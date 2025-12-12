# Copyright (c) 2025, Tri Dao.
from typing import Optional, Tuple
from functools import partial

from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
import cutlass.torch as cutlass_torch

from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_act import GemmActMixin
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.gemm_wrapper_utils import GemmWrapperBase
import quack.activation


class GemmDActMixin(GemmActMixin):
    # Different from GemmActSm90, here act_bwd_fn must take in 2 arguments (x, dout)
    # and return 2 arguments (dx, out)
    EpilogueArguments = GemmActMixin.EpilogueArguments
    EpilogueParams = GemmActMixin.EpilogueParams

    @cute.jit
    def epi_visit_subtile(
        self,
        params: EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        assert tRS_rC is not None
        # We don't add C to the accumulator
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None)
        tRS_rC_acc = cute.make_fragment_like(tRS_rC, self.acc_dtype)
        tRS_rC_acc.store(tRS_rC.load().to(self.acc_dtype))
        # If we don't have .shape here, the compiler generates local stores and loads
        if const_expr(params.act_fn is not None):
            tRS_rPostAct = cute.make_fragment(tRS_rD.layout.shape, self.acc_dtype)
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                    tRS_rD[i], tRS_rPostAct[i] = params.act_fn(tRS_rC_acc[i], tRS_rD[i])
            else:
                for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                    (
                        (tRS_rD[2 * i], tRS_rD[2 * i + 1]),
                        (tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1]),
                    ) = params.act_fn(
                        (tRS_rC_acc[2 * i], tRS_rC_acc[2 * i + 1]),
                        (tRS_rD[2 * i], tRS_rD[2 * i + 1]),
                    )
        else:
            tRS_rPostAct = tRS_rC_acc
        # Type conversion
        tRS_rPostAct_out = cute.make_fragment_like(tRS_rPostAct, self.postact_dtype)
        tRS_rPostAct_out.store(tRS_rPostAct.load().to(self.postact_dtype))
        return tRS_rPostAct_out


class GemmDActSm90(GemmDActMixin, GemmSm90):
    pass


class GemmDActSm100(GemmDActMixin, GemmSm100):
    pass


dact_fn_map = {
    None: None,
    "relu": quack.activation.drelu,
    "relu_sq": quack.activation.drelu_sq,
    "gelu_tanh_approx": quack.activation.dgelu_tanh_approx,
}


def gemm_dact(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    Out: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    PreAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    PostAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = True,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
) -> None:
    if cu_seqlens_m is not None:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        assert Out.stride(-1) == 1, "varlen_m requires Out to be n-major"
        assert PreAct.stride(-1) == 1, "varlen_m requires PreAct to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    gather_A = A_idx is not None
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen (cu_seqlens_m must be specified)"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    assert activation in dact_fn_map, f"Unsupported activation {activation}"

    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A,
        B,
        Out,
        PreAct,
        additional_tensors={"PostAct": PostAct},
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=cu_seqlens_m is not None)
    GemmWrapperBase.extract_dtypes(tensor_infos)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
        "PostAct": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10], "Only SM90 and SM100 are supported"
    GemmCls = GemmDActSm100 if device_capacity[0] > 9 else GemmDActSm90

    acc_dtype = Float32
    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    if not GemmCls.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        acc_dtype,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Skipping due to unsupported combination of types and majors")

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    GemmWrapperBase.create_cute_tensors(tensor_infos, major_configs)
    act_fn = dact_fn_map[activation]
    epi_args = GemmCls.EpilogueArguments(tensor_infos["PostAct"].cute_tensor, act_fn)
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters, tile_count_semaphore, max_swizzle_size=max_swizzle_size
    )

    # Create varlen arguments if needed (assumes persistent=True when varlen_m)
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        None,  # cu_seqlens_k
        A_idx,
        max_active_clusters,
        cluster_shape_mnk,
        tensor_infos,
        GemmCls.num_epi_tensormaps,
        pingpong,
    )

    current_stream = cutlass_torch.current_stream()
    compile_key = GemmWrapperBase.get_compile_key(
        tensor_infos,
        activation,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        tile_count_semaphore is not None,
        device_capacity,
        max_swizzle_size,
        cu_seqlens_m is not None,
        A_idx is not None,
        key_tensor_names=("A", "B", "D", "PostAct", "C"),
    )
    cache = gemm_dact.compile_cache
    if compile_key not in cache:
        if device_capacity[0] == 9:
            GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
        gemm = GemmCls(
            acc_dtype,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            gather_A=gather_A,
        )
        cache[compile_key] = cute.compile(
            gemm,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
        )
    cache[compile_key](
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
    )


gemm_dact.compile_cache = {}
