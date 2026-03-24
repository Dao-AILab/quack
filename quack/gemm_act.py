# Copyright (c) 2025, Wentao Guo, Tri Dao.
from __future__ import annotations
from typing import NamedTuple, Tuple, Optional, Callable
from functools import partial

from torch import Tensor

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_og
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32, Float32, const_expr
from cutlass.cute.runtime import make_ptr

from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    ParamsBase,
    mlir_namedtuple,
    get_device_capacity,
    get_max_active_clusters,
    torch2cute_dtype_map,
)
from quack.epi_ops import TileStore
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_tvm_ffi_utils import (
    get_major,
    perm3d_single,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    div_for_dtype,
    make_fake_gemm_tensors,
    compile_gemm_kernel,
)
from quack.cache_utils import jit_cache
from quack.layout_utils import permute_gated_Cregs_b16
from quack.activation import act_fn_map, gate_fn_map
from quack.rounding import RoundingMode


class GemmActMixin(GemmDefaultEpiMixin):
    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, TileStore("mPostAct"))
    _extra_param_fields = (("act_fn", cutlass.Constexpr, None),)
    _epi_param_bases = (ParamsBase,)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops + _extra_param_fields

    def epi_to_underlying_arguments(self, args: EpilogueArguments, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        return self.EpilogueParams(**d)

    # epi_get_tma_atoms, epi_smem_bytes_per_stage, epi_get_smem_struct,
    # epi_get_smem_tensors are all inherited from ComposableEpiMixin via _epi_ops.

    def epi_setup_postact(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Setup postact TMA copies and partitions before the epilogue loop."""
        sPostAct = epi_smem_tensors[self._epi_smem_map["mPostAct"]]
        get_smem_store_op = (
            partial(sm100_utils.get_smem_store_op, tiled_tmem_load=tiled_copy_t2r)
            if self.arch == 100
            else sm90_utils_og.sm90_get_smem_store_op
        )
        copy_atom_postact_r2s = get_smem_store_op(
            self.postact_layout, self.postact_dtype, self.acc_dtype
        )
        tiled_copy_postact_r2s = cute.make_tiled_copy_S(copy_atom_postact_r2s, tiled_copy_r2s)
        tRS_sPostAct = tiled_copy_postact_r2s.get_slice(tidx).partition_D(sPostAct)
        batch_idx = tile_coord_mnkl[3]
        copy_postact, _, _ = self.epilog_gmem_copy_and_partition(
            params.tma_atom_mPostAct,
            varlen_manager.offset_batch_epi(params.mPostAct, batch_idx),
            self.cta_tile_shape_postact_mn,
            params.epi_tile_mPostAct,
            sPostAct,
            tile_coord_mnkl,
        )
        return tiled_copy_postact_r2s, tRS_sPostAct, copy_postact

    @cute.jit
    def epi_convert_postact(
        self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
    ):
        """Convert postact from acc_dtype to postact_dtype. Override for custom postprocessing."""
        if const_expr(
            self.rounding_mode == RoundingMode.RS
            and tRS_rPostAct.element_type == cutlass.Float32
            and self.postact_dtype == cutlass.BFloat16
        ):
            from quack.rounding import convert_f32_to_bf16_sr
            from cutlass.cute.tensor import TensorSSA

            # Salt with 0x9E3779B1 to avoid sharing entropy with the D output seed
            seed = (
                sr_seed
                + 0x9E3779B1
                + (
                    tile_coord_mnkl[0] * 65537
                    + tile_coord_mnkl[1] * 257
                    + tile_coord_mnkl[3] * 17
                    + (num_prev_subtiles + epi_idx) * 7
                )
            )
            tRS_rPostAct_out = cute.make_rmem_tensor_like(tRS_rPostAct, self.postact_dtype)
            src_vec = tRS_rPostAct.load()
            raw_vec = convert_f32_to_bf16_sr(src_vec, seed, tidx)
            tRS_rPostAct_out.store(TensorSSA(raw_vec, src_vec.shape, self.postact_dtype))
        else:
            tRS_rPostAct_out = cute.make_rmem_tensor_like(tRS_rPostAct, self.postact_dtype)
            tRS_rPostAct_out.store(tRS_rPostAct.load().to(self.postact_dtype))
        return tRS_rPostAct_out

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        # Apply activation function if provided
        # If we don't have .shape here, the compiler generates local stores and loads
        if const_expr(params.act_fn is not None):
            tRS_rPostAct = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                    tRS_rPostAct[i] = params.act_fn(tRS_rD[i])
            else:
                for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                    tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1] = params.act_fn(
                        (tRS_rD[2 * i], tRS_rD[2 * i + 1])
                    )
        else:
            tRS_rPostAct = tRS_rD
        return tRS_rPostAct


class GemmActSm90(GemmActMixin, GemmSm90):
    pass


class GemmActSm100(GemmActMixin, GemmSm100):
    pass


def _gated_epi_tile_fn(gemm, epi_tile):
    """Halve the N dimension of the epi_tile for gated postact."""
    if isinstance(epi_tile[1], cute.Layout):
        return (epi_tile[0], cute.recast_layout(2, 1, epi_tile[1]))
    return (epi_tile[0], epi_tile[1] // 2)


class GemmGatedMixin(GemmActMixin):
    _epi_ops = (
        *GemmDefaultEpiMixin._epi_ops,
        TileStore("mPostAct", epi_tile_fn=_gated_epi_tile_fn),
    )

    def epi_to_underlying_arguments(
        self, args: GemmActMixin.EpilogueArguments, *, loc=None, ip=None
    ) -> GemmActMixin.EpilogueParams:
        assert (
            args.mPostAct.element_type.width == 16
        ), "GemmGated only supports 16bit postact for now"
        assert self.d_layout is None or self.d_layout.is_n_major_c()
        assert cutlass.utils.LayoutEnum.from_tensor(args.mPostAct).is_n_major_c()
        if self.arch == 90:
            assert (
                self.cta_tile_shape_mnk[1] % 32 == 0
            ), "GemmGatedSm90 requires tileN to be divisible by 32"
        self.rounding_mode = args.rounding_mode
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        self.cta_tile_shape_postact_mn = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1] // 2,
        )
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(
        self,
        params: GemmActMixin.EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        tRS_rPostAct_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        # If we don't have .shape here, the compiler generates local stores and loads
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rPostAct_layout.shape, self.acc_dtype)
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rPostAct[i] = params.act_fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
        else:
            for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1] = params.act_fn(
                    (tRS_rD[4 * i], tRS_rD[4 * i + 2]), (tRS_rD[4 * i + 1], tRS_rD[4 * i + 3])
                )
        return tRS_rPostAct

    @cute.jit
    def epi_convert_postact(
        self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
    ):
        tRS_rPostAct_out = GemmActMixin.epi_convert_postact(
            self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
        )
        if const_expr(self.arch == 90):
            # Only need this if we're using STSM
            permute_gated_Cregs_b16(tRS_rPostAct_out)
        return tRS_rPostAct_out


class GemmGatedSm90(GemmGatedMixin, GemmSm90):
    pass


class GemmGatedSm100(GemmGatedMixin, GemmSm100):
    pass


@jit_cache
def _compile_gemm_act(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    postact_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    postact_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    activation,
    rowvec_dtype,
    colvec_dtype,
    colvec_ndim,
    varlen_m,
    gather_A,
    device_capacity,
    gemm_cls_name,
    rounding_mode=RoundingMode.RN,
    sr_seed_mode=0,
):
    GemmCls = (
        {"act": GemmActSm100, "gated": GemmGatedSm100}[gemm_cls_name]
        if device_capacity[0] > 9
        else {"act": GemmActSm90, "gated": GemmGatedSm90}[gemm_cls_name]
    )
    pa_leading = 1 if postact_major == "n" else 0
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        varlen_m=varlen_m,
        gather_A=gather_A,
    )
    div_pa = div_for_dtype(postact_dtype)
    pa_n = cute.sym_int() if gemm_cls_name == "gated" else n
    pa_leading_dim = 1 if gemm_cls_name == "gated" else pa_leading
    pa_shape = (m, pa_n) if varlen_m else (m, pa_n, l)
    mPostAct = fake_tensor(postact_dtype, pa_shape, leading_dim=pa_leading_dim, divisibility=div_pa)

    mRowVec = fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4)
    if colvec_ndim == 2:
        mColVec = fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
    elif colvec_ndim == 1:
        mColVec = fake_tensor(colvec_dtype, (m,), leading_dim=0, divisibility=4)
    else:
        mColVec = None

    act_fn = act_fn_map[activation] if gemm_cls_name == "act" else gate_fn_map[activation]

    def fake_scalar(mode, dtype=Int32):
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(0)
        else:
            return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=4)

    epi_args = GemmCls.EpilogueArguments(
        mPostAct,
        act_fn,
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        rounding_mode=rounding_mode,
        sr_seed=fake_scalar(sr_seed_mode),
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    varlen_args = make_fake_varlen_args(varlen_m, False, gather_A, m if varlen_m else None)
    return compile_gemm_kernel(
        GemmCls,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        gather_A,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
    )


def gemm_act(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    D: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    PostAct: Tensor,  # (l, m, n) or (total_m, n//2) if gated
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,  # (l, n)
    colvec_bias: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
) -> None:
    if activation in gate_fn_map:
        gemm_cls_name = "gated"
    else:
        assert activation in act_fn_map, f"Unsupported activation {activation}"
        gemm_cls_name = "act"

    varlen_m = cu_seqlens_m is not None
    gather_A = A_idx is not None
    if varlen_m:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen"
        assert cluster_N == 1, "gather_A requires cluster_N=1"

    A_p = perm3d_single(A, varlen_m)
    B_p = perm3d_single(B)
    D_p = perm3d_single(D, varlen_m)
    C_p = perm3d_single(C, varlen_m)
    PostAct_p = perm3d_single(PostAct, varlen_m)

    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n") if D_p is not None else None
    c_major = get_major(C_p, "m", "n") if C_p is not None else None
    postact_major = get_major(PostAct_p, "m", "n")

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype] if D is not None else None
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    colvec_ndim = colvec_bias.ndim if colvec_bias is not None else 0

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10, 11], "Only SM90, SM100, and SM110 are supported"
    if rounding_mode == RoundingMode.RS:
        assert (
            device_capacity[0] >= 10
        ), "Stochastic rounding (RoundingMode.RS) requires SM100+ (Blackwell)"

    if is_dynamic_persistent and device_capacity[0] == 9:
        assert (
            tile_count_semaphore is not None
        ), "Dynamic persistent tile scheduler in SM90 requires a semaphore in GMEM"

    sr_seed_mode = (
        2 if isinstance(sr_seed, Tensor) else (1 if rounding_mode == RoundingMode.RS else 0)
    )
    compiled_fn = _compile_gemm_act(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        postact_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        postact_major,
        (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        activation,
        torch2cute_dtype_map[rowvec_bias.dtype] if rowvec_bias is not None else None,
        torch2cute_dtype_map[colvec_bias.dtype] if colvec_bias is not None else None,
        colvec_ndim,
        varlen_m,
        gather_A,
        device_capacity,
        gemm_cls_name,
        rounding_mode=rounding_mode,
        sr_seed_mode=sr_seed_mode,
    )

    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0

    def scalar_arg(scalar, mode, dtype=Int32):
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(scalar)
        else:
            return scalar.data_ptr()

    epi_args = GemmActMixin.EpilogueArguments(
        PostAct_p,
        None,  # act_fn is Constexpr, pass None at call time
        mRowVecBroadcast=rowvec_bias,
        mColVecBroadcast=colvec_bias,
        rounding_mode=None,  # Constexpr, pass None at call time
        sr_seed=scalar_arg(sr_seed, sr_seed_mode),
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
    )
    varlen_args = make_varlen_args(cu_seqlens_m, None, A_idx)

    if device_capacity[0] > 9:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None, None)
    else:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args)


gemm_gated = gemm_act
