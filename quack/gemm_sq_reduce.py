# Copyright (c) 2025-2026, Tri Dao.
# GEMM with column vector reduction of squared output and optional rowvec scaling:
# D_raw = A @ B (+ C), reduce[m] = sum_n(D_raw[m,n]^2), D_out = D_raw * rowvec.

from typing import NamedTuple, Optional

from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from quack.cute_dsl_utils import (
    mlir_namedtuple,
    torch2cute_dtype_map,
    get_device_capacity,
    get_max_active_clusters,
)
from quack.epi_ops import ColVecReduce, colvec_reduce_accumulate, vec_multiply
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_sm120 import GemmSm120
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.rounding import RoundingMode
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cache_utils import jit_cache
from quack.gemm_tvm_ffi_utils import (
    get_majors,
    get_dtypes,
    perm3d,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_fake_gemm_tensors,
    compile_gemm_kernel,
)
import quack.utils as utils


class GemmSqReduceMixin(GemmDefaultEpiMixin):
    """GEMM + sq_reduce + optional rowvec scaling.

    D_raw = A @ B (+ C), reduce[m] = sum_n(D_raw[m,n]^2), D_out = D_raw * rowvec.
    The sq_sum is computed BEFORE the rowvec scaling.
    """

    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, ColVecReduce("mColVecReduce"))

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: None = None

    # EpilogueParams auto-generated from _epi_ops

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        d = self._epi_ops_to_params_dict(args)
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrColVecReduce = epi_loop_tensors["mColVecReduce"]
        tDrRowVec = epi_loop_tensors["mRowVecBroadcast"]
        # Load accumulator, apply alpha/beta/C (skip rowvec/colvec — we handle rowvec below)
        rD = tRS_rD.load()
        if const_expr(hasattr(params, "alpha") and params.alpha is not None):
            alpha = utils.load_scalar_or_pointer(params.alpha)
            rD *= alpha
        if const_expr(tRS_rC is not None):
            if const_expr(not hasattr(params, "beta") or params.beta is None):
                rD += tRS_rC.load().to(tRS_rD.element_type)
            else:
                beta = utils.load_scalar_or_pointer(params.beta)
                rD += beta * tRS_rC.load().to(tRS_rD.element_type)
        tRS_rD.store(rD)
        # Accumulate sq_sum BEFORE rowvec scaling: reduce[m] += sum_n(D[m,n]^2)
        colvec_reduce_accumulate(self, tDrColVecReduce, tRS_rD, rScale=tRS_rD)
        # Multiply by rowvec (norm_weight) AFTER sq_sum
        vec_multiply(self, tRS_rD, None, tDrRowVec)
        return None


class GemmSqReduceSm90(GemmSqReduceMixin, GemmSm90):
    pass


class GemmSqReduceSm100(GemmSqReduceMixin, GemmSm100):
    pass


class GemmSqReduceSm120(GemmSqReduceMixin, GemmSm120):
    pass


@jit_cache
def _compile_gemm_sq_reduce(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    colvec_reduce_dtype,
    colvec_reduce_ndim,
    rowvec_dtype,
    device_capacity,
):
    sm_to_cls = {
        9: GemmSqReduceSm90,
        10: GemmSqReduceSm100,
        11: GemmSqReduceSm100,
        12: GemmSqReduceSm120,
    }
    GemmCls = sm_to_cls[device_capacity[0]]
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
    )
    n_tiles = cute.sym_int()
    if colvec_reduce_ndim == 3:
        mColVecReduce = fake_tensor(
            colvec_reduce_dtype,
            (l, m, n_tiles),
            leading_dim=2,
            divisibility=1,
        )
    else:
        mColVecReduce = fake_tensor(
            colvec_reduce_dtype,
            (m, n_tiles),
            leading_dim=1,
            divisibility=1,
        )
    mRowVec = fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4)
    epi_args = GemmCls.EpilogueArguments(
        mRowVecBroadcast=mRowVec,
        mColVecReduce=mColVecReduce,
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    varlen_args = make_fake_varlen_args(False, False, False, None)
    return compile_gemm_kernel(
        GemmCls,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        False,
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


def gemm_sq_reduce(
    A: Tensor,  # (l, m, k)
    B: Tensor,  # (l, n, k)
    D: Tensor,  # (l, m, n)
    C: Optional[Tensor],  # (l, m, n)
    colvec_reduce: Tensor,  # (l, m, ceildiv(n, tile_n))
    tile_count_semaphore: Optional[Tensor],  # (1,)
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec: Optional[Tensor] = None,  # (l, n) — norm_weight
) -> None:
    """GEMM + sq_reduce + optional rowvec scaling.

    D_raw = A @ B (+ C), colvec_reduce[m] = sum_n(D_raw[m,n]^2), D_out = D_raw * rowvec.
    """
    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10, 11, 12], "Only SM90, SM100, SM110, and SM120 are supported"

    A_p, B_p, D_p, C_p = perm3d(A, B, D, C)
    a_major, b_major, d_major, c_major = get_majors(A_p, B_p, D_p, C_p)
    a_dtype, b_dtype, d_dtype, c_dtype = get_dtypes(A, B, D, C)

    if is_dynamic_persistent and device_capacity[0] == 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler in SM90 requires a semaphore in GMEM"
        )

    compiled_fn = _compile_gemm_sq_reduce(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        torch2cute_dtype_map[colvec_reduce.dtype],
        colvec_reduce.ndim,
        torch2cute_dtype_map[rowvec.dtype] if rowvec is not None else None,
        device_capacity,
    )

    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    epi_args = GemmSqReduceMixin.EpilogueArguments(
        mRowVecBroadcast=rowvec,
        mColVecReduce=colvec_reduce,
        add_to_output=None,  # Constexpr, pass None at runtime
        rounding_mode=None,  # Constexpr, pass None at runtime
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters, max_swizzle_size, tile_count_semaphore
    )
    varlen_args = make_varlen_args(None, None, None)

    if device_capacity[0] in [10, 11]:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None, None, None)
    else:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None)
