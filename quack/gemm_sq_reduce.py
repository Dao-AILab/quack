# Copyright (c) 2025-2026, Tri Dao.
# GEMM with column vector reduction of squared output and optional rowvec scaling:
# D_raw = A @ B (+ C), reduce[m] = sum_n(D_raw[m,n]^2), D_out = D_raw * rowvec.

from typing import NamedTuple, Optional

from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from quack.cute_dsl_utils import (
    mlir_namedtuple,
    torch2cute_dtype_map,
    get_device_capacity,
    get_max_active_clusters,
)
from quack.epi_ops import (
    ColVecReduce,
    RowVecLoad,
    Scalar,
    TileStore,
    colvec_reduce_accumulate,
    vec_multiply,
)
from quack.gemm_act import GemmActMixin
from quack.gemm_sm80 import GemmSm80
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_sm120 import GemmSm120
from quack.rounding import RoundingMode
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cache import jit_cache
from quack.gemm_tvm_ffi_utils import (
    div_for_dtype,
    fake_batched,
    get_major,
    get_majors,
    get_dtypes,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_fake_gemm_tensors,
    compile_gemm_kernel,
    tensor_key,
    plan_scheduler_args,
    launch_gemm,
)
import quack.utils as utils


class GemmSqReduceMixin(GemmActMixin):
    """GEMM + sq_reduce + optional rowvec scaling.

    D_raw = A @ B (+ C), reduce[m] = sum_n(D_raw[m,n]^2), D_out = D_raw * rowvec.
    The sq_sum is computed BEFORE the rowvec scaling. If mAuxOut is provided, the
    pre-rowvec value (D_raw, after alpha/beta/C) is written to it.
    """

    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        RowVecLoad("mRowVecBroadcast"),
        ColVecReduce("mColVecReduce"),
        TileStore("mAuxOut"),
    )
    _extra_param_fields = ()  # no act_fn

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        mAuxOut: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        if args.mAuxOut is not None:
            self.aux_out_dtype = args.mAuxOut.element_type
            self.aux_out_layout = cutlass.tensor_utils.LayoutEnum.from_tensor(args.mAuxOut)
            self.cta_tile_shape_aux_out_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrColVecReduce = epi_loop_tensors.get("mColVecReduce")
        tDrRowVec = epi_loop_tensors.get("mRowVecBroadcast")
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
        # Snapshot pre-rowvec value if the caller wants the aux output written.
        if const_expr(getattr(params, "mAuxOut", None) is not None):
            tRS_rAuxOut = cute.make_rmem_tensor_like(tRS_rD)
            tRS_rAuxOut.store(tRS_rD.load())
            tRS_rAuxOuts = (tRS_rAuxOut,)
        else:
            tRS_rAuxOuts = ()
        # Multiply by rowvec (norm_weight) AFTER sq_sum
        vec_multiply(self, tRS_rD, None, tDrRowVec)
        return tRS_rAuxOuts


class GemmSqReduceSm90(GemmSqReduceMixin, GemmSm90):
    pass


class GemmSqReduceSm80(GemmSqReduceMixin, GemmSm80):
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
    aux_out_dtype,
    aux_out_major,
    device_capacity,
    batched=True,
    b_kn=False,
):
    sm_to_cls = {
        8: GemmSqReduceSm80,
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
        batched=batched,
        b_kn=b_kn,
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
    if aux_out_dtype is not None:
        aux_leading = 1 if aux_out_major == "n" else 0
        mAuxOut = fake_batched(
            aux_out_dtype, m, n, l if batched else None, aux_leading, div_for_dtype(aux_out_dtype)
        )
    else:
        mAuxOut = None
    epi_args = GemmCls.EpilogueArguments(
        mRowVecBroadcast=mRowVec,
        mColVecReduce=mColVecReduce,
        mAuxOut=mAuxOut,
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
        b_transposed=b_kn,
    )


class _GemmSqReducePlan(NamedTuple):
    """Launch plan derived purely from tensor metadata and config flags.

    Cached per metadata key (see ``_gemm_sq_reduce_plan_cache``) so warm calls
    skip major/dtype derivation and the compile-cache lookup. See ``_GemmPlan``
    in gemm.py for the pattern.
    """

    compiled_fn: object
    is_sm100_family: bool  # SM100/110 take trailing (SFA, SFB) args
    max_active_clusters: int
    max_swizzle_size: int
    scheduler_uses_semaphore: bool  # only the SM90 dynamic scheduler consumes the semaphore
    scheduler_static: Optional[object]  # TileSchedulerOptions when it has no per-call values


_gemm_sq_reduce_plan_cache: dict[tuple, _GemmSqReducePlan] = {}


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
    tile_K: int | None = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec: Optional[Tensor] = None,  # (l, n) — norm_weight
    aux_out: Optional[Tensor] = None,  # (l, m, n) — pre-rowvec output snapshot
    b_kn: bool = False,  # B passed (k, n) / (l, k, n), transposed at trace time (SM90+)
) -> _GemmSqReducePlan:
    """GEMM + sq_reduce + optional rowvec scaling.

    D_raw = A @ B (+ C), colvec_reduce[m] = sum_n(D_raw[m,n]^2), D_out = D_raw * rowvec.
    If aux_out is provided, the pre-rowvec value (D_raw, after alpha/beta/C) is also
    written to it.
    """
    # The key captures every input the plan build reads (shapes and strides
    # subsume the majors and the validation asserts), so a cache hit is exactly
    # a replay of a previously validated call with different data pointers.
    key = (
        tensor_key(A),
        tensor_key(B),
        tensor_key(D),
        tensor_key(C),
        tensor_key(colvec_reduce),
        tensor_key(rowvec),
        tensor_key(aux_out),
        tile_count_semaphore is not None,
        A.device,
        tile_M,
        tile_N,
        tile_K,
        cluster_M,
        cluster_N,
        pingpong,
        persistent,
        is_dynamic_persistent,
        max_swizzle_size,
        b_kn,
    )
    plan = _gemm_sq_reduce_plan_cache.get(key)
    if plan is None:
        plan = _build_gemm_sq_reduce_plan(
            A,
            B,
            D,
            C,
            colvec_reduce,
            tile_count_semaphore=tile_count_semaphore,
            tile_M=tile_M,
            tile_N=tile_N,
            cluster_M=cluster_M,
            cluster_N=cluster_N,
            tile_K=tile_K,
            pingpong=pingpong,
            persistent=persistent,
            is_dynamic_persistent=is_dynamic_persistent,
            max_swizzle_size=max_swizzle_size,
            rowvec=rowvec,
            aux_out=aux_out,
            b_kn=b_kn,
        )
        _gemm_sq_reduce_plan_cache[key] = plan
    run_gemm_sq_reduce_plan(
        plan,
        A,
        B,
        D,
        C,
        colvec_reduce,
        tile_count_semaphore=tile_count_semaphore,
        rowvec=rowvec,
        aux_out=aux_out,
    )
    return plan


def run_gemm_sq_reduce_plan(
    plan: _GemmSqReducePlan,
    A: Tensor,
    B: Tensor,
    D: Tensor,
    C: Optional[Tensor],
    colvec_reduce: Tensor,
    *,
    tile_count_semaphore: Optional[Tensor] = None,
    rowvec: Optional[Tensor] = None,
    aux_out: Optional[Tensor] = None,
) -> None:
    """Launch a resolved plan: only per-call pointers here.
    The tensors must match the metadata the plan was built from."""
    epi_args = GemmSqReduceMixin.EpilogueArguments(
        mRowVecBroadcast=rowvec,
        mColVecReduce=colvec_reduce,
        mAuxOut=aux_out,
        add_to_output=None,  # Constexpr, pass None at runtime
        rounding_mode=None,  # Constexpr, pass None at runtime
    )
    scheduler_args = plan_scheduler_args(plan, tile_count_semaphore)
    varlen_args = make_varlen_args(None, None, None)
    launch_gemm(plan, A, B, D, C, epi_args, scheduler_args, varlen_args)


def _build_gemm_sq_reduce_plan(
    A,
    B,
    D,
    C,
    colvec_reduce,
    *,
    tile_count_semaphore,
    tile_M,
    tile_N,
    cluster_M,
    cluster_N,
    tile_K,
    pingpong,
    persistent,
    is_dynamic_persistent,
    max_swizzle_size,
    rowvec,
    aux_out,
    b_kn=False,
) -> _GemmSqReducePlan:
    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )
    batched = A.ndim == 3
    if not batched:
        # Dense 2D (unbatched) operands: trace-time batch append, see gemm_act.
        # colvec_reduce must then be 2D (m, n_tiles) — its ndim keys the fake.
        assert (
            B.ndim == 2
            and D.ndim == 2
            and (C is None or C.ndim == 2)
            and colvec_reduce.ndim == 2
            and (aux_out is None or aux_out.ndim == 2)
        ), "2D (unbatched) A requires 2D B, D, C, colvec_reduce, and aux_out"
    if not batched or b_kn:
        assert device_capacity[0] in [9, 10, 11, 12], "2D (unbatched) operands / b_kn require SM90+"

    a_major, b_major, d_major, c_major = get_majors(A, B, D, C)
    if b_kn:
        # Majors are logical (n, k) labels: with B stored (k, n), a contiguous
        # last dim means n-major.
        b_major = "n" if B.stride(-1) == 1 else "k"
    a_dtype, b_dtype, d_dtype, c_dtype = get_dtypes(A, B, D, C)
    if aux_out is not None:
        aux_out_dtype = torch2cute_dtype_map[aux_out.dtype]
        aux_out_major = get_major(aux_out, "m", "n")
    else:
        aux_out_dtype = None
        aux_out_major = None

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
        (tile_M, tile_N, tile_K) if tile_K is not None else (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        torch2cute_dtype_map[colvec_reduce.dtype],
        colvec_reduce.ndim,
        torch2cute_dtype_map[rowvec.dtype] if rowvec is not None else None,
        aux_out_dtype,
        aux_out_major,
        device_capacity,
        batched=batched,
        b_kn=b_kn,
    )

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    # Must mirror make_fake_scheduler_args in _compile_gemm_sq_reduce: only the
    # SM90 dynamic scheduler consumes the semaphore, so it's the only non-static case.
    scheduler_uses_semaphore = is_dynamic_persistent and device_capacity[0] == 9
    scheduler_static = (
        make_scheduler_args(max_active_clusters, max_swizzle_size, None)
        if not scheduler_uses_semaphore
        else None
    )
    return _GemmSqReducePlan(
        compiled_fn=compiled_fn,
        is_sm100_family=device_capacity[0] in [10, 11],
        max_active_clusters=max_active_clusters,
        max_swizzle_size=max_swizzle_size,
        scheduler_uses_semaphore=scheduler_uses_semaphore,
        scheduler_static=scheduler_static,
    )
