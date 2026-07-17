from typing import NamedTuple, Optional

from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_ptr

from quack.cute_dsl_utils import (
    get_device_capacity,
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.rounding import RoundingMode
from quack.epi_ops import TileStore
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_sm80 import GemmSm80
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_sm120 import GemmSm120
from quack.gemm_tvm_ffi_utils import (
    div_for_dtype,
    fake_batched,
    get_major,
    get_majors,
    get_dtypes,
    make_scheduler_args,
    make_fake_scheduler_args,
    compile_gemm_kernel,
    tensor_key,
    scalar_mode,
    scalar_arg,
    plan_scheduler_args,
    launch_gemm,
)
from quack.cache import jit_cache
from quack.tile_scheduler import TriangularTileScheduler


def _symmetric_offdiag_pred(gemm, tile_coord_mnkl):
    """Skip the mirrored aux write on diagonal tiles — the aux output is D.mT,
    so the diagonal tile would write the same gmem twice."""
    square_tile_m = tile_coord_mnkl[0] // gemm.cluster_shape_mnk[0]
    square_tile_n = tile_coord_mnkl[1] // gemm.cluster_shape_mnk[1]
    return square_tile_m != square_tile_n


class GemmSymmetricMixin(GemmDefaultEpiMixin):
    """The default (linear) epilogue plus an aux output that is the transposed
    mirror of D (PostAct = D.mT) on a triangular tile schedule; the store
    predicate on the TileStore op skips the mirrored write on diagonal tiles.
    The epilogue itself is the generic driver. Stays a mixin by design: the
    scheduler choice is not expressible in the fn contract."""

    _epi_ops = GemmDefaultEpiMixin._epi_ops + (
        TileStore("mAuxOut", store_pred_fn=_symmetric_offdiag_pred),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mAuxOut: cute.Tensor
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    def get_scheduler_class(self, varlen_m: bool = False):
        return TriangularTileScheduler

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        # The mirrored output IS the (linear-epilogue) D values.
        return (tRS_rD,)


class GemmSymmetricSm80(GemmSymmetricMixin, GemmSm80):
    pass


class GemmSymmetricSm90(GemmSymmetricMixin, GemmSm90):
    pass


class GemmSymmetricSm100(GemmSymmetricMixin, GemmSm100):
    pass


class GemmSymmetricSm120(GemmSymmetricMixin, GemmSm120):
    pass


@jit_cache
def _compile_gemm_symmetric(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    c_major,
    postact_dtype,
    a_major,
    b_major,
    d_major,
    postact_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    alpha_mode,
    beta_mode,
    device_capacity,
    batched=True,
):
    sm_to_cls = {
        8: GemmSymmetricSm80,
        9: GemmSymmetricSm90,
        10: GemmSymmetricSm100,
        11: GemmSymmetricSm100,
        12: GemmSymmetricSm120,
    }
    GemmCls = sm_to_cls[device_capacity[0]]
    # Symmetric GEMM: m == n, so reuse the same sym_int for shape checking
    m, k = cute.sym_int(), cute.sym_int()
    l = cute.sym_int() if batched else None
    a_leading = 1 if a_major == "k" else 0
    b_leading = 1 if b_major == "k" else 0
    d_leading = 1 if d_major == "n" else 0
    c_leading = 1 if c_major == "n" else 0
    div_a, div_b = div_for_dtype(a_dtype), div_for_dtype(b_dtype)
    div_d, div_c = div_for_dtype(d_dtype), div_for_dtype(c_dtype) if c_dtype else 1
    mA = fake_batched(a_dtype, m, k, l, a_leading, div_a)
    mB = fake_batched(b_dtype, m, k, l, b_leading, div_b)
    mD = fake_batched(d_dtype, m, m, l, d_leading, div_d)
    mC = fake_batched(c_dtype, m, m, l, c_leading, div_c)
    # PostAct = D.mT, so it has the opposite major from D (m↔n swapped)
    div_pa = div_for_dtype(postact_dtype)
    postact_leading = 1 if postact_major == "n" else 0
    mAuxOut = fake_batched(postact_dtype, m, m, l, postact_leading, div_pa)

    def fake_scalar(mode):
        if mode == 0:
            return None
        elif mode == 1:
            return Float32(1.0)
        else:
            return make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=4)

    epi_args = GemmCls.EpilogueArguments(
        mAuxOut,
        alpha=fake_scalar(alpha_mode),
        beta=fake_scalar(beta_mode),
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    varlen_args = None
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


class _GemmSymmetricPlan(NamedTuple):
    """Launch plan derived purely from tensor metadata and config flags.

    Cached per metadata key (see ``_gemm_symmetric_plan_cache``) so warm calls
    skip major/dtype derivation and the compile-cache lookup. See ``_GemmPlan``
    in gemm.py for the pattern.
    """

    compiled_fn: object
    is_sm100_family: bool  # SM100/110 take trailing (SFA, SFB) args
    alpha_mode: int
    beta_mode: int
    max_active_clusters: int
    max_swizzle_size: int
    scheduler_uses_semaphore: bool  # only the SM90 dynamic scheduler consumes the semaphore
    scheduler_static: Optional[object]  # TileSchedulerOptions when it has no per-call values


_gemm_symmetric_plan_cache: dict[tuple, _GemmSymmetricPlan] = {}


def gemm_symmetric(
    A: Tensor,  # (l, m, k)
    B: Tensor,  # (l, m, k)
    D: Optional[Tensor],  # (l, m, m)
    C: Optional[Tensor],  # (l, m, m)
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
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
) -> _GemmSymmetricPlan:
    alpha_mode = scalar_mode(alpha)
    beta_mode = scalar_mode(beta)
    # The key captures every input the plan build reads (D's metadata subsumes
    # PostAct = D.mT), so a cache hit is exactly a replay of a previously
    # validated call with different data pointers.
    key = (
        tensor_key(A),
        tensor_key(B),
        tensor_key(D),
        tensor_key(C),
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
        alpha_mode,
        beta_mode,
    )
    plan = _gemm_symmetric_plan_cache.get(key)
    if plan is None:
        plan = _build_gemm_symmetric_plan(
            A,
            B,
            D,
            C,
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
            alpha_mode=alpha_mode,
            beta_mode=beta_mode,
        )
        _gemm_symmetric_plan_cache[key] = plan
    run_gemm_symmetric_plan(
        plan, A, B, D, C, tile_count_semaphore=tile_count_semaphore, alpha=alpha, beta=beta
    )
    return plan


def run_gemm_symmetric_plan(
    plan: _GemmSymmetricPlan,
    A: Tensor,
    B: Tensor,
    D: Tensor,
    C: Optional[Tensor],
    *,
    tile_count_semaphore: Optional[Tensor] = None,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
) -> None:
    """Launch a resolved plan: only per-call pointers and scalar values here.
    The tensors must match the metadata the plan was built from."""
    # Transpose D so the "activation" is a write to the mirrored tile
    PostAct = D.mT
    epi_args = GemmSymmetricMixin.EpilogueArguments(
        PostAct,
        alpha=scalar_arg(alpha, plan.alpha_mode),
        beta=scalar_arg(beta, plan.beta_mode),
        rounding_mode=None,
        sr_seed=None,
    )
    scheduler_args = plan_scheduler_args(plan, tile_count_semaphore)
    launch_gemm(plan, A, B, D, C, epi_args, scheduler_args, None)


def _build_gemm_symmetric_plan(
    A,
    B,
    D,
    C,
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
    alpha_mode,
    beta_mode,
) -> _GemmSymmetricPlan:
    PostAct = D.mT
    a_major, b_major, d_major, c_major = get_majors(A, B, D, C)
    a_dtype, b_dtype, d_dtype, c_dtype = get_dtypes(A, B, D, C)
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    # PostAct = D.mT has swapped major: if D is n-major, PostAct is m-major
    postact_major = get_major(PostAct, "m", "n")

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )
    batched = A.ndim == 3
    if not batched:
        # Dense 2D (unbatched) operands: trace-time batch append, see gemm_act.
        # No b_kn here — symmetric B is operand-shaped (m, k) like A, never (K, N).
        assert B.ndim == 2 and D.ndim == 2 and (C is None or C.ndim == 2), (
            "2D (unbatched) A requires 2D B, D, and C"
        )
        assert device_capacity[0] in [9, 10, 11, 12], "2D (unbatched) operands require SM90+"

    if is_dynamic_persistent and device_capacity[0] <= 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler in SM90 requires a semaphore in GMEM"
        )

    tile_shape_mn = (tile_M, tile_N, tile_K) if tile_K is not None else (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)

    compiled_fn = _compile_gemm_symmetric(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        c_major,
        postact_dtype,
        a_major,
        b_major,
        d_major,
        postact_major,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        is_dynamic_persistent,
        alpha_mode,
        beta_mode,
        device_capacity,
        batched=batched,
    )

    cluster_size = cluster_M * cluster_N
    max_active_clusters = (
        get_max_active_clusters(cluster_size, device_capacity=device_capacity) if persistent else 0
    )
    # Must mirror make_fake_scheduler_args in _compile_gemm_symmetric: only the
    # SM90 dynamic scheduler consumes the semaphore, so it's the only non-static case.
    scheduler_uses_semaphore = is_dynamic_persistent and device_capacity[0] == 9
    scheduler_static = (
        make_scheduler_args(max_active_clusters, max_swizzle_size, None)
        if not scheduler_uses_semaphore
        else None
    )
    return _GemmSymmetricPlan(
        compiled_fn=compiled_fn,
        is_sm100_family=device_capacity[0] in [10, 11],
        alpha_mode=alpha_mode,
        beta_mode=beta_mode,
        max_active_clusters=max_active_clusters,
        max_swizzle_size=max_swizzle_size,
        scheduler_uses_semaphore=scheduler_uses_semaphore,
        scheduler_static=scheduler_static,
    )
