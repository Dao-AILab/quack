
from typing import Tuple, Optional, Callable

from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, Boolean, const_expr
from cutlass.cute.runtime import make_ptr

from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.activation import act_fn_map
from quack.gemm_act import GemmActMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_sm120 import GemmSm120
from quack.gemm_tvm_ffi_utils import (
    div_for_dtype,
    perm3d,
    get_majors,
    get_dtypes,
    make_scheduler_args,
    make_fake_scheduler_args,
    compile_gemm_kernel,
)
from quack.cache_utils import jit_cache
from quack.tile_scheduler import TriangularTileScheduler
from quack.varlen_utils import VarlenManager
import quack.copy_utils as copy_utils
from quack.rounding import RoundingMode


from quack.gemm_epilogue_plan import symmetric_epi_commit

class GemmSymmetricMixin(GemmActMixin):
    def get_scheduler_class(self, varlen_m: bool = False):
        return TriangularTileScheduler

    @cute.jit
    def epi_plan_commit(
        self,
        gmem_coord,
        epi_buffer,
        copy_D,
        copy_postact,
        postact_ctx,
        tile_coord_mnkl,
        is_tma_warp,
        epi_store_pipeline,
    ):
        symmetric_epi_commit(
            self,
            gmem_coord,
            epi_buffer,
            copy_D,
            copy_postact,
            postact_ctx,
            tile_coord_mnkl,
            is_tma_warp,
            epi_store_pipeline,
        )


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
):
    sm_to_cls = {
        9: GemmSymmetricSm90,
        10: GemmSymmetricSm100,
        11: GemmSymmetricSm100,
        12: GemmSymmetricSm120,
    }
    GemmCls = sm_to_cls[device_capacity[0]]
    # Symmetric GEMM: m == n, so reuse the same sym_int for shape checking
    m, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int()
    a_leading = 1 if a_major == "k" else 0
    b_leading = 1 if b_major == "k" else 0
    d_leading = 1 if d_major == "n" else 0
    c_leading = 1 if c_major == "n" else 0
    div_a, div_b = div_for_dtype(a_dtype), div_for_dtype(b_dtype)
    div_d, div_c = div_for_dtype(d_dtype), div_for_dtype(c_dtype) if c_dtype else 1
    mA = fake_tensor(a_dtype, (m, k, l), leading_dim=a_leading, divisibility=div_a)
    mB = fake_tensor(b_dtype, (m, k, l), leading_dim=b_leading, divisibility=div_b)
    mD = fake_tensor(d_dtype, (m, m, l), leading_dim=d_leading, divisibility=div_d)
    mC = fake_tensor(c_dtype, (m, m, l), leading_dim=c_leading, divisibility=div_c)
    # PostAct = D.mT, so it has the opposite major from D (m↔n swapped)
    div_pa = div_for_dtype(postact_dtype)
    postact_leading = 1 if postact_major == "n" else 0
    mPostAct = fake_tensor(
        postact_dtype, (m, m, l), leading_dim=postact_leading, divisibility=div_pa
    )

    def fake_scalar(mode):
        if mode == 0:
            return None
        elif mode == 1:
            return Float32(1.0)
        else:
            return make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=4)

    activation = None  # identity
    act_fn = act_fn_map[activation]
    epi_args = GemmCls.EpilogueArguments(
        mPostAct,
        act_fn,
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
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
) -> None:
    # Transpose D so the "activation" is a write to the mirrored tile
    PostAct = D.mT

    A_p, B_p, D_p, C_p = perm3d(A, B, D, C)
    PostAct_p = PostAct.permute(1, 2, 0) if PostAct.ndim == 3 else PostAct
    a_major, b_major, d_major, c_major = get_majors(A_p, B_p, D_p, C_p)
    a_dtype, b_dtype, d_dtype, c_dtype = get_dtypes(A, B, D, C)
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    # PostAct = D.mT has swapped major: if D is n-major, PostAct is m-major
    postact_major = "n" if PostAct_p.stride(1) == 1 else "m"

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10, 11, 12], "Only SM90, SM100, SM110, and SM120 are supported"

    if is_dynamic_persistent and device_capacity[0] == 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler in SM90 requires a semaphore in GMEM"
        )

    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    alpha_mode = 2 if isinstance(alpha, Tensor) else (1 if alpha != 1.0 else 0)
    beta_mode = 2 if isinstance(beta, Tensor) else (1 if beta != 1.0 else 0)

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
    )

    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0

    def scalar_arg(scalar, mode):
        if mode == 0:
            return None
        elif mode == 1:
            return Float32(scalar)
        else:
            return scalar.data_ptr()

    epi_args = GemmActMixin.EpilogueArguments(
        PostAct_p,
        None,  # act_fn is Constexpr, baked in at compile time
        alpha=scalar_arg(alpha, alpha_mode),
        beta=scalar_arg(beta, beta_mode),
        rounding_mode=None,
        sr_seed=None,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
    )
    varlen_args = None

    if device_capacity[0] in [10, 11]:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None, None, None, None)
    else:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None, None)
