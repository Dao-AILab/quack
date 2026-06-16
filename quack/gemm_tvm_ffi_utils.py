# Copyright (c) 2025, Tri Dao.
# Shared utilities for TVM-FFI GEMM compilation.

from functools import partial

import torch

import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.runtime import make_ptr

from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.gemm_config import SplitKMode
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.tile_scheduler import TileSchedulerOptions
from quack.varlen_utils import VarlenArguments

# Blockscaled scale-factor dtype determines the quantization block size along K:
# e8m0 -> MX formats (32-element blocks), e4m3 -> NVFP4 (16-element blocks).
SF_DTYPE_TO_VEC_SIZE = {
    torch.float8_e8m0fnu: 32,
    torch.float8_e4m3fn: 16,
}


def validate_blockscaled_sf(A, B, SFA, SFB, device_capacity, num_batches=None, varlen_k=False):
    """Validate blockscaled scale factors against kernel-layout operands.

    A is (l, m, k[/2 if fp4]) and B is (l, n, k[/2]); SFA/SFB are
    (l, rm/rn, rk, 32, 4, 4) with the inner (32, 4, 4) block contiguous
    (strides (16, 4, 1) — one 512 B atom per 128 rows x 4 K-blocks).

    When num_batches is not None and varlen_k is False (varlen_m), A is
    (total_m, k) and SFA must be a single M-padded buffer (tile-aligned
    per-batch padding) (1, total_padded_rm, rk, 32, 4, 4) with
    total_padded_rm >= ceil(total_m/128) + (num_batches - 1) — the bound from
    AI/varlen_blockscaled_sf_layout.md that suffices for any per-batch split
    of total_m. SFB stays per-batch: (num_batches, rn, rk, 32, 4, 4).

    When varlen_k, A is (m, total_k) m-major and B is (n, total_k) n-major
    (MXFP8 only — fp4 operands must be K-major), and BOTH SF buffers are
    K-padded with tile-aligned per-batch padding:
    (1, rm/rn, total_padded_rk, 32, 4, 4) with
    total_padded_rk >= ceil(total_k/128) + (num_batches - 1).
    SF pad bytes inside each batch's last atom are loaded by the kernel but
    never consumed: the mma loop skips the MMA instructions for pad k-blocks
    (one instruction per SF block for mxfp8; see GemmSm100.mma), so the pad
    may be arbitrary bytes — torch.empty buffers are fine.
    Returns (sf_dtype, sf_vec_size) as (cutlass dtype, int).
    """
    varlen_m = num_batches is not None and not varlen_k
    assert not varlen_k or num_batches is not None, "varlen_k requires num_batches"
    assert SFB is not None, "SFA and SFB must be provided together"
    assert device_capacity[0] in [10, 11], "Blockscaled GEMM requires SM100/SM110"
    assert SFA.dtype == SFB.dtype, f"SF dtype mismatch: {SFA.dtype} vs {SFB.dtype}"
    assert SFA.dtype in SF_DTYPE_TO_VEC_SIZE, f"unsupported SF dtype: {SFA.dtype}"
    sf_vec_size = SF_DTYPE_TO_VEC_SIZE[SFA.dtype]
    sf_dtype = torch2cute_dtype_map[SFA.dtype]
    # A.shape[-1] is packed K for fp4 (two elements per byte) while dlpack presents
    # the logical extent to the kernel, so validate rk against logical K.
    k_logical = A.shape[-1] * (2 if A.dtype == torch.float4_e2m1fn_x2 else 1)
    rk = (k_logical + 4 * sf_vec_size - 1) // (4 * sf_vec_size)
    if varlen_k:
        assert A.dtype != torch.float4_e2m1fn_x2, (
            "varlen_k blockscaled supports MXFP8 only: fp4 operands must be K-major, "
            "but varlen_k requires m-major A / n-major B"
        )
        assert A.ndim == 2 and B.ndim == 2, (
            f"varlen_k expects A (m, total_k) and B (n, total_k), "
            f"got shapes {tuple(A.shape)} / {tuple(B.shape)}"
        )
        # rk here = ceil(total_k/128); K-padded buffers need one extra atom
        # column per additional batch.
        min_rk = rk + (num_batches - 1)
        for name, SF, mn in (("SFA", SFA, A.shape[0]), ("SFB", SFB, B.shape[0])):
            r_mn = (mn + 127) // 128
            assert SF.shape[0] == 1 and SF.shape[1] == r_mn and tuple(SF.shape[3:]) == (32, 4, 4), (
                f"{name} shape {tuple(SF.shape)} != (1, {r_mn}, total_padded_rk, 32, 4, 4)"
            )
            assert SF.shape[2] >= min_rk, (
                f"{name} padded rk {SF.shape[2]} < ceil(total_k/128) + (L-1) = {min_rk}"
            )
        shapes = []
    elif varlen_m:
        assert A.ndim == 2, f"varlen_m expects A as (total_m, k), got shape {tuple(A.shape)}"
        assert B.shape[0] == num_batches, (
            f"B batch dim {B.shape[0]} != len(cu_seqlens_m) - 1 = {num_batches}"
        )
        min_rm = (A.shape[0] + 127) // 128 + (num_batches - 1)
        assert SFA.shape[0] == 1 and tuple(SFA.shape[2:]) == (rk, 32, 4, 4), (
            f"SFA shape {tuple(SFA.shape)} != (1, total_padded_rm, {rk}, 32, 4, 4)"
        )
        assert SFA.shape[1] >= min_rm, (
            f"SFA padded rm {SFA.shape[1]} < ceil(total_m/128) + (L-1) = {min_rm}"
        )
        shapes = [("SFB", SFB, (num_batches, (B.shape[-2] + 127) // 128, rk, 32, 4, 4))]
    else:
        shapes = [
            (name, SF, (A.shape[0], (mn + 127) // 128, rk, 32, 4, 4))
            for name, SF, mn in (("SFA", SFA, A.shape[-2]), ("SFB", SFB, B.shape[-2]))
        ]
    for name, SF, expected in shapes:
        assert tuple(SF.shape) == expected, f"{name} shape {tuple(SF.shape)} != {expected}"
    for name, SF in (("SFA", SFA), ("SFB", SFB)):
        assert SF.stride()[-3:] == (16, 4, 1), (
            f"{name}: inner (32, 4, 4) block must be contiguous with strides (16, 4, 1), "
            f"got {SF.stride()[-3:]}"
        )
    return sf_dtype, sf_vec_size


def div_for_dtype(dtype):
    """16-byte alignment: divisibility in elements = 128 // dtype_width_bits."""
    return 128 // dtype.width


def perm3d_single(t, varlen_m=False):
    """Permute a single 3D tensor from (L, *, *) to (*, *, L), skipping for varlen_m or 2D."""
    return t.permute(1, 2, 0) if t is not None and t.ndim == 3 and not varlen_m else t


def perm3d(A, B, D, C, varlen_m=False, varlen_k=False):
    """Permute 3D tensors from (L, *, *) to (*, *, L)."""

    def _perm(t):
        return t.permute(1, 2, 0) if t is not None and t.ndim == 3 else t

    if varlen_m:
        return A, _perm(B), D, C
    elif varlen_k:
        return A, B, _perm(D), _perm(C)
    else:
        return _perm(A), _perm(B), _perm(D), _perm(C)


def get_major(t, dim0, dim1):
    return dim1 if t.stride(1) == 1 else dim0


def get_majors(A_p, B_p, D_p, C_p):
    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n")
    c_major = get_major(C_p, "m", "n") if C_p is not None else None
    return a_major, b_major, d_major, c_major


def get_dtypes(A, B, D, C):
    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype]
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None
    return a_dtype, b_dtype, d_dtype, c_dtype


def make_scheduler_args(
    max_active_clusters, max_swizzle_size, tile_count_semaphore, batch_idx_permute=None
):
    return TileSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        raster_order=None,
        max_swizzle_size=max_swizzle_size,
        tile_count_semaphore=(
            tile_count_semaphore.data_ptr() if tile_count_semaphore is not None else None
        ),
        batch_idx_permute=batch_idx_permute,
    )


def make_fake_scheduler_args(has_semaphore, has_batch_idx_permute, l_sym):
    return TileSchedulerOptions(
        max_active_clusters=Int32(1),
        max_swizzle_size=Int32(8),
        tile_count_semaphore=(
            make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=4) if has_semaphore else None
        ),
        batch_idx_permute=(
            fake_tensor(Int32, (l_sym,), leading_dim=0, divisibility=4)
            if has_batch_idx_permute
            else None
        ),
    )


def make_varlen_args(cu_seqlens_m, cu_seqlens_k, A_idx):
    if cu_seqlens_m is None and cu_seqlens_k is None:
        return None
    return VarlenArguments(
        mCuSeqlensM=cu_seqlens_m,
        mCuSeqlensK=cu_seqlens_k,
        mAIdx=A_idx,
    )


def make_fake_varlen_args(varlen_m, varlen_k, gather_A, aidx_len):
    if not varlen_m and not varlen_k:
        return None
    num_seqlens = cute.sym_int()
    return VarlenArguments(
        mCuSeqlensM=(
            fake_tensor(Int32, (num_seqlens,), leading_dim=0, divisibility=4) if varlen_m else None
        ),
        mCuSeqlensK=(
            fake_tensor(Int32, (num_seqlens,), leading_dim=0, divisibility=4) if varlen_k else None
        ),
        mAIdx=(
            fake_tensor(Int32, (aidx_len,), leading_dim=0, divisibility=4) if gather_A else None
        ),
    )


def make_fake_gemm_tensors(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    varlen_m=False,
    varlen_k=False,
    gather_A=False,
):
    """Create fake tensors for mA, mB, mD, mC with shared sym_ints.
    Pass dtype=None to get None for that tensor (e.g. optional C).
    Returns (mA, mB, mD, mC, m, n, k, l).
    When varlen_m, m is total_m (flattened M of D/C). When varlen_k, k is total_k.
    """
    a_leading = 1 if a_major == "k" else 0
    b_leading = 1 if b_major == "k" else 0
    d_leading = 1 if d_major == "n" else 0
    c_leading = 1 if c_major == "n" else 0
    m, n, l = cute.sym_int(), cute.sym_int(), cute.sym_int()
    # Sub-byte (fp4) tensors need their contiguous extent statically divisible by the
    # packing factor; fp4 operands are k-major, so mark k. Harmless for 8-bit+ dtypes.
    k_div = div_for_dtype(a_dtype) if a_dtype.width < 8 else 1
    k = cute.sym_int(divisibility=k_div)
    div_a = div_for_dtype(a_dtype)
    div_b = div_for_dtype(b_dtype)
    div_d = div_for_dtype(d_dtype) if d_dtype is not None else 1
    div_c = div_for_dtype(c_dtype) if c_dtype is not None else 1
    if varlen_m:
        # m is total_m in this case: the flattened M dimension of D/C
        m = cute.sym_int()
        a_m = cute.sym_int() if gather_A else m
        mA = fake_tensor(a_dtype, (a_m, k), leading_dim=a_leading, divisibility=div_a)
        mB = fake_tensor(b_dtype, (n, k, l), leading_dim=b_leading, divisibility=div_b)
        mD = fake_tensor(d_dtype, (m, n), leading_dim=d_leading, divisibility=div_d)
        mC = fake_tensor(c_dtype, (m, n), leading_dim=c_leading, divisibility=div_c)
    elif varlen_k:
        # k is total_k in this case: the flattened K dimension of A/B
        k = cute.sym_int()
        a_k = cute.sym_int() if gather_A else k
        mA = fake_tensor(a_dtype, (m, a_k), leading_dim=a_leading, divisibility=div_a)
        mB = fake_tensor(b_dtype, (n, k), leading_dim=b_leading, divisibility=div_b)
        mD = fake_tensor(d_dtype, (m, n, l), leading_dim=d_leading, divisibility=div_d)
        mC = fake_tensor(c_dtype, (m, n, l), leading_dim=c_leading, divisibility=div_c)
    else:
        mA = fake_tensor(a_dtype, (m, k, l), leading_dim=a_leading, divisibility=div_a)
        mB = fake_tensor(b_dtype, (n, k, l), leading_dim=b_leading, divisibility=div_b)
        mD = fake_tensor(d_dtype, (m, n, l), leading_dim=d_leading, divisibility=div_d)
        mC = fake_tensor(c_dtype, (m, n, l), leading_dim=c_leading, divisibility=div_c)
    return mA, mB, mD, mC, m, n, k, l


def make_fake_sf_tensor(sf_dtype, l):
    """Fake (l, rm, rk, 32, 4, 4) blockscaled scale-factor tensor.

    The inner (32, 4, 4) block has static strides (16, 4, 1) — one contiguous
    512 B atom per 128 rows x 4 K-blocks, so TMA loads it as one box. The
    kernel only consumes the base pointer and the outer (l, rm, rk) strides
    (the atom layout is hardware-fixed); outer strides are dynamic but
    atom-granular, so slices of larger scale buffers are accepted without a
    copy.
    """
    rm, rk = cute.sym_int(), cute.sym_int()
    return cute.runtime.make_fake_tensor(
        sf_dtype,
        (l, rm, rk, 32, 4, 4),
        stride=(
            cute.sym_int64(divisibility=512),
            cute.sym_int64(divisibility=512),
            cute.sym_int64(divisibility=512),
            16,
            4,
            1,
        ),
        assumed_align=16,
    )


def compile_gemm_kernel(
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
    post_init=None,
    mSFA=None,
    mSFB=None,
    use_tma_gather=False,
    concat_layout=None,
    num_warps=None,
    sf_vec_size=None,
    split_k=1,
    split_k_mode=SplitKMode.SERIAL,
):
    """Build GemmCls instance, apply SM90 partial, and cute.compile with TVM-FFI."""
    split_k_kwargs = {}
    if split_k != 1:
        assert device_capacity[0] in [9, 10, 11, 12], "split_k requires SM90/SM100/SM120"
        split_k_kwargs = {"split_k": split_k, "split_k_mode": split_k_mode}
    if device_capacity[0] == 8:
        sm8x_kwargs = {"is_persistent": persistent, "num_warps": num_warps}
        sm8x_kwargs["arch"] = device_capacity[0] * 10 + device_capacity[1]
        GemmCls = partial(GemmCls, **sm8x_kwargs)
    elif device_capacity[0] in [9, 12]:
        GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent, **split_k_kwargs)
    elif device_capacity[0] in [10, 11]:
        GemmCls = partial(
            GemmCls,
            use_clc_persistence=is_dynamic_persistent,
            use_tma_gather=use_tma_gather,
            sf_vec_size=sf_vec_size,
            **split_k_kwargs,
        )
    gemm_obj = GemmCls(
        Float32,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        gather_A=gather_A,
        concat_layout=concat_layout,
    )
    if post_init:
        post_init(gemm_obj)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    sf_args = () if device_capacity[0] in (8, 9, 12) else (mSFA, mSFB)
    return cute.compile(
        gemm_obj,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        stream,
        *sf_args,
        options="--enable-tvm-ffi",
    )
