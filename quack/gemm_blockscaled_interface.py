# Copyright (c) 2026, Tri Dao.

"""PyTorch-friendly interface for the MXFP8 blockscaled GEMM (SM90 + SM100).

Layout overview:
  A:       (M, K)     or (L, M, K)       dtype float8_e4m3fn, K-contiguous (row-major)
  B:       (K, N)     or (L, K, N)       dtype float8_e4m3fn, K-contiguous (col-major)
  A_scale: (M, K/V)   or (L, M, K/V)     dtype float8_e8m0fnu/float32, K-contiguous
  B_scale: (K/V, N)   or (L, K/V, N)     dtype float8_e8m0fnu/float32, K-contiguous
  out:     (M, N)     or (L, M, N)       dtype bfloat16/float16, contiguous

K-block size V differs by target:
  - SM100 (Blackwell): V = 32  (true MX granularity)
  - SM90  (Hopper):    V = 128 (coarser, faster on H100)

"K-contiguous" means stride 1 on the K axis. This matches how torchao/cuBLAS
use `torch._scaled_mm(a, b.t(), ...)`: weight stored as `(N, K)` row-major,
pass `W.mT` (zero-copy view of shape `(K, N)` with K-contig) as B. The
interface applies `.mT` internally to reach the `(N, K) K-major` layout the
kernels consume; no data is copied.

File organization (top to bottom):
  1. Helpers
  2. Quantization (`quantize_act`, `quantize_*_sm{90,100}`, `quantize_weight_sm90`)
  3. SM100 GEMM (`mxfp8_gemm_sm100`, etc.)
  4. SM90 GEMM  (`mxfp8_gemm_act_sm90`, etc.)
  5. Unified dispatch (`mxfp8_gemm`, `mxfp8_gemm_act`)
  6. Reference / utility
"""

from functools import lru_cache, partial
from typing import Optional, Tuple

import torch
from torch import Tensor
import triton
import triton.language as tl

import cutlass
import cutlass.cute as cute

from quack.activation import act_fn_map, gate_fn_map
from quack.blockscaled_gemm_utils import (
    _make_compile_tensor_like,
    ceil_div,
    compile_blockscaled_gemm_tvm_ffi,
    pack_scale_2d_to_blocked_contig,
    scale_blocked_for_cublas,
    scale_view_for_kernel,
)
from quack.cache_utils import COMPILE_ONLY, jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.gemm_act import GemmActMixin, GemmActSm90, GemmActSm100, GemmGatedSm90, GemmGatedSm100
from quack.gemm_config import GemmConfig
from quack.gemm_default_epi import GemmDefaultSm100
from quack.gemm_interface import (
    Activation,
    GatedActivation,
    _concat_interleave_bias,
    _empty_k_matmul_into,
    gated_to_pytorch_fn_map,
)
from quack.gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    div_for_dtype,
    get_major,
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
    perm3d_single,
)
from quack.mx_utils import to_mx, to_mx_2d
from quack.quant import blockwise_quant

_SF_VEC_SIZE_SM100 = 32  # SM100 K-block size
_SF_VEC_SIZE_SM90 = 128  # SM90 K-block size (activations and weights)
_WEIGHT_BLOCK_N_SM90 = 128  # SM90 N-block size for weight scales


def default_config(device):
    cap = get_device_capacity(device)[0]
    if cap == 9:
        return GemmConfig(
            tile_m=128,
            tile_n=128,
            cluster_m=1,
            cluster_n=2,
            pingpong=False,
            is_dynamic_persistent=False,
        )
    else:
        raise NotImplementedError("Currently only Hopper is supported")


def _default_tiler_cluster_sm100(m: int, n: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Pick a reasonable default (mma_tiler_mn, cluster_shape_mn)."""
    if m >= 512 and n >= 128:
        return (256, 128), (2, 1)
    return (128, 128), (1, 1)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def _f32_to_e8m0(scale_f32: torch.Tensor) -> torch.Tensor:
    """Convert float32 power-of-2 scales to E8M0 bytes.

    Extracts the biased exponent byte: (f32_bits >> 23) & 0xFF.
    """
    e8m0_byte = ((scale_f32.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return e8m0_byte.view(torch.float8_e8m0fnu)


def quantize_act_sm100(x: Tensor) -> Tuple[Tensor, Tensor]:
    """SM100 activation quantization: 32-element K-blocks, per-row.
    Returns (qdata: float8_e4m3fn, scale: float8_e8m0fnu) in torchao layout."""
    assert x.shape[-1] % _SF_VEC_SIZE_SM100 == 0, (
        f"last dim ({x.shape[-1]}) must be divisible by {_SF_VEC_SIZE_SM100}"
    )
    return to_mx(x.contiguous(), _SF_VEC_SIZE_SM100)


def quantize_act(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Auto-dispatch activation quantization based on device capability.

    SM90 uses `blockwise_quant` (128-element K-blocks, M-innermost f32 scales) — the same
    fast CuTe quantizer the GEMM's SFA expects. For varlen/grouped-M, quantize here (unpadded)
    then dQaccum-pad the scales with `grouped_scale_to_dqaccum` / `permute_scale_to_dqaccum`.
    """
    cap = torch.cuda.get_device_capability(torch.cuda.current_device())[0]
    if cap == 9:
        return blockwise_quant(x, block_size=_SF_VEC_SIZE_SM90, scale_transpose=True)
    elif cap == 10:
        return quantize_act_sm100(x)
    else:
        raise NotImplementedError(f"sm_{cap}0 not supported")


# ---------------------------------------------------------------------------
# SM100 GEMM (Blackwell)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _compile_cached_sm100(
    m: int,
    n: int,
    k: int,
    l: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    out_torch_dtype,
    ab_dtype_cutlass,
    sf_dtype_cutlass,
):
    """Compile kernel for a given (shape, dtype, tiler, cluster) and cache it."""
    dev = torch.device("cuda")
    rm = ceil_div(m, 128)
    rn = ceil_div(n, 128)
    rk = ceil_div(k // _SF_VEC_SIZE_SM100, 4)
    # K-major: (l, m, k) contiguous, viewed as (m, k, l) strides (k, 1, m*k)
    fake_mA = torch.empty(l, m, k, dtype=torch.float8_e4m3fn, device=dev).permute(1, 2, 0)
    fake_mB = torch.empty(l, n, k, dtype=torch.float8_e4m3fn, device=dev).permute(1, 2, 0)
    # N-major: (l, m, n) contiguous, viewed as (m, n, l) strides (n, 1, m*n)
    fake_mD = torch.empty(l, m, n, dtype=out_torch_dtype, device=dev).permute(1, 2, 0)
    fake_sc_A = torch.empty(l, rm, rk, 512, dtype=torch.float8_e8m0fnu, device=dev)
    fake_sc_B = torch.empty(l, rn, rk, 512, dtype=torch.float8_e8m0fnu, device=dev)
    fake_mSFA = scale_view_for_kernel(fake_sc_A, m, k // _SF_VEC_SIZE_SM100, l)
    fake_mSFB = scale_view_for_kernel(fake_sc_B, n, k // _SF_VEC_SIZE_SM100, l)
    return compile_blockscaled_gemm_tvm_ffi(
        ab_dtype_cutlass,
        sf_dtype_cutlass,
        _SF_VEC_SIZE_SM100,
        torch2cute_dtype_map[out_torch_dtype],
        mma_tiler_mn,
        cluster_shape_mn,
        fake_mA,
        fake_mB,
        fake_mD,
        fake_mSFA,
        fake_mSFB,
    )


def _as_3d(x: Tensor, ndim_in: int) -> Tensor:
    """Add a leading batch dim if input is 2D. Returns a view."""
    if ndim_in == 2:
        return x.unsqueeze(0)
    return x


def _to_kernel_layout_sm100(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
) -> Tuple[int, int, int, int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool]:
    """Normalize shapes/strides, validate, and repack scales. Returns
    (m, n, k, l, mA_mkl, mB_nkl, sc_contig_A, sc_contig_B, sfa_view, sfb_view, was_2d).

    A: (M,K) or (L,M,K) K-contig.  B: (K,N) or (L,K,N) K-contig.
    A_scale: (M,K/32) or (L,M,K/32) K-contig.  B_scale: (K/32,N) or (L,K/32,N) K-contig.
    """
    assert A.dtype == torch.float8_e4m3fn, f"A dtype must be float8_e4m3fn, got {A.dtype}"
    assert B.dtype == torch.float8_e4m3fn, f"B dtype must be float8_e4m3fn, got {B.dtype}"
    assert A_scale.dtype in (torch.float8_e8m0fnu, torch.float32), (
        f"A_scale dtype must be float8_e8m0fnu or float32, got {A_scale.dtype}"
    )
    assert B_scale.dtype in (torch.float8_e8m0fnu, torch.float32), (
        f"B_scale dtype must be float8_e8m0fnu or float32, got {B_scale.dtype}"
    )
    was_2d = A.dim() == 2
    # Flip B from (K,N) to (N,K) via .mT (zero-copy). User's B K-contig → .mT K-contig.
    A3 = _as_3d(A, A.dim())  # (l, m, k) K-contig row-major expected
    B3 = _as_3d(B, B.dim()).mT  # (l, n, k) K-contig (view) from (l, k, n)
    l, m, k = A3.shape
    l2, n, k2 = B3.shape
    assert l == l2, f"batch mismatch: A={l}, B={l2}"
    assert k == k2, f"K mismatch: A K={k}, B K={k2}"
    assert k % _SF_VEC_SIZE_SM100 == 0, f"K ({k}) must be divisible by {_SF_VEC_SIZE_SM100}"
    assert A3.stride(-1) == 1, "A must be K-contiguous (stride 1 on K)"
    assert B3.stride(-1) == 1, (
        "B must be K-contiguous on its K axis (pass .mT of an (N,K) row-major tensor)"
    )
    sf_k = k // _SF_VEC_SIZE_SM100
    as3 = _as_3d(A_scale, A_scale.dim())  # expected (l, m, sf_k) K-contig row-major
    bs3 = _as_3d(B_scale, B_scale.dim()).mT  # (l, n, sf_k) K-contig (view) from (l, sf_k, n)
    assert as3.stride(-1) == 1, "A_scale must be K-contiguous"
    assert bs3.stride(-1) == 1, (
        "B_scale must be K-contiguous on its K axis (pass .mT of an (N, K/32) row-major tensor)"
    )
    assert as3.shape == (l, m, sf_k), (
        f"A_scale shape: expected (l={l},m={m},sf_k={sf_k}) K-contig, got {tuple(as3.shape)}"
    )
    assert bs3.shape == (l, n, sf_k), (
        f"B_scale shape: expected .mT of (l={l},sf_k={sf_k},n={n}) -> ({l},{n},{sf_k}), got {tuple(bs3.shape)}"
    )
    # Force row-major contiguous for packer/kernel consumption.
    # A3 / B3 are views — .contiguous() materializes (l,m,k) / (l,n,k) row-major.
    A3_c = A3.contiguous()
    B3_c = B3.contiguous()
    # (l, m, k) -> (m, k, l) K-major view (no copy; strides (k, 1, m*k))
    mA_mkl = A3_c.permute(1, 2, 0)
    mB_nkl = B3_c.permute(1, 2, 0)
    sc_contig_A = pack_scale_2d_to_blocked_contig(as3.contiguous())
    sc_contig_B = pack_scale_2d_to_blocked_contig(bs3.contiguous())
    sfa_view = scale_view_for_kernel(sc_contig_A, m, sf_k, l)
    sfb_view = scale_view_for_kernel(sc_contig_B, n, sf_k, l)
    return m, n, k, l, mA_mkl, mB_nkl, sc_contig_A, sc_contig_B, sfa_view, sfb_view, was_2d


def mxfp8_gemm_out_sm100(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    *,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
) -> None:
    """MXFP8 blockscaled GEMM with pre-allocated output. See module doc for shape conventions."""
    m, n, k, l, mA, mB, _scA, _scB, sfa, sfb, was_2d = _to_kernel_layout_sm100(
        A, B, A_scale, B_scale
    )
    out_dtype = out.dtype
    assert out_dtype in torch2cute_dtype_map, f"unsupported out dtype: {out_dtype}"
    expected_out_shape = (m, n) if was_2d else (l, m, n)
    assert tuple(out.shape) == expected_out_shape, (
        f"out shape {tuple(out.shape)} != expected {expected_out_shape}"
    )
    assert out.is_contiguous(), "out must be contiguous"
    # View caller's contiguous (M,N) or (L,M,N) as (M,N,L) N-major strided view, no copy.
    out_3d = out.unsqueeze(0) if was_2d else out  # (l, m, n)
    mD = out_3d.permute(1, 2, 0)  # (m, n, l), strides (n, 1, m*n)
    if mma_tiler_mn is None or cluster_shape_mn is None:
        tlr, clu = _default_tiler_cluster_sm100(m, n)
        mma_tiler_mn = mma_tiler_mn or tlr
        cluster_shape_mn = cluster_shape_mn or clu
    if not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        _SF_VEC_SIZE_SM100,
        torch2cute_dtype_map[out_dtype],
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        "k",
        "k",
        "n",
    ):
        raise ValueError(
            f"unsupported config: m={m}, n={n}, k={k}, l={l}, "
            f"tiler={mma_tiler_mn}, cluster={cluster_shape_mn}"
        )
    runner = _compile_cached_sm100(
        m,
        n,
        k,
        l,
        mma_tiler_mn,
        cluster_shape_mn,
        out_dtype,
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
    )
    runner(mA, mB, mD, sfa, sfb)


def mxfp8_gemm_sm100(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Optional[Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
) -> Tensor:
    """MXFP8 blockscaled GEMM. Allocates output if not provided."""
    if out is None:
        # A: (M,K) or (L,M,K); B: (K,N) or (L,K,N); out: (M,N) or (L,M,N)
        if A.dim() == 2:
            out_shape = (A.shape[0], B.shape[1])
        else:
            out_shape = (A.shape[0], A.shape[1], B.shape[2])
        out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    mxfp8_gemm_out_sm100(
        A,
        B,
        A_scale,
        B_scale,
        out,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )
    return out


# ---------------------------------------------------------------------------
# SM90 GEMM (Hopper)
# ---------------------------------------------------------------------------


@jit_cache
def _compile_mxfp8_gemm_act_sm90(
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
    concat_layout,
    device_capacity,
    sr_seed_mode=0,
    use_tma_gather=False,
):
    is_gated = activation in gate_fn_map
    sm = device_capacity[0]
    if sm == 9:
        GemmCls = GemmGatedSm90 if is_gated else GemmActSm90
    else:
        GemmCls = GemmGatedSm100 if is_gated else GemmActSm100

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

    pa_leading = 1 if postact_major == "n" else 0
    pa_n = cute.sym_int() if is_gated else n
    div_pa = div_for_dtype(postact_dtype)
    pa_leading_dim = 1 if is_gated else pa_leading
    pa_shape = (m, pa_n) if varlen_m else (m, pa_n, l)
    mAuxOut = fake_tensor(postact_dtype, pa_shape, leading_dim=pa_leading_dim, divisibility=div_pa)

    mRowVec = (
        fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4) if rowvec_dtype else None
    )
    if colvec_ndim == 2:
        mColVec = (
            fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
            if colvec_dtype
            else None
        )
    elif colvec_ndim == 1:
        mColVec = (
            fake_tensor(colvec_dtype, (m,), leading_dim=0, divisibility=4) if colvec_dtype else None
        )
    else:
        mColVec = None

    from cutlass import Int32
    from cutlass.cute.runtime import make_ptr

    act_fn = gate_fn_map[activation] if is_gated else act_fn_map[activation]

    def fake_scalar(mode):
        if mode == 0:
            return None
        elif mode == 1:
            return Int32(0)
        else:
            return make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=4)

    epi_args = GemmCls.EpilogueArguments(
        mAuxOut,
        act_fn,
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        rounding_mode=0,  # RoundingMode.RN, Constexpr baked at compile time
        sr_seed=fake_scalar(sr_seed_mode),
    )
    scheduler_args = make_fake_scheduler_args((is_dynamic_persistent and sm == 9), False, l)
    varlen_args = make_fake_varlen_args(varlen_m, False, gather_A, m if varlen_m else None)

    if sm == 9:
        # SM90 blockscaled: float32 scales.
        # A scale: dispatch produces (m, sf_k, l) (or (m, sf_k) varlen) with M innermost
        # so the TMA atom can do a single contiguous (BLOCK_M, 1) burst per K-stage.
        # B scale: (l, n_blocks, sf_k) — read directly from gmem in math warps (no TMA).
        sf_k_sym = cute.sym_int()
        n_blocks_sym = cute.sym_int()
        if varlen_m:
            # SFA's M extent is dQaccum-padded (total_padded_m), independent of A's
            # total_m — give it its own symbol so the kernel does NOT bake in
            # mSFA.shape[0] == mA.shape[0] (which no longer holds after padding).
            padded_m_sym = cute.sym_int()
            fake_sfa = fake_tensor(
                cutlass.Float32, (padded_m_sym, sf_k_sym), leading_dim=0, divisibility=1
            )
        else:
            fake_sfa = fake_tensor(cutlass.Float32, (m, sf_k_sym, l), leading_dim=0, divisibility=1)
        fake_sfb = fake_tensor(
            cutlass.Float32, (l, n_blocks_sym, sf_k_sym), leading_dim=2, divisibility=1
        )
        return compile_gemm_kernel(
            partial(GemmCls, sf_vec_size=_SF_VEC_SIZE_SM90, weight_n_block=_WEIGHT_BLOCK_N_SM90),
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
            mSFA=fake_sfa,
            mSFB=fake_sfb,
            use_tma_gather=use_tma_gather,
            concat_layout=concat_layout or None,
        )

    # SM100/SM110: blockscaled path — inject sf_vec_size and pass fake scale tensors.
    # Layout is (l, rm, rk, 512) contiguous; dynamic_layout=True lets TVM FFI
    # accept any concrete shape at runtime.
    sc_fake = torch.empty(1, 1, 1, 512, dtype=torch.float8_e8m0fnu, device="cuda")
    mSFA = _make_compile_tensor_like(sc_fake, cutlass.Float8E8M0FNU, dynamic_layout=True)
    mSFB = _make_compile_tensor_like(sc_fake, cutlass.Float8E8M0FNU, dynamic_layout=True)
    return compile_gemm_kernel(
        partial(GemmCls, sf_vec_size=_SF_VEC_SIZE_SM100),
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
        mSFA=mSFA,
        mSFB=mSFB,
        use_tma_gather=use_tma_gather,
        concat_layout=concat_layout or None,
    )


def mxfp8_gemm_act_dispatch_sm90(
    A: Tensor,  # (l, m, k) K-contig
    B: Tensor,  # (l, n, k) K-contig
    A_scale: Tensor,  # (l, m, k/32) K-contig
    B_scale: Tensor,  # (l, n, k/32) K-contig
    D: Optional[Tensor],  # (l, m, n) or None (preact_out)
    C: Optional[Tensor],  # (l, m, n) or None
    PostAct: Tensor,  # (l, m, n//2) for gated
    tile_count_semaphore: Optional[Tensor],
    activation: str,
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    tile_K: int | None = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,
    colvec_bias: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    use_tma_gather: bool = False,
    concat_layout: tuple | None = None,
) -> None:
    varlen_m = cu_seqlens_m is not None
    gather_A = A_idx is not None

    A_p = perm3d_single(A, varlen_m)
    B_p = perm3d_single(B)
    D_p = perm3d_single(D, varlen_m) if D is not None else None
    C_p = perm3d_single(C, varlen_m) if C is not None else None
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
    sm = device_capacity[0]
    assert sm in (9, 10, 11), "mxfp8_gemm_act_dispatch_sm90 requires SM90, SM100, or SM110"

    if sm == 9 and not GemmActSm90.is_valid_dtypes(
        a_dtype, b_dtype, cutlass.Float32, d_dtype, a_major, b_major
    ):
        raise ValueError(
            f"unsupported SM90 mxfp8 config: a_dtype={a_dtype}, b_dtype={b_dtype}, "
            f"d_dtype={d_dtype}, a_major={a_major}, b_major={b_major} "
            f"(SM90 fp8 requires K-major A and B)"
        )

    concat_layout_key = tuple(sorted(concat_layout)) if concat_layout else ()
    compiled_fn = _compile_mxfp8_gemm_act_sm90(
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
        (tile_M, tile_N, tile_K) if tile_K is not None else (tile_M, tile_N),
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
        concat_layout_key,
        device_capacity,
        use_tma_gather=use_tma_gather,
    )

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    epi_args = GemmActMixin.EpilogueArguments(
        PostAct_p,
        None,  # act_fn is Constexpr, baked at compile time
        mRowVecBroadcast=rowvec_bias,
        mColVecBroadcast=colvec_bias,
        rounding_mode=None,  # Constexpr, baked at compile time
        sr_seed=None,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters, max_swizzle_size, tile_count_semaphore
    )
    varlen_args = make_varlen_args(cu_seqlens_m, None, A_idx)

    if sm == 9:
        # Scales are float32.
        # SFA: kernel sees logical shape (m, sf_k, l) (or (m, sf_k) for varlen) with
        # *M as the innermost contiguous dim* — matches the DeepGEMM a.cu reference
        # convention. TMA loads (BLOCK_M, 1) per K-stage as a single 512B burst from
        # M-contiguous memory; a K-major (sf_k stride 1) layout would force TMA to
        # do a strided gather and reads wrong values. The transpose+contiguous below
        # rematerializes A_scale as (..., sf_k, m) then transposes back to a
        # (..., m, sf_k) view with M innermost.
        if varlen_m:
            sfa_sm90 = A_scale
        else:
            # sfa_2d_to_3d = A_scale.transpose(-2, -1).contiguous()  # (l, sf_k, m)
            sfa_sm90 = A_scale.permute(1, 2, 0)  # view: (m, sf_k, l), M innermost
        sfb_sm90 = B_scale.contiguous()  # (l, n_blocks, sf_k); read directly from gmem
        compiled_fn(
            A_p,
            B_p,
            D_p,
            C_p,
            epi_args,
            scheduler_args,
            varlen_args,
            sfa_sm90,
            sfb_sm90,
            None,
        )
    else:
        # SM100/SM110: pack scales and pass to blockscaled kernel.
        # Scales may be float32 (from quantize_act_*) — convert to E8M0 first.
        k = A.shape[-1]
        l = B.shape[0]
        n = B.shape[1]
        sf_k = k // _SF_VEC_SIZE_SM100
        a_scale_e8m0 = _f32_to_e8m0(A_scale) if A_scale.dtype == torch.float32 else A_scale
        b_scale_e8m0 = _f32_to_e8m0(B_scale) if B_scale.dtype == torch.float32 else B_scale
        if varlen_m:
            # A_scale: (total_m, sf_k) — dQaccum-padded layout. Each expert's rows
            # start at the next 128-row tile boundary after the previous expert, with
            # one extra tile of slack per expert boundary so the kernel's
            # VarlenManager.offset_batch_SFA decodes offsets correctly.
            total_m = A.shape[0]
            seqlens_m = (cu_seqlens_m[1:] - cu_seqlens_m[:-1]).cpu().tolist()
            tile = 128
            total_padded_rm = (total_m + tile - 1) // tile + (l - 1)
            total_padded_m = total_padded_rm * tile
            sa_padded = torch.zeros(
                total_padded_m, sf_k, dtype=torch.float8_e8m0fnu, device=A_scale.device
            )
            row = 0
            for i, m_i in enumerate(seqlens_m):
                row_padded = (row // tile + i) * tile
                sa_padded[row_padded : row_padded + m_i] = a_scale_e8m0[row : row + m_i]
                row += m_i
            sc_contig_A = pack_scale_2d_to_blocked_contig(sa_padded.view(1, total_padded_m, sf_k))
            sfa = scale_view_for_kernel(sc_contig_A, total_padded_m, sf_k, 1)
        else:
            m = A.shape[1]
            sc_contig_A = pack_scale_2d_to_blocked_contig(a_scale_e8m0.contiguous())
            sfa = scale_view_for_kernel(sc_contig_A, m, sf_k, l)
        sc_contig_B = pack_scale_2d_to_blocked_contig(b_scale_e8m0.contiguous())
        sfb = scale_view_for_kernel(sc_contig_B, n, sf_k, l)
        compiled_fn(
            A_p,
            B_p,
            D_p,
            C_p,
            epi_args,
            scheduler_args,
            varlen_args,
            sfa,
            sfb,
            None,
        )


# @autotune(
#     configs=[AutotuneConfig(config=c) for c in get_all_configs("gated")],
#     key=["activation", "dynamic_scheduler"],
#     prune_configs_by={"early_config_prune": prune_invalid_gemm_configs},
# )
def mxfp8_gemm_gated_tuned_sm90(
    # (M, K) or or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N)
    A_scale: Tensor,
    B_scale: Tensor,
    # (M, N) or (L, M, N) or (total_M, N) if varlen_m - None if not storing preact
    preact_out: Optional[Tensor],
    postact_out: Tensor,  # (M, N//2) or (L, M, N//2) or (total_M, N//2) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: GatedActivation = "swiglu",
    cu_seqlens_m: Optional[Tensor] = None,  # (L+1), int32
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
) -> None:
    if config is None:
        config = default_config(A.device)
    varlen_m = cu_seqlens_m is not None
    if varlen_m:
        assert not config.swap_ab, "Variable-length sequences not supported with swap_ab"
    if A.ndim == 2 and not varlen_m:
        A = A.unsqueeze(0)  # (1, M, K)
        A_scale = A_scale.unsqueeze(0)
    B, B_scale = B.mT, B_scale.mT  # (N, K) or (L, N, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)  # (1, N, K)
        B_scale = B_scale.unsqueeze(0)
    if C is not None and C.ndim == 2 and not varlen_m:
        C = C.unsqueeze(0)  # (1, M, N)
    if preact_out is not None and preact_out.ndim == 2 and not varlen_m:
        D = preact_out.unsqueeze(0)
    else:
        D = preact_out
    if postact_out.ndim == 2 and not varlen_m:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)  # (L, N)
    if concat_layout and "bias" in concat_layout:
        if bias is not None and bias.dtype.itemsize >= 4:
            bias_key = "mColVecBroadcast" if config.swap_ab else "mRowVecBroadcast"
            concat_layout = tuple(bias_key if k == "bias" else k for k in concat_layout)
        else:
            concat_layout = tuple(k for k in concat_layout if k != "bias")
            if bias is not None:
                bias = _concat_interleave_bias(bias)
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    mxfp8_gemm_act_dispatch_sm90(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        A_scale if not config.swap_ab else B_scale,
        B_scale if not config.swap_ab else A_scale,
        (D if not config.swap_ab else D.mT) if D is not None else None,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        rowvec_bias=bias if not config.swap_ab else None,
        colvec_bias=bias if config.swap_ab else None,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        use_tma_gather=config.use_tma_gather,
        concat_layout=concat_layout,
    )


def mxfp8_gemm_act_tuned_sm90(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    A_scale: Tensor,
    B_scale: Tensor,
    preact_out: Optional[Tensor],  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Activation = None,
    cu_seqlens_m: Optional[Tensor] = None,  # (L+1), int32
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
) -> None:
    if config is None:
        config = default_config(A.device)
    varlen_m = cu_seqlens_m is not None
    if varlen_m:
        assert not config.swap_ab, "Variable-length sequences not supported with swap_ab"
    if A.ndim == 2 and not varlen_m:
        A = A.unsqueeze(0)  # (1, M, K)
        A_scale = A_scale.unsqueeze(0)
    B, B_scale = B.mT, B_scale.mT  # (N, K) or (L, N, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)  # (1, N, K)
        B_scale = B_scale.unsqueeze(0)
    if C is not None and C.ndim == 2 and not varlen_m:
        C = C.unsqueeze(0)  # (1, M, N)
    if preact_out is not None and preact_out.ndim == 2 and not varlen_m:
        D = preact_out.unsqueeze(0)
    else:
        D = preact_out
    if postact_out.ndim == 2 and not varlen_m:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)  # (L, N)
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    mxfp8_gemm_act_dispatch_sm90(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        A_scale if not config.swap_ab else B_scale,
        B_scale if not config.swap_ab else A_scale,
        (D if not config.swap_ab else D.mT) if D is not None else None,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        rowvec_bias=bias if not config.swap_ab else None,
        colvec_bias=bias if config.swap_ab else None,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        use_tma_gather=config.use_tma_gather,
    )


def mxfp8_gemm_gated_out_sm90(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    A_scale: Tensor,
    B_scale: Tensor,
    preact_out: Optional[Tensor],  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Tensor,  # (M, N//2) or (L, M, N//2) or (total_M, N//2) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: GatedActivation = "swiglu",
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: Optional[str] = None,
) -> None:
    """GEMM with gated activation and pre-allocated output tensors."""
    # TODO: add tuning
    assert not tuned, "currently tuning is not available"
    fn = mxfp8_gemm_gated_tuned_sm90 if tuned else partial(mxfp8_gemm_gated_tuned_sm90, config=None)
    fn(
        A,
        B,
        A_scale,
        B_scale,
        preact_out,
        postact_out,
        C,
        bias,
        activation,
        cu_seqlens_m,
        A_idx,
        dynamic_scheduler,
        concat_layout=tuple(concat_layout.split(",")) if concat_layout else None,
    )


def mxfp8_gemm_act_out_sm90(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    A_scale: Tensor,
    B_scale: Tensor,
    preact_out: Optional[Tensor],  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Activation = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    tuned: bool = True,
) -> None:
    """GEMM with activation and pre-allocated output tensors."""
    # TODO: add tuning
    tuned = False
    fn = mxfp8_gemm_act_tuned_sm90 if tuned else partial(mxfp8_gemm_act_tuned_sm90, config=None)
    fn(
        A,
        B,
        A_scale,
        B_scale,
        preact_out,
        postact_out,
        C,
        bias,
        activation,
        cu_seqlens_m,
        A_idx,
        dynamic_scheduler,
    )


def mxfp8_gemm_act_sm90(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    A_scale: Tensor,
    B_scale: Tensor,
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Activation = None,
    preact_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    store_preact: bool = True,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
) -> Tuple[Optional[Tensor], Tensor]:
    """GEMM with activation (or gated activation) and optional output tensors."""
    is_gated = activation in gated_to_pytorch_fn_map
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
    varlen_m = cu_seqlens_m is not None
    if not varlen_m:
        # SM90 constraints (non-varlen): M % 8, and the "1d2d" block-scaled quant scheme:
        #   A_scale: (..., M,        K // 128)   from quantize_act_sm90    (1 × 128)
        #   B_scale: (..., K // 128, N // 128)   from quantize_weight_sm90 (128 × 128), passed as .mT
        m_dim = A.shape[-2]  # works for both 2D (M, K) and 3D (L, M, K)
        k_dim = A.shape[-1]
        n_dim = B.shape[-1]
        assert m_dim % 8 == 0, f"SM90 mxfp8 GEMM requires M % 8 == 0; got M={m_dim}"
        assert A_scale.shape[-2:] == (m_dim, k_dim // 128), (
            f"SM90 expects A_scale from quantize_act_sm90 (1x128): "
            f"shape (..., M={m_dim}, K/128={k_dim // 128}); got {tuple(A_scale.shape)}"
        )
        assert B_scale.shape[-2:] == (k_dim // 128, n_dim // 128), (
            f"SM90 expects B_scale from quantize_weight_sm90 (128x128, passed as .mT): "
            f"shape (..., K/128={k_dim // 128}, N/128={n_dim // 128}); got {tuple(B_scale.shape)}"
        )
    # Determine output shape based on gather_A
    if varlen_m:
        total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
        out_shape = (total_m, B.shape[-1])
    elif A.ndim == 2:
        out_shape = (A.shape[0], B.shape[-1])
    else:
        out_shape = (A.shape[0], A.shape[-2], B.shape[-1])
    postact_shape = (*out_shape[:-1], out_shape[-1] // 2) if is_gated else out_shape
    if preact_out is None and store_preact:
        preact_out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    if postact_out is None:
        postact_out = torch.empty(postact_shape, dtype=postact_dtype, device=A.device)
    # Empty-input fast path. For M=0 or N=0 the outputs are empty; for K=0
    # (A@B == 0) the no-bias / no-C surface yields preact=0 and act(0)=0 for
    # every supported activation, so both outputs are zero.
    if postact_out.numel() == 0 or A.numel() == 0:
        if preact_out is not None:
            _empty_k_matmul_into(preact_out)
        _empty_k_matmul_into(postact_out)
        return preact_out, postact_out
    concat_str = ",".join(concat_layout) if concat_layout else None
    if is_gated:
        mxfp8_gemm_gated_out_sm90(
            A,
            B,
            A_scale,
            B_scale,
            preact_out,
            postact_out,
            C,
            bias,
            activation,
            cu_seqlens_m,
            A_idx,
            dynamic_scheduler,
            tuned,
            concat_layout=concat_str,
        )
    else:
        mxfp8_gemm_act_out_sm90(
            A,
            B,
            A_scale,
            B_scale,
            preact_out,
            postact_out,
            C,
            bias,
            activation,
            cu_seqlens_m,
            A_idx,
            dynamic_scheduler,
            tuned,
        )
    return preact_out, postact_out


# ---------------------------------------------------------------------------
# Unified dispatch entry points
# ---------------------------------------------------------------------------
# Top-level public API. These detect the device capability at runtime and
# route to the SM-specific implementation. Callers that don't care about the
# underlying hardware should use these; callers that need to pin a specific
# implementation can still call the *_sm90 / *_sm100 functions directly.


def mxfp8_gemm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Optional[Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
) -> Tensor:
    """MXFP8 blockscaled GEMM. Dispatches to SM90 or SM100 by device capability.

    SM100 path expects 32-element K-block scales (`quantize_act_sm100`).
    SM90 path expects 128-element K-block scales (`quantize_act_sm90`,
    `quantize_weight_sm90`). Caller is responsible for using the matching
    quantization for the target device.
    """
    cap = torch.cuda.get_device_capability(torch.cuda.current_device())[0]
    if cap == 10:
        return mxfp8_gemm_sm100(A, B, A_scale, B_scale, out=out, out_dtype=out_dtype, **kwargs)
    if cap == 9:
        # SM90 has no plain-GEMM entry; use the act path with activation=None.
        if out is None:
            out_shape = (A.shape[0], B.shape[-1]) if A.dim() == 2 else (*A.shape[:-1], B.shape[-1])
            out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
        mxfp8_gemm_act_sm90(
            A,
            B,
            A_scale,
            B_scale,
            activation=None,
            preact_out=None,
            postact_out=out,
            store_preact=False,
            **kwargs,
        )
        return out
    raise NotImplementedError(f"mxfp8_gemm: sm_{cap}0 not supported")


def mxfp8_gemm_act(*args, **kwargs) -> Tuple[Optional[Tensor], Tensor]:
    """GEMM + (optionally gated) activation. SM90-only at present.

    See `mxfp8_gemm_act_sm90` for the full signature.
    """
    cap = torch.cuda.get_device_capability(torch.cuda.current_device())[0]
    if cap == 9:
        return mxfp8_gemm_act_sm90(*args, **kwargs)
    raise NotImplementedError(f"mxfp8_gemm_act: sm_{cap}0 not supported (SM90 only)")


# ---------------------------------------------------------------------------
# Reference / utility
# ---------------------------------------------------------------------------


def mxfp8_gemm_quantize_sm100(
    A: Tensor,
    B: Tensor,
    out: Optional[Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
) -> Tensor:
    """High-level: quantize bf16 A, B_as_NK to MXFP8, then run C = A @ B_as_NK.mT.
    Inputs: A=(M,K)/(L,M,K), B_as_NK=(N,K)/(L,N,K) bf16/fp32. Quantization
    scales along the last (K) dim. Returned output has shape (M,N)/(L,M,N)."""
    A_q, A_sc = quantize_act(A)
    B_q, B_sc = quantize_act(B)
    # B_q, B_sc are (..., N, K) / (..., N, K/32). Flip to (..., K, N) / (..., K/32, N)
    # K-contig zero-copy views to match the interface convention.
    return mxfp8_gemm_sm100(
        A_q,
        B_q.mT,
        A_sc,
        B_sc.mT,
        out=out,
        out_dtype=out_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )


def mxfp8_gemm_cublas_sm100(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Reference path via torch._scaled_mm. Requires l=1 (or 2D inputs)."""
    m, n, k, l, _mA, _mB, sc_A, sc_B, _sfa, _sfb, was_2d = _to_kernel_layout_sm100(
        A, B, A_scale, B_scale
    )
    assert l == 1, "torch._scaled_mm MXFP8 path is 2D only; pass 2D inputs or l=1"
    # torch._scaled_mm: A=(M,K) row-major, B=(K,N) col-major (both K-contig) -- same layout user gave us.
    a2d = A if A.dim() == 2 else A.squeeze(0)
    b2d = B if B.dim() == 2 else B.squeeze(0)
    sca = scale_blocked_for_cublas(sc_A, m, k // _SF_VEC_SIZE_SM100, 0)
    scb = scale_blocked_for_cublas(sc_B, n, k // _SF_VEC_SIZE_SM100, 0)
    out = torch._scaled_mm(
        a2d,
        b2d,
        scale_a=sca,
        scale_b=scb,
        out_dtype=out_dtype,
    )
    return out if was_2d else out.unsqueeze(0)


def mxfp8_gemm_ref_sm100(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Dequantize + plain matmul reference. A=(M,K), B=(K,N)."""
    was_2d = A.dim() == 2
    # (l, m, k)
    A3 = _as_3d(A, A.dim()).float()
    # B is (K, N)/(L, K, N); flip to (l, n, k) for dequant by last-dim
    B3 = _as_3d(B, B.dim()).mT.contiguous().float()
    as3 = _as_3d(A_scale, A_scale.dim()).float()
    bs3 = _as_3d(B_scale, B_scale.dim()).mT.contiguous().float()
    a_dq = A3 * as3.repeat_interleave(_SF_VEC_SIZE_SM100, dim=-1)
    b_dq = B3 * bs3.repeat_interleave(_SF_VEC_SIZE_SM100, dim=-1)
    out3 = torch.einsum("lmk,lnk->lmn", a_dq, b_dq).to(out_dtype)
    return out3.squeeze(0) if was_2d else out3

@triton.jit
def _quantize_weight_sm90_kernel(
    w_ptr,  # (M, K) bf16/f32, row-major
    q_ptr,  # (M, K) float8_e4m3fn
    sc_ptr,  # (M // 128, K // 128) f32
    K,
    nbk,  # K // 128 (scale row stride)
    BLOCK: tl.constexpr,  # 128 (block_rows == block_cols == sf_vec == weight_n_block)
):
    # One 128x128 MXFP8 block per program. Bit-exact with quack.mx_utils.to_mx_2d
    # (FLOOR-mode e8m0 scale): amax over the tile -> power-of-2 dequant scale -> quantize.
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_k = pid_k * BLOCK + tl.arange(0, BLOCK)
    w_off = offs_m[:, None] * K + offs_k[None, :]
    tile = tl.load(w_ptr + w_off).to(tl.float32)
    amax = tl.max(tl.abs(tile))
    # Per-block power-of-2 quant scale: FP8_MAX/amax truncated to a power of two by zeroing
    # the mantissa (& 0xFF800000). Quantize with that same pow2 scale and store its reciprocal
    # as the dequant scale, so q * stored_scale == tile up to fp8 rounding (the SM90 GEMM applies
    # the scale as a plain f32 multiply). amax==0 -> scale 1 (all-zero tile stays zero).
    scale = tl.where(amax == 0, 1.0, 448.0 / amax)
    scale = (scale.to(tl.uint32, bitcast=True) & 0xFF800000).to(tl.float32, bitcast=True)
    q = tile * scale
    tl.store(q_ptr + w_off, q.to(q_ptr.dtype.element_ty))
    tl.store(sc_ptr + pid_m * nbk + pid_k, 1.0 / scale)


def quantize_weight_sm90(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """128x128 blockwise FP8 weight quantization in a single Triton kernel.

    Each 128x128 block gets one power-of-2 scale (FP8_MAX/amax, mantissa-truncated). Returns
    (qdata (..., N, K) float8_e4m3fn, scale (..., N//128, K//128) f32), where `scale` is the
    *dequant* scale (reciprocal) the SM90 GEMM multiplies by: `w ≈ qdata * scale`. N and K
    must be divisible by 128.
    """
    *batch, N, K = w.shape
    assert N % 128 == 0 and K % 128 == 0, f"N ({N}) and K ({K}) must be divisible by 128"
    w2d = w.reshape(-1, K).contiguous()
    M = w2d.shape[0]
    nbm, nbk = M // 128, K // 128
    q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=w.device)
    sc = torch.empty(nbm, nbk, dtype=torch.float32, device=w.device)
    _quantize_weight_sm90_kernel[(nbm, nbk)](w2d, q, sc, K, nbk, BLOCK=128)
    return q.reshape(*batch, N, K), sc.reshape(*batch, N // 128, K // 128)
