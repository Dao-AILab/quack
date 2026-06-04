# Copyright (c) 2026, Tri Dao.
"""PyTorch-friendly interface for the SM100 MXFP8 blockscaled GEMM.

Shape / layout conventions (matches torch.matmul, torch._scaled_mm, cuBLAS):
  A:       (M, K)     or (L, M, K)       dtype float8_e4m3fn, K-contiguous (row-major)
  B:       (K, N)     or (L, K, N)       dtype float8_e4m3fn, K-contiguous (col-major)
  A_scale: (M, K/32)  or (L, M, K/32)    dtype float8_e8m0fnu, K-contiguous
  B_scale: (K/32, N)  or (L, K/32, N)    dtype float8_e8m0fnu, K-contiguous
  out:     (M, N)     or (L, M, N)       dtype bfloat16/float16, contiguous

"K-contiguous" means stride 1 on the K axis. This matches how torchao/cuBLAS
use `torch._scaled_mm(a, b.t(), ...)`:
  - you store a weight as nn.Linear-style `W` of shape `(N, K)` row-major
  - you pass `W.mT` (a zero-copy view of shape (K, N) with K-contig) as B
The interface applies `.mT` internally to reach the `(N, K) K-major` layout
the quack kernel consumes. No data is copied.
"""

from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch import Tensor

import cutlass

from quack.activation import dgate_fn_map, gate_fn_map
from quack.blockscaled_gemm_utils import (
    ceil_div,
    compile_blockscaled_gemm_dgated_tvm_ffi,
    compile_blockscaled_gemm_gated_tvm_ffi,
    compile_blockscaled_gemm_tvm_ffi,
    pack_scale_2d_to_blocked_contig,
    scale_blocked_for_cublas,
    scale_view_for_kernel,
)
from quack.gemm_default_epi import GemmDefaultSm100
from quack.mx_utils import to_mx

_SF_VEC_SIZE = 32
_TORCH_TO_CUTLASS_D = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}


def _default_tiler_cluster(m: int, n: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Pick a reasonable default (mma_tiler_mn, cluster_shape_mn)."""
    if m >= 512 and n >= 128:
        return (256, 128), (2, 1)
    return (128, 128), (1, 1)


@lru_cache(maxsize=64)
def _compile_cached(
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
    rk = ceil_div(k // _SF_VEC_SIZE, 4)
    # K-major: (l, m, k) contiguous, viewed as (m, k, l) strides (k, 1, m*k)
    fake_mA = torch.empty(l, m, k, dtype=torch.float8_e4m3fn, device=dev).permute(1, 2, 0)
    fake_mB = torch.empty(l, n, k, dtype=torch.float8_e4m3fn, device=dev).permute(1, 2, 0)
    # N-major: (l, m, n) contiguous, viewed as (m, n, l) strides (n, 1, m*n)
    fake_mD = torch.empty(l, m, n, dtype=out_torch_dtype, device=dev).permute(1, 2, 0)
    fake_sc_A = torch.empty(l, rm, rk, 512, dtype=torch.float8_e8m0fnu, device=dev)
    fake_sc_B = torch.empty(l, rn, rk, 512, dtype=torch.float8_e8m0fnu, device=dev)
    fake_mSFA = scale_view_for_kernel(fake_sc_A, m, k // _SF_VEC_SIZE, l)
    fake_mSFB = scale_view_for_kernel(fake_sc_B, n, k // _SF_VEC_SIZE, l)
    return compile_blockscaled_gemm_tvm_ffi(
        ab_dtype_cutlass,
        sf_dtype_cutlass,
        _SF_VEC_SIZE,
        _TORCH_TO_CUTLASS_D[out_torch_dtype],
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


def _to_kernel_layout(
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
    assert A_scale.dtype == torch.float8_e8m0fnu
    assert B_scale.dtype == torch.float8_e8m0fnu
    was_2d = A.dim() == 2
    # Flip B from (K,N) to (N,K) via .mT (zero-copy). User's B K-contig → .mT K-contig.
    A3 = _as_3d(A, A.dim())  # (l, m, k) K-contig row-major expected
    B3 = _as_3d(B, B.dim()).mT  # (l, n, k) K-contig (view) from (l, k, n)
    l, m, k = A3.shape
    l2, n, k2 = B3.shape
    assert l == l2, f"batch mismatch: A={l}, B={l2}"
    assert k == k2, f"K mismatch: A K={k}, B K={k2}"
    assert k % _SF_VEC_SIZE == 0, f"K ({k}) must be divisible by {_SF_VEC_SIZE}"
    assert A3.stride(-1) == 1, "A must be K-contiguous (stride 1 on K)"
    assert B3.stride(-1) == 1, (
        "B must be K-contiguous on its K axis (pass .mT of an (N,K) row-major tensor)"
    )
    sf_k = k // _SF_VEC_SIZE
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


def mxfp8_gemm_out(
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
    m, n, k, l, mA, mB, _scA, _scB, sfa, sfb, was_2d = _to_kernel_layout(A, B, A_scale, B_scale)
    out_dtype = out.dtype
    assert out_dtype in _TORCH_TO_CUTLASS_D, f"unsupported out dtype: {out_dtype}"
    expected_out_shape = (m, n) if was_2d else (l, m, n)
    assert tuple(out.shape) == expected_out_shape, (
        f"out shape {tuple(out.shape)} != expected {expected_out_shape}"
    )
    assert out.is_contiguous(), "out must be contiguous"
    # View caller's contiguous (M,N) or (L,M,N) as (M,N,L) N-major strided view, no copy.
    out_3d = out.unsqueeze(0) if was_2d else out  # (l, m, n)
    mD = out_3d.permute(1, 2, 0)  # (m, n, l), strides (n, 1, m*n)
    if mma_tiler_mn is None or cluster_shape_mn is None:
        tlr, clu = _default_tiler_cluster(m, n)
        mma_tiler_mn = mma_tiler_mn or tlr
        cluster_shape_mn = cluster_shape_mn or clu
    if not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        _SF_VEC_SIZE,
        _TORCH_TO_CUTLASS_D[out_dtype],
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
    runner = _compile_cached(
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


def mxfp8_gemm(
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
    mxfp8_gemm_out(
        A,
        B,
        A_scale,
        B_scale,
        out,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )
    return out


def mxfp8_quantize(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Quantize a (..., K) bf16/fp32 tensor to MXFP8. Returns (qdata, scale_2d)
    in torchao-convention layout. Last dim (K) must be divisible by 32."""
    assert x.shape[-1] % _SF_VEC_SIZE == 0, (
        f"last dim ({x.shape[-1]}) must be divisible by {_SF_VEC_SIZE}"
    )
    return to_mx(x.contiguous(), _SF_VEC_SIZE)


def mxfp8_gemm_quantize(
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
    A_q, A_sc = mxfp8_quantize(A)
    B_q, B_sc = mxfp8_quantize(B)
    # B_q, B_sc are (..., N, K) / (..., N, K/32). Flip to (..., K, N) / (..., K/32, N)
    # K-contig zero-copy views to match the interface convention.
    return mxfp8_gemm(
        A_q,
        B_q.mT,
        A_sc,
        B_sc.mT,
        out=out,
        out_dtype=out_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )


def mxfp8_gemm_cublas(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Reference path via torch._scaled_mm. Requires l=1 (or 2D inputs)."""
    m, n, k, l, _mA, _mB, sc_A, sc_B, _sfa, _sfb, was_2d = _to_kernel_layout(A, B, A_scale, B_scale)
    assert l == 1, "torch._scaled_mm MXFP8 path is 2D only; pass 2D inputs or l=1"
    # torch._scaled_mm: A=(M,K) row-major, B=(K,N) col-major (both K-contig) -- same layout user gave us.
    a2d = A if A.dim() == 2 else A.squeeze(0)
    b2d = B if B.dim() == 2 else B.squeeze(0)
    sca = scale_blocked_for_cublas(sc_A, m, k // _SF_VEC_SIZE, 0)
    scb = scale_blocked_for_cublas(sc_B, n, k // _SF_VEC_SIZE, 0)
    out = torch._scaled_mm(
        a2d,
        b2d,
        scale_a=sca,
        scale_b=scb,
        out_dtype=out_dtype,
    )
    return out if was_2d else out.unsqueeze(0)


# ---------------------------------------------------------------------------
# Gated (SwiGLU/GeGLU) forward + backward — blockscaled MXFP8
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _compile_gated_cached(
    m, n_full, k, l, mma_tiler_mn, cluster_shape_mn, activation, out_torch_dtype
):
    dev = torch.device("cuda")
    n_out = n_full // 2
    rm, rn = ceil_div(m, 128), ceil_div(n_full, 128)
    rk = ceil_div(k // _SF_VEC_SIZE, 4)
    fake_mA = torch.empty(l, m, k, dtype=torch.float8_e4m3fn, device=dev).permute(1, 2, 0)
    fake_mB = torch.empty(l, n_full, k, dtype=torch.float8_e4m3fn, device=dev).permute(1, 2, 0)
    fake_aux = torch.empty(l, m, n_out, dtype=out_torch_dtype, device=dev).permute(1, 2, 0)
    fake_sc_A = torch.empty(l, rm, rk, 512, dtype=torch.float8_e8m0fnu, device=dev)
    fake_sc_B = torch.empty(l, rn, rk, 512, dtype=torch.float8_e8m0fnu, device=dev)
    fake_mSFA = scale_view_for_kernel(fake_sc_A, m, k // _SF_VEC_SIZE, l)
    fake_mSFB = scale_view_for_kernel(fake_sc_B, n_full, k // _SF_VEC_SIZE, l)
    return compile_blockscaled_gemm_gated_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        _SF_VEC_SIZE,
        _TORCH_TO_CUTLASS_D[out_torch_dtype],
        activation,
        mma_tiler_mn,
        cluster_shape_mn,
        fake_mA,
        fake_mB,
        fake_mSFA,
        fake_mSFB,
        fake_aux,
    )


def mxfp8_gemm_gated(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    activation: str,
    out: Optional[Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
) -> Tensor:
    assert activation in gate_fn_map, f"unknown gate activation {activation!r}"
    m, n_full, k, l, mA, mB, _scA, _scB, sfa, sfb, was_2d = _to_kernel_layout(
        A, B, A_scale, B_scale
    )
    assert n_full % 2 == 0, f"fused N ({n_full}) must be even"
    n_out = n_full // 2
    assert out_dtype in _TORCH_TO_CUTLASS_D, f"unsupported out dtype: {out_dtype}"
    if out is None:
        out = torch.empty((m, n_out) if was_2d else (l, m, n_out), dtype=out_dtype, device=A.device)
    else:
        assert out.dtype in _TORCH_TO_CUTLASS_D, f"unsupported out dtype: {out.dtype}"
    assert out.is_contiguous(), "out must be contiguous"
    out_3d = out.unsqueeze(0) if was_2d else out  # (l, m, n_out)
    mAuxOut = out_3d.permute(1, 2, 0)  # (m, n_out, l), N-major

    if mma_tiler_mn is None or cluster_shape_mn is None:
        tlr, clu = _default_tiler_cluster(m, n_full)
        mma_tiler_mn = mma_tiler_mn or tlr
        cluster_shape_mn = cluster_shape_mn or clu
    runner = _compile_gated_cached(
        m, n_full, k, l, mma_tiler_mn, cluster_shape_mn, activation, out.dtype
    )
    runner(mA, mB, sfa, sfb, mAuxOut)
    return out


@lru_cache(maxsize=64)
def _compile_dgated_cached(
    m, n, k, l, mma_tiler_mn, cluster_shape_mn, activation, postact_torch_dtype
):
    dev = torch.device("cuda")
    rm, rn = ceil_div(m, 128), ceil_div(n, 128)
    rk = ceil_div(k // _SF_VEC_SIZE, 4)
    fake_mA = torch.empty(l, m, k, dtype=torch.float8_e4m3fn, device=dev).permute(1, 2, 0)
    fake_mB = torch.empty(l, n, k, dtype=torch.float8_e4m3fn, device=dev).permute(1, 2, 0)
    # C (PreAct) / D (dPreAct) are (l, m, 2n) 16-bit, viewed as (l, m, n) f32, N-major.
    fake_C = torch.empty(l, m, 2 * n, dtype=postact_torch_dtype, device=dev)
    fake_D = torch.empty(l, m, 2 * n, dtype=postact_torch_dtype, device=dev)
    fake_C = fake_C.view(torch.float32).permute(1, 2, 0)
    fake_D = fake_D.view(torch.float32).permute(1, 2, 0)
    fake_aux = torch.empty(l, m, n, dtype=postact_torch_dtype, device=dev).permute(1, 2, 0)
    fake_sc_A = torch.empty(l, rm, rk, 512, dtype=torch.float8_e8m0fnu, device=dev)
    fake_sc_B = torch.empty(l, rn, rk, 512, dtype=torch.float8_e8m0fnu, device=dev)
    fake_mSFA = scale_view_for_kernel(fake_sc_A, m, k // _SF_VEC_SIZE, l)
    fake_mSFB = scale_view_for_kernel(fake_sc_B, n, k // _SF_VEC_SIZE, l)
    sf_dtype_d = _TORCH_TO_CUTLASS_D[postact_torch_dtype]
    return compile_blockscaled_gemm_dgated_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        _SF_VEC_SIZE,
        sf_dtype_d,  # implicit_dtype (the 16-bit packed element)
        sf_dtype_d,  # postact_dtype
        activation,
        mma_tiler_mn,
        cluster_shape_mn,
        fake_mA,
        fake_mB,
        fake_C,
        fake_D,
        fake_mSFA,
        fake_mSFB,
        fake_aux,
    )


def mxfp8_gemm_dgated(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    PreAct: Tensor,
    activation: str,
    dPreAct: Optional[Tensor] = None,
    PostAct: Optional[Tensor] = None,
    *,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    assert activation in dgate_fn_map, f"unknown dgate activation {activation!r}"
    m, n, k, l, mA, mB, _scA, _scB, sfa, sfb, was_2d = _to_kernel_layout(A, B, A_scale, B_scale)
    assert PreAct.dtype in _TORCH_TO_CUTLASS_D and PreAct.element_size() == 2, (
        f"PreAct must be bf16/fp16, got {PreAct.dtype}"
    )
    PreAct3 = _as_3d(PreAct, PreAct.dim())  # (l, m, 2n)
    assert tuple(PreAct3.shape) == (l, m, 2 * n), (
        f"PreAct shape: expected (l={l}, m={m}, 2n={2 * n}), got {tuple(PreAct3.shape)}"
    )
    assert PreAct3.is_contiguous(), "PreAct must be contiguous (row-major)"

    if dPreAct is None:
        dPreAct = torch.empty_like(PreAct)
    if PostAct is None:
        PostAct = torch.empty((m, n) if was_2d else (l, m, n), dtype=PreAct.dtype, device=A.device)
    dPreAct3 = _as_3d(dPreAct, dPreAct.dim())
    PostAct3 = _as_3d(PostAct, PostAct.dim())
    assert dPreAct3.is_contiguous() and PostAct3.is_contiguous()

    # 16-bit (l, m, 2n) -> f32 (l, m, n) -> (m, n, l) N-major view (no copy).
    mC = PreAct3.view(torch.float32).permute(1, 2, 0)
    mD = dPreAct3.view(torch.float32).permute(1, 2, 0)
    mAuxOut = PostAct3.permute(1, 2, 0)  # (m, n, l) N-major

    if mma_tiler_mn is None or cluster_shape_mn is None:
        tlr, clu = _default_tiler_cluster(m, n)
        mma_tiler_mn = mma_tiler_mn or tlr
        cluster_shape_mn = cluster_shape_mn or clu
    runner = _compile_dgated_cached(
        m, n, k, l, mma_tiler_mn, cluster_shape_mn, activation, PreAct.dtype
    )
    runner(mA, mB, mC, mD, sfa, sfb, mAuxOut)
    return dPreAct, PostAct


def mxfp8_gemm_ref(
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
    a_dq = A3 * as3.repeat_interleave(_SF_VEC_SIZE, dim=-1)
    b_dq = B3 * bs3.repeat_interleave(_SF_VEC_SIZE, dim=-1)
    out3 = torch.einsum("lmk,lnk->lmn", a_dq, b_dq).to(out_dtype)
    return out3.squeeze(0) if was_2d else out3
