# Copyright (c) 2026, Tri Dao.
"""Weight-only-quantized GEMM for SM90: out[M, N] = act[M, K] @ dequant(W)[N, K]^T.

The packed weights are the WGMMA A operand (RS, decoded to bf16 in registers
by a :class:`~quack.operand_transform.TransformAW4`), the bf16 activations
are B, and the output is written transposed (D is (N, M) m-major = out
row-major). See quack/operand_transform/formats/ for the formats and
quack/blockscaled/nvfp4_utils.py for the repack layout.

Scale-factor-strip formats (nvfp4, int4*, int4awq, mxfp4, mxfp8) pass their
repacked SF blob as ``sf``; it rides the aux A-side operand. Strip-free
formats (qtip*, int8/fp8 with the per-channel scale left to the caller,
fn-authored formats) take ``sf=None``. Per-tensor weight scales ride the
epilogue alpha.

This wrapper is thin sugar over the fn epilogue frontend: both entry points
are ``EpiMod.gemm(..., transform_a=...)`` calls, so W4 kernels share the plan
cache, jit/disk cache, async compile, and EpiOp argument machinery with every
epilogue variant. What remains here is W4's own host surface: the offline
``prepare`` step, validation, explicit-tile handling over the measured config
rules (which live with the transform handles:
quack.operand_transform.host.pick_w4_cfg), and the split-k buffer reuse.
"""

from typing import Optional

import torch
from torch import Tensor

from quack.cute_dsl_utils import get_device_capacity
from quack.operand_transform.formats import decode_format
from quack.epilogue.ops import ColVecLoad, RowVecLoad
from quack.gemm import _split_k_buffers
from quack.gemm_config import SplitKMode
from quack.epilogue.frontend import gemm_epilogue
from quack.operand_transform.host import (
    pick_w4_cfg as _pick_w4_cfg,
    pick_w4a8_cfg as _pick_w4a8_cfg,
)

__all__ = ["gemm_w4a16", "gemm_w4a8", "prepare_w4_weight", "quantize_act_per_token_fp8"]

_splitk_buf_cache = {}


def prepare_w4_weight(q, sf=None, wformat="qtip2s"):
    """One-time weight prep: quantized weights (+ scales, format-dependent)
    -> repacked blob pair. N is padded to a multiple of 128 (tile
    granularity); bytes are shuffled into WGMMA A-fragment order for the
    in-register decode."""
    return decode_format(wformat).prepare(q, sf)


# Per-tensor weight scale as an exact fp32 epilogue multiply (scalar infers
# to a Scalar op; alpha == 1.0 is bitwise-identity).
@gemm_epilogue()
def _w4a16_alpha(acc, alpha):
    return {"D": acc * alpha}


def gemm_w4a16(
    act: Tensor,  # (M, K) bf16, K-major
    blob: Tensor,  # (N/64, K/tile_k, 128, 4|8 B * tile_k/64) from fmt.prepare
    sf: Optional[Tensor] = None,  # repacked SF blob from fmt.prepare (strip formats)
    tensor_scale: float = 1.0,  # per-tensor weight scale, applied as epilogue alpha
    out: Optional[Tensor] = None,  # (M, N_out) bf16
    n_out: Optional[int] = None,  # unpadded N (defaults to blob's padded N)
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    cluster_n: int = 1,
    max_swizzle_size: int = 8,
    use_pdl: bool = True,
    wformat="qtip2s",  # W4_FORMATS name or DecodeFormat instance
    split_k: Optional[int] = None,  # None = auto: 2 when the grid starves the machine
) -> Tensor:
    fmt = decode_format(wformat)
    tk = fmt.tile_k
    assert act.dtype == torch.bfloat16 and act.is_contiguous()
    m_act, k = act.shape
    g, kt = blob.shape[:2]
    n_full = g * 64
    assert kt * tk == k, f"K mismatch: act K={k}, blob K={kt * tk}"
    if n_out is None:
        n_out = n_full
    # W4 runs atom_n == 1 on every arch (SM120 included since the (4,1,1)/
    # (8,1,1) decode layouts): tile_n floor is the 16-wide warp N span.
    auto_tm, auto_tn, auto_sk = _pick_w4_cfg(
        m_act,
        n_full,
        k // tk,
        sm120=get_device_capacity(act.device)[0] == 12,
        device=act.device,
    )
    if tile_m is None and tile_n is None and split_k is None:
        tile_m, tile_n, split_k = auto_tm, auto_tn, auto_sk
    if tile_m is None:
        tile_m = auto_tm
    if tile_n is None:
        tile_n = auto_tn
    if out is None:
        out = torch.empty(m_act, n_full, dtype=torch.bfloat16, device=act.device)
    else:
        assert out.shape == (m_act, n_out) and out.dtype == torch.bfloat16
        assert n_out == n_full, "padded N requires an internally allocated out"

    if split_k is None:
        # explicitly-tiled callers get the plain grid-starvation rule
        n_ctas = (n_full // tile_m) * ((m_act + tile_n - 1) // tile_n)
        split_k = 2 if (n_ctas < 128 and k // tk >= 32) else 1

    bufs = None
    if split_k > 1:
        # serial split-k turnstile + partials workspace; the kernel leaves the
        # semaphore reset, so the buffers are cached and reused across calls
        buf_key = (n_full, m_act, tile_m, tile_n, cluster_n, act.device.index)
        bufs = _splitk_buf_cache.get(buf_key)
        if bufs is None:
            bufs = _split_k_buffers(
                out.t()[None], SplitKMode.SERIAL, tile_m, tile_n, 1, cluster_n, False
            )
            _splitk_buf_cache[buf_key] = bufs
    # out crosses caller-oriented (M_act, N_full) row-major; the trace
    # relabels it to the kernel's (N_full, M_act) m-major D (cd_transposed)
    _w4a16_alpha.gemm(
        act,
        blob,
        out,
        epi_args={"alpha": tensor_scale},
        transform_a=wformat if isinstance(wformat, str) else fmt,
        transform_sf=sf,
        tile_M=tile_m,
        tile_N=tile_n,
        tile_K=tk,
        cluster_M=1,
        cluster_N=cluster_n,
        max_swizzle_size=max_swizzle_size,
        split_k=split_k,
        post_init_attrs=() if use_pdl else (("use_pdl", False),),
        split_k_buffers=bufs,
    )
    if n_out != n_full:
        out = out[:, :n_out]
    return out


# The per-token activation scale is a k-invariant per-output-row factor: it
# commutes out of the GEMM sum and applies as an exact fp32 colvec multiply in
# the epilogue (the pin flips to the kernel's rowvec — D is transposed).
@gemm_epilogue(ops={"v": ColVecLoad("v")})
def _w4a8_token_scale(acc, v):
    return {"D": acc * v}


# Folded (no-drain) W4A8: the group scale rode the decode LUT; what remains is
# the per-token scale and the fold's per-weight-channel normalizer, both fp32.
@gemm_epilogue(ops={"v": ColVecLoad("v"), "cs": RowVecLoad("cs")})
def _w4a8_folded_scale(acc, v, cs):
    return {"D": acc * v * cs}


def quantize_act_per_token_fp8(act: Tensor):
    """(M, K) float -> e4m3 (M, K) + fp32 per-token scales (M,): amax/448."""
    a = act.float()
    scale = (a.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
    q = (a / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale


def gemm_w4a8(
    act: Tensor,  # (M, K): e4m3 K-major (pass act_scale), or float (quantized here)
    blob: Tensor,  # from decode_format(wformat).prepare / repack_w4a8_weight
    sf: Tensor,  # repacked SF strip from prepare / repack_w4a8_sf
    act_scale: Optional[Tensor] = None,  # (M,) fp32 per-token scales
    chan_scale: Optional[Tensor] = None,  # (N,) fp32 (int4smf: from fold_int4sm_scales)
    out: Optional[Tensor] = None,  # (M, N_out) bf16
    n_out: Optional[int] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    cluster_n: int = 1,
    max_swizzle_size: int = 8,
    split_k: Optional[int] = None,
    wformat: str = "int4sm",
) -> Tensor:
    """W4A8: sign-magnitude int4-g128 weights x e4m3 per-token activations.

    wformat "int4sm" (exact): fp32 promotion per k-tile — no weight
    requantization; the only losses are the activation e4m3 cast and the fp8
    WGMMA accumulator. wformat "int4smf" (folded, no-drain): the group scale
    folds into the decode's e4m3 LUT (one e4m3 rounding per weight, ~2^-4
    rel worst case) and the mainloop runs fully pipelined fast-accum fp8 —
    faster at prefill; pass the fold's ``chan_scale``."""
    fmt = decode_format(wformat)
    assert fmt.mma_dtype.width == 8, f"{wformat!r} is not a W4A8 format"
    folded = not fmt.promote
    assert (chan_scale is not None) == folded, "chan_scale iff the folded format"
    if act.dtype != torch.float8_e4m3fn:
        assert act_scale is None, "act_scale is derived when act is not already e4m3"
        act, act_scale = quantize_act_per_token_fp8(act)
    assert act.is_contiguous()
    assert act_scale is not None and act_scale.dtype == torch.float32
    m_act, k = act.shape
    g, kt = blob.shape[:2]
    n_full = g * 64
    assert kt * 128 == k, f"K mismatch: act K={k}, blob K={kt * 128}"
    if n_out is None:
        n_out = n_full
    if folded and m_act > 128:
        # no promote -> no doubled accumulator: the W4A16 coverage rule holds
        # (tile_n 256 prefill); decode shapes stay on the measured 64-row
        # occupancy-2 rule below — both W4A8 variants are weight-BW-bound
        # there and 128-row tiles + split-k lose the same way
        auto_tm, auto_tn, auto_sk = _pick_w4_cfg(m_act, n_full, k // 128)
    else:
        auto_tm, auto_tn, auto_sk = _pick_w4a8_cfg(m_act, n_full)
    if tile_m is None:
        tile_m = auto_tm
    if tile_n is None:
        tile_n = auto_tn
    if split_k is None:
        split_k = auto_sk if (tile_m, tile_n) == (auto_tm, auto_tn) else 1
    if out is None:
        out = torch.empty(m_act, n_full, dtype=torch.bfloat16, device=act.device)
    else:
        assert out.shape == (m_act, n_out) and out.dtype == torch.bfloat16
        assert n_out == n_full, "padded N requires an internally allocated out"
    epi_args = {"v": act_scale.unsqueeze(0)}
    mod = _w4a8_token_scale
    if folded:
        mod = _w4a8_folded_scale
        if chan_scale.shape[0] < n_full:  # N was padded to tile granularity
            chan_scale = torch.cat([chan_scale, chan_scale.new_ones(n_full - chan_scale.shape[0])])
        epi_args["cs"] = chan_scale.to(torch.float32).unsqueeze(0)
    mod.gemm(
        act,
        blob,
        out,
        epi_args=epi_args,
        transform_a=wformat,
        transform_sf=sf,
        tile_M=tile_m,
        tile_N=tile_n,
        tile_K=128,
        cluster_M=1,
        cluster_N=cluster_n,
        max_swizzle_size=max_swizzle_size,
        split_k=split_k,
    )
    if n_out != n_full:
        out = out[:, :n_out]
    return out
