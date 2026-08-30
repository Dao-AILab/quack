# Copyright (c) 2025, Tri Dao
import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import custom_fwd, custom_bwd

import triton
import triton.language as tl

from quack.cross_entropy import cross_entropy, cross_entropy_fwd_out
from quack.epilogue.scaled_exp import scaled_exp_target_epi
from quack.epilogue.library import identity_epi, lse_target_epi
from quack.epilogue.frontend import gemm_epilogue
from quack.gemm_interface import gemm, gemm_add, gemm_add_inplace
from quack.linear import linear_fwd_convert_type
from quack.operand_transform import a_transform
from quack.operand_transform.host import transform_a_operand


def linear_cross_entropy_func(
    x: Tensor,  # (..., d)
    weight: Tensor,  # (V, d)
    bias: Optional[Tensor],  # (V,) or None
    target: Tensor,  # (...,), int or long
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
    inplace_backward: bool = False,
) -> Tensor:
    y = F.linear(x, weight, bias)  # (..., V)
    return cross_entropy(
        y, target, ignore_index=ignore_index, reduction=reduction, inplace_backward=inplace_backward
    )


def linear_cross_entropy_func_ref(
    x: Tensor,  # (..., d)
    weight: Tensor,  # (V, d)
    bias: Optional[Tensor],  # (V,) or None
    target: Tensor,  # (...,), int or long
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    y = F.linear(x, weight, bias)  # (..., V)
    return F.cross_entropy(y, target, ignore_index=ignore_index, reduction=reduction)


def chunked_linear_cross_entropy_fwd(
    x: Tensor,  # (B*L, d) where B is batch, L is seqlen
    weight: Tensor,  # (V, d) where V is vocab size
    target: Tensor,  # (B*L,)
    chunk_size: int = 4096,
    ignore_index: int = -100,
    tuned: bool = True,
    need_dx: bool = True,
    need_dw: bool = True,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Chunked forward pass for linear cross entropy.

    Splits input along batch dimension, computes matmul and cross_entropy_fwd
    for each chunk, stores dx for each chunk, and accumulates dw. The dw
    accumulator is fp32 regardless of input dtype: each chunk's GEMM writes /
    adds into it in fp32, so precision is lost only at the single downcast to
    the weight's dtype in the backward pass. need_dx/need_dw skip the
    corresponding GEMMs and buffers entirely (loss-only when both are False).

    Returns:
        loss: (B*L,) loss values
        dx: (B*L, d) gradient w.r.t. input, or None if not need_dx
        dw: (V, d) fp32 gradient w.r.t. weight (accumulated across chunks
            except last), or None if not need_dw or single-chunk
        last_dlogits_chunk: (chunk_len, V) gradient of last chunk's logits (for deferred dw computation)
        last_x_chunk: (chunk_len, d) last chunk's input (for deferred dw computation)
    """
    B_L, d = x.shape
    V, _ = weight.shape
    device = x.device
    num_chunks = (B_L + chunk_size - 1) // chunk_size
    # Since we use gemm with TMA we require some alignment
    assert chunk_size % 8 == 0, "chunk_size must be multiple of 8"
    assert B_L % 8 == 0
    # Pre-allocate outputs
    loss = torch.empty(B_L, device=device, dtype=torch.float32)
    logits_chunk_preallocated = torch.empty((min(chunk_size, B_L), V), device=device, dtype=x.dtype)
    dx = torch.empty_like(x) if need_dx else None
    # Last chunk of dw will be deferred to the backward pass
    dw = torch.empty_like(weight, dtype=torch.float32) if need_dw and num_chunks > 1 else None
    last_dlogits_chunk = None
    last_x_chunk = None

    # Process in chunks
    for i, (x_chunk, target_chunk, loss_chunk) in enumerate(
        zip(*(t.split(chunk_size) for t in (x, target, loss)))
    ):
        chunk_len = x_chunk.shape[0]
        logits_chunk = logits_chunk_preallocated[:chunk_len]  # (chunk_len, V)
        torch.mm(x_chunk, weight.mT, out=logits_chunk)
        # dlogits overwrite the logits in place; skipped entirely for loss-only
        dlogits_chunk = logits_chunk if need_dx or need_dw else None
        cross_entropy_fwd_out(
            logits_chunk,
            target_chunk,
            None,  # target_logit
            loss=loss_chunk,
            lse=None,  # we don't need lse here
            dx=dlogits_chunk,
            weight=None,
            ignore_index=ignore_index,
        )
        if need_dx:
            # Compute dx for this chunk: dlogits @ weight
            start = i * chunk_size
            torch.mm(dlogits_chunk, weight, out=dx[start : start + chunk_len])
        if not need_dw:
            continue
        # Compute dw for all chunks except the last
        if i == num_chunks - 1:
            # Last chunk: save for backward pass
            last_dlogits_chunk = dlogits_chunk
            last_x_chunk = x_chunk
        elif i == 0:
            # First chunk: dw = dlogits.T @ x_chunk
            gemm(dlogits_chunk.T, x_chunk, out=dw, tuned=tuned)
        else:
            # Middle chunks: dw += dlogits.T @ x_chunk
            gemm_add_inplace(dlogits_chunk.T, x_chunk, dw, tuned=tuned)
    return loss, dx, dw, last_dlogits_chunk, last_x_chunk


class ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        target: Tensor,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum"] = "mean",
        chunk_size: int = 4096,
        tuned: bool = True,
    ):
        """
        Forward pass computes loss and stores dx and dw for backward.
        """
        ctx.weight_dtype = weight.dtype
        # read before the autocast convert: the converted tensors are non-leaves
        need_dx, need_dw = x.requires_grad, weight.requires_grad
        x, weight = linear_fwd_convert_type(x, weight)
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        loss, dx, dw, last_dlogits_chunk, last_x_chunk = chunked_linear_cross_entropy_fwd(
            x,
            weight,
            target.reshape(-1),
            chunk_size,
            ignore_index,
            tuned=tuned,
            need_dx=need_dx,
            need_dw=need_dw,
        )
        loss_sum = loss.sum()
        loss_scale = None if reduction == "sum" else 1.0 / (target != ignore_index).sum().float()
        ctx.save_for_backward(dx, dw, last_dlogits_chunk, last_x_chunk, loss_scale)
        ctx.batch_shape = batch_shape
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.tuned = tuned
        return loss_sum if loss_scale is None else loss_sum * loss_scale

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dloss):
        """
        Backward pass scales pre-computed gradients by dloss and completes
        the last chunk's dw computation (single downcast from the fp32
        accumulator to the weight's original dtype).
        dloss is a scalar.
        """
        dx, dw, last_dlogits_chunk, last_x_chunk, loss_scale = ctx.saved_tensors
        tuned = ctx.tuned
        if loss_scale is not None:
            dloss = dloss * loss_scale
        if dx is not None:
            dx.mul_(dloss)
            dx = dx.reshape(*ctx.batch_shape, dx.shape[-1])
        # Complete dw computation: dw = dloss * dw + dloss * (last_dlogits_chunk.T @ last_x_chunk)
        if last_dlogits_chunk is None:
            pass  # weight didn't require grad; dw stays None
        elif dw is None:
            # Only had one chunk, compute dw directly with dloss scaling
            dw = gemm(
                last_dlogits_chunk.T,
                last_x_chunk,
                out_dtype=ctx.weight_dtype,
                alpha=dloss,
                tuned=tuned,
            )
        else:
            # Add last chunk's contribution with dloss scaling
            # dw = dloss * dw + dloss * (last_dlogits_chunk.T @ last_x_chunk)
            # We use alpha=dloss, beta=dloss
            if ctx.weight_dtype == dw.dtype:
                gemm_add_inplace(
                    last_dlogits_chunk.T, last_x_chunk, dw, alpha=dloss, beta=dloss, tuned=tuned
                )
            else:
                dw = gemm_add(
                    last_dlogits_chunk.T,
                    last_x_chunk,
                    dw,
                    alpha=dloss,
                    beta=dloss,
                    out_dtype=ctx.weight_dtype,
                    tuned=tuned,
                )
        return dx, dw, None, None, None, None, None


# ── scaled-exp fused backward ─────────────────────────────────────
# The chunked pipeline above materializes full logits and dlogits per chunk
# (one GEMM + a row-kernel + two plain GEMMs). The scaled-exp pipeline below
# replaces the row kernel's dlogits materialization with per-(row, n-tile)
# pow2 offsets carried OUTSIDE the E matrix:
#
#   gemm1  scaled_exp_target_epi: E = bf16(2^(z*log2e - k)) with
#          k = rne(rowtilemax*log2e) per (row, tile_n1 n-tile), plus the
#          sum_exp partials, the k offsets themselves, and the exact fp32
#          target logit Zy (ColVecSelect) — logits never stored.
#   glue   (Triton): loss, the in-place target fix of E, ONE
#          (V/64, M) strip u = 2^(k - k_r), v = 2^{k_r} e^{-L} (per row),
#          xs = v*x. dZ never exists: dZ = v * (u ⊙ E) row-wise.
#   dx     strip GEMM on E (@a_transform colvec_ktile: per (row, k64) scale)
#          with the per-row v folded in as a free colvec epilogue.
#   dw     strip GEMM on E^T (@a_transform kvec_m64: per (vocab-m64, token)),
#          B = xs, fp32 out, accumulated across chunks through the TMA
#          reduce-add D atom (add_to_output — the fp32 dw is never re-read).
#
# Numerics match the baseline at bf16-relative grade: E carries the same
# information as dlogits (u and v are exact powers of two / fp32), and Zy/k
# are read from the epilogue, never re-derived. Measured 1.13-1.19x over the
# baseline pipeline at Llama shapes (kernel-level: AI/bench_lce_v3_e2e.py;
# end-to-end: benchmarks/benchmark_linear_cross_entropy.py).


@a_transform(vec_size=8, args={"u": "colvec_ktile"})
def _lce_dx_scale(x, u):
    return x * u


@a_transform(vec_size=8, args={"u": "kvec_m64"})
def _lce_dw_scale(x, u):
    return x * u


@gemm_epilogue()
def _lce_vscale_epi(acc, v):
    """Per-row fp32 v = grad_scale * 2^{k_r} e^{-L}, fused into the dx store."""
    return {"D": acc * v}


LN2 = math.log(2.0)
_LN2 = tl.constexpr(LN2)


@triton.jit
def _lce_glue_row(
    max_log2_ptr,
    sum_exp_ptr,
    x_ptr,
    target_ptr,
    zy_ptr,
    E_ptr,
    loss_ptr,
    kr_ptr,
    v_ptr,
    xs_ptr,
    scale_ptr,
    M,
    T,
    D,
    V,
    stride_xm,
    ignore_index,
    TILE_N1: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    WRITE_XS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    i = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    mt = offs_t < T
    se = tl.load(sum_exp_ptr + i * T + offs_t, mask=mt, other=0.0)
    k = tl.load(max_log2_ptr + i * T + offs_t, mask=mt, other=0.0)
    k_r = tl.max(tl.where(mt, k, float("-inf")), 0)  # k can be negative (true max)
    L = k_r * _LN2 + tl.log(tl.sum(se * tl.exp2(k - k_r), 0))
    y = tl.load(target_ptr + i)
    valid = y != ignore_index
    # exact target logit emitted by the gemm1 epilogue (ColVecSelect): the
    # same fp32 accumulator value E was computed from. Never written (and
    # never read) for ignored rows.
    Zy = tl.load(zy_ptr + i, mask=valid, other=0.0)
    tl.store(loss_ptr + i, tl.where(valid, L - Zy, 0.0))
    tl.store(kr_ptr + i, k_r)
    # target fix: E[i, y] = (p_y - 1) / s, cancellation in fp32
    ky = tl.load(max_log2_ptr + i * T + y // TILE_N1, mask=valid, other=0.0)
    fix = (tl.exp(Zy - L) - 1.0) * tl.exp(L - ky * _LN2)
    tl.store(E_ptr + i * V + y, fix.to(tl.bfloat16), mask=valid)
    v = tl.where(valid, tl.exp(k_r * _LN2 - L), 0.0)
    if HAS_SCALE:
        v = v * tl.load(scale_ptr)
    tl.store(v_ptr + i, v)
    if WRITE_XS:
        vb = v.to(tl.bfloat16).to(tl.float32)  # xs sees the bf16-rounded v
        offs_d = tl.arange(0, BLOCK_D)
        for d0 in range(0, D, BLOCK_D):
            md = d0 + offs_d < D
            xv = tl.load(x_ptr + i * stride_xm + d0 + offs_d, mask=md, other=0.0).to(tl.float32)
            tl.store(xs_ptr + i * D + d0 + offs_d, (xv * vb).to(tl.bfloat16), mask=md)


@triton.jit
def _lce_glue_strip(
    max_log2_ptr,
    kr_ptr,
    strip_ptr,
    M,
    T,
    REP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pt = tl.program_id(0)
    pm = tl.program_id(1)
    offs_t = pt * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    mt = offs_t < T
    mm = offs_m < M
    # (BLOCK_M, BLOCK_T) load of max_log2, transposed u2 store coalesced along M
    k = tl.load(
        max_log2_ptr + offs_m[:, None] * T + offs_t[None, :],
        mask=mm[:, None] & mt[None, :],
        other=0.0,
    )
    k_r = tl.load(kr_ptr + offs_m, mask=mm, other=0.0)
    u2 = tl.exp2(k - k_r[:, None]).to(tl.bfloat16)  # exact pow2
    u2t = tl.trans(u2)  # (BLOCK_T, BLOCK_M)
    smask = mt[:, None] & mm[None, :]
    for r in tl.static_range(REP):
        tl.store(strip_ptr + (offs_t[:, None] * REP + r) * M + offs_m[None, :], u2t, mask=smask)


def lce_glue(
    max_log2,
    sum_exp,
    x,
    target,
    zy,
    E,
    loss,
    strip,
    v_row,
    xs,
    tile_n1,
    ignore_index=-100,
    grad_scale=None,
):
    """Fused Triton glue between gemm1 and the two strip-transform grad GEMMs.
    Per chunk of M rows it turns the epilogue's raw emissions — the k offsets
    (max_log2_out, integer-valued in log2 units), the sum_exp partials, and
    the exact fp32 target logits Zy (ColVecSelect) — into everything the grad
    GEMMs consume:

      loss[i] = L_i - Zy_i                       (0 for ignored rows)
      E[i, y_i] = (e^{Zy-L} - 1) * e^{L - k_y ln2}   (the target fix, in place)
      strip[v64, i] = 2^(k[i, v64 // REP] - k_r[i])  (exact pow2)
      v[i] = grad_scale * 2^{k_r} * e^{-L}       (0 for ignored rows)
      xs = bf16(v) * x                           (the dw GEMM's B operand)

    Exactness: k and Zy are READ from the epilogue, not re-derived — there is
    no rounding convention to match, and the target-fix term uses the same
    accumulation E came from. exp2 of the integer-valued (k - k_r) is an
    exact power of two. The glue never touches W.

    ignore_index: rows with target == ignore_index get loss = 0 and v = 0 —
    the zero v kills the dx row through the colvec epilogue and the row's dw
    contribution through xs; the target fix is skipped (their Zy was never
    written by ColVecSelect and is never read). grad_scale (an optional fp32
    scalar TENSOR, e.g. 1/num_valid for mean reduction) is folded into v so
    dx and dw come out pre-scaled; the per-row loss is NOT scaled.

    All row-indexed tensors cover the same M rows (one chunk); strip is a
    contiguous (V // 64, M) bf16 tensor (REP = tile_n1 // 64 repeats per
    n-tile — one strip serves both grad GEMMs); strip / xs may be None when
    the corresponding grad GEMM is skipped."""
    M, T = max_log2.shape
    D = x.shape[1]
    V = E.shape[1]
    kr = torch.empty(M, device=x.device, dtype=torch.float32)
    _lce_glue_row[(M,)](
        max_log2,
        sum_exp,
        x,
        target,
        zy,
        E,
        loss,
        kr,
        v_row,
        xs if xs is not None else v_row,
        grad_scale if grad_scale is not None else v_row,
        M,
        T,
        D,
        V,
        x.stride(0),
        ignore_index,
        TILE_N1=tile_n1,
        HAS_SCALE=grad_scale is not None,
        WRITE_XS=xs is not None,
        BLOCK_T=triton.next_power_of_2(T),
        BLOCK_D=256,
        num_warps=4,
    )
    if strip is not None:
        BM, BT = 64, 64
        _lce_glue_strip[(triton.cdiv(T, BT), triton.cdiv(M, BM))](
            max_log2,
            kr,
            strip,
            M,
            T,
            REP=tile_n1 // 64,
            BLOCK_M=BM,
            BLOCK_T=BT,
            num_warps=4,
        )


def _pick_tile_n1(V: int) -> Optional[int]:
    # gemm1's tile_N is the k-offset granularity: it must divide V (the
    # sum_exp/k partials are per full tile) and be a multiple of 64 (the
    # strip is m64-resolved). 192 measured best at Llama vocab/d; 256 covers
    # V % 192 != 0 (e.g. 32000), 128 the rest (e.g. 151936).
    for tile_n1 in (192, 256, 128, 64):
        if V % tile_n1 == 0:
            return tile_n1
    return None


def scaled_exp_lce_supported(
    x: Tensor, weight: Tensor, chunk_size: Optional[int], reduction: str
) -> bool:
    """Eligibility for the scaled-exp pipeline: SM90 (the A-transform RS
    mainloop), bf16 compute, V % 128 (the dw strip GEMM's tile_M over the
    vocab dim), a tile_n1 divisor of V, and 128-divisible chunks (the dx
    strip GEMM's tile_M over rows; a ragged last chunk is padded)."""
    if not (x.is_cuda and weight.is_cuda) or reduction not in ("mean", "sum"):
        return False
    if chunk_size is None or chunk_size % 128 != 0:
        return False
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else x.dtype
    wdtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else weight.dtype
    if dtype != torch.bfloat16 or wdtype != torch.bfloat16:
        return False
    V, d = weight.shape
    if x.shape[-1] != d or x.stride(-1) != 1 or weight.stride(-1) != 1:
        return False
    if V % 128 != 0 or d % 8 != 0 or _pick_tile_n1(V) is None:
        return False
    return torch.cuda.get_device_capability(x.device) == (9, 0)


def scaled_exp_linear_cross_entropy_fwd(
    x: Tensor,  # (M, d) bf16, rows contiguous
    weight: Tensor,  # (V, d) bf16, contiguous
    target: Tensor,  # (M,), int32/int64
    chunk_size: int,
    ignore_index: int = -100,
    *,
    need_dx: bool = True,
    need_dw: bool = True,
    grad_scale: Optional[Tensor] = None,  # fp32 scalar folded into dx/dw (e.g. 1/num_valid)
    tile_n1: Optional[int] = None,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Chunked scaled-exp forward. Returns (loss, dx, dw) over M_pad >= M rows
    (the last chunk is padded to a 128 multiple with ignored rows: their x is
    zeroed and their v is 0, so loss/dx pad rows are 0 and dw is exact) —
    callers slice [:M]. dw is the fp32 accumulator across all chunks."""
    M, d = x.shape
    V = weight.shape[0]
    device = x.device
    if tile_n1 is None:
        tile_n1 = _pick_tile_n1(V)
    assert tile_n1 is not None and V % tile_n1 == 0 and tile_n1 % 64 == 0
    assert V % 128 == 0 and chunk_size % 128 == 0 and M > 0
    T = V // tile_n1
    rk = V // 64
    num_chunks = -(-M // chunk_size)
    r = M - (num_chunks - 1) * chunk_size  # last chunk rows, in (0, chunk_size]
    r_pad = -(-r // 128) * 128
    cs = chunk_size if num_chunks > 1 else r_pad
    M_pad = (num_chunks - 1) * chunk_size + r_pad

    loss = torch.empty(M_pad, device=device, dtype=torch.float32)
    dx = torch.empty(M_pad, d, device=device, dtype=x.dtype) if need_dx else None
    dw = torch.empty(V, d, device=device, dtype=torch.float32) if need_dw else None

    # Per-chunk scratch, reused across chunks. The strip's M extent is its
    # contiguous last dim, so a shorter last chunk needs its own tensor.
    E = torch.empty(cs, V, device=device, dtype=x.dtype)
    sum_exp = torch.empty(cs, T, device=device, dtype=torch.float32)
    kk = torch.empty(cs, T, device=device, dtype=torch.float32)
    zy = torch.empty(cs, device=device, dtype=torch.float32)
    v_row = torch.empty(cs, device=device, dtype=torch.float32)
    strips = need_dx or need_dw
    strip = torch.empty(rk, cs, device=device, dtype=x.dtype) if strips else None
    strip_last = strip
    if strips and num_chunks > 1 and r_pad != chunk_size:
        strip_last = torch.empty(rk, r_pad, device=device, dtype=x.dtype)
    xs = torch.empty(cs, d, device=device, dtype=x.dtype) if need_dw else None
    x_last, target_last = None, None
    if r_pad != r:
        # Pad rows are exact by construction: x = 0 keeps gemm1/E/strip finite
        # (acc 0 -> k 0, E 1, strip 1) and target = ignore_index zeroes their
        # v (and xs row), so they contribute nothing to loss, dx, or dw.
        x_last = torch.zeros(r_pad, d, device=device, dtype=x.dtype)
        x_last[:r].copy_(x[M - r :])
        target_last = torch.full((r_pad,), ignore_index, device=device, dtype=target.dtype)
        target_last[:r].copy_(target[M - r :])

    g1_pingpong = tile_n1 <= 208  # pingpong hides the two-phase epilogue
    tile_n2 = 192 if d % 192 == 0 else 256  # dx/dw strip GEMM tile_N over d
    for i in range(num_chunks):
        start = i * chunk_size
        last = i == num_chunks - 1
        n_rows = r_pad if last else chunk_size
        if last and x_last is not None:
            x_c, target_c = x_last, target_last
        else:
            x_c = x[start : start + n_rows]
            target_c = target[start : start + n_rows]
        mE = E[:n_rows]
        sum_exp_c, kk_c, zy_c, v_c = sum_exp[:n_rows], kk[:n_rows], zy[:n_rows], v_row[:n_rows]
        strip_c = strip_last if last else strip
        xs_c = xs[:n_rows] if need_dw else None
        scaled_exp_target_epi.gemm(
            x_c,
            weight,
            mE,
            epi_args=dict(
                max_log2=tile_n1,
                sum_exp=sum_exp_c,
                max_log2_out=kk_c,
                target=target_c,
                target_logit=zy_c,
            ),
            tile_M=128,
            tile_N=tile_n1,
            cluster_M=2,
            cluster_N=1,
            pingpong=g1_pingpong,
        )
        lce_glue(
            kk_c,
            sum_exp_c,
            x_c,
            target_c,
            zy_c,
            mE,
            loss[start : start + n_rows],
            strip_c if strips else None,
            v_c,
            xs_c,
            tile_n1,
            ignore_index=ignore_index,
            grad_scale=grad_scale,
        )
        if need_dx:
            bundle = transform_a_operand(_lce_dx_scale, mE, {"u": strip_c}, 128, 64)
            _lce_vscale_epi.gemm(
                bundle,
                weight.mT,
                dx[start : start + n_rows],
                epi_args=dict(v=v_c.unsqueeze(0)),
                transform_a=_lce_dx_scale,
                tile_M=128,
                tile_N=tile_n2,
                tile_K=64,
                cluster_M=2,
                cluster_N=1,
            )
        if need_dw:
            bundle = transform_a_operand(_lce_dw_scale, mE.t(), {"u": strip_c}, 128, 64)
            # chunk 0 stores dw; later chunks accumulate through the TMA
            # reduce-add D atom (no C load — half the accumulate traffic)
            identity_epi.gemm(
                bundle,
                xs_c.mT,
                dw,
                epi_args={},
                transform_a=_lce_dw_scale,
                add_to_output=i > 0,
                tile_M=128,
                tile_N=tile_n2,
                tile_K=64,
                cluster_M=1,
                cluster_N=2,
            )
    return loss, dx, dw


@torch.library.custom_op("quack::lce_scaled_exp_fwd", mutates_args=(), device_types="cuda")
def _lce_scaled_exp_fwd_op(
    x: Tensor,
    weight: Tensor,
    target: Tensor,
    chunk_size: int,
    ignore_index: int,
    need_dx: bool,
    need_dw: bool,
    grad_scale: Optional[Tensor],
    tile_n1: Optional[int],
) -> list[Tensor]:
    """The whole chunked scaled-exp forward as ONE custom op, so torch.compile
    records a single graph node instead of tracing the host chunk loop
    (mod.gemm plan machinery, jit-cache probes, Triton launches). Returns
    [loss(M_pad,)] + [dx] if need_dx + [dw] if need_dw (see the fake)."""
    loss, dx, dw = scaled_exp_linear_cross_entropy_fwd(
        x,
        weight,
        target,
        chunk_size,
        ignore_index,
        need_dx=need_dx,
        need_dw=need_dw,
        grad_scale=grad_scale,
        tile_n1=tile_n1,
    )
    return [loss] + ([dx] if need_dx else []) + ([dw] if need_dw else [])


@_lce_scaled_exp_fwd_op.register_fake
def _lce_scaled_exp_fwd_fake(
    x, weight, target, chunk_size, ignore_index, need_dx, need_dw, grad_scale, tile_n1
):
    # Mirrors scaled_exp_linear_cross_entropy_fwd's padding arithmetic exactly.
    M, d = x.shape
    V = weight.shape[0]
    num_chunks = -(-M // chunk_size)
    r = M - (num_chunks - 1) * chunk_size
    M_pad = (num_chunks - 1) * chunk_size + -(-r // 128) * 128
    outs = [torch.empty(M_pad, device=x.device, dtype=torch.float32)]
    if need_dx:
        outs.append(torch.empty(M_pad, d, device=x.device, dtype=x.dtype))
    if need_dw:
        outs.append(torch.empty(V, d, device=x.device, dtype=torch.float32))
    return outs


def _scaled_exp_lce_loss_only(x: Tensor, weight: Tensor, target: Tensor, ignore_index: int):
    # D-less eval: the logits are never materialized — lse_target_epi emits
    # online-LSE partials (host-finalized) and the exact target logit only.
    # Ignored rows' target_logit is never written (ColVecSelect skips
    # out-of-range indices); torch.where discards the uninitialized lanes.
    res = lse_target_epi(x, weight.mT, store_d=False, target=target, tuned=False)
    return torch.where(target != ignore_index, res["lse"] - res["target_logit"], 0.0)


class ScaledExpLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        target: Tensor,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum"] = "mean",
        chunk_size: int = 4096,
        tile_n1: Optional[int] = None,
    ):
        ctx.weight_dtype = weight.dtype
        # read before the autocast convert: the converted tensors are non-leaves
        need_dx, need_dw = x.requires_grad, weight.requires_grad
        x, weight = linear_fwd_convert_type(x, weight)
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        if x.stride(0) % 8 != 0:  # 16 B TMA alignment on gemm1's A rows
            x = x.contiguous()
        target = target.reshape(-1)
        M = x.shape[0]
        # mean: fold 1/num_valid into v at the glue, so dx/dw come out
        # pre-scaled and the backward only applies dloss.
        inv_count = None
        if reduction == "mean":
            inv_count = (target != ignore_index).sum().float().reciprocal()
        outs = torch.ops.quack.lce_scaled_exp_fwd(
            x, weight, target, chunk_size, ignore_index, need_dx, need_dw, inv_count, tile_n1
        )
        loss = outs[0]
        dx = outs[1] if need_dx else None
        dw = outs[1 + need_dx] if need_dw else None
        loss_sum = loss[:M].sum()
        ctx.save_for_backward(dx, dw)
        ctx.batch_shape = batch_shape
        ctx.num_rows = M
        return loss_sum if inv_count is None else loss_sum * inv_count

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dloss):
        dx, dw = ctx.saved_tensors
        if dx is not None:
            dx = dx[: ctx.num_rows]
            dx.mul_(dloss)
            dx = dx.reshape(*ctx.batch_shape, dx.shape[-1])
        if dw is not None:
            # single downcast from the fp32 accumulator
            dw = dw.mul_(dloss).to(ctx.weight_dtype)
        return dx, dw, None, None, None, None, None


def scaled_exp_linear_cross_entropy(
    x: Tensor,
    weight: Tensor,
    target: Tensor,
    chunk_size: int = 4096,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum"] = "mean",
    tile_n1: Optional[int] = None,
) -> Tensor:
    """Linear cross entropy through the scaled-exp fused-backward pipeline
    (see the section comment above). Same contract as
    chunked_linear_cross_entropy; eligibility via scaled_exp_lce_supported."""
    if reduction not in ["mean", "sum"]:
        raise ValueError(f"Invalid reduction: {reduction}")
    if not torch.is_grad_enabled() or not (x.requires_grad or weight.requires_grad):
        x, weight = linear_fwd_convert_type(x, weight)
        target = target.reshape(-1)
        loss = _scaled_exp_lce_loss_only(x.reshape(-1, x.shape[-1]), weight, target, ignore_index)
        loss_sum = loss.sum()
        if reduction == "sum":
            return loss_sum
        return loss_sum / (target != ignore_index).sum().float()
    return ScaledExpLinearCrossEntropyFunction.apply(
        x, weight, target, ignore_index, reduction, chunk_size, tile_n1
    )


def chunked_linear_cross_entropy(
    x: Tensor,
    weight: Tensor,
    target: Tensor,
    chunk_size: int = 4096,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum"] = "mean",
    tuned: bool = True,
    use_scaled_exp: Optional[bool] = None,
) -> Tensor:
    """
    Chunked linear cross entropy with automatic differentiation support.

    Args:
        x: Input tensor of shape (B*L, d)
        weight: Weight tensor of shape (V, d)
        target: Target indices of shape (B*L,)
        chunk_size: Size of chunks to process
        ignore_index: Index to ignore in loss computation
        reduction: Type of reduction to apply
        tuned: Whether to use tuned kernels
        use_scaled_exp: route through the scaled-exp fused-backward pipeline.
            None (default) auto-selects it when eligible (SM90, bf16, V % 128
            == 0, ...; see scaled_exp_lce_supported); True asserts
            eligibility inside; False forces the base pipeline.

    Returns:
        Loss tensor with specified reduction
    """
    if reduction not in ["mean", "sum"]:
        raise ValueError(f"Invalid reduction: {reduction}")
    if use_scaled_exp is None:
        use_scaled_exp = scaled_exp_lce_supported(x, weight, chunk_size, reduction)
    if use_scaled_exp:
        return scaled_exp_linear_cross_entropy(
            x, weight, target, chunk_size, ignore_index, reduction
        )
    if not torch.is_grad_enabled() or not (x.requires_grad or weight.requires_grad):
        # eval / inference: loss only — no dx/dw GEMMs, no fp32 (V, d) accumulator
        x, weight = linear_fwd_convert_type(x, weight)
        loss, *_ = chunked_linear_cross_entropy_fwd(
            x.reshape(-1, x.shape[-1]),
            weight,
            target.reshape(-1),
            chunk_size,
            ignore_index,
            tuned=tuned,
            need_dx=False,
            need_dw=False,
        )
        loss_sum = loss.sum()
        if reduction == "sum":
            return loss_sum
        return loss_sum / (target != ignore_index).sum().float()
    loss = ChunkedLinearCrossEntropyFunction.apply(
        x, weight, target, ignore_index, reduction, chunk_size, tuned
    )
    return loss


class LinearCrossEntropy(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        ignore_index: int = -100,
        reduction: Literal["none", "mean", "sum"] = "mean",
        chunk_size: Optional[int] = None,
        inplace_backward: bool = False,
        tuned: bool = True,
        use_scaled_exp: Optional[bool] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.chunk_size = chunk_size
        self.inplace_backward = inplace_backward
        self.tuned = tuned
        self.use_scaled_exp = use_scaled_exp

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if (
            self.bias is None
            and input.is_cuda
            and input.stride(-1) == 1
            and self.in_features % 8 == 0
            and self.out_features % 8 == 0
            and input.shape[:-1].numel() % 8 == 0
            and self.chunk_size is not None
            and self.chunk_size % 8 == 0
            and self.reduction in ["mean", "sum"]
        ):
            return chunked_linear_cross_entropy(
                input,
                self.weight,
                target,
                chunk_size=self.chunk_size,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                tuned=self.tuned,
                use_scaled_exp=self.use_scaled_exp,
            )
        else:
            return linear_cross_entropy_func(
                input,
                self.weight,
                self.bias,
                target,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                inplace_backward=self.inplace_backward,
            )
