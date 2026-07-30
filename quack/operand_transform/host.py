# Copyright (c) 2026, Tri Dao.
"""Host-side plumbing that makes A-operand transforms first-class in the
generic GEMM host layer (:mod:`quack.gemm_runtime.host`), mirroring what EpiOps'
host hooks do for epilogues:

* :func:`as_transform_mod` — normalize any handle to a TransformModBase
  mod (refs/registries/payloads live in :mod:`quack.gemm_runtime.identity`,
  exactly like the epilogue side).
* the W4 blob/strip geometry — ONE implementation (over meta tensors)
  serves both the runtime torch views and the trace-time fake tensors, so
  the compiled layout and the launched layout cannot drift.

A transform handle is anything ``transform_a=`` accepts today: a registered
format name, a DecodeFormat instance, or an ATransformMod / PackedFormatMod.
Value transforms (packed=None) have no host geometry — they only contribute
the ctor factory and their semantic digest to the compile key.
"""

import functools

import torch

import cutlass
import cutlass.cute as cute

from quack.operand_transform.formats import DecodeFormat, W4_FORMATS
from quack.operand_transform.kinds import ARG_KINDS
from quack.operand_transform.transform import TransformAOperand

__all__ = [
    "as_transform_mod",
    "pick_w4_cfg",
    "pick_w4a8_cfg",
    "transform_a_fake_operand",
    "transform_a_operand",
    "w4_fake_operands",
    "w4_operand_views",
    "w4_padded_n",
]


@functools.lru_cache(maxsize=None)
def _named_w4_mod(name: str):
    from quack.operand_transform.frontend import w4_transform

    return w4_transform(name)


def as_transform_mod(handle):
    """Normalize any ``transform_a=`` handle (W4 registry name / DecodeFormat
    instance / mod) to a digest-carrying mod implementing the
    TransformModBase protocol. Memoized for names and format instances, so
    warm paths pay a dict lookup, never a source fingerprint."""
    if isinstance(handle, str):
        if handle not in W4_FORMATS:
            raise KeyError(f"unknown W4 format {handle!r}")
        return _named_w4_mod(handle)
    if isinstance(handle, DecodeFormat):
        mod = handle.__dict__.get("_transform_mod")
        if mod is None:
            from quack.operand_transform.frontend import PackedFormatMod

            mod = handle._transform_mod = PackedFormatMod(handle)
        return mod
    if not hasattr(handle, "semantic_digest"):
        raise TypeError(f"not a transform handle: {handle!r}")
    return handle


def w4_padded_n(blob) -> int:
    """Padded weight N from a repacked blob (its dim0 counts m64 row blocks)
    — THE single statement of the blob-rows rule."""
    return blob.shape[0] * 64


# ---- measured W4 config rules (consumed by the mods' default_config and ----
# ---- quack.gemm_w4's explicit-tile surface) --------------------------------


def _pick_tile_n(m_act: int, tile_n_min: int = 16) -> int:
    for cand in (16, 32, 64, 128):
        if cand >= tile_n_min and m_act <= cand:
            return cand
    return 192


def pick_w4_cfg(
    m_act: int, n_full: int, k_tiles: int, tile_n_min: int = 16, sm120: bool = False
) -> tuple:
    """(tile_m, tile_n, split_k). Measured invariant (H100, int4/qtip, incl.
    the machete faceoff): every winning config puts the grid at ~112-128 CTAs
    with the LARGEST tile that gets there — tile_m=128 beats 64 by 10-25% at
    equal CTA counts (2x TMA boxes, half the per-k-tile pipeline overhead per
    byte), tile_n is the largest with under half a tile of padding on m, and
    serial split-k makes up remaining grid coverage when each split keeps
    >= ~24 k-tiles (and tile_n <= 128: the f32 finalize round-trip scales
    with tile area). Prefill (m > 256): (128, 256, 1). ``tile_n_min``: the
    arch's tile_N floor. ``sm120``: RTX 5090-measured decode rule (see below)."""
    if n_full % 128 != 0:
        tn = _pick_tile_n(m_act, tile_n_min) if m_act <= 128 else 192
        mt = -(-m_act // tn)
        sk = 2 if (m_act <= 32 and (n_full // 64) * mt < 128 and k_tiles >= 32) else 1
        return 64, tn, sk
    if m_act > 256:
        # SM120 (RTX 5090 measured): short-K prefill wants 128-wide tiles —
        # qkv m=2048 (128,128,1) 207 TF vs (128,256,1) 177 (matches Marlin);
        # long-K keeps 256 (down m=2048: (128,256) 183 > (128,128) 175).
        if sm120 and k_tiles <= 96:
            return 128, 128, 1
        return 128, 256, 1
    n128 = n_full // 128
    if sm120 and m_act <= 64:
        # RTX 5090 measured (qtip2s/int4, all four Llama-8B layers,
        # 2026-07-29): m <= 64 wants 128-row tiles unconditionally — (128,
        # tn, sk) beats every 64-row config (qkv m=1: 12.3us vs 17.3; down:
        # 20.9 vs 25.7 — 1.0-1.4 TB/s of weight BW vs the 64-row ~0.7-1.1;
        # gateup m=64: (128,64,2) 84.5us vs the H100 rule's 114) — with the
        # LARGEST serial split-k keeping >= 32 k-tiles per split, relaxed to
        # >= 16 only when the grid would stay under ~96 CTAs (o-proj: 128
        # CTAs at 16 k-tiles beats 64 CTAs at 32). The 170-SM part rewards
        # more CTAs well past the H100 112 target (gateup m=1: 448 > 224).
        tn = _pick_tile_n(m_act, tile_n_min)
        sk = 1
        for cand in (2, 4, 8):
            if k_tiles // cand >= 32 and n128 * cand <= 512:
                sk = cand
        if n128 * sk < 96:
            for cand in (2, 4, 8):
                if cand > sk and k_tiles // cand >= 16 and n128 * cand >= 96:
                    sk = cand
                    break
        return 128, tn, sk
    for tn in (256, 128, 64, 32, 16):
        if tn < tile_n_min:
            break
        if tn >= 2 * m_act and tn > 16:
            continue  # half the tile or more would be padding
        mt = -(-m_act // tn)
        for sk in (1, 2, 4):
            if sk > 1 and (tn > 128 or k_tiles // sk < 24):
                break
            if n128 * mt * sk >= 112:
                return 128, tn, sk
    # coverage unreachable under the tile_m=128 constraints (small N, short K):
    # fall back to 64-row tiles with the plain starvation rule
    tn = _pick_tile_n(m_act, tile_n_min)
    mt = -(-m_act // tn)
    sk = 2 if ((n_full // 64) * mt < 128 and k_tiles >= 32) else 1
    return 64, tn, sk


def pick_w4a8_cfg(m_act: int, n_full: int) -> tuple:
    """(tile_m, tile_n, split_k). Measured (H100, N=K=8192, AI/bench_w4a8.py):
    slow accum flips the W4A16 preferences. 64-row tiles win every decode
    shape through m=128 (the doubled accumulator makes 128-row consumers
    register-heavy; occupancy-2 64-row tiles hide the decode+promote latency
    better: m=1 21.0 vs 23.6us, m=64 22.1 vs 29.3), with tile_n covering m in
    one tile column. 128-row wins from m=256 (128x128 37.8 vs 64x128 44.5);
    prefill caps tile_n at 192 (2*96+16 regs — 256-wide tiles spill).
    split-k always loses here (the fp32 finalize round-trip stacks on the
    promote: m=1 sk2 24.9 vs sk1 21.0)."""
    if m_act <= 128:
        tm, tn = 64, _pick_tile_n(m_act)
    elif m_act <= 256:
        tm, tn = 128, 128
    else:
        tm, tn = 128, 192
    if n_full % tm != 0:
        tm = 64
    return tm, tn, 1


def transform_a_operand(mod, A, values: dict, tile_m: int, tile_k: int = 64) -> TransformAOperand:
    """The kernel's A operand for a value transform with runtime operands: a
    TransformAOperand bundle of the plain (M, K) A and the operand views the
    mod's ``args`` declare (kind-dispatched; one aux-delivered operand for
    now). ``values`` maps fn param name -> tensor; ``tile_m``/``tile_k`` must
    match the launched config."""
    args = getattr(mod, "args", ())
    assert args, f"{getattr(mod, 'name', mod)!r} declares no runtime operands (args=)"
    assert set(values) == {name for name, _ in args}, (
        f"operand values {set(values)} must match the declared args {[n for n, _ in args]}"
    )
    assert A.ndim == 2, "runtime operands support the plain dense path"
    views = [ARG_KINDS[kind].host_view(A, values[name], tile_m, tile_k) for name, kind in args]
    assert len(views) == 1, "one aux-delivered operand per transform (single aux slot)"
    return TransformAOperand(A, views[0])


def transform_a_fake_operand(mod, mA_fake, a_dtype, tile_m: int, tile_k: int) -> TransformAOperand:
    """Trace-time counterpart of :func:`transform_a_operand`."""
    args = getattr(mod, "args", ())
    fakes = [ARG_KINDS[kind].host_fake(a_dtype, tile_m, tile_k) for _name, kind in args]
    assert len(fakes) == 1, "one aux-delivered operand per transform (single aux slot)"
    return TransformAOperand(mA_fake, fakes[0])


# ---- W4 blob / strip geometry ----------------------------------------------


def _w4_views(fmt: DecodeFormat, blob_u8, sf_u8, tile_m: int):
    """The kernel-facing views: blob (g, kt, 128, wpt*... bytes) ->
    (256, wpt, tm64, Gt, Kt, 1) with a 256 B contiguous TMA inner run, and
    the SF strip -> (sfb, tm64, Gt, Kt, 1). Works on real AND meta tensors —
    the fake path below reuses it so trace and launch layouts match by
    construction."""
    tm64 = tile_m // 64
    g, kt = blob_u8.shape[:2]
    gt = g // tm64
    wpt = (16 if fmt.w8 else 8) * (fmt.tile_k // 64)  # 256 B runs per (m64, k-tile)
    mA_t = blob_u8.view(gt, tm64, kt, wpt, 256).permute(4, 3, 1, 0, 2).unsqueeze(-1)
    mSFA_t = None
    if fmt.sf_words > 0:
        sfb = fmt.sf_bytes
        mSFA_t = sf_u8.reshape(g, kt, sfb).view(gt, tm64, kt, sfb).permute(3, 1, 0, 2).unsqueeze(-1)
    return mA_t, mSFA_t


def w4_operand_views(fmt: DecodeFormat, blob, sf, tile_m: int) -> TransformAOperand:
    """The kernel's A operand: a TransformAOperand bundle of torch views
    (blob + optional SF strip). It crosses the boundary in the mA slot as one
    argument — the host layer never unpacks it."""
    assert (sf is not None) == (fmt.sf_words > 0), (
        f"format {fmt.name!r} takes {'a repacked SF blob' if fmt.sf_words else 'sf=None'}"
    )
    blob_u8 = blob.view(torch.uint8) if blob.dtype != torch.uint8 else blob
    sf_u8 = sf.view(torch.uint8) if sf is not None else None
    return TransformAOperand(*_w4_views(fmt, blob_u8, sf_u8, tile_m))


def w4_fake_operands(fmt: DecodeFormat, n_full: int, k: int, tile_m: int) -> TransformAOperand:
    """Trace-time fake bundle with the exact (static) shapes/strides the
    runtime views produce."""
    tk = fmt.tile_k
    assert n_full % 64 == 0 and k % tk == 0
    g, kt = n_full // 64, k // tk
    wpt = (16 if fmt.w8 else 8) * (tk // 64)
    blob_meta = torch.empty(g * kt * wpt * 256, dtype=torch.uint8, device="meta")
    sf_meta = None
    if fmt.sf_words > 0:
        sf_meta = torch.empty(g * kt * fmt.sf_bytes, dtype=torch.uint8, device="meta")
    mA_m, mSFA_m = _w4_views(
        fmt,
        blob_meta.view(g, kt, wpt * 256),
        sf_meta.view(g, kt, fmt.sf_bytes) if sf_meta is not None else None,
        tile_m,
    )
    fake = lambda t: cute.runtime.make_fake_tensor(
        cutlass.Uint8, tuple(t.shape), stride=tuple(t.stride()), assumed_align=16
    )
    return TransformAOperand(fake(mA_m), fake(mSFA_m) if mSFA_m is not None else None)
