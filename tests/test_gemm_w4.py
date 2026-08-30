# Copyright (c) 2026, Tri Dao.
"""SM90 weight-only-quantized GEMM (W4A16) tests: scale-factor-strip formats
(nvfp4, int4, int4awq, mxfp4, mxfp8 — the strip rides the SFA aux operand),
the qtip trellis family, and fn-authored packed formats.

The roundtrip fixture (quantize -> the format's own offline repack -> GEMM ->
compare against the format's own fp32 dequant reference) is the only thing
pinning decode_k16 <-> prepare consistency — the framework never interprets
the blob or the strip — so a NEW format gets correctness coverage by
registering itself.
"""

import pytest
import torch

from quack.blockscaled import nvfp4_utils as U
from quack.operand_transform.formats import W4_FORMATS
from quack.cute_dsl_utils import get_device_capacity
from quack.gemm_w4 import gemm_w4a16
from quack.operand_transform import PackedInput, a_transform

_ARCH = get_device_capacity(torch.device("cuda"))[0] if torch.cuda.is_available() else 0
pytestmark = pytest.mark.skipif(
    _ARCH not in (9, 12),
    reason="W4A16 (register-sourced transform mainloop) is SM90/SM120 only",
)
# int4sm (promote) needs the per-k-tile promote seam, which SM120's warp-MMA
# mainloop does not implement; int4smf (folded, fast-accum fp8) runs on both.
sm90_only = pytest.mark.skipif(_ARCH != 9, reason="W4A8 promote (int4sm) is SM90 only")

# Every registered 16-bit-MMA format, opt-out via DecodeFormat.roundtrip
# (int8/fp8: per-channel scale is an epilogue concern). fp8-MMA formats are
# W4A8 — e4m3 activations, own tests below.
_ROUNDTRIP = [n for n, f in W4_FORMATS.items() if f.roundtrip and f.mma_dtype.width == 16]


@pytest.mark.parametrize("fmtname", _ROUNDTRIP)
@pytest.mark.parametrize("m,n,k", [(16, 1024, 2048), (300, 2112, 4096)])
def test_format_roundtrip(m, n, k, fmtname):
    fmt = W4_FORMATS[fmtname]
    torch.manual_seed(11)
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, sf_blob = fmt.prepare(q, sf)
    out = gemm_w4a16(act, blob, sf_blob, wformat=fmtname)
    ref = act.float() @ fmt.dequant_reference(q, sf).t()
    atol = ref.abs().max().item() * 2**-7 + 1e-5
    torch.testing.assert_close(out[:, :n].float(), ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("wformat,tile_k", [("qtip", 64), ("qtip2s", 64), ("qtip2", 128)])
def test_gemm_qtip_decode_exactness(wformat, tile_k):
    """Trellis decodes must be bit-exact vs the host reference: with one-hot
    activations the kernel output IS the decoded weight matrix. This pins the
    window extraction (incl. the tail-biting wrap into xw[0]) and the hash
    math against blockscaled/qtip.py. Random streams exercise every
    window/wrap path without a Viterbi run; tile_k=128 (qtip2) runs the
    8-block-per-tile mainloop."""
    torch.manual_seed(12)
    n, k = 128, 256
    blob = torch.randint(
        0, 256, (n // 64, k // tile_k, 128, tile_k // 4), device="cuda", dtype=torch.uint8
    )
    act = torch.eye(k, device="cuda", dtype=torch.bfloat16)
    out = gemm_w4a16(act, blob, wformat=wformat)  # (K, N) = dequant(W)^T
    w_ref = W4_FORMATS[wformat].dequant_reference(blob, None).to(torch.bfloat16)
    assert torch.equal(out, w_ref.t().contiguous()), f"{wformat} decode must be bit-exact"


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 4096, 4096),  # single-token decode
        (16, 2048, 5120),
        (300, 2112, 4096),  # M not a tile multiple
        (2048, 4096, 2048),  # prefill
    ],
)
def test_gemm_w4a16_shapes(m, n, k):
    torch.manual_seed(3)
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    fmt = W4_FORMATS["qtip2s"]
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, _ = fmt.prepare(q, sf)
    out = gemm_w4a16(act, blob, wformat="qtip2s")
    ref = act.float() @ fmt.dequant_reference(q, sf).t()
    atol = ref.abs().max().item() * 2**-7 + 1e-5
    torch.testing.assert_close(out[:, :n].float(), ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize(
    "tile_m,tile_n",
    [
        (64, 16),  # decode config, occupancy 2
        (128, 16),  # 2 math warpgroups + occupancy 2
        (128, 128),
        (256, 128),  # MMA_M = 2 per warpgroup
    ],
)
def test_gemm_w4a16_configs(tile_m, tile_n):
    torch.manual_seed(4)
    m, n, k = 512, 1536, 1152  # N divisible by 64 but ragged vs tiles; odd k-tiles
    if n % tile_m:
        pytest.skip("padded N not divisible by tile_m")
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    fmt = W4_FORMATS["qtip2s"]
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, _ = fmt.prepare(q, sf)
    out = gemm_w4a16(act, blob, wformat="qtip2s", tile_m=tile_m, tile_n=tile_n)
    ref = act.float() @ fmt.dequant_reference(q, sf).t()
    atol = ref.abs().max().item() * 2**-7 + 1e-5
    torch.testing.assert_close(out[:, :n].float(), ref, atol=atol, rtol=1e-2)


def test_gemm_w4a16_tensor_scale():
    """Per-tensor weight scale rides the epilogue alpha."""
    torch.manual_seed(5)
    m, n, k = 64, 512, 768
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    fmt = W4_FORMATS["qtip2s"]
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, _ = fmt.prepare(q, sf)
    out = gemm_w4a16(act, blob, tensor_scale=0.03125, wformat="qtip2s")
    ref = (act.float() @ fmt.dequant_reference(q, sf).t()) * 0.03125
    atol = ref.abs().max().item() * 2**-7 + 1e-5
    torch.testing.assert_close(out[:, :n].float(), ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("split_k", [2, 4])
def test_gemm_w4a16_split_k(split_k):
    """Serial split-k (the decode-shape grid-starvation fix) must stay
    bit-exact on the one-hot probe: each output row has a single nonzero
    contribution, so split accumulation introduces no rounding."""
    torch.manual_seed(14)
    n, k = 128, 512
    blob = torch.randint(0, 256, (n // 64, k // 64, 128, 16), device="cuda", dtype=torch.uint8)
    act = torch.eye(k, device="cuda", dtype=torch.bfloat16)
    out = gemm_w4a16(act, blob, wformat="qtip2s", split_k=split_k)
    w_ref = W4_FORMATS["qtip2s"].dequant_reference(blob, None).to(torch.bfloat16)
    assert torch.equal(out, w_ref.t().contiguous())
    # repeat on the CACHED buffers: the kernel must leave the semaphore reset
    out2 = gemm_w4a16(act, blob, wformat="qtip2s", split_k=split_k)
    assert torch.equal(out2, w_ref.t().contiguous())


def test_gemm_w4a16_split_k_auto():
    """The auto heuristic picks split_k=2 for a starved grid (N=4096 -> 64
    CTAs) and must match the split_k=1 result within accumulation tolerance."""
    torch.manual_seed(15)
    n, k = 4096, 4096
    fmt = W4_FORMATS["qtip2s"]
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, _ = fmt.prepare(q, sf)
    act = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
    out_auto = gemm_w4a16(act, blob, wformat="qtip2s")  # auto -> 2
    ref = act.float() @ fmt.dequant_reference(q, sf).t()
    atol = ref.abs().max().item() * 2**-7 + 1e-5
    torch.testing.assert_close(out_auto[:, :n].float(), ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize(
    "fmtname", ["nvfp4", "mxfp4", "mxfp8", "int4", "int4_g64", "int4_g32", "int4awq"]
)
def test_gemm_sf_decode_exactness(fmtname):
    """Every strip decode is a single bf16 rounding away from the fp32-exact
    reference product, so with one-hot activations the kernel output must be
    BIT-exact vs the host dequant reference cast to bf16 — fp4: e2m1's 2
    significand bits x a 4-bit e4m3 / power-of-2 e8m0 scale fit bf16's 8
    (fully exact); mxfp8: e4m3 c bf16 (incl. denormals, HMUL2 honors them)
    x power-of-2 scale (exact); int4/int4awq: the integer q-8 / q-z is
    exact via HADD2 (awq: c = -(128+z) folded into the bias add), then ONE
    HMUL2 rounding = the reference's cast. This pins the whole SFA path —
    strip repack, TMA box, smem layout, pair_slot word indexing — not just
    'close enough'."""
    torch.manual_seed(12)
    n, k = 256, 512
    fmt = W4_FORMATS[fmtname]
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, sf_blob = fmt.prepare(q, sf)
    act = torch.eye(k, device="cuda", dtype=torch.bfloat16)
    out = gemm_w4a16(act, blob, sf_blob, wformat=fmtname)  # (K, N) = dequant(W)^T
    w_ref = fmt.dequant_reference(q, sf).to(torch.bfloat16)
    assert torch.equal(out[:, :n], w_ref.t().contiguous()), f"{fmtname} decode must be bit-exact"


@pytest.mark.parametrize("tile_m,tile_n", [(64, 16), (128, 128), (256, 128)])
def test_gemm_sf_configs(tile_m, tile_n):
    """SF strip across tile configs: tile_m=128/256 exercise the multi-m64
    strip slices (and MMA_M = 2 per warpgroup at 256); (64, 16) the
    occupancy-2 decode config."""
    torch.manual_seed(6)
    m, n, k = 256, 1536, 1024
    if n % tile_m:
        pytest.skip("padded N not divisible by tile_m")
    fmt = W4_FORMATS["int4"]
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, sf_blob = fmt.prepare(q, sf)
    out = gemm_w4a16(act, blob, sf_blob, wformat="int4", tile_m=tile_m, tile_n=tile_n)
    ref = act.float() @ fmt.dequant_reference(q, sf).t()
    atol = ref.abs().max().item() * 2**-7 + 1e-5
    torch.testing.assert_close(out[:, :n].float(), ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("split_k", [2, 4])
def test_gemm_sf_split_k(split_k):
    """SF strip + serial split-k, bit-exact on the one-hot probe (see
    test_gemm_w4a16_split_k; nvfp4's decode is exact, so splitting adds no
    rounding)."""
    torch.manual_seed(16)
    n, k = 128, 1024
    fmt = W4_FORMATS["nvfp4"]
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, sf_blob = fmt.prepare(q, sf)
    act = torch.eye(k, device="cuda", dtype=torch.bfloat16)
    out = gemm_w4a16(act, blob, sf_blob, wformat="nvfp4", split_k=split_k)
    w_ref = fmt.dequant_reference(q, sf).to(torch.bfloat16)
    assert torch.equal(out[:, :n], w_ref.t().contiguous())


# ── W4A8 (int4sm: promote / slow-accum format) ───────────────────────────────


def _w4a8_setup(m, n, k, seed=13):
    from quack.gemm_w4 import quantize_act_per_token_fp8

    fmt = W4_FORMATS["int4sm"]
    torch.manual_seed(seed)
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float32) * 2.0
    q, sf = fmt.quantize_reference(w)
    blob, sf_blob = fmt.prepare(q, sf)
    af, sfa = quantize_act_per_token_fp8(act)
    # both operands' quantization is in the reference: the kernel is exact up
    # to the fp8 WGMMA accumulator + one bf16 store rounding
    ref = (af.float() * sfa[:, None]) @ fmt.dequant_reference(q, sf).t()
    return af, sfa, blob, sf_blob, ref


@pytest.mark.parametrize("k", [128, 256, 384, 1024])
@sm90_only
def test_gemm_w4a8_decode_exactness(k):
    """One-hot e4m3 activations with unit token scales make the kernel output
    the dequantized weights EXACTLY (q and the promote are fp32-exact; one
    bf16 rounding at D — same as the reference cast): pins the fp8-fragment
    blob order, the strip slots, and the per-k-tile promote (k sweeps the
    first/preload/steady/last mainloop tile structures)."""
    from quack.gemm_w4 import gemm_w4a8

    fmt = W4_FORMATS["int4sm"]
    torch.manual_seed(14)
    n = 128
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, sf_blob = fmt.prepare(q, sf)
    af = torch.eye(k, device="cuda").to(torch.float8_e4m3fn)
    ones = torch.ones(k, device="cuda", dtype=torch.float32)
    out = gemm_w4a8(af, blob, sf_blob, act_scale=ones, tile_m=128, tile_n=64, split_k=1)
    w_ref = fmt.dequant_reference(q, sf).to(torch.bfloat16)
    assert torch.equal(out.t().contiguous(), w_ref), "w4a8 decode+promote must be bit-exact"


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 4096, 4096),  # single-token decode (small-N budgets + auto split-k)
        (16, 2048, 5120),
        (300, 2112, 4096),  # M not a tile multiple, N padded to 128
        (2048, 4096, 2048),  # prefill (tile_n capped at 192: slow accum)
    ],
)
@sm90_only
def test_gemm_w4a8_shapes(m, n, k):
    from quack.gemm_w4 import gemm_w4a8

    af, sfa, blob, sf_blob, ref = _w4a8_setup(m, n, k)
    out = gemm_w4a8(af, blob, sf_blob, act_scale=sfa)
    atol = ref.abs().max().item() * 2**-7 + 1e-5
    torch.testing.assert_close(out[:, :n].float(), ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("tile_m,tile_n", [(64, 128), (128, 128), (128, 192)])
@sm90_only
def test_gemm_w4a8_configs(tile_m, tile_n):
    from quack.gemm_w4 import gemm_w4a8

    af, sfa, blob, sf_blob, ref = _w4a8_setup(128, 1024, 2048, seed=15)
    out = gemm_w4a8(af, blob, sf_blob, act_scale=sfa, tile_m=tile_m, tile_n=tile_n, split_k=1)
    atol = ref.abs().max().item() * 2**-7 + 1e-5
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("split_k", [2, 4])
@sm90_only
def test_gemm_w4a8_split_k(split_k):
    from quack.gemm_w4 import gemm_w4a8

    af, sfa, blob, sf_blob, ref = _w4a8_setup(8, 1024, 8192, seed=16)
    out = gemm_w4a8(af, blob, sf_blob, act_scale=sfa, tile_m=128, tile_n=16, split_k=split_k)
    atol = ref.abs().max().item() * 2**-7 + 1e-5
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("k", [128, 384, 1024])
def test_gemm_w4a8_folded_decode_exactness(k):
    """Folded W4A8 (int4smf): the scaled-LUT decode's e4m3 rounding must
    match the reference exactly — one-hot acts pin the per-(row, tile) table
    build (cvt.rn.satfinite) against dequant_int4smf_reference bitwise."""
    from quack.gemm_w4 import gemm_w4a8

    fmt = W4_FORMATS["int4smf"]
    torch.manual_seed(18)
    n = 128
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, strip = fmt.prepare(q, sf)
    af = torch.eye(k, device="cuda").to(torch.float8_e4m3fn)
    ones = torch.ones(k, device="cuda", dtype=torch.float32)
    out = gemm_w4a8(
        af,
        blob,
        strip,
        act_scale=ones,
        chan_scale=sf[1],
        tile_m=128,
        tile_n=64,
        split_k=1,
        wformat="int4smf",
    )
    w_ref = fmt.dequant_reference(q, sf).to(torch.bfloat16)
    assert torch.equal(out.t().contiguous(), w_ref)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 4096, 4096),
        (300, 2112, 4096),  # ragged M/N (chan_scale padding path)
        (2048, 4096, 2048),  # prefill (no-drain fast-accum path, tile_n 256)
    ],
)
def test_gemm_w4a8_folded_shapes(m, n, k):
    from quack.gemm_w4 import gemm_w4a8, quantize_act_per_token_fp8

    fmt = W4_FORMATS["int4smf"]
    torch.manual_seed(19)
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float32) * 2.0
    q, sf = fmt.quantize_reference(w)
    blob, strip = fmt.prepare(q, sf)
    af, sfa = quantize_act_per_token_fp8(act)
    out = gemm_w4a8(af, blob, strip, act_scale=sfa, chan_scale=sf[1], wformat="int4smf")
    # the reference already carries the fold's e4m3 rounding; the remaining
    # error is the fast-accum fp8 chain + one bf16 store rounding
    ref = (af.float() * sfa[:, None]) @ fmt.dequant_reference(q, sf).t()
    atol = ref.abs().max().item() * 2**-6 + 1e-5
    torch.testing.assert_close(out[:, :n].float(), ref, atol=atol, rtol=2e-2)


@sm90_only
def test_gemm_w4a8_bf16_act_convenience():
    """bf16 acts quantize per-token internally; bitwise == the manual path."""
    from quack.gemm_w4 import gemm_w4a8, quantize_act_per_token_fp8

    torch.manual_seed(17)
    m, n, k = 32, 512, 1024
    fmt = W4_FORMATS["int4sm"]
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, sf_blob = fmt.prepare(q, sf)
    out = gemm_w4a8(act, blob, sf_blob, tile_m=128, tile_n=32, split_k=1)
    af, sfa = quantize_act_per_token_fp8(act)
    out2 = gemm_w4a8(af, blob, sf_blob, act_scale=sfa, tile_m=128, tile_n=32, split_k=1)
    assert torch.equal(out, out2)


def test_sf_blob_arg_mismatch():
    """Strip formats require the SF blob (and strip-free formats reject one)."""
    act = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    blob = torch.zeros(2, 2, 128, 16, device="cuda", dtype=torch.uint8)
    with pytest.raises(AssertionError, match="repacked SF blob"):
        gemm_w4a16(act, blob, wformat="nvfp4")
    with pytest.raises(AssertionError, match="sf=None"):
        gemm_w4a16(act, blob, blob, wformat="qtip2s")


# ── "New format in how many lines?" demo: fn-authored packed decode ──────────
# int8 weights with a per-tensor bf16 scale folded into the decode (a format
# none of the registered ones cover without the epilogue-scale plumbing),
# composed entirely from existing building blocks. Everything below IS the
# full implementation: host references, repack reuse, and the decode fn. It
# plugs into the stock gemm_w4a16 wrapper unchanged.

_I8_SCALE = 0.03125  # power of 2: bf16-exact, so the roundtrip is exact


def _bf16x2_bits(x: float) -> int:
    b = torch.tensor(x, dtype=torch.bfloat16).view(torch.uint16).item()
    return b | (b << 16)


def _quantize_int8_ts(w):
    q = (w.float() / _I8_SCALE).round().clamp(-128, 127).to(torch.int8)
    return q, None


_int8_ts = PackedInput(
    name="int8_ts",
    w8=True,
    prepare=lambda q, sf: (U.repack_w8a16_weight(q), None),
    quantize_reference=_quantize_int8_ts,
    dequant_reference=lambda q, sf: q.float() * _I8_SCALE,
)


@a_transform(packed=_int8_ts)
def _int8_ts_fn(xw, sfw, b, consts):
    from cutlass import Int32

    ra, rb = U.decode_i8x4_to_bf16x4_dp4a(xw[2 * b])
    rc, rd = U.decode_i8x4_to_bf16x4_dp4a(xw[2 * b + 1])
    h = Int32(_bf16x2_bits(_I8_SCALE))
    return (
        U.mul_bf16x2_bcast(ra, h, False),
        U.mul_bf16x2_bcast(rb, h, True),
        U.mul_bf16x2_bcast(rc, h, False),
        U.mul_bf16x2_bcast(rd, h, True),
    )


@pytest.mark.parametrize("m,n,k", [(16, 1024, 2048), (300, 2112, 4096)])
def test_new_format_int8_ts_demo(m, n, k):
    fmt = _int8_ts_fn.as_decode_format()
    torch.manual_seed(13)
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, _ = fmt.prepare(q, sf)
    out = gemm_w4a16(act, blob, wformat=fmt)
    ref = act.float() @ fmt.dequant_reference(q, sf).t()
    atol = ref.abs().max().item() * 2**-7 + 1e-5
    torch.testing.assert_close(out[:, :n].float(), ref, atol=atol, rtol=1e-2)


def test_w4_epi_mod_composition():
    """@gemm_epilogue fns compose with layout-owning transforms
    (mod(act, blob, transform_a=...)): everything is caller-oriented — out is
    (m, n_full); a per-out-channel vector infers as the kernel colvec, a
    per-token vector as the kernel rowvec (the swap_ab pin/inference flip)."""
    from quack.epilogue.frontend import gemm_epilogue

    @gemm_epilogue()
    def _bias_tscale(acc, bias, tscale):
        return {"D": (acc + bias) * tscale}

    torch.manual_seed(21)
    m, n, k = 64, 1024, 2048
    fmt = W4_FORMATS["int4"]
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, sf_blob = fmt.prepare(q, sf)
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    bias = torch.randn(1, n, device="cuda", dtype=torch.float32)
    tscale = torch.randn(1, m, device="cuda", dtype=torch.float32)
    res = _bias_tscale(
        act, blob, transform_a="int4", transform_sf=sf_blob, bias=bias, tscale=tscale
    )
    out = res["D"]
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    ref = (act.float() @ fmt.dequant_reference(q, sf).t() + bias) * tscale.t()
    atol = ref.abs().max().item() * 2**-7 + 1e-5
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=1e-2)
    # warm replay rides mod.gemm's plan cache (blob/strip views rebuilt per call)
    out2 = _bias_tscale(
        act, blob, transform_a="int4", transform_sf=sf_blob, bias=bias, tscale=tscale
    )["D"]
    assert torch.equal(out2, out)


def test_w4_epi_mod_torch_compile():
    """Layout-owning transforms under torch.compile through the single
    quack::gemm_epi op: the format crosses by digest (w4_transform handle —
    a bare string name has none and is rejected), the SF strip rides the op
    input list, and D's N comes from the blob rows (n_override) on the fake
    side. Compiled == eager bitwise."""
    import pytest

    from quack.epilogue.frontend import gemm_epilogue
    from quack.operand_transform import w4_transform

    @gemm_epilogue()
    def _bias_w4c(acc, bias):
        return {"D": acc + bias}

    torch.manual_seed(23)
    m, n, k = 64, 1024, 2048
    fmt = W4_FORMATS["int4"]
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    q, sf = fmt.quantize_reference(w)
    blob, sf_blob = fmt.prepare(q, sf)
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    bias = torch.randn(1, n, device="cuda", dtype=torch.float32)
    handle = w4_transform("int4")

    def f(act, blob, sf_blob, bias):
        return _bias_w4c(act, blob, transform_a=handle, transform_sf=sf_blob, bias=bias)["D"]

    eager = f(act, blob, sf_blob, bias)
    cf = torch.compile(f, fullgraph=True, dynamic=False)
    comp = cf(act, blob, sf_blob, bias)
    assert torch.equal(comp, eager), "w4 transform under compile != eager"
    act2 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    assert torch.equal(cf(act2, blob, sf_blob, bias), f(act2, blob, sf_blob, bias))

    def f_str(act, blob, sf_blob, bias):
        return _bias_w4c(act, blob, transform_a="int4", transform_sf=sf_blob, bias=bias)["D"]

    # Dynamo re-wraps the in-trace NotImplementedError (fullgraph), message intact
    with pytest.raises(Exception, match="digest-carrying"):
        torch.compile(f_str, fullgraph=True, dynamic=False)(act, blob, sf_blob, bias)
