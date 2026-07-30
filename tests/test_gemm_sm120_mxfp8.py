# Copyright (c) 2026, Tri Dao.
"""SM120 block-scaled GEMM (warp-level kind::mxf8f6f4 with REAL e8m0 scale
factors): numerics vs the dequantized reference, and bit-exact vs cuBLAS.

Scope: independent A/B dtypes across fp8 (e4m3/e5m2) / fp6 (e2m3/e3m2,
packed) / fp4 (e2m1, packed) — same-dtype fp8 rides MmaMXF8Op, same-dtype
fp4 the packed kind::mxf4 (e8m0/vec32) / kind::mxf4nvf4 (e4m3/vec16) atoms,
everything else MmaMXF8F6F4OpFull. Sub-byte sides of MIXED pairs are
TMA-loaded via padded tensormaps (16U4_ALIGN8B / 16U6_ALIGN16B) and unpacked
at s2r by ldsm.b4x16_p64 / b6x16_p32 (fp4 additionally shifted <<2 into MMA
position); same-dtype fp4 stays packed throughout. MN-major operands are
fp8-only: both sides ride the transposing m16n16.trans.b8 ldmatrix — A via
make_tiled_copy_A, B via a hand-built TV layout (_nmajor_b_tiled_copy) — which
also unlocks varlen_k (m-major A / n-major B, K-padded SF buffers, ragged-tail
MMA skip). Unlike the plain-fp8 unit-scale fast path (which falls back to
MmaFP8Op on the H100 CI proxy), these instructions REQUIRE an sm_120a target,
so the tests run on SM120 only.
"""

import pytest
import torch
import torch.nn.functional as F

from quack.blockscaled.operand import BlockScaledFormat, BlockScaledOperand
from quack.blockscaled.utils import blockscaled_quantize, scale_blocked_for_cublas
from quack.cute_dsl_utils import get_device_capacity
from quack.gemm_config import GemmConfig
from quack.gemm_interface import (
    _prep_blockscaled,
    _sf_batch_canonicalize,
    _unpack_operand,
    gemm,
    gemm_act,
    gemm_add,
    gemm_add_inplace,
    gemm_blockscaled_ref,
    gemm_tuned,
)

_ARCH = get_device_capacity(torch.device("cuda"))[0] if torch.cuda.is_available() else 0
requires_sm120 = pytest.mark.skipif(_ARCH != 12, reason="SM120 blockscaled warp-MMA path")


def _quantized_operands(fmt, m, n, k, batched=False, seed=0):
    torch.manual_seed(seed)
    L = 2 if batched else 1
    shape_a = (L, m, k) if batched else (m, k)
    shape_w = (L, n, k) if batched else (n, k)
    a_hp = torch.randn(*shape_a, device="cuda", dtype=torch.bfloat16) * k**-0.5
    w_hp = torch.randn(*shape_w, device="cuda", dtype=torch.bfloat16) * k**-0.5
    qa, sfa = blockscaled_quantize(a_hp, fmt)
    qw, sfw = blockscaled_quantize(w_hp, fmt)
    fmt_obj = BlockScaledFormat.from_name(fmt)
    A = BlockScaledOperand.from_parts(qa, sfa, fmt_obj)
    W = BlockScaledOperand.from_parts(qw, sfw, fmt_obj)
    return A, W.mT  # B = (K, N) logical view; qdata stride-swap, scale unchanged


def _gemm_with_config(A, B, config=None, split_k=1):
    """gemm(A, B, tuned=False) with a forced GemmConfig (the public wrapper
    exposes no config knob)."""
    opA, opB = _unpack_operand(A), _unpack_operand(B)
    Ad, Bd = opA.data, opB.data
    SFA, SFB, bs_format_a, bs_format_b = _prep_blockscaled(opA, opB)
    SFA, SFB = _sf_batch_canonicalize(SFA, SFB, Ad.ndim == 3)
    out_shape = (
        (Ad.shape[0], Bd.shape[-1]) if Ad.ndim == 2 else (Ad.shape[0], Ad.shape[-2], Bd.shape[-1])
    )
    out = torch.empty(out_shape, dtype=torch.bfloat16, device=Ad.device)
    gemm_tuned.fn(
        Ad,
        Bd,
        out,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        config=config,
        split_k=split_k,
    )
    return out


def _sm120_config(tile_m, tile_n, pingpong=False):
    return GemmConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        cluster_m=1,
        cluster_n=1,
        pingpong=pingpong,
        is_dynamic_persistent=True,
        device_capacity=12,
    )


def _rel_err(out, ref):
    return (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()


@requires_sm120
@pytest.mark.parametrize("fmt", ["mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "shape_mnk",
    [
        (256, 256, 256),
        (128, 128, 256),
        (448, 320, 512),  # M, N not multiples of 128 (padded SF rows)
        (1024, 256, 8192),  # long K: SF pipeline over many k-tiles
    ],
)
def test_sm120_mxfp8_gemm(fmt, batched, shape_mnk):
    m, n, k = shape_mnk
    A, B = _quantized_operands(fmt, m, n, k, batched)
    out = gemm(A, B, tuned=False)
    ref = gemm_blockscaled_ref(A, B)
    expected_shape = (2, m, n) if batched else (m, n)
    assert out.shape == expected_shape and out.dtype == torch.bfloat16
    rel = _rel_err(out, ref)
    assert rel < 5e-3, f"{fmt} {shape_mnk} batched={batched}: rel_err={rel}"


@requires_sm120
@pytest.mark.parametrize(
    "tile_mn,pingpong",
    [
        ((128, 128), True),
        ((256, 128), False),
        ((128, 256), False),
        ((256, 256), False),
    ],
)
def test_sm120_mxfp8_tiles(tile_mn, pingpong):
    m, n, k = 512, 512, 512
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    out = _gemm_with_config(A, B, config=_sm120_config(*tile_mn, pingpong=pingpong))
    ref = gemm_blockscaled_ref(A, B)
    rel = _rel_err(out, ref)
    assert rel < 5e-3, f"tile={tile_mn} pingpong={pingpong}: rel_err={rel}"


@requires_sm120
def test_sm120_mxfp8_split_k():
    m, n, k = 128, 128, 4096
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    out = _gemm_with_config(A, B, split_k=2)
    ref = gemm_blockscaled_ref(A, B)
    rel = _rel_err(out, ref)
    assert rel < 5e-3, f"split_k=2: rel_err={rel}"


@requires_sm120
@pytest.mark.parametrize("seqlens_m", [[128, 128, 128], [100, 200, 150], [1, 128, 127, 129]])
def test_sm120_mxfp8_varlen_m(seqlens_m):
    """Grouped (varlen_m) MXFP8 GEMM: SFA is a single M-padded buffer
    (tile-aligned per-batch padding, batch dim 1); SFB stays per-expert."""
    import cutlass

    from quack.blockscaled.utils import create_blockscaled_varlen_m_operands

    num_experts = len(seqlens_m)
    n, k = 256, 256
    torch.manual_seed(0)
    a_ref_dq, b_ref_dq, qa, qb, a_sc, b_sc, cu_seqlens_m = create_blockscaled_varlen_m_operands(
        num_experts, 0, n, k, 32, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, seqlens_m=seqlens_m
    )
    B = qb.permute(2, 1, 0)  # (n, k, L) -> (L, K, N) with K contiguous
    A_op = BlockScaledOperand.from_parts(qa, a_sc, "mxfp8")
    B_op = BlockScaledOperand.from_parts(B, b_sc, "mxfp8", quant_dim=-2)
    out = gemm(A_op, B_op, cu_seqlens_m=cu_seqlens_m, tuned=False)

    cu = cu_seqlens_m.tolist()
    ref = torch.cat([a_ref_dq[cu[i] : cu[i + 1]] @ b_ref_dq[i].T for i in range(num_experts)])
    err = (out.float() - ref).abs().max().item()
    assert err < 5e-3, f"varlen_m seqlens_m={seqlens_m} max_err={err}"


@requires_sm120
def test_sm120_mxfp8_vs_cublas():
    """Bit-exact comparison against torch._scaled_mm (cuBLAS MXFP8 path).
    Both consume the same fp8 values and e8m0 scales with f32 accumulation, so
    any scale mis-application (wrong k-block, wrong row, stale smem stage)
    shows up as a hard mismatch."""
    m, n, k = 512, 512, 512
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    out = gemm(A, B, tuned=False)
    sfa_flat = scale_blocked_for_cublas(A.scale.unsqueeze(0), m, k // 32)
    sfw_flat = scale_blocked_for_cublas(B.scale.unsqueeze(0), n, k // 32)
    out_cublas = torch._scaled_mm(
        A.qdata, B.qdata, scale_a=sfa_flat, scale_b=sfw_flat, out_dtype=torch.bfloat16
    )
    assert torch.equal(out, out_cublas), (
        f"quack != cuBLAS: max_err={(out.float() - out_cublas.float()).abs().max().item()}"
    )


@requires_sm120
def test_sm120_mxfp8_gemm_add():
    """Epilogue frontend (alpha/beta + C) with blockscaled operands."""
    m, n, k = 256, 256, 512
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    C = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    alpha, beta = 0.5, 2.0
    out = gemm_add(A, B, C, alpha=alpha, beta=beta, tuned=False)
    ref = alpha * gemm_blockscaled_ref(A, B, out_dtype=torch.float32) + beta * C.float()
    rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 5e-3, f"gemm_add: rel_err={rel}"

    acc = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    ref2 = gemm_blockscaled_ref(A, B, out_dtype=torch.float32) + acc.float()
    gemm_add_inplace(A, B, acc, tuned=False)
    rel = (acc.float() - ref2).abs().max().item() / ref2.abs().max().item()
    assert rel < 5e-3, f"gemm_add_inplace: rel_err={rel}"


@requires_sm120
def test_sm120_mxfp8_gemm_add_tuned():
    """Autotuned path: sweeps the blockscaled_config_ok-pruned SM120 space."""
    m, n, k = 512, 512, 512
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    C = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    out = gemm_add(A, B, C, alpha=0.5, beta=2.0)
    ref = 0.5 * gemm_blockscaled_ref(A, B, out_dtype=torch.float32) + 2.0 * C.float()
    rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 5e-3, f"tuned gemm_add: rel_err={rel}"


@requires_sm120
@pytest.mark.parametrize("activation", ["relu", "gelu_tanh_approx"])
def test_sm120_mxfp8_gemm_act(activation):
    """gemm_act with blockscaled A/B, checked against the dequant reference."""
    m, n, k = 256, 256, 512
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    act_fn = {
        "relu": F.relu,
        "gelu_tanh_approx": lambda x: F.gelu(x, approximate="tanh"),
    }[activation]
    preact, postact = gemm_act(A, B, activation=activation, tuned=False)
    ref_post = act_fn(gemm_blockscaled_ref(A, B, out_dtype=torch.float32))
    rel = (postact.float() - ref_post).abs().max().item() / max(ref_post.abs().max().item(), 1e-6)
    assert rel < 5e-3, f"gemm_act {activation}: rel_err={rel}"


def _mixed_operand(fmt, rows, k, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    return BlockScaledOperand.quantize(x, fmt)


_MIXED_PAIRS = [
    # fp4 x fp8 (16U4_ALIGN8B TMA + b4x16_p64 ldmatrix + <<2 shift)
    ("mxfp4", "mxfp8_e4m3"),
    ("mxfp8_e4m3", "mxfp4"),
    ("mxfp4", "mxfp8_e5m2"),
    ("mxfp8_e5m2", "mxfp4"),
    # same-width mixed fp8 (independent .e4m3/.e5m2 qualifiers)
    ("mxfp8_e4m3", "mxfp8_e5m2"),
    ("mxfp8_e5m2", "mxfp8_e4m3"),
    # fp6 (16U6_ALIGN16B TMA + b6x16_p32 ldmatrix, no shift): x fp8, x fp6
    # (mixed and same-dtype), x fp4
    ("mxfp6_e2m3_packed", "mxfp8_e4m3"),
    ("mxfp8_e4m3", "mxfp6_e2m3_packed"),
    ("mxfp6_e2m3_packed", "mxfp6_e3m2_packed"),
    ("mxfp6_e2m3_packed", "mxfp6_e2m3_packed"),
    ("mxfp6_e2m3_packed", "mxfp4"),
    ("mxfp4", "mxfp6_e3m2_packed"),
]


@requires_sm120
@pytest.mark.parametrize("fmt_pair", _MIXED_PAIRS)
@pytest.mark.parametrize(
    "shape_mnk",
    [
        (256, 512, 512),
        (448, 320, 512),  # M, N not multiples of 128 (padded SF rows)
        (256, 256, 8192),  # long K: ALIGN8B + SF pipeline over many k-tiles
    ],
)
def test_sm120_mixed_fp4_fp8_gemm(fmt_pair, shape_mnk):
    """Mixed-dtype blockscaled pairs (kind::mxf8f6f4, independent a/b dtype
    qualifiers — the CUTLASS C++ SM120_16x8x32_TN matrix): sub-byte operands
    ride padded tensormaps (16U4_ALIGN8B / 16U6_ALIGN16B) into byte-domain
    smem, ldsm.b4x16_p64 / b6x16_p32 unpacks into byte lanes, and fp4
    fragments are shifted <<2 into MMA position. Checked against both the
    blockscaled reference and the dequantized product."""
    fmt_a, fmt_b = fmt_pair
    m, n, k = shape_mnk
    A = _mixed_operand(fmt_a, m, k, seed=0)
    W = _mixed_operand(fmt_b, n, k, seed=1)
    out = gemm(A, W.mT, tuned=False)
    ref = gemm_blockscaled_ref(A, W.mT)
    rel = _rel_err(out, ref)
    assert rel < 5e-3, f"{fmt_a} x {fmt_b} {shape_mnk}: rel_err={rel}"
    ref_dq = A.dequantize(torch.float32) @ W.dequantize(torch.float32).T
    rel_dq = (out.float() - ref_dq).abs().max().item() / ref_dq.abs().max().item()
    assert rel_dq < 5e-3, f"{fmt_a} x {fmt_b} {shape_mnk}: rel_err vs dequant={rel_dq}"


@requires_sm120
@pytest.mark.parametrize(
    "tile_mn,pingpong",
    [((128, 128), True), ((256, 128), False), ((128, 256), False)],
)
def test_sm120_mixed_fp4_fp8_tiles(tile_mn, pingpong):
    m, n, k = 512, 512, 512
    A = _mixed_operand("mxfp4", m, k, seed=0)
    W = _mixed_operand("mxfp8_e4m3", n, k, seed=1)
    B = W.mT
    out = _gemm_with_config(A, B, config=_sm120_config(*tile_mn, pingpong=pingpong))
    ref = gemm_blockscaled_ref(A, B)
    rel = _rel_err(out, ref)
    assert rel < 5e-3, f"mixed tile={tile_mn} pingpong={pingpong}: rel_err={rel}"


@requires_sm120
def test_sm120_mixed_fp4_fp8_gemm_add():
    """Mixed operands through the epilogue frontend."""
    m, n, k = 256, 256, 512
    A = _mixed_operand("mxfp8_e4m3", m, k, seed=0)
    W = _mixed_operand("mxfp4", n, k, seed=1)
    B = W.mT
    C = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    out = gemm_add(A, B, C, alpha=0.5, beta=2.0, tuned=False)
    ref = 0.5 * gemm_blockscaled_ref(A, B, out_dtype=torch.float32) + 2.0 * C.float()
    rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 5e-3, f"mixed gemm_add: rel_err={rel}"


@requires_sm120
@pytest.mark.parametrize("tile_m", [128, 64])
def test_sm120_plain_mixed_fp8_gemm(tile_m):
    """PLAIN (non-blockscaled) mixed e4m3 x e5m2. tile_m=128 rides
    kind::mxf8f6f4 with constant unit scales (full Blackwell rate); tile_m=64
    forces the Ada-instruction fallback (MmaFP8MixedOp — the sm_89 opcode
    takes independent .e4m3/.e5m2 qualifiers), the same path H100 CI proxy
    legs take."""
    import math

    torch.manual_seed(0)
    m, n, k = 256, 256, 512
    A = (torch.randn(m, k, device="cuda") / math.sqrt(k)).to(torch.float8_e4m3fn)
    B = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.float8_e5m2)
    D = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    from quack.gemm import gemm as gemm_ffi

    gemm_ffi(A, B, D, None, None, tile_m, 128, 1, 1)
    torch.cuda.synchronize()
    ref = A.float() @ B.float().mT
    atol = ref.abs().max().item() * 2**-7 + 1e-6
    torch.testing.assert_close(D.float(), ref, atol=atol, rtol=1e-2)


@requires_sm120
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize(
    "shape_mnk",
    [(256, 256, 256), (448, 320, 512), (256, 256, 8192)],
)
def test_sm120_same_dtype_fp4_gemm(fmt, shape_mnk):
    """Same-dtype fp4 rides the dedicated packed kinds — kind::mxf4 (e8m0
    scales, sf_vec 32) / kind::mxf4nvf4 (e4m3 scales, sf_vec 16) — with
    inst K 64, regular packed fp4 smem/TMA, plain 16-bit ldmatrix, and no
    register shift. NVFP4's per-tensor scale folds into alpha at the
    interface."""
    m, n, k = shape_mnk
    A, B = _quantized_operands(fmt, m, n, k)
    out = gemm(A, B, tuned=False)
    ref = gemm_blockscaled_ref(A, B)
    rel = _rel_err(out, ref)
    assert rel < 5e-3, f"{fmt} {shape_mnk}: rel_err={rel}"


@requires_sm120
@pytest.mark.parametrize("fmt_b", ["mxfp8_e4m3", "mxfp4"])
def test_sm120_mxfp8_a_m_major(fmt_b):
    """M-major fp8 A (byte-granularity transposing ldmatrix m16n16.trans.b8):
    the scale tensor is layout-independent, only the qdata strides swap."""
    m, n, k = 256, 320, 512
    A, _ = _quantized_operands("mxfp8_e4m3", m, n, k)
    W = _mixed_operand(fmt_b, n, k, seed=1)
    ref = gemm_blockscaled_ref(A, W.mT)
    qa_mm = A.qdata.t().contiguous().t()  # (m, k) with M contiguous
    assert qa_mm.stride() == (1, m)
    A_mm = BlockScaledOperand.from_parts(qa_mm, A.scale, A.format)
    out = gemm(A_mm, W.mT, tuned=False)
    rel = _rel_err(out, ref)
    assert rel < 5e-3, f"m-major A x {fmt_b}: rel_err={rel}"


def _n_major_b(W):
    """(k, n) row-major view of an (n, k) quantized operand — the interface's
    .mT relabel makes it n-major B. The scale tensor is layout-independent."""
    qb_kn = W.qdata.t().contiguous()  # (k, n) with N contiguous
    return BlockScaledOperand.from_parts(qb_kn, W.scale, W.format, quant_dim=-2)


@requires_sm120
@pytest.mark.parametrize("fmt_a", ["mxfp8_e4m3", "mxfp8_e5m2", "mxfp4", "mxfp6_e2m3_packed"])
def test_sm120_b_n_major(fmt_a):
    """N-major fp8 B (ldmatrix.m16n16.x2.trans.b8 through a hand-built TV
    layout, see GemmSm120._nmajor_b_tiled_copy) under every A flavor that
    pairs with fp8 B (same-dtype mxfp8, mixed e5m2 x e4m3, fp4 x fp8,
    fp6 x fp8): must be bit-identical to the k-major B run on the same
    quantized operands."""
    m, n, k = 256, 320, 512
    A = _mixed_operand(fmt_a, m, k, seed=0)
    W = _mixed_operand("mxfp8_e4m3", n, k, seed=1)
    out_kmaj = gemm(A, W.mT, tuned=False)
    out = gemm(A, _n_major_b(W), tuned=False)
    assert torch.equal(out, out_kmaj), (
        f"{fmt_a} x n-major fp8 B != k-major: "
        f"max_err={(out.float() - out_kmaj.float()).abs().max().item()}"
    )


@requires_sm120
@pytest.mark.parametrize(
    "tile_mn,pingpong",
    [((128, 128), True), ((256, 128), False), ((128, 256), False), ((256, 256), False)],
)
def test_sm120_b_n_major_tiles(tile_mn, pingpong):
    """tile_n 256 splits the B fragment's N rest mode AROUND K (the copy view
    regroup in _retile_b must follow the fragment layout, not assume plain
    (V, N, K) col-major); pingpong exercises the within-warp-group tidx
    remap."""
    m, n, k = 512, 512, 512
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    config = _sm120_config(*tile_mn, pingpong=pingpong)
    out_kmaj = _gemm_with_config(A, B, config=config)
    W = B.mT  # undo the .mT from _quantized_operands
    out = _gemm_with_config(A, _n_major_b(W), config=config)
    assert torch.equal(out, out_kmaj), f"tile={tile_mn} pingpong={pingpong}: n-major != k-major"


@requires_sm120
def test_sm120_fp4_b_n_major_rejected():
    """Packed fp4 cannot be n-major (nibbles pack along the quantized K axis;
    no ldmatrix variant can transpose them) — the operand container rejects
    the layout at construction, before any kernel is minted."""
    W = _mixed_operand("mxfp4", 256, 512, seed=0)
    with pytest.raises(ValueError, match="packed dim"):
        BlockScaledOperand.from_parts(W.qdata.t().contiguous(), W.scale, W.format, quant_dim=-2)


@requires_sm120
@pytest.mark.parametrize("seqlens_k", [[96, 160, 128], [100, 220, 65]])
def test_sm120_mxfp8_varlen_k_poisoned_sf_pad(seqlens_k):
    """varlen_k (m-major A / n-major B, K-padded SF buffers): SF pad bytes are
    TMA-loaded but must never be consumed — the mma loop skips the
    instructions covering the ragged tail (GemmSm120.mma's
    sf_valid_insts_last_tile). Poison
    the pad with 0xFF (e8m0 NaN): any consumed pad byte NaNs whole output rows
    via NaN-scale x 0-value products."""
    from quack.blockscaled.utils import create_blockscaled_varlen_k_operands

    num_experts = len(seqlens_k)
    m, n, sf_vec = 256, 256, 32
    torch.manual_seed(0)
    a_ref_list, b_ref_list, qa, qb, SFA, SFB, cu_seqlens_k = create_blockscaled_varlen_k_operands(
        num_experts, 0, m, n, sf_vec, seqlens_k=seqlens_k, sf_pad_byte=0xFF
    )
    A_op = BlockScaledOperand.from_parts(qa, SFA, "mxfp8")
    B_op = BlockScaledOperand.from_parts(qb.t(), SFB, "mxfp8", quant_dim=-2)
    out = gemm(A_op, B_op, cu_seqlens_k=cu_seqlens_k, tuned=False)
    assert not out.isnan().any(), "NaN leaked from poisoned SF pad into the output"
    for i in range(num_experts):
        ref_i = a_ref_list[i] @ b_ref_list[i].T
        err = (out[i].float() - ref_i).abs().max().item() / ref_i.abs().max().item()
        assert err < 5e-3, f"poisoned pad seqlens_k={seqlens_k} expert={i} rel_err={err}"
