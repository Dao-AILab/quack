# Copyright (c) 2026, Tri Dao.
"""SM120 block-scaled GEMM under forced SM120 configs: tile shapes and pingpong
scheduling for the warp-level block-scaled MMA, for same-dtype mxfp8, mixed
fp4 x fp8 and N-major fp8 B, checked against the dequantized reference.
Formats, layouts, epilogues, varlen and split-K are the same on every
blockscaled GPU and live in test_gemm_blockscaled.py (which runs here
too), test_gemm_quant_out.py, and test_gemm_sm120_fp8.py for plain fp8.

Kernel paths exercised on SM120: independent A/B dtypes across fp8 (e4m3/e5m2) / fp6 (e2m3/e3m2,
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

from quack.blockscaled.operand import BlockScaledFormat, BlockScaledOperand
from quack.blockscaled.utils import blockscaled_quantize
from quack.cute_dsl_utils import get_device_capacity
from quack.gemm_config import GemmConfig
from quack.gemm_interface import (
    _prep_blockscaled,
    _sf_batch_canonicalize,
    _unpack_operand,
    gemm_blockscaled_ref,
    gemm_tuned,
)

_ARCH = get_device_capacity(torch.device("cuda"))[0] if torch.cuda.is_available() else 0
# get_device_capacity honors the QUACK_ARCH proxy override, but the blockscaled
# mma kinds exist on sm_120/121 silicon only and ptxas always targets the
# physical GPU — so the H100 QUACK_ARCH=120 CI legs must skip these.
_PHYSICAL_ARCH = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
requires_sm120 = pytest.mark.skipif(
    _ARCH != 12 or _PHYSICAL_ARCH != 12,
    reason="SM120 blockscaled warp-MMA path (needs sm_120/121 silicon, no proxy)",
)


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


def _gemm_with_config(A, B, config=None):
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


def _mixed_operand(fmt, rows, k, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    return BlockScaledOperand.quantize(x, fmt)


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


def _n_major_b(W):
    """(k, n) row-major view of an (n, k) quantized operand — the interface's
    .mT relabel makes it n-major B. The scale tensor is layout-independent."""
    qb_kn = W.qdata.t().contiguous()  # (k, n) with N contiguous
    return BlockScaledOperand.from_parts(qb_kn, W.scale, W.format, quant_dim=-2)


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
@pytest.mark.parametrize("fmt", ["mxfp8_e4m3", "mxfp4", "nvfp4"])
def test_sm120_cluster_rejected(fmt):
    """Clusters other than 1x1 are rejected up front. Before, the multicast loads compiled
    to driver syscalls: correct but pointless for fp8, and a device-side assertion for
    fp4 (emulate_tma: INVALID_DESCRIPTOR_FORMAT)."""
    from quack.gemm import gemm as gemm_raw

    m = n = k = 512
    A = _mixed_operand(fmt, m, k, seed=0)
    W = _mixed_operand(fmt, n, k, seed=1)
    D = torch.empty(1, m, n, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"cluster shape \(1, 1, 1\)"):
        gemm_raw(
            A.qdata[None],
            W.qdata[None],
            D,
            None,
            None,
            tile_M=128,
            tile_N=128,
            cluster_M=2,
            cluster_N=1,
            SFA=A.scale[None],
            SFB=W.scale[None],
            bs_format_a=fmt,
            bs_format_b=fmt,
        )
