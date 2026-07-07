"""Debug/regression test for the SM90 128x128 blockwise weight quantizer.

Unlike the MoE forward test (whose reference dequantizes with this same function, so it
can't catch a bad scale), this checks reconstruction directly: q * scale must recover w.
"""

import pytest
import torch

from quack.gemm_blockscaled_sm90 import quantize_weight_sm90

_IS_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 9
_skip = pytest.mark.skipif(not _IS_SM90, reason="SM90-only weight quantizer")


def _dequant(q: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    # Broadcast each 128x128 block's scale back over the block.
    return q.float() * sc.repeat_interleave(128, -2).repeat_interleave(128, -1)


@_skip
@pytest.mark.parametrize(
    "shape",
    [(8, 1024, 512), (8, 512, 512), (16, 2048, 768), (8, 3072, 1536), (1, 128, 128)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_weight_quant_reconstruction(shape, dtype):
    torch.manual_seed(0)
    E, N, K = shape
    w = 0.02 * torch.randn(E, N, K, device="cuda", dtype=dtype)

    q, sc = quantize_weight_sm90(w)

    # shapes / dtypes
    assert q.shape == (E, N, K) and q.dtype == torch.float8_e4m3fn
    assert sc.shape == (E, N // 128, K // 128) and sc.dtype == torch.float32

    eff = _dequant(q, sc)

    # (1) magnitude sanity — catches an inverted / off-by-scale bug (the "5850x" failure):
    ratio = eff.abs().mean() / w.float().abs().mean()
    assert 0.5 < ratio < 2.0, f"effective/true magnitude ratio {ratio:.3f} — dequant scale wrong"

    # (2) reconstruction — q * scale must approximate w to within fp8 block error:
    rel = (eff - w.float()).abs().mean() / w.float().abs().mean()
    assert rel < 0.05, f"reconstruction rel error {rel:.4f} too high"

    # (3) scales are powers of two (we keep the & 0xFF800000 mantissa mask):
    mantissa = sc.view(torch.int32) & 0x007FFFFF
    assert (mantissa == 0).all(), "stored scale is not a power of two"


@_skip
def test_weight_quant_zero_block():
    # An all-zero 128x128 block must not produce NaN/Inf (amax==0 guard).
    w = torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16)
    q, sc = quantize_weight_sm90(w)
    assert torch.isfinite(sc).all()
    assert (q.float() == 0).all()
