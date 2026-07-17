# Copyright (c) 2026, Tri Dao.
"""Tests for the MX quantizers (quack/blockscaled/quantize.py): scale modes and
the dim0 ("columnwise") variant used by training linears (dgrad/wgrad).

Pure-PyTorch quantizers — CPU-runnable; the GEMM-side orientation tests live in
test_gemm_blockscaled_interface.py.
"""

import pytest
import torch

from quack.blockscaled.quantize import F8E4M3_MAX, to_mx, to_mx_dim0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_to_mx_rceil_never_saturates(dtype):
    """RCEIL picks the smallest power-of-two scale with max_abs/scale <= 448:
    the quantized block max never clips, across many magnitude decades."""
    torch.manual_seed(0)
    x = torch.randn(64, 256, dtype=dtype) * torch.logspace(-6, 6, 64).unsqueeze(1).to(dtype)
    q, s = to_mx(x, 32, scaling_mode="rceil")
    assert q.dtype == torch.float8_e4m3fn and s.dtype == torch.float8_e8m0fnu
    scale = s.to(torch.float32).repeat_interleave(32, -1)
    assert (x.float().abs() / scale).max() <= F8E4M3_MAX
    assert q.to(torch.float32).abs().max() <= F8E4M3_MAX


def test_to_mx_rceil_vs_floor_known_block():
    """Block max 500: FLOOR keeps scale 1 and clips 500 -> 448; RCEIL bumps the
    scale to 2 (500/2 = 250, RN-e4m3 -> 256: no clipping, just rounding)."""
    v = torch.full((1, 32), 500.0)
    v[0, 1:] = 1.0
    qf, sf = to_mx(v, 32, scaling_mode="floor")
    qc, sc = to_mx(v, 32, scaling_mode="rceil")
    assert sf.to(torch.float32).item() == 1.0 and qf.to(torch.float32).max().item() == 448.0
    assert sc.to(torch.float32).item() == 2.0 and qc.to(torch.float32).max().item() == 256.0


@pytest.mark.parametrize("scaling_mode", ["rceil", "floor"])
def test_to_mx_edge_cases(scaling_mode):
    """Zero, denormal, inf, NaN blocks produce sane biased scales."""
    e = torch.tensor(
        [[0.0] * 32, [1e-40] * 32, [float("inf")] * 32, [float("nan")] * 32],
        dtype=torch.float32,
    )
    q, s = to_mx(e, 32, scaling_mode=scaling_mode)
    biased = s.view(torch.uint8).flatten().tolist()
    assert biased[0] <= 1  # zero block: minimal scale
    assert biased[3] == 255  # NaN: sentinel
    # inf: rceil saturates the exponent to the sentinel; floor lands on
    # 2^(128 - 8) = biased 247 (inherited torchao behavior)
    assert biased[2] == (255 if scaling_mode == "rceil" else 247)
    assert torch.isfinite(q.to(torch.float32)[:2]).all()


@pytest.mark.parametrize("scaling_mode", ["rceil", "floor"])
@pytest.mark.parametrize("shape", [(128, 320), (32, 32), (4096, 96)])
def test_to_mx_dim0_matches_transposed_rowwise(scaling_mode, shape):
    """dim0 quantization must be bit-identical to rowwise on the transpose —
    same values, no transposed hp copy, only the layout differs."""
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    q0, s0 = to_mx_dim0(x, 32, scaling_mode=scaling_mode)
    qt, st = to_mx(x.t().contiguous(), 32, scaling_mode=scaling_mode)
    assert q0.shape == x.shape and s0.shape == (x.shape[0] // 32, x.shape[1])
    assert torch.equal(q0.view(torch.uint8), qt.t().contiguous().view(torch.uint8))
    assert torch.equal(s0.view(torch.uint8), st.t().contiguous().view(torch.uint8))
