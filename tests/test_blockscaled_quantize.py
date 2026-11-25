# Copyright (c) 2026, Tri Dao.
"""Tests for the MX quantizers (quack/blockscaled/quantize.py): scale modes and
the dim0 ("columnwise") variant used by training linears (dgrad/wgrad).

Pure-PyTorch quantizers — CPU-runnable; the GEMM-side orientation tests live in
test_gemm_blockscaled_interface.py.
"""

import pytest
import torch

from quack.blockscaled.quantize import F8E4M3_MAX, to_mx, to_mx_2d, to_mx_dim0


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


def test_to_mx_2d_row_replication_and_block_numerics():
    """to_mx_2d = to_mx rceil applied to each flattened 32x32 block, with the
    per-block scale row-replicated into the standard (M, K // 32) rowwise
    tensor — so blocked-layout packing and every kernel path apply unchanged."""
    torch.manual_seed(0)
    x = torch.randn(128, 96, dtype=torch.bfloat16)
    x = x * torch.logspace(-3, 3, 128).unsqueeze(1).to(torch.bfloat16)
    qdata, scale = to_mx_2d(x)
    assert qdata.dtype == torch.float8_e4m3fn and qdata.shape == (128, 96)
    assert scale.dtype == torch.float8_e8m0fnu and scale.shape == (128, 3)
    su8 = scale.view(torch.uint8).view(4, 32, 3)
    assert torch.equal(su8, su8[:, :1, :].expand_as(su8))  # row-replicated per block
    # Exact to_mx numerics on each flattened (32, 32) block.
    blocks = x.view(4, 32, 3, 32).permute(0, 2, 1, 3).reshape(12, 1024)
    q_ref, s_ref = to_mx(blocks, 1024)
    q_2d_as_blocks = qdata.view(4, 32, 3, 32).permute(0, 2, 1, 3).reshape(12, 1024)
    assert torch.equal(q_2d_as_blocks.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(su8[:, 0, :].reshape(-1), s_ref.view(torch.uint8).reshape(-1))
    # 2D-only and block-divisibility contracts.
    with pytest.raises(AssertionError):
        to_mx_2d(x.unsqueeze(0))
    with pytest.raises(AssertionError):
        to_mx_2d(x[:48])
