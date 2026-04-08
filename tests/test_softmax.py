# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import pytest
import torch
import torch.nn.functional as F


from quack.softmax import softmax


torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024

TOLERANCES = {
    torch.bfloat16: (1e-2, 1e-2),
    torch.float16: (1e-3, 1e-3),
    torch.float32: (1e-4, 1e-4),
}


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
# @pytest.mark.parametrize("input_dtype", [torch.float32])
@pytest.mark.parametrize(
    "N",
    [192, 256, 512, 760, 1024, 1128, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144],
    # [32768]
)
@pytest.mark.parametrize("M", [1, 37, 199])
@pytest.mark.parametrize("use_compile", [False, True])
# @pytest.mark.parametrize("M", [1])
def test_softmax(M, N, input_dtype, use_compile):
    """Test Softmax forward and backward passes against reference implementation."""
    # SM12x (consumer Blackwell) has 99 KB SMEM — skip dims that exceed capacity
    major, _ = torch.cuda.get_device_capability()
    if major == 12:
        if input_dtype == torch.float32 and N > 4096:
            pytest.skip("SM12x: 99 KB SMEM limit exceeded for fp32")
        if input_dtype != torch.float32 and N > 8192:
            pytest.skip("SM12x: 99 KB SMEM limit exceeded for fp16/bf16")
    device = "cuda"
    atol, rtol = TOLERANCES[input_dtype]
    function = torch.compile(softmax, fullgraph=True) if use_compile else softmax

    torch.random.manual_seed(0)
    # Create input tensors (scale down to avoid overflow in softmax)
    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    x_ref = x.detach().clone().requires_grad_(True)

    out = function(x)
    out_ref = F.softmax(x_ref, dim=-1)

    assert out.shape == x.shape
    assert out.dtype == input_dtype
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    sums = torch.sum(out, dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)
    assert (out >= 0).all()
    assert (out <= 1).all()

    dy = torch.randn_like(out)
    torch.cuda.synchronize()  # without sync, torch.autograd gets wrong results
    (dx,) = torch.autograd.grad(out, x, grad_outputs=dy)
    (dx_ref,) = torch.autograd.grad(out_ref, x_ref, grad_outputs=dy)
    assert dx.shape == dy.shape
    assert dx.dtype == input_dtype
    torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_compile", [False, True])
def test_softmax_extreme_values(input_dtype, use_compile):
    """Test Softmax with extreme input values."""
    device = "cuda"
    M, N = 16, 1024
    function = torch.compile(softmax, fullgraph=True) if use_compile else softmax

    x_large = torch.full((M, N), 10.0, device=device, dtype=input_dtype)
    out_large = function(x_large)
    # Should be uniform since all values are the same
    expected = torch.full_like(out_large, 1.0 / N)
    torch.testing.assert_close(out_large, expected, atol=1e-3, rtol=1e-3)
    # Test with large negative values
    x_small = torch.full((M, N), -10.0, device=device, dtype=input_dtype)
    out_small = function(x_small)
    # Should also be uniform
    torch.testing.assert_close(out_small, expected, atol=1e-3, rtol=1e-3)
    # Test with mixed extreme values
    x_mixed = torch.zeros((M, N), device=device, dtype=input_dtype)
    x_mixed[:, 0] = 10.0
    x_mixed[:, 1:] = -10.0
    out_mixed = function(x_mixed)
    # First column should be close to 1, rest close to 0
    assert (out_mixed[:, 0] > 0.99).all()
    assert (out_mixed[:, 1:] < 0.01).all()


@pytest.mark.parametrize("use_compile", [False, True])
def test_softmax_numerical_stability(use_compile):
    """Test that softmax is numerically stable."""
    device = "cuda"
    M, N = 8, 512
    function = torch.compile(softmax, fullgraph=True) if use_compile else softmax

    x = torch.randn(M, N, device=device, dtype=torch.float32)
    x_shifted = x + 100.0
    out = function(x)
    out_shifted = function(x_shifted)
    torch.testing.assert_close(out, out_shifted, atol=1e-6, rtol=1e-6)
