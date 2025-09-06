# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import pytest
import torch
import torch.nn.functional as F

import cutlass

from quack.softmax import softmax


torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
# @pytest.mark.parametrize("input_dtype", [torch.float32])
@pytest.mark.parametrize(
    "N",
    [192, 256, 512, 760, 1024, 1128, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    # [32768]
)
@pytest.mark.parametrize("M", [1, 37, 199])
@pytest.mark.parametrize("function", [softmax, torch.compile(softmax, fullgraph=True)])
# @pytest.mark.parametrize("M", [1])
def test_softmax(M, N, input_dtype, function):
    """Test Softmax forward and backward passes against reference implementation."""
    device = "cuda"
    # Set tolerance based on dtype
    if input_dtype == torch.bfloat16:
        atol = 1e-2
        rtol = 1e-2
    elif input_dtype == torch.float16:
        atol = 1e-3
        rtol = 1e-3
    else:
        atol = 1e-4
        rtol = 1e-4

    torch.random.manual_seed(0)
    # Create input tensors (scale down to avoid overflow in softmax)
    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    x_ref = x.detach().clone().requires_grad_(True)

    # Forward pass
    out = function(x)
    out_ref = F.softmax(x_ref, dim=-1)

    # Check output shape and dtype
    assert out.shape == x.shape
    assert out.dtype == input_dtype
    # Check accuracy
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    # Check softmax properties
    # Sum along last dimension should be 1
    sums = torch.sum(out, dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)
    # All values should be positive
    assert (out >= 0).all()
    # All values should be <= 1
    assert (out <= 1).all()

    # Test backward pass
    dy = torch.randn_like(out)
    torch.cuda.synchronize()  # without sync, torch.autograd gets wrong results
    dx_ref, = torch.autograd.grad(out_ref, x_ref, grad_outputs=dy)
    # Call our implementation later, otherwise getting CUDA_ERROR_INVALID_CONTEXT
    dx, = torch.autograd.grad(out, x, grad_outputs=dy)
    assert dx.shape == dy.shape
    assert dx.dtype == input_dtype
    torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("function", [softmax, torch.compile(softmax, fullgraph=True)])
def test_softmax_extreme_values(input_dtype, function):
    """Test Softmax with extreme input values."""
    device = "cuda"
    M, N = 16, 1024
    # Test with large positive values
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
    x_mixed[:, 0] = 10.0  # One large value per row
    x_mixed[:, 1:] = -10.0  # Rest are small
    out_mixed = function(x_mixed)
    # First column should be close to 1, rest close to 0
    assert (out_mixed[:, 0] > 0.99).all()
    assert (out_mixed[:, 1:] < 0.01).all()


@pytest.mark.parametrize("function", [softmax, torch.compile(softmax, fullgraph=True)])
def test_softmax_numerical_stability(function):
    """Test that softmax is numerically stable."""
    device = "cuda"
    M, N = 8, 512
    # Create input with a wide range of values
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    # Add large constant to test numerical stability
    x_shifted = x + 100.0
    out = function(x)
    out_shifted = function(x_shifted)
    # Results should be identical (softmax is translation invariant)
    torch.testing.assert_close(out, out_shifted, atol=1e-6, rtol=1e-6)


# def test_softmax_backward_properties():
#     """Test mathematical properties of softmax backward pass."""
#     device = "cuda"
#     M, N = 16, 1024

#     torch.random.manual_seed(42)
#     # Create test inputs
#     x = torch.randn(M, N, device=device, dtype=torch.float32, requires_grad=True)
#     y = F.softmax(x, dim=-1)

#     # Test 1: Gradient of uniform upstream should sum to zero along last dim
#     dy_uniform = torch.ones_like(y)
#     dx = softmax_backward(dy_uniform, y.detach())

#     # Sum along last dimension should be approximately zero
#     row_sums = torch.sum(dx, dim=-1)
#     torch.testing.assert_close(row_sums, torch.zeros_like(row_sums), atol=1e-6, rtol=1e-6)

#     # Test 2: Gradient should be zero when upstream gradient is proportional to y
#     # (since softmax(x) ∝ y means dy ∝ y, so dy - dot*1 = 0)
#     for scale in [0.5, 1.0, 2.0]:
#         dy_prop = scale * y
#         dx_prop = softmax_backward(dy_prop, y.detach())
#         torch.testing.assert_close(dx_prop, torch.zeros_like(dx_prop), atol=1e-6, rtol=1e-6)


# def test_softmax_backward_edge_cases():
#     """Test softmax backward with edge cases."""
#     device = "cuda"

#     # Test with one-hot softmax output (peaked distribution)
#     M, N = 8, 512
#     y = torch.zeros(M, N, device=device, dtype=torch.float32)
#     y[:, 0] = 1.0  # One-hot at first position

#     dy = torch.randn_like(y)
#     dx = softmax_backward(dy, y)

#     # Check that gradients exist and are finite
#     assert torch.isfinite(dx).all()

#     # Test with uniform softmax output
#     y_uniform = torch.full((M, N), 1.0/N, device=device, dtype=torch.float32)
#     dy_uniform = torch.randn_like(y_uniform)
#     dx_uniform = softmax_backward(dy_uniform, y_uniform)

#     # Check that gradients exist and are finite
#     assert torch.isfinite(dx_uniform).all()

#     # Row sums should be zero (property of softmax jacobian)
#     row_sums = torch.sum(dx_uniform, dim=-1)
#     torch.testing.assert_close(row_sums, torch.zeros_like(row_sums), atol=1e-6, rtol=1e-6)
