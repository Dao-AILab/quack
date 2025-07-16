# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import pytest
import torch

from quack.rmsnorm import rmsnorm, rmsnorm_ref, rstd_ref, _rmsnorm_fwd

@pytest.mark.parametrize("eps", [1e-5, 1e-6])
# @pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
# @pytest.mark.parametrize("input_dtype", [torch.float16])
@pytest.mark.parametrize(
    "N",
    [
        192,
        256,
        512,
        760,
        1024,
        1128,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
    ],
    # [262144]
)
@pytest.mark.parametrize("M", [1, 37, 199, 8 * 1024])
# @pytest.mark.parametrize("M", [1])
def test_rmsnorm_forward_backward(M, N, input_dtype, eps):
    """Test RMSNorm forward pass against reference implementation."""
    if N >= 256 * 1024 and input_dtype == torch.float32 and M >= 8 * 1024:
        pytest.skip("Skipping large tensor test for float32 to avoid OOM")
    device = "cuda"
    # Set tolerance based on dtype
    if input_dtype == torch.bfloat16:
        atol = 1e-1
    elif input_dtype == torch.float16:
        atol = 1e-2
    else:
        atol = 1e-4
    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
    weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    out = rmsnorm(x, weight, eps=eps)
    out_ref = rmsnorm_ref(x_ref, weight_ref, eps=eps)
    # rstd_ref_val = rstd_ref(x_ref, eps=eps)
    assert out.shape == x.shape
    assert out.dtype == input_dtype
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=1e-3)
    # torch.testing.assert_close(rstd, rstd_ref_val, atol=atol, rtol=1e-3)
    # Backward pass
    if N > 128 * 1024 and input_dtype == torch.float32:
        # Skip backward pass for due to not enough smem
        return
    grad_out = torch.randn_like(out)
    torch.cuda.synchronize()
    out_ref.backward(grad_out)
    out.backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=1e-3)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=atol, rtol=1e-3)


@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "N",
    [131072, 262144],
    # [262144]
)
@pytest.mark.parametrize("M", [32 * 1024])
def test_rmsnorm_large_tensor(M, N, input_dtype, eps):
    """Test RMSNorm forward pass against reference implementation."""
    device = "cuda"
    # Set tolerance based on dtype
    if input_dtype == torch.bfloat16:
        atol = 1e-1
    elif input_dtype == torch.float16:
        atol = 1e-2
    else:
        atol = 1e-4
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=False)
    weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=False)
    out = rmsnorm(x, weight, eps=eps)
    # Need to compile, otherwise it OOMs
    rmsnorm_compiled = torch.compile(rmsnorm_ref)
    # Run once with smaller input to avoid OOMs
    rmsnorm_compiled(x[:32], weight, eps=eps)
    out_ref = rmsnorm_compiled(x, weight, eps=eps)
    # Need to chunk, otherwise it OOMs
    assert all(
        (out_c - out_ref_c).abs().max() < atol
        for out_c, out_ref_c in zip(out.chunk(16), out_ref.chunk(16))
    )


@pytest.mark.parametrize("return_rstd", [True, False])
def test_rmsnorm_return_rstd_option(return_rstd):
    """Test that return_rstd option works correctly."""
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6

    x = torch.randn(M, N, device=device, dtype=torch.float16)
    weight = torch.randn(N, device=device, dtype=torch.float32)

    if return_rstd:
        out, rstd = _rmsnorm_fwd(x, weight, eps=eps, return_rstd=True)
        assert out.shape == (M, N)
        assert rstd.shape == (M,)
        assert rstd.dtype == torch.float32
    else:
        out = _rmsnorm_fwd(x, weight, eps=eps, return_rstd=False)
        assert out.shape == (M, N)
        assert isinstance(out, torch.Tensor)


def test_rmsnorm_input_validation():
    """Test input validation and error handling."""
    device = "cuda"

    # Test 3D input (should now work since rmsnorm was updated to accept 3D inputs)
    x_3d = torch.randn(2, 32, 1024, device=device, dtype=torch.float16)
    weight = torch.randn(1024, device=device, dtype=torch.float32)

    # This should not raise an exception now
    out = rmsnorm(x_3d, weight)
    # Verify output shape matches input shape
    assert out.shape == x_3d.shape
    # Verify output dtype matches input dtype
    assert out.dtype == x_3d.dtype

    # Test weight dimension mismatch
    x = torch.randn(32, 1024, device=device, dtype=torch.float16)
    weight_wrong = torch.randn(512, device=device, dtype=torch.float32)

    with pytest.raises(AssertionError, match="Last dimension of input must match weight dimension"):
        rmsnorm(x, weight_wrong)

    # Test CPU tensors (should fail)
    x_cpu = torch.randn(32, 1024, dtype=torch.float16)
    weight_cpu = torch.randn(1024, dtype=torch.float32)

    with pytest.raises(AssertionError, match="Tensors must be on CUDA device"):
        rmsnorm(x_cpu, weight_cpu)

    # Test unsupported dtype
    x = torch.randn(32, 1024, device=device, dtype=torch.float64)
    weight = torch.randn(1024, device=device, dtype=torch.float32)

    with pytest.raises(AssertionError, match="Unsupported dtype"):
        rmsnorm(x, weight)

    # Test wrong weight dtype
    x = torch.randn(32, 1024, device=device, dtype=torch.float16)
    weight_wrong_dtype = torch.randn(1024, device=device, dtype=torch.float64)

    with pytest.raises(AssertionError, match="Weight must be float32, float16 or bfloat16"):
        rmsnorm(x, weight_wrong_dtype)


def test_rmsnorm_bf16_weights():
    """Test that bfloat16 weights work correctly with rmsnorm."""
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6

    # Test with bfloat16 input and weights
    x = torch.randn(M, N, device=device, dtype=torch.bfloat16)
    weight_bf16 = torch.randn(N, device=device, dtype=torch.bfloat16)

    # Run rmsnorm with bfloat16 weights
    out_bf16 = rmsnorm(x, weight_bf16, eps=eps)

    # Verify output shape and dtype
    assert out_bf16.shape == x.shape
    assert out_bf16.dtype == torch.bfloat16

    # Convert to float32 for reference comparison
    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight_bf16.to(torch.float32)

    # Run reference implementation with float32
    out_ref = rmsnorm_ref(x_fp32, weight_fp32, eps=eps).to(torch.bfloat16)

    # Verify output values match reference implementation
    torch.testing.assert_close(out_bf16, out_ref, atol=1e-1, rtol=1e-2)


def test_rmsnorm_bf16_weights_backward():
    """Test that bfloat16 weights work correctly with rmsnorm backward pass."""
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6
    atol = 1e-1  # Higher tolerance for bfloat16

    # Create tensors with gradients
    x = torch.randn(M, N, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight_bf16 = torch.randn(N, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Create reference tensors with float32 weights for comparison
    x_ref = x.detach().clone().requires_grad_()
    weight_fp32 = weight_bf16.to(torch.float32).detach().requires_grad_()

    # Forward pass
    out_bf16 = rmsnorm(x, weight_bf16, eps=eps)
    out_ref = rmsnorm(x_ref, weight_fp32, eps=eps)

    # Create gradient for backward pass
    grad_out = torch.randn_like(out_bf16)
    grad_out_ref = grad_out.clone()

    # Backward pass
    torch.cuda.synchronize()
    out_bf16.backward(grad_out)
    out_ref.backward(grad_out_ref)

    # Verify gradients
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=1e-2)
    torch.testing.assert_close(
        weight_bf16.grad, weight_fp32.grad.to(torch.bfloat16), atol=atol, rtol=1e-2
    )

    # Test with mixed precision: bfloat16 input and float32 weights
    x = torch.randn(M, N, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight_fp32 = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)

    # Forward pass
    out_mixed = rmsnorm(x, weight_fp32, eps=eps)

    # Create gradient for backward pass
    grad_out = torch.randn_like(out_mixed)

    # Backward pass
    torch.cuda.synchronize()
    out_mixed.backward(grad_out)

    # Just verify that backward pass completes without errors
    assert x.grad is not None
    assert weight_fp32.grad is not None


def test_rmsnorm_fp16_weights():
    """Test that float16 weights work correctly with rmsnorm."""
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6

    # Test with float16 input and weights
    x = torch.randn(M, N, device=device, dtype=torch.float16)
    weight_fp16 = torch.randn(N, device=device, dtype=torch.float16)

    # Run rmsnorm with float16 weights
    out_fp16 = rmsnorm(x, weight_fp16, eps=eps)

    # Verify output shape and dtype
    assert out_fp16.shape == x.shape
    assert out_fp16.dtype == torch.float16

    # Convert to float32 for reference comparison
    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight_fp16.to(torch.float32)

    # Run reference implementation with float32
    out_ref = rmsnorm_ref(x_fp32, weight_fp32, eps=eps).to(torch.float16)

    # Verify output values match reference implementation
    torch.testing.assert_close(out_fp16, out_ref, atol=1e-2, rtol=1e-2)


def test_rmsnorm_fp16_weights_backward():
    """Test that float16 weights work correctly with rmsnorm backward pass."""
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6
    atol = 1e-2  # Tolerance for float16

    # Create tensors with gradients
    x = torch.randn(M, N, device=device, dtype=torch.float16, requires_grad=True)
    weight_fp16 = torch.randn(N, device=device, dtype=torch.float16, requires_grad=True)

    # Create reference tensors with float32 weights for comparison
    x_ref = x.detach().clone().requires_grad_()
    weight_fp32 = weight_fp16.to(torch.float32).detach().requires_grad_()

    # Forward pass
    out_fp16 = rmsnorm(x, weight_fp16, eps=eps)
    out_ref = rmsnorm(x_ref, weight_fp32, eps=eps)

    # Create gradient for backward pass
    grad_out = torch.randn_like(out_fp16)
    grad_out_ref = grad_out.clone()

    # Backward pass
    torch.cuda.synchronize()
    out_fp16.backward(grad_out)
    out_ref.backward(grad_out_ref)

    # Verify gradients
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=1e-2)
    torch.testing.assert_close(
        weight_fp16.grad, weight_fp32.grad.to(torch.float16), atol=atol, rtol=1e-2
    )

    # Test with mixed precision: float16 input and float32 weights
    x = torch.randn(M, N, device=device, dtype=torch.float16, requires_grad=True)
    weight_fp32 = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)

    # Forward pass
    out_mixed = rmsnorm(x, weight_fp32, eps=eps)

    # Create gradient for backward pass
    grad_out = torch.randn_like(out_mixed)

    # Backward pass
    torch.cuda.synchronize()
    out_mixed.backward(grad_out)

    # Just verify that backward pass completes without errors
    assert x.grad is not None
    assert weight_fp32.grad is not None


def test_rmsnorm_compile_cache():
    """Test that compile cache works correctly for repeated calls."""
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6

    # Clear cache
    _rmsnorm_fwd.compile_cache.clear()
    assert len(_rmsnorm_fwd.compile_cache) == 0

    x1 = torch.randn(M, N, device=device, dtype=torch.float16)
    weight1 = torch.randn(N, device=device, dtype=torch.float32)

    # First call should compile
    out1 = _rmsnorm_fwd(x1, weight1, eps=eps)
    assert len(_rmsnorm_fwd.compile_cache) == 1

    # Same shape should reuse cache
    x2 = torch.randn(M, N, device=device, dtype=torch.float16)
    weight2 = torch.randn(N, device=device, dtype=torch.float32)
    out2 = _rmsnorm_fwd(x2, weight2, eps=eps)
    assert len(_rmsnorm_fwd.compile_cache) == 1

    # Different shape should create new cache entry
    x3 = torch.randn(M, N * 2, device=device, dtype=torch.float16)
    weight3 = torch.randn(N * 2, device=device, dtype=torch.float32)
    out3 = _rmsnorm_fwd(x3, weight3, eps=eps)
    assert len(_rmsnorm_fwd.compile_cache) == 2

    # Different dtype should create new cache entry
    x4 = torch.randn(M, N, device=device, dtype=torch.float32)
    weight4 = torch.randn(N, device=device, dtype=torch.float32)
    out4 = _rmsnorm_fwd(x4, weight4, eps=eps)
    assert len(_rmsnorm_fwd.compile_cache) == 3
