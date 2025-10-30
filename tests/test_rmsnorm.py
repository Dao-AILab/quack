# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import pytest
import torch

from quack.rmsnorm import rmsnorm, rmsnorm_ref, _rmsnorm_fwd, rmsnorm_fwd, rmsnorm_bwd

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
# @pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float16, torch.float32, None])
# @pytest.mark.parametrize("weight_dtype", [torch.bfloat16])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
# @pytest.mark.parametrize("input_dtype", [torch.float32])
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
)
@pytest.mark.parametrize("M", [1, 37, 199, 8 * 1024])
# @pytest.mark.parametrize("M", [1])
@pytest.mark.parametrize("use_compile", [False, True])
# @pytest.mark.parametrize("use_compile", [False])
def test_rmsnorm_forward_backward(M, N, input_dtype, weight_dtype, eps, use_compile):
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
    if weight_dtype is not None:
        weight = torch.randn(N, device=device, dtype=weight_dtype, requires_grad=True)
    else:
        weight = None
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_() if weight is not None else None
    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
    out = function(x, weight, eps=eps)
    out_ref = rmsnorm_ref(x_ref, weight_ref, eps=eps)

    assert out.shape == x.shape
    assert out.dtype == input_dtype
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=1e-3)
    # Backward pass
    if N > 128 * 1024 and input_dtype == torch.float32:
        # Skip backward pass for due to not enough smem
        return
    grad_out = torch.randn_like(out)
    torch.cuda.synchronize()
    out_ref.backward(grad_out)
    out.backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=1e-3)
    if weight_dtype is not None:
        if weight_dtype == torch.float32:
            weight_atol = 1e-4
        else:
            weight_atol = 2 * (weight_ref.grad + 0.3 - 0.3 - weight_ref.grad).abs().max()
        torch.testing.assert_close(weight.grad, weight_ref.grad, atol=weight_atol, rtol=1e-3)


@pytest.mark.parametrize("use_compile", [False, True])
# @pytest.mark.parametrize("use_compile", [False])
def test_rmsnorm_strided_tensor(use_compile):
    """Test RMSNorm with strided tensor input where shape is (8, 4096, 512) and stride is (sth, 576, 1)."""
    device = "cuda"
    dtype = torch.bfloat16
    atol = 1e-1
    eps = 1e-5
    # Create a larger tensor with 576 features
    full_tensor = torch.randn(8, 4096, 576, device=device, dtype=dtype)
    # Take a slice of the top 512 dimensions - this creates a strided view
    x = full_tensor[:, :, :512].detach().requires_grad_()
    # Create weight tensor
    weight = torch.randn(512, device=device, dtype=torch.float32, requires_grad=True)
    # Reference implementation
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
    out = function(x, weight, eps=eps)
    out_ref = rmsnorm_ref(x_ref, weight_ref, eps=eps)
    assert out.shape == x.shape
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=1e-3)
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
@pytest.mark.parametrize("use_compile", [False, True])
def test_rmsnorm_large_tensor(M, N, input_dtype, eps, use_compile):
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
    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
    out = function(x, weight, eps=eps)
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

    # with pytest.raises(AssertionError, match="Tensors must be on CUDA device"):
    # With torch.library custom op, this now fails with NotImplementedError
    with pytest.raises(NotImplementedError):
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
    out1 = rmsnorm_fwd(x1, weight1, eps=eps)
    assert len(_rmsnorm_fwd.compile_cache) == 1

    # Same shape should reuse cache
    x2 = torch.randn(M, N, device=device, dtype=torch.float16)
    weight2 = torch.randn(N, device=device, dtype=torch.float32)
    out2 = rmsnorm_fwd(x2, weight2, eps=eps)
    assert len(_rmsnorm_fwd.compile_cache) == 1

    # Changing batch size should reuse cache
    x2 = torch.randn(M * 2, N, device=device, dtype=torch.float16)
    weight2 = torch.randn(N, device=device, dtype=torch.float32)
    out2 = rmsnorm_fwd(x2, weight2, eps=eps)
    assert len(_rmsnorm_fwd.compile_cache) == 1

    # Different shape should create new cache entry
    x3 = torch.randn(M, N * 2, device=device, dtype=torch.float16)
    weight3 = torch.randn(N * 2, device=device, dtype=torch.float32)
    out3 = rmsnorm_fwd(x3, weight3, eps=eps)
    assert len(_rmsnorm_fwd.compile_cache) == 2

    # Different dtype should create new cache entry
    x4 = torch.randn(M, N, device=device, dtype=torch.float32)
    weight4 = torch.randn(N, device=device, dtype=torch.float32)
    out4 = rmsnorm_fwd(x4, weight4, eps=eps)
    assert len(_rmsnorm_fwd.compile_cache) == 3


@pytest.mark.parametrize("use_compile", [False, True])
def test_rmsnorm_with_bias(use_compile):
    """Test RMSNorm with bias parameter - both forward and backward."""
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6
    input_dtype = torch.float16
    weight_dtype = torch.float32
    bias_dtype = torch.float32

    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
    weight = torch.randn(N, device=device, dtype=weight_dtype, requires_grad=True)
    bias = torch.randn(N, device=device, dtype=bias_dtype, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_()

    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
    out = function(x, weight, bias=bias, eps=eps)
    out_ref = rmsnorm_ref(x_ref, weight_ref, bias=bias_ref, eps=eps)

    assert out.shape == x.shape
    assert out.dtype == input_dtype
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-3)

    grad_out = torch.randn_like(out)
    torch.cuda.synchronize()
    out_ref.backward(grad_out)
    out.backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(bias.grad, bias_ref.grad, atol=1e-4, rtol=1e-3)

@pytest.mark.parametrize("backward_type", ["only_output", "both"])
@pytest.mark.parametrize("use_compile", [False, True])
def test_rmsnorm_with_residual(backward_type, use_compile):
    """Test RMSNorm with residual connection - both forward and backward."""
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6
    input_dtype = torch.float16
    weight_dtype = torch.float32

    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
    weight = torch.randn(N, device=device, dtype=weight_dtype, requires_grad=True)
    residual = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    residual_ref = residual.detach().clone().requires_grad_()

    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
    out, residual_out = function(x, weight, residual=residual, eps=eps, prenorm=True)
    out_ref, residual_out_ref = rmsnorm_ref(x_ref, weight_ref, residual=residual_ref, eps=eps)

    assert out.shape == x.shape
    assert out.dtype == input_dtype
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(residual_out, residual_out_ref, atol=1e-2, rtol=1e-3)

    grad_out = torch.randn_like(out)
    torch.cuda.synchronize()
    if backward_type == "only_residual":    
        residual_out_ref.backward(grad_out)
        residual_out.backward(grad_out)
    elif backward_type == "only_output":
        out_ref.backward(grad_out)
        out.backward(grad_out)
    else:
        (out_ref + residual_out_ref).backward(grad_out)
        (out + residual_out).backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(residual.grad, residual_ref.grad, atol=1e-2, rtol=1e-3)


def test_amp_bf16_training():
    """
    Test amp bf16 training works
    """
    device = "cuda"
    M, N = 32768, 1024
    eps = 1e-6

    dy = torch.randn(M, N, device=device, dtype=torch.bfloat16, requires_grad=True)
    x = torch.randn(M, N, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)
    rstd = torch.randn(M, device=device, dtype=torch.float32, requires_grad=True)

    dx, dw, _, _ = rmsnorm_bwd(x, weight, dy, rstd)

    assert dx is not None
    assert dw is not None


@pytest.mark.parametrize("use_compile", [False, True])
def test_rmsnorm_prenorm_false(use_compile):
    """Test RMSNorm with prenorm=False - residual input but no residual output."""
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6
    input_dtype = torch.float16
    weight_dtype = torch.float32

    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
    weight = torch.randn(N, device=device, dtype=weight_dtype, requires_grad=True)
    residual = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    residual_ref = residual.detach().clone().requires_grad_()

    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
    out = function(x, weight, residual=residual, eps=eps, prenorm=False)

    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    assert out.dtype == input_dtype

    out_ref, residual_out_ref = rmsnorm_ref(x_ref, weight_ref, residual=residual_ref, eps=eps)

    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-3)

    grad_out = torch.randn_like(out)
    torch.cuda.synchronize()
    out_ref.backward(grad_out)
    out.backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(residual.grad, residual_ref.grad, atol=1e-2, rtol=1e-3)
