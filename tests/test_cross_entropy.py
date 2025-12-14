# Copyright (c) 2025, Tri Dao.

import pytest
import torch
import torch.nn.functional as F

from quack.cross_entropy import cross_entropy_fwd, cross_entropy

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
# @pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "N",
    [
        192,
        256,
        512,
        668,
        760,
        1024,
        1128,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        128256,
        131072,
        256128,
        262144,
    ],
    # [32768]
)
@pytest.mark.parametrize("M", [1, 77, 289])
# @pytest.mark.parametrize("M", [1])
@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy(M, N, input_dtype, use_compile):
    """Test Cross Entropy forward pass against reference implementation."""
    device = "cuda"
    atol, rtol = 5e-5, 1e-5
    torch.random.manual_seed(0)
    # Create input tensors (scale down to avoid overflow)
    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    x_ref = x.detach().clone().requires_grad_()
    target_ref = target.detach().clone()
    # Forward pass
    function = torch.compile(cross_entropy, fullgraph=True) if use_compile else cross_entropy
    loss = function(x, target, reduction="none")
    loss_ref = F.cross_entropy(x_ref.float(), target_ref, reduction="none")
    # Check output shape and dtype
    assert loss.shape == (M,)
    assert loss.dtype == torch.float32
    # Check accuracy
    torch.testing.assert_close(loss, loss_ref, atol=atol, rtol=rtol)
    # Check cross entropy properties
    # All values should be non-negative
    assert (loss >= 0).all()
    # Check that loss is reasonable (not inf or nan)
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()
    # Test backward pass
    dloss = torch.randn_like(loss)
    torch.cuda.synchronize()
    (dx,) = torch.autograd.grad(loss, x, grad_outputs=dloss)
    (dx_ref,) = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss)
    assert dx.shape == x.shape
    torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("N", [128256])
@pytest.mark.parametrize("M", [4096])
@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_lse_partial(M, N, input_dtype, use_compile):
    """Test Cross Entropy forward pass against reference implementation."""
    assert N % 128 == 0, "N must be multiple of 128 for lse_partial"
    device = "cuda"
    atol, rtol = 5e-5, 1e-5
    torch.random.manual_seed(0)
    # Create input tensors (scale down to avoid overflow)
    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    with torch.no_grad():
        lse_partial = x.view(M, N // 128, 128).float().logsumexp(dim=-1)
    x_ref = x.detach().clone().requires_grad_()
    target_ref = target.detach().clone()
    # Forward pass
    function = torch.compile(cross_entropy, fullgraph=True) if use_compile else cross_entropy
    loss = function(x, target, lse_partial=lse_partial, reduction="none")
    loss_ref = F.cross_entropy(x_ref.float(), target_ref, reduction="none")
    # Check output shape and dtype
    assert loss.shape == (M,)
    assert loss.dtype == torch.float32
    # Check accuracy
    torch.testing.assert_close(loss, loss_ref, atol=atol, rtol=rtol)
    # Check cross entropy properties
    # All values should be non-negative
    assert (loss >= 0).all()
    # Check that loss is reasonable (not inf or nan)
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()
    # Test backward pass
    dloss = torch.randn_like(loss)
    torch.cuda.synchronize()
    (dx_ref,) = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss)
    (dx,) = torch.autograd.grad(loss, x, grad_outputs=dloss)
    assert dx.shape == x.shape
    torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_extreme_values(input_dtype, use_compile):
    """Test Cross Entropy with extreme input values."""
    device = "cuda"
    M, N = 16, 1024
    function = (
        torch.compile(cross_entropy_fwd, fullgraph=True) if use_compile else cross_entropy_fwd
    )
    # Test with large positive values
    x_large = torch.full((M, N), 10.0, device=device, dtype=input_dtype)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    loss_large = function(x_large, target)
    # Should be around log(N) since all logits are equal
    expected_large = torch.full_like(loss_large, torch.log(torch.tensor(N, dtype=torch.float32)))
    torch.testing.assert_close(loss_large, expected_large, atol=1e-2, rtol=1e-2)
    # Test with large negative values
    x_small = torch.full((M, N), -10.0, device=device, dtype=input_dtype)
    loss_small = function(x_small, target)
    # Should also be around log(N)
    torch.testing.assert_close(loss_small, expected_large, atol=1e-2, rtol=1e-2)
    # Test with one-hot like scenario (one large value, rest small)
    x_onehot = torch.full((M, N), -10.0, device=device, dtype=input_dtype)
    # Set the target class to have large logit
    for i in range(M):
        x_onehot[i, target[i]] = 10.0
    loss_onehot = function(x_onehot, target)
    # Should be close to 0 since target class has highest probability
    assert (loss_onehot < 1.0).all()


@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_numerical_stability(use_compile):
    """Test that cross entropy is numerically stable."""
    device = "cuda"
    M, N = 8, 512
    function = (
        torch.compile(cross_entropy_fwd, fullgraph=True) if use_compile else cross_entropy_fwd
    )
    # Create input with a wide range of values
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    # Add large constant to test numerical stability
    x_shifted = x + 100.0
    loss = function(x, target)
    loss_shifted = function(x_shifted, target)
    # Results should be identical (cross entropy is translation invariant)
    torch.testing.assert_close(loss, loss_shifted, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_edge_targets(use_compile):
    """Test cross entropy with edge case targets."""
    device = "cuda"
    M, N = 16, 1024
    function = (
        torch.compile(cross_entropy_fwd, fullgraph=True) if use_compile else cross_entropy_fwd
    )
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch.float32)
    # Test with target = 0 (first class)
    target_first = torch.zeros(M, device=device, dtype=torch.int64)
    loss_first = function(x, target_first)
    loss_ref_first = F.cross_entropy(x, target_first, reduction="none")
    torch.testing.assert_close(loss_first, loss_ref_first, atol=1e-4, rtol=1e-4)
    # Test with target = N-1 (last class)
    target_last = torch.full((M,), N - 1, device=device, dtype=torch.int64)
    loss_last = function(x, target_last)
    loss_ref_last = F.cross_entropy(x, target_last, reduction="none")
    torch.testing.assert_close(loss_last, loss_ref_last, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("N", [192, 1024, 32768])
@pytest.mark.parametrize("M", [1, 77, 289])
@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_ignore_index(M, N, input_dtype, use_compile):
    """Test Cross Entropy with ignore_index functionality."""
    device = "cuda"
    atol, rtol = 5e-5, 1e-5
    torch.random.manual_seed(0)
    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    ignore_index = N - 1  # Use last class as ignore index
    ignore_mask = torch.rand(M, device=device) < 0.3  # Randomly ignore ~30% of samples
    target[ignore_mask] = ignore_index
    x_ref = x.detach().clone().requires_grad_()
    target_ref = target.detach().clone()
    function = torch.compile(cross_entropy, fullgraph=True) if use_compile else cross_entropy
    loss = function(x, target, reduction="none", ignore_index=ignore_index)
    loss_ref = F.cross_entropy(
        x_ref.float(), target_ref, reduction="none", ignore_index=ignore_index
    )
    # Check that losses are zero for ignored indices
    assert (loss[ignore_mask] == 0).all(), "Loss should be 0 for ignored indices"
    # Check accuracy for non-ignored indices
    if (~ignore_mask).any():
        torch.testing.assert_close(loss[~ignore_mask], loss_ref[~ignore_mask], atol=atol, rtol=rtol)
    # Test backward pass
    dloss = torch.randn_like(loss)
    torch.cuda.synchronize()
    (dx_ref,) = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss)
    (dx,) = torch.autograd.grad(loss, x, grad_outputs=dloss)
    assert dx.shape == x.shape
    torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_ignore_index_edge_cases(use_compile):
    """Test Cross Entropy ignore_index with edge cases."""
    device = "cuda"
    M, N = 16, 1024
    function = (
        torch.compile(cross_entropy_fwd, fullgraph=True) if use_compile else cross_entropy_fwd
    )

    x = 0.1 * torch.randn(M, N, device=device, dtype=torch.float32)
    # Test with all targets being ignore_index
    ignore_index = 0
    target_all_ignored = torch.zeros(M, device=device, dtype=torch.int64)
    loss_all_ignored = function(x, target_all_ignored, ignore_index=ignore_index)
    assert (loss_all_ignored == 0).all(), "All losses should be 0 when all targets are ignored"
    # Test with no targets being ignore_index
    ignore_index = -1  # Use -1 as ignore index (no valid targets will have this value)
    target_none_ignored = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    loss_none_ignored = function(x, target_none_ignored, ignore_index=ignore_index)
    loss_ref = F.cross_entropy(x, target_none_ignored, reduction="none")
    torch.testing.assert_close(loss_none_ignored, loss_ref, atol=1e-4, rtol=1e-4)
    # Test with default ignore_index (-100)
    target_with_default = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    target_with_default[
        0
    ] = -100  # Won't actually be -100 due to randint range, just for illustration
    # Since -100 is out of valid range [0, N), it won't match any targets
    loss_default = function(x, target_with_default)  # Uses default ignore_index=-100
    loss_ref_default = F.cross_entropy(x, target_with_default, reduction="none")
    torch.testing.assert_close(loss_default, loss_ref_default, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("N", [192, 1024, 32768, 128256])
@pytest.mark.parametrize("M", [1, 77, 289])
@pytest.mark.parametrize("inplace_backward", [False, True])
@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_fwd_with_grad(M, N, input_dtype, inplace_backward, use_compile):
    """Test Cross Entropy forward pass with gradient computation."""
    device = "cuda"
    atol, rtol = 1e-4, 1e-4
    torch.random.manual_seed(0)
    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    x_ref = x.detach().clone().requires_grad_()
    target_ref = target.detach().clone()
    # Test forward with gradient computation
    function = (
        torch.compile(cross_entropy_fwd, fullgraph=True) if use_compile else cross_entropy_fwd
    )
    if inplace_backward:
        x_copy = x.detach().clone()
        loss, lse, dx = function(
            x_copy, target, return_lse=True, return_dx=True, inplace_backward=True
        )
        # Check that dx is the same tensor as x_copy (inplace)
        assert dx is x_copy, "inplace_backward should modify x in-place"
    else:
        loss, lse, dx = function(x, target, return_lse=True, return_dx=True, inplace_backward=False)
        # Check that dx is a different tensor from x
        assert dx is not x, "non-inplace should create new tensor"

    # Reference implementation
    loss_ref = F.cross_entropy(x_ref.float(), target_ref, reduction="none")
    lse_ref = torch.logsumexp(x_ref.float(), dim=-1)
    dloss = torch.ones_like(loss_ref)  # Need dloss to be 1.0
    (dx_ref,) = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss)

    # Check results
    torch.testing.assert_close(loss, loss_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)

    # Test with ignore_index
    ignore_index = N - 1
    ignore_mask = torch.rand(M, device=device) < 0.3
    target[ignore_mask] = ignore_index
    if inplace_backward:
        x_copy = x.detach().clone()
        loss_ig, lse_ig, dx_ig = function(
            x_copy,
            target,
            ignore_index=ignore_index,
            return_lse=True,
            return_dx=True,
            inplace_backward=True,
        )
        assert dx_ig is x_copy
    else:
        loss_ig, lse_ig, dx_ig = function(
            x,
            target,
            ignore_index=ignore_index,
            return_lse=True,
            return_dx=True,
            inplace_backward=False,
        )
        assert dx_ig is not x
    # Reference with ignore_index
    x_ref2 = x.detach().clone().requires_grad_()
    loss_ref_ig = F.cross_entropy(
        x_ref2.float(), target, reduction="none", ignore_index=ignore_index
    )
    (dx_ref_ig,) = torch.autograd.grad(loss_ref_ig, x_ref2, grad_outputs=dloss)
    # Check that losses are zero for ignored indices
    assert (loss_ig[ignore_mask] == 0).all(), "Loss should be 0 for ignored indices"
    # Check accuracy for non-ignored indices
    if (~ignore_mask).any():
        torch.testing.assert_close(
            loss_ig[~ignore_mask], loss_ref_ig[~ignore_mask], atol=atol, rtol=rtol
        )
    torch.testing.assert_close(dx_ig, dx_ref_ig.to(input_dtype), atol=atol, rtol=rtol)
