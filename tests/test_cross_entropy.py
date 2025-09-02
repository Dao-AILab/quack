# Copyright (c) 2025, Tri Dao.

import pytest
import torch
import torch.nn.functional as F

from quack.cross_entropy import cross_entropy_fwd, cross_entropy
import cutlass

@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
# @pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "N",
    [192, 256, 512, 760, 1024, 1128, 2048, 4096, 8192, 16384, 32768, 65536, 128256, 131072, 256128, 262144]
    # [32768]
)
@pytest.mark.parametrize("M", [1, 77, 289])
# @pytest.mark.parametrize("M", [1])
def test_cross_entropy(M, N, input_dtype):
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
    loss = cross_entropy(x, target)
    loss_ref = F.cross_entropy(x_ref.float(), target_ref, reduction='none')
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
    dx_ref, = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss)
    dx, = torch.autograd.grad(loss, x, grad_outputs=dloss)
    assert dx.shape == x.shape
    torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_cross_entropy_extreme_values(input_dtype):
    """Test Cross Entropy with extreme input values."""
    device = "cuda"
    M, N = 16, 1024
    # Test with large positive values
    x_large = torch.full((M, N), 10.0, device=device, dtype=input_dtype)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    loss_large = cross_entropy_fwd(x_large, target)
    # Should be around log(N) since all logits are equal
    expected_large = torch.full_like(loss_large, torch.log(torch.tensor(N, dtype=torch.float32)))
    torch.testing.assert_close(loss_large, expected_large, atol=1e-2, rtol=1e-2)
    # Test with large negative values
    x_small = torch.full((M, N), -10.0, device=device, dtype=input_dtype)
    loss_small = cross_entropy_fwd(x_small, target)
    # Should also be around log(N)
    torch.testing.assert_close(loss_small, expected_large, atol=1e-2, rtol=1e-2)
    # Test with one-hot like scenario (one large value, rest small)
    x_onehot = torch.full((M, N), -10.0, device=device, dtype=input_dtype)
    # Set the target class to have large logit
    for i in range(M):
        x_onehot[i, target[i]] = 10.0
    loss_onehot = cross_entropy_fwd(x_onehot, target)
    # Should be close to 0 since target class has highest probability
    assert (loss_onehot < 1.0).all()


def test_cross_entropy_numerical_stability():
    """Test that cross entropy is numerically stable."""
    device = "cuda"
    M, N = 8, 512
    # Create input with a wide range of values
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    # Add large constant to test numerical stability
    x_shifted = x + 100.0
    loss = cross_entropy_fwd(x, target)
    loss_shifted = cross_entropy_fwd(x_shifted, target)
    # Results should be identical (cross entropy is translation invariant)
    torch.testing.assert_close(loss, loss_shifted, atol=1e-5, rtol=1e-5)


def test_cross_entropy_edge_targets():
    """Test cross entropy with edge case targets."""
    device = "cuda"
    M, N = 16, 1024
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch.float32)
    # Test with target = 0 (first class)
    target_first = torch.zeros(M, device=device, dtype=torch.int64)
    loss_first = cross_entropy_fwd(x, target_first)
    loss_ref_first = F.cross_entropy(x, target_first, reduction='none')
    torch.testing.assert_close(loss_first, loss_ref_first, atol=1e-4, rtol=1e-4)
    # Test with target = N-1 (last class)
    target_last = torch.full((M,), N-1, device=device, dtype=torch.int64)
    loss_last = cross_entropy_fwd(x, target_last)
    loss_ref_last = F.cross_entropy(x, target_last, reduction='none')
    torch.testing.assert_close(loss_last, loss_ref_last, atol=1e-4, rtol=1e-4)
