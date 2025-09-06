# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Tri Dao.

import pytest
import torch

import cutlass

from quack.topk import topk

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024

@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
# @pytest.mark.parametrize("input_dtype", [torch.float32])
@pytest.mark.parametrize(
    "N, k",
    [(64, 16), (128, 32), (256, 16), (512, 32), (1024, 32), (4096, 32), (4096, 64), (4096, 128)]
    # [(256, 4)]
)
@pytest.mark.parametrize("M", [1, 37, 199])
@pytest.mark.parametrize("function", [topk, torch.compile(topk, fullgraph=True)])
# @pytest.mark.parametrize("M", [1])
def test_topk(M, N, k, input_dtype, function):
    """Test TopK against PyTorch reference implementation."""
    device = "cuda"
    # Set tolerance based on dtype
    if input_dtype == torch.bfloat16:
        atol = 1e-2
        rtol = 1e-2
    elif input_dtype == torch.float16:
        atol = 1e-3
        rtol = 1e-3
    else:
        atol = 1e-3
        rtol = 5e-4

    torch.random.manual_seed(0)
    # Create input tensors
    x = torch.randn(M, N, device=device, dtype=input_dtype)
    out_val, out_idx = function(x, k)
    out_val_ref, out_idx_ref = torch.topk(x, k, dim=-1, largest=True, sorted=True)

    # Check output shape and dtype
    assert out_val.shape == (M, k)
    assert out_val.dtype == input_dtype
    # Check accuracy - values should match the reference
    torch.testing.assert_close(out_val, out_val_ref, atol=atol, rtol=rtol)

    # Additional properties to check:
    # 1. All values in output should be from the input
    # 2. Values should be in descending order
    # 3. Values indexed at output indices should match output values
    # Check descending order for all rows
    assert torch.all(out_val[:, :-1] >= out_val[:, 1:]), "Some rows not in descending order"

    # Check that x indexed at out_idx is close to out_val using gather
    indexed_vals = torch.gather(x, 1, out_idx.long())
    torch.testing.assert_close(indexed_vals, out_val, atol=atol, rtol=rtol,
                               msg="Values indexed from x don't match output values")


# @pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
# def test_topk_extreme_values(input_dtype):
#     """Test TopK with extreme input values."""
#     device = "cuda"
#     M, N, k = 16, 64, 16

#     # Test with identical values
#     x_uniform = torch.full((M, N), 1.0, device=device, dtype=input_dtype)
#     out_uniform = topk(x_uniform, k)
#     # All output values should be 1.0
#     expected = torch.full((M, k), 1.0, device=device, dtype=input_dtype)
#     torch.testing.assert_close(out_uniform, expected, atol=1e-3, rtol=1e-3)

#     # Test with large range of values
#     x_range = torch.arange(N, dtype=input_dtype, device=device).unsqueeze(0).expand(M, -1)
#     out_range = topk(x_range, k)
#     # Should get the largest k values in descending order
#     expected_range = torch.arange(N-1, N-k-1, -1, dtype=input_dtype, device=device).unsqueeze(0).expand(M, -1)
#     torch.testing.assert_close(out_range, expected_range, atol=1e-6, rtol=1e-6)


# def test_topk_edge_cases():
#     """Test TopK edge cases."""
#     device = "cuda"

#     # Test k=1 (single maximum)
#     M, N = 8, 64
#     x = torch.randn(M, N, device=device, dtype=torch.float32)
#     out_val = topk(x, 1)
#     out_val_ref = torch.max(x, dim=-1, keepdim=True)[0]
#     torch.testing.assert_close(out_val, out_val_ref, atol=1e-6, rtol=1e-6)

#     # Test with negative values
#     x_neg = torch.randn(M, N, device=device, dtype=torch.float32) - 10.0
#     out_neg = topk(x_neg, 8)
#     out_ref_neg, _ = torch.topk(x_neg, 8, dim=-1, largest=True, sorted=True)
#     torch.testing.assert_close(out_neg, out_ref_neg, atol=1e-6, rtol=1e-6)


