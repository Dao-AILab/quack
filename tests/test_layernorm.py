# tests/test_layernorm.py

import pytest
import torch

from quack.rmsnorm import layernorm_fwd, layernorm_ref, layernorm_rstd_ref, layernorm_mean_ref


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("M", [1, 37, 199])
@pytest.mark.parametrize(
    "N", [256, 512, 760, 1024, 1128, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
)  # , 32768])
def test_layernorm_forward(M, N, input_dtype, eps):
    """Test LayerNorm forward pass against reference implementation."""
    device = "cuda"
    major, _ = torch.cuda.get_device_capability()
    if major == 12:
        # SM12x 99 KB SMEM: fwd holds input tile in smem; fp32 exceeds when N/cluster_n > ~25K
        smem_n_limit = 131072 if input_dtype == torch.float32 else 262144
        if N > smem_n_limit:
            pytest.skip("SM12x: exceeds 99 KB SMEM")

    # tolerance depends on precision
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
    x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
    weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)

    # pure‐PyTorch refs
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()

    out, rstd, mean = layernorm_fwd(x, weight, eps=eps, return_rstd=True, return_mean=True)
    out_ref = layernorm_ref(x_ref, weight_ref, eps=eps)
    rstd_ref_val = layernorm_rstd_ref(x_ref, eps=eps)
    mean_ref_val = layernorm_mean_ref(x_ref)

    # shapes & dtypes
    assert out.shape == x.shape
    assert out.dtype == input_dtype
    assert rstd.shape == (M,) and rstd.dtype == torch.float32
    assert mean.shape == (M,) and mean.dtype == torch.float32

    # numeric check
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(rstd, rstd_ref_val, atol=6e-4, rtol=6e-4)
    torch.testing.assert_close(mean, mean_ref_val, atol=6e-4, rtol=6e-4)

    # Return-arity contract: each (return_rstd, return_mean) toggle pair returns
    # the right shape tuple. Numerical correctness is already covered above with
    # both flags True; here we only check that the alternate arities don't crash
    # and return the right shapes.
    out_r, rstd_only = layernorm_fwd(x, weight, eps=eps, return_rstd=True, return_mean=False)
    assert out_r.shape == x.shape and rstd_only.shape == (M,) and rstd_only.dtype == torch.float32
    out_m, mean_only = layernorm_fwd(x, weight, eps=eps, return_rstd=False, return_mean=True)
    assert out_m.shape == x.shape and mean_only.shape == (M,) and mean_only.dtype == torch.float32
    out_only = layernorm_fwd(x, weight, eps=eps, return_rstd=False, return_mean=False)
    assert isinstance(out_only, torch.Tensor) and out_only.shape == x.shape


def test_layernorm_forward_masks_oob_variance_lanes():
    """Regression: ragged N must not include padding lanes in LayerNorm variance."""
    device = "cuda"
    M, N = 3, 769  # N is not a full copy/reduction tile, so the last tile has OOB lanes.
    eps = 1e-5

    cols = torch.arange(N, device=device, dtype=torch.float32)
    rows = torch.arange(M, device=device, dtype=torch.float32).unsqueeze(1)
    x = 1000.0 + rows * 100.0 + ((cols % 17) - 8.0) / 8.0
    weight = torch.ones(N, device=device, dtype=torch.float32)

    out, rstd, mean = layernorm_fwd(x, weight, eps=eps, return_rstd=True, return_mean=True)
    out_ref = layernorm_ref(x, weight, eps=eps)
    rstd_ref_val = layernorm_rstd_ref(x, eps=eps)
    mean_ref_val = layernorm_mean_ref(x)

    torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(rstd, rstd_ref_val, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mean, mean_ref_val, atol=1e-5, rtol=1e-5)


def test_layernorm_input_validation():
    """Test input validation and error handling."""
    device = "cuda"

    # Test 3D input (should fail)
    x_3d = torch.randn(2, 32, 1024, device=device, dtype=torch.float16)
    weight = torch.randn(1024, device=device, dtype=torch.float32)

    with pytest.raises(AssertionError, match="Input must be 2D"):
        layernorm_fwd(x_3d, weight)

    # Test weight dimension mismatch
    x = torch.randn(32, 1024, device=device, dtype=torch.float16)
    weight_wrong = torch.randn(512, device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match="Mismatched mW.shape[0]*"):
        layernorm_fwd(x, weight_wrong)

    # Test CPU tensors (should fail)
    x_cpu = torch.randn(32, 1024, dtype=torch.float16)
    weight_cpu = torch.randn(1024, dtype=torch.float32)

    # with pytest.raises(AssertionError, match="Tensors must be on CUDA device"):
    # With torch.library custom op, this now fails with NotImplementedError
    with pytest.raises(NotImplementedError):
        layernorm_fwd(x_cpu, weight_cpu)

    # Test unsupported dtype
    x = torch.randn(32, 1024, device=device, dtype=torch.float64)
    weight = torch.randn(1024, device=device, dtype=torch.float32)

    with pytest.raises(AssertionError, match="Unsupported dtype"):
        layernorm_fwd(x, weight)

    # Test wrong weight dtype
    x = torch.randn(32, 1024, device=device, dtype=torch.float16)
    weight_wrong_dtype = torch.randn(1024, device=device, dtype=torch.float16)

    with pytest.raises(AssertionError, match="Weight must be float32"):
        layernorm_fwd(x, weight_wrong_dtype)
