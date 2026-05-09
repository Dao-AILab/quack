# tests/test_layernorm.py

import pytest
import torch

from quack.rmsnorm import (
    layernorm_bwd,
    layernorm_fwd,
    layernorm_mean_ref,
    layernorm_ref,
    layernorm_rstd_ref,
)


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


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("M", [1, 37, 199])
@pytest.mark.parametrize(
    "N", [256, 512, 760, 1024, 1128, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
)
def test_layernorm_backward(M, N, input_dtype, eps):
    """Test LayerNorm backward against torch autograd."""
    device = "cuda"
    # RMSNormBackward rejects N > 128k with >=32-bit dtype (smem limit).
    if N > 128 * 1024 and input_dtype == torch.float32:
        pytest.skip("RMSNormBackward: N > 128k unsupported for fp32")
    major, _ = torch.cuda.get_device_capability()
    if major == 12:
        smem_n_limit = 131072 if input_dtype == torch.float32 else 262144
        if N > smem_n_limit:
            pytest.skip("SM12x: exceeds 99 KB SMEM")

    if input_dtype == torch.bfloat16:
        atol, rtol = 2e-2, 2e-2
    elif input_dtype == torch.float16:
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 1e-4, 1e-4

    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
    # F.layer_norm requires weight/bias dtype to match input; QuACK requires fp32 weight.
    w_ref = torch.randn(N, device=device, dtype=input_dtype, requires_grad=True)
    b_ref = torch.randn(N, device=device, dtype=input_dtype, requires_grad=True)
    w_f32 = w_ref.detach().to(torch.float32).contiguous()

    y_ref = torch.nn.functional.layer_norm(x, (N,), w_ref, b_ref, eps=eps)
    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)
    dx_ref = x.grad.detach()
    dw_ref = w_ref.grad.detach()
    db_ref = b_ref.grad.detach()

    mean = x.detach().to(torch.float32).mean(dim=-1)
    rstd = (x.detach().to(torch.float32).var(dim=-1, unbiased=False) + eps).rsqrt()

    dx, dw, db = layernorm_bwd(x.detach(), w_f32, dy, rstd, mean)

    assert dx.shape == x.shape and dx.dtype == input_dtype
    assert dw.shape == (N,) and dw.dtype == torch.float32
    assert db.shape == (N,) and db.dtype == torch.float32

    torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=rtol)
    # dw/db accumulate across rows so absolute error scales with M; use relative.
    rel_dw = (dw.float() - dw_ref.float()).abs().max().item() / max(
        dw_ref.float().abs().mean().item(), 1e-6
    )
    rel_db = (db.float() - db_ref.float()).abs().max().item() / max(
        db_ref.float().abs().mean().item(), 1e-6
    )
    assert rel_dw < 0.05, f"dw rel err {rel_dw}"
    assert rel_db < 0.05, f"db rel err {rel_db}"


@pytest.mark.parametrize("return_rstd", [True, False])
@pytest.mark.parametrize("return_mean", [True, False])
def test_layernormnorm_return_rstd_option(return_rstd, return_mean):
    """Test that return_rstd option works correctly."""
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6

    x = torch.randn(M, N, device=device, dtype=torch.float16)
    weight = torch.randn(N, device=device, dtype=torch.float32)

    if return_rstd and return_mean:
        out, rstd, mean = layernorm_fwd(x, weight, eps=eps, return_rstd=True, return_mean=True)
        assert out.shape == (M, N)
        assert rstd.shape == (M,)
        assert rstd.dtype == torch.float32
        assert mean.shape == (M,)
        assert mean.dtype == torch.float32
    elif return_rstd and not return_mean:
        out, rstd = layernorm_fwd(x, weight, eps=eps, return_rstd=True, return_mean=False)
        assert out.shape == (M, N)
        assert rstd.shape == (M,)
        assert rstd.dtype == torch.float32
    elif not return_rstd and return_mean:
        out, mean = layernorm_fwd(x, weight, eps=eps, return_rstd=False, return_mean=True)
        assert out.shape == (M, N)
        assert mean.shape == (M,)
        assert mean.dtype == torch.float32
    else:
        out = layernorm_fwd(x, weight, eps=eps, return_rstd=False, return_mean=False)
        assert out.shape == (M, N)
        assert isinstance(out, torch.Tensor)


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
