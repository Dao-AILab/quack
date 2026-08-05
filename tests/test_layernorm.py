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

# (atol, rtol) per dtype: fwd is a single normalization, bwd chains two
# reductions (mean/var grads) so its low-precision noise budget is larger.
TOLERANCES_FWD = {
    torch.bfloat16: (1e-2, 1e-2),
    torch.float16: (1e-3, 1e-3),
    torch.float32: (1e-4, 1e-4),
}
TOLERANCES_BWD = {
    torch.bfloat16: (2e-2, 2e-2),
    torch.float16: (5e-3, 5e-3),
    torch.float32: (1e-4, 1e-4),
}


@pytest.mark.parametrize("has_bias", [True, False])
# eps does not change the kernel path, so a single value suffices.
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
# M=4096 makes each persistent CTA iterate multiple rows, exercising the
# 4-slot reduction-buffer cycling and phase flips of the LayerNorm bwd path.
@pytest.mark.parametrize("M", [1, 37, 199, 4096])
@pytest.mark.parametrize(
    "N", [256, 512, 760, 1024, 1128, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
)
def test_layernorm_forward_backward(M, N, input_dtype, eps, has_bias):
    """Test LayerNorm fwd + bwd against a float32 torch autograd reference.

    The bwd consumes exactly what the fwd kernel stores (rstd, mean), like a
    training step would. Where the bwd kernel doesn't support the shape, the
    fwd is still validated (skip_bwd), mirroring test_rmsnorm_forward_backward.
    """
    device = "cuda"
    skip_bwd = N > 128 * 1024 and input_dtype == torch.float32  # bwd: not enough smem for fp32
    major, _ = torch.cuda.get_device_capability()
    if major == 12:
        # SM12x 99 KB SMEM: fwd holds input tile in smem; fp32 exceeds when N/cluster_n > ~25K
        fwd_smem_n_limit = 131072 if input_dtype == torch.float32 else 262144
        if N > fwd_smem_n_limit:
            pytest.skip("SM12x: exceeds 99 KB SMEM")
        # bwd double-buffers x and dout tiles, so its limit is tighter; run fwd-only above it.
        bwd_smem_n_limit = 32768 if input_dtype == torch.float32 else 65536
        skip_bwd = skip_bwd or N > bwd_smem_n_limit

    fwd_atol, fwd_rtol = TOLERANCES_FWD[input_dtype]
    bwd_atol, bwd_rtol = TOLERANCES_BWD[input_dtype]

    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=input_dtype)
    weight = torch.randn(N, device=device, dtype=torch.float32)

    # fp32 ground truth: same (dtype-rounded) inputs, all math in fp32.
    x_ref = x.float().requires_grad_()
    w_ref = weight.detach().clone().requires_grad_()
    b_ref = torch.zeros(N, device=device, dtype=torch.float32, requires_grad=True)
    out_ref = torch.nn.functional.layer_norm(x_ref, (N,), w_ref, b_ref, eps=eps)

    out, rstd, mean = layernorm_fwd(x, weight, eps=eps, return_rstd=True, return_mean=True)
    if not skip_bwd:
        dout = torch.randn_like(x)
        dx, dw, db = layernorm_bwd(x, weight, dout, rstd, mean, has_bias=has_bias)
        out_ref.backward(dout.float())

    # Forward: shapes, dtypes, numerics.
    assert out.shape == x.shape
    assert out.dtype == input_dtype
    assert rstd.shape == (M,) and rstd.dtype == torch.float32
    assert mean.shape == (M,) and mean.dtype == torch.float32
    torch.testing.assert_close(out, out_ref.detach().to(input_dtype), atol=fwd_atol, rtol=fwd_rtol)
    torch.testing.assert_close(rstd, layernorm_rstd_ref(x, eps=eps), atol=6e-4, rtol=6e-4)
    torch.testing.assert_close(mean, layernorm_mean_ref(x), atol=6e-4, rtol=6e-4)
    if skip_bwd:
        return

    # Backward: shapes, dtypes, numerics.
    assert dx.shape == x.shape and dx.dtype == input_dtype
    assert dw.shape == (N,) and dw.dtype == torch.float32
    torch.testing.assert_close(dx, x_ref.grad.to(input_dtype), atol=bwd_atol, rtol=bwd_rtol)
    # dw/db reduce over M rows in fp32 but in a different summation order than
    # the reference, so the error grows with sqrt(M).
    sum_atol = 5e-6 * (M**0.5) if input_dtype == torch.float32 else 1e-3 * (M**0.5)
    torch.testing.assert_close(dw, w_ref.grad, atol=sum_atol, rtol=1e-3)
    if has_bias:
        assert db.shape == (N,) and db.dtype == torch.float32
        torch.testing.assert_close(db, b_ref.grad, atol=sum_atol, rtol=1e-3)
    else:
        assert db is None


def test_layernorm_fwd_return_arity():
    """Return-arity contract: each (return_rstd, return_mean) toggle pair returns
    the right shape tuple. Numerical correctness (with both flags True) is covered
    by test_layernorm_forward_backward; here we only check that the alternate
    arities don't crash and return the right shapes."""
    device = "cuda"
    M, N = 37, 1024
    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=torch.bfloat16)
    weight = torch.randn(N, device=device, dtype=torch.float32)

    out_r, rstd_only = layernorm_fwd(x, weight, return_rstd=True, return_mean=False)
    assert out_r.shape == x.shape and rstd_only.shape == (M,) and rstd_only.dtype == torch.float32
    out_m, mean_only = layernorm_fwd(x, weight, return_rstd=False, return_mean=True)
    assert out_m.shape == x.shape and mean_only.shape == (M,) and mean_only.dtype == torch.float32
    out_only = layernorm_fwd(x, weight, return_rstd=False, return_mean=False)
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

    # Eager calls bypass the torch.library dispatcher, so TVM-FFI performs the
    # device validation directly.
    with pytest.raises(ValueError, match="expected device_type=cuda"):
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
