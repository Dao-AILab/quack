# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.rmsnorm import (
    _compile_rmsnorm_fwd,
    rmsnorm,
    rmsnorm_bwd,
    rmsnorm_bwd_ref,
    rmsnorm_bwd_tuned,
    rmsnorm_fwd,
    rmsnorm_fwd_tuned,
    rmsnorm_ref,
)
from quack.rmsnorm_config import RmsNormBwdConfig, RmsNormFwdConfig

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024

TOLERANCES = {
    torch.bfloat16: 1e-1,
    torch.float16: 1e-2,
    torch.float32: 1e-4,
}


# Grid-reduction rationale: eps does not change the kernel path, fp16 weight_dtype
# shares the 16-bit weight-load codegen with bf16, and the dropped N / M values are
# interior to regimes already covered by their neighbors.
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32, None])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
# @pytest.mark.parametrize("input_dtype", [torch.float32])
@pytest.mark.parametrize(
    "N",
    [192, 256, 760, 1024, 1128, 4096, 32768, 131072, 262144],
)
# @pytest.mark.parametrize("M", [1])
@pytest.mark.parametrize("M", [1, 199, 8 * 1024])
@pytest.mark.parametrize("use_compile", [False, True])
# @pytest.mark.parametrize("use_compile", [False])
def test_rmsnorm_forward_backward(M, N, input_dtype, weight_dtype, eps, use_compile):
    """Test RMSNorm forward pass against reference implementation."""
    if N >= 256 * 1024 and input_dtype == torch.float32 and M >= 8 * 1024:
        pytest.skip("Skipping large tensor test for float32 to avoid OOM")
    major, _ = torch.cuda.get_device_capability()
    if major == 12:
        # SM12x 99 KB SMEM: bwd double-buffers 2 tensors; fp32 exceeds at N > 32K, fp16/bf16 at N > 64K
        smem_n_limit = 32768 if input_dtype == torch.float32 else 65536
        if N > smem_n_limit:
            pytest.skip("SM12x: exceeds 99 KB SMEM")
    torch.cuda.empty_cache()
    device = "cuda"
    atol = TOLERANCES[input_dtype]
    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
    if weight_dtype is not None:
        weight = torch.randn(N, device=device, dtype=weight_dtype, requires_grad=True)
    else:
        weight = None
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_() if weight is not None else None
    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
    # Compile ref for large inputs to avoid OOMs.
    compile_ref = N >= 256 * 1024 and M >= 8 * 1024
    ref_function = torch.compile(rmsnorm_ref) if compile_ref else rmsnorm_ref
    # Run all kernel calls (fwd + bwd) first, then do numerical assertions.
    # and bwd kernels to be dispatched (and therefore compiled). `assert_close` raises
    # on fake tensors, so run all kernel calls FIRST, then do numerical assertions.
    out = function(x, weight, eps=eps)
    out_ref = ref_function(x_ref, weight_ref, eps=eps)
    skip_bwd = N > 128 * 1024 and input_dtype == torch.float32  # Not enough smem for bwd
    if not skip_bwd:
        grad_out = torch.randn_like(out)
        torch.cuda.synchronize()
        out.backward(grad_out)
        out_ref.backward(grad_out)

    assert out.shape == x.shape
    assert out.dtype == input_dtype
    # assert_close upcasts both operands to fp32 and allocates diff/mask
    # temporaries (~7x the operand size). For the largest shapes here (4 GiB
    # operands on top of 28 GiB of live tensors) that transient pushed peak
    # memory to ~56 GiB. Chunk the comparison for operands >= 1 GiB, and free
    # each tensor as soon as it has been checked, keeping the peak at ~30 GiB.
    n_chunks = 16 if x.numel() * x.element_size() >= 2**30 else 1

    def assert_close(actual, expected, atol):
        for actual_c, expected_c in zip(actual.chunk(n_chunks), expected.chunk(n_chunks)):
            torch.testing.assert_close(actual_c, expected_c, atol=atol, rtol=1e-3)

    assert_close(out, out_ref, atol)
    if skip_bwd:
        return
    del out, out_ref, grad_out
    assert_close(x.grad, x_ref.grad, atol)
    x.grad = x_ref.grad = None
    if weight_dtype is not None:
        if weight_dtype == torch.float32:
            # Kernel and reference reduce dout*x_hat in float32 but in different summation
            # orders, so the error grows with sqrt(M) (number of rows being reduced).
            weight_atol = 5e-6 * (M**0.5)
        else:
            # bf16/fp16: different reduction orders can land on different ULPs.
            # Tolerance = 1 ULP at the magnitude of the largest gradient.
            weight_atol = 2 * torch.finfo(weight_dtype).eps * weight_ref.grad.abs().max()
        torch.testing.assert_close(weight.grad, weight_ref.grad, atol=weight_atol, rtol=1e-3)


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm_noncontiguous_grad(input_dtype, use_compile):
    """Test RMSNorm with 3D input where backward produces non-contiguous gradients (issue #88)."""
    device = "cuda"
    atol = TOLERANCES[input_dtype]
    torch.random.manual_seed(0)
    # 3D input: backward grad from .sum() is non-contiguous after reshape
    x = torch.randn(2, 1024, 256, device=device, dtype=input_dtype, requires_grad=True)
    w = torch.ones(256, device=device, dtype=torch.float32)
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone()
    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
    out = function(x, w, eps=1e-6)
    out_ref = rmsnorm_ref(x_ref, w_ref, eps=1e-6)
    out.sum().backward()
    out_ref.sum().backward()
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=1e-3)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=1e-3)


def test_rmsnorm_compile_2d_then_4d():
    """Regression test: torch.compile(rmsnorm) must work when called first with 2D input
    (standard) then 4D input (per-head), without dynamo.reset() in between."""
    torch._dynamo.reset()
    f = torch.compile(rmsnorm, fullgraph=True)
    device = "cuda"
    atol = TOLERANCES[torch.bfloat16]

    # Step 1: 2D input, 1D weight
    x = torch.randn(32, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(256, device=device, dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()
    out = f(x, w, eps=1e-5)
    out_ref = rmsnorm_ref(x_ref, w_ref, eps=1e-5)
    out.sum().backward()
    out_ref.sum().backward()
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=1e-3)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=1e-3)

    # Step 2: 4D input, 2D per-head weight + bias + residual (different rank, different args)
    x4 = torch.randn(2, 16, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(4, 64, device=device, dtype=torch.float32, requires_grad=True)
    b2 = torch.randn(4, 64, device=device, dtype=torch.float32, requires_grad=True)
    r4 = torch.randn(2, 16, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    x4_ref = x4.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    b2_ref = b2.detach().clone().requires_grad_()
    r4_ref = r4.detach().clone().requires_grad_()
    out2 = f(x4, w2, bias=b2, residual=r4, eps=1e-6)
    out2_ref = rmsnorm_ref(x4_ref, w2_ref, bias=b2_ref, residual=r4_ref, eps=1e-6)
    if isinstance(out2_ref, tuple):
        out2_ref = out2_ref[0]
    out2.sum().backward()
    out2_ref.sum().backward()
    torch.testing.assert_close(out2, out2_ref, atol=atol, rtol=1e-3)
    torch.testing.assert_close(x4.grad, x4_ref.grad, atol=atol, rtol=1e-3)
    torch.testing.assert_close(w2.grad, w2_ref.grad, atol=atol, rtol=1e-3)
    torch.testing.assert_close(b2.grad, b2_ref.grad, atol=atol, rtol=1e-3)
    torch.testing.assert_close(r4.grad, r4_ref.grad, atol=atol, rtol=1e-3)


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("use_bias,use_residual", [(False, False), (True, True)])
@pytest.mark.parametrize(
    "B,S,H,D",
    [
        (2, 16, 4, 64),  # typical multi-head QK rmsnorm
        (1, 1, 32, 128),  # single token, many heads (different smem layout per head)
    ],
)
def test_rmsnorm_qk(B, S, H, D, use_compile, input_dtype, use_bias, use_residual):
    device = "cuda"
    eps = 1e-6
    atol = TOLERANCES[input_dtype]
    torch.random.manual_seed(0)

    x = torch.randn(B, S, H, D, device=device, dtype=input_dtype, requires_grad=True)
    weight = torch.randn(H, D, device=device, dtype=torch.float32, requires_grad=True)
    bias = (
        torch.randn(H, D, device=device, dtype=torch.float32, requires_grad=True)
        if use_bias
        else None
    )
    residual = (
        torch.randn(B, S, H, D, device=device, dtype=input_dtype, requires_grad=True)
        if use_residual
        else None
    )

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    residual_ref = residual.detach().clone().requires_grad_() if residual is not None else None

    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
    out = function(x, weight, bias=bias, residual=residual, eps=eps)
    out_ref = rmsnorm_ref(x_ref, weight_ref, bias=bias_ref, residual=residual_ref, eps=eps)
    if residual is not None:
        out_ref = out_ref[0]

    grad_out = torch.randn_like(out)
    torch.cuda.synchronize()
    out.backward(grad_out)
    out_ref.backward(grad_out)

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=1e-3)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=1e-3)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=atol, rtol=1e-3)
    if bias is not None:
        torch.testing.assert_close(bias.grad, bias_ref.grad, atol=atol, rtol=1e-3)
    if residual is not None:
        torch.testing.assert_close(residual.grad, residual_ref.grad, atol=atol, rtol=1e-3)


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
    grad_out = torch.randn_like(out)
    torch.cuda.synchronize()
    out.backward(grad_out)
    out_ref.backward(grad_out)
    assert out.shape == x.shape
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=1e-3)
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
    n_chunks = 16
    # x + out must be fully materialized for the kernel call (irreducible);
    # the reference is computed and compared one M/n_chunks-row chunk at a
    # time (rmsnorm is row-wise, so chunking over M is exact), so out_ref
    # never fully materializes. Budget: 2 full tensors + ~3 chunk-sized
    # temporaries (out_ref chunk, diff, abs). Gate on *free* memory so a
    # partially-occupied GPU on a shared machine skips instead of OOMing.
    peak_bytes = 2 * M * N * 2 + 3 * (M // n_chunks) * N * 2
    # Release cache reserved by earlier tests in this worker before measuring,
    # otherwise it counts as "used" and forces a spurious skip.
    torch.cuda.empty_cache()
    free_bytes = torch.cuda.mem_get_info()[0]
    if peak_bytes > free_bytes * 0.9:
        pytest.skip(
            f"Insufficient free VRAM ({free_bytes // 2**30} GiB free,"
            f" need ~{peak_bytes // 2**30} GiB)"
        )
    device = "cuda"
    atol = TOLERANCES[input_dtype]
    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=False)
    weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=False)
    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
    out = function(x, weight, eps=eps)
    # Need to compile the ref, otherwise it OOMs. All chunks share one shape,
    # so only the first chunk pays the compile.
    rmsnorm_compiled = torch.compile(rmsnorm_ref)
    for x_c, out_c in zip(x.chunk(n_chunks), out.chunk(n_chunks)):
        out_ref_c = rmsnorm_compiled(x_c, weight, eps=eps)
        assert (out_c - out_ref_c).abs().max() < atol


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

    with pytest.raises(ValueError, match="Mismatched mW.shape[0]*"):
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
    _compile_rmsnorm_fwd.cache_clear()
    assert _compile_rmsnorm_fwd.cache_info().currsize == 0

    x1 = torch.randn(M, N, device=device, dtype=torch.float16)
    weight1 = torch.randn(N, device=device, dtype=torch.float32)

    # First call should compile
    out1 = rmsnorm_fwd(x1, weight1, eps=eps)
    assert _compile_rmsnorm_fwd.cache_info().currsize == 1

    # Same shape should reuse cache
    x2 = torch.randn(M, N, device=device, dtype=torch.float16)
    weight2 = torch.randn(N, device=device, dtype=torch.float32)
    out2 = rmsnorm_fwd(x2, weight2, eps=eps)
    assert _compile_rmsnorm_fwd.cache_info().currsize == 1

    # Changing batch size should reuse cache
    x2 = torch.randn(M * 2, N, device=device, dtype=torch.float16)
    weight2 = torch.randn(N, device=device, dtype=torch.float32)
    out2 = rmsnorm_fwd(x2, weight2, eps=eps)
    assert _compile_rmsnorm_fwd.cache_info().currsize == 1

    # Different shape should create new cache entry
    x3 = torch.randn(M, N * 2, device=device, dtype=torch.float16)
    weight3 = torch.randn(N * 2, device=device, dtype=torch.float32)
    out3 = rmsnorm_fwd(x3, weight3, eps=eps)
    assert _compile_rmsnorm_fwd.cache_info().currsize == 2

    # Different dtype should create new cache entry
    x4 = torch.randn(M, N, device=device, dtype=torch.float32)
    weight4 = torch.randn(N, device=device, dtype=torch.float32)
    out4 = rmsnorm_fwd(x4, weight4, eps=eps)
    assert _compile_rmsnorm_fwd.cache_info().currsize == 3


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

    grad_out = torch.randn_like(out)
    torch.cuda.synchronize()
    out.backward(grad_out)
    out_ref.backward(grad_out)

    assert out.shape == x.shape
    assert out.dtype == input_dtype
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-3)
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

    grad_out = torch.randn_like(out)
    torch.cuda.synchronize()
    if backward_type == "only_residual":
        residual_out.backward(grad_out)
        residual_out_ref.backward(grad_out)
    elif backward_type == "only_output":
        out.backward(grad_out)
        out_ref.backward(grad_out)
    else:
        (out + residual_out).backward(grad_out)
        (out_ref + residual_out_ref).backward(grad_out)

    assert out.shape == x.shape
    assert out.dtype == input_dtype
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(residual_out, residual_out_ref, atol=1e-2, rtol=1e-3)
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

    grad_out = torch.randn_like(out)
    torch.cuda.synchronize()
    out.backward(grad_out)
    out_ref.backward(grad_out)

    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(residual.grad, residual_ref.grad, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("use_compile", [False, True])
def test_rmsnorm_residual_dtype_override(use_compile):
    """Regression test: user-supplied residual_dtype must not be overwritten by residual.dtype.

    Passing a bf16 residual together with residual_dtype=torch.float32 should produce an fp32
    residual_out holding the full-precision (x + residual) sum (the kernel accumulates in fp32
    and casts only at the final store).
    """
    device = "cuda"
    M, N = 32, 1024
    eps = 1e-6
    input_dtype = torch.bfloat16
    weight_dtype = torch.float32

    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=input_dtype)
    weight = torch.randn(N, device=device, dtype=weight_dtype)
    residual = torch.randn(M, N, device=device, dtype=input_dtype)

    out, residual_out, _ = rmsnorm_fwd(
        x, weight, residual=residual, residual_dtype=torch.float32, eps=eps
    )
    assert residual_out.dtype == torch.float32, (
        f"residual_out dtype overridden: got {residual_out.dtype}, expected float32"
    )
    expected_residual_f32 = x.float() + residual.float()
    torch.testing.assert_close(residual_out, expected_residual_f32, atol=0.0, rtol=0.0)
    assert out.dtype == input_dtype

    function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
    out_hl, residual_out_hl = function(
        x, weight, residual=residual, residual_dtype=torch.float32, eps=eps, prenorm=True
    )
    assert residual_out_hl.dtype == torch.float32
    torch.testing.assert_close(residual_out_hl, expected_residual_f32, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("store_rstd", [False, True])
def test_rmsnorm_fwd_empty(store_rstd):
    """rmsnorm_fwd must not launch a kernel when the input has zero elements
    (e.g. uneven FSDP shards), and must return correctly-shaped empty outputs."""
    dtype = torch.bfloat16
    N = 4096
    x = torch.empty(0, N, device="cuda", dtype=dtype)
    weight = torch.randn(N, device="cuda", dtype=dtype)
    out, residual_out, rstd = rmsnorm_fwd(x, weight, store_rstd=store_rstd)
    assert out.shape == x.shape and out.numel() == 0
    # No residual passed in: residual_out aliases x (and is empty).
    assert residual_out.shape == x.shape and residual_out.numel() == 0
    if store_rstd:
        assert rstd.shape == x.shape[:-1] and rstd.numel() == 0
    else:
        assert rstd is None


@pytest.mark.parametrize("has_bias", [False, True])
def test_rmsnorm_bwd_empty(has_bias):
    """rmsnorm_bwd must not launch a kernel when inputs have zero elements,
    and dw / db must reduce to zero (the gradient over an empty batch)."""
    dtype = torch.bfloat16
    N = 4096
    x = torch.empty(0, N, device="cuda", dtype=dtype)
    weight = torch.randn(N, device="cuda", dtype=dtype)
    dout = torch.empty(0, N, device="cuda", dtype=dtype)
    rstd = torch.empty(0, device="cuda", dtype=torch.float32)
    dx, dw, db, dresidual = rmsnorm_bwd(x, weight, dout, rstd, has_bias=has_bias)
    assert dx.shape == x.shape and dx.numel() == 0
    assert dw is not None and dw.shape == weight.shape
    assert torch.all(dw == 0)
    if has_bias:
        assert db is not None and db.shape == weight.shape
        assert torch.all(db == 0)
    else:
        assert db is None
    assert dresidual is None


# ---------------------------------------------------------------------------
# Tuned-path tests: mirror tests/test_linear.py::test_gemm — drive the
# autotune-decorated fn directly with an explicit config (bypassing the search)
# to exercise the plumbing for specific knob combinations, plus one end-to-end
# autotune dispatch test for sanity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threads_per_row", [32, 64])
def test_rmsnorm_degenerate_cluster_config_is_clamped(threads_per_row):
    """A clustered config whose single CTA tile already spans the whole row
    (tiler_mn[1] >= N) must still compute correct fwd/bwd, not double-count.

    Before the fix, local_tile collapsed every peer CTA onto tile 0, so the
    cluster reduction summed the full row cluster_n times and scaled rstd by
    ~1/sqrt(cluster_n) (e.g. 0.7071 for N=256, cluster_n=2). The runtime clamp
    in ReductionBase._cap_cluster_n drops cluster_n back to a value where each
    peer owns a distinct tile, restoring correctness.
    """
    device_capacity = _device_capacity_or_skip()
    if device_capacity < 9:
        pytest.skip("SM8x lacks cluster support")
    torch.random.manual_seed(0)
    device = "cuda"
    M, N = 256, 256  # vecsize=8 -> 32 vec-blocks; tpr*cluster_n=64/128 >= 32 => degenerate
    dtype = torch.bfloat16

    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype) * 0.1
    out = torch.empty_like(x)
    rstd = torch.empty(M, device=device, dtype=torch.float32)

    fwd_config = RmsNormFwdConfig(
        num_threads=128,
        threads_per_row=threads_per_row,
        cluster_n=2,  # degenerate at N=256: tile_n >= N
        reload_from=None,
        delay_w_load=False,
    )
    rmsnorm_fwd_tuned.fn(
        x,
        weight,
        out,
        bias=None,
        rstd=rstd,
        mean=None,
        residual=None,
        residual_out=None,
        eps=1e-6,
        is_layernorm=False,
        per_head=False,
        config=fwd_config,
    )
    out_ref = rmsnorm_ref(x, weight, eps=1e-6)
    rstd_ref = torch.rsqrt(x.float().square().mean(dim=-1) + 1e-6)
    # The bug corrupted rstd by ~1/sqrt(2); require it close (not off by 30%).
    torch.testing.assert_close(rstd, rstd_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out, out_ref, atol=2 * TOLERANCES[dtype], rtol=1e-3)

    # Backward must be correct too.
    dout = torch.randn(M, N, device=device, dtype=dtype)
    dx = torch.empty_like(x)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count * 2
    dw_partial = torch.empty((sm_count, N), device=device, dtype=torch.float32)
    bwd_config = RmsNormBwdConfig(
        num_threads=128,
        threads_per_row=threads_per_row,
        cluster_n=2,
        reload_wdy=None,
        reload_x=None,
        use_tma=False,
        smem_stages=2,
    )
    rmsnorm_bwd_tuned.fn(
        x,
        weight,
        dout,
        rstd_ref,
        dx,
        dw_partial=dw_partial,
        db_partial=None,
        dresidual_out=None,
        dresidual=None,
        sm_count=sm_count,
        per_head=False,
        has_dw_partial=True,
        has_db_partial=False,
        config=bwd_config,
    )
    dw = dw_partial.sum(dim=0).to(weight.dtype)
    dx_ref, dw_ref = rmsnorm_bwd_ref(x, weight, dout, rstd_ref, eps=1e-6)
    torch.testing.assert_close(dx, dx_ref, atol=2 * TOLERANCES[dtype], rtol=1e-3)
    torch.testing.assert_close(dw, dw_ref, atol=2 * TOLERANCES[dtype], rtol=1e-3)


def _device_capacity_or_skip() -> int:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for tuned-path tests")
    return get_device_capacity(torch.device("cuda"))[0]


@pytest.mark.parametrize("reload_from", [None, "smem", "gmem"])
@pytest.mark.parametrize("cluster_n", [1, 2])
@pytest.mark.parametrize("num_threads,threads_per_row", [(128, 32), (128, 128), (256, 128)])
def test_rmsnorm_fwd_tuned_config(num_threads, threads_per_row, cluster_n, reload_from):
    """Drive rmsnorm_fwd_tuned.fn with an explicit config, bypassing autotune.

    Mirrors test_linear::test_gemm: exercises specific config combinations to
    catch breakage in the tuned-path plumbing (config -> RMSNorm -> compile).
    """
    device_capacity = _device_capacity_or_skip()
    if device_capacity < 9 and cluster_n > 1:
        pytest.skip("SM8x lacks cluster support")
    torch.random.manual_seed(0)
    device = "cuda"
    M, N = 1024, 4096
    dtype = torch.bfloat16
    if threads_per_row * cluster_n > N:
        pytest.skip("Config over-clusters for this N")

    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype) * 0.1
    out = torch.empty_like(x)
    rstd = torch.empty(M, device=device, dtype=torch.float32)

    config = RmsNormFwdConfig(
        num_threads=num_threads,
        threads_per_row=threads_per_row,
        cluster_n=cluster_n,
        reload_from=reload_from,
        delay_w_load=False,
    )
    rmsnorm_fwd_tuned.fn(
        x,
        weight,
        out,
        bias=None,
        rstd=rstd,
        mean=None,
        residual=None,
        residual_out=None,
        eps=1e-6,
        is_layernorm=False,
        per_head=False,
        config=config,
    )
    out_ref = rmsnorm_ref(x, weight, eps=1e-6)
    # rmsnorm_ref always promotes to fp32 internally, so a gemm-style "kernel
    # vs pt" tolerance would collapse to zero. Tuned configs may pick a
    # different reduction tree than the analytical heuristic, so allow 2x the
    # un-tuned bf16 noise budget.
    torch.testing.assert_close(out, out_ref, atol=2 * TOLERANCES[dtype], rtol=1e-3)


@pytest.mark.parametrize("reload_x", [None, "smem"])
@pytest.mark.parametrize("reload_wdy", [None, "smem"])
@pytest.mark.parametrize("cluster_n", [1, 2])
@pytest.mark.parametrize("num_threads,threads_per_row", [(128, 32), (256, 128)])
def test_rmsnorm_bwd_tuned_config(num_threads, threads_per_row, cluster_n, reload_wdy, reload_x):
    """Drive rmsnorm_bwd_tuned.fn with an explicit config, bypassing autotune."""
    device_capacity = _device_capacity_or_skip()
    if device_capacity < 9 and cluster_n > 1:
        pytest.skip("SM8x lacks cluster support")
    torch.random.manual_seed(0)
    device = "cuda"
    M, N = 1024, 4096
    dtype = torch.bfloat16
    if threads_per_row * cluster_n > N:
        pytest.skip("Config over-clusters for this N")

    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype) * 0.1
    rstd = torch.rsqrt(x.float().square().mean(dim=-1) + 1e-6)
    dout = torch.randn(M, N, device=device, dtype=dtype)
    dx = torch.empty_like(x)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count * 2
    dw_partial = torch.empty((sm_count, N), device=device, dtype=torch.float32)

    config = RmsNormBwdConfig(
        num_threads=num_threads,
        threads_per_row=threads_per_row,
        cluster_n=cluster_n,
        reload_wdy=reload_wdy,
        reload_x=reload_x,
        use_tma=False,
        smem_stages=2,
    )
    rmsnorm_bwd_tuned.fn(
        x,
        weight,
        dout,
        rstd,
        dx,
        dw_partial=dw_partial,
        db_partial=None,
        dresidual_out=None,
        dresidual=None,
        sm_count=sm_count,
        per_head=False,
        has_dw_partial=True,
        has_db_partial=False,
        config=config,
    )
    dw = dw_partial.sum(dim=0).to(weight.dtype)
    dx_ref, dw_ref = rmsnorm_bwd_ref(x, weight, dout, rstd, eps=1e-6)
    torch.testing.assert_close(dx, dx_ref, atol=2 * TOLERANCES[dtype], rtol=1e-3)
    torch.testing.assert_close(dw, dw_ref, atol=2 * TOLERANCES[dtype], rtol=1e-3)


def test_rmsnorm_fwd_tuned_dispatch():
    """End-to-end autotune dispatch: invoke rmsnorm_fwd_tuned (no .fn) so the
    @autotune decorator picks a config from the search space, benchmarks, and
    caches it. Validates the autotune cache key + prune flow."""
    _device_capacity_or_skip()
    torch.random.manual_seed(0)
    device = "cuda"
    # Small N keeps prune_invalid_rmsnorm_fwd_configs filtered list short so
    # the bench doesn't dominate the test runtime.
    M, N = 256, 256
    dtype = torch.bfloat16
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype) * 0.1
    out = torch.empty_like(x)
    rstd = torch.empty(M, device=device, dtype=torch.float32)
    rmsnorm_fwd_tuned(
        x,
        weight,
        out,
        bias=None,
        rstd=rstd,
        mean=None,
        residual=None,
        residual_out=None,
        eps=1e-6,
        is_layernorm=False,
        per_head=False,
    )
    assert rmsnorm_fwd_tuned.best_config is not None
    out_ref = rmsnorm_ref(x, weight, eps=1e-6)
    # Tuned path may pick a config with a different reduction tree than the
    # analytical heuristic, so allow 2x the un-tuned bf16 noise budget.
    torch.testing.assert_close(out, out_ref, atol=2 * TOLERANCES[dtype], rtol=1e-3)


def test_rmsnorm_bwd_tuned_dispatch():
    """End-to-end autotune dispatch for the bwd. See fwd counterpart."""
    _device_capacity_or_skip()
    torch.random.manual_seed(0)
    device = "cuda"
    M, N = 256, 256
    dtype = torch.bfloat16
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype) * 0.1
    rstd = torch.rsqrt(x.float().square().mean(dim=-1) + 1e-6)
    dout = torch.randn(M, N, device=device, dtype=dtype)
    dx = torch.empty_like(x)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count * 2
    dw_partial = torch.empty((sm_count, N), device=device, dtype=torch.float32)
    rmsnorm_bwd_tuned(
        x,
        weight,
        dout,
        rstd,
        dx,
        dw_partial=dw_partial,
        db_partial=None,
        dresidual_out=None,
        dresidual=None,
        sm_count=sm_count,
        per_head=False,
        has_dw_partial=True,
        has_db_partial=False,
    )
    assert rmsnorm_bwd_tuned.best_config is not None
    dw = dw_partial.sum(dim=0).to(weight.dtype)
    dx_ref, dw_ref = rmsnorm_bwd_ref(x, weight, dout, rstd, eps=1e-6)
    # Tuned path may pick a config with a different reduction tree than the
    # analytical heuristic, so allow 2x the un-tuned bf16 noise budget.
    torch.testing.assert_close(dx, dx_ref, atol=2 * TOLERANCES[dtype], rtol=1e-3)
    torch.testing.assert_close(dw, dw_ref, atol=2 * TOLERANCES[dtype], rtol=1e-3)


def test_rmsnorm_fwd_tuned_requires_config():
    """rmsnorm_fwd_tuned.fn (bypassing autotune) must raise without a config."""
    _device_capacity_or_skip()
    device = "cuda"
    x = torch.zeros(4, 256, device=device, dtype=torch.bfloat16)
    weight = torch.ones(256, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    with pytest.raises(RuntimeError, match="requires a config"):
        rmsnorm_fwd_tuned.fn(x, weight, out)


def test_rmsnorm_bwd_tuned_requires_config():
    """rmsnorm_bwd_tuned.fn (bypassing autotune) must raise without a config."""
    _device_capacity_or_skip()
    device = "cuda"
    M, N = 4, 256
    x = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    weight = torch.ones(N, device=device, dtype=torch.bfloat16)
    dout = torch.zeros_like(x)
    rstd = torch.ones(M, device=device, dtype=torch.float32)
    dx = torch.empty_like(x)
    with pytest.raises(RuntimeError, match="requires a config"):
        rmsnorm_bwd_tuned.fn(x, weight, dout, rstd, dx)
