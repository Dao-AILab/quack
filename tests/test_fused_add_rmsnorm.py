# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import pytest
import torch

from quack.fused_add_rmsnorm import fused_add_rmsnorm, fused_add_rmsnorm_ref
from quack.rmsnorm import rmsnorm_ref


_compiled_add_cache: dict[torch.dtype, callable] = {}
_compiled_rms_cache: dict[tuple[torch.dtype, torch.dtype], callable] = {}


def _get_compiled_add(dtype: torch.dtype):
    fn = _compiled_add_cache.get(dtype)
    if fn is None:

        def add_fn(a, b):
            return a + b

        fn = torch.compile(add_fn, fullgraph=True, dynamic=True)
        _compiled_add_cache[dtype] = fn
    return fn


def _get_compiled_rmsnorm(x_dtype: torch.dtype, w_dtype: torch.dtype):
    key = (x_dtype, w_dtype)
    fn = _compiled_rms_cache.get(key)
    if fn is None:

        def rms_fn(x, w, eps):
            return rmsnorm_ref(x, w, eps=eps)

        fn = torch.compile(rms_fn, fullgraph=True, dynamic=True)
        _compiled_rms_cache[key] = fn
    return fn


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "N", [192, 256, 512, 760, 1024, 1128, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
)
@pytest.mark.parametrize("M", [1, 37, 199, 8 * 1024])
@pytest.mark.parametrize("use_compile", [False, True])
def test_fused_add_rmsnorm_forward_backward(M, N, input_dtype, eps, use_compile):
    """Test fused add+rmsnorm forward pass against reference implementation."""
    total_elements = M * N
    if total_elements >= 512 * 1024 * 1024:
        pytest.skip("Skipping extremely large configuration to avoid OOM")
    if input_dtype == torch.float32 and N >= 32 * 1024:
        pytest.skip("Skipping large float32 configuration pending kernel tuning")
    if N >= 256 * 1024 and input_dtype == torch.float32 and M >= 8 * 1024:
        pytest.skip("Skipping large tensor test for float32 to avoid OOM")
    device = "cuda"
    if input_dtype == torch.bfloat16:
        atol = 1.25e-1
        rtol = 5e-3
    elif input_dtype == torch.float16:
        atol = 1e-2
        rtol = 1e-3
    else:
        atol = 1e-4
        rtol = 1e-3
    torch.random.manual_seed(0)
    residual = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
    hidden_states = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
    weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)
    residual_ref = residual.detach().clone().requires_grad_()
    hidden_states_ref = hidden_states.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()

    if use_compile:
        compiled_add = _get_compiled_add(input_dtype)
        compiled_rms = _get_compiled_rmsnorm(input_dtype, torch.float32)
        updated = compiled_add(residual, hidden_states)
        updated_ref = compiled_add(residual_ref, hidden_states_ref)
        out = fused_add_rmsnorm(residual, hidden_states, weight, eps=eps)
        out_ref = compiled_rms(updated_ref, weight_ref, eps)
    else:
        out = fused_add_rmsnorm(residual, hidden_states, weight, eps=eps)
        out_ref = fused_add_rmsnorm_ref(residual_ref, hidden_states_ref, weight_ref, eps=eps)
        updated = residual + hidden_states
        updated_ref = residual_ref + hidden_states_ref

    assert out.shape == (M, N)
    assert updated.shape == (M, N)
    assert out.dtype == input_dtype
    assert updated.dtype == input_dtype

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(updated, updated_ref, atol=atol, rtol=rtol)

    if N > 128 * 1024 and input_dtype == torch.float32:
        return
    grad_out = torch.randn_like(out_ref)
    grad_updated = torch.randn_like(updated_ref)

    grads = torch.autograd.grad(
        (out, updated),
        (residual, hidden_states, weight),
        (grad_out, grad_updated),
        retain_graph=False,
        allow_unused=False,
    )
    grads_ref = torch.autograd.grad(
        (out_ref, updated_ref),
        (residual_ref, hidden_states_ref, weight_ref),
        (grad_out, grad_updated),
        retain_graph=False,
        allow_unused=False,
    )

    for grad, grad_ref in zip(grads, grads_ref):
        torch.testing.assert_close(grad, grad_ref, atol=atol, rtol=rtol)
