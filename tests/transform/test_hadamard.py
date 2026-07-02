import math

import pytest
import torch

from quack.transform.hadamard import (
    hadamard_transform,
    hadamard_transform_fwd,
    hadamard_transform_ref,
)


torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

TOLERANCES = {
    torch.bfloat16: (1e-1, 2e-2),
    torch.float16: (3e-2, 2e-2),
    torch.float32: (1e-4, 1e-4),
}

POWER_OF_2_DIMS = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("N", POWER_OF_2_DIMS)
def test_hadamard_transform_correctness(N, dtype):
    torch.manual_seed(0)
    x = torch.randn(15, N, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    scale = 1.0 / math.sqrt(N)

    out = hadamard_transform(x, scale=scale)
    out_ref = hadamard_transform_ref(x_ref, scale=scale)

    dy = torch.randn_like(out)
    out.backward(dy)
    out_ref.backward(dy)

    assert out.shape == x.shape
    assert out.dtype == dtype
    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("N", [24, 137, 1000])
def test_hadamard_transform_non_power_of_two(N, dtype):
    torch.manual_seed(1)
    x = torch.randn(5, N, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    scale = 1.0 / math.sqrt(1 << (N - 1).bit_length())

    out = hadamard_transform(x, scale=scale)
    out_ref = hadamard_transform_ref(x_ref, scale=scale)

    dy = torch.randn_like(out)
    out.backward(dy)
    out_ref.backward(dy)

    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=rtol)


def test_hadamard_transform_compile():
    torch.manual_seed(3)
    dtype = torch.bfloat16
    x = torch.randn(3, 256, device="cuda", dtype=dtype)
    fn = torch.compile(hadamard_transform, fullgraph=True)

    out = fn(x, scale=0.5)
    out_ref = hadamard_transform_ref(x, scale=0.5)

    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


def test_hadamard_transform_empty_batch():
    x = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    out = hadamard_transform_fwd(x)
    assert out.shape == x.shape and out.numel() == 0
