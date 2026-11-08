"""Numerical acceptance coverage for the native Windows SM120 path."""

import sys

import pytest
import torch

from quack.gemm_interface import gemm
from quack.rmsnorm import rmsnorm
from quack.softmax import softmax

pytestmark = pytest.mark.skipif(
    sys.platform != "win32"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0),
    reason="Native Windows SM120 acceptance",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [128, 1024])
@pytest.mark.parametrize("operation", ["softmax", "rmsnorm"])
def test_reduction_forward_and_backward(dtype, width, operation):
    torch.manual_seed(1234)
    x = torch.randn(8, width, device="cuda", dtype=dtype, requires_grad=True)
    reference_x = x.detach().float().requires_grad_()
    if operation == "softmax":
        actual = softmax(x)
        reference = reference_x.softmax(-1)
    else:
        actual = rmsnorm(x)
        reference = reference_x * torch.rsqrt(
            reference_x.square().mean(-1, keepdim=True) + 1e-6
        )
    tolerance = 1e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(
        actual.float(), reference, atol=tolerance, rtol=tolerance
    )
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    reference.backward(gradient.float())
    torch.testing.assert_close(
        x.grad.float(), reference_x.grad, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_against_fp32(dtype):
    torch.manual_seed(1234)
    a = torch.randn(128, 128, device="cuda", dtype=dtype) / 128**0.5
    b = torch.randn(128, 128, device="cuda", dtype=dtype) / 128**0.5
    actual = gemm(a, b, tuned=False)
    torch.testing.assert_close(
        actual.float(), a.float() @ b.float(), atol=1e-3, rtol=1e-2
    )
