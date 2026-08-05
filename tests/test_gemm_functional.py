# Copyright (c) 2026, Tri Dao.
"""Functional facade ops (quack::gemm, quack::gemm_add, quack::gemm_epi_f):
allocation inside the op, real fakes — one graph-insertable node per call.
Parity vs the python wrappers, and fake-tensor shape/dtype correctness (the
property FX passes rely on)."""

import math

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from quack.gemm_interface import gemm, gemm_add

requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="needs sm90",
)


def _abc(m=512, k=512, n=512, dt=torch.bfloat16, c_dtype=None):
    torch.manual_seed(0)
    A = torch.randn(m, k, dtype=dt, device="cuda") / math.sqrt(k)
    B = torch.randn(k, n, dtype=dt, device="cuda") / math.sqrt(k)
    C = torch.randn(m, n, dtype=c_dtype or dt, device="cuda")
    return A, B, C


@requires_sm90
def test_gemm_add_functional_parity():
    A, B, C = _abc()
    bias = torch.randn(B.shape[-1], dtype=A.dtype, device="cuda")
    out = torch.ops.quack.gemm_add(A, B, C, bias=bias, tuned=False)
    ref = gemm_add(A, B, C, bias=bias, tuned=False)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@requires_sm90
def test_gemm_add_functional_fp32_residual():
    A, B, C = _abc(c_dtype=torch.float32)
    out = torch.ops.quack.gemm_add(A, B, C, out_dtype=torch.float32, tuned=False)
    assert out.dtype == torch.float32
    ref = gemm_add(A, B, C, out_dtype=torch.float32, tuned=False)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@requires_sm90
def test_gemm_functional_parity():
    A, B, _ = _abc()
    out = torch.ops.quack.gemm(A, B, tuned=False)
    ref = gemm(A, B, tuned=False)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@requires_sm90
def test_functional_fakes():
    """Fake kernels must report the shapes/dtypes FX passes will rely on,
    without touching CUDA."""
    with FakeTensorMode():
        A = torch.empty(64, 32, dtype=torch.bfloat16, device="cuda")
        B = torch.empty(32, 128, dtype=torch.bfloat16, device="cuda")
        C = torch.empty(64, 128, dtype=torch.float32, device="cuda")
        out = torch.ops.quack.gemm_add(A, B, C, out_dtype=torch.float32)
        assert out.shape == (64, 128) and out.dtype == torch.float32
        out2 = torch.ops.quack.gemm(A, B)
        assert out2.shape == (64, 128) and out2.dtype == torch.bfloat16
