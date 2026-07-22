# Copyright (c) 2026, Dao-AILab.
"""Focused tests for public CUDA/ROCm RMSNorm binding."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_public_rocm_binding_is_import_isolated():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import sys
import torch
if torch.version.hip is None:
    raise SystemExit(0)
def unexpected_device_query(*args, **kwargs):
    raise AssertionError("exact target discovery ran while binding the provider")
torch.cuda.get_device_properties = unexpected_device_query
import quack
public_rmsnorm = quack.rmsnorm
assert public_rmsnorm.__module__ == "quack.rmsnorm_flydsl"
assert "quack.backends.flydsl.rmsnorm_provider" in sys.modules
assert "quack.backends.cute.rmsnorm_provider" not in sys.modules
assert "quack.dsl" not in sys.modules
assert "cutlass" not in sys.modules
assert "flydsl" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True)


def test_cuda_initialization_exports_and_direct_imports():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import importlib
import os
import sys
import types
import torch

torch.version.hip = None
torch.version.cuda = "13.0"

patch_calls = []
dsl = types.ModuleType("quack.dsl")
dsl.__path__ = []
ptxas = types.ModuleType("quack.dsl.cute_dsl_ptxas")
ptxas.patch = lambda: patch_calls.append("patched")
dsl.cute_dsl_ptxas = ptxas
sys.modules["quack.dsl"] = dsl
sys.modules["quack.dsl.cute_dsl_ptxas"] = ptxas

rmsnorm = lambda *args, **kwargs: "rmsnorm"
softmax = lambda *args, **kwargs: "softmax"
cross_entropy = lambda *args, **kwargs: "cross_entropy"
class RoundingMode:
    pass

legacy_rmsnorm = types.ModuleType("quack.rmsnorm")
legacy_rmsnorm.rmsnorm = rmsnorm
legacy_rmsnorm.rmsnorm_fwd = object()
sys.modules["quack.rmsnorm"] = legacy_rmsnorm

for module_name, attr_name, value in (
    ("quack.softmax", "softmax", softmax),
    ("quack.cross_entropy", "cross_entropy", cross_entropy),
    ("quack.rounding", "RoundingMode", RoundingMode),
):
    module = types.ModuleType(module_name)
    setattr(module, attr_name, value)
    sys.modules[module_name] = module

os.environ["CUTE_DSL_PTXAS_PATH"] = "/tmp/fake-ptxas"
import quack
assert patch_calls == ["patched"]
assert quack.rmsnorm is rmsnorm and callable(quack.rmsnorm)
assert quack.softmax is softmax and callable(quack.softmax)
assert quack.cross_entropy is cross_entropy and callable(quack.cross_entropy)
assert quack.RoundingMode is RoundingMode

# Direct submodule imports must not replace callable package exports.
for module_name in ("quack.rmsnorm", "quack.softmax", "quack.cross_entropy"):
    importlib.import_module(module_name)
from quack.rmsnorm import rmsnorm_fwd
from quack import cross_entropy as public_cross_entropy
from quack import rmsnorm as public_rmsnorm
from quack import softmax as public_softmax
assert public_rmsnorm is rmsnorm
assert not isinstance(public_rmsnorm, types.ModuleType)
assert rmsnorm_fwd is legacy_rmsnorm.rmsnorm_fwd
assert public_softmax is softmax
assert public_cross_entropy is cross_entropy
assert "flydsl" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True)


def test_provider_binding_failures_are_not_swallowed():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import sys
import types
import torch
torch.version.hip = "7.2"
torch.version.cuda = None
class ProviderLoadFailure(RuntimeError):
    pass
provider = types.ModuleType("quack.backends.flydsl.rmsnorm_provider")
provider.is_available = lambda: True
provider.supports = lambda spec, target: True
def fail():
    raise ProviderLoadFailure("provider load failed")
provider.load_op = fail
sys.modules["quack.backends.flydsl.rmsnorm_provider"] = provider
import quack
try:
    quack.rmsnorm
except ProviderLoadFailure as exc:
    assert str(exc) == "provider load failed"
else:
    raise AssertionError("provider load failure was swallowed")
"""
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True)


@pytest.fixture
def rocm_device():
    available = (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and importlib.util.find_spec("flydsl") is not None
    )
    if not available:
        pytest.skip("requires ROCm PyTorch and FlyDSL")
    device = torch.device("cuda", torch.cuda.current_device())
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    if arch not in {"gfx942", "gfx950"}:
        pytest.skip(f"unsupported FlyDSL RMSNorm architecture: {arch}")
    return device


def test_public_rocm_fullgraph_and_diagnostics(rocm_device):
    from quack import rmsnorm as public_rmsnorm
    from quack.rmsnorm_flydsl import rmsnorm as explicit_rmsnorm

    assert public_rmsnorm is explicit_rmsnorm

    x = torch.randn(3, 65, device=rocm_device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(
        65,
        device=rocm_device,
        dtype=torch.float32,
        requires_grad=True,
    )
    out = torch.compile(public_rmsnorm, fullgraph=True)(x, weight, eps=3e-5)
    grad = torch.randn_like(out)
    dx, dweight = torch.autograd.grad(out, (x, weight), grad)
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    out_ref = torch.nn.functional.rms_norm(x_ref, (65,), weight_ref, 3e-5)
    dx_ref, dweight_ref = torch.autograd.grad(out_ref, (x_ref, weight_ref), grad)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(dx, dx_ref, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(dweight, dweight_ref, atol=6e-2, rtol=3e-2)
    with pytest.raises(NotImplementedError, match="bias"):
        public_rmsnorm(x, weight, bias=torch.zeros_like(weight))


@pytest.mark.skipif(
    torch.version.hip is not None or not torch.cuda.is_available(),
    reason="requires CUDA PyTorch",
)
def test_public_cuda_preserves_legacy_behavior():
    from quack.rmsnorm import rmsnorm as direct_rmsnorm
    from quack import rmsnorm as public_rmsnorm

    assert "flydsl" not in sys.modules
    assert public_rmsnorm is direct_rmsnorm
    x = torch.randn(4, 65, device="cuda", dtype=torch.float16)
    weight = torch.randn(65, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(
        public_rmsnorm(x, weight, eps=1e-5),
        direct_rmsnorm(x, weight, eps=1e-5),
    )
