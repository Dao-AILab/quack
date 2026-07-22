# Copyright (c) 2026, Dao-AILab.
"""Focused boundary and smoke tests for the experimental FlyDSL RMSNorm."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from quack._torch_library_op import torch_library_op
from quack.backends.flydsl import rmsnorm_provider
from quack.backends.flydsl.rmsnorm_config import (
    ATOMIC,
    TWO_STAGE,
    get_bwd_workspace_rows,
    select_bwd_config,
    select_fwd_config,
)
from quack.backends.protocol import Support
from quack.backends.target import Target

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024


def test_backend_neutral_custom_op_is_opaque_under_compile():
    call_log = []

    @torch_library_op(
        "quack_test_flydsl_boundary::copy",
        mutates_args={"out"},
        schema="(Tensor x, Tensor(a1!) out) -> ()",
    )
    def copy_impl(x, out):
        call_log.append(tuple(x.shape))
        out.copy_(x)

    @torch.compile(fullgraph=True)
    def compiled_copy(x):
        out = torch.empty_like(x)
        copy_impl(x, out)
        return out

    x = torch.randn(4, 8)
    torch.testing.assert_close(compiled_copy(x), x)
    assert call_log == [(4, 8)]


def test_opaque_custom_ops_preserve_autograd_under_compile():
    call_log = []

    @torch_library_op(
        "quack_test_flydsl_boundary::rmsnorm_fwd",
        mutates_args={"out", "rstd"},
        schema=("(Tensor x, Tensor weight, Tensor(a2!) out, Tensor(a3!) rstd, float eps) -> ()"),
    )
    def fwd_impl(x, weight, out, rstd, eps):
        call_log.append("forward")
        row_rstd = torch.rsqrt(x.float().square().mean(dim=-1) + eps)
        rstd.copy_(row_rstd)
        out.copy_((x.float() * row_rstd[:, None] * weight).to(x.dtype))

    @torch_library_op(
        "quack_test_flydsl_boundary::rmsnorm_bwd",
        mutates_args={"dx", "dweight"},
        schema=(
            "(Tensor x, Tensor weight, Tensor doutput, Tensor rstd, "
            "Tensor(a4!) dx, Tensor(a5!) dweight) -> ()"
        ),
    )
    def bwd_impl(x, weight, doutput, rstd, dx, dweight):
        call_log.append("backward")
        x_hat = x.float() * rstd[:, None]
        weighted_grad = doutput.float() * weight
        coefficient = (x_hat * weighted_grad).mean(dim=-1, keepdim=True)
        dx.copy_(((weighted_grad - x_hat * coefficient) * rstd[:, None]).to(x.dtype))
        dweight.copy_((doutput.float() * x_hat).sum(dim=0))

    class OpaqueRMSNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            out = torch.empty_like(x)
            rstd = torch.empty(x.shape[0], dtype=torch.float32)
            torch.ops.quack_test_flydsl_boundary.rmsnorm_fwd.default(
                x,
                weight,
                out,
                rstd,
                eps,
            )
            ctx.save_for_backward(x, weight, rstd)
            return out

        @staticmethod
        def backward(ctx, doutput):
            x, weight, rstd = ctx.saved_tensors
            dx = torch.empty_like(x)
            dweight = torch.empty_like(weight)
            torch.ops.quack_test_flydsl_boundary.rmsnorm_bwd.default(
                x,
                weight,
                doutput,
                rstd,
                dx,
                dweight,
            )
            return dx, dweight, None

    @torch.compile(fullgraph=True)
    def compiled_rmsnorm(x, weight):
        shape = x.shape
        return OpaqueRMSNorm.apply(x.reshape(-1, shape[-1]), weight, 1e-5).reshape(shape)

    x = torch.randn(2, 3, 17, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(17, dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    grad = torch.randn_like(x)

    out = compiled_rmsnorm(x, weight)
    dx, dweight = torch.autograd.grad(out, (x, weight), grad)
    out_ref = torch.nn.functional.rms_norm(x_ref, (17,), weight_ref, 1e-5)
    dx_ref, dweight_ref = torch.autograd.grad(out_ref, (x_ref, weight_ref), grad)

    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close(dx, dx_ref)
    torch.testing.assert_close(dweight, dweight_ref)
    assert call_log == ["forward", "backward"]


def test_package_and_flydsl_entry_imports_are_isolated():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import sys
import torch
torch.version.hip = "7.2"
torch.version.cuda = None
import quack
assert callable(quack.rmsnorm)
assert quack.rmsnorm.__module__ == "quack.rmsnorm_flydsl"
assert "quack.rmsnorm" not in sys.modules
assert "quack.backends.cute.rmsnorm_provider" not in sys.modules
assert "quack.dsl" not in sys.modules
assert "cutlass" not in sys.modules
assert "flydsl" not in sys.modules
from quack.rmsnorm_flydsl import rmsnorm as explicit_rmsnorm
assert explicit_rmsnorm is quack.rmsnorm
assert "quack.dsl" not in sys.modules
assert "cutlass" not in sys.modules
assert "flydsl" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True)


def test_cute_op_compatibility_reexport_is_preserved():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import os
import sys
import types
import torch

os.environ.pop("CUTE_DSL_PTXAS_PATH", None)
torch.version.hip = "7.2"
torch.version.cuda = None
for name in (
    "quack.dsl.cute_tensor_indexing",
    "quack.dsl.cute_tensor",
    "quack.dsl.mixed_constexpr_if",
):
    sys.modules[name] = types.ModuleType(name)

from quack._torch_library_op import torch_library_op
from quack.dsl import cute_op
from quack.dsl.torch_library_op import cute_op as compatibility_cute_op

assert cute_op is torch_library_op
assert compatibility_cute_op is torch_library_op
assert "cutlass" not in sys.modules
assert "flydsl" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True)


def test_registry_loads_only_the_selected_provider():
    code = """
import sys
from quack.backends.protocol import BackendProvider
from quack.backends.registry import REGISTRY
provider = REGISTRY.load_provider("rmsnorm", "amd")
assert provider.__name__ == "quack.backends.flydsl.rmsnorm_provider"
assert isinstance(provider, BackendProvider)
assert "quack.backends.cute.rmsnorm_provider" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_flydsl_has_install_hint(monkeypatch):
    import quack.rmsnorm_flydsl as module

    monkeypatch.setattr(module, "_BUILD_VENDOR", "amd")
    monkeypatch.setattr(module, "_FLYDSL_AVAILABLE", False)
    with pytest.raises(ModuleNotFoundError, match=r"quack-kernels\[rocm\]"):
        module.rmsnorm(torch.empty(1))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"has_weight": False}, "explicit weight"),
        ({"has_bias": True}, "support bias"),
        ({"has_residual": True}, "residual fusion"),
        ({"prenorm": True}, "prenorm"),
        ({"per_head": True}, "per-head"),
        ({"out_dtype": torch.float16}, "out_dtype"),
        ({"residual_dtype": torch.float32}, "residual_dtype"),
        ({"weight_offset": 1.0}, "weight_offset"),
    ],
)
def test_flydsl_provider_reports_unsupported_features(override, message):
    fields = {
        "has_weight": True,
        "has_bias": False,
        "has_residual": False,
        "prenorm": False,
        "per_head": False,
        "out_dtype": None,
        "residual_dtype": None,
        "weight_offset": 0.0,
        "input_dtype": torch.float16,
        "weight_dtype": torch.float32,
    }
    fields.update(override)
    decision = rmsnorm_provider.supports(
        SimpleNamespace(**fields),
        Target(vendor="amd", arch="gfx950", device_index=0),
    )
    assert isinstance(decision, Support)
    assert not decision
    assert message in decision.reason


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "input_dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_fp32_weight_and_supported_arch_contracts_without_hardware(arch, input_dtype):
    spec = SimpleNamespace(
        has_weight=True,
        has_bias=False,
        has_residual=False,
        prenorm=False,
        per_head=False,
        out_dtype=None,
        residual_dtype=None,
        weight_offset=0.0,
        input_dtype=input_dtype,
        weight_dtype=torch.float32,
    )
    target = Target(vendor="amd", arch=arch, device_index=0)
    assert rmsnorm_provider.supports(spec, target)
    assert (
        select_fwd_config(
            1024,
            4096,
            input_dtype,
            torch.float32,
            target,
            304 if arch == "gfx942" else 256,
        ).block_threads
        == 256
    )


def test_non_fp32_weight_has_precise_reason():
    spec = SimpleNamespace(
        has_weight=True,
        has_bias=False,
        has_residual=False,
        prenorm=False,
        per_head=False,
        out_dtype=None,
        residual_dtype=None,
        weight_offset=0.0,
        input_dtype=torch.float16,
        weight_dtype=torch.float16,
    )
    decision = rmsnorm_provider.supports(
        spec,
        Target(vendor="amd", arch="gfx942", device_index=0),
    )
    assert not decision
    assert "FP32 weight" in decision.reason


def test_unsupported_arch_has_precise_reason():
    spec = SimpleNamespace(
        has_weight=True,
        has_bias=False,
        has_residual=False,
        prenorm=False,
        per_head=False,
        out_dtype=None,
        residual_dtype=None,
        weight_offset=0.0,
        input_dtype=torch.float32,
        weight_dtype=torch.float32,
    )
    decision = rmsnorm_provider.supports(
        spec,
        Target(vendor="amd", arch="gfx1100", device_index=0),
    )
    assert not decision
    assert "gfx942 and gfx950" in decision.reason
    assert "gfx1100" in decision.reason


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_custom_ops_reject_compile_target_mismatch(monkeypatch, direction):
    import quack.rmsnorm_flydsl as module

    invocation_target = Target(vendor="amd", arch="gfx950", device_index=0)
    monkeypatch.setattr(module, "target_from_tensor", lambda tensor: invocation_target)
    monkeypatch.setattr(module, "_flydsl_compile_target", lambda: ("rocm", "gfx942"))

    x = torch.randn(2, 4)
    weight = torch.randn(4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="compile for 'gfx942'.*tensor is on 'gfx950'"):
        if direction == "forward":
            module._rmsnorm_flydsl_fwd._init_fn(
                x,
                weight,
                torch.empty_like(x),
                torch.empty(2, dtype=torch.float32),
                1e-6,
            )
        else:
            module._rmsnorm_flydsl_bwd._init_fn(
                x,
                weight,
                torch.randn_like(x),
                torch.ones(2, dtype=torch.float32),
                torch.empty_like(x),
                torch.empty_like(weight),
                torch.empty(0, dtype=torch.float32),
            )


@pytest.mark.parametrize("arch,cu_count", [("gfx942", 304), ("gfx950", 256)])
def test_backward_selector_contracts_without_hardware(arch, cu_count):
    target = Target(vendor="amd", arch=arch, device_index=0)
    atomic = select_bwd_config(
        32,
        4096,
        torch.bfloat16,
        torch.float32,
        target,
        cu_count,
    )
    staged = select_bwd_config(
        4096,
        4096,
        torch.bfloat16,
        torch.float32,
        target,
        cu_count,
    )
    assert atomic.path == ATOMIC
    assert atomic.num_programs == 0
    assert staged.path == TWO_STAGE
    workspace_rows = get_bwd_workspace_rows(4096, 4096, torch.bfloat16)
    assert 0 < staged.num_programs <= workspace_rows


def test_flydsl_custom_ops_have_noop_fake_implementations():
    from torch._subclasses.fake_tensor import FakeTensorMode

    import quack.rmsnorm_flydsl as module

    with FakeTensorMode():
        x = torch.empty(2, 17, device="cuda", dtype=torch.float16)
        weight = torch.empty(17, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        rstd = torch.empty(2, device="cuda", dtype=torch.float32)
        module._rmsnorm_flydsl_fwd._custom_op(x, weight, out, rstd, 1e-6)
        doutput = torch.empty_like(x)
        dx = torch.empty_like(x)
        dweight = torch.empty_like(weight)
        workspace = torch.empty(0, device="cuda", dtype=torch.float32)
        module._rmsnorm_flydsl_bwd._custom_op(
            x,
            weight,
            doutput,
            rstd,
            dx,
            dweight,
            workspace,
        )


@pytest.mark.parametrize("eps", [-1.0, float("inf"), float("-inf"), float("nan")])
def test_opaque_forward_rejects_invalid_eps_before_launch(eps):
    import quack.rmsnorm_flydsl as module

    x = torch.randn(2, 4)
    weight = torch.randn(4, dtype=torch.float32)
    with pytest.raises(ValueError, match="eps must be finite and nonnegative"):
        module._rmsnorm_flydsl_fwd._init_fn(
            x,
            weight,
            torch.empty_like(x),
            torch.empty(2, dtype=torch.float32),
            eps,
        )


def _flydsl_rocm_available() -> bool:
    return (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and importlib.util.find_spec("flydsl") is not None
    )


@pytest.fixture
def rocm_device():
    if not _flydsl_rocm_available():
        pytest.skip("requires ROCm PyTorch and FlyDSL")
    device = torch.device("cuda", torch.cuda.current_device())
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    if arch not in {"gfx942", "gfx950"}:
        pytest.skip(f"unsupported FlyDSL RMSNorm architecture: {arch}")
    expected_arch = os.environ.get("QUACK_EXPECTED_GFX")
    if expected_arch is not None:
        assert arch == expected_arch, f"runner exposed {arch}, expected {expected_arch}"
    return device


def _tolerances(dtype, flattened_rows):
    if dtype == torch.float32:
        return 8e-4, max(2e-3, 2e-4 * math.sqrt(flattened_rows))
    if dtype == torch.float16:
        return 3e-2, max(5e-2, 2e-3 * math.sqrt(flattened_rows))
    return 1e-1, max(2e-1, 5e-3 * math.sqrt(flattened_rows))


def _assert_forward_backward(shape, dtype, eps, use_compile, device):
    from quack.rmsnorm_flydsl import rmsnorm

    torch.manual_seed(0)
    n = shape[-1]
    flattened_rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(n, device=device, dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    fn = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm

    out = fn(x, weight, eps=eps)
    out_ref = torch.nn.functional.rms_norm(x_ref, (n,), weight_ref, eps)
    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)

    tensor_atol, weight_atol = _tolerances(dtype, flattened_rows)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, out_ref, atol=tensor_atol, rtol=2e-2)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=tensor_atol, rtol=2e-2)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=weight_atol, rtol=3e-2)


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize(
    "shape,dtype,eps",
    [
        ((1, 1), torch.float16, 0.0),
        ((7, 65), torch.float32, 1e-6),
        ((2, 3, 257), torch.float16, 3e-5),
        ((5, 129), torch.bfloat16, 1e-2),
        ((2048, 1024), torch.bfloat16, 1e-5),
    ],
)
def test_rmsnorm_flydsl_forward_backward(rocm_device, use_compile, shape, dtype, eps):
    _assert_forward_backward(shape, dtype, eps, use_compile, rocm_device)


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("eps", [-1.0, float("inf"), float("-inf"), float("nan")])
def test_invalid_eps_fails_before_kernel_launch(rocm_device, use_compile, eps):
    from quack.rmsnorm_flydsl import rmsnorm

    x = torch.randn(2, 17, device=rocm_device, dtype=torch.float16)
    weight = torch.randn(17, device=rocm_device, dtype=torch.float32)
    fn = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
    with pytest.raises(ValueError, match="eps must be finite and nonnegative"):
        fn(x, weight, eps=eps)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_weight", "explicit weight"),
        ("bias", "support bias"),
        ("residual", "residual fusion"),
        ("prenorm", "prenorm"),
        ("per_head", "per-head"),
        ("out_dtype", "out_dtype"),
        ("residual_dtype", "residual_dtype"),
        ("weight_offset", "weight_offset"),
        ("input_dtype", "input dtype"),
        ("weight_dtype", "FP32 weight"),
    ],
)
def test_public_api_reports_unsupported_features(rocm_device, case, message):
    from quack.rmsnorm_flydsl import rmsnorm

    x = torch.randn(2, 17, device=rocm_device, dtype=torch.float16)
    weight = torch.randn(17, device=rocm_device, dtype=torch.float32)
    kwargs = {"weight": weight}
    if case == "missing_weight":
        kwargs["weight"] = None
    elif case == "bias":
        kwargs["bias"] = torch.randn_like(weight)
    elif case == "residual":
        kwargs["residual"] = torch.randn_like(x)
    elif case == "prenorm":
        kwargs["prenorm"] = True
    elif case == "per_head":
        kwargs["weight"] = torch.randn(2, 17, device=rocm_device, dtype=torch.float32)
    elif case == "out_dtype":
        kwargs["out_dtype"] = torch.float32
    elif case == "residual_dtype":
        kwargs["residual_dtype"] = torch.float32
    elif case == "weight_offset":
        kwargs["weight_offset"] = 1.0
    elif case == "input_dtype":
        x = x.double()
    elif case == "weight_dtype":
        kwargs["weight"] = weight.half()

    with pytest.raises(NotImplementedError, match=message):
        rmsnorm(x, **kwargs)


def test_runtime_eps_and_compiled_cache_reuse(rocm_device, monkeypatch):
    from quack.backends.flydsl import rmsnorm_bwd_kernel, rmsnorm_kernel
    from quack.rmsnorm_flydsl import rmsnorm

    rmsnorm_kernel._FWD_CACHE.clear()
    rmsnorm_bwd_kernel._BWD_CACHE.clear()
    counts = {"forward": 0, "backward": 0}
    original_fwd = rmsnorm_kernel.build_rmsnorm_fwd_module
    original_bwd = rmsnorm_bwd_kernel.build_rmsnorm_bwd_atomic_module

    def counted_fwd(*args, **kwargs):
        counts["forward"] += 1
        return original_fwd(*args, **kwargs)

    def counted_bwd(*args, **kwargs):
        counts["backward"] += 1
        return original_bwd(*args, **kwargs)

    monkeypatch.setattr(rmsnorm_kernel, "build_rmsnorm_fwd_module", counted_fwd)
    monkeypatch.setattr(rmsnorm_bwd_kernel, "build_rmsnorm_bwd_atomic_module", counted_bwd)
    compiled = torch.compile(rmsnorm, fullgraph=True)
    weight = torch.randn(65, device=rocm_device, dtype=torch.float32, requires_grad=True)
    for m, eps in ((3, 1e-6), (11, 1e-2)):
        x = torch.randn(
            m,
            65,
            device=rocm_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        out = compiled(x, weight, eps=eps)
        expected = torch.nn.functional.rms_norm(x.detach(), (65,), weight.detach(), eps)
        torch.testing.assert_close(out, expected, atol=8e-4, rtol=2e-2)
        out.sum().backward()
        weight.grad = None

    assert counts == {"forward": 1, "backward": 1}


def test_non_default_streams_use_per_call_staged_workspaces(rocm_device):
    from quack.rmsnorm_flydsl import rmsnorm

    torch.manual_seed(0)
    streams = [torch.cuda.Stream(device=rocm_device) for _ in range(2)]
    cases = []
    for seed, stream in enumerate(streams):
        torch.manual_seed(seed)
        x = torch.randn(
            512,
            257,
            device=rocm_device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.randn(
            257,
            device=rocm_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        grad = torch.randn_like(x)
        checks_host = torch.empty(3, dtype=torch.bool, pin_memory=True)
        cases.append((stream, x, weight, grad, checks_host))

    # Compile both paths and finish all input initialization before deliberately
    # blocking the default stream.
    warm_x = cases[0][1].detach().clone().requires_grad_()
    warm_weight = cases[0][2].detach().clone().requires_grad_()
    warm_out = rmsnorm(warm_x, warm_weight, eps=1e-5)
    warm_dx, warm_dweight = torch.autograd.grad(
        warm_out,
        (warm_x, warm_weight),
        cases[0][3],
    )
    torch.cuda.synchronize(rocm_device)

    default_stream = torch.cuda.current_stream(rocm_device)
    default_finished = torch.cuda.Event()
    torch.cuda._sleep(int(2e9))
    default_finished.record(default_stream)

    completions = []
    for stream, x, weight, grad, checks_host in cases:
        with torch.cuda.stream(stream):
            out = rmsnorm(x, weight, eps=1e-5)
            dx, dweight = torch.autograd.grad(out, (x, weight), grad)
            out_ref = torch.nn.functional.rms_norm(x, (257,), weight, 1e-5)
            dx_ref, dweight_ref = torch.autograd.grad(out_ref, (x, weight), grad)
            checks = torch.stack(
                (
                    torch.isclose(out, out_ref, atol=1e-1, rtol=2e-2).all(),
                    torch.isclose(dx, dx_ref, atol=1e-1, rtol=2e-2).all(),
                    torch.isclose(dweight, dweight_ref, atol=3e-1, rtol=3e-2).all(),
                )
            )
            checks_host.copy_(checks, non_blocking=True)
            done = torch.cuda.Event()
            done.record(stream)
        completions.append(done)

    try:
        for done in completions:
            done.synchronize()
        assert not default_finished.query(), "default-stream blocker ended before validation"
        for *_, checks_host in cases:
            assert checks_host.tolist() == [True, True, True]
    finally:
        default_finished.synchronize()

    # Keep warmup allocations live until the blocked default stream drains, so
    # test outputs cannot accidentally reuse already-correct storage.
    assert warm_dx is not None and warm_dweight is not None
