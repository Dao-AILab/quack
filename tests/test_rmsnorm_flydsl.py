# Copyright (c) 2026, Tri Dao.

import importlib.util
import sys

import pytest
import torch

_HAS_ROCM_GPU = torch.version.hip is not None and torch.cuda.is_available()
_DEVICE = (
    torch.device("cuda", torch.cuda.current_device()) if _HAS_ROCM_GPU else torch.device("cuda")
)
_ARCH = (
    torch.cuda.get_device_properties(_DEVICE).gcnArchName.split(":", 1)[0]
    if _HAS_ROCM_GPU
    else None
)
_HAS_FLYDSL = importlib.util.find_spec("flydsl") is not None
_CAN_RUN = _HAS_ROCM_GPU and _ARCH == "gfx950" and _HAS_FLYDSL

if not _HAS_ROCM_GPU:
    _SKIP_REASON = "requires a ROCm GPU"
elif _ARCH != "gfx950":
    _SKIP_REASON = "FlyDSL RMSNorm currently requires gfx950"
elif not _HAS_FLYDSL:
    _SKIP_REASON = "requires flydsl"
else:
    _SKIP_REASON = ""
pytestmark = pytest.mark.skipif(not _CAN_RUN, reason=_SKIP_REASON)

if _CAN_RUN:
    import quack.flydsl_runtime as fly_runtime
    import quack.rmsnorm_flydsl as fly_rmsnorm
    from quack.rmsnorm_flydsl_config import WAVE_SIZE, RmsNormFwdConfig, rows_per_block

_ATOL = {
    torch.float16: 2e-3,
    torch.bfloat16: 2e-2,
    torch.float32: 2e-4,
}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _reference(x, weight=None, bias=None, residual=None, eps=1e-6):
    combined = x.float()
    if residual is not None:
        combined = combined + residual.float()
    rstd = torch.rsqrt(combined.square().mean(dim=-1) + eps)
    out = combined * rstd.unsqueeze(-1)
    if weight is not None:
        out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out, combined, rstd


def _assert_close(actual, expected, input_dtype):
    torch.testing.assert_close(
        actual,
        expected.to(actual.dtype),
        atol=_ATOL[input_dtype],
        rtol=2e-3,
    )


@pytest.mark.parametrize(
    "m,n,dtype,weight_dtype",
    [
        pytest.param(1, 8, torch.float32, torch.float32, id="short-fp32"),
        pytest.param(2, 12, torch.float32, torch.float32, id="fp32-four-element-alignment"),
        pytest.param(3, 192, torch.float16, torch.float16, id="padded-fp16"),
        pytest.param(5, 760, torch.bfloat16, torch.float32, id="predicated-mixed-weight"),
        pytest.param(2, 4096, torch.bfloat16, None, id="cross-wave"),
        pytest.param(1, 65536, torch.bfloat16, torch.float32, id="wide-reload"),
        pytest.param(1, 65544, torch.bfloat16, torch.float32, id="wide-predicated-reload"),
    ],
)
def test_plain_forward(m, n, dtype, weight_dtype):
    x = torch.randn(m, n, device=_DEVICE, dtype=dtype)
    weight = (
        torch.randn(n, device=_DEVICE, dtype=weight_dtype) if weight_dtype is not None else None
    )

    out, residual_out, rstd = fly_rmsnorm.rmsnorm_fwd(x, weight)
    expected, _, _ = _reference(x, weight)

    assert out.shape == x.shape and out.dtype == x.dtype
    assert residual_out is x
    assert rstd is None
    _assert_close(out, expected, dtype)

    if n == 4096:
        config = RmsNormFwdConfig.for_forward(n, x.element_size() * 8)
        assert config.num_threads > WAVE_SIZE
    if n >= 65536:
        config = RmsNormFwdConfig.for_forward(n, x.element_size() * 8)
        assert config.reload_from_gmem


def test_bias_residual_and_aux_output_semantics():
    shape = (2, 3, 4, 64)
    x = torch.randn(shape, device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(shape[-2:], device=_DEVICE, dtype=torch.float32)
    bias = torch.randn(shape[-2:], device=_DEVICE, dtype=torch.float32)
    residual = torch.randn(shape, device=_DEVICE, dtype=torch.float16)

    out, residual_out, rstd = fly_rmsnorm.rmsnorm_fwd(
        x,
        weight,
        bias=bias,
        residual=residual,
        out_dtype=torch.float32,
        residual_dtype=torch.float32,
        store_rstd=True,
    )
    expected, combined, expected_rstd = _reference(x, weight, bias, residual)

    assert out.shape == x.shape and out.dtype == torch.float32
    assert residual_out.shape == x.shape and residual_out.dtype == torch.float32
    assert rstd.shape == x.shape[:-1] and rstd.dtype == torch.float32
    _assert_close(out, expected, x.dtype)
    torch.testing.assert_close(residual_out, combined, atol=0, rtol=0)
    _assert_close(rstd, expected_rstd, x.dtype)


def test_residual_dtype_without_residual_forces_a_fresh_output():
    x = torch.randn((3, 64), device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(64, device=_DEVICE, dtype=torch.float32)

    out, residual_out, rstd = fly_rmsnorm.rmsnorm_fwd(
        x,
        weight,
        residual_dtype=torch.float32,
    )
    expected, combined, _ = _reference(x, weight)

    assert residual_out.dtype == torch.float32
    assert residual_out.data_ptr() != x.data_ptr()
    assert rstd is None
    _assert_close(out, expected, x.dtype)
    torch.testing.assert_close(residual_out, combined, atol=0, rtol=0)


def test_padded_rows_reuse_compiled_launcher(monkeypatch):
    n = 192
    config = RmsNormFwdConfig.for_forward(n, 16)
    assert rows_per_block(config) == 8

    compile_calls = 0
    compile_original = fly_rmsnorm.flyc.compile
    validate_calls = 0
    validate_original = fly_runtime.validate_arch

    def count_compile(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return compile_original(*args, **kwargs)

    def count_validate(*args, **kwargs):
        nonlocal validate_calls
        validate_calls += 1
        return validate_original(*args, **kwargs)

    fly_rmsnorm._FWD_CACHE.clear()
    monkeypatch.setattr(fly_rmsnorm.flyc, "compile", count_compile)
    # SpecializationCache resolves validate_arch from its own module globals;
    # compile_context is bound into the kernel module by its from-import.
    monkeypatch.setattr(fly_runtime, "validate_arch", count_validate)
    compile_context = [("gfx950", "context-a")]
    monkeypatch.setattr(fly_rmsnorm, "compile_context", lambda _device: compile_context[0])

    for m, padding in ((3, 8), (11, 16)):
        storage = torch.randn(m, n + padding, device=_DEVICE, dtype=torch.float16)
        x = storage[:, :n]
        assert x.stride() == (n + padding, 1)
        weight = torch.randn(n, device=_DEVICE, dtype=torch.float16)
        out, _, _ = fly_rmsnorm.rmsnorm_fwd(x, weight)
        expected, _, _ = _reference(x, weight)
        _assert_close(out, expected, x.dtype)

    assert compile_calls == 1
    assert validate_calls == 1
    assert len(fly_rmsnorm._FWD_CACHE) == 1

    compile_context[0] = ("gfx950", "context-b")
    x = torch.randn(3, n, device=_DEVICE, dtype=torch.float16)
    weight = torch.randn(n, device=_DEVICE, dtype=torch.float16)
    fly_rmsnorm.rmsnorm_fwd(x, weight)
    assert compile_calls == 2
    assert validate_calls == 2
    assert len(fly_rmsnorm._FWD_CACHE) == 2


def test_compile_context_tracks_target_environment(monkeypatch):
    monkeypatch.setattr(fly_runtime, "validate_arch", lambda *_args, **_kwargs: "gfx950")
    monkeypatch.delenv("ARCH", raising=False)
    first = fly_rmsnorm.compile_context(_DEVICE)

    monkeypatch.setenv("ARCH", "gfx950:context-change")
    second = fly_rmsnorm.compile_context(_DEVICE)

    assert second != first


def test_launch_follows_the_callers_stream():
    n = 4096
    x = torch.randn(2, n, device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(n, device=_DEVICE, dtype=torch.float32)
    expected, _, _ = _reference(x, weight)

    default_stream = torch.cuda.current_stream(_DEVICE)
    assert fly_rmsnorm.current_raw_stream(_DEVICE) == default_stream.cuda_stream
    assert fly_rmsnorm.current_raw_stream(torch.device("cuda")) == default_stream.cuda_stream

    # A raw stream accessor that ignored the ambient context would silently
    # place every launch on the default stream, racing the caller's ordering.
    side = torch.cuda.Stream(device=_DEVICE)
    with torch.cuda.stream(side):
        assert fly_rmsnorm.current_raw_stream(_DEVICE) == side.cuda_stream
        out, _, _ = fly_rmsnorm.rmsnorm_fwd(x, weight)
    side.synchronize()

    assert fly_rmsnorm.current_raw_stream(_DEVICE) == default_stream.cuda_stream
    _assert_close(out, expected, x.dtype)


def test_unaligned_padded_rows_are_copied_before_vector_access():
    n = 192
    storage = torch.randn(4, 196, device=_DEVICE, dtype=torch.float16)
    x = storage[:, :n]
    assert (x.stride(0) * x.element_size()) % 16 == 8

    packed = fly_rmsnorm.packed_rows(x)
    assert packed.is_contiguous()
    assert packed.data_ptr() != x.data_ptr()

    weight = torch.randn(n, device=_DEVICE, dtype=torch.float16)
    out, _, _ = fly_rmsnorm.rmsnorm_fwd(x, weight)
    expected, _, _ = _reference(x, weight)
    _assert_close(out, expected, x.dtype)


def test_misaligned_contiguous_storage_offsets_force_allocation():
    n = 192
    x_storage = torch.randn(2 * n + 1, device=_DEVICE, dtype=torch.float16)
    weight_storage = torch.randn(n + 1, device=_DEVICE, dtype=torch.float16)
    x = x_storage[1:].view(2, n)
    weight = weight_storage[1:]
    assert x.is_contiguous() and weight.is_contiguous()
    assert x.data_ptr() % 16 == weight.data_ptr() % 16 == 2

    packed_x = fly_rmsnorm.packed_rows(x)
    packed_weight = fly_rmsnorm.packed_rows(weight)
    assert packed_x.data_ptr() != x.data_ptr()
    assert packed_weight.data_ptr() != weight.data_ptr()
    assert packed_x.data_ptr() % 16 == packed_weight.data_ptr() % 16 == 0

    out, _, _ = fly_rmsnorm.rmsnorm_fwd(x, weight)
    expected, _, _ = _reference(x, weight)
    _assert_close(out, expected, x.dtype)


def test_empty_rows_return_without_compiling(monkeypatch):
    x = torch.empty(0, 192, device=_DEVICE, dtype=torch.float16)
    weight = torch.ones(192, device=_DEVICE, dtype=torch.float32)

    def forbid_compile(*_args, **_kwargs):
        raise AssertionError("empty rows attempted to compile a kernel")

    monkeypatch.setattr(fly_rmsnorm.flyc, "compile", forbid_compile)
    out, residual_out, rstd = fly_rmsnorm.rmsnorm_fwd(
        x,
        weight,
        out_dtype=torch.float32,
        residual_dtype=torch.float32,
        store_rstd=True,
    )

    assert out.shape == x.shape and out.dtype == torch.float32
    assert residual_out.shape == x.shape and residual_out.dtype == torch.float32
    assert residual_out is not x
    assert rstd.shape == x.shape[:-1] and rstd.dtype == torch.float32


def test_rocm_cute_exports_fail_without_submodule_fallback():
    with pytest.raises(ImportError, match="requires the CUDA/CuTe backend"):
        __import__("quack", fromlist=("rmsnorm",))

    assert "quack.rmsnorm" not in sys.modules
    assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)


def test_async_compile_is_rejected_before_loading_cute():
    from quack.testing import pytest_plugin

    class AsyncCompileConfig:
        @staticmethod
        def getoption(*_args, **_kwargs):
            return 1

    with pytest.raises(pytest.UsageError, match="unavailable on ROCm"):
        pytest_plugin.pytest_configure(AsyncCompileConfig())

    assert "quack.cache.jit" not in sys.modules
    assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)


def test_invalid_inputs():
    x = torch.empty(2, 16, device=_DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="multiple of 8"):
        fly_rmsnorm.rmsnorm_fwd(torch.empty(2, 15, device=_DEVICE, dtype=torch.float16))
    with pytest.raises(ValueError, match="weight shape"):
        fly_rmsnorm.rmsnorm_fwd(x, torch.empty(8, device=_DEVICE))
    with pytest.raises(ValueError, match="residual shape"):
        fly_rmsnorm.rmsnorm_fwd(x, residual=torch.empty(1, 16, device=_DEVICE))
    with pytest.raises(TypeError, match="x dtype"):
        fly_rmsnorm.rmsnorm_fwd(x.double())
    with pytest.raises(ValueError, match="finite and positive"):
        fly_rmsnorm.rmsnorm_fwd(x, eps=0)
    with pytest.raises(ValueError, match="ROCm device"):
        fly_rmsnorm.rmsnorm_fwd(x.cpu())
