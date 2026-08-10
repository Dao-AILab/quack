# Copyright (c) 2026, Tri Dao.

import importlib.util

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


def test_padded_rows_reuse_compiled_launcher(monkeypatch):
    n = 192
    config = RmsNormFwdConfig.for_forward(n, 16)
    assert rows_per_block(config) == 8

    compile_calls = 0
    compile_original = fly_rmsnorm.flyc.compile

    def count_compile(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return compile_original(*args, **kwargs)

    fly_rmsnorm._FWD_CACHE.clear()
    fly_rmsnorm._COMPILED_CALLABLES.clear()
    monkeypatch.setattr(fly_rmsnorm.flyc, "compile", count_compile)

    for m, padding in ((3, 8), (11, 16)):
        storage = torch.randn(m, n + padding, device=_DEVICE, dtype=torch.float16)
        x = storage[:, :n]
        assert x.stride() == (n + padding, 1)
        weight = torch.randn(n, device=_DEVICE, dtype=torch.float16)
        out, _, _ = fly_rmsnorm.rmsnorm_fwd(x, weight)
        expected, _, _ = _reference(x, weight)
        _assert_close(out, expected, x.dtype)

    assert compile_calls == 1
    assert len(fly_rmsnorm._FWD_CACHE) == 1


def test_invalid_inputs():
    x = torch.empty(2, 16, device=_DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="multiple of 8"):
        fly_rmsnorm.rmsnorm_fwd(torch.empty(2, 15, device=_DEVICE))
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
