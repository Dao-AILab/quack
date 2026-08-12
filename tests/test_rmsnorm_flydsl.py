# Copyright (c) 2026, Tri Dao.

"""The rmsnorm_fwd forward contract, held against whichever backend this machine has.

CuTe-DSL and FlyDSL implement the same function -- same arguments, same returned
triple, same aliasing (residual_out is x when nothing was accumulated, rstd is
None unless asked for). Most of this file therefore runs on both, and each test
carries its tests/test_rmsnorm.py counterpart's name. Tests marked FlyDSL-only
reach into host plumbing or shapes that the CuTe backend does not have.
"""

import pytest
import torch
from flydsl_env import BACKEND as _BACKEND
from flydsl_env import CAN_RUN as _CAN_RUN
from flydsl_env import DEVICE as _DEVICE
from flydsl_env import IS_FLYDSL as _IS_FLYDSL
from flydsl_env import SKIP_REASON as _SKIP_REASON

pytestmark = pytest.mark.skipif(not _CAN_RUN, reason=_SKIP_REASON)
_flydsl_only = pytest.mark.skipif(not _IS_FLYDSL, reason="FlyDSL-specific, no CuTe analogue")

if _CAN_RUN:
    # Through the package, not the backend module: quack/__init__.py picks the
    # implementation, so a direct import would never notice a mis-pick.
    from quack import rmsnorm_fwd as _rmsnorm_fwd
if _IS_FLYDSL:
    import quack.rmsnorm_flydsl as fly_rmsnorm

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024

# Each backend keeps its own budget: CuTe's rsqrt is fastmath, which is what
# spent bf16 down to 1e-1 in test_rmsnorm.py, while the FlyDSL kernel measured
# 0.50 bf16 ULP against fp32.
_ATOL, _RTOL = (
    ({torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 2e-4}, 2e-3)
    if _IS_FLYDSL
    else ({torch.float16: 1e-2, torch.bfloat16: 1e-1, torch.float32: 1e-4}, 1e-3)
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _fwd(use_compile):
    if not use_compile:
        return _rmsnorm_fwd
    return torch.compile(_rmsnorm_fwd, fullgraph=True, dynamic=False)


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
        rtol=_RTOL,
    )


def test_rmsnorm_fwd_dispatches_to_the_device_backend():
    """quack.rmsnorm_fwd must resolve to the backend this device actually runs."""
    expected = "quack.rmsnorm_flydsl" if _BACKEND == "flydsl" else "quack.rmsnorm"
    assert _rmsnorm_fwd.__module__ == expected


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize(
    "m,n,dtype,weight_dtype",
    [
        pytest.param(1, 8, torch.float32, torch.float32, id="short-fp32"),
        pytest.param(2, 12, torch.float32, torch.float32, id="fp32-four-element-alignment"),
        pytest.param(3, 192, torch.float16, torch.float16, id="padded-fp16"),
        pytest.param(5, 760, torch.bfloat16, torch.float32, id="predicated-mixed-weight"),
        pytest.param(2, 4096, torch.bfloat16, None, id="cross-wave"),
        pytest.param(2, 8192, torch.bfloat16, torch.float32, id="doubled-thread-geometry"),
        pytest.param(1, 65536, torch.bfloat16, torch.float32, id="wide-reload"),
        pytest.param(1, 65544, torch.bfloat16, torch.float32, id="wide-predicated-reload"),
        pytest.param(1, 262144, torch.bfloat16, torch.float32, id="max-n"),
    ],
)
def test_rmsnorm_forward(m, n, dtype, weight_dtype, use_compile):
    """Test RMSNorm forward pass against reference implementation."""
    x = torch.randn(m, n, device=_DEVICE, dtype=dtype)
    weight = (
        torch.randn(n, device=_DEVICE, dtype=weight_dtype) if weight_dtype is not None else None
    )

    out, residual_out, rstd = _fwd(use_compile)(x, weight)
    expected, _, _ = _reference(x, weight)

    assert out.shape == x.shape and out.dtype == x.dtype
    assert residual_out is x
    assert rstd is None
    _assert_close(out, expected, dtype)


# n=760 stays register-cached, n=65536 reloads. The offset is folded into the
# epilogue both paths share, so each needs its own value check.
@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("n", [760, 65536])
def test_rmsnorm_weight_offset(n, use_compile):
    x = torch.randn((3, n), device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(n, device=_DEVICE, dtype=torch.float32)

    function = _fwd(use_compile)
    out, _, _ = function(x, weight, weight_offset=1.0)
    expected, _, _ = _reference(x, weight + 1.0)

    _assert_close(out, expected, x.dtype)
    # An ignored offset would leave the plain-weight result, which differs here.
    plain, _, _ = function(x, weight)
    assert not torch.allclose(out, plain, atol=1e-3, rtol=1e-3)


def test_rmsnorm_compile_2d_then_3d():
    """One compiled callable must take a 2D input then a 3D one, no dynamo.reset() between.

    3D, not the 4D of the counterpart: that test calls rmsnorm(), which flattens
    leading batch dims before the kernel sees them. At this level per-head input
    is (rows, H, D).
    """
    torch._dynamo.reset()
    function = _fwd(use_compile=True)

    x = torch.randn(32, 256, device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(256, device=_DEVICE, dtype=torch.float32)
    out, residual_out, rstd = function(x, weight, eps=1e-5)
    expected, _, _ = _reference(x, weight, eps=1e-5)
    assert out.shape == x.shape and residual_out is x and rstd is None
    _assert_close(out, expected, x.dtype)

    # Different rank, per-head weight, and two arguments the first trace never saw.
    shape = (32, 4, 64)
    x3 = torch.randn(shape, device=_DEVICE, dtype=torch.bfloat16)
    weight2 = torch.randn(shape[-2:], device=_DEVICE, dtype=torch.float32)
    bias2 = torch.randn(shape[-2:], device=_DEVICE, dtype=torch.float32)
    residual3 = torch.randn(shape, device=_DEVICE, dtype=torch.bfloat16)
    out3, residual_out3, _ = function(x3, weight2, bias=bias2, residual=residual3)
    expected3, combined3, _ = _reference(x3, weight2, bias2, residual3)
    assert out3.shape == shape and residual_out3.shape == shape
    _assert_close(out3, expected3, x3.dtype)
    _assert_close(residual_out3, combined3, x3.dtype)


@pytest.mark.parametrize("use_compile", [False, True])
def test_rmsnorm_qk(use_compile):
    """Per-head weight and bias over a 3D input, with a residual and rstd."""
    shape = (6, 4, 64)
    x = torch.randn(shape, device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(shape[-2:], device=_DEVICE, dtype=torch.float32)
    bias = torch.randn(shape[-2:], device=_DEVICE, dtype=torch.float32)
    residual = torch.randn(shape, device=_DEVICE, dtype=torch.float16)

    out, residual_out, rstd = _fwd(use_compile)(
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


@_flydsl_only
@pytest.mark.parametrize("use_compile", [False, True])
def test_rmsnorm_qk_4d(use_compile):
    """FlyDSL flattens leading batch dims itself, so per-head input may be any rank.

    CuTe's rmsnorm_fwd cannot take this: it builds a 3D per-head layout descriptor,
    and rmsnorm() reshapes to it before the call.
    """
    shape = (2, 3, 4, 64)
    x = torch.randn(shape, device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(shape[-2:], device=_DEVICE, dtype=torch.float32)

    out, _, rstd = _fwd(use_compile)(x, weight, store_rstd=True)
    expected, _, expected_rstd = _reference(x, weight)

    assert out.shape == shape and rstd.shape == shape[:-1]
    _assert_close(out, expected, x.dtype)
    _assert_close(rstd, expected_rstd, x.dtype)


@_flydsl_only
@pytest.mark.parametrize("use_compile", [False, True])
def test_rmsnorm_strided_tensor(use_compile):
    """A row stride that is not 16B-aligned must be copied before vector access."""
    n = 192
    storage = torch.randn(4, 196, device=_DEVICE, dtype=torch.float16)
    x = storage[:, :n]
    assert (x.stride(0) * x.element_size()) % 16 == 8

    packed = fly_rmsnorm.packed_rows(x)
    assert packed.is_contiguous()
    assert packed.data_ptr() != x.data_ptr()

    weight = torch.randn(n, device=_DEVICE, dtype=torch.float16)
    out, _, _ = _fwd(use_compile)(x, weight)
    expected, _, _ = _reference(x, weight)
    _assert_close(out, expected, x.dtype)


@_flydsl_only
@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("n", [131072, 262144])
def test_rmsnorm_large_tensor(n, use_compile):
    """The row counts a real model reaches; every other test here fits in 11 rows.

    FlyDSL-only for cost, not capability: ~35 GiB of peak VRAM, and the CuTe
    counterpart already covers the same m and n. Drop the marker to share it.
    """
    m, n_chunks = 32 * 1024, 16
    # x + out must be fully materialized (irreducible); the reference is
    # computed one m/n_chunks-row chunk at a time, so it never is. Gate on
    # *free* memory so a partially-occupied GPU skips instead of OOMing.
    torch.cuda.empty_cache()
    peak_bytes = 2 * m * n * 2 + 3 * (m // n_chunks) * n * 2
    free_bytes = torch.cuda.mem_get_info()[0]
    if peak_bytes > free_bytes * 0.9:
        pytest.skip(
            f"Insufficient free VRAM ({free_bytes // 2**30} GiB free,"
            f" need ~{peak_bytes // 2**30} GiB)"
        )

    x = torch.randn(m, n, device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(n, device=_DEVICE, dtype=torch.float32)
    out, _, _ = _fwd(use_compile)(x, weight)

    # Absolute only, at the counterpart's looser bf16 budget: one bf16 ULP is
    # ~0.8% relative, so the file's rtol=2e-3 is tighter than correct rounding
    # permits, and half a billion elements always reach that tail.
    for x_c, out_c in zip(x.chunk(n_chunks), out.chunk(n_chunks)):
        assert (out_c.float() - _reference(x_c, weight)[0]).abs().max() < 1e-1


@_flydsl_only
def test_rmsnorm_input_validation():
    """Test input validation and error handling: FlyDSL rejects before the op."""
    x = torch.empty(2, 16, device=_DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="multiple of 8"):
        fly_rmsnorm.rmsnorm_fwd(torch.empty(2, 15, device=_DEVICE, dtype=torch.float16))
    with pytest.raises(ValueError, match="weight shape"):
        fly_rmsnorm.rmsnorm_fwd(x, torch.empty(8, device=_DEVICE))
    with pytest.raises(TypeError, match="weight dtype"):
        fly_rmsnorm.rmsnorm_fwd(x, torch.empty(16, device=_DEVICE, dtype=torch.float64))
    with pytest.raises(ValueError, match="residual shape"):
        fly_rmsnorm.rmsnorm_fwd(x, residual=torch.empty(1, 16, device=_DEVICE))
    with pytest.raises(TypeError, match="x dtype"):
        fly_rmsnorm.rmsnorm_fwd(x.double())
    with pytest.raises(ValueError, match="finite and positive"):
        fly_rmsnorm.rmsnorm_fwd(x, eps=0)
    with pytest.raises(ValueError, match="ROCm device"):
        fly_rmsnorm.rmsnorm_fwd(x.cpu())


@_flydsl_only
def test_rmsnorm_compile_cache():
    """One launcher per specialization; row padding and row count are not part of it."""
    n = 192
    fly_rmsnorm._compiled_forward.cache_clear()
    assert fly_rmsnorm._compiled_forward.cache_info().currsize == 0

    def call(m, padding, width=n, dtype=torch.float16):
        storage = torch.randn(m, width + padding, device=_DEVICE, dtype=dtype)
        x = storage[:, :width]
        assert x.stride() == (width + padding, 1)
        weight = torch.randn(width, device=_DEVICE, dtype=dtype)
        out, _, _ = fly_rmsnorm.rmsnorm_fwd(x, weight)
        _assert_close(out, _reference(x, weight)[0], dtype)

    # First call compiles.
    call(3, 8)
    assert fly_rmsnorm._compiled_forward.cache_info().currsize == 1

    # Same shape reuses.
    call(3, 8)
    assert fly_rmsnorm._compiled_forward.cache_info().currsize == 1

    # A different row count and a different row padding still reuse.
    call(11, 16)
    assert fly_rmsnorm._compiled_forward.cache_info().currsize == 1

    # A different normalized dimension is a different specialization.
    call(3, 8, width=2 * n)
    assert fly_rmsnorm._compiled_forward.cache_info().currsize == 2

    # So is a different dtype: dropped from the key, it would hand back a
    # launcher built for another element width.
    call(3, 8, dtype=torch.bfloat16)
    assert fly_rmsnorm._compiled_forward.cache_info().currsize == 3


@pytest.mark.parametrize("use_compile", [False, True])
def test_rmsnorm_with_bias(use_compile):
    m, n = 32, 1024
    x = torch.randn(m, n, device=_DEVICE, dtype=torch.float16)
    weight = torch.randn(n, device=_DEVICE, dtype=torch.float32)
    bias = torch.randn(n, device=_DEVICE, dtype=torch.float32)

    out, residual_out, rstd = _fwd(use_compile)(x, weight, bias=bias)
    expected, _, _ = _reference(x, weight, bias)

    assert out.shape == x.shape and out.dtype == torch.float16
    assert residual_out is x
    assert rstd is None
    _assert_close(out, expected, x.dtype)


@pytest.mark.parametrize("use_compile", [False, True])
def test_rmsnorm_with_residual(use_compile):
    m, n = 32, 1024
    x = torch.randn(m, n, device=_DEVICE, dtype=torch.float16)
    weight = torch.randn(n, device=_DEVICE, dtype=torch.float32)
    residual = torch.randn(m, n, device=_DEVICE, dtype=torch.float16)

    out, residual_out, rstd = _fwd(use_compile)(
        x,
        weight,
        residual=residual,
        residual_dtype=torch.float32,
        store_rstd=True,
    )
    expected, combined, expected_rstd = _reference(x, weight, residual=residual)

    assert residual_out.shape == x.shape and residual_out.dtype == torch.float32
    assert rstd.shape == x.shape[:-1]
    _assert_close(out, expected, x.dtype)
    # The kernel accumulates x + residual in fp32 and casts only at the store.
    torch.testing.assert_close(residual_out, combined, atol=0, rtol=0)
    _assert_close(rstd, expected_rstd, x.dtype)


@pytest.mark.parametrize("use_compile", [False, True])
def test_rmsnorm_residual_dtype_override(use_compile):
    """residual_dtype without a residual still forces a fresh fp32 residual_out."""
    x = torch.randn((3, 64), device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(64, device=_DEVICE, dtype=torch.float32)

    out, residual_out, rstd = _fwd(use_compile)(x, weight, residual_dtype=torch.float32)
    expected, combined, _ = _reference(x, weight)

    assert residual_out.dtype == torch.float32
    assert residual_out.data_ptr() != x.data_ptr()
    assert rstd is None
    _assert_close(out, expected, x.dtype)
    torch.testing.assert_close(residual_out, combined, atol=0, rtol=0)


@pytest.mark.parametrize("store_rstd", [False, True])
def test_rmsnorm_fwd_empty(store_rstd):
    """Zero rows (e.g. uneven FSDP shards) must return correctly-shaped empty outputs."""
    n = 4096
    x = torch.empty(0, n, device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(n, device=_DEVICE, dtype=torch.bfloat16)

    out, residual_out, rstd = _rmsnorm_fwd(x, weight, store_rstd=store_rstd)

    assert out.shape == x.shape and out.numel() == 0
    # No residual passed in: residual_out aliases x (and is empty).
    assert residual_out.shape == x.shape and residual_out.numel() == 0
    if store_rstd:
        assert rstd.shape == x.shape[:-1] and rstd.numel() == 0
    else:
        assert rstd is None
