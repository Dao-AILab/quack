# Copyright (c) 2026, Tri Dao.

"""Forward-only mirror of tests/test_rmsnorm.py for the FlyDSL backend.

The two rmsnorm_fwd implementations take the same arguments and return the same
triple, so a test here carries its CuTe-DSL counterpart's name and does the same
thing at the same strength. Tests with no counterpart there (backward, autotune,
prenorm) are absent rather than invented.
"""

import pytest
import torch
from flydsl_env import CAN_RUN as _CAN_RUN
from flydsl_env import DEVICE as _DEVICE
from flydsl_env import SKIP_REASON as _SKIP_REASON

pytestmark = pytest.mark.skipif(not _CAN_RUN, reason=_SKIP_REASON)

if _CAN_RUN:
    import quack.rmsnorm_flydsl as fly_rmsnorm

# The use_compile axis recompiles once per specialization, same as test_rmsnorm.py.
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024

_ATOL = {
    torch.float16: 2e-3,
    torch.bfloat16: 2e-2,
    torch.float32: 2e-4,
}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _fwd(use_compile):
    """The `function = torch.compile(rmsnorm) if use_compile else rmsnorm` of test_rmsnorm.py."""
    if not use_compile:
        return fly_rmsnorm.rmsnorm_fwd
    return torch.compile(fly_rmsnorm.rmsnorm_fwd, fullgraph=True, dynamic=False)


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
    """Forward half of test_rmsnorm.py::test_rmsnorm_forward_backward."""
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


# One predicated register-cached row and one reloaded row: the offset is folded
# in the shared epilogue, so a value check on both branches pins the arithmetic
# that the type checks around weight_offset cannot see.
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


@pytest.mark.parametrize("use_compile", [False, True])
def test_rmsnorm_qk(use_compile):
    """Per-head weight and bias over a 4D input, with a residual and rstd."""
    shape = (2, 3, 4, 64)
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


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("n", [131072, 262144])
def test_rmsnorm_large_tensor(n, use_compile):
    """The row counts a real model reaches; every other test here fits in 11 rows."""
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

    # Absolute only, and the counterpart's looser bf16 budget: this kernel is
    # correctly rounded (measured worst case 0.50 bf16 ULP against fp32), but
    # one ULP is ~0.8% relative, so the file's rtol=2e-3 is tighter than correct
    # rounding permits and half a billion elements always reach that tail.
    for x_c, out_c in zip(x.chunk(n_chunks), out.chunk(n_chunks)):
        assert (out_c.float() - _reference(x_c, weight)[0]).abs().max() < 1e-1


def test_rmsnorm_input_validation():
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

    # So is a different dtype. This is the only pair in the file that differs
    # in nothing else, so it is the only thing standing between a dtype dropped
    # from the key and a launcher built for another element width.
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

    out, residual_out, rstd = fly_rmsnorm.rmsnorm_fwd(x, weight, store_rstd=store_rstd)

    assert out.shape == x.shape and out.numel() == 0
    # No residual passed in: residual_out aliases x (and is empty).
    assert residual_out.shape == x.shape and residual_out.numel() == 0
    if store_rstd:
        assert rstd.shape == x.shape[:-1] and rstd.numel() == 0
    else:
        assert rstd is None
