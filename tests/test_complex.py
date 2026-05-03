"""Smoke tests for quack.complex.Complex64 through the tvm-ffi compile path."""

import math

import numpy as np
import pytest
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float64, Int32, const_expr
from cutlass._mlir import ir
from cutlass._mlir.extras import types as T

from quack.complex import (
    Complex64,
    allocate_smem_complex,
    complex_storage,
    recast_to_complex64,
)


class _ScaleByComplex:
    """out[i, j] = scale_complex * in[i, j], one thread per element."""

    def __init__(self, threads_per_block: int = 128):
        self.threads_per_block = threads_per_block

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        scale: Complex64,
        stream: cuda.CUstream,
    ):
        assert mX.element_type is Complex64
        assert mO.element_type is Complex64
        batch, n = mX.shape
        threads = self.threads_per_block
        grid_x = cute.ceil_div(n, threads)
        self.kernel(mX, mO, scale).launch(
            grid=[grid_x, batch, 1],
            block=[threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor, scale: Complex64):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, _ = cute.arch.block_idx()
        i = bidx_x * self.threads_per_block + tidx
        if i < mX.shape[1]:
            mO[bidx_y, i] = mX[bidx_y, i] * scale


def _compile_scale_by_complex(n: int):
    batch_sym = cute.sym_int()
    x_fake = cute.runtime.make_fake_tensor(
        Complex64, (batch_sym, n), stride=(n, 1), assumed_align=8
    )
    o_fake = cute.runtime.make_fake_tensor(
        Complex64, (batch_sym, n), stride=(n, 1), assumed_align=8
    )
    return cute.compile(
        _ScaleByComplex(),
        x_fake,
        o_fake,
        Complex64(complex(0.0, 0.0)),  # placeholder; real value passed at call
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@pytest.mark.parametrize("n", [128, 256])
@pytest.mark.parametrize("batch", [1, 4])
def test_complex_scale_tvm_ffi(batch: int, n: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    x = torch.randn(batch, n, dtype=torch.complex64, device="cuda")
    out = torch.empty_like(x)
    scale = complex(2.0, 1.0)

    fn = _compile_scale_by_complex(n)
    fn(complex_storage(x), complex_storage(out), Complex64(scale))
    torch.cuda.synchronize()

    expected = x * scale
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Smem round-trip: gmem -> rmem -> smem -> sync -> smem -> rmem -> gmem.
# Catches the "smem allocation loses Complex64 type tag" bug -- without
# allocate_smem_complex, scalar smem writes silently truncate via Float64.to.
# ---------------------------------------------------------------------------


_SMEM_THREADS = 32
_SMEM_ELEMS_PER_THREAD = 4
_SMEM_BLOCK_ELEMS = _SMEM_THREADS * _SMEM_ELEMS_PER_THREAD


class _SmemRoundTrip:
    """Each thread loads 4 contiguous complex from gmem, stores to smem at the
    same indices, syncs, reads them back at strided indices, writes to gmem at
    the strided indices. Output should equal input (read-stride and write-stride
    cancel)."""

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream):
        batch = mX.shape[0]
        self.kernel(mX, mO).launch(
            grid=[batch, 1, 1],
            block=[_SMEM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX, mO):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        rmem = cute.make_rmem_tensor(_SMEM_ELEMS_PER_THREAD, Complex64)
        for i in cutlass.range_constexpr(_SMEM_ELEMS_PER_THREAD):
            rmem[i] = mX[bidx, tidx * const_expr(_SMEM_ELEMS_PER_THREAD) + const_expr(i)]

        smem = cutlass.utils.SmemAllocator()
        exchange = allocate_smem_complex(smem, cute.make_layout(_SMEM_BLOCK_ELEMS))

        for i in cutlass.range_constexpr(_SMEM_ELEMS_PER_THREAD):
            exchange[tidx * const_expr(_SMEM_ELEMS_PER_THREAD) + const_expr(i)] = rmem[i]
        cute.arch.barrier()
        for i in cutlass.range_constexpr(_SMEM_ELEMS_PER_THREAD):
            rmem[i] = exchange[tidx + const_expr(i * _SMEM_THREADS)]
        for i in cutlass.range_constexpr(_SMEM_ELEMS_PER_THREAD):
            mO[bidx, tidx + const_expr(i * _SMEM_THREADS)] = rmem[i]


def _compile_smem_roundtrip():
    batch_sym = cute.sym_int()
    args = [
        cute.runtime.make_fake_tensor(
            Complex64,
            (batch_sym, _SMEM_BLOCK_ELEMS),
            stride=(_SMEM_BLOCK_ELEMS, 1),
            assumed_align=8,
        ),
        cute.runtime.make_fake_tensor(
            Complex64,
            (batch_sym, _SMEM_BLOCK_ELEMS),
            stride=(_SMEM_BLOCK_ELEMS, 1),
            assumed_align=8,
        ),
    ]
    return cute.compile(
        _SmemRoundTrip(),
        *args,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@pytest.mark.parametrize("batch", [1, 4])
def test_smem_roundtrip_complex64(batch: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    x = torch.randn(batch, _SMEM_BLOCK_ELEMS, dtype=torch.complex64, device="cuda")
    out = torch.empty_like(x)
    fn = _compile_smem_roundtrip()
    fn(complex_storage(x), complex_storage(out))
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)


def test_complex_static_unpack_pack_roundtrip():
    """Sanity: a static Python complex packs/unpacks losslessly through f64 bits."""
    import numpy as np

    for c in [
        complex(1.0, -1.0),
        complex(3.0, 4.0),
        complex(-7.5, 0.25),
        complex(0.0, 0.0),
    ]:
        cx = Complex64(c)
        # Round-trip through the value's stored f64 -> bytes -> two f32s
        f64_bits = np.float64(cx.value).view(np.uint64).item()
        re_bits = np.uint32(f64_bits & 0xFFFFFFFF)
        im_bits = np.uint32((f64_bits >> 32) & 0xFFFFFFFF)
        re = re_bits.view(np.float32).item()
        im = im_bits.view(np.float32).item()
        assert re == c.real and im == c.imag, f"{c} -> ({re}, {im})"


# ===========================================================================
# Tier 1-8 Complex64 type tests. See AI/complex64_design_notes.md for the
# behaviors being pinned. Most kernel-side tests share the same harness:
# launch a one-thread kernel that writes test results into a small Complex64
# output buffer; verify on the host.
# ===========================================================================


def _f64bits_to_complex(f64_value: float) -> complex:
    """Decode the f64 packed-as-complex value into a Python complex."""
    bits = np.float64(f64_value).view(np.uint64).item()
    re = np.uint32(bits & 0xFFFFFFFF).view(np.float32).item()
    im = np.uint32((bits >> 32) & 0xFFFFFFFF).view(np.float32).item()
    return complex(re, im)


# ---------------------------------------------------------------------------
# Harness: run a one-thread kernel that writes Complex64 results to gmem.
# ---------------------------------------------------------------------------


class _OneShot:
    """Generic one-thread kernel that delegates the body to `body_fn`."""

    def __init__(self, body_fn):
        self.body_fn = body_fn

    @cute.jit
    def __call__(self, mO: cute.Tensor, stream):
        self.kernel(mO).launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mO: cute.Tensor):
        self.body_fn(mO)


def _run_oneshot(body_fn, n_outputs: int) -> list[complex]:
    """Compile a kernel that writes n_outputs Complex64 values, run it, return them."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    out = torch.zeros(n_outputs, dtype=torch.complex64, device="cuda")
    out_fake = cute.runtime.make_fake_tensor(Complex64, (n_outputs,), stride=(1,), assumed_align=8)
    fn = cute.compile(
        _OneShot(body_fn),
        out_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    fn(complex_storage(out))
    torch.cuda.synchronize()
    return out.tolist()


# ---------------------------------------------------------------------------
# Tier 1: constructor paths
# ---------------------------------------------------------------------------


def test_constructor_python_complex():
    """Tier 1.1 -- Complex64(c) for Python complex (host-side)."""
    cx = Complex64(complex(3.0, 4.0))
    assert _f64bits_to_complex(cx.value) == complex(3.0, 4.0)


def test_constructor_two_arg():
    """Tier 1.2 -- Complex64(re, im) two-arg form (host-side, static floats)."""
    cx = Complex64(2.5, -1.5)
    assert _f64bits_to_complex(cx.value) == complex(2.5, -1.5)


def test_constructor_same_type_copy_kernel():
    """Tier 1.3 -- Complex64(complex64_instance) kernel-side same-type copy."""

    def body(mO):
        a = Complex64(complex(3.0, 4.0))
        mO[0] = Complex64(a)  # short-circuit copy

    out = _run_oneshot(body, 1)
    assert out[0] == complex(3.0, 4.0)


def test_constructor_from_f32_ir_value_kernel():
    """Tier 1.5 -- Complex64(ir.Value of f32) becomes (val, 0)."""

    def body(mO):
        f32_val = Float32(2.5)
        mO[0] = Complex64(f32_val.ir_value())

    out = _run_oneshot(body, 1)
    assert out[0] == complex(2.5, 0.0)


def test_constructor_from_float32_kernel():
    """Tier 1.6 -- Complex64(Float32 instance) becomes (val, 0)."""

    def body(mO):
        mO[0] = Complex64(Float32(7.0))
        mO[1] = Complex64(Float32(-3.5))

    out = _run_oneshot(body, 2)
    assert out[0] == complex(7.0, 0.0)
    assert out[1] == complex(-3.5, 0.0)


def test_constructor_from_other_numeric_kernel():
    """Tier 1.7 -- Complex64(Int32 / Float64) becomes (real, 0)."""

    def body(mO):
        mO[0] = Complex64(Int32(5))
        mO[1] = Complex64(Float64(1.5))

    out = _run_oneshot(body, 2)
    assert out[0] == complex(5.0, 0.0)
    assert out[1] == complex(1.5, 0.0)


def test_constructor_from_python_int_float_kernel():
    """Tier 1.8 -- Complex64(int) and Complex64(float) become (val, 0)."""

    def body(mO):
        mO[0] = Complex64(3)
        mO[1] = Complex64(2.5)
        mO[2] = Complex64(-7)

    out = _run_oneshot(body, 3)
    assert out[0] == complex(3.0, 0.0)
    assert out[1] == complex(2.5, 0.0)
    assert out[2] == complex(-7.0, 0.0)


def test_constructor_bad_input_raises():
    """Tier 1.9 -- Complex64('hello') and Complex64(object()) raise TypeError."""
    with pytest.raises(TypeError):
        Complex64("hello")
    with pytest.raises(TypeError):
        Complex64(object())


# ---------------------------------------------------------------------------
# Tier 2: arithmetic matrix
# ---------------------------------------------------------------------------


def test_arithmetic_complex_complex():
    """Tier 2.10 -- Complex64 +/-/* Complex64 against Python complex reference."""

    def body(mO):
        a = Complex64(complex(1.0, 2.0))
        b = Complex64(complex(3.0, -1.0))
        mO[0] = a + b
        mO[1] = a - b
        mO[2] = a * b
        mO[3] = -a

    out = _run_oneshot(body, 4)
    a = complex(1.0, 2.0)
    b = complex(3.0, -1.0)
    assert out[0] == a + b
    assert out[1] == a - b
    assert out[2] == a * b
    assert out[3] == -a


def test_arithmetic_with_float32_both_directions():
    """Tier 2.11 -- Cx OP Float32 and Float32 OP Cx via subclass-precedence."""

    def body(mO):
        a = Complex64(complex(1.0, 2.0))
        s = Float32(5.0)
        mO[0] = a + s  # Complex64 + Float32 (rhs)
        mO[1] = s + a  # Float32 + Complex64 (lhs, via __radd__)
        mO[2] = a - s  # Complex64 - Float32
        mO[3] = s - a  # Float32 - Complex64 (lhs, via __rsub__)
        mO[4] = a * Float32(2.0)
        mO[5] = Float32(2.0) * a  # __rmul__

    out = _run_oneshot(body, 6)
    a = complex(1.0, 2.0)
    assert out[0] == a + 5.0
    assert out[1] == 5.0 + a
    assert out[2] == a - 5.0
    assert out[3] == 5.0 - a
    assert out[4] == a * 2.0
    assert out[5] == 2.0 * a


def test_arithmetic_with_python_scalar():
    """Tier 2.10 cont. -- Cx OP {int, float} (via __init__'s int/float branch)."""

    def body(mO):
        a = Complex64(complex(1.0, 2.0))
        mO[0] = a + 7.0  # cx + python float
        mO[1] = a * 3  # cx * python int

    out = _run_oneshot(body, 2)
    a = complex(1.0, 2.0)
    assert out[0] == a + 7.0
    assert out[1] == a * 3.0


def test_arithmetic_random_sweep():
    """Tier 2.13 -- (a + b * c) against Python complex reference for random triples."""
    rng = np.random.default_rng(0)
    inputs = [
        (
            complex(rng.uniform(-2, 2), rng.uniform(-2, 2)),
            complex(rng.uniform(-2, 2), rng.uniform(-2, 2)),
            complex(rng.uniform(-2, 2), rng.uniform(-2, 2)),
        )
        for _ in range(8)
    ]

    def body(mO):
        for i, (a_v, b_v, c_v) in enumerate(inputs):
            a = Complex64(a_v)
            b = Complex64(b_v)
            c = Complex64(c_v)
            mO[i] = a + b * c

    out = _run_oneshot(body, len(inputs))
    for i, (a_v, b_v, c_v) in enumerate(inputs):
        expected = a_v + b_v * c_v
        # f32 precision; use loose tolerance
        assert abs(complex(out[i]) - expected) < 1e-4, f"i={i}: {out[i]} vs {expected}"


# ---------------------------------------------------------------------------
# Tier 3: methods
# ---------------------------------------------------------------------------


def test_conj_basic():
    """Tier 3.14 -- conj() flips the imaginary lane."""

    def body(mO):
        a = Complex64(complex(3.0, -4.0))
        mO[0] = a.conj()
        mO[1] = a.conj().conj()  # idempotent

    out = _run_oneshot(body, 2)
    assert out[0] == complex(3.0, 4.0)
    assert out[1] == complex(3.0, -4.0)


def test_conj_self_product_is_real():
    """Tier 3.14 cont. -- (c * c.conj()).imag == 0; .real == |c|^2."""

    def body(mO):
        b = Complex64(complex(1.5, 2.5))
        mO[0] = b * b.conj()

    out = _run_oneshot(body, 1)
    expected_norm_sq = 1.5 * 1.5 + 2.5 * 2.5  # = 8.5
    assert abs(out[0].real - expected_norm_sq) < 1e-5
    assert out[0].imag == 0.0


def test_real_imag_accessors_kernel():
    """Tier 3.15 -- real() and imag() return the correct Float32 lanes."""

    def body(mO):
        a = Complex64(complex(3.0, -4.0))
        # Pack the lanes back into a Complex64 to round-trip them through gmem
        mO[0] = Complex64(a.real(), a.imag())

    out = _run_oneshot(body, 1)
    assert out[0] == complex(3.0, -4.0)


def test_from_re_im_alias_equivalence():
    """Tier 3.16 -- from_re_im(r, i) and Complex64(r, i) produce identical bits."""

    def body(mO):
        re = Float32(2.0)
        im = Float32(-1.5)
        mO[0] = Complex64.from_re_im(re, im)
        mO[1] = Complex64(re, im)

    out = _run_oneshot(body, 2)
    assert out[0] == out[1]
    assert out[0] == complex(2.0, -1.5)


# ---------------------------------------------------------------------------
# Tier 4: class invariants
# ---------------------------------------------------------------------------


def test_complex64_class_attributes():
    """Tier 4.17, 4.18, 4.20, 4.21 -- pure Python class metadata."""
    assert Complex64.width == 64
    assert Complex64.is_float is True
    assert isinstance(Complex64(complex(0, 0)), Float32)
    # Auto-promotion behavior pins (used by _cvt_to_dest):
    assert Complex64.is_same_kind(Float32) is True
    assert Complex64.is_same_kind(Float64) is True


def test_complex64_mlir_type_is_f64():
    """Tier 4.19 -- mlir_type query needs an MLIR context."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        assert Complex64.mlir_type == T.f64()


# ---------------------------------------------------------------------------
# Tier 5: helpers
# ---------------------------------------------------------------------------


def test_recast_to_complex64_kernel():
    """Tier 5.22 -- recast_to_complex64 on an rmem Float32 tensor preserves data."""

    # body_fn is plain Python called from inside @cute.kernel, so plain
    # `range(...)` unrolls at trace time (cutlass.range_constexpr only works
    # inside a directly-decorated @cute.jit/@cute.kernel function).
    def body(mO):
        rmem_f32 = cute.make_rmem_tensor(16, Float32)
        for i in range(8):
            rmem_f32[2 * i] = Float32(float(i + 1))
            rmem_f32[2 * i + 1] = Float32(-float(i + 1))
        rmem_cx = recast_to_complex64(rmem_f32)
        for i in range(8):
            mO[i] = rmem_cx[i]

    out = _run_oneshot(body, 8)
    expected = [complex(i + 1, -(i + 1)) for i in range(8)]
    assert out == expected


def test_complex_storage_passthrough_for_float64():
    """Tier 5.24a -- complex_storage(float64 tensor) returns the input unchanged."""
    t = torch.zeros(4, dtype=torch.float64)
    assert complex_storage(t) is t


def test_complex_storage_view_for_complex64():
    """Tier 5.24b -- complex_storage(complex64) returns a float64 view sharing memory."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for view alignment")
    c = torch.zeros(4, dtype=torch.complex64, device="cuda")
    v = complex_storage(c)
    assert v.dtype == torch.float64
    assert v.data_ptr() == c.data_ptr()
    assert v.numel() == c.numel()


def test_complex_storage_bad_dtype_raises():
    """Tier 5.24c -- complex_storage(other dtype) raises TypeError."""
    with pytest.raises(TypeError):
        complex_storage(torch.zeros(4, dtype=torch.float32))
    with pytest.raises(TypeError):
        complex_storage(torch.zeros(4, dtype=torch.int32))


# ---------------------------------------------------------------------------
# Tier 6: __c_pointers__ boundary
# ---------------------------------------------------------------------------


def test_c_pointers_static_value_works():
    """Tier 6.25 -- static-value Complex64 produces an 8-byte pointer."""
    c = Complex64(complex(2.0, 1.0))
    ptrs = c.__c_pointers__()
    assert len(ptrs) == 1


def test_c_pointers_dynamic_value_rejected():
    """Tier 6.26 -- a Complex64 carrying a non-float self.value raises.

    Simulates the "dynamic SSA" case (which can only really arise inside a
    kernel context) by mutating self.value to a non-float."""
    c = Complex64(complex(1.0, 0.0))
    c.value = "not-a-float"  # simulate dynamic state
    with pytest.raises(ValueError, match="dynamic SSA value"):
        c.__c_pointers__()


# ---------------------------------------------------------------------------
# Tier 7: special / edge values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "val",
    [
        complex(0.0, 0.0),
        complex(-0.0, -0.0),
        complex(float("inf"), 0.0),
        complex(0.0, float("inf")),
        complex(float("-inf"), 0.0),
        complex(float("nan"), 0.0),
        complex(0.0, float("nan")),
        complex(1e-38, -1e-38),
        complex(1e30, -1e30),
        complex(-1.0, -2.0),
    ],
)
def test_static_special_value_roundtrip(val):
    """Tier 7.27 -- bitcast plumbing preserves edge values exactly.

    Compare against the f32-rounded version of `val`, since Complex64 stores
    each lane as f32 by construction. NaN equality is special-cased.
    """
    cx = Complex64(val)
    decoded = _f64bits_to_complex(cx.value)
    expected_re = float(np.float32(val.real))
    expected_im = float(np.float32(val.imag))
    if math.isnan(expected_re):
        assert math.isnan(decoded.real)
    else:
        assert decoded.real == expected_re, f"re: {decoded.real} != {expected_re}"
    if math.isnan(expected_im):
        assert math.isnan(decoded.imag)
    else:
        assert decoded.imag == expected_im, f"im: {decoded.imag} != {expected_im}"


def test_sign_sensitivity_kernel():
    """Tier 7.28 -- all four sign combinations land at the right (re, im)."""
    cases = [
        complex(-1.0, -2.0),
        complex(-1.0, 2.0),
        complex(1.0, -2.0),
        complex(1.0, 2.0),
    ]

    def body(mO):
        for i, c in enumerate(cases):
            mO[i] = Complex64(c)

    out = _run_oneshot(body, len(cases))
    for i, c in enumerate(cases):
        assert out[i] == c, f"sign case {i}: {out[i]} != {c}"


# ---------------------------------------------------------------------------
# Tier 8: documented-behavior pins (auto-promotion via _cvt_to_dest)
# ---------------------------------------------------------------------------


def test_float32_writes_promote_to_complex64():
    """Tier 8.29 -- writing a Float32 to a Complex64 tensor lands as (val, 0)."""

    def body(mO):
        rmem = cute.make_rmem_tensor(2, Complex64)
        rmem[0] = Float32(5.0)  # promote: (5, 0)
        rmem[1] = Float32(-2.5)  # promote: (-2.5, 0)
        for i in range(2):
            mO[i] = rmem[i]

    out = _run_oneshot(body, 2)
    assert out[0] == complex(5.0, 0.0)
    assert out[1] == complex(-2.5, 0.0)


def test_float64_writes_promote_to_complex64():
    """Tier 8.30 -- pin behavior of Float64 written to a Complex64 tensor."""

    def body(mO):
        rmem = cute.make_rmem_tensor(1, Complex64)
        rmem[0] = Float64(3.14)  # whatever Numeric promotion does, pin it
        mO[0] = rmem[0]

    out = _run_oneshot(body, 1)
    # Currently Numeric.is_same_kind(Float, Float) plus width >= width triggers
    # data.to(Complex64), which goes through Float32(Float64) (cvtf truncate to f32),
    # then the Numeric branch packs (re=truncated, im=0).
    assert out[0] == complex(np.float32(3.14), 0.0)
