"""Complex64 element type for CuTe-DSL kernels.

Single-precision complex (re + imj) carried as f64-packed bits (re in the low
32 bits, im in the high 32 bits). f64 is on `cute.MemRefType`'s element-type
allowlist; the natural `complex<f32>` MLIR type is not. Arithmetic methods
unpack each f64 into two Float32 lanes, compute, and repack -- the bitcasts
are folded out by ptxas.

Inherits from `Float32` (with `width=64, mlir_type=T.f64` overrides) so that
Python's subclass-precedence rule routes `Float32 OP Complex64` to our
reflected `__r*__` operators before Numeric's promotion logic sees the
operands. Without this, Float32-on-the-LEFT would silently promote through
Float64 conversion and corrupt the packed bits.

Boundary convention (tvm-ffi): the compiled kernel's ABI sees f64 storage.
At the call site, pass `torch.complex64` tensors as `t.view(torch.float64)`
-- use `complex_storage(t)` for the conversion.

See `AI/complex64_design_notes.md` for the why and what's been validated.
"""

from __future__ import annotations

import ctypes

import numpy as np
import torch

import cutlass.cute as cute
from cutlass import Float32, Numeric
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith
from cutlass._mlir.extras import types as T
from cutlass.base_dsl._mlir_helpers.arith import bitcast as _bitcast
from cutlass.base_dsl.typing import FloatMeta


class Complex64(Float32, metaclass=FloatMeta, width=64, mlir_type=T.f64):
    """Complex64 carried as f64-packed bits (re in low 32, im in high 32).

    `tensor.element_type is Complex64` inside the kernel; indexing returns
    Complex64 instances; `+`, `-`, `*`, `__neg__`, and `conj()` work natively.
    """

    def __init__(self, x, im=None, *, loc=None, ip=None):
        # Two-arg lane form: Complex64(re, im) packs (re, im) into f64 bits.
        # Coerce both through Float32 so int / float / Float32 / ir.Value(f32)
        # all work as inputs.
        if im is not None:
            # Static fast path: both args are Python int/float -- no MLIR
            # context needed (lets host-side code call Complex64(2.5, -1.5)).
            if isinstance(x, (int, float)) and isinstance(im, (int, float)):
                Complex64.__init__(self, complex(x, im), loc=loc, ip=ip)
                return
            re_ssa = Float32(x).ir_value()
            im_ssa = Float32(im).ir_value()
            Numeric.__init__(self, Complex64._pack_ssa(re_ssa, im_ssa))
            return

        # Same-type copy MUST be checked first. `_cvt_to_dest` (cute/tensor.py)
        # calls `data.to(element_type)` on every tensor write, which becomes
        # `Complex64(complex_instance)`; falling through to the generic-Numeric
        # branch below would re-pack as (real_view_of_packed_bits, 0) and
        # silently corrupt the data.
        if isinstance(x, Complex64):
            Numeric.__init__(self, x.value)
            return

        if isinstance(x, complex):
            f64_val = _pack_python_complex(x)
            Numeric.__init__(self, f64_val)
            return

        if isinstance(x, ir.Value):
            if x.type == T.f64():
                # Already in our storage form (loaded from a Complex64 tensor,
                # or output of _pack_ssa).
                Numeric.__init__(self, x)
                return
            if x.type == T.f32():
                packed = Complex64._pack_ssa(x, arith.constant(T.f32(), 0.0))
                Numeric.__init__(self, packed)
                return
            raise TypeError(f"Complex64: ir.Value of unsupported type {x.type}")

        if isinstance(x, Numeric):
            # Float32, Int32, Float64, etc. -> coerce real lane through Float32,
            # imag lane = 0. Float32(Float32) is a no-op, so this also handles
            # the Float32 case cleanly.
            re_ssa = Float32(x).ir_value()
            packed = Complex64._pack_ssa(re_ssa, arith.constant(T.f32(), 0.0))
            Numeric.__init__(self, packed)
            return

        if isinstance(x, (int, float)):
            Complex64.__init__(self, complex(x, 0.0), loc=loc, ip=ip)
            return

        raise TypeError(f"Complex64: unsupported source type {type(x)}")

    # ---- packing / unpacking primitives --------------------------------

    @staticmethod
    def _pack_ssa(re_f32, im_f32):
        """Pack two f32 SSA lanes into one f64 SSA value (re lo, im hi)."""
        re_i32 = _bitcast(re_f32, T.i32())
        im_i32 = _bitcast(im_f32, T.i32())
        re_i64 = arith.extui(T.i64(), re_i32)
        im_i64 = arith.extui(T.i64(), im_i32)
        hi = arith.shli(im_i64, arith.constant(T.i64(), 32))
        return _bitcast(arith.ori(re_i64, hi), T.f64())

    def _unpack(self):
        """Split self -> (re_f32, im_f32) as Float32 SSA values."""
        i64_ssa = _bitcast(self.ir_value(), T.i64())
        lo32 = arith.trunci(T.i32(), i64_ssa)
        hi32 = arith.trunci(T.i32(), arith.shrui(i64_ssa, arith.constant(T.i64(), 32)))
        return Float32(_bitcast(lo32, T.f32())), Float32(_bitcast(hi32, T.f32()))

    @staticmethod
    def from_re_im(re: Float32, im: Float32) -> "Complex64":
        """Build a Complex64 from two Float32 SSA lanes.

        Equivalent to `Complex64(re, im)`; kept as an explicit name for the
        hot-path call sites that want to skip the Float32 coercion in __init__.
        """
        return Complex64(Complex64._pack_ssa(re.ir_value(), im.ir_value()))

    # Internal alias used by arithmetic methods.
    _from_re_im = from_re_im

    # ---- accessors ------------------------------------------------------

    def real(self) -> Float32:
        re, _ = self._unpack()
        return re

    def imag(self) -> Float32:
        _, im = self._unpack()
        return im

    def conj(self) -> "Complex64":
        re, im = self._unpack()
        return Complex64._from_re_im(re, -im)

    # ---- arithmetic -----------------------------------------------------

    def __add__(self, other, *, loc=None, ip=None):
        a_re, a_im = self._unpack()
        b_re, b_im = _other_lanes(other)
        return Complex64._from_re_im(a_re + b_re, a_im + b_im)

    def __radd__(self, other, *, loc=None, ip=None):
        return self.__add__(other, loc=loc, ip=ip)

    def __sub__(self, other, *, loc=None, ip=None):
        a_re, a_im = self._unpack()
        b_re, b_im = _other_lanes(other)
        return Complex64._from_re_im(a_re - b_re, a_im - b_im)

    def __rsub__(self, other, *, loc=None, ip=None):
        a_re, a_im = self._unpack()
        b_re, b_im = _other_lanes(other)
        return Complex64._from_re_im(b_re - a_re, b_im - a_im)

    def __mul__(self, other, *, loc=None, ip=None):
        a_re, a_im = self._unpack()
        if isinstance(other, Complex64):
            b_re, b_im = other._unpack()
            return Complex64._from_re_im(a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re)
        # Real scalar: (re, im) * s = (re*s, im*s)
        s = Float32(other)
        return Complex64._from_re_im(a_re * s, a_im * s)

    def __rmul__(self, other, *, loc=None, ip=None):
        return self.__mul__(other, loc=loc, ip=ip)

    def __neg__(self, *, loc=None, ip=None):
        re, im = self._unpack()
        return Complex64._from_re_im(-re, -im)

    # ---- runtime arg passing -------------------------------------------

    def __c_pointers__(self):
        # Scalar Complex64 args travel as 8 bytes (the packed-as-f64 value).
        if not isinstance(self.value, float):
            raise ValueError(
                "Complex64 with a dynamic SSA value cannot be passed as a "
                "kernel argument; only static values are supported"
            )
        return [ctypes.cast(ctypes.pointer(ctypes.c_double(self.value)), ctypes.c_void_p)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pack_python_complex(c: complex) -> float:
    """Compute the f64 representation of a complex bit-packed (re, im)."""
    re_b = int(np.float32(c.real).view(np.uint32))
    im_b = int(np.float32(c.imag).view(np.uint32))
    return float(np.uint64((im_b << 32) | re_b).view(np.float64))


def _other_lanes(other):
    """Unpack the RHS of a binary op into (re_f32, im_f32) Float32 lanes."""
    if isinstance(other, Complex64):
        return other._unpack()
    return Float32(other), Float32(0.0)


def _retag_as_complex64(t):
    """Restore `t.element_type is Complex64` after a code path that derived it
    from MLIR (where complex64 collapses to Float64 / Int64 because
    `Numeric.from_mlir_type` is a many-to-one lookup)."""
    t._dtype = Complex64
    return t


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def allocate_smem_complex(
    allocator,
    layout_or_shape,
    byte_alignment: int = 16,
    swizzle=None,
):
    """Allocate a `Complex64` smem tensor.

    Wraps `cutlass.utils.SmemAllocator.allocate_tensor(Complex64, ...)` and
    re-tags the result so `tensor.element_type is Complex64`. Without the
    re-tag, the JIT-side tensor's element_type is `Float64` (derived from the
    f64 memref) and writes go through `Complex64.to(Float64)` and corrupt the
    packed bits.
    """
    t = allocator.allocate_tensor(
        Complex64, layout_or_shape, byte_alignment=byte_alignment, swizzle=swizzle
    )
    return _retag_as_complex64(t)


def recast_to_complex64(src: cute.Tensor) -> cute.Tensor:
    """Recast any tensor (e.g. Float32, Int64) to a `Complex64` tensor.

    Wraps `cute.recast_tensor(src, Complex64)` and re-tags the result. Same
    dtype-loss bug as `allocate_smem_complex` -- the underlying recast goes
    through `make_tensor`, which derives element_type from the MLIR memref
    (here f64) and gets back Float64.
    """
    return _retag_as_complex64(cute.recast_tensor(src, Complex64))


def complex_storage(t: torch.Tensor) -> torch.Tensor:
    """View a `torch.complex64` tensor as `torch.float64` with the same memory.

    Compiled kernels declared with `Complex64` element type have an f64 ABI;
    use this at the boundary to satisfy tvm-ffi's dtype check without copying.
    """
    if t.dtype == torch.float64:
        return t
    if t.dtype != torch.complex64:
        raise TypeError(
            f"complex_storage expects torch.complex64 (or torch.float64 for "
            f"already-converted storage), got {t.dtype}"
        )
    return t.view(torch.float64)


# ---------------------------------------------------------------------------
# tvm-ffi registration
# ---------------------------------------------------------------------------


def _register_with_tvm_ffi() -> None:
    """Teach tvm-ffi that Complex64 has an f64 ABI.

    Both `NumericToTVMFFIDtype` (the type->dtype-string lookup) and
    `AcceptableNumericTypesForScalar` (the allowlist for scalar kernel args)
    are plain Python collections, so we extend them at import time.
    """
    from cutlass.cute import _tvm_ffi_args_spec_converter as _cv

    _cv.NumericToTVMFFIDtype.setdefault(Complex64, "float64")
    if Complex64 not in _cv.AcceptableNumericTypesForScalar:
        _cv.AcceptableNumericTypesForScalar.append(Complex64)


_register_with_tvm_ffi()
