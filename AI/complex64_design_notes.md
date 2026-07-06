# `quack.complex.Complex64` design notes

A native-feeling `complex64` element type for CuTe-DSL kernels, without
patching the C++ MLIR bindings. Lives in `quack/complex.py`. Validated by
`tests/test_complex.py` (44 tests covering constructors, arithmetic matrix,
methods, class invariants, helpers, the c_pointers boundary, special / edge
values, auto-promotion, and a smem round-trip). FFT-correctness tests that
exercise Complex64 through real kernels live in `tests/test_fft.py`.

This file records *why* the design looks the way it does and what was tried
and rejected. The production native C2C FFT path in `quack/fft.py` already
uses Complex64 for register values and IO -- see
`AI/complex64_fft_migration_plan.md` for the cutover history and remaining
work.

---

## The fundamental constraint

`cute.MemRefType.get(value_type, layout_type)` -- the C++ MLIR-binding call
that backs every `make_tensor`, `make_rmem_tensor`, `recast_tensor`, and smem
`allocate_tensor` -- enforces an element-type **allowlist**: the value_type
must be one of `int`, `float`, `ptr`, or `sparse_elem`. The natural choice
for "complex single-precision" -- `complex<f32>` -- gets:

```
TypeError: expects value type to be int, float, ptr or sparse_elem,
           but got 'complex<f32>'
```

This is enforced in `_cute_ir` (the compiled binding), not in Python. We
have no Python-level escape hatch.

The other plausible "unique" MLIR types we tried have the same fate:

| MLIR element type     | allowlisted? | unique to Complex64?              |
| --------------------- | :----------: | --------------------------------- |
| `complex<f32>`        | ❌            | unique but rejected               |
| `tuple<f32, f32>`     | ❌            | unique but rejected               |
| `f64`                 | ✅            | shared with `Float64`             |
| `i64`                 | ✅            | shared with `Int64` / `Uint64`    |
| `vector<2xf32>`, etc. | ❌ (by family)| unique but rejected               |

So the only viable storage MLIR types are ones already claimed by some other
Python `Numeric`. We picked **`f64`**.

---

## The Python type identity is recoverable

`f64` is the storage type, but the kernel writer wants `tensor[i]` to return
a `Complex64`-typed value, not a `Float64`. The JIT-side `_Tensor` derives
`element_type` from the MLIR memref via `Numeric.from_mlir_type`, which is a
many-to-one lookup -- f64 always becomes `Float64`. So whatever we tell the
constructor about Python type identity must survive the round-trip.

We do this two ways:

1. **At `make_fake_tensor` / `make_rmem_tensor` time**, the dtype is passed
   explicitly and stored on the runtime tensor's `_dtype` attribute. As long
   as `Complex64.mlir_type == T.f64()`, the MLIR memref is f64 and the
   Python `_dtype` stays `Complex64`. ✅
2. **At `SmemAllocator.allocate_tensor` / `cute.recast_tensor` / pointer-to-
   tensor `make_tensor` time**, the resulting JIT tensor is built from an
   MLIR value with no carried dtype. Its `_dtype` falls back to
   `Numeric.from_mlir_type(f64)` = `Float64`. ❌ -- needs a re-tag.

The fix is a one-line wrapper, exposed as `allocate_smem_complex(...)` and
`recast_to_complex64(...)`. Both internally do
`t._dtype = Complex64; return t`.

Without the re-tag, scalar writes like `smem[i] = c` (where `c: Complex64`)
flow through `_cvt_to_dest` (`cute/tensor.py:308`), which calls
`data.to(self.element_type)` = `Complex64.to(Float64)` -- the latter does a
cvtf-style real-number conversion on the f64 packed bits, then truncates to
f32, then stores. Output is silently corrupted (specifically: high 32 bits
zeroed, low 32 bits hold the f32 truncation of the bit-packed-as-f64 value).

This is the *only* sharp edge the wrapper class can't smooth over, and it
shows up exactly at MLIR boundaries.

---

## Why inherit from `Float32`

The arithmetic operators (`__add__`/`__mul__`/...) on Complex64 unpack to
two Float32 lanes, compute, repack. Easy when Complex64 is on the LEFT --
Python calls `Complex64.__mul__(other)` first.

For `Float32 * Complex64`, Python normally calls `Float32.__mul__(complex)`
first. Without intervention, that goes through `Numeric._binary_op_type_promote`
which sees the operands as `(f32, i64)` or `(f32, f64)` (depending on the
Complex64.mlir_type) and either:

- **Inherit from `Int64`** (mlir_type=i64): kind mismatch fires at store time
  with `"type mismatch, store f64 to CxI64"`. Loud error, no corruption.
- **Inherit from `Float64`** (mlir_type=f64): same kind, same width -- promotion
  picks Complex64 as result type, then `op(lhs.value, rhs.value)` calls
  `arith.mulf` on two f64 SSAs. Stored bits are valid f64s but interpret the
  packed-bit pattern as a real number. **Silent corruption.**
- **Inherit from `Float32`** (mlir_type=f64, width=64 override): `Complex64` is
  a strict subclass of `Float32`, so Python's subclass-precedence rule fires
  `Complex64.__rmul__(Float32)` *before* `Float32.__mul__(Complex64)`. We
  control the dispatch and do the right thing. ✅

`Float32` inheritance is the only one of the three that gives both directions
of mixed arithmetic without monkey-patching. The structural cost: `isinstance(
complex_val, Float32) is True`. Only one site in cutlass touches this
(`nvvm_wrappers.py:441`, an fmax helper); not a problem in practice.

---

## Why same-type init must short-circuit first

`_cvt_to_dest` calls `data.to(element_type)` on every tensor write. If both
sides are `Complex64`, this becomes `Complex64(complex_instance)`. If our
`__init__` falls through to the generic-`Numeric` branch (`Float32(x)` then
re-pack), it treats the existing packed-bits Complex64 as if it were a
real-valued Float, extracts the f32 truncation of the bit pattern, and packs
that as `(real_view, 0)`. This was the section-8b bug.

The fix is a one-line check at the top of `__init__`:

```python
if isinstance(x, Complex64):
    Numeric.__init__(self, x.value)
    return
```

---

## What's been validated

Complex64 type itself (`tests/test_complex.py`, 44 tests):

- All constructor branches: Python `complex`, two-arg `(re, im)`, same-type
  copy, `ir.Value` of f64/f32, `Float32` instance, other `Numeric` (Int32,
  Float64), Python int/float, plus the bad-input `TypeError` path.
- Arithmetic matrix: `Complex64 OP X` for `X` in `{Complex64, Float32, int,
  float}`, both directions; `__neg__`; randomized correctness sweep against
  Python complex reference.
- Methods: `conj()` (involution + `c * c.conj()` is real), `real()`, `imag()`,
  `from_re_im` alias.
- Class invariants: `width=64`, `is_float=True`, `mlir_type == T.f64()`,
  `isinstance(c, Float32) is True`, `is_same_kind(Float32 / Float64) is True`.
- Helpers: `complex_storage(t)` (passthrough for f64, view for complex64,
  `TypeError` for other dtypes), `recast_to_complex64`, `allocate_smem_complex`.
- `__c_pointers__`: static-value path works, dynamic-SSA path raises.
- Edge values round-trip exactly through the bitcast plumbing: zero, -0.0,
  +/-inf, NaN, denormal, very large magnitudes, mixed signs.
- Auto-promotion via `_cvt_to_dest`: writing a `Float32` or `Float64` to a
  `Complex64` tensor lands as `(value, 0)` (pinned because future changes to
  `Numeric` promotion could silently break this).
- Smem round-trip (gmem -> rmem -> smem -> sync -> smem -> rmem -> gmem)
  using `allocate_smem_complex`; serves as a regression test for the
  dtype-loss bug at the smem allocation boundary.

Through real kernels (`tests/test_fft.py`):

- tvm-ffi compile path (`--enable-tvm-ffi` + `complex_storage(t)` boundary).
- Standalone radix-8 FFT butterfly matches `torch.fft.fft` numerically.
- N=64 = 8x8 Cooley-Tukey FFT with smem transpose matches numerically.
- The production native FFT class (N=2..8192) operates on Complex64 register
  fragments end-to-end and matches `torch.fft.fft` (existing
  `test_fft_native_power_of_two_matches_torch` parametrizations).

## What hasn't been validated

- TMA (`SM90`-style bulk loads / cp.async.bulk) carrying Complex64 elements.
- `cute.copy` with explicit copy atoms typed as Complex64; the production
  FFT path keeps smem in `Float32` layouts and recasts at the rmem boundary.
- Tensor-core (MMA) operations -- not expected to be relevant for FFT but
  noted for completeness.
- Typed-Complex64 smem layouts for the FFT path. The fast paths still allocate
  `Float32` smem because the tuned interleaved / split-real-imag / swizzled
  layouts are easier to express as scalar f32. Whether a typed-Complex64 smem
  variant wins on any specific N is an open profiling question (see the
  remaining work in `AI/complex64_fft_migration_plan.md`).
- Mixed Complex64 / Float32 arithmetic when `Float32` is a TENSOR (not just
  a scalar) -- e.g. multiplying a Complex64 register by a Float32 lane from
  another rmem tensor. Should work via the `__rmul__` path but not tested
  directly.

---

## Sharp edges to keep in mind

1. **MLIR boundary loses dtype.** Any code path that constructs a JIT tensor
   from MLIR (smem alloc, recast, ptr-to-tensor) loses the Complex64 tag.
   Use the wrappers in `quack/complex.py` or call `_retag_as_complex64(t)`
   manually.
2. **`isinstance(c, Float32) is True`.** Only one cutlass site checks this
   (`nvvm_wrappers.py:441`). Easy to grep for if behavior is suspicious.
3. **`is_same_kind(Complex64, Float32) is True`** because both have
   `is_float=True`. So `tensor[i] = some_float32` will auto-promote via
   `_cvt_to_dest` -> `Complex64(float32_instance)` -> `(re=value, im=0)`.
   That's friendly ergonomics for scalar real-valued writes, but if you
   accidentally store an f32 expecting "complex bits", it'll be silently
   wrapped. Add an explicit type check in `__setitem__` if you want to
   harden a particular kernel.
4. **`Float32 OP Complex64` only works** because `Complex64` is a strict
   subclass of `Float32`, triggering Python's reflected-operator precedence.
   Don't break the inheritance.
5. **`Numeric.from_mlir_type` is a single global registry.** Anything
   monkey-patching it for Complex64 → Float64 mapping must avoid disturbing
   the existing Float64 mapping (we don't currently patch it; we use the
   per-tensor re-tag instead).
6. **Dynamic Complex64 SSA values can't be passed as kernel args.**
   `__c_pointers__` raises if `self.value` is an SSA. Static (Python
   complex literal) values work fine. This is consistent with how `Float32`
   handles the same case.
7. **Twiddle generation builds an O(N) `arith.select` cascade** when the
   index is dynamic. For small N (8, 16, 32, 64) that's fine; for larger N
   use the LUT pattern via `_twiddle_from_lut_cx` (and friends:
   `_twiddle_cx`, `_twiddle_binary_cx`, `_twiddle_in_range_cx`) in
   `quack/fft.py`. These wrap the existing tuple-returning helpers and
   return a single `Complex64` value.

---

## File layout

```
quack/complex.py
  class Complex64(Float32, width=64, mlir_type=T.f64)
    __init__(x, im=None)        # complex / Float32 / Numeric / ir.Value(f32|f64)
    _pack_ssa(re, im)           # static: two f32 SSAs -> packed f64 SSA
    _unpack(self)               # f64 -> (Float32, Float32)
    from_re_im(re, im)          # public alias of _pack_ssa-based ctor
    real() / imag() / conj()
    __add__/__radd__/__sub__/__rsub__/__mul__/__rmul__/__neg__
    __c_pointers__              # 8-byte scalar-arg path

  allocate_smem_complex(allocator, layout)
  recast_to_complex64(src)
  complex_storage(t)            # torch.complex64 -> torch.float64 view

  _register_with_tvm_ffi()      # called at import time

quack/fft.py                    # Complex64-native FFT path; *_cx primitives
                                # (_fft{2,4,8,16,32}_inplace_cx*, _mul_j_cx,
                                # _apply_stage_twiddle_cx,
                                # _mul_by_base_twiddle_powers_cx,
                                # _twiddle{,_binary,_in_range,_from_lut}_cx,
                                # plus pure-Complex64 smem helpers)
```

```
tests/test_complex.py            # 44 tests for Complex64 the type
tests/test_fft.py                # FFT-correctness tests using Complex64

AI/complex64_design_notes.md     # this file (design rationale + sharp edges)
AI/complex64_fft_migration_plan.md # cutover history and remaining work
AI/fft_optimization_notes.md     # broader FFT perf notes
```
