# TVM-FFI Compilation in CuTe DSL

## Overview

CuTe DSL kernels compiled with `--enable-tvm-ffi` use a two-phase approach:
1. **Compile time**: Build MLIR from fake (symbolic) tensors, lower to PTX
2. **Call time**: Pass real tensors; TVM-FFI extracts pointers and validates shapes

The compiled function is cached via `@jit_cache` (in-memory dict + filesystem `.o` cache)
so recompilation only happens for new dtype/major/tile/activation combinations.

## Symbolic Shapes and `sym_int`

Fake tensors use `cute.sym_int()` for symbolic dimensions. **Reusing the same
`sym_int` across tensors tells TVM-FFI that those dimensions must match at
runtime.**

```python
m, n, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()

# All share the same 'm' — TVM-FFI enforces D.shape[0] == A.shape[0] == ColVec.shape[0]
mA = fake_tensor(a_dtype, (m, k, l), leading_dim=..., divisibility=...)
mD = fake_tensor(d_dtype, (m, n, l), leading_dim=..., divisibility=...)
mColVec = fake_tensor(colvec_dtype, (m,), leading_dim=0, divisibility=4)
```

`make_fake_tensor` (in `compile_utils.py`) creates a tensor with:
- Symbolic shape dims (`sym_int`)
- Symbolic strides (`sym_int64` with divisibility hints), except `1` for the leading dim
- Alignment assumptions derived from divisibility and dtype width

### Varlen case

When `varlen_m=True`, there is no batch dimension `l` for A/D/C. The M dimension
becomes `total_m` (sum of all sequence lengths). We reassign `m = cute.sym_int()`
so that callers still get the correct sym_int from the return value:

```python
if varlen_m:
    # m is total_m: the flattened M dimension of D/C
    m = cute.sym_int()
    a_m = cute.sym_int() if gather_A else m  # A may have different row count
    mA = fake_tensor(a_dtype, (a_m, k), ...)
    mD = fake_tensor(d_dtype, (m, n), ...)
```

The `mAIdx` tensor (gather indices) reuses `m` (= `total_m`) since it has one
entry per output row.

### Symmetric GEMM

For symmetric GEMM where `m == n`, we use the same sym_int for both:

```python
m, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int()
mA = fake_tensor(a_dtype, (m, k, l), ...)
mD = fake_tensor(d_dtype, (m, m, l), ...)  # square output
```

## NamedTuples as Kernel Arguments

Epilogue arguments, scheduler options, and varlen arguments are NamedTuples
decorated with `@mlir_namedtuple`. This decorator adds `__new_from_mlir_values__`
so the TVM-FFI runtime can reconstruct the struct from flat MLIR values.

```python
@mlir_namedtuple
class EpilogueArguments(NamedTuple):
    mPostAct: cute.Tensor
    act_fn: cutlass.Constexpr[Callable] = None   # baked in at compile time
    alpha: Optional[Float32 | cute.Tensor] = None
    beta: Optional[Float32 | cute.Tensor] = None
```

### How reconstruction works

`_namedtuple_new_from_mlir_values(self, values)`:
1. Iterates fields of the compile-time template (`self`)
2. `None` and `StaticTypes` (int, bool, str, Constexpr, ...) are preserved as-is
3. Complex types (cute.Tensor, pointers) consume N MLIR values via
   `cutlass.new_from_mlir_values(field_val, values[:n_items])`
4. Returns a new NamedTuple instance with reconstructed fields

## Constexpr Fields

Fields annotated `cutlass.Constexpr[T]` are compile-time constants — their values
are baked into the compiled kernel. At call time, pass `None` for these fields.

This is enabled by a monkey-patch on the TVM-FFI argument converter:

```python
def _patched_convert_single_arg(arg, arg_name, arg_type, ctx):
    if arg_type is not None and get_origin(arg_type) is cutlass.Constexpr:
        return spec.ConstNone(arg_name)  # not a runtime arg
    ...
```

Example: `act_fn` is `Constexpr[Callable]`, so the activation function is compiled
into the kernel. Different activations produce different compiled kernels.

### NamedTuple type hint workaround

When a NamedTuple is passed as a `tuple`-annotated parameter (e.g.
`epilogue_args: tuple`), the TVM-FFI converter doesn't see the NamedTuple's field
types. The patch redirects to use `type(arg)` (the actual NamedTuple class):

```python
if (isinstance(arg, tuple) and hasattr(type(arg), "_fields")
        and (arg_type is None or not hasattr(arg_type, "_fields"))):
    return _original_convert_single_arg(arg, arg_name, type(arg), ctx)
```

## ParamsBase (Dataclass Arguments)

`ParamsBase` is the dataclass counterpart for kernel parameters that go through
`epi_to_underlying_arguments` (the JIT-level struct, not the user-facing args).
It partitions fields into constexpr (static) and non-constexpr:

```python
@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        _, non_constexpr_fields = _partition_fields(self)
        values, self._values_pos = [], []
        for obj in non_constexpr_fields.values():
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values
```

`_values_pos` tracks how many MLIR values each field consumed, so
`_new_from_mlir_values` can reconstruct fields in the right order.

## Compilation and Caching Flow

```
gemm_act(A, B, D, ...)          # user-facing API
  │
  ├─ Extract dtypes, majors from real tensors
  ├─ _compile_gemm_act(...)     # @jit_cache — in-memory + disk cache
  │    ├─ make_fake_gemm_tensors(...)  # symbolic tensors with shared sym_ints
  │    ├─ Build fake epilogue args, scheduler args, varlen args
  │    └─ compile_gemm_kernel(...)
  │         └─ cute.compile(gemm_obj, *fake_args, options="--enable-tvm-ffi")
  │
  ├─ Build runtime epilogue args (real tensors, None for Constexpr fields)
  ├─ Build runtime scheduler/varlen args
  └─ compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args)
```

The `@jit_cache` key is derived from the function's args (dtypes, majors, tile sizes,
cluster dims, activation, etc.). Same args → same compiled kernel.

## Key Files

| File | Role |
|------|------|
| `compile_utils.py` | `make_fake_tensor` — creates symbolic tensors |
| `gemm_tvm_ffi_utils.py` | `make_fake_gemm_tensors`, `compile_gemm_kernel`, varlen/scheduler helpers |
| `cute_dsl_utils.py` | `@mlir_namedtuple`, `ParamsBase`, Constexpr converter patch |
| `cache_utils.py` | `@jit_cache` decorator — in-memory + filesystem `.o` caching |
