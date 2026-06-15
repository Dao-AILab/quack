# Design — cute-DSL CUDA 13.3 toolchain shim

This document explains *why* the shim is shaped the way it is, the
runtime intercept strategy, and the validation/extension story for
future wheels.

## Problem

CUTLASS DSL ships an `_cutlass_ir.cpython-*.so` that statically links
**libnvvm 13.1** (LLVM IR → PTX) and **libnvptxcompiler 13.1** (PTX → cubin).
A QuACK process that wants to use a newer CUDA 13.3 toolchain — for
example for PTX 9.3 features, ptxas register-allocator improvements, or
fixes to a specific compiler bug — cannot just set `CUDA_HOME` or
`LD_LIBRARY_PATH`: the embedded toolchain is statically linked and the
PTX→cubin step happens entirely inside `_cutlass_ir.so`, with no
PLT/GOT indirection that `LD_PRELOAD` could intercept.

## Existing Python hook

`quack/dsl/cute_dsl_ptxas.py` is the legacy ptxas-only path. It
monkey-patches the CuTe DSL compile flow, forces PTX dumping, runs an
external `ptxas`, and swaps the resulting cubin bytes back into the
CUDA library loader. That path is useful as a pure-Python fallback, but
it cannot replace libnvvm: LLVM IR is still lowered to PTX by the
embedded 13.1 NVVM.

This shim replaces both stages: libnvvm (LLVM IR → PTX) and ptxas
(PTX → cubin).

## v1 design

A **single-file C shim** built into `libcute_dsl_shim.so`, **activated
from Python** via ctypes by `quack/__init__.py`. The shim is
**manual-build only**: `make -C tools/cute_dsl_shim` produces the local
`.so`, and `QUACK_CUTE_DSL_SHIM_LIB` can point at a manually installed
copy elsewhere. Package installs do not compile or bundle it. The C side
does not know any VAs; all per-wheel data lives in
`quack/dsl/cute_dsl_shim.py`'s `WHEEL_OFFSETS` dict and is passed in via a
struct.

```
┌──────────────────────────────────────────────────────────┐
│            quack/__init__.py                              │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  quack/dsl/cute_dsl_shim.py — try_activate():            │  │
│  │   1. import cutlass; locate _cutlass_ir.so           │  │
│  │   2. sha256(_cutlass_ir.so) → lookup WHEEL_OFFSETS   │  │
│  │   3. ctypes.CDLL("libcute_dsl_shim.so")              │  │
│  │   4. resolve libnvvm / ptxas paths                   │  │
│  │   5. cute_dsl_shim_install(base, &offsets, &config)  │  │
│  └─────────────────────────────────────────────────────┘  │
│                          │ ctypes call                     │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  libcute_dsl_shim.so                                 │  │
│  │   ┌─────────────────────────────────────────────┐    │  │
│  │   │ NVVM dispatch-table patch                   │    │  │
│  │   │  • dlopen("libnvvm.so.4") (13.3)            │    │  │
│  │   │  • dlsym 13 symbols → static table          │    │  │
│  │   │  • write *(uintptr_t*)(base+table_va)=table │    │  │
│  │   │  • write *(uint8_t*)(base+guard_va)=1       │    │  │
│  │   └─────────────────────────────────────────────┘    │  │
│  │   ┌─────────────────────────────────────────────┐    │  │
│  │   │ nvPTXCompiler 7-entry trampoline patch      │    │  │
│  │   │  • mprotect each entry page RWX             │    │  │
│  │   │  • write 12 bytes: movabs rax,addr; jmp rax │    │  │
│  │   │  • mprotect back to RX                      │    │  │
│  │   └─────────────────────────────────────────────┘    │  │
│  └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Why activate from Python instead of `LD_PRELOAD`

The user explicitly chose programmatic activation. There are real
advantages:

- **One single import surface.** Set `QUACK_CUTE_DSL_SHIM=1` and
  `import quack` activates the shim. No `LD_PRELOAD=…` wrapper script,
  no `patchelf` step, no need for write access to the cutlass-dsl
  install dir.
- **Deterministic timing.** `quack/__init__.py` runs *before* any
  `cute.compile()` invocation in user code. The libnvvm dispatch-table
  initializer is one-shot inside CuTe DSL — it builds the table on
  first compile. By activating before any compile happens, we win the
  race against the initializer trivially and can patch the dispatch
  table directly instead of rewriting every NVVM wrapper stub.
- **Pythonic error reporting.** Wheel mismatches, missing system CUDA,
  and ABI mismatches surface as `ShimError` when the shim is explicitly
  enabled, not silent garbage corruption.

## Why patch the NVVM table instead of NVVM stubs

| Aspect                              | dispatch-table patch                                      | stub patch                                      |
|-------------------------------------|-----------------------------------------------------------|-------------------------------------------------|
| What we touch                       | One pointer-sized word in `.data` + one byte              | 13+ function bodies in `.text`                  |
| `mprotect` churn                    | None on `.text` for NVVM                                  | Page permission flips for each stub             |
| Per-wheel discovery work            | 2 VAs (guard + table)                                     | One VA per NVVM wrapper stub                    |
| Lazy-init requirement               | Must install before the first compile                     | Can tolerate the wrapper initializer running    |
| Fit for current activation model    | Good: `quack/__init__.py` runs before any CuTe compile     | Unnecessary extra code mutation                 |

Because this shim is explicitly activated at import time, before any
CuTe compile, the table patch is the smaller and easier-to-validate
change. If a future activation mode needs to support late attachment
after compile activity has already begun, adding NVVM stub patches is a
reasonable extension.

## NVVM table slot order

The table order used by this shim is:

```
+0x38  nvvmVerifyProgram
+0x40  nvvmCompileProgram
```

That ordering is validated by end-to-end kernel compilation tests that
exercise the patched table. See the comment block above
`NVVM_TABLE_SLOTS[]` in `cute_dsl_shim.c`.

## Validation strategy

v1 keys per-wheel offsets by `sha256(_cutlass_ir.so)`. If the SHA does
not match any entry in `WHEEL_OFFSETS`, the loader **refuses to patch**
and raises `ShimError` when explicitly enabled with
`QUACK_CUTE_DSL_SHIM=1`. When the env var is unset, the loader is a
no-op. This is the only thing standing between a new CuTe DSL release
and arbitrary in-process memory corruption, so it is non-negotiable.

`QUACK_CUTE_DSL_SHIM_FORCE=1` exists as an emergency-only escape hatch
that re-uses the cp312 offsets on any wheel. It is documented as
**dangerous**.

A future improvement is a second layer of validation that disassembles
the bytes at each candidate VA and verifies them against known prologue
signatures for the symbol being intercepted (for example, the PTX
emission path and each `nvPTXCompiler*` entry).

## Coexistence with `quack/dsl/cute_dsl_ptxas.py`

`quack/__init__.py` calls `try_activate()` first. If it returns `True`,
the legacy Python hook is skipped — there is exactly one cubin-rewrite
path in the process. If the shim is off and `CUTE_DSL_PTXAS_PATH` is
set, the legacy hook installs as before, preserving the current
user-facing behavior for environments without the shim. If the shim is
explicitly enabled and a prerequisite is missing, activation raises
`ShimError` instead of falling through silently.

The legacy hook is **not** removed in v1 because:

- It is the documented `CUTE_DSL_PTXAS_PATH` UX users may already rely on.
- It works on wheels the shim doesn't recognize (no SHA gating).
- It only requires ptxas, not libnvvm 13.3.

Once we ship the shim and the offset table covers a comfortable range
of wheels, removing the legacy hook is a fine cleanup.

## ABI

`cute_dsl_shim_install()` takes two structs with explicit
`abi_version` fields. Bump `CUTE_DSL_SHIM_ABI_VERSION` and the matching
constant in `quack/dsl/cute_dsl_shim.py::ABI_VERSION` when fields change.
Old loaders + new `.so` (or vice versa) will refuse to bind with
`CUTE_DSL_SHIM_E_ABI`.

The seven `nvPTXCompiler*` functions reproduce the public C ABI of
NVIDIA's `libnvptxcompiler`. The opaque handle is a pointer to a
`shim_state` struct with a `'QUACKPTX'` magic sentinel — any out-of-spec
handle is rejected with `NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE`.

The 13 libnvvm pointers are `dlsym`'d *directly* from the loaded CUDA
13.3 `libnvvm.so.4`. No per-symbol wrapper, no per-call indirection —
the cute-DSL wrapper calls straight through into 13.3 code on the
first compile. Opaque-handle types (`nvvmProgram`) are never
dereferenced by the cute-DSL side, so the handle returned by 13.3's
`nvvmCreateProgram` is just shuttled back through subsequent calls
unchanged.

## Future work

1. **More wheels.** Add entries to `WHEEL_OFFSETS` for `cutlass-dsl`
   point releases beyond 4.5.2 as they ship. The recon scripts in
   `recon/` automate offset extraction; updating the dict is a
   ~10-minute task per wheel.
2. **Aarch64.** Add an aarch64 trampoline emitter (for example,
   `ldr x16; br x16; .quad target`), call `__builtin___clear_cache`
   after patching, add conditional compile paths in `cute_dsl_shim.c`,
   and store separate per-wheel offset records per arch.
3. **Signature-scan fallback.** Today an unknown wheel SHA fails the
   strict mode. A safer path is to keep the SHA fast-path but, on
   miss, run a small in-process disassembly scan keyed by the same
   `.rodata` anchor strings the `recon/` scripts use.
4. **Drop the Python legacy hook.** Once the wheel-offsets table is
   comprehensive enough that the shim handles every supported wheel,
   `quack/dsl/cute_dsl_ptxas.py` can be deleted. Until then it is the
   safety net.
5. **A `quack-run` CLI** that prepends shim activation and execv's
   into the user's command, for users who prefer not to `import quack`
   to trigger activation. (Note: the C ABI is stable enough that this
   can be a 50-line shell + ctypes script and doesn't need to touch
   the C side.)
