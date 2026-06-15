# cute-DSL CUDA 13.3 toolchain shim

Replace CUTLASS DSL's embedded CUDA 13.1 compiler stack (libnvvm + ptxas)
with the system CUDA 13.3 toolchain, at runtime, from inside the same
Python process — without modifying the CuTe DSL pipeline, without
`LD_PRELOAD`, and without touching `_cutlass_ir.so` on disk.

```text
                  cute IR  →  MLIR  →  LLVM IR
                                    │   ╲────────────╴ (libnvvm 13.3 via dispatch-table)
                                    ▼
                                   PTX
                                    │   ╲────────────╴ (ptxas 13.3 via trampolines)
                                    ▼
                                  cubin
```

End-to-end verification on this checkout::

    $ QUACK_CUTE_DSL_SHIM=0 python tools/cute_dsl_shim/verify.py
    ptxas 13.1 / V13.1.66  (kernel result max abs diff = 0.0)  shim_active=False

    $ QUACK_CUTE_DSL_SHIM=1 python tools/cute_dsl_shim/verify.py --require-shim
    ptxas 13.3 / V13.3.33  (kernel result max abs diff = 0.0)  shim_active=True

## Build

The shim is **manual-build only**. Package installs do not compile or
bundle the native `.so`.

```bash
make -C tools/cute_dsl_shim          # produces tools/cute_dsl_shim/libcute_dsl_shim.so
make -C tools/cute_dsl_shim verify   # builds + runs verify.py
```

If you place the `.so` somewhere else, point the loader at it with
`QUACK_CUTE_DSL_SHIM_LIB=/absolute/path/to/libcute_dsl_shim.so`.

## Activation

The shim is activated from Python, from inside `quack/__init__.py`. There
is **no `LD_PRELOAD` and no `patchelf` step**. The activation policy is
controlled by `QUACK_CUTE_DSL_SHIM`:

| Value             | Behavior                                                          |
|-------------------|-------------------------------------------------------------------|
| unset / `0` / `off` | Skip the shim; use the legacy `cute_dsl_ptxas.py` hook if `CUTE_DSL_PTXAS_PATH` is set. |
| `1` / `on`        | Activate; raise `ShimError` if anything is missing.                |

If the shim activates successfully, the Python hook is suppressed: there
is exactly one cubin-rewrite path in the process.

## Environment knobs

| Variable                       | Default                                          | Purpose                                                                |
|--------------------------------|--------------------------------------------------|------------------------------------------------------------------------|
| `QUACK_CUTE_DSL_SHIM`          | unset / off                                      | Activation policy (see above).                                         |
| `QUACK_CUTE_DSL_SHIM_LIB`      | `<repo>/tools/cute_dsl_shim/libcute_dsl_shim.so` | Absolute path to the shim `.so`. |
| `QUACK_CUTE_DSL_SHIM_PTXAS`    | `/usr/local/cuda/bin/ptxas`                      | External ptxas binary the shim fork/execs.                             |
| `QUACK_CUTE_DSL_SHIM_LIBNVVM`  | `/usr/local/cuda-13.3/.../libnvvm.so.4` (preferred), else `/usr/local/cuda/.../libnvvm.so.4` | libnvvm.so.4 to forward NVVM table slots to. |
| `QUACK_CUTE_DSL_SHIM_NO_NVVM`  | unset                                            | Set to `1` to only patch ptxas (keep embedded libnvvm 13.1).           |
| `QUACK_CUTE_DSL_SHIM_NO_PTXAS` | unset                                            | Set to `1` to only patch libnvvm (keep embedded ptxas 13.1).           |
| `QUACK_CUTE_DSL_SHIM_DEBUG`    | unset                                            | Set to `1` for verbose stderr tracing.                                 |
| `QUACK_CUTE_DSL_SHIM_FORCE`    | unset                                            | Bypass the SHA256 wheel-validation check. Dangerous — only use if you've manually verified the wheel matches a known one. |

## Supported wheels

`nvidia-cutlass-dsl-libs-cu13 == 4.5.2`, x86_64 Linux, CPython 3.10 through 3.14
(including 3.14 free-threaded). The shim refuses to install on any other
wheel SHA256 unless `QUACK_CUTE_DSL_SHIM_FORCE=1` is set. See
`quack/dsl/cute_dsl_shim.py::WHEEL_OFFSETS` for the table.

Aarch64 and other architectures are deferred. See `DESIGN.md` § "Future work".

## How it works (short version)

Two intercepts, both performed in `cute_dsl_shim_install()` after
`_cutlass_ir.so` has been loaded and its base address located from
`/proc/self/maps`:

1. **libnvvm dispatch-table patch.** CuTe DSL's libnvvm wrapper builds a
   13-slot function-pointer table lazily on first compile. We allocate a
   process-lifetime table of the same shape, populate each slot with the
   corresponding `nvvm*` symbol resolved by `dlopen("libnvvm.so.4") +
   dlsym`, write the table-pointer global with our address, and set the
   guard byte. The wrapper's pthread_once-like initializer then sees
   `guard != 0` and never rebuilds.

2. **nvPTXCompiler entry-point trampolines.** Each of the seven public
   API functions (`Create`, `Compile`, `Destroy`, `GetCompiledProgram{Size,}`,
   `GetErrorLog{Size,}`) gets a 12-byte `movabs rax, addr; jmp rax`
   prologue rewrite, routing it into a small in-process shim that
   fork/execs the system `ptxas` and returns its cubin through the same
   opaque-handle ABI.

See `DESIGN.md` for the full rationale and validation strategy.

## Caveats and limitations

- **x86_64 Linux only** (v1). Aarch64 trampolines and `dl_iterate_phdr`-
  based base discovery are not implemented yet.
- **SHA256-gated.** Adding a new wheel requires re-running the recon
  scripts (`recon/`) and adding an entry to `WHEEL_OFFSETS`. The shim
  refuses to patch unknown wheels by default.
- **ptxas adds ~5–15 ms of fork/exec overhead per fresh kernel compile.**
  Negligible compared to NVVM time; cache hits skip both entirely.
- The C shim allocates and never frees its libnvvm table and per-handle
  state. This is intentional — the patches live for the process lifetime.
- The shim mutates `.text` pages of `_cutlass_ir.so` in place. Install
  exactly once, early, before any concurrent compile activity. The
  per-process Python loader handles this correctly (one `try_activate()`
  call from `quack/__init__.py`).

## File layout

```
tools/cute_dsl_shim/
├── README.md              ← this document
├── DESIGN.md              ← architecture, recon, ABI, validation strategy
├── Makefile               ← build, verify, clean
├── cute_dsl_shim.c        ← single-file C source (~800 LoC, builds to ~30 KB .so)
├── verify.py              ← end-to-end smoke test
└── recon/
    ├── README.md          ← how to derive offsets for a new wheel
    ├── find_ptxas_entries.py    ← anchor → LEA → CALL pipeline for 7 entries
    └── find_libnvvm_table.py    ← locate the dispatch-table guard/pointer

quack/
├── dsl/
│   ├── cute_dsl_shim.py   ← Python loader; per-wheel WHEEL_OFFSETS table
│   └── cute_dsl_ptxas.py  ← legacy ptxas-only Python hook (fallback)
└── __init__.py            ← prefers manually-built shim, falls back to legacy hook
```
