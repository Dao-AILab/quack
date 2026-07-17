This file provides guidance to coding agents (Claude Code, Codex) when working with code in this repository.

## Project Overview

QuACK (Quirky Assortment of CuTe Kernels) — high-performance CUDA kernels written in [CuTe-DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html), targeting H100 (SM90), B200/B300 (SM100), and GeForce RTX 50 (SM120) GPUs. Package name: `quack-kernels`.

## Build & Development

```bash
# Install (dev)
pip install -e '.[dev]'
pre-commit install

# For CUDA 13.x
pip install -e '.[dev,cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Lint & format
ruff check --fix quack/ tests/ benchmarks/
ruff format quack/ tests/ benchmarks/

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_rmsnorm.py -x
pytest tests/test_rmsnorm.py::test_rmsnorm_fwd -x -k "bfloat16"

# After editing kernel source (cold .o cache): overlap compiles with tests.
# Misses are compiled by a pool of N CPU workers (forkserver sidecar) while
# deferred tests retry once their .o lands. No-op when the cache is warm.
pytest tests/test_rmsnorm.py --async-compile=16
pytest tests/ -n 8 --async-compile=32
```

## CuTe DSL Conventions

For code inside `@cute.jit` or `@cute.kernel` decorated functions, only a subset of Python syntax is supported. Follow [Control Flow](docs/dsl_control_flow.rst) and [Limitations](docs/limitations.rst).

Key rules:
- `cutlass.const_expr()` marks compile-time constants; `cutlass.range_constexpr()` unrolls loops at compile time
- `cutlass.range()` is for dynamic runtime loops
- No early `break`/`continue` in loops; no `return` values from jit functions
- Python lists/dicts inside DSL are static (compile-time only)
- Types must be determinable at compile time (no dependent types)
- Variables defined inside control flow bodies are not accessible outside

**Note:** CuTe DSL relies on Python source inspection `inspect.getsourcelines()` to parse kernel definitions. Defining `@cute.kernel` / `@cute.jit` functions directly in plain Python REPL will cause source inspection to fail with errors like `OSError: could not get source code`. One should write to a temporary file.

## Architecture

### Kernel patterns

**Reduction kernels** (`rmsnorm.py`, `softmax.py`, `cross_entropy.py`) inherit from `ReductionBase` in `reduction_base.py`. They share a pattern: configure cluster size, get tiled copies, allocate reduction buffers with mbarriers, then launch a `@cute.kernel`.

**GEMM** has a multi-layer design:
- `gemm.py` — public API, validates inputs, selects SM version, caches compiled kernels
- `gemm_interface.py` — unified interface across SM versions
- `gemm_sm90.py` / `gemm_sm100.py` — SM-specific implementations
- `gemm_default_epi.py` + `gemm_*_epi.py` — epilogue variants (bias, activation, etc.)
- `gemm_config.py` — `GemmConfig` dataclass with tile sizes, cluster dims, swizzle settings

### Core utilities

- `copy_utils.py` — memory copy operations (shared↔register, async copies, tiled copies)
- `layout_utils.py` — layout algebra (transpose, select, expand, permute)
- `cute_dsl_utils.py` — dtype mapping, device capability queries, parameter base classes
- `tile_scheduler.py` — tile scheduling for persistent kernels
- `varlen_utils.py` — variable-length sequence support

### Testing

Tests use pytest with parametrize across dtypes (`float32`, `float16`, `bfloat16`), dimensions, and batch sizes. Each test includes a reference implementation for numerical validation.

Every test must verify **numerical correctness** against a reference, not just shapes or smoke. A test that only checks `.shape` or "doesn't crash" is not a test — it hides bugs. Always compare kernel output values against a PyTorch reference (float32 for ground truth, same dtype for tolerance baseline).

## Iteration Speed

When iterating on kernel code, run a small subset of tests (1-3 parametrizations) rather than the full test suite. Use `-k` or pass specific test IDs to pytest. Only run the full suite when finalizing changes.

## Debugging Failures

When debugging any failure — kernel correctness, torch.compile interaction, test infrastructure — get to the bottom of it. The goal is not to make the test pass; it is to understand **why** it fails and fix the actual cause.

Do not route around the bug just to make a test pass, for example by pruning a config, skipping a path, resetting state, increasing limits, or switching to a different implementation. A workaround that makes CI green but leaves the bug for users is worse than useless — it hides the problem and the deeper issue will eventually surface at greater cost. Only use workarounds after proving the root cause is external (e.g., upstream PyTorch bug) and documenting why.

Start by reproducing the reported failure, then simplify it to the smallest setting that still fails: reduce batch, M/N/K, tile shape, scheduler options, `swap_ab`, beta/C, dtype, and epilogue features where possible. Keep the failing behavior intact while removing unrelated complexity.

Use `cute.printf` inside `@cute.jit` / `@cute.kernel` code to print the relevant locations, tile coordinates, tensor coordinates, predicates, and values. Print at the boundaries between stages, for example TMA load, MMA accumulator, epilogue register values, register-to-smem, smem contents, and TMA store coordinates, until the first bad stage is identified.

After finding a fix, verify that the minimized repro passes, the original repro passes, and that temporarily disabling the fix makes the regression test fail. Regression tests should encode the failure mode, not only the high-level symptom.

## Code Style

- Favor concise, self-explanatory code
- Line length: 100 (ruff)
- Ruff allows: lambda assignment (E731), single-char vars I/O/l (E741), unused locals (F841)

## Measured facts & gotchas (hard-won; verify before assuming they changed)

Hardware (B300 / SM100):
- Dense bf16 peak is 2250 TFLOPS — never use the sparse marketing number.
- L1TEX: gmem and smem share the 128 B/clk/SM pipe; 16-bit kernels with smem
  exchanges are smem/latency-bound, not DRAM-bound (clone ceiling ~6.3 TB/s).
- Blockscaled SF tmem is 64-N granular at both UTCCP write and QMMA read:
  tile_n that is a 32-multiple but not 64 (96, 224) is hardware-blocked.

Benchmarking (see tools/matmul_heuristic/common.py for the reference protocol):
- Per-launch CUDA event pairs on a starved stream time the host enqueue gap,
  not the kernel — any sub-100us kernel needs a GPU-side backlog (burner).
  Consecutive kernels chain L2 state, so rotate config order across rounds.
- On shared nodes: interleave variants at single-launch granularity and
  compare medians; sequential do_bench rounds drift 2-3x with clocks and
  co-tenants. Guard long benches with a contention canary.
- Short benches ride boost clocks (~15-25% fast vs settled); input data
  distribution shifts the settled clock (randn vs small-int vs zeros differ
  up to ~40%) — match warmup AND data distribution across baselines.
- Pin the CPU (taskset) when launch overhead matters; the Python/FFI launch
  path floor is ~3.5us and drifts 2-3x unpinned.

CuTe-DSL:
- Dtype views of swizzled smem drop the swizzle: re-attach the SAME
  byte-addressed swizzle via `recast_ptr`, never plain `recast_tensor`.
- TensorSSA `.store` on all-zero-stride filtered rmem views is silently
  dropped — use scalar setitem.
- Data-independent per-element selects hoist above the load and spill
  registers; make both select arms consume the loaded value.
- Never `partition_D` a tile smaller than the epilogue tiler (warp
  corruption); derive extents from the copy atom.
- `QUACK_CACHE_ENABLED=0` for const_expr ablations — the jit cache ignores
  env flags, and identical timings across an ablation mean a stale cubin.
