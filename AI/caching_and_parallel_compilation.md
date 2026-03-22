# Caching and Parallel Compilation

QuACK compiles CUDA kernels at runtime via CuTe DSL. Each compilation generates MLIR IR,
lowers to PTX, and JIT-links — taking ~100ms per kernel. Since a single test run or training
step may trigger dozens of compilations (different dtypes, tile sizes, activations), naive
serial compilation dominates wall-clock time. This doc explains the multi-layer caching and
parallel compilation design that makes it fast.

## Layer 1: `@jit_cache` Decorator (`cache_utils.py`)

Each `_compile_*` function (e.g. `_compile_gemm` in `gemm.py`, `_compile_softmax_fwd` in
`softmax.py`) is decorated with `@jit_cache`. This single decorator provides both in-memory
and persistent disk caching:

```python
@jit_cache
def _compile_softmax_fwd(dtype, out_dtype, N):
    # ... build fake tensors ...
    return cute.compile(softmax_op, ...)
```

### In-Memory Cache

A per-function `dict` keyed on the function's positional/keyword args. Within a single
process, calling the same kernel config twice skips compilation and disk I/O entirely.

### Disk Cache

On in-memory miss, checks for a cached `.o` (object file) on disk. The disk key is
`(fn.__qualname__, *args, **sorted_kwargs)`, hashed with SHA-256. The cache directory
includes a **source fingerprint** — a SHA-256 of all `quack/*.py` files plus
Python/CUTLASS/TVM-FFI versions. Any source change invalidates the entire cache.

With cutlass-dsl >= 4.4.2, `tvm_ffi` can load `.o` files directly — no need to link
`.o` → `.so` via distutils. This eliminates the `distutils.ccompiler` dependency.

### Cache Miss Flow

```
@jit_cache wrapper(dtype, out_dtype, N)
  ├─ In-memory dict lookup → hit ✓ (~0ns)
  │
  ├─ hash (fn.__qualname__, *args) → SHA-256 hex
  ├─ Shared lock: check if .o exists → load via tvm_ffi (~1ms) ✓
  │
  ├─ Cache miss → call fn(dtype, out_dtype, N)
  │    └─ cute.compile(...) → MLIR → PTX → binary
  │
  └─ Exclusive lock: export_to_c() → write .o to cache
```

### Concurrency Safety

Multiple processes may compile the same key simultaneously (e.g. parallel test workers).
`FileLock` (using `fcntl.flock`) serializes access:

- **Shared lock** for reads — multiple readers can load `.o` concurrently
- **Exclusive lock** for writes — only one writer exports `.o`, others wait
- **Double-check**: after acquiring exclusive lock, re-check if `.o` already exists
  (another process may have written it while we were compiling)

### Environment Controls

| Variable | Default | Description |
|---|---|---|
| `QUACK_CACHE_ENABLED` | `1` | Set to `0` to disable disk cache (in-memory cache still active) |
| `QUACK_CACHE_DIR` | `/tmp/$USER/quack_cache` | Override cache location |

### Before `@jit_cache` (Historical)

Previously, caching was split across two layers:
- `@lru_cache(maxsize=None)` on each `_compile_*` function (in-memory)
- `compile_and_cache(key, compile_fn)` called inside the function body (disk)

Each call site manually constructed a `key` tuple and wrapped compilation in an inner
`_compile()` closure. `@jit_cache` unifies both layers into a single decorator,
eliminating the manual key construction and closure boilerplate.

## Layer 2: Autotuning Result Cache (`autotuner.py`)

The autotuner benchmarks many configs (e.g. ~44 for GEMM) to find the fastest. Results are
cached to disk as JSON via `FileCacheManager` (Triton's cache infrastructure).

```
Autotuner.__call__()
  ├─ Compute tuning key from tensor shapes/strides/dtypes
  ├─ check_disk_cache(key, configs, benchmark_fn)
  │    ├─ Hit: load best config from JSON ✓
  │    └─ Miss: run benchmark_fn(), save JSON
  └─ Call kernel with best config
```

The JSON cache key includes: package version, tuning key (tensor metadata), and all config
strings. Invalidated by changing `__version__` or configs.

| Variable | Default | Description |
|---|---|---|
| `QUACK_CACHE_AUTOTUNING` | unset | Set to `1` to enable disk caching of tuning results |
| `QUACK_FORCE_CACHE_UPDATE` | unset | Set to `1` to ignore cached tuning results |

## Parallel Compilation During Autotuning

When the autotuner encounters a cache miss, it needs to compile all candidate configs before
benchmarking. This is where parallel compilation becomes critical.

### The Constraints

Two facts make naive parallelism impossible:

1. **`cute.compile()` is not thread-safe** — MLIR uses thread-local state, so
   `ThreadPoolExecutor` causes data races.

2. **`fork()` after CUDA init segfaults** — the parent process has already called
   `torch.cuda.init()` (or any CUDA op). `fork()` copies CUDA driver state into the child,
   but driver handles are invalid in the child process, causing segfaults or hangs.
   `multiprocessing.fork` and `os.fork()` both hit this. `mp.set_start_method("spawn")`
   avoids it but `multiprocessing.Pool` with spawn has high per-task overhead.

### Solution: Persistent Subprocess Workers

`Autotuner._precompile()` spawns workers via `subprocess.Popen()` — always a fresh process
(like `spawn`), never `fork`, so no inherited CUDA state.

```
Parent process (CUDA initialized, has real tensors)
  │
  ├─ subprocess.Popen("python -m quack._compile_worker")  ← worker 0
  ├─ subprocess.Popen("python -m quack._compile_worker")  ← worker 1
  └─ ...up to QUACK_COMPILE_WORKERS (default 8)
```

Each worker (`_compile_worker.py`):
1. Sets `COMPILE_ONLY = True` (compilation produces `.o` but never launches kernels)
2. Signals `"READY"` to parent
3. Loops reading tasks from stdin (length-prefixed pickle protocol)
4. Creates `FakeTensor`s matching parent tensor metadata (shape/stride/dtype, no GPU memory)
5. Calls the kernel function under `FakeTensorMode()` → triggers `@jit_cache` →
   exports `.o`
6. Stays alive for the next task (amortizes `import quack` overhead, ~2-3s)

The parent does round-robin dispatch, collects acks, then closes stdin to shut workers down.

### Quick-Check Optimization

Before spawning workers, the parent compiles the first config in-process. If this completes
in <0.5s, the `.o` cache is likely warm — skip parallel compilation entirely:

```python
t_check = time.time()
self.fn(*args, **configs[0].all_kwargs())
if time.time() - t_check < 0.5:
    return  # cache is warm, no need for workers
```

### FakeTensors in Workers

Workers don't need real GPU memory. They serialize tensor metadata
(shape, stride, dtype) from the parent, then reconstruct with:

```python
def _make_fake_tensor(meta):
    return torch.empty_strided(shape, stride, dtype=dtype, device="cuda")
```

Under `FakeTensorMode`, this creates a CPU-side metadata object, no allocation.
Compilation only needs metadata (shapes, strides, dtypes) to generate correct code.

### The CUDA Init Ordering Gotcha

`torch.cuda.init()` must happen *before* entering `FakeTensorMode`:

- `FakeTensorMode` intercepts tensor creation, but CUDA init is a one-time driver call,
  not a tensor op.
- If FakeTensorMode is entered first, CUDA init either doesn't happen or happens in a
  broken state.
- In `_compile_worker.py`, `torch.empty_strided(..., device="cuda")` triggers CUDA init
  on first call — this works because it's a fresh subprocess and FakeTensorMode intercepts
  the allocation after init completes.

## The `COMPILE_ONLY` Flag

`cache_utils.COMPILE_ONLY` is a global boolean (default `False`). When `True`:

1. `@jit_cache` still compiles and exports `.o`, but returns `_noop_kernel`
   instead of the real compiled function
2. Early returns in `gemm.py`, `gemm_act.py`, `gemm_dact.py`, `gemm_symmetric.py` after
   compilation prevents reaching runtime code that calls `data_ptr()` (which crashes on
   FakeTensors)

```python
# In gemm.py — the critical boundary
compiled_fn = _compile_gemm(...)   # ← compilation happens here

from quack.cache_utils import COMPILE_ONLY
if COMPILE_ONLY:
    return                          # ← avoid data_ptr() below

max_active_clusters = get_max_active_clusters(...)
def scalar_arg(scalar, mode):
    if mode == 2:
        return scalar.data_ptr()   # ← would crash on FakeTensor
```

## Two-Pass Test Workflow (`conftest.py`)

The same machinery enables parallel test compilation:

```bash
# Pass 1: compile all kernels in parallel (no GPU memory needed)
pytest tests/test_softmax.py --compile-only -n 64

# Pass 2: run tests (instant .o cache hits)
pytest tests/test_softmax.py
```

### How `--compile-only` Works

`conftest.py:pytest_configure()`:
1. Sets `COMPILE_ONLY = True`
2. Calls `torch.cuda.init()` (before FakeTensorMode — critical ordering)
3. Enters `FakeTensorMode` globally
4. Swallows all test errors via `pytest_runtest_call` hook (we only care about compilation)

### Multi-GPU Round-Robin

When using `pytest-xdist` with multiple GPUs, `conftest.py` assigns workers round-robin:
```python
worker_num = int(worker_id.replace("gw", ""))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_num % len(gpu_ids)]
```

Worker 0 discovers available GPUs and writes to a shared JSON file; other workers read it.

### The `custom_op` / `register_fake` Requirement

All kernels use `@torch.library.custom_op(device_types="cuda")`. Under `FakeTensorMode`,
PyTorch dispatches to the op's `register_fake` implementation instead of the real CUDA
function body. Without an explicit `register_fake`, PyTorch auto-generates a no-op for
`mutates_args` ops — the real function body never executes, so **0 compilations** happen
during `--compile-only`.

Every `custom_op` therefore needs an explicit `register_fake` that triggers compilation
when `COMPILE_ONLY` is set. The pattern is the same across all kernels:

```python
@_softmax_fwd.register_fake
def _softmax_fwd_fake(x, out):
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not isinstance(x.size(1), torch.SymInt):
        N = x.size(1)
        dtype, out_dtype = [torch2cute_dtype_map[t.dtype] for t in [x, out]]
        _compile_softmax_fwd(dtype, out_dtype, N)
        _compile_softmax_backward(dtype, out_dtype, out_dtype, N)
```

Key details:

- **`not isinstance(..., torch.SymInt)` guard**: Under `torch.compile`, dynamo traces
  with symbolic shapes where dimensions are `SymInt`. We must not compile with SymInts
  (they crash `@jit_cache` and produce invalid code). `COMPILE_ONLY` mode uses concrete shapes.

- **Non-GEMM kernels** (softmax, rmsnorm, cross_entropy, topk, causal_conv1d) have
  straightforward `register_fake` — they call `_compile_*()` directly with the concrete
  shapes from the fake tensors.

- **GEMM kernels** are more involved because they go through the autotuner. A generic
  helper `_register_precompile_fake()` handles the boilerplate: it uses `inspect.signature`
  to rebind positional args by name (PyTorch normalizes all `custom_op` args to positional),
  then calls `_precompile_default_config()`.

### GEMM-Specific `register_fake` Details

`_precompile_default_config()` calls `autotuned_fn.fn(*args, config=None, **kwargs)`.
`config=None` selects the default config (128x192 for SM90, 256x256 for SM100). Tests use
`tuned=False` which bypasses the autotuner and uses the same default, so this is sufficient.

`gemm_symmetric_out` is special: it's not autotuned, so its `register_fake` calls
`gemm_symmetric_sm90_sm100()` directly with fixed tile parameters.

Ops using `_register_precompile_fake`:
- `gemm_out`, `gemm_add_out`, `gemm_add_inplace_op` — all route through `gemm_tuned`
- `gemm_act_out` — routes through `gemm_act_tuned`
- `gemm_dact_out` — routes through `gemm_dact_tuned`
- `gemm_gated_out` — routes through `gemm_gated_tuned`
- `gemm_dgated_out` — routes through `gemm_dgated_tuned` (also returns a tensor for
  `colvec_reduce`, so its `register_fake` was already needed for shape inference)
- `gemm_symmetric_out` — calls `gemm_symmetric_sm90_sm100` directly (manual `register_fake`)

### Why Early Return After Compilation

The `COMPILE_ONLY` early return in low-level gemm functions (`gemm.py`, `gemm_act.py`, etc.)
prevents reaching runtime code that:
- Calls `data_ptr()` on tensors (crashes on FakeTensors)
- Creates `scalar_arg` from tensor pointers
- Builds runtime `TileSchedulerOptions` with real `data_ptr()` values

```python
# gemm.py — the critical boundary
compiled_fn = _compile_gemm(...)   # ← compilation happens here

from quack.cache_utils import COMPILE_ONLY
if COMPILE_ONLY:
    return                          # ← avoid data_ptr() below

max_active_clusters = get_max_active_clusters(...)
def scalar_arg(scalar, mode):
    if mode == 2:
        return scalar.data_ptr()   # ← would crash on FakeTensor
```

## Complete Data Flow

### Normal Execution (Training/Inference)

```
User calls gemm(A, B, ...)
  └─ gemm_out (custom_op, dispatches to CUDA impl)
       └─ gemm_tuned (autotuner)
            ├─ Cache hit → call fn with best config
            └─ Cache miss → _precompile() + benchmark all configs
                 ├─ Spawn subprocess workers
                 ├─ Workers: FakeTensor + COMPILE_ONLY → .o files
                 ├─ Parent: benchmark each config (instant .o loads)
                 └─ Cache best config
            └─ fn(A, B, out, config=best)
                 └─ gemm_sm90_sm100(...)
                      ├─ _compile_gemm(...) → @jit_cache → .o
                      └─ compiled_fn(real_A, real_B, real_D, ...)
```

### `--compile-only` Test Mode

```
pytest --compile-only
  └─ conftest: COMPILE_ONLY=True, torch.cuda.init(), FakeTensorMode.__enter__()
       └─ Test creates FakeTensors (no GPU memory)
            └─ gemm(A_fake, B_fake, ...)
                 └─ gemm_out (custom_op → dispatches to register_fake)
                      └─ gemm_out_fake(A, B, out, ...)
                           └─ _precompile_default_config(gemm_tuned, ...)
                                └─ gemm_tuned.fn(A, B, out, config=None, ...)
                                     └─ gemm_sm90_sm100(...)
                                          ├─ _compile_gemm(...) → .o exported
                                          └─ COMPILE_ONLY → return (no launch)
```

### `--compile-only` then Normal Run

```
# Pass 1: all .o files written to cache
pytest --compile-only -n 64   # 64 workers, each compiles its test cases

# Pass 2: instant cache hits
pytest
  └─ _compile_gemm(...) → @jit_cache in-memory miss → disk hit
       └─ Shared lock: .o exists → load via tvm_ffi (~1ms) ✓
```

## Key Files

| File | Role |
|------|------|
| `cache_utils.py` | `@jit_cache` decorator (in-memory + `.o` disk cache), `FileLock`, `COMPILE_ONLY` flag |
| `compile_utils.py` | `make_fake_tensor()` — symbolic CuTe tensors for compilation |
| `gemm_tvm_ffi_utils.py` | `make_fake_gemm_tensors()`, `compile_gemm_kernel()` |
| `autotuner.py` | `Autotuner._precompile()` — subprocess worker pool, `FileCacheManager` |
| `_compile_worker.py` | Persistent subprocess: `FakeTensorMode` + `COMPILE_ONLY` loop |
| `gemm_interface.py` | `_register_precompile_fake()`, `_precompile_default_config()` for GEMM `custom_op`s |
| `softmax.py`, `rmsnorm.py`, `cross_entropy.py`, `topk.py`, `causal_conv1d.py` | `register_fake` with `COMPILE_ONLY` compilation |
| `conftest.py` | `--compile-only` pytest plugin, GPU round-robin, `FakeTensorMode` setup |
| `gemm.py` / `gemm_act.py` / `gemm_dact.py` / `gemm_symmetric.py` | `COMPILE_ONLY` early returns |
