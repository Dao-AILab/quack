# Caching and Parallel Compilation

QuACK compiles CUDA kernels at runtime via CuTe DSL. Each compilation generates MLIR IR,
lowers to PTX, and JIT-links — taking ~0.5–1 s for reduction-style kernels and 10 s+ for
large GEMM configs. A test run or autotune sweep triggers dozens-to-hundreds of
compilations (different dtypes, tile sizes, activations), so naive serial compilation
dominates wall-clock time. This doc explains the caching + async compilation design.

The design in one sentence: **on a `.o`-cache miss, the compile is shipped to a pool of
GPU-blind CPU workers and `CompilePending` is raised; the caller defers that work item,
runs something else, and retries once the `.o` lands.** One mechanism serves tests
(`pytest --async-compile`), autotune sweeps, and CI.

## Layer 1: `@jit_cache` Decorator (`quack/cache/jit.py`)

Each `_compile_*` function (e.g. `_compile_gemm` in `gemm.py`, `Softmax.compile` in
`softmax.py`) is decorated with `@jit_cache`, providing in-memory and persistent disk
caching:

```python
@jit_cache
def _compile_rmsnorm_fwd(dtype, out_dtype, ..., N, ...):
    # ... build fake CuTe tensors with cute.sym_int() batch dims ...
    return cute.compile(RMSNorm(...), ...)
```

Two properties of these functions matter for everything below:

- **They are tensor-free.** Args are picklable scalars (cutlass dtypes, ints, flags,
  config dataclasses); torch tensors never cross this boundary. That's what lets a
  subprocess replay a compile from the pickled key.
- **They never launch kernels.** Compilation and launch are separated at this boundary,
  so a worker that calls `_compile_*` cannot touch the GPU by construction.

### In-Memory Cache

A per-function `dict` keyed on the function's positional/keyword args. Within a single
process, calling the same kernel config twice skips compilation and disk I/O entirely.

### Disk Cache

On in-memory miss, checks for a cached `.o` (object file) on disk. The disk key is
`(fn.__qualname__, *args, **sorted_kwargs)`, hashed with SHA-256. The cache directory
includes a **source fingerprint** — a SHA-256 of all `quack/*.py` files plus
Python/CUTLASS/TVM-FFI versions. Any source change invalidates the entire `.o` cache
(the compile *keys* are source-independent — this is what makes re-warming after an
edit cheap and parallel).

With cutlass-dsl >= 4.4.2, `tvm_ffi` loads `.o` files directly (~1 ms) — no
`.o` → `.so` link step, no distutils.

### Cache Miss Flow

```
@jit_cache wrapper(dtype, out_dtype, N)
  ├─ 1. In-memory dict lookup → hit ✓ (~0ns)
  ├─ 2. hash (fn.__qualname__, *args) → SHA-256 hex
  ├─ 3. Shared lock: .o exists → load via tvm_ffi (~1ms) ✓
  ├─ 3b. Async pool active?
  │     ├─ another process holds this key's flock → mark external, raise CompilePending
  │     ├─ key not yet submitted → submit to pool,   raise CompilePending
  │     ├─ pool says "pending"                     → raise CompilePending
  │     ├─ pool says "done"   → shared lock, load .o ✓
  │     └─ pool says "failed" → fall through (in-process compile, real traceback)
  └─ 4. Exclusive lock: fn(...) → cute.compile → export_to_c() → write .o
```

### Concurrency Safety

Multiple processes may compile the same key simultaneously (parallel test workers, pool
workers, autotune sweeps). `FileLock` (`fcntl.flock`) serializes access:

- **Shared lock** for reads — multiple readers load `.o` concurrently
- **Exclusive lock** for writes — one writer compiles + exports; others wait, then load
- **Double-check** after acquiring the exclusive lock (another process may have won)
- The lock file doubles as a **cross-process "compile in progress" signal**: a
  non-blocking probe (`_flock_held_exclusively`) lets a consumer defer on a key some
  *other* process is already compiling instead of wasting a pool slot on a duplicate.

### Environment Controls

| Variable | Default | Description |
|---|---|---|
| `QUACK_CACHE_ENABLED` | `1` | Set to `0` to disable disk cache (in-memory cache still active) |
| `QUACK_CACHE_DIR` | `/tmp/$USER/quack_cache` | Override cache location |

## Layer 2: Async Compile Pool (`quack/cache/async_compile.py`)

### The Constraints

1. **`cute.compile()` is not thread-safe** — MLIR uses thread-local state, so threads
   are out; parallelism must be process-based.
2. **`fork()` after CUDA init is unsafe** — the consuming process has a CUDA context
   and threads; forking it is undefined behavior.
3. **Compiled kernels aren't picklable** — a worker can't send the result back through
   a pipe.

### The Mechanism: `CompilePending` + defer-and-retry

When a pool is active, a `.o` miss does *not* compile in-process. `jit_cache` pickles
the `_compile_*` function's `(module, qualname, args, kwargs)`, submits it to the pool,
and raises `CompilePending(sha)`. The caller catches it, **defers that work item, runs
other work, and retries once `pool.poll(sha)` reports done** — the retry hits the `.o`
fast path (~1 ms). Constraint 3 is solved by making the persistent cache itself the IPC
channel: the worker's only output *is* the `.o` file.

`CompilePending` derives from **`BaseException`** (like `KeyboardInterrupt`) so that
`except Exception` in user/test code cannot swallow the deferral signal and turn
not-yet-run work into a false pass. Only the defer loops catch it.

Two callers implement the loop:

- **pytest** (`quack/testing/pytest_plugin.py`, `--async-compile[=N]`): tests are the
  work items. A deferred test's reports are discarded and it is re-run later.
- **the autotuner** (`Autotuner.__call__` → `benchmark()` under `pool_scope()`):
  candidate configs are the work items. A cold config rotates to the back of the bench
  queue; whichever config is ready gets benchmarked.

Both loops share the same failure/termination rules:

- **Pool failures are never trusted**: a "failed" poll falls through to an in-process
  compile so the real exception surfaces with a local traceback.
- **Force-sync caps**: after `_MAX_ATTEMPTS` deferrals, or when a sha stays pending past
  a wedge deadline (`_POOL_WEDGE_TIMEOUT_S` / `_WEDGE_TIMEOUT_S`), the item runs with
  the pool suppressed (`suppress_pool()`), compiling in-process. A wedged worker or a
  hung foreign flock holder can therefore never hang the run.

### Worker Startup: Forkserver Sidecar

Same architecture as PyTorch Inductor's compile-worker `SubprocPool`: one sidecar
process pays the heavy import once, workers fork from it copy-on-write.

- `_make_executor()` builds a `ProcessPoolExecutor` on a **`forkserver`** context with
  `set_forkserver_preload(["quack.cache._pool_preload"])`. The preload imports
  torch + cutlass + tvm_ffi (~13 s) exactly once; each worker forks in ~0.1 s.
  Measured effect: pool CPU cost for a 130-key cold run dropped from ~11 CPU-min
  (spawn, 32 workers × import) to ~1 CPU-min.
- **Workers are GPU-blind.** The preload pins `QUACK_ARCH`/`CUTE_DSL_ARCH` (via
  `nvidia-smi --query-gpu=compute_cap`, no CUDA context) and sets
  `CUDA_VISIBLE_DEVICES=""`. An explicit `QUACK_ARCH` env override wins — CI
  cross-compiles (e.g. `QUACK_ARCH=120` on an H100) and workers must target the
  *requested* arch, not the physical one. Fork-safety follows: the sidecar never
  initializes CUDA. (This flushed out a real bug: `import quack` used to initialize
  CUDA at import time via `rmsnorm_config._detect_arch_major()`; it now honors
  `QUACK_ARCH`.)
- **`_neutral_main()`**: multiprocessing child prep re-executes the user's `__main__`
  script (`runpy.run_path`) so pickles referencing `__main__` resolve. Our tasks never
  reference `__main__`, and a user script with CUDA work at top level would kill every
  worker at spawn. Since executor workers spawn synchronously inside `submit`, masking
  `sys.modules["__main__"]` with an empty stub during the submit is sufficient.
  (Inductor solves the same problem with a dedicated worker entrypoint.)
- **`prewarm()`** (Inductor's `warm_pool()` analog): the pytest plugin submits a no-op
  at `pytest_configure` so the sidecar's ~13 s import overlaps collection instead of
  the first cold compile. Measured: in-session cold time 38 s → 29 s on the benchmark
  set.

### Pool API surface

| | |
|---|---|
| `activate(jobs)` / `deactivate()` / `get_active_pool()` | session-wide pool (pytest plugin) |
| `pool_scope()` | scoped activation (autotuner); reuses the active pool or wraps the shared executor; `CompilePending` cannot escape the block into unrelated code |
| `suppress_pool()` | make `get_active_pool()` return None inside — the force-sync escape hatch |
| `CompilePool.submit / poll / mark_external / prewarm / stats` | per-sha bookkeeping; `poll` states: `new / pending / done / failed` |
| `get_shared_executor()` | process-wide executor for scoped pools (`QUACK_COMPILE_WORKERS`, default 8) |

| Variable | Default | Description |
|---|---|---|
| `QUACK_ASYNC_COMPILE_START` | `forkserver` | set to `spawn` to disable the fork sidecar |
| `QUACK_COMPILE_WORKERS` | `8` | shared-executor size (autotune sweeps outside pytest) |

## Layer 3: Autotuning (`autotuner.py`)

### Result Cache

The autotuner benchmarks many configs (e.g. ~44 for GEMM) to find the fastest. Results
are cached to disk as JSON via `FileCacheManager` (Triton's cache infrastructure). The
JSON key includes package version, tuning key (tensor metadata), and config strings.

| Variable | Default | Description |
|---|---|---|
| `QUACK_CACHE_AUTOTUNING` | unset | Set to `1` to enable disk caching of tuning results |
| `QUACK_FORCE_CACHE_UPDATE` | unset | Set to `1` to ignore cached tuning results |

### Compile/Bench Overlap

There is **no separate precompile phase**. The bench loop runs inside `pool_scope()`:

```
with pool_scope() as pool:
    queue = deque(pruned_configs)
    while queue:
        config = queue.popleft()
        if awaiting sha still pending (and not past wedge deadline): rotate; continue
        try:
            timings[config] = self._bench(*args, config=config, **kwargs)
        except CompilePending as e:
            remember e.sha; rotate config to the back
```

Key discovery uses the **real tensors, in-process**: the first bench attempt of a cold
config reaches `_compile_*` through the normal wrapper, the key ships to the pool, and
the loop benches whichever config is already warm. Total wall stays
max(parallel_compile, serial_bench). `CompilePending` can only fire *outside* CUDA
graph capture: the L2-cold bench does plain priming launches before capture, so a cold
key raises at the first launch.

A config whose compile genuinely fails falls through to an in-process compile inside
`_bench`, whose `(RuntimeError, MemoryError)` handler records `float("inf")` — failed
configs never win, and never contaminate timings with compile cost.

## Single-Pass Test Workflow (`quack/testing/pytest_plugin.py`)

```bash
# One command, cold or warm cache:
pytest tests/test_softmax.py --async-compile=16
pytest tests/ -n 8 --dist worksteal --async-compile=32     # multi-GPU xdist
```

Cold runs cost ≈ warm + compile-critical-path (measured: warm 26 s, cold 29–39 s
in-session for a 398-test / 130-key set); warm runs submit nothing and pay nothing
beyond one idle sidecar import in the background.

### Single-process mode (`_SingleProcDeferLoop`)

Replaces `pytest_runtestloop` with a deque: a deferred test rotates to the back
(reports discarded, no logging) and is retried once its sha resolves — *without*
re-running the test until then (rotation is a poll, not a re-run). Fixture-teardown
scoping is handled by a misprediction guard in `_run_protocol`: `runtestprotocol`
scopes teardown to a *predicted* next item, deferral breaks the prediction chain, and
pytest asserts on the mismatch ("previous item was not torn down properly") — so the
guard forces a full teardown whenever the actually-run item differs from the
prediction.

### xdist mode (`_XdistWorkerDefer`)

xdist's `WorkerInteractor` owns the runtestloop but invokes `pytest_runtest_protocol`
as a hook, so the plugin takes over the protocol (tryfirst + firstresult):

- A deferred item is stashed; the worker still sends `runtest_protocol_complete`, the
  master marks the item complete (order-independent `.remove()`) and **keeps streaming
  new items — skip-ahead comes for free**.
- Stashed items are retried opportunistically between incoming items, and drained in a
  `pytest_runtestloop` hookwrapper after xdist's inner loop exits (channel still open,
  late reports flow normally). Late reports temporarily repoint
  `WorkerInteractor.item_index` (the worker asserts report↔index consistency).
- The drain is skipped on `-x`/`--maxfail` aborts.
- Deferral suppresses the call phase cheaply (`force_result`, avoiding ~100 ms of
  failure-`longrepr` formatting per deferral) but leaves a setup-phase `CompilePending`
  as a failure so pytest skips the body on half-built fixtures. Teardown errors on
  deferred attempts are still forwarded.

### Session-end integrity check

The report-early trick opens one hole: a worker that crashes with deferred items
stashed loses them *silently* (the master already counted them complete). At
`pytest_sessionfinish` the plugin diffs collected − reported − deselected nodeids; any
gap prints the missing tests and flips the exit status to failure.

### CI (`.github/actions/gpu-test/action.yml`)

Single pass: `pytest tests/ -n $NUM_GPUS --dist worksteal --async-compile=24`. The
per-runner `QUACK_CACHE_DIR` carries `.o` files across runs; cross-arch legs set
`QUACK_ARCH` (honored by the pool's arch pinning).

## `custom_op` Fakes Are Pure No-ops

All kernels use `@torch.library.custom_op` via the `cute_op` decorator
(`quack/dsl/torch_library_op.py`). The registered fake is a **pure no-op**: our ops
only mutate their inputs, so Dynamo / AOT autograd need no shape effect, and running
the body under tracing would pay compile latency at trace time (or crash for
shape/dtype combos the kernel intentionally rejects). Kernel compilation is owned
entirely by `jit_cache` + the pool at real execution time. (A few ops that *return*
tensors — e.g. `gemm_rms_out` — keep a minimal shape-only fake.)

## Complete Data Flow

### Normal Execution (Training/Inference)

```
User calls gemm(A, B, ...)
  └─ gemm_out (custom_op → CUDA impl)
       └─ gemm_tuned (autotuner)
            ├─ JSON cache hit → call fn with best config
            └─ Miss → benchmark() under pool_scope()
                 ├─ cold config → CompilePending → rotate; pool compiles .o
                 ├─ warm config → bench (compile/bench overlap)
                 └─ cache best config
            └─ fn(A, B, out, config=best)
                 └─ _compile_gemm(...) → @jit_cache (.o hit, ~1ms)
                      └─ compiled_fn(real A/B/D pointers, ...)
```

### `pytest --async-compile` (cold cache)

```
pytest --async-compile=32
  └─ pytest_configure: activate(pool), pool.prewarm()   # sidecar import overlaps collection
       └─ test runs, kernel dispatch → jit_cache miss
            ├─ submit pickled (_compile_*, args) to pool → raise CompilePending
            ├─ defer loop: rotate test to back, run other tests
            ├─ pool worker (forked, GPU-blind): _compile_*(...) → export .o
            └─ retry: .o fast path (~1ms) → kernel launches → asserts run
  └─ pytest_sessionfinish: collected-vs-reported integrity check
```

## Key Files

| File | Role |
|------|------|
| `quack/cache/jit.py` | `@jit_cache` (in-memory + `.o` disk cache), `FileLock`, pool integration (step 3b) |
| `quack/cache/async_compile.py` | `CompilePending`, `CompilePool`, forkserver sidecar, `pool_scope`, `suppress_pool`, `_neutral_main` |
| `quack/cache/_pool_preload.py` | forkserver preload: arch pinning (no CUDA), torch/cutlass import |
| `quack/testing/pytest_plugin.py` | `--async-compile` flag, single-proc + xdist defer loops, integrity check |
| `autotuner.py` | bench loop under `pool_scope` (compile/bench overlap), `FileCacheManager` result cache |
| `quack/dsl/torch_library_op.py` | `cute_op`: custom_op with pure no-op fakes |
| `tests/test_async_compile.py`, `tests/test_autotuner.py`, `tests/test_cache.py` | regression tests encoding the failure modes above |

## History

This design replaced a two-pass workflow (`pytest --compile-only -n 64` under a
session-wide `FakeTensorMode`, then a real run) and its supporting machinery: a
`COMPILE_ONLY` mode flag threaded through `jit_cache`/gemm wrappers/`register_fake`
bodies, an `.item()`-sentinel fake-tensor mode, per-op precompile fakes, a
record/replay compile-key manifest with prefetch, and hand-rolled `subprocess.Popen`
compile workers with a pickle framing protocol. All of it is gone; the demand-driven
defer-and-retry pool matched or beat it in every measured configuration while deleting
~1000 lines. If archaeology is needed, the session that made the change documented each
step, and `git log` on the files above has the receipts.
