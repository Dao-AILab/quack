# SM120 Direct Rank-2 TMA

QuACK exposes a narrow SM120-only direct TMA helper for dense
FlashAttention-style K/V tile loads.  This path exists because normal rank-2
descriptor TMA through `cute.copy(...)` can compile but hang on SM120 while
waiting on the TMA pipeline mbarrier.

The CuTe DSL workaround is available in CUTLASS PR
https://github.com/NVIDIA/cutlass/pull/3189,
"CuTe DSL: work around SM120 rank-2 TMA cute.copy hang".

The supported path is:

1. Build the CuTe descriptor in the explicit hardware TMA basis.
2. Get the descriptor pointer from CuTe.
3. Issue the CTA-local rank-2 TMA load with `cpasync.sm120_tma_load_2d(...)`.
4. Use `PipelineTmaAsync` normally around the transaction.

This is not generic CuTe TMA support and does not replace the existing GEMM TMA
paths.  It is opt-in scaffolding for benchmarking and for future FA-style K/V
load experiments.

## Contract

For a row-major logical tensor:

```text
src[seq, d]
physical stride = (D, 1)
```

create a separate TMA-basis view:

```python
gmem_tma = make_sm120_tma_basis_tensor_2d(src, D, S)
```

This view has:

```text
shape  = (D, S)
stride = (1, D)
coords = (d_coord, seq_coord)
```

The shared-memory load layout should use the same direct basis, for example:

```python
smem_layout = make_sm120_direct_tma_smem_layout_2d(d_tile, seq_tile)
tma_atom, tma_tensor, desc_ptr = make_sm120_direct_tma_load_2d_atom(
    gmem_tma,
    smem_layout,
    (d_tile, seq_tile),
)
```

For SW128-style swizzled shared memory, the TMA atom layout and destination
shared-memory pointer must carry the same swizzle:

```python
smem_layout = make_sm120_direct_tma_smem_layout_2d(
    d_tile,
    seq_tile,
    swizzle=True,
)
s_tile = smem.allocate_tensor(
    dtype,
    make_sm120_direct_tma_smem_layout_2d(d_tile, seq_tile),
    byte_alignment=128,
    swizzle=cute.make_swizzle(3, 4, 3),
)
sm120_direct_tma_load_2d(s_tile.iterator, desc_ptr, tma_bar_ptr, d_coord, seq_coord)
```

Consumers can create another shared-memory view over the same storage if they
need a logical `(seq, d)` view.  Do not transpose the data after TMA just to
recover the logical tensor order.

## Public Helpers

The SM120 direct TMA helpers live in `quack.sm120_tma_utils`:

- `has_sm120_direct_tma_2d(device=None, require_device=False)`
- `assert_sm120_direct_tma_2d(device=None, require_device=True)`
- `make_sm120_tma_basis_tensor_2d(gmem_tensor, d_total, seq_total)`
- `make_sm120_direct_tma_smem_layout_2d(d_tile, seq_tile, swizzle=False)`
- `make_sm120_direct_tma_load_2d_atom(gmem_tma_tensor, smem_layout, cta_tiler)`
- `get_sm120_direct_tma_desc_addr(tma_atom)`
- `sm120_direct_tma_load_2d(dst_smem_ptr, desc_ptr, tma_bar_ptr, d_coord, seq_coord)`

The feature checks require the local CuTe DSL branch or upstream equivalent that
provides:

- `cpasync.get_tma_desc_addr`
- `cpasync.make_sm120_tma_load_2d_atom`
- `cpasync.sm120_tma_load_2d`

## Why TMA

TMA is useful when the copy is large enough and the kernel can overlap movement
with useful work.  A single elected producer can issue a bulk shared-memory tile
load while consumer warps continue with compute on previous stages, which can
reduce the number of threads spent on manual copy loops and make producer /
consumer pipelines cleaner.

For small tiles, TMA setup and synchronization overhead can dominate.  A common
rule of thumb is to benchmark TMA around tiles larger than roughly 16KB; below
that, simpler async or cooperative copy paths can be faster.  The benchmark
therefore includes both a below-threshold copy control and FA-like larger-tile
cases.

## Limitations

This helper is intentionally scoped to the first dense SM120 FA use case:

- SM120/SM120a only
- CTA-local G2S only
- rank-2 only
- mode 0 must be contiguous with stride 1
- element width must be at least 8 bits
- `tile0 <= 256`
- `tile0 * sizeof(element)` must be a multiple of 16 bytes

Not covered:

- normal `cute.copy(...)` executable TMA lowering
- multicast or CTA group 2
- rank-3/4/5 descriptors
- paged or varlen KV gather in one TMA transaction
- OOB fill/masking semantics
- automatic `(seq, d)` to `(d, seq)` canonicalization
- large swizzled shared-memory tiles beyond the conservative smoke coverage
- integration into `GemmSm120`

## Validation

The copy-only tests are in `tests/test_sm120_tma_utils.py`.  They load a
deterministic row-major source pattern:

```text
src[seq, d] = 100 * seq + d
```

and compare the full copied tile against the PyTorch reference slice for both
zero and nonzero origins.  They also validate conservative SW128 shared-memory
loads for FP16/BF16 `64x64` tiles.

Run:

```bash
CUTE_DSL_ARCH=sm_120a pytest -q tests/test_sm120_tma_utils.py
```

## Benchmark

Use `benchmarks/benchmark_sm120_direct_tma.py` to compare this path against two
non-TMA baselines:

- `async-copy`: a CuTe `cp.async` G2S staging path for the FA-overlap scenario.
  It stages in logical `(seq, d)` order, vectorized along contiguous `d`, and
  uses the same `num_stages` ping-pong buffering contract as direct TMA through
  `PipelineCpAsync`.
- `cooperative-copy`: a blocking cooperative global-to-shared copy loop. This
  is a control, not a claim about optimized async-copy performance.

The default scenario is `overlap-fa`: a producer warp stages dense K/V tiles
while one or more consumer warps perform deterministic FA-like work on staged data.  The
default consumer is `mma`: a synthetic BF16/FP16 warp-level Tensor Core workload
using `MmaF16BF16Op`.  This is meant to model K/V tiles feeding QK/PV-style
Tensor Core work without depending on QuACK attention or NVFP4 internals.  The
`scalar` consumer remains available for staging-overhead diagnostics.

The default BF16 tile is `head_dim=64, seq_tile=128`, with both K and V loaded
and two stages.  Each stage moves 32KB of K/V data and the kernel uses roughly
65KB of dynamic shared memory.  The benchmark also includes one-stage and larger
tile rows because extra stages only help when there is enough independent
consumer work to hide the TMA pipeline wait and shared-memory footprint.

For `overlap-fa`, the benchmark keeps the producer/consumer split consistent:
one producer warp is responsible for staging in every mode and `--num-consumer-warps`
controls how many consumer warps reuse the staged K/V tile.  The output reports the
producer resource for each mode so the TMA tradeoff is visible: direct TMA uses
one elected issuer, async copy uses a producer warp issuing `cp.async`, and
cooperative copy uses a producer warp doing blocking loads/stores. With
`--num-stages 2`, direct TMA and async-copy both use real ping-pong staging; the
producer can fill one stage while the consumer works on the other.
`--num-consumer-warps` currently supports `1..7`, for up to an 8-warp CTA with
one producer warp and seven consumer warps.

The benchmark accepts `--smem-layout identity`, `--smem-layout sw128`, and
`--smem-layout all`.  In the FA-overlap scenario, `sw128` is validated for
`direct-tma` and `async-copy`; `cooperative-copy` remains an identity-layout
control.  Use `--mode all --smem-layout sw128` to compare direct TMA against a
cp.async baseline with the same swizzled consumer layout.  The standalone copy
scenario compares direct TMA against cooperative-copy only; use `overlap-fa` for
async-copy baselines.

For `overlap-fa`, `--async-copy-bits` defaults to `128` and controls the
cp.async transaction width.  The standalone copy scenario defaults to the older
32-bit async width internally but does not include async-copy in `--mode all`.

```bash
CUTE_DSL_ARCH=sm_120a python benchmarks/benchmark_sm120_direct_tma.py

CUTE_DSL_ARCH=sm_120a python benchmarks/benchmark_sm120_direct_tma.py --sweep

CUTE_DSL_ARCH=sm_120a python benchmarks/benchmark_sm120_direct_tma.py \
  --scenario overlap-fa --consumer mma --mode all --head-dim 128 --seq-tile 128 \
  --num-stages 1 --num-kv-tiles 4

CUTE_DSL_ARCH=sm_120a python benchmarks/benchmark_sm120_direct_tma.py \
  --scenario overlap-fa --consumer scalar --mode all --head-dim 128 --seq-tile 64

CUTE_DSL_ARCH=sm_120a python benchmarks/benchmark_sm120_direct_tma.py \
  --scenario copy --mode all --head-dim 128 --seq-tile 128

CUTE_DSL_ARCH=sm_120a python benchmarks/benchmark_sm120_direct_tma.py \
  --scenario copy --mode direct-tma --smem-layout sw128 \
  --dtype bfloat16 --head-dim 64 --seq-tile 64

CUTE_DSL_ARCH=sm_120a python benchmarks/benchmark_sm120_direct_tma.py \
  --scenario overlap-fa --mode all --smem-layout sw128 \
  --consumer mma --dtype bfloat16 --head-dim 64 --seq-tile 128 \
  --num-kv-tiles 2 --num-stages 2 --num-consumer-warps 2 \
  --async-copy-bits 128
```

That last command is the recommended 32KB-per-stage SW128 ping-pong row for
comparing direct TMA against the staged vectorized async-copy baseline while two
consumer warps reuse each staged K/V tile.

For Nsight Compute:

```bash
ncu --profile-from-start off python benchmarks/benchmark_sm120_direct_tma.py \
  --profile --scenario overlap-fa --consumer mma --mode direct-tma \
  --smem-layout sw128 --head-dim 64 --seq-tile 128 \
  --num-kv-tiles 2 --num-stages 2 --num-consumer-warps 2

ncu --profile-from-start off python benchmarks/benchmark_sm120_direct_tma.py \
  --profile --scenario overlap-fa --consumer mma --mode async-copy \
  --smem-layout sw128 --head-dim 64 --seq-tile 128 \
  --num-kv-tiles 2 --num-stages 2 --num-consumer-warps 2 \
  --async-copy-bits 128

ncu --profile-from-start off python benchmarks/benchmark_sm120_direct_tma.py \
  --profile --scenario overlap-fa --consumer mma --mode cooperative-copy \
  --head-dim 128 --seq-tile 64 --num-stages 2
```

For identity-vs-SW128 shared-memory conflict analysis, profile direct TMA
layouts separately, then profile the SW128 async-copy baseline:

```bash
CUTE_DSL_ARCH=sm_120a ncu --profile-from-start off --target-processes all \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,\
l1tex__data_pipe_lsu_wavefronts_mem_shared,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld \
  python benchmarks/benchmark_sm120_direct_tma.py \
    --profile --profile-warmup 2 --scenario overlap-fa \
    --mode direct-tma --smem-layout identity --consumer mma \
    --dtype bfloat16 --head-dim 64 --seq-tile 64 \
    --num-kv-tiles 2 --num-stages 2

CUTE_DSL_ARCH=sm_120a ncu --profile-from-start off --target-processes all \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,\
l1tex__data_pipe_lsu_wavefronts_mem_shared,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld \
  python benchmarks/benchmark_sm120_direct_tma.py \
    --profile --profile-warmup 2 --scenario overlap-fa \
    --mode direct-tma --smem-layout sw128 --consumer mma \
    --dtype bfloat16 --head-dim 64 --seq-tile 64 \
    --num-kv-tiles 2 --num-stages 2

CUTE_DSL_ARCH=sm_120a ncu --profile-from-start off --target-processes all \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,\
l1tex__data_pipe_lsu_wavefronts_mem_shared,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld \
  python benchmarks/benchmark_sm120_direct_tma.py \
    --profile --profile-warmup 2 --scenario overlap-fa \
    --mode async-copy --smem-layout sw128 --consumer mma \
    --dtype bfloat16 --head-dim 64 --seq-tile 64 \
    --num-kv-tiles 2 --num-stages 2
```

On a local RTX 50 workstation, this `64x64` overlapped MMA row reduced shared
load bank conflicts from roughly `918821` to `1105` and shared load wavefronts
from roughly `1050917` to `133201` when switching direct TMA from identity SMEM
to SW128.  The SW128 async-copy baseline measured similarly low shared-load
conflicts and wavefronts (`1531` and `133627`), which makes it a better
apples-to-apples baseline for comparing TMA issue mechanics rather than shared
memory bank behavior.  Treat those values as directional, not portable
constants.

The `--sweep` mode runs FA-like Tensor Core overlap rows around the TMA
usefulness threshold with multi-warp consumers. It covers comparable 16KB and
32KB staged K/V tiles across 2, 3, 5, and 7 consumer warps, and repeats the 32KB
rows at `compute_iters=1,2,4`. It is the preferred way to evaluate this helper
because it shows where direct TMA loses to simpler copies and where larger K/V
tiles or heavier consumer work can cross over. `--compute-iters` repeats the
synthetic consumer work per staged tile; use it to test whether additional
independent work hides producer latency, but do not treat it as real attention
math.

On a workstation GPU, single-run timings can be noisy.  Prefer the default
`second-min` statistic, run several repeats, and use Nsight Compute counters
such as TMA requests, dynamic shared memory, eligible warps, and stall reasons
to explain timing differences instead of relying on one benchmark row.
