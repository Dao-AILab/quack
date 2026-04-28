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
smem_layout = cute.make_layout((d_tile, seq_tile), stride=(1, d_tile))
tma_atom, tma_tensor, desc_ptr = make_sm120_direct_tma_load_2d_atom(
    gmem_tma,
    smem_layout,
    (d_tile, seq_tile),
)
```

Consumers can create another shared-memory view over the same storage if they
need a logical `(seq, d)` view.  Do not transpose the data after TMA just to
recover the logical tensor order.

## Public Helpers

The SM120 direct TMA helpers live in `quack.sm120_tma_utils`:

- `has_sm120_direct_tma_2d(device=None, require_device=False)`
- `assert_sm120_direct_tma_2d(device=None, require_device=True)`
- `make_sm120_tma_basis_tensor_2d(gmem_tensor, d_total, seq_total)`
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
- validated swizzled shared-memory layouts
- integration into `GemmSm120`

## Validation

The copy-only tests are in `tests/test_sm120_tma_utils.py`.  They load a
deterministic row-major source pattern:

```text
src[seq, d] = 100 * seq + d
```

and compare the full copied tile against the PyTorch reference slice for both
zero and nonzero origins.

Run:

```bash
CUTE_DSL_ARCH=sm_120a pytest -q tests/test_sm120_tma_utils.py
```

## Benchmark

Use `benchmarks/benchmark_sm120_direct_tma.py` to compare this path against two
non-TMA baselines:

- `async-copy`: a simple CuTe `cp.async` G2S staging path using the same
  `(d, seq)` tile shape. This is a better non-TMA comparison than a blocking
  thread-copy loop, but it is intentionally not a fully tuned vectorized
  cp.async pipeline.
- `cooperative-copy`: a blocking cooperative global-to-shared copy loop. This
  is a control, not a claim about optimized async-copy performance.

The default scenario is `overlap-fa`: a producer warp stages dense K/V tiles
while a consumer warp performs deterministic FA-like work on staged data.  The
default consumer is `mma`: a synthetic BF16/FP16 warp-level Tensor Core workload
using `MmaF16BF16Op`.  This is meant to model K/V tiles feeding QK/PV-style
Tensor Core work without depending on QuACK attention or NVFP4 internals.  The
`scalar` consumer remains available for staging-overhead diagnostics.

The default BF16 tile is `head_dim=128, seq_tile=64`, with both K and V loaded
and two stages.  Each stage moves 32KB of K/V data and the kernel uses roughly
65KB of dynamic shared memory.  The benchmark also includes one-stage and larger
tile rows because extra stages only help when there is enough independent
consumer work to hide the TMA pipeline wait and shared-memory footprint.

For `overlap-fa`, the benchmark keeps the producer/consumer split consistent:
one producer warp is responsible for staging in every mode and one consumer warp
does the shared-memory read/FP32 accumulation work.  The output reports the
producer resource for each mode so the TMA tradeoff is visible: direct TMA uses
one elected issuer, async copy uses a producer warp issuing `cp.async`, and
cooperative copy uses a producer warp doing blocking loads/stores.

```bash
CUTE_DSL_ARCH=sm_120a python benchmarks/benchmark_sm120_direct_tma.py

CUTE_DSL_ARCH=sm_120a python benchmarks/benchmark_sm120_direct_tma.py --sweep

CUTE_DSL_ARCH=sm_120a python benchmarks/benchmark_sm120_direct_tma.py \
  --scenario overlap-fa --consumer mma --mode both --head-dim 128 --seq-tile 128 \
  --num-stages 1 --num-kv-tiles 4

CUTE_DSL_ARCH=sm_120a python benchmarks/benchmark_sm120_direct_tma.py \
  --scenario overlap-fa --consumer scalar --mode both --head-dim 128 --seq-tile 64

CUTE_DSL_ARCH=sm_120a python benchmarks/benchmark_sm120_direct_tma.py \
  --scenario copy --mode both --head-dim 128 --seq-tile 128
```

For Nsight Compute:

```bash
ncu --profile-from-start off python benchmarks/benchmark_sm120_direct_tma.py \
  --profile --scenario overlap-fa --consumer mma --mode direct-tma \
  --head-dim 128 --seq-tile 64 --num-stages 2

ncu --profile-from-start off python benchmarks/benchmark_sm120_direct_tma.py \
  --profile --scenario overlap-fa --consumer mma --mode async-copy \
  --head-dim 128 --seq-tile 64 --num-stages 2

ncu --profile-from-start off python benchmarks/benchmark_sm120_direct_tma.py \
  --profile --scenario overlap-fa --consumer mma --mode cooperative-copy \
  --head-dim 128 --seq-tile 64 --num-stages 2
```

The `--sweep` mode runs below-threshold, threshold, and FA-like overlap rows.
It is the preferred way to evaluate this helper because it shows where direct
TMA loses to simpler copies and where larger FA-like K/V tiles can cross over.
In particular, compare the scalar and Tensor Core consumer rows before assuming
extra stages help: on light synthetic work, the additional shared-memory
allocation and pipeline wait can dominate, while larger Tensor Core rows better
reflect the intended FlashAttention use case.

On a workstation GPU, single-run timings can be noisy.  Prefer the default
`second-min` statistic, run several repeats, and use Nsight Compute counters
such as TMA requests, dynamic shared memory, eligible warps, and stall reasons
to explain timing differences instead of relying on one benchmark row.
