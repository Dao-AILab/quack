# SM120 Nsight Compute Profiling Guide

This note describes how to profile SM120 / RTX 50 GEMM epilogue kernels using
the generic benchmark harness:

- `benchmarks/benchmark_gemm_epilogues.py`

The immediate target of this note is SM120 performance work for:

- `gemm_rms`
- `gemm_norm_act`
- optionally `gemm_act` as a control path

## Why this harness exists

The existing GEMM benchmark scripts are useful for wall-clock timing, but they
do not provide a stable, explicit profiling entrypoint for SM120 epilogue-heavy
kernels.

This harness makes the following inputs explicit from the command line:

- kernel family
- shape
- dtype
- `tile_m`
- `tile_n`
- `pingpong`
- `swap_ab`
- dynamic persistence mode
- optional GPU preheat duration
- optional JSON / CSV output for comparing repeated profiling sessions

That makes it suitable for both:

- quick timing with repeated CUDA-event samples
- reproducible `ncu` runs
- machine-readable result capture for tuning notes and PR evidence

All timings recorded during this SM120 tuning pass were taken on a desktop
Blackwell RTX 5060 8GB. Because the local machine is noisy, comparisons should
use the second-best sample (`--stat second-min`), not a single run or the raw
minimum.

These findings should be revalidated on other RTX 50 models, especially the RTX
5090. Its larger memory capacity and different performance envelope may change
the winning defaults or the performance expectations.

## Capturing timing records

Use `--output-json` for one-off profiling notes and `--output-csv` when sweeping
several configs:

### Epilogues

```bash
python benchmarks/benchmark_gemm_epilogues.py \
  --kernel rms --dtype bfloat16 --m 4096 --n 4096 --k 4096 \
  --preheat-ms 500 \
  --stat second-min \
  --tile-m 128 --tile-n 64 \
  --output-json sm120_rms_128x64.json

python benchmarks/benchmark_gemm_epilogues.py \
  --kernel rms --dtype bfloat16 --m 4096 --n 4096 --k 4096 \
  --preheat-ms 500 \
  --stat second-min \
  --tile-m 64 --tile-n 128 \
  --output-csv sm120_rms_sweep.csv
```

## First comparison matrix

Start with `gemm_rms` on these configs:

1. `128x64`, `pingpong=False`
2. `64x128`, `pingpong=False`
3. `128x64`, `pingpong=True`
4. `64x128`, `pingpong=True`

Use these shapes first:

1. `(4096, 4096) x (4096, 4096)`
2. `(4096, 2048) x (2048, 2048)`

These are the right first comparisons because SM120 timing already showed that
RMS is sensitive to tile shape and pingpong choice.

## Example timing commands

```bash
python benchmarks/benchmark_gemm_epilogues.py \
  --kernel rms --dtype bfloat16 --m 4096 --n 4096 --k 4096 \
  --preheat-ms 500 \
  --stat second-min \
  --tile-m 128 --tile-n 64

python benchmarks/benchmark_gemm_epilogues.py \
  --kernel rms --dtype bfloat16 --m 4096 --n 4096 --k 4096 \
  --preheat-ms 500 \
  --stat second-min \
  --tile-m 64 --tile-n 128

python benchmarks/benchmark_gemm_epilogues.py \
  --kernel rms --dtype bfloat16 --m 4096 --n 4096 --k 4096 \
  --preheat-ms 500 \
  --stat second-min \
  --tile-m 128 --tile-n 64 --pingpong

python benchmarks/benchmark_gemm_epilogues.py \
  --kernel rms --dtype bfloat16 --m 4096 --n 4096 --k 4096 \
  --preheat-ms 500 \
  --stat second-min \
  --tile-m 64 --tile-n 128 --pingpong
```

## Example `ncu` commands

Profile one configuration at a time.

Large square RMS case:

```bash
ncu --profile-from-start off python benchmarks/benchmark_gemm_epilogues.py \
  --profile \
  --kernel rms --dtype bfloat16 --m 4096 --n 4096 --k 4096 \
  --preheat-ms 500 \
  --tile-m 128 --tile-n 64
```

Moderate-width RMS case:

```bash
ncu --profile-from-start off python benchmarks/benchmark_gemm_epilogues.py \
  --profile \
  --kernel rms --dtype bfloat16 --m 4096 --n 2048 --k 2048 \
  --preheat-ms 500 \
  --tile-m 128 --tile-n 64
```

Norm-act comparison:

```bash
ncu --profile-from-start off python benchmarks/benchmark_gemm_epilogues.py \
  --profile \
  --kernel norm_act --dtype bfloat16 --activation gelu_tanh_approx \
  --m 4096 --n 4096 --k 4096 \
  --preheat-ms 500 \
  --tile-m 128 --tile-n 64
```

Act control path:

```bash
ncu --profile-from-start off python benchmarks/benchmark_gemm_epilogues.py \
  --profile \
  --kernel act --dtype bfloat16 --activation gelu_tanh_approx \
  --m 4096 --n 14336 --k 4096 \
  --preheat-ms 500 \
  --tile-m 64 --tile-n 128 --pingpong
```

## Metrics to inspect first

Look at these categories before changing code or config pruning:

1. achieved occupancy
2. registers per thread
3. shared memory per block
4. active warps / active CTAs
5. tensor pipeline utilization
6. memory pipeline utilization
7. stall reasons related to barriers, waits, and scheduler backpressure

## How to interpret the first results

Use the profiling results to answer these questions:

1. Why does `pingpong=False` beat `pingpong=True` for `gemm_rms` on SM120?
2. Is the losing config limited by:
   - lower occupancy
   - higher SMEM pressure
   - barrier overhead
   - epilogue serialization
   - lower tensor-pipe utilization
3. Does `gemm_norm_act` show the same limiting behavior as `gemm_rms`?
4. Is the current generic SM120 default config (`128x128`, pingpong) sufficient
   for RMS, or does RMS need its own justified default once `ncu` and timing
   are considered together?

## What should change only after profiling

Do not change SM120 defaults until the timing and `ncu` story agree.

The first config changes that profiling should justify are limited to SM120 defaults, such as `gemm_rms` or possibly `gemm_norm_act` if it shows the same bottleneck pattern

The first profiling pass should not trigger:

- kernel algorithm rewrites
- major SM120 pipeline redesign
- broad retuning of all SM120 GEMM paths at once
