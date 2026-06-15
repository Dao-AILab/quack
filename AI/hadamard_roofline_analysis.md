# Hadamard roofline: staged-SMEM vs shuffle-heavy implementations

This section compares two single-CTA Hadamard implementations for bf16 input/output with fp32 internal values:

1. **QuACK staged-SMEM implementation** in `quack/hadamard.py`: each logical radix stage does local fp32 butterflies in registers, then performs a full fp32 shared-memory round trip to exchange across threads. The current kernel always takes this path (the `use_3stage` selector is short-circuited to `True`).
2. **Previous `fast-hadamard-transform` CUDA implementation**: uses warp shuffles for the 5 within-warp butterfly stages, sandwiched between two shared-memory transposes that expose the 3 cross-warp butterfly stages as additional within-warp shuffles.

The goal is not to model latency or occupancy perfectly; it is a roofline model for the dominant data-movement resources: global-memory bandwidth and the SM-local MIO/shared-memory/shuffle pipe. fp32 add/sub throughput is well above the bottleneck in every case below and is not the binding roof for any configuration.

## Counting model

For `N = 2^L`, one Hadamard transform performs

```text
FP ops = N * L
```

because each of the `L` butterfly stages has `N / 2` butterflies and each butterfly does one add and one subtract. I ignore the final scale multiply; counting it changes `L` to approximately `L + 1` for the fp32 operation count, but does not change the memory-traffic conclusions.

For bf16 input and output:

```text
GMEM bytes = 2N load + 2N store = 4N bytes
```

I use the SM-local MIO/shared-memory/shuffle roof from `AI/gpu_pipe_microbench_notes.md`:

```text
SMEM/MIO roof = 128 B / clk / SM
```

fp32 add/sub peak (~128 ops/clk/SM on these GPUs) is far above the values that come out of the data-movement roofs below, so it never binds in the tables that follow.

Shuffle is modeled as SM-local pipe pressure:

```text
1 full-warp 32-bit SHFL ~= 128 useful B ~= 4 B/lane
```

So one shuffle stage over all `N` fp32 values is counted as `4N` bytes of SM-local pipe pressure.

## Hardware assumptions

All bandwidths are theoretical peak, using decimal TB/s/GB/s.

| GPU | SMs | clock | GMEM BW | GMEM B/clk/SM | aggregate SMEM BW | SMEM / GMEM |
|---|---:|---:|---:|---:|---:|---:|
| A100 80GB SXM | 108 | 1.41 GHz | 2.039 TB/s | 13.39 | 19.49 TB/s | 9.56x |
| H100 SXM | 132 | 1.83 GHz | 3.35 TB/s | 13.87 | 30.92 TB/s | 9.23x |
| B200 | 148 | 1.85 GHz | 8.0 TB/s | 29.22 | 35.05 TB/s | 4.38x |
| RTX 5090 | 170 | 2.41 GHz | 1.792 TB/s | 4.37 | 52.44 TB/s | 29.26x |

The important normalized quantity is **GMEM B/clk/SM**. B200 has much more HBM bandwidth per SM-clock than A100/H100, while RTX 5090 has much less.

The global-memory roof is:

```text
GMEM roof = FP ops / GMEM cycles
          = (N * L) / (4N / B_gmem)
          = L * B_gmem / 4        op/clk/SM
```

## Implementation traffic models

### 1. QuACK staged-SMEM implementation

The active path in `quack/hadamard.py` performs one full fp32 SMEM round trip per logical stage:

```text
SMEM bytes per stage = 4N store + 4N load = 8N bytes
```

For `S` *effective* logical stages (defined below):

```text
SM-local bytes = 8N * S
SM-local roof  = (N * L) / (8N * S / 128)
               = 16L / S              op/clk/SM
```

#### `tail_direct_store` optimization

The kernel performs `ceil(log_N / log_ept)` butterfly stages. The last stage has
`bit_shift = log_N mod log_ept` (or `log_ept` if that's 0), which can be smaller
than `log_ept`. When that's the case, an `exchange` after the final butterfly is
still needed in general to permute values back into the original gmem layout.

When all of (a) `dtype.width == 16` (bf16 or fp16), (b) `N == N_padded`,
(c) `rows_per_block == 1`, and (d) `tail_bit_shift < log_ept` hold, the kernel
skips that final SMEM round trip and instead computes a register-side
permutation that lines up a tiled 128-bit gmem store atom whose thread/value
layouts encode the post-exchange address pattern directly. The implementation
is `_tail_direct_store_plan` / `_tail_store_copy` in `quack/hadamard.py`. The
result is that for the bf16/fp16 cases listed below the **effective stage
count** for the SM-local roof is `raw_stages - 1`, not `raw_stages`.

#### Current `EPT` table

With the current `_EPT_BY_N_PADDED` dict in `quack/hadamard.py` and bf16/fp16
inputs (tail-direct-store activated when feasible):

| N | L | EPT | log_ept | raw stages | tail_bit_shift | tail_direct_store | effective S | SM-local pressure | SM-local roof |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 8K  | 13 | 32 | 5 | 3 | 3 | yes | 2 | 16N bytes | 104.0 op/clk/SM |
| 16K | 14 | 64 | 6 | 3 | 2 | yes | 2 | 16N bytes | 112.0 op/clk/SM |
| 32K | 15 | 32 | 5 | 3 | 5 |  no | 3 | 24N bytes |  80.0 op/clk/SM |

(At N=32K with ept=32, `tail_bit_shift = log_N mod log_ept = 15 mod 5 = 0`,
which the kernel promotes to `log_ept = 5`; the feasibility predicate
`tail_bit_shift < log_ept` then fails and the final SMEM round trip is
kept. 32K therefore stays at the raw S=3.)

For fp32 inputs (or when `QUACK_HADAMARD_TAIL_DIRECT_STORE=0`) the effective
`S` equals the raw stage count, i.e. `S=3` for all three of 8K / 16K / 32K
with their respective EPTs.

### 2. Previous `fast-hadamard-transform` CUDA implementation

For bf16 and `N = 8K, 16K, 32K`, the previous CUDA implementation uses:

```text
kNThreads = 256              # 8 warps
bf16 vector load width = 8   # one uint4 per global-memory transaction
```

The per-thread element count `kNElts` scales with `N` (8K -> 32, 16K -> 64, 32K -> 128) so that `kNThreads * kNElts = N`. Per-thread butterfly stages over registers are free for this roofline; only stages that move data across threads load the SM-local pipe.

Its cross-thread movement is:

```text
within-warp SHFL stages         = log2(32) = 5
cross-warp butterfly stages     = log2(8)  = 3   (run as within-warp
                                                  SHFLs on a transposed
                                                  smem view, bracketed by
                                                  two smem transposes)
total SHFL stages on the pipe   = 8
smem transposes (round trips)   = 2
```

Each shuffle stage over all `N` fp32 values costs `4N` bytes of SM-local pipe pressure:

```text
SHFL pressure = 8 * 4N = 32N bytes
```

It also performs two full fp32 SMEM exchanges/transposes:

```text
SMEM pressure = 2 * (4N store + 4N load) = 16N bytes
```

So total SM-local pipe pressure is:

```text
SM-local bytes = 32N + 16N = 48N bytes
SM-local roof  = (N * L) / (48N / 128)
               = 8L / 3              op/clk/SM
```

| N | L | SHFL pressure | SMEM pressure | total SM-local pressure | SM-local roof |
|---:|---:|---:|---:|---:|---:|
| 8K | 13 | 32N bytes | 16N bytes | 48N bytes | 34.7 op/clk/SM |
| 16K | 14 | 32N bytes | 16N bytes | 48N bytes | 37.3 op/clk/SM |
| 32K | 15 | 32N bytes | 16N bytes | 48N bytes | 40.0 op/clk/SM |

Thus the old implementation has the equivalent SM-local pressure of **six fp32 SMEM round trips**. QuACK's current bf16/fp16 default uses **two effective round trips** at 8K and 16K (three raw stages minus the tail-direct-store saving) and **three** at 32K (where the feasibility predicate fails); fp32 keeps all three at every size.

## Roofline results

All entries are effective fp32 Hadamard operation roofs in `op/clk/SM`. The label is the predicted bottleneck (`GMEM` = global-memory-bound, `SM-local` = SM-local-pipe-bound).

### QuACK staged-SMEM implementation

bf16/fp16 with the current default `_EPT_BY_N_PADDED` (effective S=2 at 8K/16K,
S=3 at 32K):

| GPU | 8K (S=2) | 16K (S=2) | 32K (S=3) |
|---|---:|---:|---:|
| A100 80GB SXM | 43.5 GMEM | 46.9 GMEM | 50.2 GMEM |
| H100 SXM | 45.1 GMEM | 48.5 GMEM | 52.0 GMEM |
| B200 | 95.0 GMEM | 102.3 GMEM | **80.0 SM-local** |
| RTX 5090 | 14.2 GMEM | 15.3 GMEM | 16.4 GMEM |

At 8K and 16K, tail-direct-store drops the effective stage count to 2 and
all four GPUs are GMEM-bound. At 32K the predicate fails and the kernel
falls back to S=3; for B200 this trips the SM-local/GMEM crossover (B_gmem =
29.22 > 64/3 ≈ 21.3), making B200 32K SM-local-bound at 80 op/clk/SM. A100,
H100, and RTX 5090 stay GMEM-bound at 32K because their GMEM-per-SM-clock
is below the S=3 threshold.

For fp32 (or `QUACK_HADAMARD_TAIL_DIRECT_STORE=0`), every size is `S=3` and
B200 is SM-local-bound at all three: 69.3 / 74.7 / 80.0 op/clk/SM at
8K / 16K / 32K respectively.

### Previous shuffle-heavy CUDA implementation

| GPU | 8K | 16K | 32K |
|---|---:|---:|---:|
| A100 80GB SXM | 34.7 SM-local | 37.3 SM-local | 40.0 SM-local |
| H100 SXM | 34.7 SM-local | 37.3 SM-local | 40.0 SM-local |
| B200 | 34.7 SM-local | 37.3 SM-local | 40.0 SM-local |
| RTX 5090 | 14.2 GMEM | 15.3 GMEM | 16.4 GMEM |

The old implementation is SM-local-bound on A100, H100, and B200 because the 8 shuffle stages plus two SMEM exchanges create `48N` bytes of MIO pressure. RTX 5090 remains GMEM-bound because its GMEM bandwidth per SM-clock is much lower.

## Crossover rules

For QuACK staged-SMEM:

```text
T_smem / T_gmem = S * B_gmem / 64
```

So SMEM dominates GMEM when:

```text
B_gmem > 64 / S    B/clk/SM
```

Thresholds for the two relevant stage counts:

```text
S = 3 threshold: B_gmem > 21.3 B/clk/SM
  - bf16/fp16 default at N=32K
  - fp32 at all sizes
  - any size with QUACK_HADAMARD_TAIL_DIRECT_STORE=0
S = 2 threshold: B_gmem > 32.0 B/clk/SM
  - bf16/fp16 default at N=8K and N=16K
```

Mapping the four GPUs onto these thresholds:

- B200 (29.22 B/clk/SM) is above the S=3 threshold but below the S=2
  threshold, so it is SM-local-bound at any S=3 configuration (including
  the bf16 default at N=32K) and GMEM-bound at S=2.
- A100 (13.39) and H100 (13.87) are below both thresholds: GMEM-bound
  everywhere.
- RTX 5090 (4.37) is far below both thresholds: GMEM-bound everywhere.

For the old shuffle-heavy implementation:

```text
SM-local roof = 8L / 3
GMEM roof     = L * B_gmem / 4
```

SM-local dominates when:

```text
B_gmem > 32 / 3 ~= 10.7 B/clk/SM
```

A100, H100, and B200 are above this threshold; RTX 5090 is below it.

## Predicted roofline gain from QuACK vs previous FHT

This is the ratio of the overall roofline limit, not a measured speedup.

bf16/fp16 with current `_EPT_BY_N_PADDED` (effective S=2 at 8K/16K, S=3 at 32K):

| GPU | 8K (S=2) | 16K (S=2) | 32K (S=3) |
|---|---:|---:|---:|
| A100 80GB SXM | 1.26x | 1.26x | 1.26x |
| H100 SXM | 1.30x | 1.30x | 1.30x |
| B200 | 2.74x | 2.74x | **2.00x** |
| RTX 5090 | 1.00x | 1.00x | 1.00x |

(B200 32K drops from 2.74x to 2.00x because the kernel falls back to S=3 at
that size: tail-direct-store is infeasible at ept=32 / N=32K, and B200 is
SM-local-bound under S=3. The other three GPUs are GMEM-bound at 32K under
both S=2 and S=3, so their gain is unaffected by the stage count.)

Main takeaway: **the shuffle-heavy implementation is mostly an SM-local-pipe problem, not a DRAM problem, on datacenter GPUs.** Replacing most shuffle pressure with fewer full SMEM round trips moves A100/H100 closer to the GMEM roof and matters dramatically on B200. On RTX 5090, standalone bf16-in/bf16-out Hadamard is so GMEM-limited that both implementations have the same roofline limit, although the lower SM-local pressure would still matter for fused/cache-resident variants.

## Measured perf on H100 and gap to the GMEM roof

Measured GB/s and NCU stall breakdown at **M=16384, N=32768, bf16, H100 SXM**,
for three EPT choices. Pipe peak BW is 3350 GB/s (one-direction HBM3). All
three are SMEM-occupancy-capped to **1 CTA / SM** (132 KB dynamic smem per
CTA vs 228 KB SM budget).

| EPT | tpt | regs/thread | warps active (avg) | effective S | DRAM %peak | GB/s | long_scoreboard stall | mio_throttle stall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  32 | 1024 |  63 | 30.1 / 32 | 3 (no tail-direct) | **75%** | **2512** | 19.5% | 19.9% |
|  64 |  512 | 128 | 14.9 / 16 | 2 (tail-direct on) | 63% | 2102 | 35.6% | 7.6% |
| 128 |  256 | 159 |  7.8 /  8 | 2 (tail-direct on) | 63% | 2156 | 44.4% | 5.2% |

Measured SMEM bank conflicts are negligible (within 0.5% of the
conflict-free `B/128` wavefront ideal) for all three configs — the
`Sw<3,2,*>` swizzles in `exchange()` clear banks cleanly.

### Where the simple roofline disagrees with the measurement

The simple SM-local-vs-GMEM model predicts `ept=64` (with tail-direct,
`S=2`, 16N SM-local pressure) should beat `ept=32` (`S=3`, 24N SM-local
pressure) at any size: both are GMEM-bound and the lighter SMEM path should
waste less time on the MIO pipe. **The opposite is true at N=32K on H100**:
`ept=32` is ~20% faster than `ept=64`.

Two facts narrow the cause:

1. **It is not the SMEM round-trip count.** Running `ept=64` with
   `QUACK_HADAMARD_TAIL_DIRECT_STORE=0` (forcing `S=3` to match `ept=32`'s
   SMEM pressure) gives 2002 GB/s, still well below `ept=32`'s 2530 GB/s.
   Removing the SMEM saving does not close the gap.

2. **The configs differ in warps/SM.** With 132 KB dynamic SMEM, only one
   CTA fits per SM in every config, so warps/SM is set purely by
   threads/CTA: 32 / 16 / 8 for `ept = 32 / 64 / 128`. Register pressure
   (63 / 128 / 159 regs/thread) scales with `ept` because each thread holds
   `ept` fp32 values plus its LDG destinations, and with the 64K register
   file per SM that bounds warps/SM identically. So `ept` and warps/SM
   move together in this kernel; we cannot vary one without the other
   inside the current design.

### Achievable bandwidth: `torch.clone` as the empirical floor

A pure HBM read+write (`torch.clone(x)`) is the empirical upper bound on
throughput at a given dtype / N / M: same bytes moved, no compute. On the
same H100 SXM (`CUDA_VISIBLE_DEVICES=7`, `QUACK_CACHE_ENABLED=0`,
`triton.testing.do_bench` warmup=10 rep=100, M=16384, N=32768):

| dtype | clone GB/s | clone % HBM peak |
|---|---:|---:|
| bf16 | 3003 | 89.6% |
| fp32 | 3029 | 90.4% |

Clone falls ~10% short of the 3.35 TB/s spec peak. This shortfall is
kernel-agnostic (HBM ramp/drain and cross-CTA scheduling jitter at the
device level) and any hadamard kernel running on this hardware is bounded
by clone time, not by the analytical GMEM roof.

Measured hadamard vs clone at the chosen default (`ept=32`):

| dtype | hadamard GB/s | hadamard / clone | hadamard − clone (per row per SM) |
|---|---:|---:|---:|
| bf16 | 2530 | 84.2% | **+1.08 µs** |
| fp32 | 3005 | 99.2% | +0.09 µs |

The **bf16 gap of 1.08 µs/row/SM is the headroom the current kernel structure
leaves on H100**. fp32 has essentially no headroom — hadamard runs at clone
bandwidth there.

The same comparison across the three `ept` variants (all measured on GPU 7
as above; ept=64 and ept=128 forced by patching `_EPT_BY_N_PADDED[32768]`
and clearing the JIT cache between runs):

| | ept=32 (32 warps, eff S=3) | ept=64 (16 warps, eff S=3) | ept=128 (8 warps, eff S=2) |
|---|---:|---:|---:|
| bf16 hadamard GB/s | **2529** | 2006 | 2147 |
| bf16 hadamard − clone (µs/row/SM) | **+1.07** | +2.85 | +2.28 |
| fp32 hadamard GB/s | **3005** | 2998 | 2982 |
| fp32 hadamard − clone (µs/row/SM) | +0.08 | +0.11 | +0.17 |

(In this re-measurement, `TailDirectStorePlan.is_feasible` returns False
for ept=64 with bf16 N=32K, so its effective S is 3 — matching the
`QUACK_HADAMARD_TAIL_DIRECT_STORE=0` row in the earlier table above
(2002 GB/s). The ept=64 / `S=2` / 2102 GB/s row in that earlier table
reflects an older run with tail-direct active. Either way, ept=64 is
slower than ept=32.)

Two measured trends:

- The bf16 gap to clone grows when warps are halved (32w → 1.07, 16w → 2.85),
  then partially recovers at 8w because `ept=128` switches to `S=2` via
  tail-direct-store, saving one of the three SMEM round trips.
- The fp32 gap stays within 0.2 µs/row across all three configs.

We also tested whether the bf16 gap is set by per-thread LDG instruction
count: a monkey-patched bf16 with `copy_vecsize=4` (8-byte LDGs, doubling
the instruction count per thread to 8 LDGs/thread, matching fp32's per-thread
LDG count) gave 2486 GB/s vs 2519 baseline — within noise. LDG instruction
count alone is not the binding resource at `ept=32`.

### Practical consequence and follow-up

The `EPT[32768] = 32` default already reflects the table above (this
changed from `64` based on the same measurements). At this default the
bf16 kernel is at 84% of clone bandwidth on H100 SXM N=32K M=16K; closing
that gap is the active optimization target.

Within the current single-CTA-per-SM design, no knob closes the gap:

- Going to `ept=64` or `ept=128` loses warps (table above).
- Forcing `copy_vecsize=4` to double LDG instruction count per thread
  doesn't help (above).
- All three `ept` variants are SMEM-budget-limited to 1 CTA/SM, so more
  CTAs/SM is not available.

The structural change that can close the gap is **persistent CTAs with
`cp.async` row-N+1 prefetch**, overlapping the next row's HBM load with
the current row's compute + SMEM exchange. Two staged plans:

- `AI/hadamard_persistence_stage1_plan.md` — persistence only (one CTA
  per SM, looping over rows), no prefetch. Validates infrastructure and
  isolates the contribution of per-CTA launch overhead from prefetch
  overlap. Expected lift: ≤ 3%.
- `AI/hadamard_async_pipeline_plan.md` — adds `cp.async` prefetch on top.
  Target: ≥ 90% of clone, i.e. ≥ 2900 GB/s bf16 N=32K on H100 SXM (vs
  2530 today). The 1.08 µs/row/SM gap to clone is exactly what this stage
  attempts to hide under HBM.

fp32 has no comparable headroom — it already runs at clone bandwidth — so
the gain from these structural changes is expected to be bf16/fp16-only.
