# NVIDIA global-memory coalescing notes

Benchmark: `microbenchmarks/global_memory_coalescing.py`

## Pattern legend

Warp-level patterns (one warp instruction, 16B per active lane):

| key | Meaning |
| --- | --- |
| A | 32 lanes contiguous: lane `i -> base + 16*i` (512B total). |
| B | Two far 256B streams, even/odd lanes split between streams. |
| C | Two far 256B streams, lanes 0..15 in stream A and 16..31 in stream B. |
| D | Four far 128B streams, one 8-lane group per stream. |
| E | Sixteen far 32B streams, one adjacent lane pair per stream. |
| F | Thirty-two far 16B streams, one lane per stream. |
| G | Contiguous 512B with byte offset sweep. |
| H | Stride sweep: lane `i -> base + stride*i`. |
| I_pred | Four 128B lines but only 8 active lanes, one sector per line. |
| I_dup | Full-warp duplicate-address control: four unique sectors in four lines. |
| I_full | Full-warp all-sector version: 16 sectors in four lines. |
| J_packed | 16 sectors packed into four contiguous 128B lines. |
| J_16lines | 16 sectors spread across 16 different 128B lines. |
| J_16pages | 16 sectors spread across 16 different 4KB pages. |
| K_same_half | 16 sectors in 8 lines, two adjacent sectors in the same 64B half of each line. |
| K_split_half | 16 sectors in 8 lines, one sector in each 64B half of each line. |

Block-permutation patterns (`--bench block-permute`) all have 128 threads writing one
contiguous 2048B tile; only thread-to-16B-slot mapping changes.  Important names:
`identity`, `unit8_interleave`, `sector_striped`/`pair_contiguous`, `split_halves`,
`two_line_striped`, `affine17`, `bit_reverse`, and `warp_interleave`.

## Read-this-first takeaways

1. **Loads:** model H100 warp-load traffic as:
   - 32B sectors at warp/L1-L2 coalescing granularity,
   - 64B L2-to-DRAM half-line fills for misses,
   - plus request/page/partition/stall overheads.

2. **The 64B half-line tutorial claim looks valid for loads.**  The clean probe is:
   - `K_same_half`: 16 sectors, 8 lines, 8 half-lines → ~6.45 ms.
   - `K_split_half`: same 16 sectors and same 8 lines, but 16 half-lines → ~11.15 ms.
   - `J_16lines`: 16 sectors, 16 lines, 16 half-lines → ~11.37 ms.
   `K_split_half` tracks half-lines, not 128B lines.

3. **A/B/C vs D/E for loads:** lane order is not the issue.  B and C are identical-ish.
   D/E are slower because one warp instruction fans out across more independent regions /
   line requests.  A/B/C preserve contiguous stream grouping; D/E lose it.

4. **Stores:** do **not** use the load-side 64B half-line model as the primary statistic.
   Store timing is dominated by sector coverage, 128B line locality, partial-sector stores,
   store combining/queues, ordering, partition mapping, and replay-like effects.

5. **128-thread block-permutation stores:** the CTA always writes the same unique 2KB tile,
   but performance follows per-warp/per-8-thread sector+line shape and address ordering,
   not the block-unique footprint.  Neither “warp only” nor “8-thread only” fully predicts
   store time.

6. **All bandwidth numbers here are effective requested/useful bytes per second**, not
   measured DRAM bytes.  Counter validation is still needed.  On this machine,
   `ncu --query-metrics` fails with `ERR_NVGPUCTRPERM`.

## Benchmark hygiene / current address generation

An earlier version was invalid because neighboring warp bases were only 128B apart while
pattern A reads 512B per warp, causing overlap and >HBM effective bandwidth.  Current
streaming mode uses non-overlapping stream-major slots:

```text
slot = (iteration * total_warps + warp_id) % slot_count
addr = stream_region_base + slot * slot_stride + lane_pattern_offset
```

Additional safeguards:

- Far-stream patterns use separate large regions per logical stream.
- Streaming mode requires at least two full-grid waves of slots.
- `J_16pages` rotates the 32B sector offset inside each 4KB page across iterations to avoid
  repeatedly touching only one L2-sized sector set.
- E/F use a 128B per-warp slot stride.  A smaller stride let neighboring warps consume the
  other sector in the same 64B half-line, hiding sparse-sector cost.

## Run configuration for quoted H100 timings

```text
GPU: NVIDIA H100 80GB HBM3
common args: --cache-modes streaming --iterations 8192 --warmup 3 --repeats 5 \
             --data-bytes 1G --no-init
launch: 528 blocks = 132 SMs * 4 blocks/SM, 8 warps/block
full-warp requested bytes per row: 17.716740096 GB
```

## Load evidence

### 64B half-line validation

| pattern | sectors | 64B half-lines | 128B lines | time ms | useful GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 16 | 8 | 4 | 6.141 | 2885 |
| K_same_half | 16 | 8 | 8 | 6.447 | 2748 |
| K_split_half | 16 | 16 | 8 | 11.151 | 1589 |
| J_16lines | 16 | 16 | 16 | 11.369 | 1558 |

Interpretation:

- Pure 32B-sector DRAM fetch would make these much closer.
- Full 128B-line fetch would make `K_same_half` and `K_split_half` close.
- Observed behavior tracks 64B half-lines.
- `global`, `cg`, and `volatile` cache operators preserved this split, so it is not just
  default cache-policy prefetching.

### A/B/C/D/E/F load shape

| pattern | sectors | half-lines | lines | shape | time ms | useful GB/s |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| A | 16 | 8 | 4 | one contiguous 512B stream | 6.122 | 2894 |
| B | 16 | 8 | 4 | two far 256B streams, even/odd lanes | 6.211 | 2853 |
| C | 16 | 8 | 4 | two far 256B streams, half-warps | 6.200 | 2857 |
| D | 16 | 8 | 4 | four far 128B streams | 6.614 | 2679 |
| E | 16 | 16 | 16 | sixteen far 32B streams | 12.084 | 1466 |
| F | 32 | 32 | 32 | thirty-two far 16B streams | 32.184 | 550 |

Why D/E are slower than A/B/C:

- A: four adjacent full 128B lines in one stream.
- B/C: two far streams, but each has two adjacent full lines.  Lane grouping does not
  matter much.
- D: same sector/half-line/line count as A/B/C, but four independent lines/regions; ~8%
  slower from worse request locality.
- E: same sector count as A, but twice the half-line count and much more line/request
  fan-out; ~2x slower.

Cache-operator sweep (`global`, `ca`, `cg`, `volatile`) kept the same ordering.  That makes
“wrong adjacent-line prefetch” unlikely as the main explanation.

## Store evidence

### Warp-pattern stores

Store rows use `--ops store --store-modes global` and vector `st.global.v4.u32`.
Half-line stats are intentionally omitted for stores.

| pattern | sectors | lines | time ms | useful GB/s |
| --- | ---: | ---: | ---: | ---: |
| A | 16 | 4 | 5.581 | 3175 |
| B | 16 | 4 | 5.637 | 3143 |
| C | 16 | 4 | 5.624 | 3150 |
| D | 16 | 4 | 5.959 | 2973 |
| E | 16 | 16 | 17.595 | 1007 |
| F | 32 | 32 | 71.066 | 249 |
| J_16lines | 16 | 16 | 14.243 | 1244 |

Store takeaways:

- B vs C again shows lane order is not visible.
- Sparse sectors / more independent lines hurt stores much more than loads.
- Partial-sector stores are especially expensive.  In previous G-offset sweeps, vector
  offset 16 had only +1 sector but was much slower than offset 0 because first/last sectors
  were partial.
- Likely mechanisms: write combining, partial-write/RMW behavior, store queues, partition
  mapping, ordering, and replay.

### 128-thread block-permutation stores

Command:

```text
python microbenchmarks/global_memory_coalescing.py --bench block-permute --perms all \
  --store-modes global --cache-modes streaming --iterations 8192 \
  --warmup 3 --repeats 5 --data-bytes 1G --no-init
```

Each CTA writes the same contiguous 2048B tile: block-unique footprint is always 64 sectors
and 16 lines.  The table shows sector/line counts summed over the four warps and over the
sixteen 8-thread units.

| permutation | block sec/line | warp sec/line | unit8 sec/line | time ms | GB/s |
| --- | --- | --- | --- | ---: | ---: |
| identity | 64/16 | 64/16 | 64/16 | 2.839 | 3120 |
| unit8_interleave | 64/16 | 64/16 | 64/16 | 2.941 | 3012 |
| lane_reverse | 64/16 | 64/16 | 64/16 | 2.842 | 3117 |
| block_reverse | 64/16 | 64/16 | 64/16 | 2.833 | 3127 |
| split_halves | 64/16 | 64/32 | 64/32 | 2.839 | 3120 |
| two_line_striped | 64/16 | 64/32 | 64/32 | 2.931 | 3022 |
| sector_striped / pair_contiguous | 64/16 | 64/64 | 64/64 | 3.057 | 2898 |
| affine17 | 64/16 | 96/64 | 128/128 | 4.312 | 2054 |
| bit_reverse | 64/16 | 128/64 | 128/128 | 4.854 | 1825 |
| warp_interleave | 64/16 | 128/64 | 128/64 | 6.025 | 1470 |

Block-permutation takeaways:

- Block-unique footprint alone predicts nothing; it is identical for all rows.
- Lane reversal/order within the same footprint is cheap: identity, lane_reverse, and
  block_reverse are identical.
- Adjacent 8 threads writing one 128B line is best (`identity`).  `unit8_interleave` has
  identical counts but slightly worse ordering.
- Adjacent thread pairs writing minimal 32B sectors (`sector_striped`) is good but ~8%
  slower than the 8-thread-line-contiguous case because each 8-thread group spans four
  lines.
- 8-thread counts and warp counts both help, but neither fully predicts time.  Example:
  `bit_reverse` and `warp_interleave` have the same warp counts but different timing.
- Store address order, partition mapping, store-queue behavior, and cross-warp combining
  matter beyond sector/line counts.

## Working model

### Loads

1. Warp coalesces requests into 32B sectors.
2. L2 miss traffic appears to fetch 64B half-lines from DRAM.
3. Time also depends on request/line/page locality, MSHR/queue pressure, memory partition
   mapping, TLB effects, replay, and stalls.

### Stores

1. Track 32B sectors and 128B line locality; ignore half-line stats as a primary store
   model.
2. Full-sector and full-line coverage is much better than sparse/partial stores.
3. Address ordering and partition/store-queue/write-combine behavior can dominate even
   when sector/line counts are equal.

## Counter validation needed

Collect, when permissions allow:

- L1TEX global load/store requests and sectors.
- L2 read/write sectors or bytes.
- DRAM read/write bytes.
- Replay / excessive-sector metrics if exposed.
- Warp stall reasons, especially long scoreboard / memory dependency / TLB-like stalls.

Diagnostic expectations:

- `K_same_half` and A should have similar DRAM bytes per useful byte; `K_split_half` and
  `J_16lines` should have about 2x if the 64B half-line model is right.
- D should have near-A bytes but more requests/stalls if request fan-out explains its
  slowdown.
- Store counters/replay should clarify partial-sector and sparse-store penalties.
