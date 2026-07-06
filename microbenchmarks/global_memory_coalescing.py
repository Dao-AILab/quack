#!/usr/bin/env python3
r"""CuTe-DSL microbenchmarks for NVIDIA warp global-memory coalescing.

Purpose
-------
Measure memory traffic, replay behavior, and throughput for warp-local global load/store
patterns.  In aligned/vector rows, each active lane issues exactly one 16-byte vector
memory op per loop iteration using inline PTX:

    ld.global.v4.u32 / ld.global.ca.v4.u32 / ld.global.cg.v4.u32
    st.global.v4.u32 / st.global.{wb,cg,cs,wt}.v4.u32

The loaded u32x4 values are accumulated into registers; store rows generate u32x4 values
in registers and write them to global memory.  A final checksum is written so the loop
state stays live.  The CSV reports timing plus the theoretical number of unique 32B
sectors and 128B lines touched by one warp memory instruction.

Important misalignment note
---------------------------
Current NVIDIA hardware/toolchains fault unaligned 16B vector global op addresses in
local H100 testing.  For offset rows such as 4/8/20/28, the script defaults to a clearly
labeled `*_scalar4` fallback: four scalar 32-bit ops covering the same 16B byte span.
Use `--misaligned-vector error` if you want strict one-16B-instruction rows only.

Setup
-----
There is no separate build step; CuTe DSL JIT-compiles the kernel on first use.
Use the normal repo dev environment:

    pip install -e '.[dev]'
    # For CUDA 13.x wheels, if needed:
    pip install -e '.[dev,cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

Quick smoke run:

    python microbenchmarks/global_memory_coalescing.py \
      --patterns A --iterations 1024 --repeats 3

Write CSV:

    python microbenchmarks/global_memory_coalescing.py --patterns all \
      --cache-modes streaming resident \
      --ops load --load-modes global \
      --csv coalescing.csv

Run stores instead of loads:

    python microbenchmarks/global_memory_coalescing.py --patterns all \
      --cache-modes streaming \
      --ops store --store-modes global \
      --csv coalescing_store.csv

Run the 128-thread / 2KB block-permutation store benchmark:

    python microbenchmarks/global_memory_coalescing.py --bench block-permute \
      --perms all --store-modes global --cache-modes streaming \
      --csv block_permute.csv

Patterns
--------
A:
    Contiguous 512B warp load: lane i -> base + 16*i.
    Expected aligned behavior: 16 sectors, 4 lines.
B:
    Even lanes load one 256B stream; odd lanes load a far 256B stream.
    Expected: 16 sectors, 4 lines, two far ranges.
C:
    Lanes 0..15 load one 256B stream; lanes 16..31 load a far 256B stream.
    Same sectors/lines as B, different lane grouping.
D:
    Four groups of eight lanes load four far 128B streams.
    Expected: 16 sectors, 4 far line groups.
E:
    Sixteen far 32B streams, two adjacent lanes per stream.
    Expected: 16 sectors, 16 lines/regions.
F:
    Thirty-two far 16B streams, one lane per region.
    Expected: 32 sectors, 32 lines/regions.
G:
    Contiguous 512B with byte offset.
    Sector count changes when 16B spans cross 32B sectors and when the warp span
    crosses an extra sector.
H:
    Stride sweep: lane i -> base + stride*i.
    Sector count grows with stride.
I_pred / I_dup / I_full:
    Same four 128B lines but fewer/more 32B sectors.  `I_pred` uses only 8 active
    lanes.  `I_dup` is a full-warp duplicate-address variant, not a unique-byte
    throughput test.  `I_full` is the full-warp all-sector case.
J_packed / J_16lines / J_16pages:
    Sixteen sectors packed into 4 lines vs spread over 16 lines/pages.  These keep
    sector count fixed while changing cache-line/request/page locality.
K_same_half / K_split_half:
    Half-line granularity probes.  Both touch 16 sectors in 8 lines; `K_same_half` uses
    one 64B half-line per 128B line, while `K_split_half` uses both halves.  If L2 fetches
    from DRAM in 64B half-lines, `K_split_half` should look closer to `J_16lines` than to
    `K_same_half`.

Common runs
-----------
Single streaming pattern:

    python microbenchmarks/global_memory_coalescing.py \
      --patterns A --cache-modes streaming

Compare lane order / grouping:

    python microbenchmarks/global_memory_coalescing.py \
      --patterns B C --cache-modes streaming resident

Misalignment sweep.  Offsets that are not 16B-aligned report as `global_scalar4`
unless `--misaligned-vector error` is set:

    python microbenchmarks/global_memory_coalescing.py --patterns G \
      --offsets 0,4,8,16,20,28,32,64

Strict vector-only offset sweep:

    python microbenchmarks/global_memory_coalescing.py --patterns G \
      --offsets 0,16,32,64 --misaligned-vector error

Stride sweep:

    python microbenchmarks/global_memory_coalescing.py --patterns H \
      --strides 16,32,64,128,256

Same sector count, different 128B-line/page locality:

    python microbenchmarks/global_memory_coalescing.py --patterns J \
      --cache-modes streaming resident

Half-line fetch-granularity probe:

    python microbenchmarks/global_memory_coalescing.py --patterns A K J_16lines \
      --ops load --load-modes global --cache-modes streaming

Compare load cache operators:

    python microbenchmarks/global_memory_coalescing.py \
      --patterns A E F J_16pages \
      --ops load --load-modes global ca cg volatile \
      --cache-modes streaming

Compare store cache operators:

    python microbenchmarks/global_memory_coalescing.py \
      --patterns A E F J_16pages \
      --ops store --store-modes global wb cg cs wt volatile \
      --cache-modes streaming

Force scalar fallback explicitly for scalar-control rows:

    python microbenchmarks/global_memory_coalescing.py --patterns G \
      --load-modes global_scalar4 ca_scalar4 cg_scalar4 \
      --offsets 4,8,20,28

Low-occupancy / easier Nsight Compute analysis:

    python microbenchmarks/global_memory_coalescing.py --patterns A --low-occupancy \
      --warmup 0 --repeats 1 --iterations 4096

Throughput saturation defaults launch many warps: SMs * --blocks-per-sm blocks and
--warps-per-block warps/block.  Tune explicitly:

    python microbenchmarks/global_memory_coalescing.py --patterns A \
      --blocks-per-sm 8 --warps-per-block 8 --iterations 8192

Cache-resident vs streaming modes
---------------------------------
--cache-modes resident:
    Each warp repeatedly reuses a bounded slot range.  Default `--resident-bytes 32M`
    is intended to be L2-resident on large data-center GPUs.  For large-footprint patterns
    this mode may intentionally reuse slots within one launch; treat it as cache/issue
    behavior, not HBM bandwidth.
--cache-modes streaming:
    Each loop iteration maps `(iteration, warp)` to a stream-major slot.  The script
    requires at least two non-overlapping full-grid waves of slots, so neighboring warps
    no longer overlap as they did in the original version.  `J_16pages` also rotates the
    32B sector offset inside each 4KB page across iterations so the sparse page pattern
    does not fit in L2 by touching only one sector per page forever.  Default
    `--data-bytes 1G` keeps the working set large for the included patterns.

Far stream spacing knobs:

    --far-stride-bytes 4M       # B/C/D far streams
    --region-stride-bytes 4K    # E/F separated regions
    --page-stride-bytes 4K      # J_16pages

CSV columns
-----------
The CSV contains:

    operation, pattern, load_mode, cache_mode, offset, stride, active_lanes,
    useful_bytes_per_warp_inst, requested_bytes,
    expected_32B_sectors, expected_64B_halflines, expected_128B_lines,
    slot_stride_bytes, stream_span_bytes, slot_count,
    time_ms, bandwidth_GBps, checksum

`operation` is `load` or `store`.  `load_mode` names the PTX memory-op variant; for store
rows it is the store mode.  `expected_64B_halflines` is load-only; store-only CSV output
omits that column.  requested_bytes = active_lanes * 16B * iterations * blocks *
warps_per_block.
bandwidth_GBps uses those requested/useful bytes, not the expected sector traffic and not
measured DRAM bytes.  Streaming mode avoids same-grid overlap, but this is still an
effective requested-byte rate; use Nsight Compute counters to compare requested bytes with
actual L1/L2/DRAM traffic.

For `--bench block-permute`, the CSV instead reports block-unique sector/line counts,
per-warp sector/line sums, and per-8-thread-unit sector/line sums.  Half-line stats are
omitted for this store-only benchmark.  The whole 128-thread CTA always covers the same
unique 2048B tile; comparing warp vs 8-thread-unit sums helps test which granularity better
explains store timing.

Nsight Compute
--------------
First pass with built-in sections and exactly one benchmark row:

    ncu --target-processes all \
      --set full \
      --section SpeedOfLight \
      --section MemoryWorkloadAnalysis \
      --section SchedulerStats \
      --section WarpStateStats \
      python microbenchmarks/global_memory_coalescing.py \
        --patterns A --cache-modes streaming --ops load --load-modes global \
        --warmup 0 --repeats 1 --iterations 8192 --low-occupancy --no-init

Throughput-mode counters: remove `--low-occupancy` and launch many blocks/warps:

    ncu --target-processes all --section SpeedOfLight --section MemoryWorkloadAnalysis \
      python microbenchmarks/global_memory_coalescing.py \
        --patterns J --cache-modes streaming --ops load --load-modes global \
        --warmup 0 --repeats 1 --iterations 8192 --no-init

Metric names vary across Nsight Compute and GPU generations.  Start with:

    ncu --query-metrics | grep -Ei \
      'l1tex.*global.*(ld|st).*(sector|request)|lts.*sector|dram.*bytes|stall|replay|excessive'

Metrics/sections to collect when available:

* Global-load/store request and sector counts:
  - l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
  - l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
  - l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
  - l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
  - sector/request ratios from Memory Workload Analysis
* L2 sectors / bytes:
  - lts__t_sectors_op_read.sum
  - lts__t_sectors_srcunit_tex_op_read.sum
  - L2 throughput/byte counters from Memory Workload Analysis
* DRAM read/write bytes / achieved bandwidth:
  - dram__bytes_read.sum
  - dram__bytes_write.sum
  - DRAM throughput metrics from SpeedOfLight / Memory Workload Analysis
* Replay / excessive sectors, if exposed on the target:
  - metrics containing replay or excessive
  - source-level Memory Workload Analysis tables
* Warp stall reasons:
  - SchedulerStats and WarpStateStats
  - long scoreboard / memory dependency stall metrics

Always verify the generated SASS contains a single vector global load/store for the
aligned variant you are analyzing.  Rows whose `load_mode` ends in `_scalar4` deliberately
use four scalar ops; interpret those as a sector-boundary fallback/control, not as a
one-instruction 16B memory-op experiment.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Iterable

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from quack.compile_utils import make_fake_tensor as fake_tensor


WARP_SIZE = 32
VEC_BYTES = 16
SECTOR_BYTES = 32
HALF_LINE_BYTES = 64
LINE_BYTES = 128
DEFAULT_DATA_BYTES = 1024 * 1024 * 1024
DEFAULT_RESIDENT_BYTES = 32 * 1024 * 1024
DEFAULT_FAR_STRIDE_BYTES = 4 * 1024 * 1024
DEFAULT_REGION_STRIDE_BYTES = 4 * 1024
DEFAULT_PAGE_STRIDE_BYTES = 4 * 1024

PAT_A = 0
PAT_B = 1
PAT_C = 2
PAT_D = 3
PAT_E = 4
PAT_F = 5
PAT_G = 6
PAT_H = 7
PAT_I_PRED = 8
PAT_I_DUP = 9
PAT_I_FULL = 10
PAT_J_PACKED = 11
PAT_J_16LINES = 12
PAT_J_16PAGES = 13
PAT_K_SAME_HALF = 14
PAT_K_SPLIT_HALF = 15

BLOCK_PERM_THREADS = 128
BLOCK_PERM_WARPS = BLOCK_PERM_THREADS // WARP_SIZE
BLOCK_PERM_UNIT8 = 8
BLOCK_PERM_UNITS8 = BLOCK_PERM_THREADS // BLOCK_PERM_UNIT8
BLOCK_PERM_TILE_BYTES = BLOCK_PERM_THREADS * VEC_BYTES

PERM_IDENTITY = 0
PERM_LANE_REVERSE = 1
PERM_BLOCK_REVERSE = 2
PERM_WARP_INTERLEAVE = 3
PERM_SECTOR_STRIPED = 4
PERM_TWO_LINE_STRIPED = 5
PERM_SPLIT_HALVES = 6
PERM_UNIT8_INTERLEAVE = 7
PERM_AFFINE17 = 8
PERM_BIT_REVERSE = 9


@dataclass(frozen=True)
class PatternSpec:
    key: str
    pattern_id: int
    label: str
    description: str


@dataclass(frozen=True)
class StreamLayout:
    slot_stride_bytes: int
    stream_span_bytes: int
    slot_count: int
    note: str = ""

    @property
    def slot_stride_i32(self) -> int:
        return self.slot_stride_bytes // 4

    @property
    def stream_span_i32(self) -> int:
        return self.stream_span_bytes // 4


@dataclass(frozen=True)
class PermSpec:
    key: str
    perm_id: int
    label: str
    description: str


PATTERNS: dict[str, PatternSpec] = {
    "A": PatternSpec(
        "A",
        PAT_A,
        "A_contiguous_512B",
        "lane i loads base + 16*i",
    ),
    "B": PatternSpec(
        "B",
        PAT_B,
        "B_even_odd_2x256B_far",
        "even lanes stream from A, odd lanes stream from far B",
    ),
    "C": PatternSpec(
        "C",
        PAT_C,
        "C_halfwarp_2x256B_far",
        "lanes 0..15 stream from A, lanes 16..31 stream from far B",
    ),
    "D": PatternSpec(
        "D",
        PAT_D,
        "D_four_128B_far",
        "four groups of eight lanes stream from four far regions",
    ),
    "E": PatternSpec(
        "E",
        PAT_E,
        "E_sixteen_32B_far",
        "pairs of lanes load adjacent 16B chunks in sixteen separated sectors",
    ),
    "F": PatternSpec(
        "F",
        PAT_F,
        "F_thirtytwo_16B_far",
        "each lane loads the first 16B of a separated region",
    ),
    "G": PatternSpec(
        "G",
        PAT_G,
        "G_misaligned_contiguous_512B",
        "lane i loads base + 16*i + offset",
    ),
    "H": PatternSpec(
        "H",
        PAT_H,
        "H_strided",
        "lane i loads base + stride*i",
    ),
    "I_pred": PatternSpec(
        "I_pred",
        PAT_I_PRED,
        "I_4lines_4sectors_pred8",
        "8 active lanes touch one 32B sector in each of four 128B lines",
    ),
    "I_dup": PatternSpec(
        "I_dup",
        PAT_I_DUP,
        "I_4lines_4sectors_fullwarp_duplicates",
        "full warp duplicates two 16B addresses in one sector per 128B line",
    ),
    "I_full": PatternSpec(
        "I_full",
        PAT_I_FULL,
        "I_4lines_16sectors_fullwarp",
        "full warp touches all four 32B sectors in each of four 128B lines",
    ),
    "J_packed": PatternSpec(
        "J_packed",
        PAT_J_PACKED,
        "J_16sectors_packed_4lines",
        "16 sectors packed into four contiguous 128B lines",
    ),
    "J_16lines": PatternSpec(
        "J_16lines",
        PAT_J_16LINES,
        "J_16sectors_spread_16lines",
        "16 sectors spread across sixteen different 128B lines",
    ),
    "J_16pages": PatternSpec(
        "J_16pages",
        PAT_J_16PAGES,
        "J_16sectors_spread_16pages",
        "16 sectors spread across sixteen different 4KB pages",
    ),
    "K_same_half": PatternSpec(
        "K_same_half",
        PAT_K_SAME_HALF,
        "K_16sectors_8lines_same_halves",
        "16 sectors as eight adjacent sector pairs in one 64B half per line",
    ),
    "K_split_half": PatternSpec(
        "K_split_half",
        PAT_K_SPLIT_HALF,
        "K_16sectors_8lines_split_halves",
        "16 sectors as two sectors in opposite 64B halves of each of eight lines",
    ),
}

PATTERN_GROUPS: dict[str, list[str]] = {
    "all": [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I_pred",
        "I_dup",
        "I_full",
        "J_packed",
        "J_16lines",
        "J_16pages",
        "K_same_half",
        "K_split_half",
    ],
    "I": ["I_pred", "I_dup", "I_full"],
    "J": ["J_packed", "J_16lines", "J_16pages"],
    "K": ["K_same_half", "K_split_half"],
}

PERMS: dict[str, PermSpec] = {
    "identity": PermSpec(
        "identity",
        PERM_IDENTITY,
        "identity_contiguous_warps",
        "warp w writes slots 32*w..32*w+31",
    ),
    "lane_reverse": PermSpec(
        "lane_reverse",
        PERM_LANE_REVERSE,
        "lane_reverse_contiguous_warps",
        "same slots as identity but lanes reversed within each warp",
    ),
    "block_reverse": PermSpec(
        "block_reverse",
        PERM_BLOCK_REVERSE,
        "block_reverse_contiguous_warps",
        "whole block slot order reversed; each warp remains contiguous",
    ),
    "warp_interleave": PermSpec(
        "warp_interleave",
        PERM_WARP_INTERLEAVE,
        "warp_interleave_stride4_slots",
        "warp w writes slots w, w+4, w+8, ... across the 2KB tile",
    ),
    "sector_striped": PermSpec(
        "sector_striped",
        PERM_SECTOR_STRIPED,
        "pair_contiguous_sector_minimal",
        "adjacent thread pairs write adjacent 16B slots; each pair covers one 32B sector",
    ),
    "two_line_striped": PermSpec(
        "two_line_striped",
        PERM_TWO_LINE_STRIPED,
        "unit8_two_lines_striped",
        "each 8-thread group writes two separated 64B spans across two lines",
    ),
    "split_halves": PermSpec(
        "split_halves",
        PERM_SPLIT_HALVES,
        "split_halves_2sectors_per_line",
        "each warp writes one sector in each half of eight 128B lines",
    ),
    "unit8_interleave": PermSpec(
        "unit8_interleave",
        PERM_UNIT8_INTERLEAVE,
        "unit8_contiguous_line_minimal_interleaved",
        "adjacent 8-thread groups write one 128B line, with lines interleaved across warps",
    ),
    "affine17": PermSpec(
        "affine17",
        PERM_AFFINE17,
        "affine17_permutation",
        "slot = (17 * thread) mod 128",
    ),
    "bit_reverse": PermSpec(
        "bit_reverse",
        PERM_BIT_REVERSE,
        "bit_reverse7_permutation",
        "7-bit bit-reversal of thread id",
    ),
}

PERM_GROUPS: dict[str, list[str]] = {"all": list(PERMS.keys())}

CSV_COLUMNS = [
    "operation",
    "pattern",
    "load_mode",
    "cache_mode",
    "offset",
    "stride",
    "active_lanes",
    "useful_bytes_per_warp_inst",
    "requested_bytes",
    "expected_32B_sectors",
    "expected_64B_halflines",
    "expected_128B_lines",
    "slot_stride_bytes",
    "stream_span_bytes",
    "slot_count",
    "time_ms",
    "bandwidth_GBps",
    "checksum",
]

BLOCK_PERM_CSV_COLUMNS = [
    "permutation",
    "store_mode",
    "cache_mode",
    "threads",
    "tile_bytes",
    "requested_bytes",
    "block_unique_32B_sectors",
    "block_unique_128B_lines",
    "sum_warp_32B_sectors",
    "sum_warp_128B_lines",
    "sum_unit8_32B_sectors",
    "sum_unit8_128B_lines",
    "warp_32B_sectors",
    "warp_128B_lines",
    "unit8_32B_sectors",
    "unit8_128B_lines",
    "tile_count",
    "time_ms",
    "bandwidth_GBps",
    "checksum",
]


@dsl_user_op
def load_global_v4_u32(
    gmem_ptr: cute.Pointer,
    load_mode: str,
    *,
    loc=None,
    ip=None,
) -> tuple[Int32, Int32, Int32, Int32]:
    """Inline PTX 16-byte global load returning four u32 values.

    `has_side_effects=True` deliberately keeps repeated cache-resident loads in
    the timed loop instead of letting LLVM treat the inline asm as hoistable pure
    computation.  The returned values are also accumulated into a checksum.
    """
    asm_by_mode = {
        "global": "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "ca": "ld.global.ca.v4.u32 {$0, $1, $2, $3}, [$4];",
        "cg": "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];",
        "volatile": "ld.volatile.global.v4.u32 {$0, $1, $2, $3}, [$4];",
    }
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [gmem_ptr.llvm_ptr],
        asm_by_mode[load_mode],
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), values, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), values, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), values, [2], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), values, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def load_global_scalar4_u32(
    gmem_ptr: cute.Pointer,
    load_mode: str,
    *,
    loc=None,
    ip=None,
) -> tuple[Int32, Int32, Int32, Int32]:
    """Fallback for 4B-aligned but not 16B-aligned 16-byte spans.

    NVIDIA faults unaligned `ld.global.v4.u32` addresses on current H100 testing.
    This fallback uses four scalar 32-bit loads so offset 4/8/20/28 rows can still
    probe sector-boundary effects.  Rows using this helper report a `*_scalar4`
    load mode because they are not one 16B memory instruction per lane.
    """
    asm_by_mode = {
        "global": "ld.global.u32 $0, [$4];\n\t"
        "ld.global.u32 $1, [$4+4];\n\t"
        "ld.global.u32 $2, [$4+8];\n\t"
        "ld.global.u32 $3, [$4+12];",
        "ca": "ld.global.ca.u32 $0, [$4];\n\t"
        "ld.global.ca.u32 $1, [$4+4];\n\t"
        "ld.global.ca.u32 $2, [$4+8];\n\t"
        "ld.global.ca.u32 $3, [$4+12];",
        "cg": "ld.global.cg.u32 $0, [$4];\n\t"
        "ld.global.cg.u32 $1, [$4+4];\n\t"
        "ld.global.cg.u32 $2, [$4+8];\n\t"
        "ld.global.cg.u32 $3, [$4+12];",
        "volatile": "ld.volatile.global.u32 $0, [$4];\n\t"
        "ld.volatile.global.u32 $1, [$4+4];\n\t"
        "ld.volatile.global.u32 $2, [$4+8];\n\t"
        "ld.volatile.global.u32 $3, [$4+12];",
    }
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [gmem_ptr.llvm_ptr],
        asm_by_mode[load_mode],
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), values, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), values, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), values, [2], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), values, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def store_global_v4_u32(
    gmem_ptr: cute.Pointer,
    v0: Int32,
    v1: Int32,
    v2: Int32,
    v3: Int32,
    store_mode: str,
    *,
    loc=None,
    ip=None,
) -> None:
    asm_by_mode = {
        "global": "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "wb": "st.global.wb.v4.u32 [$0], {$1, $2, $3, $4};",
        "cg": "st.global.cg.v4.u32 [$0], {$1, $2, $3, $4};",
        "cs": "st.global.cs.v4.u32 [$0], {$1, $2, $3, $4};",
        "wt": "st.global.wt.v4.u32 [$0], {$1, $2, $3, $4};",
        "volatile": "st.volatile.global.v4.u32 [$0], {$1, $2, $3, $4};",
    }
    llvm.inline_asm(
        None,
        [
            gmem_ptr.llvm_ptr,
            Int32(v0).ir_value(loc=loc, ip=ip),
            Int32(v1).ir_value(loc=loc, ip=ip),
            Int32(v2).ir_value(loc=loc, ip=ip),
            Int32(v3).ir_value(loc=loc, ip=ip),
        ],
        asm_by_mode[store_mode],
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_scalar4_u32(
    gmem_ptr: cute.Pointer,
    v0: Int32,
    v1: Int32,
    v2: Int32,
    v3: Int32,
    store_mode: str,
    *,
    loc=None,
    ip=None,
) -> None:
    asm_by_mode = {
        "global": "st.global.u32 [$0], $1;\n\t"
        "st.global.u32 [$0+4], $2;\n\t"
        "st.global.u32 [$0+8], $3;\n\t"
        "st.global.u32 [$0+12], $4;",
        "wb": "st.global.wb.u32 [$0], $1;\n\t"
        "st.global.wb.u32 [$0+4], $2;\n\t"
        "st.global.wb.u32 [$0+8], $3;\n\t"
        "st.global.wb.u32 [$0+12], $4;",
        "cg": "st.global.cg.u32 [$0], $1;\n\t"
        "st.global.cg.u32 [$0+4], $2;\n\t"
        "st.global.cg.u32 [$0+8], $3;\n\t"
        "st.global.cg.u32 [$0+12], $4;",
        "cs": "st.global.cs.u32 [$0], $1;\n\t"
        "st.global.cs.u32 [$0+4], $2;\n\t"
        "st.global.cs.u32 [$0+8], $3;\n\t"
        "st.global.cs.u32 [$0+12], $4;",
        "wt": "st.global.wt.u32 [$0], $1;\n\t"
        "st.global.wt.u32 [$0+4], $2;\n\t"
        "st.global.wt.u32 [$0+8], $3;\n\t"
        "st.global.wt.u32 [$0+12], $4;",
        "volatile": "st.volatile.global.u32 [$0], $1;\n\t"
        "st.volatile.global.u32 [$0+4], $2;\n\t"
        "st.volatile.global.u32 [$0+8], $3;\n\t"
        "st.volatile.global.u32 [$0+12], $4;",
    }
    llvm.inline_asm(
        None,
        [
            gmem_ptr.llvm_ptr,
            Int32(v0).ir_value(loc=loc, ip=ip),
            Int32(v1).ir_value(loc=loc, ip=ip),
            Int32(v2).ir_value(loc=loc, ip=ip),
            Int32(v3).ir_value(loc=loc, ip=ip),
        ],
        asm_by_mode[store_mode],
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class GlobalCoalescingBench:
    def __init__(
        self,
        *,
        operation: str,
        pattern_id: int,
        mem_mode: str,
        cache_mode: str,
        offset_bytes: int,
        stride_bytes: int,
        far_stride_bytes: int,
        region_stride_bytes: int,
        page_stride_bytes: int,
        num_blocks: int,
        warps_per_block: int,
    ):
        self.is_store = operation == "store"
        self.pattern_id = pattern_id
        self.use_scalar4 = mem_mode.endswith("_scalar4")
        self.mem_mode = mem_mode.removesuffix("_scalar4")
        self.cache_mode = cache_mode
        self.offset_bytes = offset_bytes
        self.stride_bytes = stride_bytes
        self.far_stride_bytes = far_stride_bytes
        self.region_stride_bytes = region_stride_bytes
        self.page_stride_bytes = page_stride_bytes
        self.num_blocks = num_blocks
        self.warps_per_block = warps_per_block
        self.threads = warps_per_block * WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mData: cute.Tensor,
        mOut: cute.Tensor,
        iterations: Int32,
        slot_count: Int64,
        slot_stride_i32: Int64,
        stream_span_i32: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(mData, mOut, iterations, slot_count, slot_stride_i32, stream_span_i32).launch(
            grid=[self.num_blocks, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mData: cute.Tensor,
        mOut: cute.Tensor,
        iterations: Int32,
        slot_count: Int64,
        slot_stride_i32: Int64,
        stream_span_i32: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        lane = tidx & 31
        warp_in_block = tidx >> 5
        warp_global_i64 = bidx.to(Int64) * Int64(self.warps_per_block) + warp_in_block.to(Int64)
        total_warps_i64 = Int64(self.num_blocks * self.warps_per_block)

        acc0 = Int32(tidx + 1)
        acc1 = Int32(tidx + 17)
        acc2 = Int32(tidx + 33)
        acc3 = Int32(tidx + 49)

        for _it in cutlass.range(iterations, unroll=1):
            if const_expr(self.cache_mode == "streaming"):
                linear_i64 = _it.to(Int64) * total_warps_i64 + warp_global_i64
                slot_i64 = linear_i64 % slot_count
            else:
                slot_i64 = warp_global_i64 % slot_count

            region = Int32(0)
            local_bytes = lane * 16
            extra_elem_i64 = Int64(0)
            active = cutlass.Boolean(True)

            if const_expr(self.pattern_id == PAT_A):
                local_bytes = lane * 16
            elif const_expr(self.pattern_id == PAT_B):
                pair = lane >> 1
                region = lane & 1
                local_bytes = pair * 16
            elif const_expr(self.pattern_id == PAT_C):
                if lane < 16:
                    region = Int32(0)
                    local_bytes = lane * 16
                else:
                    region = Int32(1)
                    local_bytes = (lane - 16) * 16
            elif const_expr(self.pattern_id == PAT_D):
                region = lane >> 3
                local = lane & 7
                local_bytes = local * 16
            elif const_expr(self.pattern_id == PAT_E):
                region = lane >> 1
                sub = lane & 1
                local_bytes = sub * 16
            elif const_expr(self.pattern_id == PAT_F):
                region = lane
                local_bytes = Int32(0)
            elif const_expr(self.pattern_id == PAT_G):
                local_bytes = lane * 16 + self.offset_bytes
            elif const_expr(self.pattern_id == PAT_H):
                local_bytes = lane * self.stride_bytes
            elif const_expr(self.pattern_id == PAT_I_PRED):
                group = lane >> 3
                local = lane & 7
                active = local < 2
                local_bytes = group * LINE_BYTES + local * 16
            elif const_expr(self.pattern_id == PAT_I_DUP):
                group = lane >> 3
                local = lane & 7
                local_bytes = group * LINE_BYTES + (local & 1) * 16
            elif const_expr(self.pattern_id == PAT_I_FULL):
                local_bytes = lane * 16
            elif const_expr(self.pattern_id == PAT_J_PACKED):
                local_bytes = lane * 16
            elif const_expr(self.pattern_id == PAT_J_16LINES):
                pair = lane >> 1
                sub = lane & 1
                local_bytes = pair * LINE_BYTES + sub * 16
            elif const_expr(self.pattern_id == PAT_J_16PAGES):
                pair = lane >> 1
                sub = lane & 1
                local_bytes = pair * self.page_stride_bytes + sub * 16
                if const_expr(self.cache_mode == "streaming"):
                    # Revisit the same pages with a different 32B sector offset on
                    # each outer iteration.  Without this, a 1GB allocation only
                    # touches ~9MB of actual sectors for J_16pages because each
                    # 64KB slot would use one 32B sector in each 4KB page.
                    sectors_per_page = self.page_stride_bytes // SECTOR_BYTES
                    extra_elem_i64 = ((_it % sectors_per_page) * (SECTOR_BYTES // 4)).to(Int64)
            elif const_expr(self.pattern_id == PAT_K_SAME_HALF):
                group = lane >> 2
                local = lane & 3
                local_bytes = group * LINE_BYTES + local * 16
            else:  # PAT_K_SPLIT_HALF
                group = lane >> 2
                local = lane & 3
                if local < 2:
                    local_bytes = group * LINE_BYTES + local * 16
                else:
                    local_bytes = group * LINE_BYTES + HALF_LINE_BYTES + (local - 2) * 16

            if active:
                elem_i64 = (
                    region.to(Int64) * stream_span_i32
                    + slot_i64 * slot_stride_i32
                    + (local_bytes >> 2).to(Int64)
                    + extra_elem_i64
                )
                if const_expr(self.is_store):
                    s0 = acc0 + _it + lane
                    s1 = acc1 + _it + lane
                    s2 = acc2 + _it + lane
                    s3 = acc3 + _it + lane
                    if const_expr(self.use_scalar4):
                        store_global_scalar4_u32(
                            mData.iterator + elem_i64,
                            s0,
                            s1,
                            s2,
                            s3,
                            self.mem_mode,
                        )
                    else:
                        store_global_v4_u32(
                            mData.iterator + elem_i64,
                            s0,
                            s1,
                            s2,
                            s3,
                            self.mem_mode,
                        )
                    acc0 += Int32(1)
                    acc1 += Int32(3)
                    acc2 += Int32(5)
                    acc3 += Int32(7)
                else:
                    if const_expr(self.use_scalar4):
                        v0, v1, v2, v3 = load_global_scalar4_u32(
                            mData.iterator + elem_i64,
                            self.mem_mode,
                        )
                    else:
                        v0, v1, v2, v3 = load_global_v4_u32(
                            mData.iterator + elem_i64,
                            self.mem_mode,
                        )
                    acc0 += v0
                    acc1 += v1
                    acc2 += v2
                    acc3 += v3

        # One 4B checksum store per thread keeps all lanes' load results live.
        mOut[bidx * self.threads + tidx] = acc0 + acc1 + acc2 + acc3


@lru_cache(maxsize=None)
def _compile(
    operation: str,
    pattern_id: int,
    mem_mode: str,
    cache_mode: str,
    offset_bytes: int,
    stride_bytes: int,
    far_stride_bytes: int,
    region_stride_bytes: int,
    page_stride_bytes: int,
    num_blocks: int,
    warps_per_block: int,
    data_elems: int,
    out_elems: int,
):
    data = fake_tensor(Int32, (data_elems,), divisibility=32)
    out = fake_tensor(Int32, (out_elems,), divisibility=1)
    op = GlobalCoalescingBench(
        operation=operation,
        pattern_id=pattern_id,
        mem_mode=mem_mode,
        cache_mode=cache_mode,
        offset_bytes=offset_bytes,
        stride_bytes=stride_bytes,
        far_stride_bytes=far_stride_bytes,
        region_stride_bytes=region_stride_bytes,
        page_stride_bytes=page_stride_bytes,
        num_blocks=num_blocks,
        warps_per_block=warps_per_block,
    )
    return cute.compile(
        op,
        data,
        out,
        Int32(0),
        Int64(0),
        Int64(0),
        Int64(0),
        cute.runtime.make_fake_stream(),
    )


class BlockPermuteStoreBench:
    """128-thread CTA store permutation over one contiguous 2048B tile."""

    def __init__(self, *, perm_id: int, store_mode: str, cache_mode: str, num_blocks: int):
        self.perm_id = perm_id
        self.store_mode = store_mode
        self.cache_mode = cache_mode
        self.num_blocks = num_blocks

    @cute.jit
    def __call__(
        self,
        mData: cute.Tensor,
        mOut: cute.Tensor,
        iterations: Int32,
        tile_count: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(mData, mOut, iterations, tile_count).launch(
            grid=[self.num_blocks, 1, 1],
            block=[BLOCK_PERM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mData: cute.Tensor,
        mOut: cute.Tensor,
        iterations: Int32,
        tile_count: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = tidx >> 5
        lane = tidx & 31

        acc0 = Int32(tidx + 1)
        acc1 = Int32(tidx + 17)
        acc2 = Int32(tidx + 33)
        acc3 = Int32(tidx + 49)
        tile_stride_i32 = Int64(BLOCK_PERM_TILE_BYTES // 4)
        num_blocks_i64 = Int64(self.num_blocks)

        for it in cutlass.range(iterations, unroll=1):
            if const_expr(self.cache_mode == "streaming"):
                tile = (it.to(Int64) * num_blocks_i64 + bidx.to(Int64)) % tile_count
            else:
                tile = bidx.to(Int64) % tile_count

            if const_expr(self.perm_id == PERM_IDENTITY):
                slot = tidx
            elif const_expr(self.perm_id == PERM_LANE_REVERSE):
                slot = warp * WARP_SIZE + (WARP_SIZE - 1 - lane)
            elif const_expr(self.perm_id == PERM_BLOCK_REVERSE):
                slot = (BLOCK_PERM_THREADS - 1) - tidx
            elif const_expr(self.perm_id == PERM_WARP_INTERLEAVE):
                slot = lane * BLOCK_PERM_WARPS + warp
            elif const_expr(self.perm_id == PERM_SECTOR_STRIPED):
                line = lane >> 1
                pair = lane & 1
                slot = line * 8 + warp * 2 + pair
            elif const_expr(self.perm_id == PERM_TWO_LINE_STRIPED):
                half_line = (lane >> 2) * BLOCK_PERM_WARPS + warp
                local = lane & 3
                slot = (half_line >> 1) * 8 + (half_line & 1) * 4 + local
            elif const_expr(self.perm_id == PERM_SPLIT_HALVES):
                line = (warp >> 1) * 8 + (lane >> 2)
                local = lane & 3
                sector_in_half = warp & 1
                sector = sector_in_half + (local >> 1) * 2
                slot = line * 8 + sector * 2 + (local & 1)
            elif const_expr(self.perm_id == PERM_UNIT8_INTERLEAVE):
                unit_in_warp = lane >> 3
                local = lane & 7
                line = unit_in_warp * BLOCK_PERM_WARPS + warp
                slot = line * 8 + local
            elif const_expr(self.perm_id == PERM_AFFINE17):
                slot = (tidx * 17) & 127
            else:  # PERM_BIT_REVERSE
                slot = ((tidx & 1) << 6) | ((tidx & 2) << 4) | ((tidx & 4) << 2)
                slot |= tidx & 8
                slot |= (tidx & 16) >> 2
                slot |= (tidx & 32) >> 4
                slot |= (tidx & 64) >> 6

            elem = tile * tile_stride_i32 + slot.to(Int64) * Int64(VEC_BYTES // 4)
            v0 = acc0 + it + lane
            v1 = acc1 + it + lane
            v2 = acc2 + it + lane
            v3 = acc3 + it + lane
            store_global_v4_u32(mData.iterator + elem, v0, v1, v2, v3, self.store_mode)
            acc0 += Int32(1)
            acc1 += Int32(3)
            acc2 += Int32(5)
            acc3 += Int32(7)

        mOut[bidx * BLOCK_PERM_THREADS + tidx] = acc0 + acc1 + acc2 + acc3


@lru_cache(maxsize=None)
def _compile_block_permute(
    perm_id: int,
    store_mode: str,
    cache_mode: str,
    num_blocks: int,
    data_elems: int,
    out_elems: int,
):
    data = fake_tensor(Int32, (data_elems,), divisibility=32)
    out = fake_tensor(Int32, (out_elems,), divisibility=1)
    op = BlockPermuteStoreBench(
        perm_id=perm_id,
        store_mode=store_mode,
        cache_mode=cache_mode,
        num_blocks=num_blocks,
    )
    return cute.compile(op, data, out, Int32(0), Int64(0), cute.runtime.make_fake_stream())


def parse_size(text: str) -> int:
    s = text.strip().lower().replace("_", "")
    scale = 1
    if s.endswith("kib"):
        scale = 1024
        s = s[:-3]
    elif s.endswith("kb") or s.endswith("k"):
        scale = 1024
        s = s.rstrip("b")[:-1] if s.endswith("kb") else s[:-1]
    elif s.endswith("mib"):
        scale = 1024**2
        s = s[:-3]
    elif s.endswith("mb") or s.endswith("m"):
        scale = 1024**2
        s = s.rstrip("b")[:-1] if s.endswith("mb") else s[:-1]
    elif s.endswith("gib"):
        scale = 1024**3
        s = s[:-3]
    elif s.endswith("gb") or s.endswith("g"):
        scale = 1024**3
        s = s.rstrip("b")[:-1] if s.endswith("gb") else s[:-1]
    return int(float(s) * scale)


def parse_int_list(text: str) -> list[int]:
    if not text:
        return []
    return [int(x.strip(), 0) for x in text.split(",") if x.strip()]


def round_down(x: int, align: int) -> int:
    return x - (x % align)


def round_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def expand_patterns(items: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for item in items:
        key = item.strip()
        if key in PATTERN_GROUPS:
            expanded.extend(PATTERN_GROUPS[key])
        elif key in PATTERNS:
            expanded.append(key)
        else:
            valid = ", ".join([*PATTERN_GROUPS.keys(), *PATTERNS.keys()])
            raise ValueError(f"unknown pattern {key!r}; valid choices: {valid}")
    deduped: list[str] = []
    seen: set[str] = set()
    for key in expanded:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def expand_perms(items: Iterable[str]) -> list[PermSpec]:
    expanded: list[str] = []
    for item in items:
        key = item.strip()
        if key in PERM_GROUPS:
            expanded.extend(PERM_GROUPS[key])
        elif key in PERMS:
            expanded.append(key)
        else:
            valid = ", ".join([*PERM_GROUPS.keys(), *PERMS.keys()])
            raise ValueError(f"unknown permutation {key!r}; valid choices: {valid}")

    specs: list[PermSpec] = []
    seen: set[str] = set()
    for key in expanded:
        if key not in seen:
            specs.append(PERMS[key])
            seen.add(key)
    return specs


def slot_for_perm(perm_id: int, tidx: int) -> int:
    warp = tidx // WARP_SIZE
    lane = tidx % WARP_SIZE
    if perm_id == PERM_IDENTITY:
        return tidx
    if perm_id == PERM_LANE_REVERSE:
        return warp * WARP_SIZE + (WARP_SIZE - 1 - lane)
    if perm_id == PERM_BLOCK_REVERSE:
        return BLOCK_PERM_THREADS - 1 - tidx
    if perm_id == PERM_WARP_INTERLEAVE:
        return lane * BLOCK_PERM_WARPS + warp
    if perm_id == PERM_SECTOR_STRIPED:
        line = lane // 2
        pair = lane & 1
        return line * 8 + warp * 2 + pair
    if perm_id == PERM_TWO_LINE_STRIPED:
        half_line = (lane // 4) * BLOCK_PERM_WARPS + warp
        local = lane & 3
        return (half_line // 2) * 8 + (half_line & 1) * 4 + local
    if perm_id == PERM_SPLIT_HALVES:
        line = (warp // 2) * 8 + (lane // 4)
        local = lane & 3
        sector_in_half = warp & 1
        sector = sector_in_half if local < 2 else 2 + sector_in_half
        return line * 8 + sector * 2 + (local & 1)
    if perm_id == PERM_UNIT8_INTERLEAVE:
        unit_in_warp = lane // 8
        local = lane & 7
        line = unit_in_warp * BLOCK_PERM_WARPS + warp
        return line * 8 + local
    if perm_id == PERM_AFFINE17:
        return (tidx * 17) & 127
    if perm_id == PERM_BIT_REVERSE:
        return int(f"{tidx:07b}"[::-1], 2)
    raise ValueError(f"unknown permutation id {perm_id}")


def count_units(byte_addrs: Iterable[int]) -> tuple[int, int, int]:
    sectors: set[int] = set()
    half_lines: set[int] = set()
    lines: set[int] = set()
    for addr in byte_addrs:
        sectors.add(addr // SECTOR_BYTES)
        half_lines.add(addr // HALF_LINE_BYTES)
        lines.add(addr // LINE_BYTES)
    return len(sectors), len(half_lines), len(lines)


def expected_block_perm_counts(perm_id: int) -> dict[str, object]:
    slots = [slot_for_perm(perm_id, tidx) for tidx in range(BLOCK_PERM_THREADS)]
    if sorted(slots) != list(range(BLOCK_PERM_THREADS)):
        raise ValueError(f"permutation {perm_id} is not one-to-one")

    block_addrs = [slot * VEC_BYTES for slot in slots]
    block_s, _block_h, block_l = count_units(block_addrs)

    warp_s: list[int] = []
    warp_l: list[int] = []
    for warp in range(BLOCK_PERM_WARPS):
        addrs = block_addrs[warp * WARP_SIZE : (warp + 1) * WARP_SIZE]
        s, _h, l = count_units(addrs)
        warp_s.append(s)
        warp_l.append(l)

    unit8_s: list[int] = []
    unit8_l: list[int] = []
    for unit in range(BLOCK_PERM_UNITS8):
        start = unit * BLOCK_PERM_UNIT8
        addrs = block_addrs[start : start + BLOCK_PERM_UNIT8]
        s, _h, l = count_units(addrs)
        unit8_s.append(s)
        unit8_l.append(l)

    return {
        "block_unique_32B_sectors": block_s,
        "block_unique_128B_lines": block_l,
        "sum_warp_32B_sectors": sum(warp_s),
        "sum_warp_128B_lines": sum(warp_l),
        "sum_unit8_32B_sectors": sum(unit8_s),
        "sum_unit8_128B_lines": sum(unit8_l),
        "warp_32B_sectors": "/".join(map(str, warp_s)),
        "warp_128B_lines": "/".join(map(str, warp_l)),
        "unit8_32B_sectors": "/".join(map(str, unit8_s)),
        "unit8_128B_lines": "/".join(map(str, unit8_l)),
    }


def lane_addresses(
    pattern_id: int,
    *,
    offset_bytes: int,
    stride_bytes: int,
    far_stride_bytes: int,
    region_stride_bytes: int,
    page_stride_bytes: int,
) -> list[tuple[int, int]]:
    """Return (lane, byte_address_relative_to_base) for active lanes."""
    addrs: list[tuple[int, int]] = []
    for lane in range(WARP_SIZE):
        active = True
        if pattern_id == PAT_A:
            addr = 16 * lane
        elif pattern_id == PAT_B:
            pair = lane // 2
            addr = 16 * pair + (far_stride_bytes if lane & 1 else 0)
        elif pattern_id == PAT_C:
            addr = 16 * lane if lane < 16 else far_stride_bytes + 16 * (lane - 16)
        elif pattern_id == PAT_D:
            group = lane // 8
            local = lane % 8
            addr = group * far_stride_bytes + 16 * local
        elif pattern_id == PAT_E:
            pair = lane // 2
            sub = lane & 1
            addr = pair * region_stride_bytes + 16 * sub
        elif pattern_id == PAT_F:
            addr = lane * region_stride_bytes
        elif pattern_id == PAT_G:
            addr = 16 * lane + offset_bytes
        elif pattern_id == PAT_H:
            addr = lane * stride_bytes
        elif pattern_id == PAT_I_PRED:
            group = lane // 8
            local = lane % 8
            active = local < 2
            addr = group * LINE_BYTES + 16 * local
        elif pattern_id == PAT_I_DUP:
            group = lane // 8
            local = lane % 8
            addr = group * LINE_BYTES + 16 * (local & 1)
        elif pattern_id == PAT_I_FULL:
            addr = 16 * lane
        elif pattern_id == PAT_J_PACKED:
            addr = 16 * lane
        elif pattern_id == PAT_J_16LINES:
            pair = lane // 2
            sub = lane & 1
            addr = pair * LINE_BYTES + 16 * sub
        elif pattern_id == PAT_J_16PAGES:
            pair = lane // 2
            sub = lane & 1
            addr = pair * page_stride_bytes + 16 * sub
        elif pattern_id == PAT_K_SAME_HALF:
            group = lane // 4
            local = lane % 4
            addr = group * LINE_BYTES + 16 * local
        elif pattern_id == PAT_K_SPLIT_HALF:
            group = lane // 4
            local = lane % 4
            if local < 2:
                addr = group * LINE_BYTES + 16 * local
            else:
                addr = group * LINE_BYTES + HALF_LINE_BYTES + 16 * (local - 2)
        else:
            raise ValueError(f"unknown pattern id {pattern_id}")
        if active:
            addrs.append((lane, addr))
    return addrs


def expected_counts(addrs: list[tuple[int, int]]) -> tuple[int, int, int, int, int, int]:
    sectors: set[int] = set()
    half_lines: set[int] = set()
    lines: set[int] = set()
    max_end = 0
    for _lane, addr in addrs:
        if addr % 4 != 0:
            raise ValueError(f"address {addr} is not 4B-aligned")
        first_sector = addr // SECTOR_BYTES
        last_sector = (addr + VEC_BYTES - 1) // SECTOR_BYTES
        sectors.update(range(first_sector, last_sector + 1))
        first_half_line = addr // HALF_LINE_BYTES
        last_half_line = (addr + VEC_BYTES - 1) // HALF_LINE_BYTES
        half_lines.update(range(first_half_line, last_half_line + 1))
        first_line = addr // LINE_BYTES
        last_line = (addr + VEC_BYTES - 1) // LINE_BYTES
        lines.update(range(first_line, last_line + 1))
        max_end = max(max_end, addr + VEC_BYTES)
    active_lanes = len(addrs)
    useful_bytes = active_lanes * VEC_BYTES
    return active_lanes, useful_bytes, len(sectors), len(half_lines), len(lines), max_end


def pattern_num_streams(pattern_id: int) -> int:
    if pattern_id in {PAT_B, PAT_C}:
        return 2
    if pattern_id == PAT_D:
        return 4
    if pattern_id == PAT_E:
        return 16
    if pattern_id == PAT_F:
        return 32
    return 1


def pattern_slot_stride_bytes(pattern_id: int, span_bytes: int, page_stride_bytes: int) -> int:
    # For far-stream patterns, each stream owns a compact per-warp slot and the
    # stream-major layout separates streams by stream_span_bytes.  This avoids the
    # old benchmark bug where neighboring warps overlapped inside one huge span.
    if pattern_id in {PAT_B, PAT_C}:
        return 256
    if pattern_id == PAT_D:
        return 128
    if pattern_id in {PAT_E, PAT_F}:
        # Align each warp-instance slot to a full 128B line.  The per-warp E/F
        # footprints are only 32B/16B per stream, but a 32B slot stride lets
        # neighboring warps consume the other sector in the same 64B half-line,
        # hiding the per-instruction sparse-sector cost we are trying to study.
        return LINE_BYTES
    if pattern_id == PAT_J_16PAGES:
        return round_up(span_bytes, page_stride_bytes)
    return round_up(span_bytes, LINE_BYTES)


def compute_stream_layout(
    *,
    pattern_id: int,
    span_bytes: int,
    data_bytes: int,
    cache_mode: str,
    resident_bytes: int,
    total_warps: int,
    far_stride_bytes: int,
    region_stride_bytes: int,
    page_stride_bytes: int,
) -> StreamLayout:
    num_streams = pattern_num_streams(pattern_id)
    slot_stride_bytes = pattern_slot_stride_bytes(pattern_id, span_bytes, page_stride_bytes)
    usable_bytes = min(resident_bytes, data_bytes) if cache_mode == "resident" else data_bytes
    stream_span_bytes = round_down(usable_bytes // num_streams, LINE_BYTES)
    slot_count = stream_span_bytes // slot_stride_bytes

    if stream_span_bytes <= 0 or slot_count <= 0:
        raise ValueError(
            f"usable_bytes={usable_bytes} is too small for {num_streams} streams and "
            f"slot_stride={slot_stride_bytes}B"
        )
    if pattern_id in {PAT_B, PAT_C, PAT_D} and stream_span_bytes < far_stride_bytes:
        raise ValueError(
            f"{stream_span_bytes}B stream separation is smaller than requested "
            f"far stride {far_stride_bytes}B; increase --data-bytes/--resident-bytes"
        )
    if pattern_id in {PAT_E, PAT_F} and stream_span_bytes < region_stride_bytes:
        raise ValueError(
            f"{stream_span_bytes}B stream separation is smaller than requested "
            f"region stride {region_stride_bytes}B; increase --data-bytes/--resident-bytes"
        )

    note = ""
    if cache_mode == "streaming":
        min_slots = 2 * total_warps
        if slot_count < min_slots:
            needed = num_streams * slot_stride_bytes * min_slots
            raise ValueError(
                f"streaming mode has only {slot_count} slots but needs at least {min_slots} "
                f"for two non-overlapping grid waves; increase --data-bytes to >= {needed}"
            )
    elif slot_count < total_warps:
        note = (
            f"resident mode has {slot_count} slots for {total_warps} warps; "
            "same-launch reuse is intentional but bandwidth is cache/effective only"
        )

    return StreamLayout(slot_stride_bytes, stream_span_bytes, slot_count, note)


def make_aligned_i32_buffer(num_bytes: int, *, initialize: bool) -> torch.Tensor:
    if num_bytes % 4 != 0:
        raise ValueError("data bytes must be a multiple of 4")
    extra_elems = 64
    raw = torch.empty((num_bytes // 4 + extra_elems,), device="cuda", dtype=torch.int32)
    start_elem = ((-raw.data_ptr()) % LINE_BYTES) // 4
    data = raw[start_elem : start_elem + num_bytes // 4]
    if data.data_ptr() % LINE_BYTES != 0:
        raise RuntimeError(f"failed to align data pointer: {data.data_ptr():#x}")
    if initialize:
        data.fill_(1)
    return data


def time_kernel(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return min(times_ms) if repeats <= 2 else float(median(times_ms))


def run_one(
    *,
    operation: str,
    spec: PatternSpec,
    mem_mode: str,
    cache_mode: str,
    misaligned_vector: str,
    offset_bytes: int,
    stride_bytes: int,
    far_stride_bytes: int,
    region_stride_bytes: int,
    page_stride_bytes: int,
    data: torch.Tensor,
    out: torch.Tensor,
    iterations: int,
    warmup: int,
    repeats: int,
    resident_bytes: int,
    num_blocks: int,
    warps_per_block: int,
) -> dict[str, object]:
    addrs = lane_addresses(
        spec.pattern_id,
        offset_bytes=offset_bytes,
        stride_bytes=stride_bytes,
        far_stride_bytes=far_stride_bytes,
        region_stride_bytes=region_stride_bytes,
        page_stride_bytes=page_stride_bytes,
    )
    actual_mem_mode = mem_mode
    if not mem_mode.endswith("_scalar4") and any(addr % VEC_BYTES != 0 for _lane, addr in addrs):
        if misaligned_vector == "error":
            raise ValueError(
                f"{spec.label} has addresses that are not 16B-aligned; "
                "16B vector global ops fault on these. Use --misaligned-vector scalar4 "
                "or restrict offsets to multiples of 16."
            )
        actual_mem_mode = f"{mem_mode}_scalar4"
    active_lanes, useful_per_warp_inst, sectors, half_lines, lines, span_bytes = expected_counts(
        addrs
    )
    total_warps = num_blocks * warps_per_block
    layout = compute_stream_layout(
        pattern_id=spec.pattern_id,
        span_bytes=span_bytes,
        data_bytes=data.numel() * 4,
        cache_mode=cache_mode,
        resident_bytes=resident_bytes,
        total_warps=total_warps,
        far_stride_bytes=far_stride_bytes,
        region_stride_bytes=region_stride_bytes,
        page_stride_bytes=page_stride_bytes,
    )
    if layout.note:
        print(f"warning: {spec.label}: {layout.note}", file=sys.stderr)

    compiled = _compile(
        operation,
        spec.pattern_id,
        actual_mem_mode,
        cache_mode,
        offset_bytes,
        stride_bytes,
        far_stride_bytes,
        region_stride_bytes,
        page_stride_bytes,
        num_blocks,
        warps_per_block,
        data.numel(),
        out.numel(),
    )

    stream = torch.cuda.current_stream().cuda_stream
    fn = lambda: compiled(
        data,
        out,
        iterations,
        layout.slot_count,
        layout.slot_stride_i32,
        layout.stream_span_i32,
        stream,
    )
    time_ms = time_kernel(fn, warmup=warmup, repeats=repeats)
    requested_bytes = useful_per_warp_inst * iterations * num_blocks * warps_per_block
    bandwidth_gbps = requested_bytes / (time_ms / 1000.0) / 1e9
    checksum = int(out.to(torch.int64).sum().item())

    return {
        "operation": operation,
        "pattern": spec.label,
        "load_mode": actual_mem_mode,
        "cache_mode": cache_mode,
        "offset": offset_bytes
        if spec.pattern_id in {PAT_G, PAT_A, PAT_I_FULL, PAT_J_PACKED}
        else "",
        "stride": stride_bytes if spec.pattern_id == PAT_H else "",
        "active_lanes": active_lanes,
        "useful_bytes_per_warp_inst": useful_per_warp_inst,
        "requested_bytes": requested_bytes,
        "expected_32B_sectors": sectors,
        "expected_64B_halflines": half_lines if operation == "load" else "",
        "expected_128B_lines": lines,
        "slot_stride_bytes": layout.slot_stride_bytes,
        "stream_span_bytes": layout.stream_span_bytes,
        "slot_count": layout.slot_count,
        "time_ms": f"{time_ms:.6f}",
        "bandwidth_GBps": f"{bandwidth_gbps:.3f}",
        "checksum": checksum,
    }


def build_rows(args) -> list[tuple[PatternSpec, int, int]]:
    rows: list[tuple[PatternSpec, int, int]] = []
    for key in expand_patterns(args.patterns):
        spec = PATTERNS[key]
        if spec.pattern_id == PAT_G:
            for offset in parse_int_list(args.offsets):
                rows.append((spec, offset, 16))
        elif spec.pattern_id == PAT_H:
            for stride in parse_int_list(args.strides):
                rows.append((spec, 0, stride))
        else:
            rows.append((spec, 0, 16))
    return rows


def run_block_permute_one(
    *,
    spec: PermSpec,
    store_mode: str,
    cache_mode: str,
    data: torch.Tensor,
    out: torch.Tensor,
    iterations: int,
    warmup: int,
    repeats: int,
    num_blocks: int,
) -> dict[str, object]:
    tile_count = data.numel() // (BLOCK_PERM_TILE_BYTES // 4)
    if tile_count <= 0:
        raise ValueError(f"--data-bytes must be at least {BLOCK_PERM_TILE_BYTES}")
    if cache_mode == "streaming" and tile_count < 2 * num_blocks:
        needed = 2 * num_blocks * BLOCK_PERM_TILE_BYTES
        raise ValueError(
            "streaming block-permute mode needs at least two full-grid waves of tiles; "
            f"increase --data-bytes to >= {needed}"
        )

    compiled = _compile_block_permute(
        spec.perm_id,
        store_mode,
        cache_mode,
        num_blocks,
        data.numel(),
        out.numel(),
    )
    stream = torch.cuda.current_stream().cuda_stream
    fn = lambda: compiled(data, out, iterations, tile_count, stream)
    time_ms = time_kernel(fn, warmup=warmup, repeats=repeats)
    requested_bytes = BLOCK_PERM_TILE_BYTES * iterations * num_blocks
    bandwidth_gbps = requested_bytes / (time_ms / 1000.0) / 1e9
    checksum = int(out.to(torch.int64).sum().item())
    return {
        "permutation": spec.label,
        "store_mode": store_mode,
        "cache_mode": cache_mode,
        "threads": BLOCK_PERM_THREADS,
        "tile_bytes": BLOCK_PERM_TILE_BYTES,
        "requested_bytes": requested_bytes,
        **expected_block_perm_counts(spec.perm_id),
        "tile_count": tile_count,
        "time_ms": f"{time_ms:.6f}",
        "bandwidth_GBps": f"{bandwidth_gbps:.3f}",
        "checksum": checksum,
    }


def run_block_permute(args, props) -> None:
    num_blocks = (
        1
        if args.low_occupancy
        else (args.blocks if args.blocks > 0 else props.multi_processor_count * args.blocks_per_sm)
    )
    data = make_aligned_i32_buffer(args.data_bytes, initialize=False)
    out = torch.empty((num_blocks * BLOCK_PERM_THREADS,), device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    time.sleep(0.1)

    print(
        f"GPU={props.name} SMs={props.multi_processor_count} blocks={num_blocks} "
        f"threads/block={BLOCK_PERM_THREADS} iterations={args.iterations} "
        f"data={args.data_bytes}B",
        file=sys.stderr,
    )

    output_fh = args.csv.open("w", newline="") if args.csv is not None else sys.stdout
    try:
        writer = csv.DictWriter(output_fh, fieldnames=BLOCK_PERM_CSV_COLUMNS)
        writer.writeheader()
        for spec in expand_perms(args.perms):
            for store_mode in args.store_modes:
                for cache_mode in args.cache_modes:
                    row = run_block_permute_one(
                        spec=spec,
                        store_mode=store_mode,
                        cache_mode=cache_mode,
                        data=data,
                        out=out,
                        iterations=args.iterations,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        num_blocks=num_blocks,
                    )
                    writer.writerow(row)
                    output_fh.flush()
    finally:
        if args.csv is not None:
            output_fh.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CuTe-DSL global-memory coalescing microbenchmarks"
    )
    parser.add_argument(
        "--bench",
        choices=["warp", "block-permute"],
        default="warp",
        help="warp: original one-warp patterns; block-permute: 128-thread 2KB store perms",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["all"],
        help="Pattern keys/groups: all, I, J, K, A, B, C, D, E, F, G, H, I_pred, I_dup, "
        "I_full, J_packed, J_16lines, J_16pages, K_same_half, K_split_half",
    )
    parser.add_argument(
        "--perms",
        nargs="+",
        default=["all"],
        help="Block-permute keys/groups: all, identity, lane_reverse, block_reverse, "
        "warp_interleave, sector_striped, two_line_striped, split_halves, "
        "unit8_interleave, affine17, bit_reverse",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=["load"],
        choices=["load", "store"],
        help="Benchmark global loads, global stores, or both",
    )
    parser.add_argument(
        "--load-modes",
        nargs="+",
        default=["global"],
        choices=[
            "global",
            "ca",
            "cg",
            "volatile",
            "global_scalar4",
            "ca_scalar4",
            "cg_scalar4",
            "volatile_scalar4",
        ],
        help="Inline PTX load instruction variant; *_scalar4 uses four 32-bit loads",
    )
    parser.add_argument(
        "--store-modes",
        nargs="+",
        default=["global"],
        choices=[
            "global",
            "wb",
            "cg",
            "cs",
            "wt",
            "volatile",
            "global_scalar4",
            "wb_scalar4",
            "cg_scalar4",
            "cs_scalar4",
            "wt_scalar4",
            "volatile_scalar4",
        ],
        help="Inline PTX store instruction variant; *_scalar4 uses four 32-bit stores",
    )
    parser.add_argument(
        "--cache-modes",
        nargs="+",
        default=["streaming"],
        choices=["resident", "streaming"],
        help="resident reuses a small working set; streaming advances base each iteration",
    )
    parser.add_argument("--offsets", default="0,4,8,16,20,28,32,64")
    parser.add_argument("--strides", default="16,32,64,128,256")
    parser.add_argument("--iterations", type=int, default=4096, help="Timed loop iterations/kernel")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup kernel launches")
    parser.add_argument("--repeats", type=int, default=10, help="Timed kernel launches")
    parser.add_argument("--data-bytes", type=parse_size, default=DEFAULT_DATA_BYTES)
    parser.add_argument("--resident-bytes", type=parse_size, default=DEFAULT_RESIDENT_BYTES)
    parser.add_argument(
        "--misaligned-vector",
        choices=["scalar4", "error"],
        default="scalar4",
        help="What to do when a 16B vector load would be unaligned (offsets 4/8/20/28)",
    )
    parser.add_argument("--far-stride-bytes", type=parse_size, default=DEFAULT_FAR_STRIDE_BYTES)
    parser.add_argument(
        "--region-stride-bytes", type=parse_size, default=DEFAULT_REGION_STRIDE_BYTES
    )
    parser.add_argument("--page-stride-bytes", type=parse_size, default=DEFAULT_PAGE_STRIDE_BYTES)
    parser.add_argument("--blocks", type=int, default=0, help="0 means SMs * --blocks-per-sm")
    parser.add_argument("--blocks-per-sm", type=int, default=4)
    parser.add_argument("--warps-per-block", type=int, default=8)
    parser.add_argument(
        "--low-occupancy",
        action="store_true",
        help="Alias for --blocks 1 --warps-per-block 1; useful for Nsight Compute",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Write CSV here instead of stdout")
    parser.add_argument(
        "--no-init",
        action="store_true",
        help="Do not fill the input tensor; useful when profiling only the benchmark kernel",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.iterations <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("iterations/repeats must be positive and warmup non-negative")
    if args.warps_per_block <= 0:
        raise ValueError("--warps-per-block must be positive")

    props = torch.cuda.get_device_properties(0)
    if args.bench == "block-permute":
        run_block_permute(args, props)
        return

    if args.low_occupancy:
        num_blocks = 1
        warps_per_block = 1
    else:
        num_blocks = (
            args.blocks if args.blocks > 0 else props.multi_processor_count * args.blocks_per_sm
        )
        warps_per_block = args.warps_per_block
    threads = warps_per_block * WARP_SIZE
    out_elems = num_blocks * threads

    for value_name in [
        "data_bytes",
        "resident_bytes",
        "far_stride_bytes",
        "region_stride_bytes",
        "page_stride_bytes",
    ]:
        value = getattr(args, value_name)
        if value <= 0 or value % 4 != 0:
            raise ValueError(f"--{value_name.replace('_', '-')} must be a positive multiple of 4")
    for value_name in ["far_stride_bytes", "region_stride_bytes", "page_stride_bytes"]:
        value = getattr(args, value_name)
        if value % LINE_BYTES != 0:
            print(
                f"warning: --{value_name.replace('_', '-')}={value} is not 128B-aligned",
                file=sys.stderr,
            )

    rows_to_run = build_rows(args)
    max_span = 0
    for spec, offset, stride in rows_to_run:
        addrs = lane_addresses(
            spec.pattern_id,
            offset_bytes=offset,
            stride_bytes=stride,
            far_stride_bytes=args.far_stride_bytes,
            region_stride_bytes=args.region_stride_bytes,
            page_stride_bytes=args.page_stride_bytes,
        )
        *_, span = expected_counts(addrs)
        max_span = max(max_span, span)
    min_data = round_up(max_span, LINE_BYTES) + 4096
    if args.data_bytes < min_data:
        raise ValueError(f"--data-bytes must be at least {min_data} for the selected patterns")

    print(
        f"GPU={props.name} SMs={props.multi_processor_count} blocks={num_blocks} "
        f"warps/block={warps_per_block} iterations={args.iterations} data={args.data_bytes}B",
        file=sys.stderr,
    )
    print(
        f"far_stride={args.far_stride_bytes}B region_stride={args.region_stride_bytes}B "
        f"page_stride={args.page_stride_bytes}B",
        file=sys.stderr,
    )

    data = make_aligned_i32_buffer(
        args.data_bytes, initialize=not args.no_init and "load" in args.ops
    )
    out = torch.empty((out_elems,), device="cuda", dtype=torch.int32)
    # Make the allocation and optional fill visible before timing.
    torch.cuda.synchronize()
    time.sleep(0.1)

    output_fh = args.csv.open("w", newline="") if args.csv is not None else sys.stdout
    fieldnames = (
        CSV_COLUMNS
        if "load" in args.ops
        else [c for c in CSV_COLUMNS if c != "expected_64B_halflines"]
    )
    try:
        writer = csv.DictWriter(output_fh, fieldnames=fieldnames)
        writer.writeheader()
        for spec, offset, stride in rows_to_run:
            for operation in args.ops:
                modes = args.load_modes if operation == "load" else args.store_modes
                for mem_mode in modes:
                    for cache_mode in args.cache_modes:
                        row = run_one(
                            operation=operation,
                            spec=spec,
                            mem_mode=mem_mode,
                            cache_mode=cache_mode,
                            misaligned_vector=args.misaligned_vector,
                            offset_bytes=offset,
                            stride_bytes=stride,
                            far_stride_bytes=args.far_stride_bytes,
                            region_stride_bytes=args.region_stride_bytes,
                            page_stride_bytes=args.page_stride_bytes,
                            data=data,
                            out=out,
                            iterations=args.iterations,
                            warmup=args.warmup,
                            repeats=args.repeats,
                            resident_bytes=args.resident_bytes,
                            num_blocks=num_blocks,
                            warps_per_block=warps_per_block,
                        )
                        writer.writerow({key: row[key] for key in fieldnames})
                        output_fh.flush()
    finally:
        if args.csv is not None:
            output_fh.close()


if __name__ == "__main__":
    main()
