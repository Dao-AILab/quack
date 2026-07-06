#!/usr/bin/env python3
"""CuTe-DSL microbenchmarks for the SMEM / SHFL / REDUX pipe.

This is the CuTe-DSL counterpart of the standalone CUDA microbenches in this
repo.  One kernel template has compile-time switches for:

* conflict-free shared-memory uint4 reads
* conflict-free shared-memory uint4 writes
* warp SHFL.BFLY.B32
* warp REDUX.SUM.S32
* warp CREDUX.{MAX,MIN}.{S32,U32} on SM100+ via PTX redux.sync max/min
* warp CREDUX.{MAX,MIN,MAXABS,MINABS}.F32 on SM100a/f+ via PTX redux.sync f32

The point is to keep the benchmark extensible: adding a new instruction should
only require adding a small ``dsl_user_op`` wrapper, an optional per-thread state
register, and a compile-time flag/count in ``PipeBench``.

Counting convention:

* SMEM bytes are real shared-memory bytes.  A uint4 op per lane is 16 B/lane.
* SHFL bytes are useful delivered bytes: one 32-bit value per active lane, i.e.
  128 B per full-warp SHFL instruction.
* REDUX/CREDUX are primarily reported as warp-instructions/clock/SM.  ``redux inB/clk``
  is a descriptive input-byte count (one 32-bit input per lane), not necessarily
  a bandwidth-pressure model.  CUDA ptxas emits CREDUX for integer max/min
  reductions on SM100+, while SM90 emits REDUX.MAX/MIN.  F32 CREDUX requires
  an architecture-specific/family-specific SM100+ target such as ``sm_103a``.

Examples:

    # Standard suite: baselines plus a few mixed ratios.
    python microbenchmarks/gpu_pipe_microbench.py

    # One custom kernel variant.
    python microbenchmarks/gpu_pipe_microbench.py \
        --no-suite --smem-read --shuffle --shuffles-per-op 4
    python microbenchmarks/gpu_pipe_microbench.py \
        --no-suite --cred 4
    python microbenchmarks/gpu_pipe_microbench.py \
        --no-suite --cred-f32 4

    # Inspect SASS for a compiled variant via normal CuTe cache/dump tooling.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.base_dsl.arch import Arch
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

from quack.compile_utils import make_fake_tensor as fake_tensor


WARP_SIZE = 32
VEC_BYTES = 16  # uint4 shared-memory op per lane
SHFL_BYTES = 4  # one int32 result per lane
REDUX_INPUT_BYTES = 4
ALIGN_BYTES = 128
INT_REDUX_OPS = ("sum_s32", "max_s32", "min_s32", "max_u32", "min_u32")
F32_REDUX_OPS = ("max_f32", "min_f32", "maxabs_f32", "minabs_f32")
REDUX_OPS = (*INT_REDUX_OPS, *F32_REDUX_OPS)
F32_REDUX_OP_SET = frozenset(F32_REDUX_OPS)


# ---------------------------------------------------------------------------
# Device-side primitive wrappers
# ---------------------------------------------------------------------------


@dsl_user_op
def clock64(*, loc=None, ip=None) -> Int64:
    return Int64(nvvm.read_ptx_sreg_clock64(T.i64(), loc=loc, ip=ip))


@dsl_user_op
def smem_store_v4_u32(
    smem_ptr: cute.Pointer,
    v0: Int32,
    v1: Int32,
    v2: Int32,
    v3: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    smem_addr = smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            smem_addr,
            Int32(v0).ir_value(loc=loc, ip=ip),
            Int32(v1).ir_value(loc=loc, ip=ip),
            Int32(v2).ir_value(loc=loc, ip=ip),
            Int32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def smem_load_v4_u32(smem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    smem_addr = smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [smem_addr],
        "{ .reg .u32 x, y, z, w;\n\tld.volatile.shared.v4.u32 {x, y, z, w}, [$0];\n\t}",
        "r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def shfl_bfly_i32(x: Int32, mask: int, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {mask}, 0x1f, 0xffffffff;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def redux_i32(x: Int32, op: str, *, loc=None, ip=None) -> Int32:
    asm_by_op = {
        "sum_s32": "redux.sync.add.s32 $0, $1, 0xffffffff;",
        "max_s32": "redux.sync.max.s32 $0, $1, 0xffffffff;",
        "min_s32": "redux.sync.min.s32 $0, $1, 0xffffffff;",
        "max_u32": "redux.sync.max.u32 $0, $1, 0xffffffff;",
        "min_u32": "redux.sync.min.u32 $0, $1, 0xffffffff;",
    }
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            asm_by_op[op],
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def redux_f32(x: Float32, op: str, *, loc=None, ip=None) -> Float32:
    asm_by_op = {
        "max_f32": "redux.sync.max.f32 $0, $1, 0xffffffff;",
        "min_f32": "redux.sync.min.f32 $0, $1, 0xffffffff;",
        "maxabs_f32": "redux.sync.max.abs.f32 $0, $1, 0xffffffff;",
        "minabs_f32": "redux.sync.min.abs.f32 $0, $1, 0xffffffff;",
    }
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            asm_by_op[op],
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


# ---------------------------------------------------------------------------
# Config / result helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchConfig:
    name: str
    smem_read: bool = False
    smem_write: bool = False
    shuffle: bool = False
    redux: bool = False
    shuffles_per_op: int = 0
    reduxes_per_op: int = 0
    redux_op: str = "sum_s32"
    ops_per_iter: int = 16
    iterations: int = 50_000
    threads: int = 256

    @property
    def uses_smem(self) -> bool:
        return self.smem_read or self.smem_write

    @property
    def redux_is_f32(self) -> bool:
        return self.redux_op in F32_REDUX_OP_SET

    @property
    def smem_op_multiplier(self) -> int:
        return int(self.smem_read) + int(self.smem_write)

    @property
    def smem_int32(self) -> int:
        # One uint4 per lane per op: 4 int32 elements.
        return self.ops_per_iter * self.threads * 4 if self.uses_smem else 1

    @property
    def smem_footprint_bytes(self) -> int:
        return self.smem_int32 * 4 + ALIGN_BYTES if self.uses_smem else 16

    def work_per_cta(self) -> dict[str, float]:
        warps = self.threads // WARP_SIZE
        smem_bytes = (
            self.iterations * self.threads * self.ops_per_iter * VEC_BYTES * self.smem_op_multiplier
        )
        shfl_warp_inst = (
            self.iterations
            * warps
            * self.ops_per_iter
            * (self.shuffles_per_op if self.shuffle else 0)
        )
        redux_warp_inst = (
            self.iterations * warps * self.ops_per_iter * (self.reduxes_per_op if self.redux else 0)
        )
        return {
            "smem_bytes": float(smem_bytes),
            "shfl_warp_inst": float(shfl_warp_inst),
            "shfl_bytes": float(shfl_warp_inst * WARP_SIZE * SHFL_BYTES),
            "redux_warp_inst": float(redux_warp_inst),
            "redux_input_bytes": float(redux_warp_inst * WARP_SIZE * REDUX_INPUT_BYTES),
        }


@dataclass
class BenchResult:
    config: BenchConfig
    min_cycles: int
    median_cycles: int
    max_cycles: int
    smem_bclk: float
    shfl_inst_clk: float
    shfl_bclk: float
    redux_inst_clk: float
    redux_input_bclk: float
    total_useful_bclk: float


# ---------------------------------------------------------------------------
# CuTe-DSL kernel template
# ---------------------------------------------------------------------------


class PipeBench:
    def __init__(self, config: BenchConfig, launch_smem_bytes: int):
        self.config = config
        self.smem_read = config.smem_read
        self.smem_write = config.smem_write
        self.shuffle = config.shuffle
        self.redux = config.redux
        self.redux_op = config.redux_op
        self.redux_is_f32 = config.redux_is_f32
        self.shuffles_per_op = config.shuffles_per_op if config.shuffle else 0
        self.reduxes_per_op = config.reduxes_per_op if config.redux else 0
        self.ops_per_iter = config.ops_per_iter
        self.threads = config.threads
        self.smem_int32 = config.smem_int32
        self.uses_smem = config.uses_smem
        self.launch_smem_bytes = launch_smem_bytes

    @cute.jit
    def __call__(
        self, mCycles: cute.Tensor, mSink: cute.Tensor, iterations: Int32, stream: cuda.CUstream
    ):
        self.kernel(mCycles, mSink, iterations).launch(
            grid=[mCycles.shape[0], 1, 1],
            block=[self.threads, 1, 1],
            # Reserving opt-in dynamic smem keeps the launch to one CTA/SM for
            # apples-to-apples per-SM cycle measurements, even for shuffle/redux-only cases.
            smem=self.launch_smem_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mCycles: cute.Tensor, mSink: cute.Tensor, iterations: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Independent chains expose instruction throughput rather than latency.
        x0 = Int32(tidx + 0)
        x1 = Int32(tidx + 17)
        x2 = Int32(tidx + 33)
        x3 = Int32(tidx + 49)
        x4 = Int32(tidx + 65)
        x5 = Int32(tidx + 81)
        x6 = Int32(tidx + 97)
        x7 = Int32(tidx + 113)
        f0 = Float32(tidx) + Float32(0.125)
        f1 = Float32(tidx) + Float32(17.25)
        f2 = Float32(tidx) + Float32(33.375)
        f3 = Float32(tidx) + Float32(49.5)
        f4 = Float32(tidx) + Float32(65.625)
        f5 = Float32(tidx) + Float32(81.75)
        f6 = Float32(tidx) + Float32(97.875)
        f7 = Float32(tidx) + Float32(113.0)

        if const_expr(self.uses_smem):
            smem = cutlass.utils.SmemAllocator()
            s = smem.allocate_tensor(
                Int32, cute.make_layout(self.smem_int32), byte_alignment=ALIGN_BYTES
            )

            # Initialize every vector slot.  Read benchmarks rely on these stores being
            # outside the timed region.  Store benchmarks also benefit from touching the
            # same footprint before timing.
            for op in cutlass.range_constexpr(self.ops_per_iter):
                elem = (tidx + op * self.threads) * 4
                smem_store_v4_u32(
                    s.iterator + elem,
                    Int32(tidx + op + 0),
                    Int32(tidx + op + 1),
                    Int32(tidx + op + 2),
                    Int32(tidx + op + 3),
                )

        cute.arch.barrier()
        start = clock64()

        # Keep the timed loop body comparable to the CUDA microbench.  With a
        # compile-time trip count, CuTe currently software-unrolls this loop.
        for _it in cutlass.range(iterations, unroll=1):
            for op in cutlass.range_constexpr(self.ops_per_iter):
                if const_expr(self.uses_smem):
                    elem = (tidx + op * self.threads) * 4
                    if const_expr(self.smem_read):
                        smem_load_v4_u32(s.iterator + elem)
                    if const_expr(self.smem_write):
                        smem_store_v4_u32(
                            s.iterator + elem,
                            x0 + op,
                            x1 + op,
                            x2 + op,
                            x3 + op,
                        )

                if const_expr(self.shuffle):
                    for j in cutlass.range_constexpr(self.shuffles_per_op):
                        which = (op * self.shuffles_per_op + j) & 7
                        mask = ((op * self.shuffles_per_op + j) * 7 + 1) & 31
                        if const_expr(which == 0):
                            x0 = shfl_bfly_i32(x0, mask)
                        elif const_expr(which == 1):
                            x1 = shfl_bfly_i32(x1, mask)
                        elif const_expr(which == 2):
                            x2 = shfl_bfly_i32(x2, mask)
                        elif const_expr(which == 3):
                            x3 = shfl_bfly_i32(x3, mask)
                        elif const_expr(which == 4):
                            x4 = shfl_bfly_i32(x4, mask)
                        elif const_expr(which == 5):
                            x5 = shfl_bfly_i32(x5, mask)
                        elif const_expr(which == 6):
                            x6 = shfl_bfly_i32(x6, mask)
                        else:
                            x7 = shfl_bfly_i32(x7, mask)

                if const_expr(self.redux):
                    for j in cutlass.range_constexpr(self.reduxes_per_op):
                        which = (op * self.reduxes_per_op + j) & 7
                        if const_expr(self.redux_is_f32):
                            if const_expr(which == 0):
                                f0 = redux_f32(f0, self.redux_op)
                            elif const_expr(which == 1):
                                f1 = redux_f32(f1, self.redux_op)
                            elif const_expr(which == 2):
                                f2 = redux_f32(f2, self.redux_op)
                            elif const_expr(which == 3):
                                f3 = redux_f32(f3, self.redux_op)
                            elif const_expr(which == 4):
                                f4 = redux_f32(f4, self.redux_op)
                            elif const_expr(which == 5):
                                f5 = redux_f32(f5, self.redux_op)
                            elif const_expr(which == 6):
                                f6 = redux_f32(f6, self.redux_op)
                            else:
                                f7 = redux_f32(f7, self.redux_op)
                        else:
                            if const_expr(which == 0):
                                x0 = redux_i32(x0, self.redux_op)
                            elif const_expr(which == 1):
                                x1 = redux_i32(x1, self.redux_op)
                            elif const_expr(which == 2):
                                x2 = redux_i32(x2, self.redux_op)
                            elif const_expr(which == 3):
                                x3 = redux_i32(x3, self.redux_op)
                            elif const_expr(which == 4):
                                x4 = redux_i32(x4, self.redux_op)
                            elif const_expr(which == 5):
                                x5 = redux_i32(x5, self.redux_op)
                            elif const_expr(which == 6):
                                x6 = redux_i32(x6, self.redux_op)
                            else:
                                x7 = redux_i32(x7, self.redux_op)

        cute.arch.barrier()
        stop = clock64()

        if tidx == 0:
            mCycles[bidx] = stop - start
            # Keep collective-op chains live through the timed region.
            mSink[bidx] = (
                x0
                + x1
                + x2
                + x3
                + x4
                + x5
                + x6
                + x7
                + Int32(f0)
                + Int32(f1)
                + Int32(f2)
                + Int32(f3)
                + Int32(f4)
                + Int32(f5)
                + Int32(f6)
                + Int32(f7)
            )


# ---------------------------------------------------------------------------
# Host-side compile/run/report
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _compile(config: BenchConfig, num_ctas: int, launch_smem_bytes: int):
    cycles = fake_tensor(Int64, (num_ctas,), 1)
    sink = fake_tensor(Int32, (num_ctas,), 1)
    op = PipeBench(config, launch_smem_bytes)
    return cute.compile(op, cycles, sink, Int32(0), cute.runtime.make_fake_stream())


def _summarize(config: BenchConfig, cycles: torch.Tensor) -> BenchResult:
    sorted_cycles = torch.sort(cycles.cpu()).values
    min_cycles = int(sorted_cycles[0].item())
    median_cycles = int(sorted_cycles[len(sorted_cycles) // 2].item())
    max_cycles = int(sorted_cycles[-1].item())
    work = config.work_per_cta()
    denom = float(max_cycles)
    smem_bclk = work["smem_bytes"] / denom
    shfl_inst_clk = work["shfl_warp_inst"] / denom
    shfl_bclk = work["shfl_bytes"] / denom
    redux_inst_clk = work["redux_warp_inst"] / denom
    redux_input_bclk = work["redux_input_bytes"] / denom
    total_useful_bclk = (
        work["smem_bytes"] + work["shfl_bytes"] + work["redux_input_bytes"]
    ) / denom
    return BenchResult(
        config=config,
        min_cycles=min_cycles,
        median_cycles=median_cycles,
        max_cycles=max_cycles,
        smem_bclk=smem_bclk,
        shfl_inst_clk=shfl_inst_clk,
        shfl_bclk=shfl_bclk,
        redux_inst_clk=redux_inst_clk,
        redux_input_bclk=redux_input_bclk,
        total_useful_bclk=total_useful_bclk,
    )


def run_config(
    config: BenchConfig,
    *,
    num_ctas: int,
    launch_smem_bytes: int,
    warmup: int,
    rep: int,
) -> BenchResult:
    cycles = torch.empty((num_ctas,), device="cuda", dtype=torch.int64)
    sink = torch.empty((num_ctas,), device="cuda", dtype=torch.int32)
    compiled = _compile(config, num_ctas, launch_smem_bytes)

    for _ in range(warmup):
        compiled(cycles, sink, config.iterations, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    best: BenchResult | None = None
    for _ in range(rep):
        compiled(cycles, sink, config.iterations, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        result = _summarize(config, cycles)
        if best is None or result.max_cycles < best.max_cycles:
            best = result

    assert best is not None
    # Cheap sanity: force the sink copy to catch launch/runtime problems.
    _ = int(sink[0].item())
    return best


def default_suite(base: BenchConfig, include_f32_credux: bool) -> list[BenchConfig]:
    common = dict(
        ops_per_iter=base.ops_per_iter,
        iterations=base.iterations,
        threads=base.threads,
    )
    configs = [
        BenchConfig("smem_read", smem_read=True, **common),
        BenchConfig("smem_write", smem_write=True, **common),
        BenchConfig("shuffle", shuffle=True, shuffles_per_op=4, **common),
        BenchConfig("redux", redux=True, reduxes_per_op=4, redux_op="sum_s32", **common),
        BenchConfig("credux_max", redux=True, reduxes_per_op=4, redux_op="max_s32", **common),
        BenchConfig("read+shuffle_r4", smem_read=True, shuffle=True, shuffles_per_op=4, **common),
        BenchConfig("write+shuffle_r4", smem_write=True, shuffle=True, shuffles_per_op=4, **common),
        BenchConfig("read+redux_r2", smem_read=True, redux=True, reduxes_per_op=2, **common),
        BenchConfig("write+redux_r2", smem_write=True, redux=True, reduxes_per_op=2, **common),
        BenchConfig("read+redux_r4", smem_read=True, redux=True, reduxes_per_op=4, **common),
        BenchConfig("write+redux_r4", smem_write=True, redux=True, reduxes_per_op=4, **common),
        BenchConfig(
            "read+credux_r4",
            smem_read=True,
            redux=True,
            reduxes_per_op=4,
            redux_op="max_s32",
            **common,
        ),
        BenchConfig(
            "write+credux_r4",
            smem_write=True,
            redux=True,
            reduxes_per_op=4,
            redux_op="max_s32",
            **common,
        ),
    ]
    if include_f32_credux:
        configs.extend(
            [
                BenchConfig(
                    "credux_f32_max",
                    redux=True,
                    reduxes_per_op=4,
                    redux_op="max_f32",
                    **common,
                ),
                BenchConfig(
                    "read+credux_f32_r4",
                    smem_read=True,
                    redux=True,
                    reduxes_per_op=4,
                    redux_op="max_f32",
                    **common,
                ),
                BenchConfig(
                    "write+credux_f32_r4",
                    smem_write=True,
                    redux=True,
                    reduxes_per_op=4,
                    redux_op="max_f32",
                    **common,
                ),
            ]
        )
    return configs


def print_results(results: Iterable[BenchResult]) -> None:
    print(
        f"{'name':<18} {'R':>1} {'W':>1} {'SHF/op':>6} {'RED/op':>6} {'redop':<8} "
        f"{'max cyc':>12} {'smem B/clk':>12} {'shfl inst':>10} {'shfl B/clk':>12} "
        f"{'red inst':>10} {'red inB/clk':>12} {'total B/clk':>12}"
    )
    for r in results:
        c = r.config
        print(
            f"{c.name:<18} {int(c.smem_read):>1} {int(c.smem_write):>1} "
            f"{(c.shuffles_per_op if c.shuffle else 0):>6} "
            f"{(c.reduxes_per_op if c.redux else 0):>6} "
            f"{(c.redux_op if c.redux else '-'): <8} "
            f"{r.max_cycles:>12d} {r.smem_bclk:>12.2f} {r.shfl_inst_clk:>10.2f} "
            f"{r.shfl_bclk:>12.2f} {r.redux_inst_clk:>10.2f} "
            f"{r.redux_input_bclk:>12.2f} {r.total_useful_bclk:>12.2f}"
        )


def print_overlap_analysis(results: list[BenchResult]) -> None:
    """Compare mixed kernels to standalone baselines when the suite has them."""
    read_base = next(
        (
            r
            for r in results
            if r.config.smem_read
            and not r.config.smem_write
            and not r.config.shuffle
            and not r.config.redux
        ),
        None,
    )
    write_base = next(
        (
            r
            for r in results
            if r.config.smem_write
            and not r.config.smem_read
            and not r.config.shuffle
            and not r.config.redux
        ),
        None,
    )
    shfl_base = next(
        (r for r in results if r.config.shuffle and not r.config.uses_smem and not r.config.redux),
        None,
    )
    redux_bases = {
        r.config.redux_op: r
        for r in results
        if r.config.redux and not r.config.uses_smem and not r.config.shuffle
    }

    rows = []
    for r in results:
        c = r.config
        if not c.uses_smem or not (c.shuffle or c.redux):
            continue
        smem_base = read_base if c.smem_read else write_base
        if smem_base is None:
            continue
        work = c.work_per_cta()
        components = [work["smem_bytes"] / smem_base.smem_bclk]
        if c.shuffle:
            if shfl_base is None or shfl_base.shfl_inst_clk == 0.0:
                continue
            components.append(work["shfl_warp_inst"] / shfl_base.shfl_inst_clk)
        if c.redux:
            redux_base = redux_bases.get(c.redux_op)
            if redux_base is None or redux_base.redux_inst_clk == 0.0:
                continue
            components.append(work["redux_warp_inst"] / redux_base.redux_inst_clk)
        ideal_ind = max(components)
        ideal_sum = sum(components)
        rows.append((r, r.max_cycles / ideal_ind, r.max_cycles / ideal_sum))

    if not rows:
        return

    print("\nOverlap analysis from standalone baselines:")
    print(f"{'name':<18} {'obs/ind':>10} {'obs/sum':>10}")
    for r, obs_ind, obs_sum in rows:
        print(f"{r.config.name:<18} {obs_ind:>10.3f} {obs_sum:>10.3f}")
    print("obs/ind ~= 1 means perfect overlap; obs/sum ~= 1 means additive/serialized time.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-suite", action="store_true", help="run one custom config")
    parser.add_argument("--name", default="custom", help="name for --no-suite output")
    parser.add_argument("--smem-read", action="store_true", help="include SMEM uint4 reads")
    parser.add_argument("--smem-write", action="store_true", help="include SMEM uint4 writes")
    parser.add_argument("--shuffle", action="store_true", help="include SHFL.BFLY.B32 ops")
    parser.add_argument("--redux", action="store_true", help="include REDUX/CREDUX ops")
    parser.add_argument(
        "--cred",
        nargs="?",
        const=-1,
        type=int,
        default=None,
        help="alias for --redux --redux-op max_s32; optional N sets --reduxes-per-op N",
    )
    parser.add_argument(
        "--cred-f32",
        nargs="?",
        const=-1,
        type=int,
        default=None,
        help="alias for --redux --redux-op max_f32; optional N sets --reduxes-per-op N",
    )
    parser.add_argument("--shuffles-per-op", type=int, default=4)
    parser.add_argument("--reduxes-per-op", type=int, default=4)
    parser.add_argument(
        "--redux-op",
        default="sum_s32",
        choices=REDUX_OPS,
    )
    parser.add_argument("--ops-per-iter", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--num-ctas", type=int, default=0, help="default: one CTA per SM")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=3)
    parser.add_argument(
        "--reserve-smem-kib",
        type=int,
        default=0,
        help="dynamic smem reservation to force occupancy; default uses opt-in max",
    )
    return parser.parse_args()


def cute_target_supports_f32_credux() -> tuple[bool, str]:
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    if isinstance(arch, Arch):
        arch_name = arch.name
    else:
        arch_name = str(arch).split(".")[-1]
    arch_suffix = arch_name.removeprefix("sm_")
    digits = "".join(ch for ch in arch_suffix if ch.isdigit())
    major = int(digits) // 10 if digits else 0
    is_family_or_specific = arch_name.endswith(("a", "f"))
    return major >= 10 and is_family_or_specific, arch_name


def apply_redux_alias(args: argparse.Namespace, value: int | None, op: str) -> None:
    if value is None:
        return
    args.redux = True
    args.redux_op = op
    if value >= 0:
        args.reduxes_per_op = value


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.threads % WARP_SIZE != 0 or args.threads < WARP_SIZE or args.threads > 1024:
        raise ValueError("--threads must be a multiple of 32 in [32, 1024]")
    if args.ops_per_iter <= 0:
        raise ValueError("--ops-per-iter must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.shuffles_per_op < 0 or args.reduxes_per_op < 0:
        raise ValueError("per-op counts must be non-negative")
    apply_redux_alias(args, args.cred, "max_s32")
    apply_redux_alias(args, args.cred_f32, "max_f32")

    supports_f32_credux, cute_arch_name = cute_target_supports_f32_credux()
    if args.redux and args.redux_op in F32_REDUX_OP_SET and not supports_f32_credux:
        raise RuntimeError(
            f"{args.redux_op} requires an SM100a/f+ CuTe target; current target is {cute_arch_name}"
        )

    props = torch.cuda.get_device_properties(0)
    num_ctas = args.num_ctas or props.multi_processor_count
    optin_smem = min(props.shared_memory_per_block_optin, props.shared_memory_per_multiprocessor)
    launch_smem_bytes = args.reserve_smem_kib * 1024 if args.reserve_smem_kib else optin_smem
    # CuTe/CUDA require non-zero dynamic smem in some paths; align the reservation.
    launch_smem_bytes = max(16, launch_smem_bytes)

    base = BenchConfig(
        name=args.name,
        smem_read=args.smem_read,
        smem_write=args.smem_write,
        shuffle=args.shuffle,
        redux=args.redux,
        shuffles_per_op=args.shuffles_per_op,
        reduxes_per_op=args.reduxes_per_op,
        redux_op=args.redux_op,
        ops_per_iter=args.ops_per_iter,
        iterations=args.iterations,
        threads=args.threads,
    )
    configs = [base] if args.no_suite else default_suite(base, supports_f32_credux)

    max_footprint = max(c.smem_footprint_bytes for c in configs)
    if max_footprint > launch_smem_bytes:
        raise ValueError(
            f"SMEM footprint {max_footprint / 1024:.1f} KiB exceeds reserved "
            f"dynamic smem {launch_smem_bytes / 1024:.1f} KiB"
        )

    major, minor = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} sm_{major}{minor}; CuTe target={cute_arch_name}")
    print(
        f"ctas={num_ctas} threads={args.threads} ops/iter={args.ops_per_iter} "
        f"iterations={args.iterations} reserve_smem={launch_smem_bytes / 1024:.1f} KiB"
    )
    print("Counting: SHFL B/clk = 4 B/lane/result; REDUX/CREDUX inB/clk = 4 B/lane/input.")
    print("CREDUX note: integer max/min lowers to CREDUX on SM100+, REDUX on SM90.")
    print(f"CREDUX f32: {'enabled' if supports_f32_credux else 'disabled'} for this CuTe target.\n")

    results: list[BenchResult] = []
    for config in configs:
        time.sleep(0.05)
        results.append(
            run_config(
                config,
                num_ctas=num_ctas,
                launch_smem_bytes=launch_smem_bytes,
                warmup=args.warmup,
                rep=args.rep,
            )
        )

    print_results(results)
    print_overlap_analysis(results)


if __name__ == "__main__":
    main()
