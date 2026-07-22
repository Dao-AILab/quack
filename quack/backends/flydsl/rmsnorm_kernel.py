# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Copyright (c) 2026 Dao-AILab
"""Quack-owned FlyDSL RMSNorm forward kernel and compiled cache."""

from __future__ import annotations

import math
import threading

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath

from quack.backends.flydsl.rmsnorm_common import (
    WARP_SIZE,
    dtype_to_elem_type,
    load_scalar,
    make_reduction_storage,
    store_scalar,
    torch_dtype_to_str,
)
from quack.backends.flydsl.rmsnorm_config import (
    RMSNormFwdConfig,
    cache_target_identity,
)

_FWD_CACHE: dict[tuple, flyc.CompiledFunction] = {}
_FWD_CACHE_LOCK = threading.Lock()


def build_rmsnorm_fwd_module(
    n: int,
    input_dtype_str: str,
    weight_dtype_str: str,
    config: RMSNormFwdConfig,
):
    """Build a scalar-tail-safe RMSNorm forward launcher.

    ``eps`` and the row count are runtime values; hidden size, dtypes, and
    launch geometry are structural specializations.
    """

    if weight_dtype_str != "f32":
        raise ValueError("the FlyDSL RMSNorm forward kernel requires FP32 weight")
    block_threads = config.block_threads
    red_slots = max(1, math.ceil(block_threads / WARP_SIZE))
    elem_bits = 32 if input_dtype_str == "f32" else 16
    SharedStorage = make_reduction_storage(red_slots)

    @flyc.kernel
    def rmsnorm_fwd_kernel(
        Input: fx.Tensor,
        Weight: fx.Tensor,
        Output: fx.Tensor,
        Rstd: fx.Tensor,
        Eps: fx.Float32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        input_dtype = dtype_to_elem_type(input_dtype_str)
        fastmath = arith.FastMathFlags.fast

        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        reduction = shared.values.view(fx.make_layout(red_slots, 1))

        def wave_reduce_add(value):
            result = value
            for shift_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                offset = WARP_SIZE // (2 << shift_exp)
                result = result.addf(
                    result.shuffle_xor(offset, WARP_SIZE),
                    fastmath=fastmath,
                )
            return result

        def block_reduce_add(value):
            if const_expr(red_slots == 1):
                return wave_reduce_add(value)
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            wave_sum = wave_reduce_add(value)
            if lane == 0:
                fx.memref_store(wave_sum, reduction, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < red_slots
                safe_lane = in_range.select(lane, 0)
                partial = fx.memref_load(reduction, safe_lane)
                total = wave_reduce_add(in_range.select(partial, fx.Float32(0.0)))
                if lane == 0:
                    fx.memref_store(total, reduction, 0)
            gpu.barrier()
            return fx.memref_load(reduction, 0)

        input_buffer = fx.rocdl.make_buffer_tensor(Input)
        weight_buffer = fx.rocdl.make_buffer_tensor(Weight)
        output_buffer = fx.rocdl.make_buffer_tensor(Output)
        rstd_buffer = fx.rocdl.make_buffer_tensor(Rstd)

        input_row = fx.slice(input_buffer, (row, None))
        output_row = fx.slice(output_buffer, (row, None))
        input_div = fx.logical_divide(input_row, fx.make_layout(1, 1))
        weight_div = fx.logical_divide(weight_buffer, fx.make_layout(1, 1))
        output_div = fx.logical_divide(output_row, fx.make_layout(1, 1))
        rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))

        input_copy = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if elem_bits == 32 else fx.rocdl.BufferCopy16b(),
            elem_bits,
        )
        f32_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        thread_sum = fx.Float32(0.0)
        for base in range_constexpr(0, n, block_threads):
            index = tid + base
            valid = index < n
            safe_index = valid.select(index, 0)
            x_elem = load_scalar(input_copy, input_dtype, input_div, safe_index)
            x = x_elem if input_dtype_str == "f32" else x_elem.to(fx.Float32)
            thread_sum = thread_sum + valid.select(x * x, fx.Float32(0.0))

        row_sum = block_reduce_add(thread_sum)
        rstd = fmath.rsqrt(row_sum / float(n) + Eps, fastmath=fastmath)
        if tid == 0:
            store_scalar(f32_copy, fx.Float32, rstd_div, row, rstd)

        for base in range_constexpr(0, n, block_threads):
            index = tid + base
            if index < n:
                x_elem = load_scalar(input_copy, input_dtype, input_div, index)
                x = x_elem if input_dtype_str == "f32" else x_elem.to(fx.Float32)
                weight = load_scalar(f32_copy, fx.Float32, weight_div, index)
                output = x * rstd * weight
                output_elem = output if input_dtype_str == "f32" else output.to(input_dtype)
                store_scalar(input_copy, input_dtype, output_div, index, output_elem)

    @flyc.jit
    def launch_rmsnorm_fwd(
        Input: fx.Tensor,
        Weight: fx.Tensor,
        Output: fx.Tensor,
        Rstd: fx.Tensor,
        M: fx.Int32,
        Eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        rmsnorm_fwd_kernel(Input, Weight, Output, Rstd, Eps).launch(
            grid=(M, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_fwd


def run_rmsnorm_fwd(x, weight, out, rstd, eps: float, target, config) -> None:
    """Compile/cache/launch on the input tensor's device and current stream."""

    m, n = x.shape
    input_dtype_str = torch_dtype_to_str(x.dtype)
    weight_dtype_str = torch_dtype_to_str(weight.dtype)
    with torch.cuda.device(target.device_index):
        stream = torch.cuda.current_stream(target.device_index)
        key = (
            *cache_target_identity(target),
            n,
            input_dtype_str,
            weight_dtype_str,
            "forward",
            config.block_threads,
        )
        compiled = _FWD_CACHE.get(key)
        if compiled is not None:
            compiled(x, weight, out, rstd, m, eps, stream)
            return
        with _FWD_CACHE_LOCK:
            compiled = _FWD_CACHE.get(key)
            if compiled is None:
                launcher = build_rmsnorm_fwd_module(
                    n,
                    input_dtype_str,
                    weight_dtype_str,
                    config,
                )
                # flyc.compile compiles and performs the first launch.
                compiled = flyc.compile(launcher, x, weight, out, rstd, m, eps, stream)
                _FWD_CACHE[key] = compiled
                return
        compiled(x, weight, out, rstd, m, eps, stream)


__all__ = ["build_rmsnorm_fwd_module", "run_rmsnorm_fwd"]
