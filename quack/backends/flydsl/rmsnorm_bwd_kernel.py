# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Copyright (c) 2026 Dao-AILab
"""Quack-owned FlyDSL RMSNorm backward kernels.

Small row counts use direct FP32 atomic dweight accumulation. Large row
counts use a persistent first stage plus a deterministic workspace reduction.
"""

from __future__ import annotations

import math
import threading

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, const_expr, gpu, range_constexpr

from quack.backends.flydsl.rmsnorm_common import (
    WARP_SIZE,
    atomic_add_f32,
    dtype_to_elem_type,
    load_scalar,
    make_reduction_storage,
    store_scalar,
    torch_dtype_to_str,
)
from quack.backends.flydsl.rmsnorm_config import (
    ATOMIC,
    TWO_STAGE,
    RMSNormBwdConfig,
    cache_target_identity,
)

_BWD_CACHE: dict[tuple, flyc.CompiledFunction] = {}
_BWD_CACHE_LOCK = threading.Lock()


def _validate_builder_dtypes(weight_dtype_str: str) -> None:
    if weight_dtype_str != "f32":
        raise ValueError("the FlyDSL RMSNorm backward kernel requires FP32 weight")


def build_rmsnorm_bwd_atomic_module(
    n: int,
    input_dtype_str: str,
    weight_dtype_str: str,
    config: RMSNormBwdConfig,
):
    """Build one-block-per-row backward with FP32 dweight atomics."""

    _validate_builder_dtypes(weight_dtype_str)
    block_threads = config.block_threads
    red_slots = max(1, math.ceil(block_threads / WARP_SIZE))
    elem_bits = 32 if input_dtype_str == "f32" else 16
    SharedStorage = make_reduction_storage(red_slots)

    @flyc.kernel
    def rmsnorm_bwd_atomic_kernel(
        Input: fx.Tensor,
        Weight: fx.Tensor,
        DOutput: fx.Tensor,
        Rstd: fx.Tensor,
        DInput: fx.Tensor,
        DWeight: fx.Tensor,
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
        doutput_buffer = fx.rocdl.make_buffer_tensor(DOutput)
        rstd_buffer = fx.rocdl.make_buffer_tensor(Rstd)
        dinput_buffer = fx.rocdl.make_buffer_tensor(DInput)
        dweight_div = fx.logical_divide(DWeight, fx.make_layout(1, 1))

        input_div = fx.logical_divide(
            fx.slice(input_buffer, (row, None)),
            fx.make_layout(1, 1),
        )
        doutput_div = fx.logical_divide(
            fx.slice(doutput_buffer, (row, None)),
            fx.make_layout(1, 1),
        )
        dinput_div = fx.logical_divide(
            fx.slice(dinput_buffer, (row, None)),
            fx.make_layout(1, 1),
        )
        weight_div = fx.logical_divide(weight_buffer, fx.make_layout(1, 1))
        rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))

        input_copy = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if elem_bits == 32 else fx.rocdl.BufferCopy16b(),
            elem_bits,
        )
        f32_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        f32_atomic_add = fx.make_copy_atom(
            fx.UniversalAtomicAdd(
                fx.Float32,
                syncscope=fx.rocdl.SyncScope.Agent,
            ),
            fx.Float32,
        )
        rstd = load_scalar(f32_copy, fx.Float32, rstd_div, row)

        thread_sum = fx.Float32(0.0)
        for base in range_constexpr(0, n, block_threads):
            index = tid + base
            valid = index < n
            safe_index = valid.select(index, 0)
            x_elem = load_scalar(input_copy, input_dtype, input_div, safe_index)
            dy_elem = load_scalar(input_copy, input_dtype, doutput_div, safe_index)
            x = x_elem if input_dtype_str == "f32" else x_elem.to(fx.Float32)
            dy = dy_elem if input_dtype_str == "f32" else dy_elem.to(fx.Float32)
            weight = load_scalar(f32_copy, fx.Float32, weight_div, safe_index)
            x_hat = x * rstd
            thread_sum = thread_sum + valid.select(x_hat * dy * weight, fx.Float32(0.0))

        coefficient = block_reduce_add(thread_sum) / float(n)
        for base in range_constexpr(0, n, block_threads):
            index = tid + base
            if index < n:
                x_elem = load_scalar(input_copy, input_dtype, input_div, index)
                dy_elem = load_scalar(input_copy, input_dtype, doutput_div, index)
                x = x_elem if input_dtype_str == "f32" else x_elem.to(fx.Float32)
                dy = dy_elem if input_dtype_str == "f32" else dy_elem.to(fx.Float32)
                weight = load_scalar(f32_copy, fx.Float32, weight_div, index)
                x_hat = x * rstd
                dx = (dy * weight - x_hat * coefficient) * rstd
                dx_elem = dx if input_dtype_str == "f32" else dx.to(input_dtype)
                store_scalar(input_copy, input_dtype, dinput_div, index, dx_elem)
                atomic_add_f32(
                    f32_atomic_add,
                    dweight_div,
                    index,
                    dy * x_hat,
                )

    @flyc.jit
    def launch_rmsnorm_bwd_atomic(
        Input: fx.Tensor,
        Weight: fx.Tensor,
        DOutput: fx.Tensor,
        Rstd: fx.Tensor,
        DInput: fx.Tensor,
        DWeight: fx.Tensor,
        M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        rmsnorm_bwd_atomic_kernel(
            Input,
            Weight,
            DOutput,
            Rstd,
            DInput,
            DWeight,
        ).launch(
            grid=(M, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_bwd_atomic


def build_rmsnorm_bwd_two_stage_module(
    n: int,
    input_dtype_str: str,
    weight_dtype_str: str,
    config: RMSNormBwdConfig,
):
    """Build persistent dx/dweight-partial and dweight-finalizer kernels."""

    _validate_builder_dtypes(weight_dtype_str)
    if config.num_programs <= 0:
        raise ValueError("two-stage backward requires a positive program count")
    block_threads = config.block_threads
    num_programs = config.num_programs
    values_per_thread = math.ceil(n / block_threads)
    red_slots = max(1, math.ceil(block_threads / WARP_SIZE))
    elem_bits = 32 if input_dtype_str == "f32" else 16
    SharedStorage = make_reduction_storage(red_slots)

    @flyc.kernel
    def rmsnorm_bwd_partial_kernel(
        Input: fx.Tensor,
        Weight: fx.Tensor,
        DOutput: fx.Tensor,
        Rstd: fx.Tensor,
        DInput: fx.Tensor,
        DWeightPartial: fx.Tensor,
        M: fx.Int32,
    ):
        program = fx.block_idx.x
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
        doutput_buffer = fx.rocdl.make_buffer_tensor(DOutput)
        rstd_buffer = fx.rocdl.make_buffer_tensor(Rstd)
        dinput_buffer = fx.rocdl.make_buffer_tensor(DInput)
        partial_buffer = fx.rocdl.make_buffer_tensor(DWeightPartial)
        weight_div = fx.logical_divide(weight_buffer, fx.make_layout(1, 1))
        rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))
        partial_div = fx.logical_divide(partial_buffer, fx.make_layout(1, 1))

        input_copy = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if elem_bits == 32 else fx.rocdl.BufferCopy16b(),
            elem_bits,
        )
        f32_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        initial_partial = fx.Vector.filled(values_per_thread, 0.0, fx.Float32)

        for row, state in range(
            fx.Int32(program),
            M,
            fx.Int32(num_programs),
            init=[initial_partial],
        ):
            partial_acc = state[0]
            input_div = fx.logical_divide(
                fx.slice(input_buffer, (row, None)),
                fx.make_layout(1, 1),
            )
            doutput_div = fx.logical_divide(
                fx.slice(doutput_buffer, (row, None)),
                fx.make_layout(1, 1),
            )
            dinput_div = fx.logical_divide(
                fx.slice(dinput_buffer, (row, None)),
                fx.make_layout(1, 1),
            )
            rstd = load_scalar(f32_copy, fx.Float32, rstd_div, row)

            thread_sum = fx.Float32(0.0)
            for tile in range_constexpr(values_per_thread):
                index = tid + tile * block_threads
                valid = index < n
                safe_index = valid.select(index, 0)
                x_elem = load_scalar(input_copy, input_dtype, input_div, safe_index)
                dy_elem = load_scalar(input_copy, input_dtype, doutput_div, safe_index)
                x = x_elem if input_dtype_str == "f32" else x_elem.to(fx.Float32)
                dy = dy_elem if input_dtype_str == "f32" else dy_elem.to(fx.Float32)
                weight = load_scalar(f32_copy, fx.Float32, weight_div, safe_index)
                x_hat = x * rstd
                thread_sum = thread_sum + valid.select(
                    x_hat * dy * weight,
                    fx.Float32(0.0),
                )

            coefficient = block_reduce_add(thread_sum) / float(n)
            row_dweight = []
            for tile in range_constexpr(values_per_thread):
                index = tid + tile * block_threads
                valid = index < n
                safe_index = valid.select(index, 0)
                x_elem = load_scalar(input_copy, input_dtype, input_div, safe_index)
                dy_elem = load_scalar(input_copy, input_dtype, doutput_div, safe_index)
                x = x_elem if input_dtype_str == "f32" else x_elem.to(fx.Float32)
                dy = dy_elem if input_dtype_str == "f32" else dy_elem.to(fx.Float32)
                weight = load_scalar(f32_copy, fx.Float32, weight_div, safe_index)
                x_hat = x * rstd
                dx = (dy * weight - x_hat * coefficient) * rstd
                if index < n:
                    dx_elem = dx if input_dtype_str == "f32" else dx.to(input_dtype)
                    store_scalar(input_copy, input_dtype, dinput_div, index, dx_elem)
                row_dweight.append(valid.select(dy * x_hat, fx.Float32(0.0)))

            updated_partial = partial_acc + fx.Vector.from_elements(
                row_dweight,
                fx.Float32,
            )
            gpu.barrier()
            results = yield [updated_partial]

        final_partial = results
        for tile in range_constexpr(values_per_thread):
            index = tid + tile * block_threads
            if index < n:
                partial_index = program * n + index
                store_scalar(
                    f32_copy,
                    fx.Float32,
                    partial_div,
                    partial_index,
                    final_partial[tile],
                )

    @flyc.kernel
    def rmsnorm_bwd_reduce_kernel(
        DWeightPartial: fx.Tensor,
        DWeight: fx.Tensor,
    ):
        block = fx.block_idx.x
        tid = fx.thread_idx.x
        index = fx.Int32(block * block_threads + tid)
        partial_buffer = fx.rocdl.make_buffer_tensor(DWeightPartial)
        dweight_buffer = fx.rocdl.make_buffer_tensor(DWeight)
        partial_div = fx.logical_divide(partial_buffer, fx.make_layout(1, 1))
        dweight_div = fx.logical_divide(dweight_buffer, fx.make_layout(1, 1))
        f32_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        valid = index < n
        safe_index = valid.select(index, 0)
        for partial_row, state in range(
            fx.Int32(0),
            fx.Int32(num_programs),
            fx.Int32(1),
            init=[fx.Float32(0.0)],
        ):
            partial_index = fx.Int32(partial_row) * fx.Int32(n) + safe_index
            value = load_scalar(
                f32_copy,
                fx.Float32,
                partial_div,
                partial_index,
            )
            results = yield [state[0] + value]
        if index < n:
            store_scalar(
                f32_copy,
                fx.Float32,
                dweight_div,
                index,
                results,
            )

    reduce_grid = math.ceil(n / block_threads)

    @flyc.jit
    def launch_rmsnorm_bwd_two_stage(
        Input: fx.Tensor,
        Weight: fx.Tensor,
        DOutput: fx.Tensor,
        Rstd: fx.Tensor,
        DInput: fx.Tensor,
        DWeight: fx.Tensor,
        DWeightPartial: fx.Tensor,
        M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        rmsnorm_bwd_partial_kernel(
            Input,
            Weight,
            DOutput,
            Rstd,
            DInput,
            DWeightPartial,
            M,
        ).launch(
            grid=(num_programs, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )
        rmsnorm_bwd_reduce_kernel(DWeightPartial, DWeight).launch(
            grid=(reduce_grid, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_bwd_two_stage


def run_rmsnorm_bwd(
    x,
    weight,
    doutput,
    rstd,
    dinput,
    dweight,
    workspace,
    target,
    config,
) -> None:
    """Compile/cache/launch the selected backward path on the current stream."""

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
            "backward",
            config.path,
            config.block_threads,
            config.num_programs,
        )
        compiled = _BWD_CACHE.get(key)
        if config.path == ATOMIC:
            args = (x, weight, doutput, rstd, dinput, dweight, m, stream)
            builder = build_rmsnorm_bwd_atomic_module
        elif config.path == TWO_STAGE:
            required_workspace = config.num_programs * n
            if workspace.numel() < required_workspace:
                raise RuntimeError(
                    "FlyDSL RMSNorm backward workspace is too small: "
                    f"need {required_workspace} FP32 elements, got {workspace.numel()}"
                )
            args = (
                x,
                weight,
                doutput,
                rstd,
                dinput,
                dweight,
                workspace,
                m,
                stream,
            )
            builder = build_rmsnorm_bwd_two_stage_module
        else:
            raise ValueError(f"unknown FlyDSL RMSNorm backward path {config.path!r}")

        if compiled is not None:
            compiled(*args)
            return
        with _BWD_CACHE_LOCK:
            compiled = _BWD_CACHE.get(key)
            if compiled is None:
                launcher = builder(
                    n,
                    input_dtype_str,
                    weight_dtype_str,
                    config,
                )
                # flyc.compile compiles and performs the first launch.
                compiled = flyc.compile(launcher, *args)
                _BWD_CACHE[key] = compiled
                return
        compiled(*args)


__all__ = [
    "build_rmsnorm_bwd_atomic_module",
    "build_rmsnorm_bwd_two_stage_module",
    "run_rmsnorm_bwd",
]
