# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Copyright (c) 2026, Tri Dao.
#
# Device code adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""Direct eager RMSNorm forward for ROCm gfx950 using FlyDSL."""

import functools
import math
import numbers

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp

from quack.flydsl_runtime import (
    SUPPORTED_DTYPES as _SUPPORTED_DTYPES,
)
from quack.flydsl_runtime import (
    current_raw_stream,
    dtype_spec,
    empty_placeholder,
    packed_rows,
    run_compiled,
)
from quack.rmsnorm_flydsl_config import (
    ACCESS_BITS,
    MAX_N,
    WAVE_SIZE,
    RmsNormFwdConfig,
)
from quack.rmsnorm_flydsl_config import rows_per_block as _rows_per_block

__all__ = ["rmsnorm_fwd"]

_SUPPORTED_ARCHES = frozenset({"gfx950"})
_MAX_ROWS = 2**31 - 1
# rstd's buffer descriptor carries a 32-bit num_records over fp32 elements.
_MAX_RSTD_ROWS = (2**32 - 1) // 4
# Fixed for the life of the process, and read on every validated call.
_IS_ROCM = torch.version.hip is not None


@functools.cache
def _compiled_forward(
    n: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    residual_dtype: torch.dtype,
    residual_out_dtype: torch.dtype,
    has_weight: bool,
    has_bias: bool,
    has_residual: bool,
    store_residual: bool,
    store_rstd: bool,
    per_head: bool,
    num_heads: int,
    apply_weight_offset: bool,
):
    """Memoize one feature-specialized launcher per set of build parameters.

    The arguments are the cache key, so a parameter added here without a
    matching argument at the call site is a TypeError rather than a silently
    shared kernel.
    """
    return _build_rmsnorm_module(
        n,
        input_dtype,
        output_dtype,
        weight_torch_dtype=weight_dtype,
        bias_torch_dtype=bias_dtype,
        residual_torch_dtype=residual_dtype,
        residual_out_torch_dtype=residual_out_dtype,
        has_weight=has_weight,
        has_bias=has_bias,
        has_residual=has_residual,
        store_residual=store_residual,
        store_rstd=store_rstd,
        per_head=per_head,
        num_heads=num_heads,
        apply_weight_offset=apply_weight_offset,
    )


def _elements_per_buffer_copy(vecsize: int, elem_bits: int) -> int:
    """Elements handled by each at-most-128-bit copy of a logical vector."""
    return min(vecsize, ACCESS_BITS // elem_bits)


def _row_descriptor_bytes(elem_bits: int, n: int, valid):
    """Return the row-scoped buffer bound, or zero for an invalid grid row."""
    row_bytes = n * (elem_bits // 8)
    if valid is None:
        return row_bytes
    return valid.select(fx.Int32(row_bytes), fx.Int32(0))


def _row_buffer(tensor, row, elem_bits: int, n: int, valid=None):
    """Create a row-scoped descriptor so padded row pitches stay valid."""
    return fx.rocdl.make_buffer_tensor(
        fx.slice(tensor, (row, None)),
        num_records_bytes=_row_descriptor_bytes(elem_bits, n, valid),
    )


def _row_head_buffer(tensor, row, head, elem_bits: int, n: int, valid=None):
    return fx.rocdl.make_buffer_tensor(
        fx.slice(tensor, (row, head, None)),
        num_records_bytes=_row_descriptor_bytes(elem_bits, n, valid),
    )


def _make_reduction_storage(red_slots: int):
    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


# FlyDSL has no stable wrapper for these two ROCDL operations. They keep the
# intra-wave reduction out of LDS; only cross-wave rows use shared memory.
def _dpp_shuffle_xor(value, offset: int):
    raw = value.ir_value()
    result_type = raw.type
    if offset == 8:
        peer = fx.rocdl.update_dpp(result_type, raw, raw, 0x118, 0xF, 0xC, False)
        peer = fx.rocdl.update_dpp(result_type, peer, raw, 0x108, 0xF, 0x3, False)
    elif offset == 4:
        peer = fx.rocdl.update_dpp(result_type, raw, raw, 0x114, 0xF, 0xA, False)
        peer = fx.rocdl.update_dpp(result_type, peer, raw, 0x104, 0xF, 0x5, False)
    elif offset == 2:
        peer = fx.rocdl.update_dpp(result_type, raw, raw, 0x4E, 0xF, 0xF, False)
    elif offset == 1:
        peer = fx.rocdl.update_dpp(result_type, raw, raw, 0xB1, 0xF, 0xF, False)
    else:
        raise ValueError(f"unsupported DPP XOR offset: {offset}")
    return fx.Float32(peer)


def _ds_swizzle_xor(value, offset: int):
    bits = value.bitcast(fx.Uint32)
    peer = fx.rocdl.ds_swizzle(
        bits.ir_value().type,
        bits.ir_value(),
        fx.Int32((offset << 10) | 0x1F).ir_value(),
    )
    return fx.Uint32(peer).bitcast(fx.Float32)


def _shuffle_reduce_add(value, lanes: int, shuffle_width, fast_math):
    result = value
    for shift_exp in range_constexpr(int(math.log2(lanes))):
        offset = lanes // (2 << shift_exp)
        if lanes in (32, 64) and offset <= 8:
            peer = _dpp_shuffle_xor(result, offset)
        elif lanes in (32, 64) and offset == 16:
            peer = _ds_swizzle_xor(result, offset)
        else:
            peer = fx.gpu.shuffle_xor(result, offset, shuffle_width)
        with fx.arith.fastmath(fast_math):
            result = result + peer
    return result


def _load_vec(copy_atom, vec_width, elem_dtype, divided_tensor, index):
    register = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy(copy_atom, fx.slice(divided_tensor, (None, index)), register)
    return fx.memref_load_vec(register)


def _load_dtype_vec(
    copy_atom,
    elem_dtype,
    dtype_width,
    divided_tensor,
    index,
    vecsize,
):
    """Load one activation-width span, splitting wider operand storage."""
    elements_per_copy = _elements_per_buffer_copy(vecsize, dtype_width)
    copy_count = vecsize // elements_per_copy
    if const_expr(copy_count <= 1):
        return _load_vec(
            copy_atom,
            vecsize,
            elem_dtype,
            divided_tensor,
            index,
        ).to(fx.Float32)
    elements = []
    for part in range(copy_count):
        chunk = _load_vec(
            copy_atom,
            elements_per_copy,
            elem_dtype,
            divided_tensor,
            index * copy_count + part,
        )
        elements.extend(chunk[lane] for lane in range(elements_per_copy))
    return fx.Vector.from_elements(elements, fx.Float32)


def _store_vec(copy_atom, vec_width, elem_dtype, value, divided_tensor, index):
    register = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(value, register)
    fx.copy(copy_atom, register, fx.slice(divided_tensor, (None, index)))


def _store_dtype_vec(
    copy_atom,
    elem_dtype,
    dtype_width,
    value,
    divided_tensor,
    index,
    vecsize,
):
    elements_per_copy = _elements_per_buffer_copy(vecsize, dtype_width)
    copy_count = vecsize // elements_per_copy
    if const_expr(copy_count <= 1):
        _store_vec(copy_atom, vecsize, elem_dtype, value, divided_tensor, index)
        return
    for part in range(copy_count):
        lanes = list(range(part * elements_per_copy, (part + 1) * elements_per_copy))
        _store_vec(
            copy_atom,
            elements_per_copy,
            elem_dtype,
            value.shuffle(value, lanes),
            divided_tensor,
            index * copy_count + part,
        )


def _to_store_dtype(is_float32: bool, elem_dtype, value):
    if const_expr(is_float32):
        return value
    # gfx950 provides the packed fp32-to-bf16 conversion used by vector stores.
    return value.to(elem_dtype)


def _build_rmsnorm_module(
    n: int,
    input_torch_dtype: torch.dtype,
    output_torch_dtype: torch.dtype,
    *,
    weight_torch_dtype: torch.dtype,
    bias_torch_dtype: torch.dtype,
    residual_torch_dtype: torch.dtype,
    residual_out_torch_dtype: torch.dtype,
    has_weight: bool,
    has_bias: bool,
    has_residual: bool,
    store_residual: bool,
    store_rstd: bool,
    per_head: bool,
    num_heads: int,
    apply_weight_offset: bool,
):
    """Build one feature-specialized analytical forward launcher.

    Public ``torch.dtype`` values are resolved here, before FlyDSL traces the
    nested kernel; device code captures only FlyDSL scalar types, bit widths,
    and booleans.
    """
    input_dtype, input_bits = dtype_spec(input_torch_dtype)
    output_dtype, output_bits = dtype_spec(output_torch_dtype)
    weight_dtype, weight_bits = dtype_spec(weight_torch_dtype)
    bias_dtype, bias_bits = dtype_spec(bias_torch_dtype)
    residual_dtype, residual_bits = dtype_spec(residual_torch_dtype)
    residual_out_dtype, residual_out_bits = dtype_spec(residual_out_torch_dtype)
    output_is_float32 = output_torch_dtype == torch.float32
    residual_out_is_float32 = residual_out_torch_dtype == torch.float32

    config = RmsNormFwdConfig.for_forward(n, input_bits)
    threads_per_row = config.num_threads
    rows_per_block = _rows_per_block(config)
    block_threads = rows_per_block * threads_per_row
    vecsize = config.vecsize
    num_vecs = config.num_vecs
    last_tile = config.num_tiles - 1
    reload_from_gmem = config.reload_from_gmem
    wide_full_tiles = num_vecs // threads_per_row
    wide_tail_vecs = num_vecs % threads_per_row

    reduce_lanes = min(threads_per_row, WAVE_SIZE)
    red_slots = max(1, threads_per_row // WAVE_SIZE)
    shared_storage = _make_reduction_storage(red_slots)

    input_copy_width = _elements_per_buffer_copy(vecsize, input_bits)
    output_copy_width = _elements_per_buffer_copy(vecsize, output_bits)
    weight_copy_width = _elements_per_buffer_copy(vecsize, weight_bits)
    bias_copy_width = _elements_per_buffer_copy(vecsize, bias_bits)
    residual_copy_width = _elements_per_buffer_copy(vecsize, residual_bits)
    residual_out_copy_width = _elements_per_buffer_copy(vecsize, residual_out_bits)

    @flyc.kernel(**({} if block_threads <= 256 else {"known_block_size": [block_threads, 1, 1]}))
    def rmsnorm_kernel(
        input_tensor: fx.Tensor,
        weight_tensor: fx.Tensor,
        bias_tensor: fx.Tensor,
        residual_tensor: fx.Tensor,
        output_tensor: fx.Tensor,
        residual_out_tensor: fx.Tensor,
        rstd_tensor: fx.Tensor,
        num_programs: fx.Int32,
        eps: fx.Float32,
        weight_offset: fx.Float32,
    ):
        tid = fx.thread_idx.x
        if const_expr(rows_per_block > 1):
            lane = tid % threads_per_row
            program = fx.block_idx.x * fx.Int32(rows_per_block) + tid // threads_per_row
            in_grid = program < num_programs
        else:
            lane = tid
            program = fx.block_idx.x
            in_grid = None
        row = program // fx.Int32(num_heads) if per_head else program
        head = program % fx.Int32(num_heads) if per_head else fx.Int32(0)

        fast_math = arith.FastMathFlags.fast

        storage = fx.SharedAllocator().allocate(shared_storage).peek()
        reduction = storage.s_red.view(fx.make_layout(red_slots, 1))

        def group_reduce_add(value):
            return _shuffle_reduce_add(
                value,
                reduce_lanes,
                fx.Int32(reduce_lanes),
                fast_math,
            )

        def row_reduce_add(value):
            if const_expr(red_slots == 1):
                return group_reduce_add(value)
            wave_lane = tid % WAVE_SIZE
            wave = tid // WAVE_SIZE
            reduced = group_reduce_add(value)
            if wave_lane == 0:
                fx.memref_store(reduced, reduction, wave)
            gpu.barrier()
            if wave == 0:
                in_range = wave_lane < red_slots
                safe_lane = in_range.select(wave_lane, 0)
                partial = in_range.select(
                    fx.memref_load(reduction, safe_lane),
                    fx.Float32(0.0),
                )
                partial = group_reduce_add(partial)
                if wave_lane == 0:
                    fx.memref_store(partial, reduction, 0)
            gpu.barrier()
            return fx.memref_load(reduction, 0)

        def row_div(tensor, elem_bits, copy_width):
            buffer = (
                _row_head_buffer(tensor, row, head, elem_bits, n, in_grid)
                if per_head
                else _row_buffer(tensor, row, elem_bits, n, in_grid)
            )
            return fx.logical_divide(buffer, fx.make_layout(copy_width, 1))

        def parameter_div(tensor, elem_bits, copy_width):
            buffer = (
                _row_buffer(tensor, head, elem_bits, n)
                if per_head
                else fx.rocdl.make_buffer_tensor(tensor)
            )
            return fx.logical_divide(buffer, fx.make_layout(copy_width, 1))

        input_div = row_div(input_tensor, input_bits, input_copy_width)
        output_div = row_div(output_tensor, output_bits, output_copy_width)
        # Copy widths above guarantee each FlyDSL MUBUF transaction is <= 128 bits.
        input_copy = fx.make_copy_atom(
            fx.rocdl.BufferCopy(input_copy_width * input_bits, 0), input_bits
        )
        output_copy = fx.make_copy_atom(
            fx.rocdl.BufferCopy(output_copy_width * output_bits, 0), output_bits
        )
        if const_expr(has_residual):
            residual_div = row_div(residual_tensor, residual_bits, residual_copy_width)
            residual_copy = fx.make_copy_atom(
                fx.rocdl.BufferCopy(residual_copy_width * residual_bits, 0),
                residual_bits,
            )
        if const_expr(store_residual):
            residual_out_div = row_div(
                residual_out_tensor,
                residual_out_bits,
                residual_out_copy_width,
            )
            residual_out_copy = fx.make_copy_atom(
                fx.rocdl.BufferCopy(
                    residual_out_copy_width * residual_out_bits,
                    0,
                ),
                residual_out_bits,
            )
        if const_expr(has_weight):
            weight_div = parameter_div(weight_tensor, weight_bits, weight_copy_width)
            weight_copy = fx.make_copy_atom(
                fx.rocdl.BufferCopy(weight_copy_width * weight_bits, 0),
                weight_bits,
            )
        if const_expr(has_bias):
            bias_div = parameter_div(bias_tensor, bias_bits, bias_copy_width)
            bias_copy = fx.make_copy_atom(
                fx.rocdl.BufferCopy(bias_copy_width * bias_bits, 0),
                bias_bits,
            )
        if const_expr(store_rstd):
            rstd_buffer = fx.rocdl.make_buffer_tensor(
                rstd_tensor,
                num_records_bytes=num_programs * fx.Int32(4),
            )
            rstd_div = fx.logical_divide(rstd_buffer, fx.make_layout(1, 1))

        thread_sumsq = fx.Float32(0.0)
        row_values = []
        native_row_values = []
        if const_expr(reload_from_gmem):
            # Runtime loops keep code size and VGPR use bounded for wide rows.
            for tile_i in range(wide_full_tiles):
                index = lane + tile_i * threads_per_row
                value = _load_dtype_vec(
                    input_copy,
                    input_dtype,
                    input_bits,
                    input_div,
                    index,
                    vecsize,
                )
                if const_expr(has_residual):
                    value = value + _load_dtype_vec(
                        residual_copy,
                        residual_dtype,
                        residual_bits,
                        residual_div,
                        index,
                        vecsize,
                    )
                if const_expr(store_residual):
                    _store_dtype_vec(
                        residual_out_copy,
                        residual_out_dtype,
                        residual_out_bits,
                        _to_store_dtype(residual_out_is_float32, residual_out_dtype, value),
                        residual_out_div,
                        index,
                        vecsize,
                    )
                contribution = (value * value).reduce(ReductionOp.ADD, fastmath=fast_math)
                thread_sumsq = thread_sumsq + contribution
            if const_expr(wide_tail_vecs > 0):
                index = lane + wide_full_tiles * threads_per_row
                in_row = lane < wide_tail_vecs
                safe_index = in_row.select(index, 0)
                value = _load_dtype_vec(
                    input_copy,
                    input_dtype,
                    input_bits,
                    input_div,
                    safe_index,
                    vecsize,
                )
                if const_expr(has_residual):
                    value = value + _load_dtype_vec(
                        residual_copy,
                        residual_dtype,
                        residual_bits,
                        residual_div,
                        safe_index,
                        vecsize,
                    )
                if const_expr(store_residual):  # noqa: SIM102 - compile-time guard
                    if in_row:
                        _store_dtype_vec(
                            residual_out_copy,
                            residual_out_dtype,
                            residual_out_bits,
                            _to_store_dtype(residual_out_is_float32, residual_out_dtype, value),
                            residual_out_div,
                            index,
                            vecsize,
                        )
                contribution = (value * value).reduce(ReductionOp.ADD, fastmath=fast_math)
                thread_sumsq = thread_sumsq + in_row.select(
                    contribution,
                    fx.Float32(0.0),
                )
        else:
            for tile_i in range_constexpr(config.num_tiles):
                partial = config.needs_predicate and tile_i == last_tile
                index = lane + tile_i * threads_per_row
                safe_index = index
                if const_expr(partial):
                    in_row = index < num_vecs
                    safe_index = in_row.select(index, 0)
                native_value = _load_vec(
                    input_copy,
                    vecsize,
                    input_dtype,
                    input_div,
                    safe_index,
                )
                value = native_value.to(fx.Float32)
                if const_expr(has_residual):
                    value = value + _load_dtype_vec(
                        residual_copy,
                        residual_dtype,
                        residual_bits,
                        residual_div,
                        safe_index,
                        vecsize,
                    )
                if const_expr(store_residual):
                    stored = _to_store_dtype(
                        residual_out_is_float32,
                        residual_out_dtype,
                        value,
                    )
                    if const_expr(partial):
                        if in_row:
                            _store_dtype_vec(
                                residual_out_copy,
                                residual_out_dtype,
                                residual_out_bits,
                                stored,
                                residual_out_div,
                                index,
                                vecsize,
                            )
                    else:
                        _store_dtype_vec(
                            residual_out_copy,
                            residual_out_dtype,
                            residual_out_bits,
                            stored,
                            residual_out_div,
                            index,
                            vecsize,
                        )
                if const_expr(has_residual):
                    row_values.append(value)
                else:
                    native_row_values.append(native_value)
                contribution = (value * value).reduce(
                    ReductionOp.ADD,
                    fastmath=fast_math,
                )
                if const_expr(partial):
                    contribution = in_row.select(contribution, fx.Float32(0.0))
                thread_sumsq = thread_sumsq + contribution

        sum_sq = row_reduce_add(thread_sumsq)
        rrms = fmath.rsqrt(sum_sq / float(n) + eps, fastmath=fast_math)
        if const_expr(store_rstd):  # noqa: SIM102 - compile-time guard
            if lane == 0:
                rstd_div[program] = rrms

        def write_normalized(value, load_index, store_index, guard=None):
            """Scale one vector by rrms, fold in weight and bias, and store it.

            ``load_index`` is clamped inside the row for a ragged tile while
            ``store_index`` stays unclamped, because the store is predicated by
            ``guard`` instead. A ``guard`` of None means the tile is full.
            """
            result = value * rrms
            if const_expr(has_weight):
                weights = _load_dtype_vec(
                    weight_copy,
                    weight_dtype,
                    weight_bits,
                    weight_div,
                    load_index,
                    vecsize,
                )
                if const_expr(apply_weight_offset):
                    weights = weights + weight_offset
                result = result * weights
            if const_expr(has_bias):
                result = result + _load_dtype_vec(
                    bias_copy,
                    bias_dtype,
                    bias_bits,
                    bias_div,
                    load_index,
                    vecsize,
                )
            stored = _to_store_dtype(output_is_float32, output_dtype, result)

            def emit():
                _store_dtype_vec(
                    output_copy,
                    output_dtype,
                    output_bits,
                    stored,
                    output_div,
                    store_index,
                    vecsize,
                )

            if const_expr(guard is None):
                emit()
            else:
                if guard:
                    emit()

        if const_expr(reload_from_gmem):
            for tile_i in range(wide_full_tiles):
                index = lane + tile_i * threads_per_row
                value = _load_dtype_vec(
                    input_copy,
                    input_dtype,
                    input_bits,
                    input_div,
                    index,
                    vecsize,
                )
                if const_expr(has_residual):
                    value = value + _load_dtype_vec(
                        residual_copy,
                        residual_dtype,
                        residual_bits,
                        residual_div,
                        index,
                        vecsize,
                    )
                write_normalized(value, index, index)
            if const_expr(wide_tail_vecs > 0):
                index = lane + wide_full_tiles * threads_per_row
                in_row = lane < wide_tail_vecs
                safe_index = in_row.select(index, 0)
                value = _load_dtype_vec(
                    input_copy,
                    input_dtype,
                    input_bits,
                    input_div,
                    safe_index,
                    vecsize,
                )
                if const_expr(has_residual):
                    value = value + _load_dtype_vec(
                        residual_copy,
                        residual_dtype,
                        residual_bits,
                        residual_div,
                        safe_index,
                        vecsize,
                    )
                write_normalized(value, safe_index, index, in_row)
        else:
            for tile_i in range_constexpr(config.num_tiles):
                partial = config.needs_predicate and tile_i == last_tile
                index = lane + tile_i * threads_per_row
                safe_index = index
                in_row = None
                if const_expr(partial):
                    in_row = index < num_vecs
                    safe_index = in_row.select(index, 0)
                if const_expr(has_residual):
                    value = row_values[tile_i]
                else:
                    value = native_row_values[tile_i].to(fx.Float32)
                write_normalized(value, safe_index, index, in_row)

    @flyc.jit
    def launch_rmsnorm(
        input_tensor: fx.Tensor,
        weight_tensor: fx.Tensor,
        bias_tensor: fx.Tensor,
        residual_tensor: fx.Tensor,
        output_tensor: fx.Tensor,
        residual_out_tensor: fx.Tensor,
        rstd_tensor: fx.Tensor,
        m: fx.Int32,
        eps: fx.Float32,
        weight_offset: fx.Float32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008 - FlyDSL traced ABI
    ):
        num_programs = m * fx.Int32(num_heads)
        rmsnorm_kernel(
            input_tensor,
            weight_tensor,
            bias_tensor,
            residual_tensor,
            output_tensor,
            residual_out_tensor,
            rstd_tensor,
            num_programs,
            eps,
            weight_offset,
        ).launch(
            grid=(
                (num_programs - fx.Int32(1)) // fx.Int32(rows_per_block) + fx.Int32(1),
                1,
                1,
            ),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


def _launch_rmsnorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    residual_out: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    weight_offset: float,
    *,
    has_weight: bool,
    has_bias: bool,
    has_residual: bool,
    store_residual: bool,
    store_rstd: bool,
    per_head: bool,
    num_heads: int,
) -> None:
    m, n = x.shape[0], x.shape[-1]
    # Positional, not keyword: matching fifteen keywords through lru_cache costs
    # ~0.4us per launch on gfx950, against a ~20us host path at small M.
    exe = _compiled_forward(
        n,
        x.dtype,
        out.dtype,
        weight.dtype,
        bias.dtype,
        residual.dtype,
        residual_out.dtype,
        has_weight,
        has_bias,
        has_residual,
        store_residual,
        store_rstd,
        per_head,
        num_heads,
        weight_offset != 0.0,
    )
    args = (
        x,
        weight,
        bias,
        residual,
        out,
        residual_out,
        rstd,
        m,
        eps,
        weight_offset,
        current_raw_stream(x.device),
    )
    run_compiled(exe, x.device, args, supported=_SUPPORTED_ARCHES, kernel="RMSNorm")


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    residual_dtype: torch.dtype | None,
    eps: float,
    store_rstd: bool,
    weight_offset: float,
) -> tuple[int, int, int, bool, float, float]:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    optional = (("weight", weight), ("bias", bias), ("residual", residual))
    for name, tensor in optional:
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor).__name__}")

    parameter_ranks = {tensor.ndim for tensor in (weight, bias) if tensor is not None}
    if not parameter_ranks.issubset({1, 2}):
        raise ValueError("weight and bias must be 1-D or 2-D")
    if len(parameter_ranks) > 1:
        raise ValueError("weight and bias must use the same rank")
    per_head = parameter_ranks == {2}
    if per_head:
        if x.ndim < 2:
            raise ValueError("per-head RMSNorm requires an input with at least two dimensions")
        num_heads, n = x.shape[-2:]
        if num_heads < 1:
            raise ValueError("per-head RMSNorm requires at least one head")
        parameter_shape = (num_heads, n)
    else:
        num_heads, n = 1, x.shape[-1]
        parameter_shape = (n,)

    if not 1 <= n <= MAX_N:
        raise ValueError(f"x normalized dimension must be between 1 and {MAX_N}, got {n}")
    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor is not None and tuple(tensor.shape) != parameter_shape:
            raise ValueError(f"{name} shape must be {parameter_shape}, got {tuple(tensor.shape)}")
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape must match x, got {tuple(residual.shape)}/{tuple(x.shape)}"
        )

    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x dtype must be float16, bfloat16, or float32, got {x.dtype}")
    alignment = ACCESS_BITS // (x.element_size() * 8)
    if n % alignment:
        raise ValueError(f"x normalized dimension must be a multiple of {alignment}, got {n}")
    for name, tensor in optional:
        if tensor is not None and tensor.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"{name} dtype must be float16, bfloat16, or float32, got {tensor.dtype}"
            )
    for name, dtype in (("out_dtype", out_dtype), ("residual_dtype", residual_dtype)):
        if dtype is not None and dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must be float16, bfloat16, or float32, got {dtype}")

    for name, tensor in (("x", x), *optional):
        if tensor is not None and tensor.layout != torch.strided:
            raise ValueError(f"{name} must use torch.strided layout, got {tensor.layout}")
    device = x.device
    if not _IS_ROCM or device.type != "cuda":
        raise ValueError(f"x must be on a ROCm device, got {device}")
    for name, tensor in optional:
        if tensor is not None and tensor.device != device:
            raise ValueError(
                f"x and {name} must be on the same device, got {device}/{tensor.device}"
            )

    # float is the only type worth a fast path; numbers.Real is an ABC, so its
    # isinstance goes through __instancecheck__ and costs far more than the
    # concrete check that answers the common case.
    if type(eps) is not float:
        if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
            raise TypeError(f"eps must be a real number, got {type(eps).__name__}")
        eps = float(eps)
    if not 0.0 < eps < math.inf:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    if store_rstd is not True and store_rstd is not False:
        raise TypeError(f"store_rstd must be a bool, got {type(store_rstd).__name__}")
    if type(weight_offset) is not float:
        if isinstance(weight_offset, bool) or not isinstance(weight_offset, numbers.Real):
            raise TypeError(
                f"weight_offset must be a real number, got {type(weight_offset).__name__}"
            )
        weight_offset = float(weight_offset)
    if not -math.inf < weight_offset < math.inf:
        raise ValueError(f"weight_offset must be finite, got {weight_offset}")
    if weight is None and weight_offset != 0.0:
        raise ValueError("weight_offset requires an explicit weight")

    m = x.numel() // (num_heads * n)
    normalized_rows = m * num_heads
    if normalized_rows > _MAX_ROWS:
        raise ValueError(
            f"x has {normalized_rows} normalized rows, but the kernel addresses at most {_MAX_ROWS}"
        )
    if store_rstd and normalized_rows > _MAX_RSTD_ROWS:
        raise ValueError(
            f"rstd has {normalized_rows} rows, but its buffer descriptor addresses at most "
            f"{_MAX_RSTD_ROWS}"
        )
    return m, n, num_heads, per_head, eps, weight_offset


def _output_dtypes(x, residual, out_dtype, residual_dtype):
    """Resolve the two output dtypes and whether the residual sum is stored."""
    output_dtype = x.dtype if out_dtype is None else out_dtype
    residual_out_dtype = (
        residual_dtype
        if residual_dtype is not None
        else (residual.dtype if residual is not None else x.dtype)
    )
    store_residual = residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    )
    return output_dtype, residual_out_dtype, store_residual


def _absent(x, dtype):
    """Stand-in for an output this call does not produce.

    A custom op has fixed arity and its fake has to predict shapes exactly, so
    "no rstd" cannot be None here. The caller drops these; they never escape.
    """
    return torch.empty(0, device=x.device, dtype=dtype)


def _rmsnorm_fwd_core(
    x, weight, bias, residual, out_dtype, residual_dtype, eps, store_rstd, weight_offset
):
    """Shared body. Absent outputs come back as None, not as a sentinel tensor:
    materializing two empty CUDA tensors costs 2.1us, which is an eighth of this
    kernel's whole host path, and only the op boundary actually needs them."""
    m, n, num_heads, per_head, eps, weight_offset = _validate_inputs(
        x, weight, bias, residual, out_dtype, residual_dtype, eps, store_rstd, weight_offset
    )
    output_dtype, residual_out_dtype, store_residual = _output_dtypes(
        x, residual, out_dtype, residual_dtype
    )

    if m == 0:
        return (
            torch.empty(x.shape, device=x.device, dtype=output_dtype),
            torch.empty(x.shape, device=x.device, dtype=residual_out_dtype)
            if store_residual
            else None,
            torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32) if store_rstd else None,
        )

    last_shape = (num_heads, n) if per_head else (n,)
    x_flat = packed_rows(x.reshape(-1, *last_shape))
    weight_arg = packed_rows(weight) if weight is not None else empty_placeholder(x.device, x.dtype)
    bias_arg = packed_rows(bias) if bias is not None else empty_placeholder(x.device, x.dtype)
    residual_arg = (
        packed_rows(residual.reshape(-1, *last_shape))
        if residual is not None
        else empty_placeholder(x.device, x.dtype)
    )

    out_flat = torch.empty(x_flat.shape, device=x.device, dtype=output_dtype)
    residual_out_flat = (
        torch.empty(x_flat.shape, device=x.device, dtype=residual_out_dtype)
        if store_residual
        else empty_placeholder(x.device, residual_out_dtype)
    )
    rstd_flat = (
        torch.empty(m * num_heads, device=x.device, dtype=torch.float32)
        if store_rstd
        else empty_placeholder(x.device, torch.float32)
    )

    _launch_rmsnorm_fwd(
        x_flat,
        weight_arg,
        bias_arg,
        residual_arg,
        out_flat,
        residual_out_flat,
        rstd_flat,
        eps,
        weight_offset,
        has_weight=weight is not None,
        has_bias=bias is not None,
        has_residual=residual is not None,
        store_residual=store_residual,
        store_rstd=store_rstd,
        per_head=per_head,
        num_heads=num_heads,
    )

    return (
        out_flat.reshape(x.shape),
        residual_out_flat.reshape(x.shape) if store_residual else None,
        rstd_flat.reshape(x.shape[:-1]) if store_rstd else None,
    )


def _rmsnorm_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    residual_dtype: torch.dtype | None,
    eps: float,
    store_rstd: bool,
    weight_offset: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Functional, not mutating: inductor skips cudagraphs for a region whose op
    writes into buffers it did not allocate, and letting the surrounding region
    be captured is the whole reason to register this."""
    out, residual_out, rstd = _rmsnorm_fwd_core(
        x, weight, bias, residual, out_dtype, residual_dtype, eps, store_rstd, weight_offset
    )
    _, residual_out_dtype, _ = _output_dtypes(x, residual, out_dtype, residual_dtype)
    return (
        out,
        residual_out if residual_out is not None else _absent(x, residual_out_dtype),
        rstd if rstd is not None else _absent(x, torch.float32),
    )


_rmsnorm_fwd_op = torch.library.custom_op(
    "quack::flydsl_rmsnorm_fwd",
    _rmsnorm_fwd_impl,
    mutates_args=(),
    device_types="cuda",
)


@_rmsnorm_fwd_op.register_fake
def _(x, weight, bias, residual, out_dtype, residual_dtype, eps, store_rstd, weight_offset):
    # Shapes only: running the body here would pay a FlyDSL compile at trace
    # time, and would reject shape/dtype combinations the kernel means to.
    output_dtype, residual_out_dtype, store_residual = _output_dtypes(
        x, residual, out_dtype, residual_dtype
    )
    return (
        torch.empty(x.shape, device=x.device, dtype=output_dtype),
        torch.empty(x.shape, device=x.device, dtype=residual_out_dtype)
        if store_residual
        else _absent(x, residual_out_dtype),
        torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
        if store_rstd
        else _absent(x, torch.float32),
    )


def rmsnorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
    weight_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run eager RMSNorm over the last dimension and return CuTe-compatible outputs.

    Only routes through the registered op while Dynamo is watching: the
    torch.library boundary buys an opaque graph node, and in eager there is
    nobody to give it to. The two aliasing decisions the CuTe-compatible
    contract asks for -- residual_out is x when nothing was accumulated, rstd is
    None unless requested -- are made here, because an op may return neither one
    of its inputs nor a None.
    """
    if torch.compiler.is_compiling():
        out, residual_out, rstd = _rmsnorm_fwd_op(
            x, weight, bias, residual, out_dtype, residual_dtype, eps, store_rstd, weight_offset
        )
        store_residual = residual is not None or (
            residual_dtype is not None and residual_dtype != x.dtype
        )
        return out, (residual_out if store_residual else x), (rstd if store_rstd else None)

    out, residual_out, rstd = _rmsnorm_fwd_core(
        x, weight, bias, residual, out_dtype, residual_dtype, eps, store_rstd, weight_offset
    )
    return out, (residual_out if residual_out is not None else x), rstd
