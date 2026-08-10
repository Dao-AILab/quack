# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Copyright (c) 2026, Tri Dao.
#
# Device code adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""Direct eager RMSNorm forward for ROCm gfx950 using FlyDSL."""

import math
import numbers
import os
import threading

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch

from quack.rmsnorm_flydsl_config import (
    ACCESS_BITS,
    MAX_N,
    WAVE_SIZE,
    RmsNormFwdConfig,
)
from quack.rmsnorm_flydsl_config import rows_per_block as _rows_per_block

__all__ = ["rmsnorm_fwd"]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_ARCHES = frozenset({"gfx950"})
_MAX_ROWS = 2**31 - 1
_MAX_RSTD_ROWS = (2**32 - 1) // 4
_COMPILE_ENV_VARS = (
    "FLYDSL_COMPILE_BACKEND",
    "ARCH",
    "FLYDSL_GPU_ARCH",
    "HSA_OVERRIDE_GFX_VERSION",
)
_DTYPE_SPECS = {
    torch.float16: (fx.Float16, 16),
    torch.bfloat16: (fx.BFloat16, 16),
    torch.float32: (fx.Float32, 32),
}

_BUILD_LOCK = threading.RLock()
_FWD_CACHE: dict[tuple, flyc.CompiledFunction] = {}
_DEVICE_ARCH_CACHE: dict[int, str] = {}
_EMPTY_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _dtype_spec(dtype: torch.dtype):
    """Translate a public torch dtype to its FlyDSL scalar type and bit width."""
    try:
        return _DTYPE_SPECS[dtype]
    except KeyError:
        raise TypeError(f"unsupported dtype: {dtype}") from None


# Generic ROCm/FlyDSL transaction helpers. They stay local in this first kernel
# so the PR is self-contained; a later kernel can move them into shared code
# without changing their contracts.
def _buffer_copy_atom(access_bits: int, elem_bits: int):
    """Build one legal MUBUF copy atom.

    ``access_bits`` is the hardware transaction width while ``elem_bits`` tells
    FlyDSL how that transaction is represented in registers.
    """
    if access_bits not in (8, 16, 32, 64, 128):
        raise ValueError(f"no buffer copy for a {access_bits}-bit access")
    return fx.make_copy_atom(fx.rocdl.BufferCopy(access_bits, 0), elem_bits)


def _split_vector_for_dtype(vecsize: int, dtype_width: int) -> tuple[int, int]:
    """Split one logical activation vector into 128-bit-or-smaller accesses.

    Optional operands can have a wider dtype than the activation, so the same
    element span may require multiple MUBUF transactions. Returns
    ``(transaction_count, elements_per_transaction)``.
    """
    accesses = max(1, (vecsize * dtype_width) // ACCESS_BITS)
    return accesses, vecsize // accesses


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
    fx.copy_atom_call(copy_atom, fx.slice(divided_tensor, (None, index)), register)
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
    accesses, per_access = _split_vector_for_dtype(vecsize, dtype_width)
    if const_expr(accesses <= 1):
        return _load_vec(
            copy_atom,
            vecsize,
            elem_dtype,
            divided_tensor,
            index,
        ).to(fx.Float32)
    elements = []
    for part in range(accesses):
        chunk = _load_vec(
            copy_atom,
            per_access,
            elem_dtype,
            divided_tensor,
            index * accesses + part,
        )
        elements.extend(chunk[lane] for lane in range(per_access))
    return fx.Vector.from_elements(elements, fx.Float32)


def _store_vec(copy_atom, vec_width, elem_dtype, value, divided_tensor, index):
    register = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(value, register)
    fx.copy_atom_call(copy_atom, register, fx.slice(divided_tensor, (None, index)))


def _store_dtype_vec(
    copy_atom,
    elem_dtype,
    dtype_width,
    value,
    divided_tensor,
    index,
    vecsize,
):
    accesses, per_access = _split_vector_for_dtype(vecsize, dtype_width)
    if const_expr(accesses <= 1):
        _store_vec(copy_atom, vecsize, elem_dtype, value, divided_tensor, index)
        return
    for part in range(accesses):
        lanes = list(range(part * per_access, (part + 1) * per_access))
        _store_vec(
            copy_atom,
            per_access,
            elem_dtype,
            value.shuffle(value, lanes),
            divided_tensor,
            index * accesses + part,
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
    input_dtype, input_bits = _dtype_spec(input_torch_dtype)
    output_dtype, output_bits = _dtype_spec(output_torch_dtype)
    weight_dtype, weight_bits = _dtype_spec(weight_torch_dtype)
    bias_dtype, bias_bits = _dtype_spec(bias_torch_dtype)
    residual_dtype, residual_bits = _dtype_spec(residual_torch_dtype)
    residual_out_dtype, residual_out_bits = _dtype_spec(residual_out_torch_dtype)
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

    _, input_per_access = _split_vector_for_dtype(vecsize, input_bits)
    _, output_per_access = _split_vector_for_dtype(vecsize, output_bits)
    _, weight_per_access = _split_vector_for_dtype(vecsize, weight_bits)
    _, bias_per_access = _split_vector_for_dtype(vecsize, bias_bits)
    _, residual_per_access = _split_vector_for_dtype(vecsize, residual_bits)
    _, residual_out_per_access = _split_vector_for_dtype(vecsize, residual_out_bits)

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

        def row_div(tensor, elem_bits, per_access):
            buffer = (
                _row_head_buffer(tensor, row, head, elem_bits, n, in_grid)
                if per_head
                else _row_buffer(tensor, row, elem_bits, n, in_grid)
            )
            return fx.logical_divide(buffer, fx.make_layout(per_access, 1))

        def parameter_div(tensor, elem_bits, per_access):
            buffer = (
                _row_buffer(tensor, head, elem_bits, n)
                if per_head
                else fx.rocdl.make_buffer_tensor(tensor)
            )
            return fx.logical_divide(buffer, fx.make_layout(per_access, 1))

        input_div = row_div(input_tensor, input_bits, input_per_access)
        output_div = row_div(output_tensor, output_bits, output_per_access)
        input_copy = _buffer_copy_atom(input_per_access * input_bits, input_bits)
        output_copy = _buffer_copy_atom(output_per_access * output_bits, output_bits)
        if const_expr(has_residual):
            residual_div = row_div(residual_tensor, residual_bits, residual_per_access)
            residual_copy = _buffer_copy_atom(residual_per_access * residual_bits, residual_bits)
        if const_expr(store_residual):
            residual_out_div = row_div(
                residual_out_tensor,
                residual_out_bits,
                residual_out_per_access,
            )
            residual_out_copy = _buffer_copy_atom(
                residual_out_per_access * residual_out_bits,
                residual_out_bits,
            )
        if const_expr(has_weight):
            weight_div = parameter_div(weight_tensor, weight_bits, weight_per_access)
            weight_copy = _buffer_copy_atom(weight_per_access * weight_bits, weight_bits)
        if const_expr(has_bias):
            bias_div = parameter_div(bias_tensor, bias_bits, bias_per_access)
            bias_copy = _buffer_copy_atom(bias_per_access * bias_bits, bias_bits)
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
                result = value * rrms
                if const_expr(has_weight):
                    weights = _load_dtype_vec(
                        weight_copy,
                        weight_dtype,
                        weight_bits,
                        weight_div,
                        index,
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
                        index,
                        vecsize,
                    )
                _store_dtype_vec(
                    output_copy,
                    output_dtype,
                    output_bits,
                    _to_store_dtype(output_is_float32, output_dtype, result),
                    output_div,
                    index,
                    vecsize,
                )
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
                result = value * rrms
                if const_expr(has_weight):
                    weights = _load_dtype_vec(
                        weight_copy,
                        weight_dtype,
                        weight_bits,
                        weight_div,
                        safe_index,
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
                        safe_index,
                        vecsize,
                    )
                if in_row:
                    _store_dtype_vec(
                        output_copy,
                        output_dtype,
                        output_bits,
                        _to_store_dtype(output_is_float32, output_dtype, result),
                        output_div,
                        index,
                        vecsize,
                    )
        else:
            for tile_i in range_constexpr(config.num_tiles):
                partial = config.needs_predicate and tile_i == last_tile
                index = lane + tile_i * threads_per_row
                safe_index = index
                if const_expr(partial):
                    in_row = index < num_vecs
                    safe_index = in_row.select(index, 0)
                if const_expr(has_residual):
                    value = row_values[tile_i]
                else:
                    value = native_row_values[tile_i].to(fx.Float32)
                result = value * rrms
                if const_expr(has_weight):
                    weights = _load_dtype_vec(
                        weight_copy,
                        weight_dtype,
                        weight_bits,
                        weight_div,
                        safe_index,
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
                        safe_index,
                        vecsize,
                    )
                output_value = _to_store_dtype(output_is_float32, output_dtype, result)
                if const_expr(partial):
                    if in_row:
                        _store_dtype_vec(
                            output_copy,
                            output_dtype,
                            output_bits,
                            output_value,
                            output_div,
                            index,
                            vecsize,
                        )
                else:
                    _store_dtype_vec(
                        output_copy,
                        output_dtype,
                        output_bits,
                        output_value,
                        output_div,
                        index,
                        vecsize,
                    )

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


def _normalize_arch(arch: str) -> str:
    arch = str(arch).split(":", 1)[0]
    if arch.startswith("gfx"):
        return arch
    parts = arch.split(".")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return f"gfx{parts[0]}{parts[1]}{parts[2]}"
    return arch


def _validate_arch(device: torch.device) -> str:
    index = device.index if device.index is not None else torch.cuda.current_device()
    actual = _DEVICE_ARCH_CACHE.get(index)
    if actual is None:
        actual = _normalize_arch(torch.cuda.get_device_properties(index).gcnArchName)
        if actual not in _SUPPORTED_ARCHES:
            supported = ", ".join(sorted(_SUPPORTED_ARCHES))
            raise ValueError(f"FlyDSL RMSNorm supports {supported}; cuda:{index} is {actual}")
        _DEVICE_ARCH_CACHE[index] = actual

    target = flyc.get_backend().target
    compile_arch = _normalize_arch(target.arch)
    if target.backend != "rocm":
        raise RuntimeError(
            f"FlyDSL RMSNorm requires FlyDSL's ROCm backend, but it targets {target.backend!r}"
        )
    if compile_arch != actual:
        raise ValueError(
            f"cuda:{index} is {actual}, but FlyDSL compiles for {compile_arch}; "
            "set ARCH and FLYDSL_GPU_ARCH to the device architecture"
        )
    runtime_arch = _normalize_arch(get_rocm_arch())
    if runtime_arch != actual:
        raise ValueError(
            f"cuda:{index} is {actual}, but FlyDSL runtime helpers use {runtime_arch}; "
            "set FLYDSL_GPU_ARCH or HSA_OVERRIDE_GFX_VERSION to the device architecture"
        )
    return actual


def _compile_context(device: torch.device) -> tuple:
    index = device.index if device.index is not None else torch.cuda.current_device()
    return (
        index,
        *(os.environ.get(name, "") for name in _COMPILE_ENV_VARS),
    )


def _compile_forward(key: tuple, device: torch.device, args: tuple):
    """Build once and retain FlyDSL's public fast callable for this specialization."""
    with _BUILD_LOCK:
        compiled = _FWD_CACHE.get(key)
        if compiled is not None:
            return compiled, False
        _validate_arch(device)
        launcher = _build_rmsnorm_module(
            key[1],
            key[2],
            key[3],
            weight_torch_dtype=key[4],
            bias_torch_dtype=key[5],
            residual_torch_dtype=key[6],
            residual_out_torch_dtype=key[7],
            has_weight=key[8],
            has_bias=key[9],
            has_residual=key[10],
            store_residual=key[11],
            store_rstd=key[12],
            per_head=key[13],
            num_heads=key[14],
            apply_weight_offset=key[15],
        )
        # flyc.compile returns a CompiledFunction and performs its first launch.
        compiled = flyc.compile(launcher, *args)
        _FWD_CACHE[key] = compiled
        return compiled, True


def _current_raw_stream(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


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
    key = (
        _compile_context(x.device),
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
        _current_raw_stream(x.device),
    )
    with torch.cuda.device(x.device):
        compiled = _FWD_CACHE.get(key)
        if compiled is not None:
            compiled(*args)
            return
        compiled, launched = _compile_forward(key, x.device, args)
        if not launched:
            compiled(*args)


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
    for name, tensor in (("weight", weight), ("bias", bias), ("residual", residual)):
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
    for name, tensor in (("weight", weight), ("bias", bias), ("residual", residual)):
        if tensor is not None and tensor.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"{name} dtype must be float16, bfloat16, or float32, got {tensor.dtype}"
            )
    for name, dtype in (("out_dtype", out_dtype), ("residual_dtype", residual_dtype)):
        if dtype is not None and dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must be float16, bfloat16, or float32, got {dtype}")

    for name, tensor in (
        ("x", x),
        ("weight", weight),
        ("bias", bias),
        ("residual", residual),
    ):
        if tensor is not None and tensor.layout != torch.strided:
            raise ValueError(f"{name} must use torch.strided layout, got {tensor.layout}")
    if torch.version.hip is None or x.device.type != "cuda":
        raise ValueError(f"x must be on a ROCm device, got {x.device}")
    for name, tensor in (("weight", weight), ("bias", bias), ("residual", residual)):
        if tensor is not None and tensor.device != x.device:
            raise ValueError(
                f"x and {name} must be on the same device, got {x.device}/{tensor.device}"
            )

    if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
        raise TypeError(f"eps must be a real number, got {type(eps).__name__}")
    eps = float(eps)
    if not 0.0 < eps < math.inf:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    if not isinstance(store_rstd, bool):
        raise TypeError(f"store_rstd must be a bool, got {type(store_rstd).__name__}")
    if isinstance(weight_offset, bool) or not isinstance(weight_offset, numbers.Real):
        raise TypeError(f"weight_offset must be a real number, got {type(weight_offset).__name__}")
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


def _unambiguous_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure FlyDSL identifies the last dimension as the unit-stride row."""
    row = tensor.dim() - 1
    offenders = [
        axis for axis in range(row) if tensor.shape[axis] == 1 and tensor.stride(axis) == 1
    ]
    if not offenders:
        return tensor
    for axis in reversed(offenders):
        tensor = tensor.squeeze(axis)
    for axis in offenders:
        tensor = tensor.unsqueeze(axis)
    return tensor


def _rows_are_disjoint_and_packed(tensor: torch.Tensor) -> bool:
    if tensor.stride(-1) != 1:
        return False
    access_bytes = ACCESS_BITS // 8
    if tensor.data_ptr() % access_bytes:
        return False
    span = tensor.shape[-1]
    for stride, size in sorted(
        zip(tensor.stride()[:-1], tensor.shape[:-1]), key=lambda axis: axis[0]
    ):
        if size > 1 and (stride * tensor.element_size()) % access_bytes:
            return False
        if stride < span:
            return False
        span = stride * (size - 1) + span
    return True


def _packed_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Preserve safe row padding and copy only incompatible layouts."""
    tensor = _unambiguous_layout(tensor)
    if _rows_are_disjoint_and_packed(tensor):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _empty_placeholder(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    tensor = _EMPTY_CACHE.get(key)
    if tensor is None:
        tensor = torch.empty(0, device=device, dtype=dtype)
        _EMPTY_CACHE[key] = tensor
    return tensor


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
    """Run eager RMSNorm over the last dimension and return CuTe-compatible outputs."""
    m, n, num_heads, per_head, eps, weight_offset = _validate_inputs(
        x,
        weight,
        bias,
        residual,
        out_dtype,
        residual_dtype,
        eps,
        store_rstd,
        weight_offset,
    )
    output_dtype = x.dtype if out_dtype is None else out_dtype
    residual_out_dtype = (
        residual_dtype
        if residual_dtype is not None
        else (residual.dtype if residual is not None else x.dtype)
    )
    store_residual = residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    )

    if m == 0:
        out = torch.empty(x.shape, device=x.device, dtype=output_dtype)
        residual_out = (
            torch.empty(x.shape, device=x.device, dtype=residual_out_dtype) if store_residual else x
        )
        rstd = (
            torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32) if store_rstd else None
        )
        return out, residual_out, rstd

    last_shape = (num_heads, n) if per_head else (n,)
    x_flat = _packed_rows(x.reshape(-1, *last_shape))
    weight_arg = (
        _packed_rows(weight) if weight is not None else _empty_placeholder(x.device, x.dtype)
    )
    bias_arg = _packed_rows(bias) if bias is not None else _empty_placeholder(x.device, x.dtype)
    residual_arg = (
        _packed_rows(residual.reshape(-1, *last_shape))
        if residual is not None
        else _empty_placeholder(x.device, x.dtype)
    )

    out_flat = torch.empty(x_flat.shape, device=x.device, dtype=output_dtype)
    residual_out_flat = (
        torch.empty(x_flat.shape, device=x.device, dtype=residual_out_dtype)
        if store_residual
        else _empty_placeholder(x.device, residual_out_dtype)
    )
    rstd_flat = (
        torch.empty(m * num_heads, device=x.device, dtype=torch.float32)
        if store_rstd
        else _empty_placeholder(x.device, torch.float32)
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

    out = out_flat.reshape(x.shape)
    residual_out = residual_out_flat.reshape(x.shape) if store_residual else x
    rstd = rstd_flat.reshape(x.shape[:-1]) if store_rstd else None
    return out, residual_out, rstd
