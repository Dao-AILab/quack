# Copyright (c) 2026, Tri Dao.

"""FlyDSL sibling of :mod:`quack.reduce`: the row reduction a norm kernel needs.

Same two layers as the CuTe original -- shuffles within a subgroup, then LDS
across subgroups -- minus the cluster tier, which has no CDNA counterpart.
gfx950 has no ``redux`` instruction either, so the subgroup step is always the
shuffle butterfly the CuTe helper keeps as its fallback.

FlyDSL rewrites control flow into ``scf`` only inside a ``@flyc.kernel`` body, so
a helper meant to be shared cannot branch on a dynamic value: every upstream
FlyDSL norm kernel re-inlines its own copy of this for that reason. The one
branch a block reduction needs -- "only the first lane of the wave stores" -- is
expressed here as a predicated copy instead, which is what FlyDSL uses to
predicate every other memory access.
"""

import math
import operator

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp

from quack.flydsl_constants import WAVE_SIZE

__all__ = ["block_reduce", "make_reduction_buffer", "row_reduce", "warp_reduce"]


def make_reduction_buffer(rows: int, waves_per_row: int, dtype=fx.Float32):
    """Shared staging for the cross-wave half of a row reduction.

    Shaped ``(rows, waves_per_row)`` with waves-per-row contiguous, which is what
    :func:`block_reduce` indexes. Single-shot: nothing is pipelined through it,
    so unlike the CuTe original there are no mbarriers to pair with it.
    """

    @fx.struct
    class ReductionStorage:
        buffer: fx.Array[dtype, rows * waves_per_row, 16]

    storage = fx.SharedAllocator().allocate(ReductionStorage).peek()
    return storage.buffer.view(fx.make_ordered_layout((rows, waves_per_row), order=(1, 0)))


_WAVE_OPS = {
    ReductionOp.ADD: operator.add,
    ReductionOp.MUL: operator.mul,
    # Python max/min branch on a dynamic Boolean while tracing. These typed
    # builders are stable across the supported FlyDSL 0.3 release line.
    ReductionOp.MAX: fx.arith.maximumf,
    ReductionOp.MIN: fx.arith.minimumf,
}


def warp_reduce(val, op, threads_in_group: int = WAVE_SIZE):
    """Reduce across the aligned ``threads_in_group``-lane subgroup this thread
    belongs to; every lane receives the result.

    ``threads_in_group`` must be a power of two no wider than a wave, so the
    butterfly stays within one subgroup and needs no member mask.
    """
    assert threads_in_group & (threads_in_group - 1) == 0, "subgroup must be a power of two"
    assert threads_in_group <= WAVE_SIZE, "a butterfly cannot cross waves"
    for step in range_constexpr(int(math.log2(threads_in_group))):
        offset = threads_in_group // (2 << step)
        val = op(val, gpu.shuffle_xor(val, offset, fx.Int32(threads_in_group)))
    return val


def _store_first_lane(reduction_buffer, row, col, val, lane_idx):
    """Land one wave's partial in its slot, without a branch.

    All lanes hold the same value once the butterfly has run, so this is purely
    about not having 64 lanes contend for one address.
    """
    dtype = reduction_buffer.element_type
    src = fx.make_rmem_tensor(1, dtype)
    src[0] = val
    first_lane = fx.make_rmem_tensor(1, fx.Boolean)
    first_lane[0] = lane_idx == fx.Int32(0)
    slot = fx.slice(
        fx.logical_divide(reduction_buffer[row, None], fx.make_layout(1, 1)), (None, col)
    )
    atom = fx.make_copy_atom(fx.UniversalCopy(dtype.width), dtype.width)
    fx.copy(atom, src, slot, pred=first_lane)


def block_reduce(val, op, reduction_buffer, init_val=0.0):
    """Combine per-wave partials staged in ``reduction_buffer``, shaped
    ``(rows per block, waves per row)``.

    One barrier, not two: every wave reads the partials back and folds them with
    a second butterfly, so the result reaches all lanes without a broadcast slot.
    """
    dtype = reduction_buffer.element_type
    tidx = fx.Int32(fx.thread_idx.x)
    lane_idx = tidx % fx.Int32(WAVE_SIZE)
    wave_idx = tidx // fx.Int32(WAVE_SIZE)
    waves_per_row = reduction_buffer.shape.unpack()[1]
    row_idx = wave_idx // fx.Int32(waves_per_row)
    col_idx = wave_idx % fx.Int32(waves_per_row)

    _store_first_lane(reduction_buffer, row_idx, col_idx, val, lane_idx)
    gpu.barrier()
    # Lanes past the partial count must not address the buffer at all, so the
    # slot is clamped before the load and the value dropped after it.
    in_range = lane_idx < fx.Int32(waves_per_row)
    slot = in_range.select(lane_idx, fx.Int32(0))
    partial = in_range.select(reduction_buffer[row_idx, slot], dtype(init_val))
    return warp_reduce(partial, op, WAVE_SIZE)


def row_reduce(x, op, threads_per_row: int, reduction_buffer=None, init_val=0.0):
    """Reduce one row's values across the ``threads_per_row`` threads that own it.

    ``reduction_buffer`` must be shaped ``(rows per block, waves per row)``; a row
    that fits in one wave needs none and may pass ``None``.
    """
    val = x.reduce(op) if const_expr(isinstance(x, fx.Vector)) else x
    val = warp_reduce(val, _WAVE_OPS[op], threads_in_group=min(threads_per_row, WAVE_SIZE))
    if const_expr(reduction_buffer is not None):  # noqa: SIM102 - compile-time guard
        if const_expr(reduction_buffer.shape.unpack()[1] > 1):
            val = block_reduce(val, _WAVE_OPS[op], reduction_buffer, init_val)
    return val
