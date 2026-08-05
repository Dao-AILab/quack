# Copyright (c) 2025, Tri Dao.

import math
import operator
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, Boolean, const_expr
from cutlass.base_dsl.arch import Arch

import quack.utils as utils


_operator_max = getattr(operator, "max", None)
_operator_min = getattr(operator, "min", None)
_cutlass_min = getattr(cutlass, "min", None)


@cute.jit
def warp_reduce(
    val: cute.Numeric,
    op: Callable,
    threads_in_group: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
    dtype: cutlass.Constexpr = None,
) -> cute.Numeric:
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    if const_expr(threads_in_group == cute.arch.WARP_SIZE):
        val_dtype = dtype if const_expr(dtype is not None) else getattr(val, "dtype", None)
        if const_expr(val_dtype == Int32):
            if const_expr(op is operator.add):
                return cute.arch.warp_redux_sync(val, "add")
            if const_expr(op is max or op is cutlass.max or op is _operator_max):
                return cute.arch.warp_redux_sync(val, "max")
            if const_expr(op is min or op is _cutlass_min or op is _operator_min):
                return cute.arch.warp_redux_sync(val, "min")
        if const_expr(val_dtype == Float32 and arch.is_family_of(Arch.sm_100f)):
            if const_expr(
                op is max or op is cutlass.max or op is cute.arch.fmax or op is _operator_max
            ):
                return cute.arch.warp_redux_sync(val, "fmax")
            if const_expr(
                op is min or op is _cutlass_min or op is cute.arch.fmin or op is _operator_min
            ):
                return cute.arch.warp_redux_sync(val, "fmin")
    return cute.arch.warp_reduction(val, op, threads_in_group=threads_in_group)


@cute.jit
def block_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: cute.Numeric = 0.0,
    dtype: cutlass.Constexpr = None,
) -> cute.Numeric:
    """reduction_buffer has shape (num_warps / warp_per_row, warps_per_row)"""
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op, dtype=dtype)


@cute.jit
def cluster_reduce(
    val: cute.Numeric | tuple,
    op: Callable,
    reduction_buffer: cute.Tensor | tuple,
    mbar_ptr: cute.Pointer,
    init_val: cute.Numeric = 0.0,
    phase: Optional[Int32] = None,
    dtype: cutlass.Constexpr = None,
) -> cute.Numeric | tuple:
    """Each reduction buffer has shape (num_warps / warps_per_row, (warps_per_row, cluster_n)).

    A tuple of values (with a matching tuple of buffers) reduces through a SINGLE
    transaction barrier: it is armed once with the combined byte count, every STAS
    credits it, and one wait covers all buffers — one cluster round trip instead of
    one per value. Sync slots are therefore decoupled from buffer slots.
    """
    is_multi = const_expr(isinstance(val, tuple))
    vals = val if const_expr(is_multi) else (val,)
    bufs = reduction_buffer if const_expr(is_multi) else (reduction_buffer,)
    assert len(vals) == len(bufs), "one reduction buffer per value"
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_block, (warps_per_row, cluster_n) = bufs[0].shape
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            tx_bytes = sum(num_warps * cluster_n * buf.element_type.width // 8 for buf in bufs)
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tx_bytes)
    if lane_idx < cluster_n:
        for buf, v in zip(bufs, vals):
            utils.store_shared_remote(
                v,
                utils.elem_pointer(buf, (row_idx, (col_idx, cta_rank_in_cluster))),
                mbar_ptr,
                peer_cta_rank_in_cluster=lane_idx,
            )
    cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
    results = []
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
    for buf in bufs:
        block_reduce_val = init_val
        for i in cutlass.range_constexpr(num_iter):
            idx = lane_idx + i * cute.arch.WARP_SIZE
            if idx < cute.size(buf, mode=[1]):
                block_reduce_val = op(block_reduce_val, buf[row_idx, idx])
        results.append(warp_reduce(block_reduce_val, op, dtype=dtype))
    return tuple(results) if const_expr(is_multi) else results[0]


@cute.jit
def block_or_cluster_reduce(
    val: cute.Numeric | tuple,
    op: Callable,
    reduction_buffer: cute.Tensor | tuple,
    mbar_ptr: Optional[cute.Pointer],
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    dtype: cutlass.Constexpr = None,
) -> cute.Numeric | tuple:
    """Perform either block or cluster reduction based on whether mbar_ptr is provided."""
    if const_expr(mbar_ptr is None):
        # No cross-CTA sync to share: reduce each value through its own buffer.
        if const_expr(isinstance(val, tuple)):
            return tuple(
                block_reduce(v, op, buf, init_val=init_val, dtype=dtype)
                for v, buf in zip(val, reduction_buffer)
            )
        return block_reduce(
            val,
            op,
            reduction_buffer,
            init_val=init_val,
            dtype=dtype,
        )
    else:
        return cluster_reduce(
            val,
            op,
            reduction_buffer,
            mbar_ptr,
            phase=phase,
            init_val=init_val,
            dtype=dtype,
        )


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric | tuple,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor | tuple] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Optional[Callable] = None,
) -> cute.Numeric | tuple:
    """Each reduction buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n)).

    A tuple x (with a matching tuple of reduction buffers) reduces every value with
    the same op through a single mbarrier / cluster round trip (see cluster_reduce)
    and returns a tuple.
    """
    is_multi = const_expr(isinstance(x, tuple))
    xs = x if const_expr(is_multi) else (x,)
    vals = []
    for xi in xs:
        if const_expr(isinstance(xi, cute.TensorSSA)):
            val = xi.reduce(op, init_val=init_val, reduction_profile=0)
        else:
            val = xi
        # Scalar inputs (e.g. an ArithValue from a prior TensorSSA.reduce) carry no
        # .dtype; None makes warp_reduce fall back to its generic reduction.
        val_dtype = xi.dtype if const_expr(isinstance(xi, cute.TensorSSA)) else None
        warp_op = {
            cute.ReductionOp.ADD: operator.add,
            cute.ReductionOp.MAX: cute.arch.fmax if const_expr(val_dtype == Float32) else max,
            cute.ReductionOp.MIN: cute.arch.fmin if const_expr(val_dtype == Float32) else min,
            cute.ReductionOp.MUL: operator.mul,
        }[op]
        val = warp_reduce(
            val,
            warp_op,
            threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
            dtype=val_dtype,
        )
        vals.append(val)
        buf_stage_dtype = val_dtype
    if const_expr(hook_fn is not None):
        hook_fn()
    if const_expr(reduction_buffer is not None):
        bufs = reduction_buffer if const_expr(is_multi) else (reduction_buffer,)
        warps_per_row, cluster_n = bufs[0].shape[1]
        assert cluster_n == 1 or mbar_ptr is not None, (
            "mbar_ptr must be provided for cluster reduction"
        )
        if const_expr(warps_per_row > 1 or cluster_n > 1):
            # The combine reads the (single-dtype) buffers, so one op suffices; for
            # multi that dtype is the buffers', for single keep the value's dtype
            # (preserves the exact redux selection of the scalar path).
            if const_expr(is_multi):
                buf_stage_dtype = bufs[0].element_type
            reduced = block_or_cluster_reduce(
                tuple(vals) if const_expr(is_multi) else vals[0],
                warp_op,
                reduction_buffer,
                mbar_ptr,
                phase=phase,
                init_val=init_val,
                dtype=buf_stage_dtype,
            )
            vals = list(reduced) if const_expr(is_multi) else [reduced]
    return tuple(vals) if const_expr(is_multi) else vals[0]


@cute.jit
def online_softmax_reduce(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    hook_fn: Optional[Callable] = None,
    phase: Optional[Int32] = None,
    return_exp_x: bool = False,
) -> [Float32, Float32, Optional[cute.TensorSSA]]:
    assert x.dtype == Float32, "x must be of type Float32"
    """reduction_buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n), 2)"""
    max_x = warp_reduce(
        x.reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
        cute.arch.fmax,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
        dtype=Float32,
    )
    log2_e = math.log2(math.e)
    exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=True)
    sum_exp_x = cute.arch.warp_reduction(
        exp_x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
        operator.add,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if const_expr(hook_fn is not None):
        hook_fn()
    if const_expr(reduction_buffer is not None):
        rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
        assert cluster_n == 1 or mbar_ptr is not None, (
            "mbar_ptr must be provided for cluster reduction"
        )
        if const_expr(warps_per_row > 1 or cluster_n > 1):
            assert reduction_buffer.element_type == Int64, (
                "reduction_buffer must be of type cute.Int64"
            )
            lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
            row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
            if const_expr(mbar_ptr is None):
                if lane_idx == 0:
                    reduction_buffer[row_idx, col_idx] = utils.f32x2_to_i64(max_x, sum_exp_x)
                cute.arch.barrier()
                max_x_single_warp = -Float32.inf
                sum_exp_x = 0.0
                if lane_idx < warps_per_row:
                    max_x_single_warp, sum_exp_x = utils.i64_to_f32x2(
                        reduction_buffer[row_idx, lane_idx]
                    )
                max_x_final = warp_reduce(max_x_single_warp, cute.arch.fmax, dtype=Float32)
                sum_exp_x *= cute.math.exp(max_x_single_warp - max_x_final, fastmath=True)
                sum_exp_x = cute.arch.warp_reduction(sum_exp_x, operator.add)
                if const_expr(return_exp_x):
                    exp_x *= cute.math.exp(max_x - max_x_final, fastmath=True)
                max_x = max_x_final
            else:
                cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        num_warps = rows_per_block * warps_per_row
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_ptr,
                            num_warps * cluster_n * reduction_buffer.element_type.width // 8,
                        )
                if lane_idx < cluster_n:
                    utils.store_shared_remote(
                        utils.f32x2_to_i64(max_x, sum_exp_x),
                        utils.elem_pointer(
                            reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))
                        ),
                        mbar_ptr,
                        peer_cta_rank_in_cluster=lane_idx,
                    )
                cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
                num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
                max_x_single_warp = cute.make_rmem_tensor(num_iter, Float32)
                max_x_single_warp.fill(-Float32.inf)
                sum_exp_x_single_warp = cute.make_rmem_tensor(num_iter, Float32)
                sum_exp_x_single_warp.fill(0.0)
                for i in cutlass.range_constexpr(num_iter):
                    idx = lane_idx + i * cute.arch.WARP_SIZE
                    if idx < cute.size(reduction_buffer, mode=[1]):
                        max_x_single_warp[i], sum_exp_x_single_warp[i] = utils.i64_to_f32x2(
                            reduction_buffer[row_idx, idx]
                        )
                max_x_final = max_x_single_warp.load().reduce(
                    cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0
                )
                max_x_final = warp_reduce(max_x_final, cute.arch.fmax, dtype=Float32)
                sum_exp_x = 0.0
                for i in cutlass.range_constexpr(num_iter):
                    sum_exp_x += sum_exp_x_single_warp[i] * cute.math.exp(
                        max_x_single_warp[i] - max_x_final, fastmath=True
                    )
                sum_exp_x = cute.arch.warp_reduction(sum_exp_x, operator.add)
                if const_expr(return_exp_x):
                    exp_x *= cute.math.exp(max_x - max_x_final, fastmath=True)
                max_x = max_x_final
    return max_x, sum_exp_x, (exp_x if const_expr(return_exp_x) else None)


@cute.jit
def sum_swap_shuffle(
    X: cute.Tensor, elem_per_lane: int = 1, subwarp_size: int = 1, warp_size: int = 32
) -> cute.Tensor:
    """
    For warp reduction, we use Swap Shuffle
    The normal way to reduction among threads:
    use shuffle to let *** the first half of threads *** have *** whole data *** from the second half of threads.
    After each step of reduction, a half of threads won't work in the following steps.
    That is, as the reduction progresses, the efficiency of shuffle & reduction instructions gradually change from 1/2, 1/4 to 1/32 (the worst case).
    To overcome this shortcoming, for a NxN matrix to be reduced among N threads as a 1XN vectors,
    we use swap & shuffle aiming to let *** each half of threads *** have *** a half of data *** from the other half of threads.
    After reduction, each half of threads should deal with a (N/2)x(N/2) sub-matrix independently in the following step.
    We can recursively do this until the problem size is 1.
    """
    assert (
        subwarp_size >= 1
        and subwarp_size <= 32
        and subwarp_size == 1 << int(math.log2(subwarp_size))
    )
    assert (
        warp_size <= 32
        and warp_size % subwarp_size == 0
        and warp_size == 1 << int(math.log2(warp_size))
    )
    lane_idx = cute.arch.lane_idx() // subwarp_size
    X = cute.logical_divide(X, cute.make_layout(elem_per_lane))  # (elem_per_lane, M)
    numvec = cute.size(X, mode=[1])
    assert numvec <= 32 // subwarp_size
    # If X has more values than warp_size // subwarp_size, we first do a normal warp reduction
    # to sum up values held by lanes further than size(X) away
    for i in cutlass.range(
        int(math.log2(numvec)), int(math.log2(warp_size // subwarp_size)), unroll_full=True
    ):
        for v in cutlass.range(cute.size(X), unroll_full=True):
            shfl_val = cute.arch.shuffle_sync_bfly(X[v], offset=(1 << i) * subwarp_size)
            X[v] = X[v] + shfl_val
    for logm in cutlass.range_constexpr(int(math.log2(cute.size(X, mode=[1]))) - 1, -1, -1):
        m = 1 << logm
        for r in cutlass.range(m, unroll_full=True):
            frg_A = X[None, r]
            frg_B = X[None, r + m]
            #  First half of threads swap fragments from the first half of data to the second
            should_swap = not Boolean(lane_idx & m)
            for v in cutlass.range(cute.size(frg_A), unroll_full=True):
                # Step 1: swap
                lower, upper = frg_A[v], frg_B[v]
                frg_A[v] = upper if should_swap else lower
                frg_B[v] = lower if should_swap else upper
                # Step 2: shuffle
                # each half of threads get a half of data from the other half of threads
                shfl_val = cute.arch.shuffle_sync_bfly(frg_A[v], offset=m * subwarp_size)
                # Step 3: reduction
                frg_A[v] = frg_B[v] + shfl_val
    return X[None, 0]
