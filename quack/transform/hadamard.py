# Copyright (c) 2026, QuACK team.

import math
import os
from functools import partial
from typing import Literal, NamedTuple, Type

import torch
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, const_expr

from quack import copy_utils, layout_utils
from quack.cache import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import get_device_multiprocessor_count, torch2cute_dtype_map
from quack.dsl import cute_op


# Largest supported transform length. Tied to `_EPT_BY_N_PADDED` below: every
# padded length up to `_MAX_N` must have an entry there.
_MAX_N = 32768


# Default elements-per-thread for each padded N. Tuned empirically.
_EPT_BY_N_PADDED = {
    2: 2,
    4: 4,
    8: 4,
    16: 4,
    32: 8,
    64: 8,
    128: 8,
    256: 8,
    512: 16,
    1024: 32,
    2048: 32,
    4096: 32,
    8192: 32,
    16384: 64,
    32768: 32,
}
assert max(_EPT_BY_N_PADDED) == _MAX_N


def _next_power_of_2(n: int) -> int:
    return 1 << math.ceil(math.log2(n))


def _log2_exact(n: int, name: str) -> int:
    assert n >= 1 and (n & (n - 1) == 0), f"{name} must be a power of 2"
    return int(math.log2(n))


def _ensure_last_dim_contiguous(t: Tensor) -> Tensor:
    if torch.compiler.is_compiling():
        return t.contiguous()
    return t if t.stride(-1) == 1 else t.contiguous()


def _get_num_sms(device: torch.device) -> int:
    if not torch.cuda.is_available():
        return 1
    device_id = torch.cuda.current_device() if device.index is None else device.index
    return max(1, get_device_multiprocessor_count(device_id))


def _should_use_persistent(x: Tensor, N: int) -> bool:
    # Persistent mode is intended only for the known occupancy-1 case: 32K rows
    # with 16-bit elements. Other sizes keep the regular CTA-per-row launch.
    return N == _MAX_N and x.dtype in (torch.float16, torch.bfloat16)


# ─── Compile-time bit ownership ─────────────────────────────────────────────


class BitOrder(NamedTuple):
    """Compile-time ownership of logical Hadamard bits.

    `t` is the bit order owned by thread id bits, and `v` is the bit order owned
    by values/registers inside each thread.  The class is immutable and still
    tuple-like, but named fields make layout reasoning less error-prone.
    """

    t: tuple[int, ...]
    v: tuple[int, ...]

    @classmethod
    def initial(
        cls,
        log_n: int,
        log_threads_per_transform: int,
        log_copy_vecsize: int,
    ) -> "BitOrder":
        """Initial ownership order induced by `copy_utils.tiled_copy_2d`.

        Low vector bits are per-thread values, then transform-thread bits, then
        any remaining high bits return to per-thread values. For
        N=2048/ept=64/vec=8 this is:
          t=(3, 4, 5, 6, 7), v=(0, 1, 2, 8, 9, 10)
        """
        log_ept = log_n - log_threads_per_transform
        assert 0 <= log_copy_vecsize <= log_ept
        thread_start = log_copy_vecsize
        thread_stop = thread_start + log_threads_per_transform
        thread_bits = tuple(range(thread_start, thread_stop))
        value_bits = tuple(range(log_copy_vecsize)) + tuple(range(thread_stop, log_n))
        assert len(thread_bits) == log_threads_per_transform
        assert len(value_bits) == log_ept
        return cls(thread_bits, value_bits)

    def __str__(self) -> str:
        return f"t={self.t} v={self.v}"

    def __repr__(self) -> str:
        return f"BitOrder(t={self.t}, v={self.v})"

    def hadamard_bits(self, bit_shift: int | None = None) -> tuple[int, ...]:
        if bit_shift is None:
            bit_shift = len(self.v)
        return self.v[:bit_shift]

    def exchange(
        self, bit_shift: int | None = None, tail_register_permute: bool = False
    ) -> "BitOrder":
        """Ownership after `exchange(..., bit_shift=bit_shift)`.

        Store addresses are old thread bits followed by old value bits. The load
        view uses the low `bit_shift` address bits as the exchanged value group
        and the next thread bits as the new owner thread. If omitted,
        `bit_shift=len(v)`, i.e. exchange all current value bits.
        """
        if bit_shift is None:
            bit_shift = len(self.v)
        log_threads = len(self.t)
        address_bits = self.t + self.v
        assert 0 <= bit_shift <= len(self.v)
        new_thread_bits = address_bits[bit_shift : bit_shift + log_threads]
        l, r = address_bits[:bit_shift], address_bits[bit_shift + log_threads :]
        new_value_bits = l + r if not tail_register_permute else r + l
        assert len(new_thread_bits) == len(self.t)
        assert len(new_value_bits) == len(self.v)
        return BitOrder(new_thread_bits, new_value_bits)

    def tail_direct_store(self, bit_shift: int, log_copy_vecsize: int) -> "BitOrder":
        """Ownership after the skipped-tail-exchange register permutation."""
        split = bit_shift + log_copy_vecsize
        prefix = self.v[:split]
        value_bits = prefix[bit_shift:] + prefix[:bit_shift] + self.v[split:]
        return BitOrder(self.t, value_bits)

    def owner_runs(self, log_n: int) -> list[tuple[bool, int, int]]:
        """Group physical output bits 0..log_n-1 into maximal value/thread runs.

        Consecutive runs alternate by construction because ownership is binary.
        Returns `(is_value, start, width)` tuples.
        """
        value_set = set(self.v)
        runs: list[tuple[bool, int, int]] = []
        bit = 0
        while bit < log_n:
            is_value = bit in value_set
            start = bit
            while bit < log_n and (bit in value_set) == is_value:
                bit += 1
            runs.append((is_value, start, bit - start))
        return runs

    def run_offset(self, start: int, width: int, on: Literal["t", "v"]) -> int | None:
        """Where physical run `start:start+width` appears in this t- or v-order.

        Returns the offset, or `None` if the run is not present contiguously.
        """
        order = self.t if on == "t" else self.v
        try:
            offset = order.index(start)
        except ValueError:
            return None
        target = tuple(range(start, start + width))
        return offset if order[offset : offset + width] == target else None


# ─── Tail direct store layout ───────────────────────────────────────────────


class TailDirectStorePlan(NamedTuple):
    """Custom TiledCopy + gmem layout for skipping the final smem exchange.

    Think of the destination N dimension as physical output bits 0..log_n-1.
    After the tail local Hadamard, the direct-store path first permutes rmem so
    the vectorized gmem bits are leading in value order.  We can then skip the
    final smem exchange iff:
      * value order starts with vector bits, and
      * the warp-lane portion of thread order owns the remaining low address bits
        needed for a 128B coalesced segment.

    Running example, N=4096/ept=32/bf16 just before the skipped tail exchange:
      order = BitOrder(t=(10,11,3,4,5,6,7), v=(8,9,0,1,2)), bit_shift=2.
    The final local Hadamard consumes value bits (8,9), then the rmem permute
    changes value order to (0,1,2,8,9).  Physical output bits then group into:
      value run 0..2, thread run 3..7, value run 8..9, thread run 10..11.
    This produces layouts equivalent to:
      gmem = (row, v012, t34567, v89, t1011)
      thr  = (row,    1,     32,   1,     4) with compact tid strides (row,0,4,0,1)
      val  = (  1,    8,      1,   4,     1) with compact val strides (0,1,0,8,0).
    """

    gmem_shape: tuple[int, ...]
    gmem_stride: tuple[int, ...]
    thr_shape: tuple[int, ...]
    thr_stride: tuple[int, ...]
    val_shape: tuple[int, ...]
    val_stride: tuple[int, ...]
    value_store_bits: tuple[int, ...]

    @classmethod
    def is_feasible(
        cls,
        log_n: int,
        order: BitOrder,
        bit_shift: int,
        log_copy_vecsize: int,
        dtype_width: int,
    ) -> bool:
        if log_copy_vecsize == 0:
            return False
        value_store_bits = order.tail_direct_store(bit_shift, log_copy_vecsize).v
        vector_bits = tuple(range(log_copy_vecsize))
        # Physical bits are element-index bits.  A 128B segment spans 64 bf16/fp16
        # elements (bits 0..5) or 32 fp32 elements (bits 0..4).  The vector bits
        # come from registers; the remaining low bits should come from warp lanes.
        log_elems_per_128b = _log2_exact(1024 // dtype_width, "elements_per_128B")
        warp_thread_bits = tuple(range(log_copy_vecsize, min(log_elems_per_128b, log_n)))
        lane_thread_bits = order.t[: min(5, len(order.t))]
        if value_store_bits[:log_copy_vecsize] != vector_bits:
            return False
        if any(bit not in lane_thread_bits for bit in warp_thread_bits):
            return False
        # Each run must be contiguous in either the value or thread bit order.
        store_order = BitOrder(order.t, value_store_bits)
        return all(
            store_order.run_offset(start, width, on="v" if is_value else "t") is not None
            for is_value, start, width in store_order.owner_runs(log_n)
        )

    @classmethod
    def build(
        cls,
        log_n: int,
        order: BitOrder,
        bit_shift: int,
        log_copy_vecsize: int,
        dtype_width: int,
    ) -> "TailDirectStorePlan":
        """Construct the gmem/thr/val layout specs for the skipped-tail store."""
        assert cls.is_feasible(log_n, order, bit_shift, log_copy_vecsize, dtype_width)
        value_store_bits = order.tail_direct_store(bit_shift, log_copy_vecsize).v
        store_order = BitOrder(order.t, value_store_bits)

        gmem_shape = []
        gmem_stride = []
        thr_shape = []
        thr_stride = []
        val_shape = []
        val_stride = []
        runs = store_order.owner_runs(log_n)
        assert all(is_value == (i % 2 == 0) for i, (is_value, _, _) in enumerate(runs))
        for is_value, start, width in runs:
            order_offset = store_order.run_offset(start, width, on="v" if is_value else "t")
            assert order_offset is not None
            run_shape = 1 << width
            run_stride = 1 << order_offset
            gmem_shape.append(run_shape)
            gmem_stride.append(1 << start)
            thr_shape.append(1 if is_value else run_shape)
            thr_stride.append(0 if is_value else run_stride)
            val_shape.append(run_shape if is_value else 1)
            val_stride.append(run_stride if is_value else 0)

        return cls(
            gmem_shape=tuple(gmem_shape),
            gmem_stride=tuple(gmem_stride),
            thr_shape=tuple(thr_shape),
            thr_stride=tuple(thr_stride),
            val_shape=tuple(val_shape),
            val_stride=tuple(val_stride),
            value_store_bits=value_store_bits,
        )


# ─── Host-side schedule ─────────────────────────────────────────────────────


class HadamardTransformPlan:
    """Single source of truth for all host-time Hadamard configuration.

    Owns:
      * Sizes: `dtype`, `N`, `N_padded`, `ept`, `vecsize`, `copy_vecsize`,
        `threads_per_transform`, `rows_per_block`, `num_threads`
      * Log variants: `log_n`, `log_ept`, `log_threads_per_transform`,
        `log_copy_vecsize`
      * Stage schedule: `stages`, `bit_shifts`, `tail_bit_shift`, `bit_orders`
      * Tail-store decision: `tail_store: TailDirectStorePlan | None`
    """

    @staticmethod
    def default_ept_for(N_padded: int) -> int:
        return _EPT_BY_N_PADDED[N_padded]

    @staticmethod
    def default_rows_per_block(threads_per_transform: int, N_padded: int) -> int:
        """Pick a default rows-per-block.

        Small transforms pack multiple rows into one CTA to improve occupancy
        and amortize launch overhead. Larger transforms default to one row per
        CTA because the extra shared-memory exchange/barrier work outweighs the
        occupancy gain.
        """
        if threads_per_transform > cute.arch.WARP_SIZE:
            return 1
        thread_cap = max(1, 1024 // threads_per_transform)
        reg_cap = max(1, 8192 // N_padded)
        return min(thread_cap, reg_cap)

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        ept: int | None = None,
        rows_per_block: int | None = None,
        tail_direct_store_enabled: bool | None = None,
    ):
        # ── Sizes
        assert 2 <= N <= _MAX_N, f"Hadamard requires 2 <= last dim <= {_MAX_N}"
        self.dtype = dtype
        self.N = N
        self.N_padded = _next_power_of_2(N)
        assert self.N_padded <= _MAX_N, f"Padded Hadamard dim must be <= {_MAX_N}"

        if ept is None:
            ept = self.default_ept_for(self.N_padded)
        assert ept >= 1 and (ept & (ept - 1) == 0), "ept must be a power of 2"
        assert self.N_padded % ept == 0, "Padded Hadamard dimension must be divisible by ept"
        self.ept = ept
        self.threads_per_transform = self.N_padded // ept
        assert self.threads_per_transform >= 1 and (
            self.threads_per_transform & (self.threads_per_transform - 1) == 0
        ), "N_padded / ept must be a power of 2"

        max_vecsize = 4 if dtype.width == 32 else 8
        self.vecsize = min(max_vecsize, self.ept)
        # Row stride is the original N, so padded/non-power-of-two rows may not
        # be aligned enough for the full compute vector width in global memory.
        self.copy_vecsize = math.gcd(self.vecsize, self.N)
        assert self.vecsize & (self.vecsize - 1) == 0
        assert self.copy_vecsize & (self.copy_vecsize - 1) == 0
        assert self.ept % self.vecsize == 0

        if rows_per_block is None:
            rows_per_block = self.default_rows_per_block(self.threads_per_transform, self.N_padded)
        assert rows_per_block >= 1, "rows_per_block must be positive"
        assert rows_per_block * self.threads_per_transform <= 1024, (
            "block size exceeds 1024 threads"
        )
        self.rows_per_block = rows_per_block
        self.num_threads = self.threads_per_transform * rows_per_block

        # ── Log variants
        self.log_n = _log2_exact(self.N_padded, "N_padded")
        self.log_threads_per_transform = _log2_exact(
            self.threads_per_transform, "threads_per_transform"
        )
        self.log_copy_vecsize = _log2_exact(self.copy_vecsize, "copy_vecsize")
        self.log_ept = _log2_exact(self.ept, "ept")
        assert self.log_ept == self.log_n - self.log_threads_per_transform

        # ── Stage schedule
        self.stages = (self.log_n + self.log_ept - 1) // self.log_ept
        self.tail_bit_shift = self.log_n % self.log_ept or self.log_ept
        self.bit_shifts = tuple(
            self.tail_bit_shift if stage == self.stages - 1 else self.log_ept
            for stage in range(self.stages)
        )

        # ── Tail direct store decision
        #
        # The final stage can be a "tail" when log_N is not an exact multiple of
        # log_ept.  Normally we would run the final local Hadamard, do one more
        # full fp32 smem exchange to restore the original gmem ownership order,
        # then use the regular tiled gmem store.  In some schedules, however,
        # the state *before* that final exchange already has:
        #   - the vectorized gmem bits in per-thread values after a small rmem
        #     permute, and
        #   - the remaining low gmem bits for a 128B segment owned by the warp-lane
        #     thread bits (e.g. bf16 copy_vecsize=8 needs 3,4,5; bf16
        #     copy_vecsize=4 needs 2,3,4,5; fp32 copy_vecsize=4 needs 2,3,4).
        # When `TailDirectStorePlan.is_feasible` says yes we skip that last smem
        # round trip and store directly to gmem with a custom TiledCopy while
        # preserving coalesced vector stores.
        if tail_direct_store_enabled is None:
            tail_direct_store_enabled = os.getenv("QUACK_HADAMARD_TAIL_DIRECT_STORE", "1") != "0"

        # Compute bit orders without tail direct store first, so feasibility can
        # be checked on `bit_orders[-2]` (the order before the final stage,
        # which is identical in both schedules).
        self.bit_orders = self._compute_bit_orders(tail_direct_store=False)
        use_tail_direct_store = (
            tail_direct_store_enabled
            and self.tail_bit_shift < self.log_ept
            and TailDirectStorePlan.is_feasible(
                self.log_n,
                self.bit_orders[-2],
                self.tail_bit_shift,
                self.log_copy_vecsize,
                self.dtype.width,
            )
        )
        if use_tail_direct_store:
            self.bit_orders = self._compute_bit_orders(tail_direct_store=True)
            self.tail_store: TailDirectStorePlan | None = TailDirectStorePlan.build(
                self.log_n,
                self.bit_orders[-2],
                self.tail_bit_shift,
                self.log_copy_vecsize,
                self.dtype.width,
            )
        else:
            self.tail_store = None

        # If one transform fits in a warp, every smem exchange stays within that
        # warp.  The rows_per_block packing uses a power-of-two x dimension, so
        # each transform is warp-contained and row-local smem slices do not
        # require a CTA-wide barrier.
        self.exchange_uses_syncwarp = self.threads_per_transform <= cute.arch.WARP_SIZE

    @property
    def use_tail_direct_store(self) -> bool:
        return self.tail_store is not None

    def __str__(self) -> str:
        lines = [
            "Hadamard bit order: "
            f"N={self.N} N_padded={self.N_padded} ept={self.ept} "
            f"threads={self.threads_per_transform} copy_vecsize={self.copy_vecsize}"
        ]
        for stage, bit_shift in enumerate(self.bit_shifts):
            before = self.bit_orders[stage]
            after = self.bit_orders[stage + 1]
            lines.append(
                f"  stage {stage} bit_shift={bit_shift} "
                f"hadamard_bits={before.hadamard_bits(bit_shift)} "
                f"before {before} -> after {after}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            "HadamardTransformPlan("
            f"N={self.N}, N_padded={self.N_padded}, ept={self.ept}, "
            f"threads_per_transform={self.threads_per_transform}, "
            f"rows_per_block={self.rows_per_block}, "
            f"copy_vecsize={self.copy_vecsize}, "
            f"use_tail_direct_store={self.use_tail_direct_store}, "
            f"exchange_uses_syncwarp={self.exchange_uses_syncwarp})"
        )

    def _compute_bit_orders(self, tail_direct_store: bool) -> tuple[BitOrder, ...]:
        """Compute ownership order before each stage, plus the final store order."""
        order = BitOrder.initial(self.log_n, self.log_threads_per_transform, self.log_copy_vecsize)
        bit_orders = [order]
        for stage, bit_shift in enumerate(self.bit_shifts):
            assert bit_shift <= len(order.v)
            is_tail_stage = stage == len(self.bit_shifts) - 1
            if tail_direct_store and is_tail_stage:
                # The skipped-exchange store first permutes local values so the
                # contiguous vector bits are leading: v=(low-vector, tail, rest).
                order = order.tail_direct_store(bit_shift, self.log_copy_vecsize)
            else:
                tail_register_permute = is_tail_stage and bit_shift < len(order.v)
                order = order.exchange(bit_shift, tail_register_permute=tail_register_permute)
            bit_orders.append(order)

        consumed_bits = tuple(
            bit
            for order, bit_shift in zip(bit_orders, self.bit_shifts)
            for bit in order.hadamard_bits(bit_shift)
        )
        assert sorted(consumed_bits) == list(range(self.log_n)), (
            "Hadamard stage bit order should consume each logical bit exactly once"
        )
        return tuple(bit_orders)


# ─── CuTe DSL helpers ───────────────────────────────────────────────────────
#
# Kept as module-level free functions because CuTe DSL relies on
# `inspect.getsourcelines()`, which makes class-method `@cute.jit` definitions
# fragile (see AGENTS.md).


@cute.jit
def _tail_store_pred(coords: cute.Tensor, layout: cute.Layout, row_stride: Int32) -> cute.Tensor:
    # `coords` has a vectorized first mode `((copy_elem, packet), ...)` while
    # cute.copy wants one predicate per vector packet: `(packet, ...)`.
    pred_shape = (coords.shape[0][1], *coords.shape[1:])
    pred = cute.make_rmem_tensor(pred_shape, Boolean)
    pred_id = cute.make_identity_tensor(pred_shape)
    flat_pred = cute.coalesce(pred)
    flat_id = cute.coalesce(pred_id)
    for i in cutlass.range_constexpr(cute.size(flat_pred)):
        pred_coord = flat_id[i]
        coord = ((0, pred_coord[0]), *pred_coord[1:])
        output_coord = coords[coord]
        offset = cute.crd2idx(output_coord, layout) - output_coord[0] * row_stride
        flat_pred[i] = offset < row_stride
    return pred


@cute.jit
def _hadamard_thread_col(vals: cute.Tensor) -> None:  # (N, col)
    n = cute.size(vals, mode=[0])
    log_n = int(math.log2(n))
    assert n == 1 << log_n, "hadamard_thread_col requires power-of-two size"
    for step in cutlass.range_constexpr(log_n):
        stride = 1 << step
        for j in cutlass.range(1 << (log_n - 1), unroll_full=True):
            lo = j & (stride - 1)
            idx = (j - lo) * 2 + lo
            for col in cutlass.range(cute.size(vals, mode=[1]), unroll_full=True):
                a, b = vals[idx, col], vals[idx + stride, col]
                vals[idx, col] = a + b
                vals[idx + stride, col] = a - b


@cute.jit
def _hadamard_warp(
    vals: cute.Tensor,
    tidx: Int32,
    log_width: cutlass.Constexpr[int],
) -> None:
    for step in cutlass.range_constexpr(log_width):
        offset = const_expr(1 << step)
        sign_bit = tidx & offset
        sign = Float32(1.0) if sign_bit == 0 else Float32(-1.0)
        for i in cutlass.range(cute.size(vals), unroll_full=True):
            vals[i] = sign * vals[i] + cute.arch.shuffle_sync_bfly(vals[i], offset=offset)


@cute.jit
def exchange(
    vals: cute.Tensor,  # size ept
    smem: cute.Tensor,  # compact backing smem, size ept * threads_per_transform
    tidx: Int32,
    sync_fn,
    bit_shift: int | None = None,  # if None, shift all value bits, i.e. log2(ept)
) -> cute.Tensor:  # (ept,)
    ept = cute.size(vals)
    log_ept = int(math.log2(ept))
    if const_expr(bit_shift is None):
        bit_shift = log_ept
    assert 0 <= bit_shift <= log_ept, "bit_shift must be in [0, log2(ept)]"
    radix = const_expr(1 << bit_shift)
    threads_per_transform = cute.size(smem) // ept
    log_threads_per_transform = int(math.log2(threads_per_transform))
    # Store address bits are: old thread bits, then old value bits.
    smem_store_base = cute.make_layout((threads_per_transform, ept))
    # Load address bits are: exchanged value bits, new thread bits, leftover value bits.
    # Tail exchanges are register-permuted below, not loaded in a different smem order.
    smem_load_base = cute.make_layout(
        (threads_per_transform, (radix, ept // radix)),
        stride=(radix, (1, threads_per_transform * radix)),
    )

    log_vecsize = min(2, bit_shift)  # vectorize 4 elements during load, unless bit_shift < 2
    log_threads_in_phase = min(5, bit_shift, log_threads_per_transform) - log_vecsize
    swizzle = cute.make_swizzle(log_threads_in_phase, log_vecsize, bit_shift - log_vecsize)
    smem_store_layout = cute.make_composed_layout(swizzle, 0, smem_store_base)
    smem_load_layout = cute.make_composed_layout(swizzle, 0, smem_load_base)
    smem_store = cute.make_tensor(smem.iterator, smem_store_layout)
    smem_load = cute.make_tensor(smem.iterator, smem_load_layout)
    sX_store = smem_store[tidx, None]  # (ept)
    sX_load = smem_load[tidx, None]
    cute.autovec_copy(cute.composition(vals, (ept,)), sX_store)
    sync_fn()
    vals_exchanged = copy_utils.load_s2r(sX_load)
    # `load_s2r` preserves a layout like the swizzled smem source; the values are correct, but
    # that swizzled rmem layout is not contiguous, e.g, ((1,(4,2,2,2,2))):((0,(1,4,8,16,32))).
    # We call contiguous to materialize into a compact rmem layout.
    vals_exchanged = copy_utils.contiguous(vals_exchanged)
    if const_expr(bit_shift < log_ept):
        # Tail exchanges leave the loaded value bits in `(exchanged, leftover)` order.
        # Before the N=2048/ept=64 tail exchange, for example, ownership is
        #   t=(1,2,8,9,10), v=(3,4,5,6,7,0), bit_shift=5.
        # The raw load produces
        #   t=(3,4,5,6,7), v=(1,2,8,9,10,0),
        # but the original gmem store layout expects local v=(0,1,2,8,9,10).  View the
        # 64 registers as (32,2) = (exchanged bits, leftover bit), transpose to (2,32),
        # and compact so the leftover bit 0 becomes the leading local value bit again.
        vals_exchanged = cute.composition(vals_exchanged, cute.make_layout((radix, ept // radix)))
        vals_exchanged = copy_utils.contiguous(layout_utils.select(vals_exchanged, [1, 0]))
    return cute.composition(vals_exchanged, cute.make_layout((ept, 1)))


# ─── Kernel object ──────────────────────────────────────────────────────────


class HadamardTransform:
    """Hadamard kernel wrapper. Holds a `HadamardTransformPlan` and launches it."""

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        ept: int | None = None,
        rows_per_block: int | None = None,
        persistent: bool = False,
    ):
        self.plan = HadamardTransformPlan(dtype, N, ept=ept, rows_per_block=rows_per_block)
        self.persistent = persistent
        self.async_load = persistent
        self.use_shuffle = N <= 128  # slightly faster to use shuffles than smem for small N
        # print(self.plan)  # Uncomment to see the generated schedule and bit orders.

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        scale: Float32,
        num_sms: Int32,
        stream: cuda.CUstream,
    ):
        plan = self.plan
        assert mX.element_type == plan.dtype
        assert mO.element_type == plan.dtype
        # tiled_copy lays out `num_threads = threads_per_transform * rows_per_block`
        # threads as `(rows_per_block, threads_per_transform)` rows.  The kernel
        # launch keeps the transform thread index in x and the packed row in y,
        # so the copy path reconstructs the equivalent linear thread id as
        # `row_in_cta * threads_per_transform + tidx`.
        tiled_copy = copy_utils.tiled_copy_2d(
            plan.dtype, plan.threads_per_transform, plan.num_threads, plan.copy_vecsize
        )
        if const_expr(plan.use_tail_direct_store):
            tail = plan.tail_store
            tail_store_layout = cute.make_layout(tail.gmem_shape, stride=tail.gmem_stride)
            tail_store_thr_layout = cute.make_layout(tail.thr_shape, stride=tail.thr_stride)
            tail_store_val_layout = cute.make_layout(tail.val_shape, stride=tail.val_stride)
            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                plan.dtype,
                num_bits_per_copy=plan.copy_vecsize * plan.dtype.width,
            )
            tail_store_copy = cute.make_tiled_copy_tv(
                copy_atom, tail_store_thr_layout, tail_store_val_layout
            )
        else:
            tail_store_layout = cute.make_layout(1)
            tail_store_copy = tiled_copy
        tiler_mn = (plan.rows_per_block, plan.N_padded)
        # Each CTA processes `rows_per_block` rows; ceil-div handles a trailing partial CTA.
        num_blocks_m = cute.ceil_div(mX.shape[0], plan.rows_per_block)
        # Persistent mode assumes this kernel is used only for shapes with occupancy 1 CTA/SM.
        # Therefore one persistent CTA per SM is enough; do not query full occupancy here.
        grid_x = cutlass.min(num_blocks_m, num_sms) if const_expr(self.persistent) else num_blocks_m
        self.kernel(mX, mO, scale, tiler_mn, tiled_copy, tail_store_copy, tail_store_layout).launch(
            grid=[grid_x, 1, 1],
            block=[plan.threads_per_transform, plan.rows_per_block, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        scale: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        tail_store_copy: cute.TiledCopy,
        tail_store_layout: cute.Layout,
    ):
        plan = self.plan
        tidx, _, _ = cute.arch.thread_idx()
        row_in_cta = 0 if const_expr(plan.rows_per_block == 1) else cute.arch.thread_idx()[1]
        block_row, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()

        shape = mX.shape
        # Each CTA, per iteration, processes a (rows_per_block, N_padded) tile starting at
        # row block_row * rows_per_block.
        gX, gO = [cute.local_tile(mT, tiler_mn, (None, 0)) for mT in (mX, mO)]
        thr_copy = tiled_copy.get_slice(row_in_cta * plan.threads_per_transform + tidx)
        tXgX = thr_copy.partition_S(gX)
        tXgO = thr_copy.partition_D(gO)
        cX = cute.make_identity_tensor(tiler_mn)
        tXcX_full = thr_copy.partition_S(cX)
        tXrX = cute.make_rmem_tensor_like(tXgX[..., 0])
        tXrO = cute.make_rmem_tensor_like(tXgO[..., 0])
        tXsX = None
        if const_expr(self.async_load):
            sX = smem.allocate_tensor(
                plan.dtype, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
            )
            tXsX = thr_copy.partition_D(sX)

        num_rows = cute.size(mX.shape[0])
        is_even_N = const_expr(shape[1] == tiler_mn[1])
        tXpX = None if is_even_N else copy_utils.predicate_k(tXcX_full, limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        s_exchange_layout = cute.make_ordered_layout(
            (plan.threads_per_transform * plan.ept, plan.rows_per_block), order=(0, 1)
        )
        s_exchange = None
        if const_expr(not self.use_shuffle):
            s_exchange = smem.allocate_tensor(Float32, s_exchange_layout, byte_alignment=16)
            s_exchange = s_exchange[None, row_in_cta]
        sync_fn = (
            cute.arch.sync_warp if const_expr(plan.exchange_uses_syncwarp) else cute.arch.barrier
        )

        nblocks_m = cute.ceil_div(num_rows, plan.rows_per_block)
        num_cta = cute.arch.grid_dim()[0] if const_expr(self.persistent) else nblocks_m
        if const_expr(self.async_load):
            if block_row < nblocks_m:
                copy(tXgX[..., block_row], tXsX, is_async=True)
                cute.arch.cp_async_commit_group()
        num_iter = (
            1 if const_expr(not self.persistent) else cute.ceil_div(nblocks_m - block_row, num_cta)
        )
        for i in cutlass.range(num_iter, unroll=2 if const_expr(self.persistent) else 1):
            row_block = block_row + i * num_cta
            row = row_block * plan.rows_per_block + row_in_cta
            row_is_valid = True if const_expr(tiler_mn[0] == 1) else row < num_rows
            if const_expr(self.async_load):
                cute.arch.cp_async_wait_group(0)
                cute.autovec_copy(tXsX, tXrX)
                next_row_block = row_block + num_cta
                if next_row_block < nblocks_m:
                    copy(tXgX[..., next_row_block], tXsX, is_async=True)
                    cute.arch.cp_async_commit_group()
            else:
                if const_expr(not is_even_N):
                    tXrX.fill(tXrX.element_type.zero)
                if row_is_valid:
                    copy(tXgX[..., row_block], tXrX)

            x_flat = cute.composition(tXrX, cute.make_layout((cute.size(tXrX), 1)))
            x_vals = cute.make_rmem_tensor_like(x_flat, dtype=Float32)
            x_vals.store(x_flat.load().to(Float32))

            if const_expr(self.use_shuffle):
                _hadamard_thread_col(x_vals)
                _hadamard_warp(x_vals, tidx, log_width=plan.log_threads_per_transform)
            else:
                for stage in cutlass.range_constexpr(plan.stages):
                    bit_shift = const_expr(plan.bit_shifts[stage])
                    radix = const_expr(1 << bit_shift)
                    x_vals = cute.composition(x_vals, cute.make_layout((radix, plan.ept // radix)))
                    _hadamard_thread_col(x_vals)
                    if const_expr(stage < plan.stages - 1 or not plan.use_tail_direct_store):
                        if const_expr(stage > 0 or self.persistent):
                            # Before reusing the exchange buffer, wait for the previous
                            # exchange loads to finish.  Warp-local exchange plans use a
                            # warp fence; cross-warp plans use a CTA barrier.
                            sync_fn()
                        x_vals = exchange(x_vals, s_exchange, tidx, sync_fn, bit_shift=bit_shift)

            if const_expr(not self.use_shuffle and plan.use_tail_direct_store):
                tail_size = const_expr(1 << plan.tail_bit_shift)
                rest_size = const_expr(plan.ept // (tail_size * plan.copy_vecsize))
                x_store = cute.composition(
                    x_vals, cute.make_layout((tail_size, plan.copy_vecsize, rest_size))
                )
                x_store = copy_utils.contiguous(layout_utils.select(x_store, [1, 0, 2]))
                gO_store = gO[row_in_cta, None, row_block]
                thr_store = tail_store_copy.get_slice(tidx)
                tOgO = thr_store.partition_D(cute.composition(gO_store, tail_store_layout))
                tOrO = cute.make_rmem_tensor_like(tOgO, tXrO.element_type)
                cute.coalesce(tOrO).store(
                    (cute.coalesce(x_store).load() * scale).to(tOrO.element_type)
                )
                if row_is_valid:
                    if const_expr(is_even_N):
                        cute.copy(tail_store_copy, tOrO, tOgO)
                    else:
                        cO_store = cute.make_identity_tensor(tail_store_layout.shape)
                        tOcO = thr_store.partition_D(cO_store)
                        tOpO = _tail_store_pred(tOcO, tail_store_layout, shape[1])
                        cute.copy(tail_store_copy, tOrO, tOgO, pred=tOpO)
            else:
                o_flat = cute.composition(tXrO, cute.make_layout((cute.size(tXrO), 1)))
                o_flat.store((x_vals.load() * scale).to(tXrO.element_type))
                if row_is_valid:
                    copy(tXrO, tXgO[..., row_block])

    @staticmethod
    @jit_cache
    def compile(dtype, N, persistent: bool = False):
        batch_sym = cute.sym_int()
        div = math.gcd(N, 128 // dtype.width)
        x_cute = fake_tensor(dtype, (batch_sym, N), div)
        out_cute = fake_tensor(dtype, (batch_sym, N), div)
        return cute.compile(
            HadamardTransform(dtype, N, persistent=persistent),
            x_cute,
            out_cute,
            Float32(0.0),
            0,  # num_sms, just for compilation
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


# ─── Autograd, public API ──────────────────────────────────────────


@cute_op(
    "quack::_hadamard_transform_fwd",
    mutates_args={"out"},
    device_types="cuda",
)
def _hadamard_transform_fwd(x: Tensor, out: Tensor, scale: float) -> None:
    """Custom-op binding: dispatch to the cached compiled kernel."""
    assert x.dim() == 2, "Input must be 2D"
    assert out.shape == x.shape, "Output shape must match input"
    assert x.dtype == out.dtype, "Output dtype must match input dtype"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    if x.numel() == 0:
        return
    N = x.size(1)
    assert 2 <= N <= _MAX_N, f"Hadamard transform supports last dimension in [2, {_MAX_N}]"
    dtype = torch2cute_dtype_map[x.dtype]
    persistent = _should_use_persistent(x, N)
    compiled = HadamardTransform.compile(dtype, N, persistent)
    num_sms = _get_num_sms(x.device) if persistent else 0
    compiled(x, out, scale, num_sms)


def hadamard_transform_fwd(x: Tensor, scale: float = 1.0) -> Tensor:
    """Forward-only entry point: normalize layout, dispatch to the custom op."""
    assert x.dim() >= 1, "Input must have at least one dimension"
    x = _ensure_last_dim_contiguous(x)
    N = x.size(-1)
    assert 1 <= N <= _MAX_N, f"Hadamard transform supports last dimension in [1, {_MAX_N}]"
    if x.numel() == 0:
        return torch.empty_like(x)
    if N == 1:
        return x * float(scale)
    x_2d = x.reshape(-1, N)
    out_2d = torch.empty_like(x_2d)
    _hadamard_transform_fwd(x_2d, out_2d, float(scale))
    return out_2d.reshape(x.shape)


def hadamard_transform_ref(x: Tensor, scale: float = 1.0) -> Tensor:
    """PyTorch reference with the same zero-padding convention as fast-hadamard-transform."""
    assert x.dim() >= 1, "Input must have at least one dimension"
    N = x.size(-1)
    assert 1 <= N <= _MAX_N, f"Hadamard transform supports last dimension in [1, {_MAX_N}]"
    N_padded = _next_power_of_2(N)
    y = x.float().reshape(-1, N)
    if N_padded != N:
        y = torch.nn.functional.pad(y, (0, N_padded - N))
    h = 1
    while h < N_padded:
        y = y.reshape(-1, N_padded // (2 * h), 2, h)
        y0 = y[:, :, 0, :]
        y1 = y[:, :, 1, :]
        y = torch.stack((y0 + y1, y0 - y1), dim=2).reshape(-1, N_padded)
        h *= 2
    return (y[:, :N] * scale).reshape(x.shape).to(x.dtype)


class HadamardTransformFunction(torch.autograd.Function):
    """Autograd wrapper. The Hadamard transform is self-adjoint, so backward = forward."""

    @staticmethod
    def forward(ctx, x: Tensor, scale: float = 1.0):
        ctx.scale = float(scale)
        return hadamard_transform_fwd(x, ctx.scale)

    @staticmethod
    def backward(ctx, dout: Tensor):
        return hadamard_transform_fwd(dout, ctx.scale), None


def hadamard_transform(x: Tensor, scale: float = 1.0) -> Tensor:
    """Apply a Sylvester Hadamard transform along the last dimension."""
    return HadamardTransformFunction.apply(x, scale)
