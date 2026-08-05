# Copyright (c) 2026, Tri Dao.
"""Tests for quack.sync.barrier.Semaphore.

Contract under test: the acquire/release edges and the group-sync pairing, not
just flag values. The turnstile test folds a NON-commutative update
(acc = 2*acc + bidx) through S CTAs using plain (non-atomic) loads/stores by
the elected thread — the result is correct only if release_store publishes the
previous owner's plain write AND wait_eq's acquire makes it visible, in exact
turnstile order. The counter test has every arriver publish data before
release_add and every waiter read all arrivers' data after wait_eq(S).
"""

import pytest
import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

import quack.utils as utils
from quack.sync import Semaphore

THREADS = 128

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_sem(mLock: cute.Tensor, tidx: Int32, sync_kind: cutlass.Constexpr[str]) -> Semaphore:
    if cutlass.const_expr(sync_kind == "named_barrier"):
        sync = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    else:
        sync = "cta"
    return Semaphore(utils.elem_pointer(mLock, 0), tidx, sync=sync)


@cute.kernel
def _turnstile_kernel(mAcc: cute.Tensor, mLock: cute.Tensor, sync_kind: cutlass.Constexpr[str]):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    sem = _make_sem(mLock, tidx, sync_kind)
    sem.wait_eq(bidx)
    if tidx == 0:
        # Plain (non-atomic) RMW: exclusivity and visibility come entirely from
        # the turnstile's release/acquire chain.
        mAcc[0] = mAcc[0] * 2 + bidx
    sem.release_store(bidx + 1)


@cute.kernel
def _counter_kernel(mData: cute.Tensor, mOut: cute.Tensor, mLock: cute.Tensor, num_ctas: Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    sem = _make_sem(mLock, tidx, "cta")
    if tidx == 0:
        mData[bidx] = bidx + 1
    # sync-before-release orders the data write before the flag increment.
    sem.release_add()
    sem.wait_eq(num_ctas)
    if tidx == 0:
        total = Int32(0)
        for i in cutlass.range(num_ctas):
            total += mData[i]
        mOut[bidx] = total


@cute.jit
def _launch_turnstile(
    mAcc: cute.Tensor, mLock: cute.Tensor, ncta: Int32, sync_kind: cutlass.Constexpr[str]
):
    _turnstile_kernel(mAcc, mLock, sync_kind).launch(grid=(ncta, 1, 1), block=(THREADS, 1, 1))


@cute.jit
def _launch_counter(mData: cute.Tensor, mOut: cute.Tensor, mLock: cute.Tensor, ncta: Int32):
    _counter_kernel(mData, mOut, mLock, ncta).launch(grid=(ncta, 1, 1), block=(THREADS, 1, 1))


@requires_cuda
@pytest.mark.parametrize("sync_kind", ["cta", "named_barrier"])
@pytest.mark.parametrize("num_ctas", [2, 32, 148])
def test_turnstile_orders_non_commutative_updates(num_ctas, sync_kind):
    acc = torch.zeros(1, dtype=torch.int32, device="cuda")
    lock = torch.zeros(1, dtype=torch.int32, device="cuda")
    fn = cute.compile(
        _launch_turnstile, from_dlpack(acc), from_dlpack(lock), Int32(num_ctas), sync_kind
    )
    expected = 0
    for b in range(num_ctas):
        expected = expected * 2 + b
    expected &= 0xFFFFFFFF
    for _ in range(5):  # repeat: ordering bugs are schedule-dependent
        acc.zero_()
        lock.zero_()
        fn(from_dlpack(acc), from_dlpack(lock), Int32(num_ctas))
        torch.cuda.synchronize()
        assert acc.item() & 0xFFFFFFFF == expected
        assert lock.item() == num_ctas


@requires_cuda
@pytest.mark.parametrize("num_ctas", [2, 32, 148])
def test_counter_arrivals_publish_data(num_ctas):
    data = torch.zeros(num_ctas, dtype=torch.int32, device="cuda")
    out = torch.zeros(num_ctas, dtype=torch.int32, device="cuda")
    lock = torch.zeros(1, dtype=torch.int32, device="cuda")
    fn = cute.compile(
        _launch_counter, from_dlpack(data), from_dlpack(out), from_dlpack(lock), Int32(num_ctas)
    )
    expected = num_ctas * (num_ctas + 1) // 2
    for _ in range(5):
        data.zero_()
        out.zero_()
        lock.zero_()
        fn(from_dlpack(data), from_dlpack(out), from_dlpack(lock), Int32(num_ctas))
        torch.cuda.synchronize()
        assert torch.all(out == expected), out
        assert lock.item() == num_ctas
