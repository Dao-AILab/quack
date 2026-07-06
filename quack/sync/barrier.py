# Copyright (c) 2026, Tri Dao.

"""Global-memory synchronization helpers for CuTe DSL kernels.

These mirror the small counter-semaphore pattern used by CUTLASS C++
(`cutlass/barrier.h` / `cutlass/semaphore.h`): one elected thread spins on an
acquire load of a global flag, and one elected thread publishes progress with a
release global reduction.  They intentionally do not perform a CTA/warp sync;
callers should pair them with the appropriate warp, named-barrier, or pipeline
synchronization for their producer/consumer protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr


@cute.jit
def wait_eq(
    lock_ptr: cute.Pointer,
    thread_idx: Int32,
    flag_offset: Int32 | Int64,
    val: Int32,
    skip_zero: bool = False,
    sync: Literal["none", "warp", "cta"] = "none",
) -> None:
    """Wait until ``lock_ptr[flag_offset] == val`` using ``thread_idx == 0``."""
    if const_expr(skip_zero):
        if val != 0:
            flag_ptr = lock_ptr + flag_offset
            if thread_idx == 0:
                read_val = Int32(0)
                while read_val != val:
                    read_val = cute.arch.load(flag_ptr, Int32, sem="acquire", scope="gpu")
            if const_expr(sync == "warp"):
                cute.arch.sync_warp()
            elif const_expr(sync == "cta"):
                cute.arch.sync_threads()
    else:
        flag_ptr = lock_ptr + flag_offset
        if thread_idx == 0:
            read_val = Int32(0)
            while read_val != val:
                read_val = cute.arch.load(flag_ptr, Int32, sem="acquire", scope="gpu")
        if const_expr(sync == "warp"):
            cute.arch.sync_warp()
        elif const_expr(sync == "cta"):
            cute.arch.sync_threads()


@cute.jit
def arrive_inc(
    lock_ptr: cute.Pointer,
    thread_idx: Int32,
    flag_offset: Int32 | Int64,
    val: Int32 = 1,
) -> None:
    """Increment ``lock_ptr[flag_offset]`` by ``val`` using ``thread_idx == 0``."""
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        cute.arch.red(flag_ptr, Int32(val), op="add", dtype="s32", sem="release", scope="gpu")


@dataclass(frozen=True)
class Semaphore:
    """Global-memory counter semaphore.

    This keeps the pointer, participating thread index, and flag offset together
    so call sites can read like the CUTLASS C++ semaphore helpers:

    .. code-block:: python

        sem = Semaphore(ptr, tidx, flag_offset)
        sem.wait_eq(expected)
        sem.arrive_inc()

    ``sync`` optionally mirrors CUTLASS's ``GenericBarrier<Sync>`` behavior:
    wait synchronizes after the acquire loop, arrive synchronizes before the
    release increment.  Leave it as ``"none"`` when the call site already has a
    surrounding named barrier / pipeline sync or only needs the raw counter op.
    """

    lock_ptr: cute.Pointer
    thread_idx: int | Int32
    sync: Literal["none", "warp", "cta"] = "none"

    def _sync(self) -> None:
        if self.sync == "warp":
            cute.arch.sync_warp()
        elif self.sync == "cta":
            cute.arch.sync_threads()

    def wait_eq(
        self,
        val: int | Int32,
        flag_offset: Optional[int | Int32 | Int64] = None,
        skip_zero: bool = False,
    ) -> None:
        flag_offset = 0 if flag_offset is None else flag_offset
        wait_eq(
            self.lock_ptr,
            self.thread_idx,
            flag_offset,
            val,
            skip_zero=skip_zero,
            sync=self.sync,
        )

    def arrive_inc(
        self, val: int | Int32 = 1, flag_offset: Optional[int | Int32 | Int64] = None
    ) -> None:
        flag_offset = 0 if flag_offset is None else flag_offset
        self._sync()
        arrive_inc(self.lock_ptr, self.thread_idx, flag_offset, val)


# More explicit alias for call sites where plain ``Semaphore`` is ambiguous.
GlobalSemaphore = Semaphore


__all__ = [
    "Semaphore",
    "GlobalSemaphore",
    "wait_eq",
    "arrive_inc",
]
