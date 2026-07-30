#!/usr/bin/env python3
"""Minimal example: intra-kernel IKET trace hooks in CuTe-DSL.

Run with IKET:
    python -m iket.cli.main --output-dir /tmp/quack_iket --clobber \
        profile --postprocess all -- \
        env PYTHONPATH=. python examples/example_iket_trace.py
"""

from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.experimental import iket

ITERS = 8
BLOCK_THREADS = 64
GRID = 2


@cute.kernel
def _trace_kernel(iters: Int32) -> None:
    iket.range_push("kernel_total")
    for _ in cutlass.range(iters):
        iket.range_push("loop_body")
        iket.mark("loop_mark")
        iket.range_pop()
    iket.range_pop()


@cute.jit
def _launch(iters: Int32) -> None:
    _trace_kernel(iters).launch(
        grid=(GRID, 1, 1),
        block=(BLOCK_THREADS, 1, 1),
    )


def main() -> None:
    torch.cuda.init()
    _launch(ITERS)
    torch.cuda.synchronize()
    print(f"ran IKET trace workload: grid={GRID}, block={BLOCK_THREADS}, iters={ITERS}")
    print("collect traces with iket.cli profile.")


if __name__ == "__main__":
    main()
