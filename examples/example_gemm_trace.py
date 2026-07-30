#!/usr/bin/env python3
"""Trace an SM90 GEMM kernel.

Run with IKET:
    python -m iket.cli.main --output-dir /tmp/quack_gemm_iket --clobber \
        --context-buffer-size 512M profile --postprocess all --keep \
        --max-ts-cnt-per-warp 8192 -- \
        env PYTHONPATH=. python examples/example_gemm_trace.py
"""

import torch

from quack.gemm import gemm

M, N, K = 4096, 4096, 4096
TILE_M, TILE_N = 128, 192
CLUSTER_M, CLUSTER_N = 2, 1


def main():
    A = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(1, N, K, device="cuda", dtype=torch.float16)
    D = torch.empty(1, M, N, device="cuda", dtype=torch.float16)

    gemm(
        A,
        B,
        D,
        C=None,
        tile_count_semaphore=None,
        tile_M=TILE_M,
        tile_N=TILE_N,
        cluster_M=CLUSTER_M,
        cluster_N=CLUSTER_N,
        persistent=True,
        pingpong=True,
    )

    # Verify correctness.
    ref = A[0] @ B[0].T
    print(f"max error: {(D[0] - ref).abs().max().item():.4f}")
    print("collect traces with iket.cli profile.")


if __name__ == "__main__":
    main()
