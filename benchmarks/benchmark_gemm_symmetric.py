#!/usr/bin/env python3
"""Benchmark symmetric GEMM: out = A @ A^T, with guaranteed symmetric output.

The symmetric GEMM only computes the upper triangle and mirrors, so it should
be ~2x faster than a full GEMM for compute-bound sizes.

Compares quack gemm_symmetric against cuBLAS (torch.matmul / torch.bmm).

Usage:
    python benchmarks/benchmark_gemm_symmetric.py
    python benchmarks/benchmark_gemm_symmetric.py --M 8192 --K 4096
    python benchmarks/benchmark_gemm_symmetric.py --M 4096 --K 4096 --L 4
"""

import argparse
import math
import time

import torch
from triton.testing import do_bench

from quack.gemm_interface import gemm_symmetric


def tflops(flops, ms):
    return flops / (ms * 1e9)


def benchmark_symmetric(M, K, L=1, dtype=torch.bfloat16, repeats=30):
    """Benchmark symmetric GEMM: out = A @ A^T"""
    torch.manual_seed(42)
    if L == 1:
        A = torch.randn(M, K, device="cuda", dtype=dtype) / math.sqrt(K)
        B = A.T.contiguous()
    else:
        A = torch.randn(L, M, K, device="cuda", dtype=dtype) / math.sqrt(K)
        B = A.transpose(-2, -1).contiguous()

    nflops = 2 * M * M * K * L
    nbytes = (A.numel() + A.numel() + M * M * L) * dtype.itemsize

    # Warmup / compile
    out = gemm_symmetric(A, B)
    torch.cuda.synchronize()

    # Correctness check
    if L == 1:
        ref = A.float() @ A.float().T
    else:
        ref = torch.bmm(A.float(), A.float().transpose(-2, -1))
    max_err = (out.float() - ref).abs().max().item()
    sym_err = (out - out.mT).abs().max().item()
    print(f"  correctness: max_err={max_err:.6f}, symmetry_err={sym_err}")

    fn = lambda: gemm_symmetric(A, B)
    time.sleep(0.5)
    ms = do_bench(fn, warmup=5, rep=repeats)
    tf = tflops(nflops, ms)
    gbps = nbytes / (ms * 1e6)

    # cuBLAS baseline (same dtype for fair comparison)
    if L == 1:
        fn_cublas = lambda: torch.matmul(A, A.T)
    else:
        fn_cublas = lambda: torch.bmm(A, A.transpose(-2, -1))
    time.sleep(0.5)
    ms_pt = do_bench(fn_cublas, warmup=5, rep=repeats)
    tf_pt = tflops(nflops, ms_pt)

    print(f"  quack:  {ms:.3f}ms  {tf:.1f} TFLOPS  {gbps:.0f} GB/s")
    print(f"  cuBLAS: {ms_pt:.3f}ms  {tf_pt:.1f} TFLOPS")
    print(f"  speedup: {ms_pt / ms:.2f}x")
    return ms, tf


def main():
    parser = argparse.ArgumentParser(description="Benchmark symmetric GEMM")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--L", type=int, default=1, help="Batch size")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    shapes = [
        (args.M, args.K, args.L),
        (2048, 2048, 1),
        (4096, 4096, 1),
        (8192, 8192, 1),
        (4096, 4096, 4),
    ]
    # Deduplicate while preserving order
    seen = set()
    unique_shapes = []
    for s in shapes:
        if s not in seen:
            seen.add(s)
            unique_shapes.append(s)

    print(f"Symmetric GEMM benchmark (dtype={args.dtype})")
    print()

    for M, K, L in unique_shapes:
        label = f"({M}, {K})" if L == 1 else f"({L}, {M}, {K})"
        print("=" * 60)
        print(f"A @ A^T: A={label}")
        print("=" * 60)
        benchmark_symmetric(M, K, L, dtype, repeats=args.repeats)
        print()


if __name__ == "__main__":
    main()
