#!/usr/bin/env python3
"""Demo: autotuned GEMM with parallel pre-compilation.

Shows how the autotuner pre-compiles all tile configs in parallel via
persistent subprocess workers, then benchmarks each to pick the fastest.

Usage:
    # Default 8192x8192x8192 with verbose output
    python benchmarks/benchmark_gemm_autotuned.py

    # Custom size
    python benchmarks/benchmark_gemm_autotuned.py --M 8192 --N 8192 --K 4096

    # Control worker count
    QUACK_COMPILE_WORKERS=8 python benchmarks/benchmark_gemm_autotuned.py

    # Cold start: clear .so + autotuning caches to see parallel pre-compilation
    python benchmarks/benchmark_gemm_autotuned.py --cold

Environment variables:
    QUACK_COMPILE_WORKERS   Number of parallel compile workers (default: 4)
    QUACK_PRINT_AUTOTUNING  Set to "1" for verbose autotuner output (always on in this script)
"""

import argparse
import math
import os
import shutil
import time

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from quack.autotuner import default_cache_dir
from quack.cache import get_cache_path
from quack.gemm_config import GemmConfig
from quack.gemm_interface import (
    act_to_pytorch_fn_map,
    gemm,
    gemm_act,
    gemm_act_tuned,
    gemm_dgated,
    gemm_dgated_tuned,
    gemm_tuned,
    gated_to_pytorch_fn_map,
)


def clear_caches():
    """Clear both the .so kernel cache and the autotuning result cache."""
    # .so kernel cache (from quack.cache.get_cache_path)
    so_cache = str(get_cache_path())
    if os.path.isdir(so_cache):
        shutil.rmtree(so_cache)
        print(f"Cleared .so cache: {so_cache}")
    # Autotuning result cache (from autotuner.default_cache_dir)
    autotune_cache = str(default_cache_dir())
    if os.path.isdir(autotune_cache):
        shutil.rmtree(autotune_cache)
        print(f"Cleared autotuning cache: {autotune_cache}")


def tflops(flops, ms):
    return flops / (ms * 1e9)


def benchmark_gemm(m, n, k, dtype=torch.bfloat16, repeats=30, config=None):
    """Benchmark plain GEMM: out = A @ B"""
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    # gemm() expects B as (K, N)
    b = torch.randn(k, n, device="cuda", dtype=dtype) / math.sqrt(k)
    nflops = 2 * m * n * k
    nbytes = (a.numel() + b.numel() + m * n) * dtype.itemsize

    if config is None:
        fn = lambda: gemm(a, b, out_dtype=dtype)
    else:
        out = torch.empty(m, n, device="cuda", dtype=dtype)
        fn = lambda: gemm_tuned.fn(a, b, out, config=config)
    fn()  # warmup / autotune
    time.sleep(0.5)
    ms = do_bench(fn, warmup=5, rep=repeats)
    tf = tflops(nflops, ms)
    gbps = nbytes / (ms * 1e6)

    w = b.T.contiguous()  # (N, K) for F.linear
    time.sleep(0.5)
    ms_pt = do_bench(lambda: F.linear(a, w), warmup=5, rep=repeats)
    tf_pt = tflops(nflops, ms_pt)

    print(f"  quack: {ms:.3f}ms  {tf:.1f} TFLOPS  {gbps:.0f} GB/s")
    print(f"  cuBLAS: {ms_pt:.3f}ms  {tf_pt:.1f} TFLOPS")
    print(f"  speedup: {ms_pt / ms:.2f}x")
    return ms, tf


def _torch_gated_act(gated_fn, x, w):
    """Reference: GEMM + gated activation. x @ w.T has 2*N columns, split into gate/up."""
    preact = F.linear(x, w)
    gate = preact[..., ::2]
    up = preact[..., 1::2]
    return gated_fn(gate, up)


def _dsilu_exp(x):
    sigmoid_x = torch.sigmoid(x)
    silu_x = x * sigmoid_x
    return sigmoid_x + silu_x * (1.0 - sigmoid_x), silu_x


def _dsilu_tanh(x):
    x_half = 0.5 * x
    tanh_x_half = torch.tanh(x_half)
    sigmoid_x = 0.5 * tanh_x_half + 0.5
    silu_x = x_half * tanh_x_half + x_half
    return sigmoid_x + silu_x * (1.0 - sigmoid_x), silu_x


def _torch_dgated_act(dact_fn, x, w, preact):
    """Reference: GEMM + gated activation backward over interleaved gate/up preact."""
    dout = F.linear(x, w)
    gate = preact[..., ::2]
    up = preact[..., 1::2]
    dsilu, silu = dact_fn(gate)
    dgate = dout * up * dsilu
    dup = dout * silu
    dx = torch.stack((dgate, dup), dim=-1).flatten(-2)
    postact = silu * up
    return dx, postact


_dgated_act_fns = {
    "swiglu": _dsilu_exp,
    "swiglu-tanh": _dsilu_tanh,
}


def benchmark_gemm_act(
    m,
    n,
    k,
    activation="gelu_tanh_approx",
    dtype=torch.bfloat16,
    repeats=30,
    tuned=True,
    config=None,
):
    """Benchmark fused GEMM + activation (supports both regular and gated activations)."""
    is_gated = activation in gated_to_pytorch_fn_map
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    # For gated: B is (K, 2*N) with interleaved gate/up; output is (M, N)
    b_n = 2 * n if is_gated else n
    b = torch.randn(k, b_n, device="cuda", dtype=dtype) / math.sqrt(k)
    nflops = 2 * m * b_n * k

    if config is None:
        fn = lambda: gemm_act(a, b, activation=activation, out_dtype=dtype, tuned=tuned)
    else:
        preact = torch.empty(m, b_n, device="cuda", dtype=dtype)
        postact = torch.empty(m, n if is_gated else b_n, device="cuda", dtype=dtype)
        fn = lambda: gemm_act_tuned.fn(
            a,
            b,
            preact,
            postact,
            activation=activation,
            config=config,
        )
    fn()  # warmup / autotune
    time.sleep(0.5)
    ms = do_bench(fn, warmup=5, rep=repeats)
    tf = tflops(nflops, ms)

    # Baseline: torch.compile(GEMM + activation)
    w = b.T.contiguous()  # (N or 2*N, K) for F.linear
    if is_gated:
        ref_fn = torch.compile(lambda: _torch_gated_act(gated_to_pytorch_fn_map[activation], a, w))
    else:
        act_fn = act_to_pytorch_fn_map[activation]
        ref_fn = torch.compile(lambda: act_fn(F.linear(a, w)))
    ref_fn()  # compile warmup
    ref_fn()
    time.sleep(0.5)
    ms_pt = do_bench(ref_fn, warmup=5, rep=repeats)
    tf_pt = tflops(nflops, ms_pt)

    print(f"  quack: {ms:.3f}ms  {tf:.1f} TFLOPS")
    print(f"  cuBLAS + torch.compile: {ms_pt:.3f}ms  {tf_pt:.1f} TFLOPS")
    print(f"  speedup: {ms_pt / ms:.2f}x")
    return ms, tf


def benchmark_gemm_dgated(
    m,
    n,
    k,
    activation="swiglu",
    dtype=torch.bfloat16,
    repeats=30,
    tuned=True,
    config=None,
):
    """Benchmark fused GEMM + gated activation backward."""
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype) / math.sqrt(k)
    preact = torch.randn(m, 2 * n, device="cuda", dtype=dtype)
    nflops = 2 * m * n * k

    if config is None:
        fn = lambda: gemm_dgated(a, b, preact, activation=activation, out_dtype=dtype, tuned=tuned)
    else:
        dx_out = torch.empty(m, 2 * n, device="cuda", dtype=dtype)
        postact_out = torch.empty(m, n, device="cuda", dtype=preact.dtype)
        fn = lambda: gemm_dgated_tuned.fn(
            a,
            b,
            preact,
            dx_out,
            postact_out,
            None,
            activation,
            False,
            None,
            None,
            True,
            config=config,
        )
    fn()  # warmup / autotune
    time.sleep(0.5)
    ms = do_bench(fn, warmup=5, rep=repeats)
    tf = tflops(nflops, ms)

    w = b.T.contiguous()
    ref_fn = torch.compile(lambda: _torch_dgated_act(_dgated_act_fns[activation], a, w, preact))
    ref_fn()
    ref_fn()
    time.sleep(0.5)
    ms_pt = do_bench(ref_fn, warmup=5, rep=repeats)
    tf_pt = tflops(nflops, ms_pt)

    print(f"  quack: {ms:.3f}ms  {tf:.1f} TFLOPS")
    print(f"  cuBLAS + torch.compile: {ms_pt:.3f}ms  {tf_pt:.1f} TFLOPS")
    print(f"  speedup: {ms_pt / ms:.2f}x")
    return ms, tf


def forced_config_from_args(args):
    if args.config_tile_n is None:
        return None
    return GemmConfig(
        tile_m=args.config_tile_m,
        tile_n=args.config_tile_n,
        tile_k=args.config_tile_k,
        num_warps=args.config_num_warps,
        cluster_m=args.config_cluster_m,
        cluster_n=args.config_cluster_n,
        pingpong=args.config_pingpong,
        swap_ab=args.config_swap_ab,
        is_dynamic_persistent=False,
        device_capacity=torch.cuda.get_device_capability()[0],
    )


def main():
    parser = argparse.ArgumentParser(description="Demo autotuned GEMM with parallel compilation")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--dim", type=int, default=4096, help="Model dim for transformer shapes")
    parser.add_argument("--batch", type=int, default=8192, help="Batch size for transformer shapes")
    parser.add_argument(
        "--only-gated",
        action="store_true",
        help="Only run the transformer FFN gated GEMM benchmark",
    )
    parser.add_argument(
        "--gated-activation",
        choices=sorted(gated_to_pytorch_fn_map),
        default=None,
        help="Restrict the FFN gated benchmark to one activation",
    )
    parser.add_argument(
        "--only-dgated",
        action="store_true",
        help="Only run the transformer FFN gated backward GEMM benchmark",
    )
    parser.add_argument(
        "--dgated-activation",
        choices=sorted(_dgated_act_fns),
        default=None,
        help="Restrict the FFN gated backward benchmark to one activation",
    )
    parser.add_argument(
        "--untuned",
        action="store_true",
        help="Use the default GEMM config instead of autotuning activation kernels",
    )
    parser.add_argument("--config-tile-m", type=int, default=128)
    parser.add_argument("--config-tile-n", type=int, default=None)
    parser.add_argument("--config-tile-k", type=int, default=None)
    parser.add_argument("--config-num-warps", type=int, choices=[4, 8], default=None)
    parser.add_argument("--config-cluster-m", type=int, default=2)
    parser.add_argument("--config-cluster-n", type=int, default=1)
    parser.add_argument("--config-pingpong", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--config-swap-ab", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cold", action="store_true", help="Clear .so and autotuning caches first")
    args = parser.parse_args()

    if args.cold:
        clear_caches()

    os.environ["QUACK_PRINT_AUTOTUNING"] = "1"
    os.environ["QUACK_FORCE_CACHE_UPDATE"] = "1"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    M, N, K = args.M, args.N, args.K
    gated_activations = (
        [args.gated_activation]
        if args.gated_activation
        else [
            "swiglu",
            "swiglu-tanh",
        ]
    )
    dgated_activations = (
        [args.dgated_activation]
        if args.dgated_activation
        else [
            "swiglu",
            "swiglu-tanh",
        ]
    )
    forced_config = forced_config_from_args(args)
    ffn = int(args.dim * 3.5)  # Llama-3 ratio

    if args.only_gated and args.only_dgated:
        raise ValueError("--only-gated and --only-dgated are mutually exclusive")

    if args.only_gated:
        print(
            f"GEMM gated activation benchmark  (workers={os.environ.get('QUACK_COMPILE_WORKERS', '4')})"
        )
        print(f"  batch={args.batch}, dim={args.dim}, ffn={ffn}, dtype={args.dtype}")
        if forced_config is not None:
            print(f"  forced config: {forced_config}")
        for activation in gated_activations:
            print(
                f"\n  FFN up + {activation}: ({args.batch}, {args.dim}) x ({args.dim}, {2 * ffn})"
            )
            benchmark_gemm_act(
                args.batch,
                ffn,
                args.dim,
                activation,
                dtype,
                repeats=args.repeats,
                tuned=not args.untuned and forced_config is None,
                config=forced_config,
            )
        return

    if args.only_dgated:
        print(
            f"GEMM gated backward benchmark  (workers={os.environ.get('QUACK_COMPILE_WORKERS', '4')})"
        )
        print(f"  batch={args.batch}, dim={args.dim}, ffn={ffn}, dtype={args.dtype}")
        if forced_config is not None:
            print(f"  forced config: {forced_config}")
        for activation in dgated_activations:
            print(
                f"\n  FFN dSwiGLU + {activation}: ({args.batch}, {args.dim}) x ({args.dim}, {ffn})"
            )
            benchmark_gemm_dgated(
                args.batch,
                ffn,
                args.dim,
                activation,
                dtype,
                repeats=args.repeats,
                tuned=not args.untuned and forced_config is None,
                config=forced_config,
            )
        return

    print(f"GEMM autotuning demo  (workers={os.environ.get('QUACK_COMPILE_WORKERS', '4')})")
    print(f"  M={M}, N={N}, K={K}, dtype={args.dtype}")
    if forced_config is not None:
        print(f"  forced config: {forced_config}")
    print()

    # --- 1. Plain GEMM ---
    print("=" * 60)
    print(f"GEMM: ({M}, {K}) x ({K}, {N})")
    print("=" * 60)
    benchmark_gemm(M, N, K, dtype, repeats=args.repeats, config=forced_config)
    print()

    # --- 2. GEMM + activation ---
    print("=" * 60)
    print(f"GEMM + GeLU: ({M}, {K}) x ({K}, {N})")
    print("=" * 60)
    benchmark_gemm_act(
        M,
        N,
        K,
        "gelu_tanh_approx",
        dtype,
        repeats=args.repeats,
        tuned=not args.untuned and forced_config is None,
        config=forced_config,
    )
    print()

    # --- 3. Transformer-relevant shapes ---
    batch = args.batch
    dim = args.dim
    head_dim = 128
    n_q_heads = dim // head_dim
    n_kv_heads = n_q_heads // 4  # GQA
    qkv_dim = (n_q_heads + 2 * n_kv_heads) * head_dim

    print("=" * 60)
    print(f"Transformer shapes (batch={batch}, dim={dim})")
    print("=" * 60)
    gemm_shapes = [
        ("QKV proj", batch, qkv_dim, dim),
        ("Attn out", batch, dim, dim),
        ("FFN down", batch, dim, ffn),
    ]
    for label, m, n, k in gemm_shapes:
        print(f"\n  {label}: ({m}, {k}) x ({k}, {n})")
        benchmark_gemm(m, n, k, dtype, repeats=args.repeats, config=forced_config)

    # FFN up: fused GEMM + gated activation — B is (K, 2*ffn), output is (M, ffn)
    for activation in gated_activations:
        print(f"\n  FFN up + {activation}: ({batch}, {dim}) x ({dim}, {2 * ffn})")
        benchmark_gemm_act(
            batch,
            ffn,
            dim,
            activation,
            dtype,
            repeats=args.repeats,
            tuned=not args.untuned and forced_config is None,
            config=forced_config,
        )


if __name__ == "__main__":
    main()
