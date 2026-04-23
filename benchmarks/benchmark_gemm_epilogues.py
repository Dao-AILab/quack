#!/usr/bin/env python3
"""Benchmark GEMM epilogue kernels with explicit config overrides.

This script is designed for both wall-clock benchmarking and Nsight Compute
profiling of GEMM epilogue-heavy kernels.

Examples:
    python benchmarks/benchmark_gemm_epilogues.py --kernel rms
    python benchmarks/benchmark_gemm_epilogues.py --kernel norm_act --activation gelu_tanh_approx
    python benchmarks/benchmark_gemm_epilogues.py --kernel act --tile-m 64 --tile-n 128 --pingpong
    ncu python benchmarks/benchmark_gemm_epilogues.py --kernel rms --profile --tile-m 128 --tile-n 64
"""

import argparse
import math
import time

import torch

from quack.autotuner import _gpu_warmup
from quack.gemm_config import GemmConfig
from quack.gemm_interface import _gemm_rms_tuned, gemm_act, gemm_act_tuned, gemm_norm_act, gemm_norm_act_tuned


def make_config(args) -> GemmConfig:
    return GemmConfig(
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        pingpong=args.pingpong,
        cluster_m=1,
        cluster_n=1,
        swap_ab=args.swap_ab,
        max_swizzle_size=8,
        device_capacity=torch.cuda.get_device_capability()[0],
        is_dynamic_persistent=not args.no_dynamic_persistent,
        use_tma_gather=False,
    )


def selected_config(args) -> GemmConfig | None:
    return None if args.use_selected_config else make_config(args)


def benchmark(fn, repeats: int, warmup: int, stat: str) -> tuple[float, list[float]]:
    torch.cuda.synchronize()
    time.sleep(0.2)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    ordered = sorted(samples)
    if stat == "min":
        return ordered[0], samples
    if stat == "second-min":
        return ordered[1] if len(ordered) > 1 else ordered[0], samples
    if stat == "median":
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid], samples
        return 0.5 * (ordered[mid - 1] + ordered[mid]), samples
    raise ValueError(f"Unsupported stat: {stat}")


def profile_once(fn, warmup_launches: int) -> None:
    for _ in range(warmup_launches):
        fn()
    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        fn()
        torch.cuda.synchronize()
    finally:
        cudart.cudaProfilerStop()


def run_rms(args, config: GemmConfig):
    a = torch.randn(args.m, args.k, device="cuda", dtype=args.dtype)
    b = torch.randn(args.k, args.n, device="cuda", dtype=args.dtype) / math.sqrt(args.k)
    c = (
        torch.randn(args.m, args.n, device="cuda", dtype=args.dtype)
        if not args.no_residual
        else None
    )
    norm_weight = (
        torch.randn(args.n, device="cuda", dtype=args.dtype) if not args.no_norm_weight else None
    )
    out = torch.empty(args.m, args.n, device="cuda", dtype=args.dtype)

    if config is None:
        fn = lambda: _gemm_rms_tuned.fn(
            a,
            b,
            out,
            C=c,
            norm_weight=norm_weight,
            eps=args.eps,
            dynamic_scheduler=args.dynamic_scheduler,
            config=None,
        )
    else:
        fn = lambda: _gemm_rms_tuned.fn(
            a,
            b,
            out,
            C=c,
            norm_weight=norm_weight,
            eps=args.eps,
            dynamic_scheduler=args.dynamic_scheduler,
            config=config,
        )
    if args.profile:
        profile_once(fn, args.profile_warmup)
        return {"ms": None}
    fn()
    ms, samples = benchmark(fn, args.repeats, args.warmup, args.stat)
    return {"ms": ms, "samples": samples}


def run_norm_act(args, config: GemmConfig):
    a = torch.randn(args.m, args.k, device="cuda", dtype=args.dtype)
    b = torch.randn(args.k, args.n, device="cuda", dtype=args.dtype) / math.sqrt(args.k)
    c = (
        torch.randn(args.m, args.n, device="cuda", dtype=args.dtype)
        if not args.no_residual
        else None
    )
    rstd = torch.randn(args.m, device="cuda", dtype=torch.float32)
    preact_out = torch.empty(args.m, args.n, device="cuda", dtype=args.dtype)
    postact_out = torch.empty(args.m, args.n, device="cuda", dtype=args.dtype)

    if config is None:
        fn = lambda: gemm_norm_act(
            a,
            b,
            C=c,
            rstd=rstd,
            activation=args.activation,
            preact_out=preact_out,
            postact_out=postact_out,
            dynamic_scheduler=args.dynamic_scheduler,
            tuned=args.tuned,
        )
    else:
        fn = lambda: gemm_norm_act_tuned.fn(
            a,
            b,
            preact_out,
            postact_out,
            C=c,
            rstd=rstd,
            activation=args.activation,
            dynamic_scheduler=args.dynamic_scheduler,
            config=config,
        )
    if args.profile:
        profile_once(fn, args.profile_warmup)
        return {"ms": None}
    fn()
    ms, samples = benchmark(fn, args.repeats, args.warmup, args.stat)
    return {"ms": ms, "samples": samples}


def run_act(args, config: GemmConfig):
    a = torch.randn(args.m, args.k, device="cuda", dtype=args.dtype)
    b = torch.randn(args.k, args.n, device="cuda", dtype=args.dtype) / math.sqrt(args.k)
    c = (
        torch.randn(args.m, args.n, device="cuda", dtype=args.dtype)
        if not args.no_residual
        else None
    )
    preact_out = torch.empty(args.m, args.n, device="cuda", dtype=args.dtype)
    postact_out = torch.empty(args.m, args.n, device="cuda", dtype=args.dtype)

    if config is None:
        fn = lambda: gemm_act(
            a,
            b,
            C=c,
            activation=args.activation,
            preact_out=preact_out,
            postact_out=postact_out,
            dynamic_scheduler=args.dynamic_scheduler,
            tuned=args.tuned,
        )
    else:
        fn = lambda: gemm_act_tuned.fn(
            a,
            b,
            preact_out,
            postact_out,
            C=c,
            activation=args.activation,
            dynamic_scheduler=args.dynamic_scheduler,
            config=config,
        )
    if args.profile:
        profile_once(fn, args.profile_warmup)
        return {"ms": None}
    fn()
    ms, samples = benchmark(fn, args.repeats, args.warmup, args.stat)
    return {"ms": ms, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description="Benchmark GEMM epilogue kernels")
    parser.add_argument("--kernel", choices=["rms", "norm_act", "act"], default="rms")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--activation", default="gelu_tanh_approx")
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--pingpong", action="store_true")
    parser.add_argument("--swap-ab", action="store_true")
    parser.add_argument("--dynamic-scheduler", action="store_true")
    parser.add_argument("--use-selected-config", action="store_true")
    parser.add_argument("--tuned", action="store_true", help="When using selected config mode, call the tuned path")
    parser.add_argument("--no-dynamic-persistent", action="store_true")
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--no-norm-weight", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--stat",
        choices=["min", "second-min", "median"],
        default="second-min",
        help="Statistic to report from repeated CUDA-event timings",
    )
    parser.add_argument("--profile", action="store_true", help="Run one profiled launch after a small warmup")
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument(
        "--preheat-ms",
        type=int,
        default=0,
        help="Optional GPU preheat duration before timing/profile to reduce thermal skew",
    )
    args = parser.parse_args()

    args.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    config = selected_config(args)

    print("GEMM epilogue benchmark")
    print(f"  kernel={args.kernel}")
    print(f"  shape=({args.m}, {args.k}) x ({args.k}, {args.n})")
    print(f"  dtype={args.dtype}")
    if config is None:
        print(f"  config=selected-by-{'autotune' if args.tuned else 'default'}")
    else:
        print(f"  config={config}")
    if args.preheat_ms > 0:
        print(f"  preheat_ms={args.preheat_ms}")
        _gpu_warmup(args.preheat_ms)

    if args.kernel == "rms":
        result = run_rms(args, config)
    elif args.kernel == "norm_act":
        result = run_norm_act(args, config)
    else:
        result = run_act(args, config)

    if args.profile:
        print("  profile=completed")
    else:
        print(f"  stat={args.stat}")
        print(f"  samples_ms={[round(x, 3) for x in result['samples']]}")
        print(f"  runtime={result['ms']:.3f}ms")


if __name__ == "__main__":
    main()
