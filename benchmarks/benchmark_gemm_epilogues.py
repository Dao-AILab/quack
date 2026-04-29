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
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from quack.autotuner import _gpu_warmup
from quack.gemm_config import GemmConfig
from quack.gemm_interface import (
    _gemm_rms_tuned,
    gemm_act,
    gemm_act_tuned,
    gemm_norm_act,
    gemm_norm_act_tuned,
)
from quack.gemm_sq_reduce import gemm_sq_reduce


def make_config(args) -> GemmConfig:
    return GemmConfig(
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
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


def _config_dict(config: GemmConfig | None) -> dict | None:
    return None if config is None else asdict(config)


def _result_record(args, config: GemmConfig | None, result: dict) -> dict:
    device = torch.cuda.get_device_properties(0)
    samples = result.get("samples")
    return {
        "kernel": args.kernel,
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "dtype": str(args.dtype).removeprefix("torch."),
        "activation": args.activation,
        "profile": args.profile,
        "stat": args.stat,
        "runtime_ms": result.get("ms"),
        "samples_ms": samples,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "preheat_ms": args.preheat_ms,
        "use_selected_config": args.use_selected_config,
        "tuned": args.tuned,
        "dynamic_scheduler": args.dynamic_scheduler,
        "no_residual": args.no_residual,
        "no_norm_weight": args.no_norm_weight,
        "with_aux": args.with_aux,
        "config": _config_dict(config),
        "device": {
            "name": device.name,
            "capability": f"{device.major}.{device.minor}",
            "total_memory_gb": device.total_memory / 2**30,
        },
    }


def _write_json(path: str, record: dict) -> None:
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _write_csv(path: str, record: dict) -> None:
    flat = {
        "kernel": record["kernel"],
        "m": record["m"],
        "n": record["n"],
        "k": record["k"],
        "dtype": record["dtype"],
        "activation": record["activation"],
        "runtime_ms": record["runtime_ms"],
        "stat": record["stat"],
        "samples_ms": json.dumps(record["samples_ms"]),
        "profile": record["profile"],
        "use_selected_config": record["use_selected_config"],
        "tuned": record["tuned"],
        "dynamic_scheduler": record["dynamic_scheduler"],
        "preheat_ms": record["preheat_ms"],
        "device_name": record["device"]["name"],
        "device_capability": record["device"]["capability"],
        "config": json.dumps(record["config"], sort_keys=True),
    }
    path_obj = Path(path)
    write_header = not path_obj.exists() or path_obj.stat().st_size == 0
    with path_obj.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


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


def run_sq_reduce(args, config: GemmConfig):
    a = torch.randn(args.m, args.k, device="cuda", dtype=args.dtype)
    b = torch.randn(args.k, args.n, device="cuda", dtype=args.dtype) / math.sqrt(args.k)
    c = (
        torch.randn(args.m, args.n, device="cuda", dtype=args.dtype)
        if not args.no_residual
        else None
    )
    rowvec = (
        torch.randn(args.n, device="cuda", dtype=torch.float32) if not args.no_norm_weight else None
    )
    out = torch.empty(args.m, args.n, device="cuda", dtype=args.dtype)
    aux_out = (
        torch.empty(args.m, args.n, device="cuda", dtype=args.dtype) if args.with_aux else None
    )
    tile_m = config.tile_m if config is not None else args.tile_m
    tile_n = config.tile_n if config is not None else args.tile_n
    tile_k = config.tile_k if config is not None else args.tile_k
    cluster_m = config.cluster_m if config is not None else 1
    cluster_n = config.cluster_n if config is not None else 1
    pingpong = config.pingpong if config is not None else args.pingpong
    n_tiles = math.ceil(args.n / tile_n)
    colvec = torch.zeros(args.m, n_tiles, device="cuda", dtype=torch.float32)

    # gemm_sq_reduce wants (l, m, k)/(l, n, k) inputs; expand the leading batch dim of 1.
    A = a.unsqueeze(0)
    B = b.transpose(-1, -2).unsqueeze(0).contiguous()  # (1, n, k)
    D = out.unsqueeze(0)
    C = c.unsqueeze(0) if c is not None else None
    rowvec_3d = rowvec.unsqueeze(0) if rowvec is not None else None
    colvec_3d = colvec.unsqueeze(0)
    AuxOut = aux_out.unsqueeze(0) if aux_out is not None else None

    fn = lambda: gemm_sq_reduce(
        A,
        B,
        D,
        C,
        colvec_3d,
        None,
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=cluster_m,
        cluster_N=cluster_n,
        tile_K=tile_k,
        pingpong=pingpong,
        persistent=False,
        rowvec=rowvec_3d,
        aux_out=AuxOut,
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
    parser.add_argument("--kernel", choices=["rms", "norm_act", "act", "sq_reduce"], default="rms")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--activation", default="gelu_tanh_approx")
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--tile-k", type=int, default=None)
    parser.add_argument("--pingpong", action="store_true")
    parser.add_argument("--swap-ab", action="store_true")
    parser.add_argument("--dynamic-scheduler", action="store_true")
    parser.add_argument("--use-selected-config", action="store_true")
    parser.add_argument(
        "--tuned", action="store_true", help="When using selected config mode, call the tuned path"
    )
    parser.add_argument("--no-dynamic-persistent", action="store_true")
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--no-norm-weight", action="store_true")
    parser.add_argument(
        "--with-aux",
        action="store_true",
        help="(sq_reduce) also write the pre-rowvec output to mAuxOut",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--stat",
        choices=["min", "second-min", "median"],
        default="second-min",
        help="Statistic to report from repeated CUDA-event timings",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run one profiled launch after a small warmup"
    )
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument(
        "--preheat-ms",
        type=int,
        default=0,
        help="Optional GPU preheat duration before timing/profile to reduce thermal skew",
    )
    parser.add_argument(
        "--output-json",
        help="Write one machine-readable benchmark result record to this JSON file",
    )
    parser.add_argument(
        "--output-csv",
        help="Append one machine-readable benchmark result row to this CSV file",
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
    elif args.kernel == "sq_reduce":
        result = run_sq_reduce(args, config)
    else:
        result = run_act(args, config)

    if args.profile:
        print("  profile=completed")
    else:
        print(f"  stat={args.stat}")
        print(f"  samples_ms={[round(x, 3) for x in result['samples']]}")
        print(f"  runtime={result['ms']:.3f}ms")
    if args.output_json or args.output_csv:
        record = _result_record(args, config, result)
        if args.output_json:
            _write_json(args.output_json, record)
            print(f"  output_json={args.output_json}")
        if args.output_csv:
            _write_csv(args.output_csv, record)
            print(f"  output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
