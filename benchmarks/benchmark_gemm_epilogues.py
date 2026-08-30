#!/usr/bin/env python3
"""Benchmark any @gemm_epilogue-minted mod, with explicit config overrides.

Designed for both wall-clock benchmarking and Nsight Compute profiling of
GEMM epilogue-heavy kernels. The mod is named by dotted path (``--mod
package.module:attr``); factory-minted mods are called inline in the
module's namespace (``--mod "quack.epilogue.library:norm_act_mod(activation='relu',
gated=False, has_c=False, has_rowvec=False, has_colvec=False)"``).

Buffers: A (m, k) and B (n, k) are randn (B scaled 1/sqrt(k)); reduce-sink
partials are auto-allocated f32 from the mod's declaration at the config's
tile shape, and aux outputs like D. Everything else — value-port host args,
scalars — is passed with ``--epi-arg name=dtype:shape[:fill]``, where shape
dims may be symbolic (m, n, k, l, tile_m, tile_n, m_tiles, n_tiles). A
plain-matmul baseline (same shapes, same output buffer) is timed alongside
for the epilogue-overhead ratio.

The production gemm_interface wrappers (gemm_rms, gemm_act, ...) are built
on these same mods; wrapper-level and autotuned benching lives in
benchmark_gemm_epi.py / benchmark_gemm_autotuned.py.

Examples:
    python benchmarks/benchmark_gemm_epilogues.py \\
        --mod quack.epilogue.scaled_exp:scaled_exp_epi \\
        --mnkl 4096,128256,1024,1 \\
        --tile_shape_mnk 128,256 --cluster_shape_mn 2,1 \\
        --epi-arg max_log2=tile_n
    python benchmarks/benchmark_gemm_epilogues.py \\
        --mod quack.epilogue.library:amax_epi --mnkl 8192,8192,4096 --tile_shape_mnk 128,256
    python benchmarks/benchmark_gemm_epilogues.py \\
        --mod quack.epilogue.library:qknorm_epi --mnkl 4096,2048,4096 \\
        --tile_shape_mnk 128,256 --epi-arg qk=f32:128
    ncu python benchmarks/benchmark_gemm_epilogues.py \\
        --mod quack.epilogue.library:relu_mod --profile --tile_shape_mnk 128,256
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

_EPI_ARG_DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i32": torch.int32,
    "i64": torch.int64,
}


def _parse_comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def make_config(args) -> GemmConfig:
    return GemmConfig(
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        pingpong=args.pingpong,
        cluster_m=args.cluster_m,
        cluster_n=args.cluster_n,
        cluster_k=1,
        swap_ab=False,
        max_swizzle_size=8,
        device_capacity=torch.cuda.get_device_capability()[0],
        is_dynamic_persistent=not args.no_dynamic_persistent,
        use_tma_gather=False,
    )


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


def _result_record(args, config: GemmConfig, result: dict) -> dict:
    device = torch.cuda.get_device_properties(0)
    return {
        "mod": args.mod,
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "l": args.l,
        "dtype": str(args.dtype).removeprefix("torch."),
        "out_dtype": args.out_dtype,
        "with_c": args.with_c,
        "epi_args": args.epi_arg,
        "profile": args.profile,
        "stat": args.stat,
        "runtime_ms": result.get("ms"),
        "baseline_ms": result.get("baseline_ms"),
        "samples_ms": result.get("samples"),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "preheat_ms": args.preheat_ms,
        "config": asdict(config),
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
        "mod": record["mod"],
        "m": record["m"],
        "n": record["n"],
        "k": record["k"],
        "l": record["l"],
        "dtype": record["dtype"],
        "epi_args": json.dumps(record["epi_args"]),
        "runtime_ms": record["runtime_ms"],
        "baseline_ms": record["baseline_ms"],
        "stat": record["stat"],
        "samples_ms": json.dumps(record["samples_ms"]),
        "profile": record["profile"],
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


def _load_mod(spec: str):
    """Import a @gemm_epilogue mod from 'package.module:attr'. An attr with a
    call, e.g. 'quack.epilogue.library:norm_act_mod(activation="relu", ...)', is
    evaluated in the module's namespace (factory-minted mods)."""
    import importlib

    modpath, sep, attr = spec.partition(":")
    if not sep:
        raise ValueError(f"--mod expects 'package.module:attr', got {spec!r}")
    module = importlib.import_module(modpath)
    if "(" in attr:
        return eval(attr, vars(module))
    return getattr(module, attr)


def _parse_epi_args(specs: list[str], dims: dict) -> dict:
    """name=dtype:shape[:fill] -> tensor (shape dims symbolic via ``dims``),
    or name=<number> -> python scalar."""
    out = {}
    for spec in specs:
        name, sep, val = spec.partition("=")
        if not sep:
            raise ValueError(f"--epi-arg expects name=value, got {spec!r}")
        head = val.split(":", 1)[0]
        if head not in _EPI_ARG_DTYPES:
            if val in dims:  # symbolic int, e.g. max_log2=tile_n
                out[name] = dims[val]
            else:
                out[name] = int(val) if val.lstrip("+-").isdigit() else float(val)
            continue
        parts = val.split(":")
        dtype = _EPI_ARG_DTYPES[parts[0]]
        shape = (
            tuple(dims[t] if t in dims else int(t) for t in parts[1].split(",") if t)
            if len(parts) > 1
            else ()
        )
        fill = parts[2] if len(parts) > 2 else "randn"
        if fill == "randn" and dtype.is_floating_point:
            out[name] = torch.randn(shape, device="cuda", dtype=dtype)
        elif fill == "randint":  # uniform over [0, N) — column-index args (targets)
            out[name] = torch.randint(0, dims["n"], shape, device="cuda", dtype=dtype)
        elif fill in ("zeros", "randn"):  # randn on int dtypes degrades to zeros
            out[name] = torch.zeros(shape, device="cuda", dtype=dtype)
        elif fill == "ones":
            out[name] = torch.ones(shape, device="cuda", dtype=dtype)
        elif fill == "empty":
            out[name] = torch.empty(shape, device="cuda", dtype=dtype)
        else:
            raise ValueError(f"--epi-arg fill must be randn|randint|zeros|ones|empty, got {fill!r}")
    return out


def run_mod(args, config: GemmConfig):
    mod = _load_mod(args.mod)
    tile_m, tile_n = config.tile_m, config.tile_n
    m_tiles, n_tiles = math.ceil(args.m / tile_m), math.ceil(args.n / tile_n)
    dims = dict(
        m=args.m,
        n=args.n,
        k=args.k,
        l=args.l,
        tile_m=tile_m,
        tile_n=tile_n,
        m_tiles=m_tiles,
        n_tiles=n_tiles,
    )
    lead = () if args.l is None else (args.l,)
    a = torch.randn(*lead, args.m, args.k, device="cuda", dtype=args.dtype)
    b = torch.randn(*lead, args.n, args.k, device="cuda", dtype=args.dtype) / math.sqrt(args.k)
    out_dtype = _EPI_ARG_DTYPES[args.out_dtype] if args.out_dtype else args.dtype
    d = torch.empty(*lead, args.m, args.n, device="cuda", dtype=out_dtype)
    c = torch.randn(*lead, args.m, args.n, device="cuda", dtype=args.dtype) if args.with_c else None

    epi_args = {}
    for name, op in mod.sinks.items():
        if getattr(op, "dim", None) is None:
            continue
        inner = (args.m, n_tiles) if op.dim == 0 else (m_tiles, args.n)
        epi_args[name] = torch.empty(*lead, *inner, device="cuda", dtype=torch.float32)
    for name in mod.outputs:
        epi_args[name] = torch.empty_like(d)
    epi_args.update(_parse_epi_args(args.epi_arg, dims))
    missing = [op.name for op in mod.extra_ops if op.name not in epi_args]
    if missing:
        print(f"  note: mod ops without an --epi-arg (fine if optional): {missing}")
    alloc = {k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in sorted(epi_args.items())}
    print(f"  epi_args={alloc}")

    fn = lambda: mod.gemm(
        a,
        b,
        d,
        c,
        epi_args=epi_args,
        tile_M=tile_m,
        tile_N=tile_n,
        tile_K=config.tile_k,
        cluster_M=config.cluster_m,
        cluster_N=config.cluster_n,
        pingpong=config.pingpong,
    )
    if args.profile:
        profile_once(fn, args.profile_warmup)
        return {"ms": None}
    fn()
    ms, samples = benchmark(fn, args.repeats, args.warmup, args.stat)
    result = {"ms": ms, "samples": samples}
    if not args.no_baseline:
        bt = b.mT  # same output buffer + data so the comparison is honest
        plain = lambda: torch.matmul(a, bt, out=d)
        base_ms, _ = benchmark(plain, args.repeats, args.warmup, args.stat)
        tflops = 2 * args.m * args.n * args.k * (args.l or 1) / (ms * 1e-3) / 1e12
        print(
            f"  plain_matmul={base_ms:.3f}ms  mod={ms:.3f}ms "
            f"({ms / base_ms:.3f}x of plain, {tflops:.0f} TFLOPS)"
        )
        result["baseline_ms"] = base_ms
    return result


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark @gemm_epilogue GEMM kernels")
    parser.add_argument(
        "--mod",
        required=True,
        help="dotted path of a @gemm_epilogue mod ('quack.epilogue.scaled_exp:scaled_exp_epi'), "
        "or a factory call evaluated in the module namespace "
        "(\"quack.epilogue.library:norm_act_mod(activation='relu', ...)\")",
    )
    parser.add_argument(
        "--epi-arg",
        action="append",
        default=[],
        metavar="NAME=DTYPE:SHAPE[:FILL]",
        help="extra epilogue operand: dtype f32|f16|bf16|i32|i64, comma shape with "
        "symbolic dims (m, n, k, l, tile_m, tile_n, m_tiles, n_tiles), fill "
        "randn|zeros|ones|empty; or a bare number for scalar operands. Repeatable.",
    )
    parser.add_argument(
        "--mnkl",
        type=_parse_comma_separated_ints,
        default=(4096, 4096, 4096),
        metavar="M,N,K[,L]",
        help="GEMM dimensions; omit L for an unbatched GEMM",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--out-dtype",
        choices=list(_EPI_ARG_DTYPES),
        default=None,
        help="D/aux-output dtype; defaults to --dtype",
    )
    parser.add_argument(
        "--with-c", action="store_true", help="pass a randn C residual through the C pipeline"
    )
    parser.add_argument(
        "--tile_shape_mnk",
        "--tile_shape_mn",
        dest="tile_shape_mnk",
        type=_parse_comma_separated_ints,
        default=(128, 256),
        metavar="M,N[,K]",
        help="CTA tile shape; omit K to use the kernel default",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        "--cluster_shape_mnk",
        dest="cluster_shape_mn",
        type=_parse_comma_separated_ints,
        default=(1, 1),
        metavar="M,N[,K]",
        help="CTA cluster shape; K may be omitted and must be 1 if provided",
    )
    parser.add_argument("--pingpong", action="store_true")
    parser.add_argument("--no-dynamic-persistent", action="store_true")
    parser.add_argument(
        "--no-baseline", action="store_true", help="skip the plain-matmul baseline timing"
    )
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
    args = parser.parse_args(argv)

    if len(args.mnkl) not in (3, 4):
        parser.error("--mnkl must contain exactly 3 or 4 values: M,N,K[,L]")
    if len(args.tile_shape_mnk) not in (2, 3):
        parser.error("--tile_shape_mnk must contain exactly 2 or 3 values: M,N[,K]")
    if len(args.cluster_shape_mn) not in (2, 3):
        parser.error("--cluster_shape_mn must contain exactly 2 or 3 values: M,N[,K]")
    if len(args.cluster_shape_mn) == 3 and args.cluster_shape_mn[2] != 1:
        parser.error("GEMM epilogues require cluster K to be 1")
    if any(dim <= 0 for dim in (*args.mnkl, *args.tile_shape_mnk, *args.cluster_shape_mn)):
        parser.error("GEMM, tile, and cluster dimensions must be positive")

    args.m, args.n, args.k = args.mnkl[:3]
    args.l = args.mnkl[3] if len(args.mnkl) == 4 else None
    args.tile_m, args.tile_n = args.tile_shape_mnk[:2]
    args.tile_k = args.tile_shape_mnk[2] if len(args.tile_shape_mnk) == 3 else None
    args.cluster_m, args.cluster_n = args.cluster_shape_mn[:2]
    return args


def main():
    args = parse_arguments()
    args.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    config = make_config(args)

    print("GEMM epilogue benchmark")
    print(f"  mod={args.mod}")
    print(
        f"  shape=({args.m}, {args.k}) x ({args.k}, {args.n})" + (f" l={args.l}" if args.l else "")
    )
    print(f"  dtype={args.dtype}")
    print(f"  config={config}")
    if args.preheat_ms > 0:
        print(f"  preheat_ms={args.preheat_ms}")
        _gpu_warmup(args.preheat_ms)

    result = run_mod(args, config)

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
