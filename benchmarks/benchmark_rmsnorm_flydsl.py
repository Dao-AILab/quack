# ruff: noqa: I001

# Copyright (c) 2026, Tri Dao.

"""Benchmark the FlyDSL RMSNorm forward kernel against torch.compile."""

import argparse
import os

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch
from triton.testing import Benchmark, do_bench, perf_report

from quack.bench.bench_utils import run_and_print
from quack.rmsnorm_flydsl import rmsnorm_fwd


# Keep this ladder aligned with benchmarks/benchmark_rmsnorm.py so FlyDSL and
# CuTe reports cover the same shapes.
MN_PAIRS = [
    (32768, 256),
    (32768, 512),
    (32768, 1024),
    (32768, 2048),
    (32768, 4096),
    (32768, 8192),
    (32768, 16384),
    (32768, 32768),
    (32768, 65536),
    (16384, 131072),
    (8192, 262144),
]

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

EPS = 1e-6


def _rmsnorm_ref(x, weight, residual=None, eps=EPS):
    x_f32 = x.float()
    if residual is not None:
        x_f32 = x_f32 + residual.float()
    rstd = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
    out = (x_f32 * rstd * weight.float()).to(x.dtype)
    if residual is None:
        return out
    return out, x_f32.to(residual.dtype)


def _compiled_ref():
    # Dynamo's recompile budget is process-global across benchmark cells.
    torch._dynamo.reset()
    return torch.compile(_rmsnorm_ref, dynamic=False)


def _result(num_bytes: int, ms: float) -> dict:
    gbps = num_bytes / (ms / 1000) / 1e9
    return {"ms": round(ms, 4), "GB/s": round(gbps)}


def _weight_dtype(dtype_name: str, weight_dtype_name: str) -> torch.dtype:
    if weight_dtype_name == "same":
        return DTYPE_MAP[dtype_name]
    return DTYPE_MAP[weight_dtype_name]


def _memory_bytes(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None,
) -> int:
    # Logical I/O: read x and weight, write output; with residual, also read the
    # residual and write the pre-normalization sum returned by rmsnorm_fwd.
    num_bytes = 2 * x.numel() * x.element_size() + weight.numel() * weight.element_size()
    if residual is not None:
        num_bytes += 2 * residual.numel() * residual.element_size()
    return num_bytes


def make_benchmark(
    dtype_name: str,
    weight_dtype_name: str,
    use_residual: bool,
    x_vals=None,
) -> Benchmark:
    suffix = "-residual" if use_residual else ""
    return Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals if x_vals is not None else MN_PAIRS,
        line_arg="provider",
        line_vals=["flydsl", "torch_compile"],
        line_names=["flydsl", "torch.compile"],
        plot_name=(f"rmsnorm-flydsl-fwd-{dtype_name}-w-{weight_dtype_name}{suffix}"),
        args={
            "dtype_name": dtype_name,
            "weight_dtype_name": weight_dtype_name,
            "use_residual": use_residual,
        },
        xlabel="(M, N)",
        ylabel="GB/s",
    )


def rmsnorm_fwd_runner(
    M,
    N,
    provider,
    dtype_name,
    weight_dtype_name,
    use_residual,
):
    dtype = DTYPE_MAP[dtype_name]
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    weight = torch.randn(
        N,
        device="cuda",
        dtype=_weight_dtype(dtype_name, weight_dtype_name),
    )
    residual = torch.randn_like(x) if use_residual else None

    if provider == "flydsl":
        fn = lambda: rmsnorm_fwd(x, weight, residual=residual, eps=EPS)
    elif provider == "torch_compile":
        compiled = _compiled_ref()
        fn = lambda: compiled(x, weight, residual=residual, eps=EPS)
    else:
        raise ValueError(f"unknown provider: {provider}")

    ms = do_bench(fn, warmup=10, rep=100, return_mode="median")
    return _result(_memory_bytes(x, weight, residual), ms)


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlyDSL RMSNorm forward")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    parser.add_argument(
        "--weight_dtype",
        default="float32",
        choices=["same", *DTYPE_MAP],
        help="Weight dtype; 'same' follows --dtype",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Fuse a same-dtype residual add and return the pre-normalization sum",
    )
    parser.add_argument("--M", type=int, default=None, help="Bench a single M (requires --N)")
    parser.add_argument("--N", type=int, default=None, help="Bench a single N (requires --M)")
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    if (args.M is None) != (args.N is None):
        parser.error("--M and --N must be given together")
    if args.M is not None and (args.M <= 0 or args.N <= 0):
        parser.error("--M and --N must be positive")
    x_vals = [(args.M, args.N)] if args.M is not None else None

    torch.manual_seed(0)

    bench = perf_report(
        make_benchmark(
            args.dtype,
            args.weight_dtype,
            args.residual,
            x_vals,
        )
    )(rmsnorm_fwd_runner)
    run_and_print(bench, save_path=args.save_path)


if __name__ == "__main__":
    main()
