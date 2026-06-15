import argparse
import os

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch
from triton.testing import Benchmark, do_bench, perf_report

from quack.bench.bench_utils import run_and_print
from quack.rmsnorm import layernorm_fwd, layernorm_ref


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


def _result(num_bytes: int, ms: float) -> dict:
    gbps = num_bytes / (ms / 1000) / 1e9
    return {"ms": round(ms, 4), "GB/s": round(gbps)}


def _providers():
    return [("quack", "quack"), ("torch_compile", "torch.compile")]


def make_fwd_benchmark(dtype_name: str, x_vals=None) -> Benchmark:
    line_vals, line_names = zip(*_providers())
    return Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals if x_vals is not None else MN_PAIRS,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"layernorm-fwd-{dtype_name}",
        args={"dtype_name": dtype_name},
        xlabel="(M, N)",
        ylabel="GB/s",
    )


def layernorm_fwd_runner(M, N, provider, dtype_name):
    dtype = DTYPE_MAP[dtype_name]
    eps = 1e-6

    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=torch.float32)

    if provider == "quack":
        fn = lambda: layernorm_fwd(x, w, eps=eps)
    elif provider == "torch_compile":
        compiled = torch.compile(layernorm_ref)
        fn = lambda: compiled(x, w, eps=eps)
    else:
        raise ValueError(provider)

    ms = do_bench(fn, warmup=10, rep=100)
    # I/O: read x + write y + read w
    nbytes = 2 * x.numel() * x.dtype.itemsize + w.numel() * 4
    return _result(nbytes, ms)


def main():
    parser = argparse.ArgumentParser(description="Benchmark layernorm fwd")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    parser.add_argument("--M", type=int, default=None, help="Bench a single M (requires --N)")
    parser.add_argument("--N", type=int, default=None, help="Bench a single N (requires --M)")
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    if (args.M is None) != (args.N is None):
        parser.error("--M and --N must be given together")
    x_vals = [(args.M, args.N)] if args.M is not None else None

    torch.manual_seed(0)

    bench = perf_report(make_fwd_benchmark(args.dtype, x_vals))(layernorm_fwd_runner)
    run_and_print(bench, save_path=args.save_path)


if __name__ == "__main__":
    main()
