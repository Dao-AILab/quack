import argparse
import os

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch
from triton.testing import Benchmark, do_bench, perf_report

import cutlass

from quack.bench.bench_utils import run_and_print
from quack.topk import topk, topk_bwd

try:
    import rtopk
except ImportError:
    rtopk = None


# (N, k) sweep — M is held fixed as a Benchmark.args entry. Skip pairs with k > N//2.
NK_PAIRS = [(n, k) for n in (64, 128, 256, 512, 1024) for k in (8, 16, 32) if k <= n // 2]

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _result(num_bytes: int, ms: float) -> dict:
    gbps = num_bytes / (ms / 1000) / 1e9
    return {"ms": round(ms, 4), "GB/s": round(gbps)}


def _fwd_providers():
    providers = [("quack", "quack"), ("torch", "torch.topk")]
    if rtopk is not None:
        providers.append(("rtopk", "rtopk"))
    return providers


def _bwd_providers():
    return [("quack", "quack"), ("torch", "torch")]


def make_fwd_benchmark(M: int, dtype_name: str, softmax: bool, x_vals=None) -> Benchmark:
    line_vals, line_names = zip(*_fwd_providers())
    suffix = dtype_name + (f"-M{M}" if x_vals is None else "") + ("-softmax" if softmax else "")
    return Benchmark(
        x_names=["N", "k"],
        x_vals=x_vals if x_vals is not None else NK_PAIRS,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"topk-fwd-{suffix}",
        args={"M": M, "dtype_name": dtype_name, "softmax": softmax},
        xlabel="(N, k)",
        ylabel="GB/s",
    )


def make_bwd_benchmark(M: int, dtype_name: str, softmax: bool, x_vals=None) -> Benchmark:
    line_vals, line_names = zip(*_bwd_providers())
    suffix = dtype_name + (f"-M{M}" if x_vals is None else "") + ("-softmax" if softmax else "")
    return Benchmark(
        x_names=["N", "k"],
        x_vals=x_vals if x_vals is not None else NK_PAIRS,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"topk-bwd-{suffix}",
        args={"M": M, "dtype_name": dtype_name, "softmax": softmax},
        xlabel="(N, k)",
        ylabel="GB/s",
    )


def topk_fwd_runner(N, k, provider, M, dtype_name, softmax):
    dtype = DTYPE_MAP[dtype_name]
    elem_bytes = dtype.itemsize
    x = torch.randn(M, N, device="cuda", dtype=dtype)

    if provider == "quack":
        fn = lambda: topk(x, k, softmax=softmax)
    elif provider == "torch":
        fn = lambda: torch.topk(x, k, dim=-1, largest=True, sorted=True)[0]
    elif provider == "rtopk":
        fn = lambda: rtopk.ops.rtopk(x, k, max_iter=512)
    else:
        raise ValueError(provider)

    ms = do_bench(fn, warmup=10, rep=100)
    # I/O: read x (M*N) + write values (M*k); ignore index output (small)
    nbytes = (M * N + M * k) * elem_bytes
    return _result(nbytes, ms)


def topk_bwd_runner(N, k, provider, M, dtype_name, softmax):
    dtype = DTYPE_MAP[dtype_name]
    elem_bytes = dtype.itemsize
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)

    if provider == "quack":
        out, idx = topk(x, k, softmax=softmax)
        dvalues = torch.randn_like(out)
        fn = lambda: topk_bwd(dvalues, out, idx, N, softmax=softmax)
        ms = do_bench(fn, warmup=10, rep=100, grad_to_none=(x,))
    elif provider == "torch":
        values = torch.topk(x, k, dim=-1, largest=True, sorted=True)[0]
        if softmax:
            values = torch.softmax(values, dim=-1)
        dvalues = torch.randn_like(values)
        fn = lambda: torch.autograd.grad(values, x, grad_outputs=dvalues, retain_graph=True)
        ms = do_bench(fn, warmup=10, rep=100, grad_to_none=(x,))
    else:
        raise ValueError(provider)

    # I/O: read x + read dvalues + write dx
    nbytes = (M * N + 2 * M * k) * elem_bytes
    return _result(nbytes, ms)


def main():
    parser = argparse.ArgumentParser(description="Benchmark top-k fwd / bwd")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    parser.add_argument("--M", default=65536, type=int, help="Held fixed during the (N, k) sweep")
    parser.add_argument("--N", default=None, type=int, help="Bench a single N (requires --k)")
    parser.add_argument("--k", default=None, type=int, help="Bench a single k (requires --N)")
    parser.add_argument("--softmax", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    if (args.N is None) != (args.k is None):
        parser.error("--N and --k must be given together")
    x_vals = [(args.N, args.k)] if args.N is not None else None

    torch.manual_seed(0)
    cutlass.cuda.initialize_cuda_context()

    if args.backward:
        bench = perf_report(make_bwd_benchmark(args.M, args.dtype, args.softmax, x_vals))(
            topk_bwd_runner
        )
    else:
        bench = perf_report(make_fwd_benchmark(args.M, args.dtype, args.softmax, x_vals))(
            topk_fwd_runner
        )

    run_and_print(bench, save_path=args.save_path)


if __name__ == "__main__":
    main()
