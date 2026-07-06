import argparse
import os

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch
import torch.nn.functional as F
from triton.testing import Benchmark, do_bench, perf_report

import cutlass
import cutlass.torch as cutlass_torch

from quack.bench.bench_utils import run_and_print
from quack.softmax import softmax

try:
    from liger_kernel.transformers.functional import liger_softmax
except ImportError:
    liger_softmax = None


# (M, N) pairs: keep M fixed up to N=64K, then shrink M for very large N
# to keep total elements bounded.
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
    "bfloat16": cutlass.BFloat16,
    "float16": cutlass.Float16,
    "float32": cutlass.Float32,
}


def _result(numel_rw: int, elem_bytes: int, ms: float) -> dict:
    # GB/s given total elements transferred (read+write) and bytes/elem
    gbps = numel_rw * elem_bytes / (ms / 1000) / 1e9
    return {"ms": round(ms, 4), "GB/s": round(gbps)}


def _fwd_providers():
    providers = [
        ("quack", "quack"),
        ("torch_compile", "torch.compile"),
    ]
    if liger_softmax is not None:
        providers.append(("liger", "liger"))
    return providers


def _bwd_providers():
    return [
        ("quack", "quack"),
        ("torch_compile", "torch.compile"),
    ]


def make_fwd_benchmark(dtype_name: str, x_vals=None) -> Benchmark:
    line_vals, line_names = zip(*_fwd_providers())
    return Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals if x_vals is not None else MN_PAIRS,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"softmax-fwd-{dtype_name}",
        args={"dtype_name": dtype_name},
        xlabel="(M, N)",
        ylabel="GB/s",
    )


def make_bwd_benchmark(dtype_name: str, x_vals=None) -> Benchmark:
    line_vals, line_names = zip(*_bwd_providers())
    return Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals if x_vals is not None else MN_PAIRS,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"softmax-bwd-{dtype_name}",
        args={"dtype_name": dtype_name},
        xlabel="(M, N)",
        ylabel="GB/s",
    )


def softmax_fwd_runner(M, N, provider, dtype_name):
    dtype = DTYPE_MAP[dtype_name]
    torch_dtype = cutlass_torch.dtype(dtype)
    elem_bytes = dtype.width // 8

    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype)

    if provider == "quack":
        fn = lambda: softmax(x)
    elif provider == "torch_compile":
        compiled = torch.compile(lambda x: F.softmax(x, dim=-1))
        fn = lambda: compiled(x)
    elif provider == "liger":
        fn = lambda: liger_softmax(x)
    else:
        raise ValueError(provider)

    ms = do_bench(fn, warmup=10, rep=100)
    # I/O: read x + write y
    return _result(2 * x.numel(), elem_bytes, ms)


def softmax_bwd_runner(M, N, provider, dtype_name):
    dtype = DTYPE_MAP[dtype_name]
    torch_dtype = cutlass_torch.dtype(dtype)
    elem_bytes = dtype.width // 8

    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    if provider == "quack":
        y = softmax(x)
        dy = torch.randn_like(y)
        fn = lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)
        grad_to_none = (x,)
    elif provider == "torch_compile":
        y_ref = F.softmax(x_ref, dim=-1)
        dy = torch.randn_like(y_ref)
        fn = torch.compile(
            lambda: torch.autograd.grad(y_ref, x_ref, grad_outputs=dy, retain_graph=True)
        )
        grad_to_none = (x_ref,)
    else:
        raise ValueError(provider)

    ms = do_bench(fn, warmup=10, rep=100, grad_to_none=grad_to_none)

    # I/O: read y + read dy + write dx
    return _result(3 * x.numel(), elem_bytes, ms)


def main():
    parser = argparse.ArgumentParser(description="Benchmark softmax fwd / bwd")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--M", type=int, default=None, help="Bench a single M (requires --N)")
    parser.add_argument("--N", type=int, default=None, help="Bench a single N (requires --M)")
    parser.add_argument("--save_path", default=None, help="Directory to save CSV results")
    args = parser.parse_args()

    if (args.M is None) != (args.N is None):
        parser.error("--M and --N must be given together")
    x_vals = [(args.M, args.N)] if args.M is not None else None

    torch.manual_seed(0)

    if args.backward:
        bench = perf_report(make_bwd_benchmark(args.dtype, x_vals))(softmax_bwd_runner)
    else:
        bench = perf_report(make_fwd_benchmark(args.dtype, x_vals))(softmax_fwd_runner)

    run_and_print(bench, save_path=args.save_path)


if __name__ == "__main__":
    main()
