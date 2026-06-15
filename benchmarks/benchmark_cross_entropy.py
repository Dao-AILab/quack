import argparse
import os

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch
import torch.nn.functional as F
from triton.testing import Benchmark, do_bench, perf_report

import cutlass
import cutlass.torch as cutlass_torch

from quack.bench.bench_utils import run_and_print
from quack.cross_entropy import cross_entropy, cross_entropy_fwd


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


def _result(num_bytes: int, ms: float) -> dict:
    gbps = num_bytes / (ms / 1000) / 1e9
    return {"ms": round(ms, 4), "GB/s": round(gbps)}


def _bench(fn, **kwargs) -> float:
    return do_bench(fn, warmup=10, rep=100, **kwargs)


def _providers():
    return [("quack", "quack"), ("torch_compile", "torch.compile")]


def make_fwd_benchmark(dtype_name: str, return_dx: bool, x_vals=None) -> Benchmark:
    line_vals, line_names = zip(*_providers())
    suffix = dtype_name + ("-with-dx" if return_dx else "")
    return Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals if x_vals is not None else MN_PAIRS,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"cross-entropy-fwd-{suffix}",
        args={"dtype_name": dtype_name, "return_dx": return_dx},
        xlabel="(M, N)",
        ylabel="GB/s",
    )


def make_bwd_benchmark(dtype_name: str, x_vals=None) -> Benchmark:
    line_vals, line_names = zip(*_providers())
    return Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals if x_vals is not None else MN_PAIRS,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"cross-entropy-bwd-{dtype_name}",
        args={"dtype_name": dtype_name},
        xlabel="(M, N)",
        ylabel="GB/s",
    )


def cross_entropy_fwd_runner(M, N, provider, dtype_name, return_dx):
    dtype = DTYPE_MAP[dtype_name]
    torch_dtype = cutlass_torch.dtype(dtype)
    db = dtype.width // 8

    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype)
    target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)

    if provider == "quack":
        fn = lambda: cross_entropy_fwd(x, target, return_dx=return_dx)
        ms = _bench(fn)
        # I/O: read x (+ write dx if return_dx) + read target + write loss
        nbytes = M * N * db * (2 if return_dx else 1) + M * 8 + M * 4
    elif provider == "torch_compile":
        compiled = torch.compile(lambda x, target: F.cross_entropy(x, target, reduction="none"))
        fn = lambda: compiled(x, target)
        ms = _bench(fn)
        nbytes = M * N * db + M * 8 + M * 4
    else:
        raise ValueError(provider)

    return _result(nbytes, ms)


def cross_entropy_bwd_runner(M, N, provider, dtype_name):
    dtype = DTYPE_MAP[dtype_name]
    torch_dtype = cutlass_torch.dtype(dtype)

    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype, requires_grad=True)
    target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
    dloss = torch.randn(M, device="cuda", dtype=torch.float32)

    if provider == "quack":
        loss = cross_entropy(x, target, reduction="none")
        fn = lambda: torch.autograd.grad(loss, x, grad_outputs=dloss, retain_graph=True)
        ms = _bench(fn, grad_to_none=(x,))
    elif provider == "torch_compile":
        x_ref = x.detach().clone().requires_grad_()
        loss_ref = F.cross_entropy(x_ref, target, reduction="none")
        compiled = torch.compile(
            lambda: torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss, retain_graph=True)
        )
        ms = _bench(compiled, grad_to_none=(x_ref,))
    else:
        raise ValueError(provider)

    # I/O: read x + read target + read dloss + read lse(M*4) + write dx
    nbytes = (
        2 * x.numel() * x.element_size()
        + target.numel() * target.element_size()
        + dloss.numel() * dloss.element_size()
        + x.shape[0] * 4
    )
    return _result(nbytes, ms)


def main():
    parser = argparse.ArgumentParser(description="Benchmark cross_entropy fwd / bwd")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--fwd_dx", action="store_true", help="Forward pass that also computes dx")
    parser.add_argument("--M", type=int, default=None, help="Bench a single M (requires --N)")
    parser.add_argument("--N", type=int, default=None, help="Bench a single N (requires --M)")
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    if (args.M is None) != (args.N is None):
        parser.error("--M and --N must be given together")
    x_vals = [(args.M, args.N)] if args.M is not None else None

    torch.manual_seed(0)
    cutlass.cuda.initialize_cuda_context()

    if args.backward:
        bench = perf_report(make_bwd_benchmark(args.dtype, x_vals))(cross_entropy_bwd_runner)
    else:
        bench = perf_report(make_fwd_benchmark(args.dtype, args.fwd_dx, x_vals))(
            cross_entropy_fwd_runner
        )

    run_and_print(bench, save_path=args.save_path)


if __name__ == "__main__":
    main()
