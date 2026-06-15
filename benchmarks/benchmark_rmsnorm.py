import argparse
import os
from typing import Optional

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch  # noqa: E402
import torch._functorch.config as _functorch_config  # noqa: E402
from triton.testing import Benchmark, do_bench, perf_report  # noqa: E402

from quack.bench.bench_utils import run_and_print  # noqa: E402
from quack.rmsnorm import rmsnorm, rmsnorm_bwd, rmsnorm_fwd, rmsnorm_ref  # noqa: E402

# Inductor's donated-buffer optimization is incompatible with retain_graph=True
# (used so we benchmark only bwd, not fwd+bwd). Disable it for the torch.compile
# bwd path. Must be set before torch.compile builds the bwd graph.
_functorch_config.donated_buffer = False

try:
    import cudnn
except ImportError:
    cudnn = None


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


def _bench(fn, **kwargs) -> float:
    return do_bench(fn, warmup=10, rep=100, **kwargs)


def _fwd_providers():
    providers = [("quack", "quack"), ("torch_compile", "torch.compile")]
    if cudnn is not None:
        providers.append(("cudnn", "cudnn"))
    return providers


def _bwd_providers():
    return [("quack", "quack"), ("torch_compile", "torch.compile")]


def make_fwd_benchmark(
    dtype_name: str, residual_dtype_name: Optional[str], x_vals=None
) -> Benchmark:
    line_vals, line_names = zip(*_fwd_providers())
    suffix = dtype_name + (f"-res-{residual_dtype_name}" if residual_dtype_name else "")
    return Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals if x_vals is not None else MN_PAIRS,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"rmsnorm-fwd-{suffix}",
        args={"dtype_name": dtype_name, "residual_dtype_name": residual_dtype_name},
        xlabel="(M, N)",
        ylabel="GB/s",
    )


def make_bwd_benchmark(
    dtype_name: str, residual_dtype_name: Optional[str], x_vals=None
) -> Benchmark:
    line_vals, line_names = zip(*_bwd_providers())
    suffix = dtype_name + (f"-res-{residual_dtype_name}" if residual_dtype_name else "")
    if x_vals is None:
        x_vals = MN_PAIRS
        # quack RMSNorm bwd kernel rejects N > 128k with fp32 (smem too small).
        if DTYPE_MAP[dtype_name].itemsize >= 4:
            x_vals = [(m, n) for (m, n) in x_vals if n <= 128 * 1024]
    return Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"rmsnorm-bwd-{suffix}",
        args={"dtype_name": dtype_name, "residual_dtype_name": residual_dtype_name},
        xlabel="(M, N)",
        ylabel="GB/s",
    )


def _fwd_mem_bytes(x: torch.Tensor, w: torch.Tensor, residual: Optional[torch.Tensor]) -> int:
    nbytes = 2 * x.numel() * x.dtype.itemsize + w.numel() * 4
    if residual is not None:
        nbytes += 2 * residual.numel() * residual.dtype.itemsize
    return nbytes


def rmsnorm_cudnn_setup(M, N, dtype):
    x_gpu = torch.empty(M, N, dtype=dtype, device="cuda")
    scale_gpu = torch.empty(1, N, dtype=dtype, device="cuda")
    epsilon_cpu = torch.ones((1, 1), dtype=torch.float32, device="cpu")
    out_gpu = torch.empty_like(x_gpu)
    inv_var_gpu = torch.empty(M, 1, dtype=torch.float32, device="cuda")
    handle = cudnn.create_handle()
    graph = cudnn.pygraph(
        handle=handle,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x = graph.tensor_like(x_gpu.detach()).set_name("X")
    scale = graph.tensor_like(scale_gpu.detach()).set_name("scale")
    epsilon = graph.tensor_like(epsilon_cpu).set_name("epsilon")
    (out, inv_var) = graph.rmsnorm(
        name="rmsnorm",
        input=x,
        norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
        scale=scale,
        epsilon=epsilon,
    )
    out.set_name("output").set_output(True).set_data_type(out_gpu.dtype)
    inv_var.set_name("inv_var").set_output(True).set_data_type(inv_var_gpu.dtype)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    variant_pack = {
        x: x_gpu.detach(),
        scale: scale_gpu.detach(),
        epsilon: epsilon_cpu,
        out: out_gpu,
        inv_var: inv_var_gpu,
    }
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return out_gpu, inv_var_gpu

    return run


def rmsnorm_fwd_runner(M, N, provider, dtype_name, residual_dtype_name):
    dtype = DTYPE_MAP[dtype_name]
    residual_dtype = DTYPE_MAP[residual_dtype_name] if residual_dtype_name else None
    eps = 1e-6

    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    residual = (
        torch.randn(M, N, device="cuda", dtype=residual_dtype)
        if residual_dtype is not None
        else None
    )

    if provider == "quack":
        fn = lambda: rmsnorm_fwd(x, w, residual=residual, eps=eps)
        ms = _bench(fn)
        nbytes = _fwd_mem_bytes(x, w, residual)
    elif provider == "torch_compile":
        compiled = torch.compile(rmsnorm_ref)
        fn = lambda: compiled(x, w, residual=residual, eps=eps)
        ms = _bench(fn)
        nbytes = _fwd_mem_bytes(x, w, residual)
    elif provider == "cudnn":
        if residual is not None:
            return float("nan")
        run_cudnn = rmsnorm_cudnn_setup(M, N, dtype)
        ms = _bench(run_cudnn)
        nbytes = 2 * x.numel() * x.dtype.itemsize + w.numel() * 4
    else:
        raise ValueError(provider)

    return _result(nbytes, ms)


def rmsnorm_bwd_runner(M, N, provider, dtype_name, residual_dtype_name):
    dtype = DTYPE_MAP[dtype_name]
    residual_dtype = DTYPE_MAP[residual_dtype_name] if residual_dtype_name else None
    eps = 1e-6

    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
    if residual_dtype is not None:
        residual = torch.randn(M, N, device="cuda", dtype=residual_dtype, requires_grad=True)
    else:
        residual = None

    out = rmsnorm(x, w, residual=residual, eps=eps)
    if residual is not None:
        y, residual_out = out
    else:
        y, residual_out = out, None

    dy = torch.randn_like(y)
    rstd = torch.randn(M, device="cuda", dtype=torch.float32)
    dresidual_out = torch.randn_like(residual_out) if residual is not None else None

    if provider == "quack":
        in_x = x if residual is None else residual_out

        def fn():
            rmsnorm_bwd(in_x, w, dy, rstd, dresidual_out=dresidual_out)

        ms = _bench(fn, grad_to_none=(x,))

        def in_bytes(*ts):
            return sum(t.numel() * t.dtype.itemsize for t in ts if t is not None)

        nbytes = in_bytes(
            in_x,
            w,
            dy,
            dresidual_out,
            x,
            residual if residual is not None and residual.dtype != x.dtype else None,
        )
    elif provider == "torch_compile":
        x_ref = x.detach().clone().requires_grad_()
        w_ref = w.detach().clone().requires_grad_()
        y_ref = torch.compile(rmsnorm_ref)(x_ref, w_ref, eps=eps)
        fn = lambda: torch.autograd.grad(y_ref, [x_ref, w_ref], grad_outputs=dy, retain_graph=True)
        ms = _bench(fn, grad_to_none=(x_ref, w_ref))
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count * 2
        nbytes = (
            3 * x.numel() * dtype.itemsize
            + w.numel() * 4
            + x.shape[0] * 4
            + sm_count * w.numel() * 4
        )
    else:
        raise ValueError(provider)

    return _result(nbytes, ms)


def main():
    parser = argparse.ArgumentParser(description="Benchmark rmsnorm fwd / bwd")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    parser.add_argument("--residual_dtype", default=None, choices=[None, *DTYPE_MAP])
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--M", type=int, default=None, help="Bench a single M (requires --N)")
    parser.add_argument("--N", type=int, default=None, help="Bench a single N (requires --M)")
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    if (args.M is None) != (args.N is None):
        parser.error("--M and --N must be given together")
    x_vals = [(args.M, args.N)] if args.M is not None else None

    torch.manual_seed(0)

    if args.backward:
        bench = perf_report(make_bwd_benchmark(args.dtype, args.residual_dtype, x_vals))(
            rmsnorm_bwd_runner
        )
    else:
        bench = perf_report(make_fwd_benchmark(args.dtype, args.residual_dtype, x_vals))(
            rmsnorm_fwd_runner
        )

    run_and_print(bench, save_path=args.save_path)


if __name__ == "__main__":
    main()
