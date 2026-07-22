#!/usr/bin/env python3
"""Same-GPU ROCm benchmark for Quack's explicit FlyDSL RMSNorm backend.

The reported baselines run in this process on the same AMD GPU. They are not
comparisons against Quack's CUDA/CuTe backend and do not imply CUDA parity.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from quack.backends.flydsl.rmsnorm_config import get_bwd_workspace_rows
from quack.rmsnorm_flydsl import _rmsnorm_flydsl_bwd, _rmsnorm_flydsl_fwd, rmsnorm

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _dispatch_latency_us(fn, warmup, iterations):
    """Measure batched synchronized wall latency per dispatch.

    This intentionally includes Python/dispatcher overhead and GPU execution.
    It does not claim to isolate sub-microsecond kernel duration with events.
    """

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def _wall_ms(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _record(results, provider, metric, value, unit):
    timing = (
        "batched_synchronized_wall_clock"
        if metric.endswith("_dispatch_latency")
        else "synchronized_wall_clock"
    )
    results.append(
        {
            "provider": provider,
            "metric": metric,
            "value": round(value, 4),
            "unit": unit,
            "timing": timing,
        }
    )


def _load_aiter():
    try:
        from aiter.ops.triton.rmsnorm import rms_norm
    except Exception as exc:
        print(f"AIter unavailable; skipping same-GPU comparison: {type(exc).__name__}: {exc}")
        return None
    return rms_norm


def run(args):
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires ROCm PyTorch and a GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    arch = props.gcnArchName.split(":", 1)[0]
    if arch not in {"gfx942", "gfx950"}:
        raise RuntimeError(f"FlyDSL RMSNorm supports gfx942/gfx950, got {arch}")

    dtype = DTYPES[args.dtype]
    torch.manual_seed(args.seed)
    x = torch.randn(args.M, args.N, device=device, dtype=dtype)
    weight = torch.randn(args.N, device=device, dtype=torch.float32)
    doutput = torch.randn_like(x)
    output, dx = torch.empty_like(x), torch.empty_like(x)
    rstd = torch.empty(args.M, device=device, dtype=torch.float32)
    dweight = torch.zeros_like(weight)
    workspace = torch.empty(
        get_bwd_workspace_rows(args.M, args.N, dtype) * args.N,
        device=device,
        dtype=torch.float32,
    )
    results = []

    _record(
        results,
        "flydsl",
        "first_forward_compile_and_launch",
        _wall_ms(lambda: rmsnorm(x, weight, eps=args.eps)),
        "ms",
    )
    _rmsnorm_flydsl_fwd(x, weight, output, rstd, args.eps)

    def direct_backward():
        dweight.zero_()
        _rmsnorm_flydsl_bwd(x, weight, doutput, rstd, dx, dweight, workspace)

    _record(
        results,
        "flydsl",
        "first_backward_compile_and_launch",
        _wall_ms(direct_backward),
        "ms",
    )
    _record(
        results,
        "flydsl",
        "hot_cache_forward_dispatch_latency",
        _dispatch_latency_us(
            lambda: _rmsnorm_flydsl_fwd(x, weight, output, rstd, args.eps),
            args.warmup,
            args.iterations,
        ),
        "us",
    )
    _record(
        results,
        "flydsl",
        "direct_backward_dispatch_latency",
        _dispatch_latency_us(direct_backward, args.warmup, args.iterations),
        "us",
    )

    x_train = x.detach().clone().requires_grad_()
    weight_train = weight.detach().clone().requires_grad_()

    def e2e(fn):
        y = fn(x_train, weight_train, eps=args.eps)
        return torch.autograd.grad(y, (x_train, weight_train), doutput)

    _record(
        results,
        "flydsl",
        "public_autograd_e2e_dispatch_latency",
        _dispatch_latency_us(lambda: e2e(rmsnorm), args.warmup, args.iterations),
        "us",
    )
    if not args.skip_torch_compile:
        compiled = torch.compile(rmsnorm, fullgraph=True)
        e2e(compiled)
        _record(
            results,
            "flydsl_torch_compile",
            "public_autograd_e2e_dispatch_latency",
            _dispatch_latency_us(lambda: e2e(compiled), args.warmup, args.iterations),
            "us",
        )

    def torch_forward(input, scale):
        return torch.nn.functional.rms_norm(input, (args.N,), scale, args.eps)

    _record(
        results,
        "pytorch_eager",
        "hot_cache_forward_dispatch_latency",
        _dispatch_latency_us(
            lambda: torch_forward(x, weight),
            args.warmup,
            args.iterations,
        ),
        "us",
    )

    rstd_ref = torch.rsqrt(x.float().square().mean(dim=-1) + args.eps)

    def torch_direct_backward(input, scale, grad, inv_rms):
        x_hat = input.float() * inv_rms[:, None]
        weighted_grad = grad.float() * scale
        coefficient = (x_hat * weighted_grad).mean(dim=-1, keepdim=True)
        dx_ref = ((weighted_grad - x_hat * coefficient) * inv_rms[:, None]).to(input.dtype)
        dweight_ref = (grad.float() * x_hat).sum(dim=0)
        return dx_ref, dweight_ref

    _record(
        results,
        "pytorch_eager_math",
        "direct_backward_dispatch_latency",
        _dispatch_latency_us(
            lambda: torch_direct_backward(x, weight, doutput, rstd_ref),
            args.warmup,
            args.iterations,
        ),
        "us",
    )

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()

    def torch_e2e():
        y = torch_forward(x_ref, weight_ref)
        return torch.autograd.grad(y, (x_ref, weight_ref), doutput)

    _record(
        results,
        "pytorch_eager",
        "public_autograd_e2e_dispatch_latency",
        _dispatch_latency_us(torch_e2e, args.warmup, args.iterations),
        "us",
    )
    if not args.skip_torch_compile:
        compiled_torch_forward = torch.compile(torch_forward, fullgraph=True)
        compiled_torch_backward = torch.compile(torch_direct_backward, fullgraph=True)
        compiled_torch_forward(x, weight)
        compiled_torch_backward(x, weight, doutput, rstd_ref)
        _record(
            results,
            "pytorch_fullgraph",
            "hot_cache_forward_dispatch_latency",
            _dispatch_latency_us(
                lambda: compiled_torch_forward(x, weight),
                args.warmup,
                args.iterations,
            ),
            "us",
        )
        _record(
            results,
            "pytorch_fullgraph_math",
            "direct_backward_dispatch_latency",
            _dispatch_latency_us(
                lambda: compiled_torch_backward(x, weight, doutput, rstd_ref),
                args.warmup,
                args.iterations,
            ),
            "us",
        )

        def compiled_torch_e2e():
            y = compiled_torch_forward(x_ref, weight_ref)
            return torch.autograd.grad(y, (x_ref, weight_ref), doutput)

        _record(
            results,
            "pytorch_fullgraph",
            "public_autograd_e2e_dispatch_latency",
            _dispatch_latency_us(compiled_torch_e2e, args.warmup, args.iterations),
            "us",
        )
    if args.aiter and (aiter_rmsnorm := _load_aiter()) is not None:
        _record(
            results,
            "aiter",
            "hot_cache_forward_dispatch_latency",
            _dispatch_latency_us(
                lambda: aiter_rmsnorm(x, weight, args.eps),
                args.warmup,
                args.iterations,
            ),
            "us",
        )

    print(f"GPU={props.name} arch={arch} shape=({args.M}, {args.N}) dtype={args.dtype}")
    print("Baselines are same-machine ROCm measurements; CUDA Quack parity is not implied.")
    print(
        "Dispatch-latency rows use batched synchronized wall time and include "
        "Python/dispatcher overhead plus GPU execution."
    )
    print(f"{'provider':24s} {'metric':44s} {'value':>12s} {'unit':>5s}")
    for row in results:
        print(f"{row['provider']:24s} {row['metric']:44s} {row['value']:12.4f} {row['unit']:>5s}")
    if args.output_json:
        payload = {
            "device": props.name,
            "arch": arch,
            "shape": [args.M, args.N],
            "dtype": args.dtype,
            "eps": args.eps,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "scope": (
                "same-machine ROCm baselines; dispatch rows are batched synchronized "
                "wall-clock latency; no CUDA Quack parity claim"
            ),
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aiter", action="store_true")
    parser.add_argument("--skip-torch-compile", action="store_true")
    parser.add_argument("--enable-disk-cache", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if not cli_args.enable_disk_cache:
        os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    run(cli_args)
