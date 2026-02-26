import argparse
import time
from typing import Tuple

import torch
from triton.testing import do_bench

import cutlass
import cutlass.torch as cutlass_torch

from quack.fused_add_rmsnorm import (
    fused_add_rmsnorm,
    fused_add_rmsnorm_ref,
    _fused_add_rmsnorm_backward,
    _fused_add_rmsnorm_fwd,
)
from quack.rmsnorm import rmsnorm


def run_fused_add_rmsnorm(
    M: int,
    N: int,
    dtype: torch.dtype,
    warmup_iterations: int = 5,
    iterations: int = 100,
    eps: float = 1e-6,
) -> Tuple[float, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run this benchmark!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input / residual dtype: {dtype}")

    device = "cuda"
    residual = torch.randn(M, N, device=device, dtype=dtype)
    hidden_states = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=torch.float32)

    print("Input tensor shapes:")
    print(f"  residual      : {residual.shape}, dtype: {residual.dtype}")
    print(f"  hidden_states : {hidden_states.shape}, dtype: {hidden_states.dtype}")
    print(f"  weight        : {weight.shape}, dtype: {weight.dtype}")

    mem_bytes = (
        residual.numel() * residual.element_size()
        + hidden_states.numel() * hidden_states.element_size()
        + weight.numel() * weight.element_size()
        + hidden_states.numel() * hidden_states.element_size()
    )

    fused_add_rmsnorm(residual, hidden_states, weight, eps=eps)

    def bench(label: str, fn):
        time.sleep(0.5)
        avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        mem_bw = round(mem_bytes / (avg_time / 1000) / 1e9, 2)
        print(f"{label} execution time: {avg_time:.4f} ms")
        print(f"{label} mem throughput: {mem_bw:.2f} GB/s")
        return avg_time

    print("\nExecuting Fused Add + RMSNorm kernel...")
    fused_time = bench(
        "Fused kernel",
        lambda: fused_add_rmsnorm(residual, hidden_states, weight, eps=eps),
    )

    print("\nExecuting RMSNorm kernel with residual path...")
    rmsnorm_time = bench(
        "RMSNorm kernel",
        lambda: rmsnorm(hidden_states, weight, residual=residual, eps=eps),
    )

    print("\nExecuting PyTorch eager reference...")
    pytorch_eager_time = bench(
        "PyTorch eager reference",
        lambda: fused_add_rmsnorm_ref(residual, hidden_states, weight, eps=eps),
    )

    print("\nExecuting PyTorch compiled reference...")
    compiled_ref = torch.compile(fused_add_rmsnorm_ref)
    for _ in range(5):
        compiled_ref(residual, hidden_states, weight, eps=eps)
    pytorch_compiled_time = bench(
        "PyTorch compiled reference",
        lambda: compiled_ref(residual, hidden_states, weight, eps=eps),
    )

    print("\nComparisons:")
    print(f"Fused Add RMSNorm Forward Kernel vs RMSNorm kernel with Residual Path: {rmsnorm_time / fused_time:6.2f}x speedup")
    print(f"Fused Add RMSNorm Forward Kernel vs PyTorch compiled baseline: {pytorch_compiled_time / fused_time:6.2f}x speedup")
    print(f"Fused Add RMSNorm Forward Kernel vs PyTorch eager baseline: {pytorch_eager_time / fused_time:6.2f}x speedup")

    return fused_time, rmsnorm_time, pytorch_compiled_time, pytorch_eager_time


def run_fused_add_rmsnorm_backward(
    M: int,
    N: int,
    dtype: torch.dtype,
    warmup_iterations: int = 5,
    iterations: int = 100,
    eps: float = 1e-6,
) -> Tuple[float, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run this benchmark!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input / residual dtype: {dtype}")

    device = "cuda"
    residual = torch.randn(M, N, device=device, dtype=dtype)
    hidden_states = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=torch.float32)

    residual_eager = residual.detach().clone().requires_grad_(True)
    hidden_eager = hidden_states.detach().clone().requires_grad_(True)
    weight_eager = weight.detach().clone().requires_grad_(True)

    out_eager = fused_add_rmsnorm_ref(residual_eager, hidden_eager, weight_eager, eps=eps)
    dout = torch.randn_like(out_eager)

    x = residual + hidden_states
    out, rstd = _fused_add_rmsnorm_fwd(
        residual, hidden_states, weight, eps=eps, return_rstd=True
    )
    _fused_add_rmsnorm_backward(x, weight, dout, rstd)

    mem_bytes = (
        x.numel() * x.element_size()  # read x
        + weight.numel() * weight.element_size()  # read weight
        + dout.numel() * dout.element_size()  # read dout
        + rstd.numel() * rstd.element_size()  # read rstd
        + x.numel() * x.element_size()  # write dx
        + weight.numel() * weight.element_size()  # write dw
    )

    def bench(label: str, fn):
        time.sleep(0.5)
        avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        mem_bw = round(mem_bytes / (avg_time / 1000) / 1e9, 2)
        print(f"{label} execution time: {avg_time:.4f} ms")
        print(f"{label} mem throughput: {mem_bw:.2f} GB/s")
        return avg_time

    print("\nExecuting fused backward kernel...")
    fused_time = bench(
        "Fused backward kernel",
        lambda: _fused_add_rmsnorm_backward(x, weight, dout, rstd),
    )

    print("\nExecuting PyTorch eager backward reference...")

    def pytorch_backward():
        torch.autograd.grad(
            out_eager,
            (residual_eager, hidden_eager, weight_eager),
            dout,
            retain_graph=True,
        )

    pytorch_eager_time = bench("PyTorch eager backward", pytorch_backward)

    print("\nExecuting PyTorch compiled backward reference...")
    compiled_ref = torch.compile(fused_add_rmsnorm_ref)
    residual_compiled = residual.detach().clone().requires_grad_(True)
    hidden_compiled = hidden_states.detach().clone().requires_grad_(True)
    weight_compiled = weight.detach().clone().requires_grad_(True)
    for _ in range(5):
        out_compiled = compiled_ref(residual_compiled, hidden_compiled, weight_compiled, eps=eps)
        torch.autograd.grad(out_compiled, (residual_compiled, hidden_compiled, weight_compiled), dout)

    def pytorch_compiled_backward():
        out_compiled = compiled_ref(residual_compiled, hidden_compiled, weight_compiled, eps=eps)
        torch.autograd.grad(
            out_compiled,
            (residual_compiled, hidden_compiled, weight_compiled),
            dout,
            retain_graph=False,
        )

    pytorch_compiled_time = bench(
        "PyTorch compiled backward",
        pytorch_compiled_backward,
    )

    print("\nComparisons:")
    print(f"Fused Add RMSNorm Backward Kernel vs PyTorch eager backward: {pytorch_eager_time / fused_time:6.2f}x speedup")
    print(
        f"Fused Add RMSNorm Backward Kernel vs PyTorch compiled backward: {pytorch_compiled_time / fused_time:6.2f}x speedup"
    )

    return fused_time, pytorch_eager_time, pytorch_compiled_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the fused residual-add + RMSNorm kernel."
    )
    parser.add_argument("--M", default=32768, type=int)
    parser.add_argument("--N", default=32768, type=int)
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        choices=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32],
        default=cutlass.BFloat16,
    )
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Benchmark the backward pass instead of the forward pass.",
    )

    args = parser.parse_args()

    if args.backward:
        print("=== Fused Add + RMSNorm Backward Benchmark ===")
        run_fused_add_rmsnorm_backward(
            args.M,
            args.N,
            dtype=cutlass_torch.dtype(args.dtype),
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            eps=args.eps,
        )
    else:
        print("=== Fused Add + RMSNorm Forward Benchmark ===")
        run_fused_add_rmsnorm(
            args.M,
            args.N,
            dtype=cutlass_torch.dtype(args.dtype),
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            eps=args.eps,
        )

