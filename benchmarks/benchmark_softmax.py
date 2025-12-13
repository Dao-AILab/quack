import argparse
import time
from typing import Type

import torch
import torch.nn.functional as F
from triton.testing import do_bench

import cutlass
import cutlass.torch as cutlass_torch

from quack.softmax import softmax

try:
    from liger_kernel.transformers.functional import liger_softmax
except ImportError:
    liger_softmax = None


def run_softmax(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    warmup_iterations=10,
    iterations=1000,
):
    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)

    device = "cuda"
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch_dtype)

    print(f"Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    out = softmax(x)
    # compiled_func_ref = torch.compile(lambda x: F.softmax(x, dim=-1))
    compiled_func_ref = torch.compile(lambda x: F.softmax(x, dim=-1))
    fn = lambda: softmax(x)
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bw = round(2 * x.numel() * dtype.width // 8 / (avg_time / 1000) / 1e9)
    print(f"Kernel execution time: {avg_time:.4f} ms")
    print(f"Mem throughput: {mem_bw:.2f} GB/s")

    fn = lambda: compiled_func_ref(x)
    for _ in range(5): fn()  # warm up
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bw_ref = round(2 * x.numel() * dtype.width // 8 / (avg_time / 1000) / 1e9)
    print(f"Torch compile kernel execution time: {avg_time:.4f} ms")
    print(f"Torch compile mem throughput: {mem_bw_ref:.2f} GB/s")

    if liger_softmax is not None:
        fn = lambda: liger_softmax(x)
        for _ in range(5): fn()  # warm up
        time.sleep(0.5)
        avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        mem_bw_ref = round(2 * x.numel() * dtype.width // 8 / (avg_time / 1000) / 1e9)
        print(f"Liger kernel execution time: {avg_time:.4f} ms")
        print(f"Liger mem throughput: {mem_bw_ref:.2f} GB/s")

    return mem_bw, mem_bw_ref


def run_softmax_backward(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    warmup_iterations=10,
    iterations=1000,
):
    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)

    device = "cuda"
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch_dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    print(f"Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")

    y = softmax(x)
    dy = torch.randn_like(y)

    time.sleep(0.5)
    fn = lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    # Memory: read dy and y, write ax backward
    mem_bw = round(3 * x.numel() * dtype.width // 8 / (avg_time / 1000) / 1e9)
    print(f"Kernel execution time: {avg_time:.4f} ms")
    print(f"Mem throughput: {mem_bw:.2f} GB/s")

    # Reference implementation
    y_ref = F.softmax(x_ref, dim=-1)
    compiled_func_ref = torch.compile(lambda: torch.autograd.grad(y_ref, x_ref, grad_outputs=dy, retain_graph=True))

    for _ in range(5): compiled_func_ref()  # warm up
    time.sleep(0.5)
    avg_time_ref = do_bench(compiled_func_ref, warmup=warmup_iterations, rep=iterations)
    mem_bw_ref = round(3 * x.numel() * dtype.width // 8 / (avg_time_ref / 1000) / 1e9)
    print(f"Ref kernel execution time: {avg_time_ref:.4f} ms")
    print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")

    return mem_bw, mem_bw_ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark softmax forward and backward passes"
    )
    parser.add_argument("--M", default=8192, type=int)
    parser.add_argument("--N", default=16384, type=int)
    parser.add_argument("--dtype", type=cutlass.dtype, choices=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32], default=cutlass.BFloat16)
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--backward", action="store_true", help="Benchmark backward pass instead of forward pass")

    args = parser.parse_args()
    torch.manual_seed(0)

    if args.backward:
        print("=== Softmax Backward Pass Benchmark ===")
        run_softmax_backward(
            args.M,
            args.N,
            dtype=args.dtype,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
        )
    else:
        print("=== Softmax Forward Pass Benchmark ===")
        run_softmax(
            args.M,
            args.N,
            dtype=args.dtype,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
        )

    # MN_pairs = [(32768, 256), (32768, 512), (32768, 1024), (32768, 2048), (32768, 4096), (32768, 8192), (32768, 16384), (32768, 32768), (32768, 65536), (16384, 131072), (8192, 262144)]
    # # MN_pairs = [(32768, 1024)]
    # results = []
    # for M, N in MN_pairs:
    #     res = run_softmax(
    #         M,
    #         N,
    #         dtype=args.dtype,
    #         warmup_iterations=args.warmup_iterations,
    #         iterations=args.iterations,
    #     )
    #     results.append(res)
    # # print(results)
    # print([x for x, _ in results])
