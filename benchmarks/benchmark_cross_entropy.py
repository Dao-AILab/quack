import argparse
import time
from typing import Type

import torch
import torch.nn.functional as F
from triton.testing import do_bench

import cutlass
import cutlass.torch as cutlass_torch

from quack.cross_entropy import cross_entropy_fwd, cross_entropy


def run_cross_entropy(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    warmup_iterations=2,
    iterations=200,
    return_dx=False,
):
    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)

    device = "cuda"
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch_dtype)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)

    loss = cross_entropy_fwd(x, target, return_dx=return_dx)

    compiled_func_ref = torch.compile(lambda x, target: F.cross_entropy(x, target, reduction='none'))

    fn = lambda: cross_entropy_fwd(x, target, return_dx=return_dx)
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    # Memory bandwidth calculation: read x (M*N elements) + read target (M elements) + write loss (M elements)
    mem_bytes = (M * N * (2 if return_dx else 1) + M + M) * dtype.width // 8
    mem_bw = round(mem_bytes / (avg_time / 1000) / 1e9)
    print(f"Kernel execution time: {avg_time:.4f} ms")
    print(f"Mem throughput: {mem_bw:.2f} GB/s")

    fn_ref = lambda: compiled_func_ref(x, target)
    for _ in range(5): fn_ref()  # warm up
    time.sleep(0.5)
    avg_time = do_bench(fn_ref, warmup=warmup_iterations, rep=iterations)
    mem_bytes = (M * N + M + M) * dtype.width // 8
    mem_bw_ref = round(mem_bytes / (avg_time / 1000) / 1e9)
    print(f"Ref kernel execution time: {avg_time:.4f} ms")
    print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")

    # from flash_attn.utils.benchmark import pytorch_profiler
    # pytorch_profiler(fn)
    # pytorch_profiler(fn_ref)
    # pytorch_profiler(torch.compile(torch.logsumexp), x, dim=-1)

    return mem_bw, mem_bw_ref



def run_cross_entropy_backward(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    warmup_iterations=10,
    iterations=1000,
):
    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)

    device = "cuda"
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch_dtype, requires_grad=True)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    x_ref = x.detach().clone().requires_grad_()

    print(f"Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    print(f"target: {target.shape}, dtype: {target.dtype}")

    loss = cross_entropy(x, target, reduction="none")
    dloss = torch.randn(M, device=device, dtype=torch.float32)
    torch.cuda.synchronize()

    # Reference implementation
    loss_ref = F.cross_entropy(x_ref, target, reduction='none')
    compiled_func_ref = torch.compile(lambda: torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss, retain_graph=True))

    for _ in range(5): compiled_func_ref()  # warm up
    time.sleep(0.5)
    avg_time_ref = do_bench(compiled_func_ref, warmup=warmup_iterations, rep=iterations)
    mem_bw_ref = round((2 * x.numel() * x.element_size() + target.numel() * target.element_size() +
                        dloss.numel() * dloss.element_size()) / (avg_time_ref / 1000) / 1e9)

    time.sleep(0.5)
    fn = lambda: torch.autograd.grad(loss, x, grad_outputs=dloss, retain_graph=True)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    # Memory bandwidth calculation: read x (M*N) + read target (M) + read dloss (M) + write grad (M*N)
    mem_bw = round((2 * x.numel() * x.element_size() + target.numel() * target.element_size() + 
                    dloss.numel() * dloss.element_size()) / (avg_time / 1000) / 1e9)
    print(f"Kernel execution time: {avg_time:.4f} ms")
    print(f"Mem throughput: {mem_bw:.2f} GB/s")
    print(f"Ref kernel execution time: {avg_time_ref:.4f} ms")
    print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")

    return mem_bw, mem_bw_ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark cross entropy forward and backward passes"
    )
    parser.add_argument("--M", default=8192, type=int)
    parser.add_argument("--N", default=16384, type=int)
    parser.add_argument("--dtype", type=cutlass.dtype, choices=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32], default=cutlass.BFloat16)
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--backward", action="store_true", help="Benchmark backward pass instead of forward pass")
    parser.add_argument("--fwd_dx", action="store_true", help="Benchmark forward pass that also computes dx")

    args = parser.parse_args()
    torch.manual_seed(0)
    cutlass.cuda.initialize_cuda_context()


    if args.backward:
        print("=== Cross Entropy Backward Pass Benchmark ===")
        run_cross_entropy_backward(
            args.M,
            args.N,
            dtype=args.dtype,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
        )
    else:
        print("=== Cross Entropy Forward Pass Benchmark ===")
        run_cross_entropy(
            args.M,
            args.N,
            dtype=args.dtype,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            return_dx=args.fwd_dx,
        )


    '''
    #MN_pairs = [(32768, 256), (32768, 512), (32768, 1024), (32768, 2048), (32768, 4096), (32768, 8192), (32768, 16384), (32768, 32768), (32768, 65536), (16384, 131072), (8192, 262144)]
    MN_pairs = [(32768, 65536)]
    results = []
    for M, N in MN_pairs:
         res = run_cross_entropy_backward(
             M,
             N,
             dtype=args.dtype,
             warmup_iterations=args.warmup_iterations,
             iterations=args.iterations,
         )
         results.append(res)
    print(results)
    #print([x for x, _ in results])
    '''
