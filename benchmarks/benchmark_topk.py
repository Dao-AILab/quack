import argparse
import time
from typing import Type

import torch
from triton.testing import do_bench

import cutlass
import cutlass.torch as cutlass_torch

from quack.topk import topk, topk_bwd

try:
    import rtopk
except ImportError:
    rtopk = None


def run_topk(
    M,
    N,
    k,
    dtype: Type[cutlass.Numeric],
    softmax: bool = False,
    backward: bool = False,
    warmup_iterations=10,
    iterations=1000,
):
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA GPU is required to run this example!")

    print(f"Tensor dimensions: [{M}, {N}], k={k}")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)

    device = "cuda"
    x = torch.randn(M, N, device=device, dtype=torch_dtype, requires_grad=True)

    print(f"Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    out, idx = topk(x, k, softmax=softmax)
    print(f"Output shape: {out.shape}")

    if backward:
        dvalues = torch.randn_like(out)
        fn = lambda: topk_bwd(dvalues, out, idx, N, softmax=softmax)
    else:
        # Benchmark our implementation
        fn = lambda: topk(x, k, softmax=softmax)
    fn()  # warm up
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    # Memory: read input (M*N elements), write output (M*k elements)
    if backward:
        mem_accessed = (M * N + 2 * M * k) * dtype.width // 8
    else:
        mem_accessed = (M * N + M * k) * dtype.width // 8
    mem_bw = round(mem_accessed / (avg_time / 1000) / 1e9, 2)
    print(f"Kernel execution time: {avg_time:.4f} ms")
    print(f"Mem throughput: {mem_bw:.2f} GB/s")

    # Benchmark PyTorch reference
    if backward:
        fn_ref = lambda: torch.autograd.grad(
            torch.softmax(torch.topk(x, k, dim=-1, largest=True, sorted=True)[0], dim=-1)
            if softmax
            else torch.topk(x, k, dim=-1, largest=True, sorted=True)[0],
            x,
            grad_outputs=dvalues,
            retain_graph=True,
        )
        for _ in range(5):
            fn_ref()
        time.sleep(0.5)
        avg_time_ref = do_bench(fn_ref, warmup=warmup_iterations, rep=iterations)
        mem_bw_ref = round(mem_accessed / (avg_time_ref / 1000) / 1e9, 2)
        print(f"Ref backward execution time: {avg_time_ref:.4f} ms")
        print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")
        speedup = avg_time_ref / avg_time
        print(f"Speedup: {speedup:.2f}x")
    else:
        fn_ref = lambda: torch.topk(x, k, dim=-1, largest=True, sorted=True)[0]
        for _ in range(5): fn_ref()  # warm up
        time.sleep(0.5)
        avg_time_ref = do_bench(fn_ref, warmup=warmup_iterations, rep=iterations)
        mem_bw_ref = round(mem_accessed / (avg_time_ref / 1000) / 1e9, 2)
        print(f"Ref kernel execution time: {avg_time_ref:.4f} ms")
        print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")

        speedup = avg_time_ref / avg_time
        print(f"Speedup: {speedup:.2f}x")

    if rtopk is not None:
        fn_rtopk = lambda: rtopk.ops.rtopk(x, k, max_iter=512)
        for _ in range(5): fn_rtopk()  # warm up
        time.sleep(0.5)
        avg_time_ref = do_bench(fn_rtopk, warmup=warmup_iterations, rep=iterations)
        mem_bw_ref = round(mem_accessed / (avg_time_ref / 1000) / 1e9, 2)
        print(f"RTopK kernel execution time: {avg_time_ref:.4f} ms")
        print(f"RTopK mem throughput: {mem_bw_ref:.2f} GB/s")

    # do_bench doesn't seem very accurate for very fast kernels, so we use pytorch_profiler
    from flash_attn.utils.benchmark import pytorch_profiler
    pytorch_profiler(fn)
    if rtopk is not None:
        pytorch_profiler(fn_rtopk)

    return mem_bw, mem_bw_ref, speedup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark top-k operation"
    )
    parser.add_argument("--M", default=8192, type=int)
    parser.add_argument("--N", default=1024, type=int)
    parser.add_argument("--k", default=32, type=int)
    parser.add_argument("--dtype", type=cutlass.dtype, choices=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32], default=cutlass.BFloat16)
    parser.add_argument("--softmax", action="store_true", help="Apply softmax to top-k values")
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--sweep", action="store_true", help="Run sweep across different N and k values")
    parser.add_argument("--backward", action="store_true", help="Benchmark backward pass instead of forward pass")

    args = parser.parse_args()
    torch.manual_seed(0)

    cutlass.cuda.initialize_cuda_context()

    if args.sweep:
        print("=== Top-K Sweep Benchmark ===")
        # Test different combinations of N and k
        N_values = [64, 128, 256, 512, 1024]
        k_values = [8, 16, 32]

        results = []
        for N in N_values:
            for k in k_values:
                if k > N // 2:  # Skip if k is too large relative to N
                    continue
                print(f"\n--- N={N}, k={k} ---")
                try:
                    mem_bw, mem_bw_ref, speedup = run_topk(
                        args.M,
                        N,
                        k,
                        dtype=args.dtype,
                        softmax=args.softmax,
                        backward=args.backward,
                        warmup_iterations=args.warmup_iterations,
                        iterations=args.iterations,
                    )
                    results.append((N, k, mem_bw, mem_bw_ref, speedup))
                except Exception as e:
                    print(f"Error with N={N}, k={k}: {e}")

        print("\n=== Summary ===")
        print("N\tk\tOurs (GB/s)\tRef (GB/s)\tSpeedup")
        for N, k, mem_bw, mem_bw_ref, speedup in results:
            print(f"{N}\t{k}\t{mem_bw}\t\t{mem_bw_ref}\t\t{speedup:.2f}x")
    else:
        print("=== Top-K Benchmark ===")
        run_topk(
            args.M,
            args.N,
            args.k,
            dtype=args.dtype,
            softmax=args.softmax,
            backward=args.backward,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
        )
