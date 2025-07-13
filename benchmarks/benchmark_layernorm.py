import argparse
import time
from typing import Type

import torch
from triton.testing import do_bench

import cutlass
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from quack.layernorm import _layernorm_fwd, layernorm_ref, rstd_ref, mean_ref
import cutlass.cute as cute

try:
    import cudnn
except ImportError:
    cudnn = None


def run_layernorm(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    warmup_iterations=2,
    iterations=200,
):
    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)

    device = "cuda"
    x = torch.randn(M, N, device=device, dtype=torch_dtype)
    w = torch.randn(N, device=device, dtype=torch.float32)

    print(f"Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    print(f"w: {w.shape}, dtype: {w.dtype}")

    eps = 1e-6

    print("Executing kernel...")
    out, rstd, mean = _layernorm_fwd(x, w, eps=eps, return_rstd=True, return_mean=True)

    compiled_func_ref = torch.compile(layernorm_ref)

    fn = lambda: _layernorm_fwd(x, w, eps=eps)
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bw = (2 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9
    print(f"Kernel execution time: {avg_time:.4f} ms")
    print(f"Mem throughput: {mem_bw:.2f} GB/s")

    fn = lambda: compiled_func_ref(x, w, eps=eps)
    for _ in range(5):
        fn()  # warm up
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bw_ref = (2 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9
    print(f"Ref kernel execution time: {avg_time:.4f} ms")
    print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")

    return mem_bw, mem_bw_ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise add to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument("--M", default=32768, type=int)
    parser.add_argument("--N", default=16384, type=int)
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)

    args = parser.parse_args()

    run_layernorm(
        args.M,
        args.N,
        dtype=cutlass.BFloat16,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
