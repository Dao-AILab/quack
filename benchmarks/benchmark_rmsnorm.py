import argparse
import os
import time
from typing import Optional

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch
from triton.testing import do_bench

import cutlass
from quack.rmsnorm import rmsnorm_fwd, rmsnorm_ref, rmsnorm, rmsnorm_bwd

try:
    import cudnn
except ImportError:
    cudnn = None


def run_rmsnorm(
    M,
    N,
    dtype: torch.dtype,
    residual_dtype: Optional[torch.dtype] = None,
    warmup_iterations=5,
    iterations=100,
):
    if not torch.cuda.is_available():
        raise RuntimeError("Ampere GPU is required to run this example!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    device = "cuda"
    x = torch.randn(M, N, device=device, dtype=dtype)
    if residual_dtype is not None:
        residual = torch.randn(M, N, device=device, dtype=residual_dtype)
    else:
        residual = None
    w = torch.randn(N, device=device, dtype=torch.float32)

    print("Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    print(f"w: {w.shape}, dtype: {w.dtype}")

    eps = 1e-6

    print("Executing kernel...")
    rmsnorm_fwd(x, w, residual=residual, eps=eps, store_rstd=True)

    compiled_func_ref = torch.compile(rmsnorm_ref)

    fn = lambda: rmsnorm_fwd(x, w, residual=residual, eps=eps)
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bytes = (2 * x.numel() * dtype.itemsize + w.numel() * 4)
    if residual is not None:
        mem_bytes += 2 * residual.numel() * residual.dtype.itemsize
    mem_bw = round(mem_bytes / (avg_time / 1000) / 1e9)
    print(f"Kernel execution time: {avg_time:.4f} ms")
    print(f"Mem throughput: {mem_bw:.2f} GB/s")

    fn = lambda: compiled_func_ref(x, w, residual=residual, eps=eps)
    for _ in range(5):
        fn()  # warm up
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bytes_ref = mem_bytes
    mem_bw_ref = round(mem_bytes_ref / (avg_time / 1000) / 1e9)
    print(f"Ref kernel execution time: {avg_time:.4f} ms")
    print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")

    if cudnn is not None:
        run_cudnn = rmsnorm_cudnn_setup(M, N, dtype)
        time.sleep(0.5)
        avg_time = do_bench(run_cudnn, warmup=warmup_iterations, rep=iterations)
        mem_bytes_cudnn = (2 * x.numel() * dtype.itemsize + w.numel() * 4)
        mem_bw_cudnn = round(mem_bytes_cudnn / (avg_time / 1000) / 1e9)
        print(f"Cudnn kernel execution time: {avg_time:.4f} ms")
        print(f"Cudnn mem throughput: {mem_bw_cudnn:.2f} GB/s")

    return mem_bw, mem_bw_ref


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
    # create tensor handles with the graph API
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
    # enable all outputs
    out.set_name("output").set_output(True).set_data_type(out_gpu.dtype)
    inv_var.set_name("inv_var").set_output(True).set_data_type(inv_var_gpu.dtype)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    # Mapping of (handles -> memory)
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


def run_rmsnorm_bwd(
    M,
    N,
    dtype: torch.dtype,
    residual_dtype: Optional[torch.dtype] = None,
    warmup_iterations=5,
    iterations=100,
):
    if not torch.cuda.is_available():
        raise RuntimeError("Ampere GPU is required to run this example!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    device = "cuda"

    # Set up forward pass inputs with gradients enabled
    x = torch.randn(M, N, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    w = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)
    w_ref = w.detach().clone().requires_grad_()
    if residual_dtype is not None:
        residual = torch.randn(M, N, device=device, dtype=residual_dtype, requires_grad=True)
        residual_ref = residual.detach().clone().requires_grad_()
    else:
        residual, residual_ref = None, None

    print("Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    print(f"w: {w.shape}, dtype: {w.dtype}")

    eps = 1e-6

    # Forward pass to get outputs and rstd
    y = rmsnorm(x, w, residual=residual, eps=eps)
    if residual is not None:
        y, residual_out = y
    else:
        residual_out = None

    # Create upstream gradients
    dy = torch.randn_like(y)
    rstd = torch.randn(M, device=device, dtype=torch.float32)
    dresidual_out = torch.randn_like(residual_out) if residual is not None else None

    def mem_in_bytes(*args):
        return sum(t.numel() * t.dtype.itemsize for t in args if t is not None)

    time.sleep(0.5)
    # Benchmark custom backward pass
    # fn = lambda: torch.autograd.grad(y, [x, w], grad_outputs=dy, retain_graph=True)
    def fn():
        # x.grad = None  # Reset gradients to avoid accumulation
        # y.backward(dy, retain_graph=True)
        rmsnorm_bwd(x if residual is None else residual_out, w, dy, rstd, dresidual_out=dresidual_out)

    avg_time = do_bench(fn, grad_to_none=(x,), warmup=warmup_iterations, rep=iterations)
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count * 2
    mem_bytes = mem_in_bytes(x if residual is None else residual_out, w, dy, dresidual_out, x, residual if residual is not None and residual.dtype != x.dtype else None)
    mem_bw = round(mem_bytes / (avg_time / 1000) / 1e9)
    print(f"Kernel execution time: {avg_time:.4f} ms")
    print(f"Mem throughput: {mem_bw:.2f} GB/s")
    from flash_attn.utils.benchmark import pytorch_profiler
    pytorch_profiler(fn)

    # Reference implementation
    y_ref = torch.compile(rmsnorm_ref)(x_ref, w_ref, eps=eps)
    compiled_func_ref = lambda: torch.autograd.grad(y_ref, [x_ref, w_ref], grad_outputs=dy, retain_graph=True)
    # def f():
    #     x_ref.grad = None  # Reset gradients to avoid accumulation
    #     w_ref.grad = None
    #     rmsnorm_ref(x_ref, w_ref, eps=eps).backward(dy)
    # compiled_func_ref = torch.compile(f)

    for _ in range(5):
        compiled_func_ref()  # warm up
    time.sleep(0.5)
    avg_time_ref = do_bench(compiled_func_ref, warmup=warmup_iterations, rep=iterations)
    mem_bytes_ref = (3 * x.numel() * dtype.itemsize + w.numel() * 4 + x.shape[0] * 4 + sm_count * w.numel() * 4)
    mem_bw_ref = round(mem_bytes_ref / (avg_time_ref / 1000) / 1e9)
    print(f"Ref kernel execution time: {avg_time_ref:.4f} ms")
    print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")
    pytorch_profiler(compiled_func_ref)

    return mem_bw, mem_bw_ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise add to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument("--M", default=32768, type=int)
    parser.add_argument("--N", default=32768, type=int)
    parser.add_argument("--dtype", type=cutlass.dtype, choices=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32], default=cutlass.BFloat16)
    parser.add_argument("--residual_dtype", type=cutlass.dtype, choices=[None, cutlass.BFloat16, cutlass.Float16, cutlass.Float32], default=None)
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--backward", action="store_true", help="Benchmark backward pass instead of forward pass")

    args = parser.parse_args()

    if args.backward:
        print("=== RMSNorm Backward Pass Benchmark ===")
        run_rmsnorm_bwd(
            args.M,
            args.N,
            dtype=cutlass.torch.dtype(args.dtype),
            residual_dtype=cutlass.torch.dtype(args.residual_dtype) if args.residual_dtype else None,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
        )
    else:
        print("=== RMSNorm Forward Pass Benchmark ===")
        run_rmsnorm(
            args.M,
            args.N,
            dtype=cutlass.torch.dtype(args.dtype),
            residual_dtype=cutlass.torch.dtype(args.residual_dtype) if args.residual_dtype else None,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
        )
    # # MN_pairs = [(32768, 256), (32768, 512), (32768, 1024), (32768, 2048), (32768, 4096), (32768, 8192), (32768, 16384), (32768, 32768), (32768, 65536), (16384, 131072), (8192, 262144)]
    # # MN_pairs = [(32768, 2048)]
    # MN_pairs = [(16384, 65536)]
    # results = []
    # for M, N in MN_pairs:
    #     res = run_rmsnorm(
    #         M,
    #         N,
    #         dtype=cutlass.BFloat16,
    #         skip_ref_check=False,
    #         benchmark=True,
    #         warmup_iterations=args.warmup_iterations,
    #         iterations=args.iterations,
    #     )
    #     results.append(res)
    # print(results)
    # print("\nPASS")
