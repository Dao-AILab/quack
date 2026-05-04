import argparse
import time


import torch
from triton.testing import do_bench

from quack.mhc_post import mhc_post, mhc_post_ref

torch.set_float32_matmul_precision("high")


def _make_inputs(n0: int, n1: int, h: int, mhc: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n0, n1, mhc, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n0, n1, mhc, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n0, n1, mhc, mhc), dtype=torch.float32, device=device)
    return x, residual, post_layer_mix, comb_res_mix


def _mem_bytes(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor,
) -> int:
    tensors = (x, residual, post_layer_mix, comb_res_mix, out)
    return sum(t.numel() * t.element_size() for t in tensors)


def _bench(name: str, fn, mem_bytes: int, warmup: int, rep: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    time.sleep(0.2)
    latency_ms = do_bench(fn, warmup=warmup, rep=rep)
    bandwidth = mem_bytes / (latency_ms / 1000.0) / 1e9
    if name:
        print(f"{name} kernel execution time: {latency_ms:.4f} ms")
        print(f"{name} mem throughput: {bandwidth:.2f} GB/s")
    else:
        print(f"Kernel execution time: {latency_ms:.4f} ms")
        print(f"Mem throughput: {bandwidth:.2f} GB/s")
    return latency_ms


def run_mhc_post(
    n0: int,
    n1: int,
    h: int,
    mhc: int,
    warmup_iterations: int,
    iterations: int,
    skip_torch_compile: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark")

    x, residual, post_layer_mix, comb_res_mix = _make_inputs(n0, n1, h, mhc)
    print(f"Tensor dimensions: [{n0}, {n1}, {mhc}, {h}]")
    print("Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    print(f"residual: {residual.shape}, dtype: {residual.dtype}")
    print(f"post_layer_mix: {post_layer_mix.shape}, dtype: {post_layer_mix.dtype}")
    print(f"comb_res_mix: {comb_res_mix.shape}, dtype: {comb_res_mix.dtype}")

    out = mhc_post(x, residual, post_layer_mix, comb_res_mix)
    out_ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
    torch.testing.assert_close(out, out_ref)
    print("Correctness: compared QuACK against torch reference")

    mem_bytes = _mem_bytes(x, residual, post_layer_mix, comb_res_mix, out)
    quack_ms = _bench(
        "",
        lambda: mhc_post(x, residual, post_layer_mix, comb_res_mix),
        mem_bytes,
        warmup_iterations,
        iterations,
    )
    torch_ms = _bench(
        "Ref",
        lambda: mhc_post_ref(x, residual, post_layer_mix, comb_res_mix),
        mem_bytes,
        warmup_iterations,
        iterations,
    )
    print(f"Speedup over torch reference: {torch_ms / quack_ms:.2f}x")

    if not skip_torch_compile:
        compiled_ref = torch.compile(mhc_post_ref, fullgraph=True)
        compiled_out = compiled_ref(x, residual, post_layer_mix, comb_res_mix)
        torch.testing.assert_close(out, compiled_out)
        compiled_ms = _bench(
            "Compiled ref",
            lambda: compiled_ref(x, residual, post_layer_mix, comb_res_mix),
            mem_bytes,
            warmup_iterations,
            iterations,
        )
        print(f"Speedup over compiled torch reference: {compiled_ms / quack_ms:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MHC post forward pass")
    parser.add_argument("--n0", default=2, type=int)
    parser.add_argument("--n1", default=4096, type=int)
    parser.add_argument("--h", default=7168, type=int)
    parser.add_argument("--mhc", default=4, type=int)
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip-torch-compile", action="store_true")
    args = parser.parse_args()

    run_mhc_post(
        args.n0,
        args.n1,
        args.h,
        args.mhc,
        args.warmup_iterations,
        args.iterations,
        args.skip_torch_compile,
    )
