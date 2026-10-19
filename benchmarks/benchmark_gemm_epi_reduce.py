"""GEMM + fused reduce-scatter / all-reduce benchmark (SM100, epi_reduce_mode).

Each rank computes A_local @ B_local^T over its K-shard (k_local = k / world_size); the
reducer warps multimem-reduce the partial D across ranks — reduce_scatter: each rank
keeps its M-slab (m / world_size rows); all_reduce: every rank ends with the full D.
Runs the public quack.gemm.gemm() frontend with make_epi_reduce_args for the comm setup.
Baseline: torch.bmm + dist.reduce_scatter_tensor / dist.all_reduce.

Usage:
    torchrun --nproc_per_node=8 benchmarks/benchmark_gemm_epi_reduce.py \
        --mnkl 8192,4096,4096,1 --mode reduce_scatter \
        --tile_shape_mnk 256,256 --cluster_shape_mnk 2,1
"""

import argparse
import os

import torch
import torch.distributed as dist

import cutlass
import cutlass.torch as cutlass_torch

from quack.bench.bench_utils_dist import do_bench_all
from quack.dist_utils import init_distributed, clean_distributed
from quack.distributed.gemm_epi_reduce import make_epi_reduce_args
from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm


def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        ) from e


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GEMM + fused reduce-scatter / all-reduce benchmark"
    )
    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(8192, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated); k is global (sharded k/world per rank)",
    )
    parser.add_argument(
        "--mode",
        choices=["reduce_scatter", "all_reduce"],
        default="reduce_scatter",
        help="Fused collective: reduce_scatter (own M-slab) or all_reduce (full D on every rank)",
    )
    parser.add_argument(
        "--tile_shape_mnk",
        "--tile_shape_mn",
        dest="tile_shape_mnk",
        type=parse_comma_separated_ints,
        default=(256, 256),
        help="MMA tile shape M,N (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mnk",
        type=parse_comma_separated_ints,
        default=(2, 1),
        help="Cluster shape M,N (comma-separated)",
    )
    parser.add_argument(
        "--ab_dtype",
        choices=["bfloat16", "mxfp8"],
        default="bfloat16",
        help="Operand dtype; mxfp8 = per-rank K-shard quantized mxfp8_e4m3 (sf_vec 32)",
    )
    parser.add_argument(
        "--d_dtype",
        type=str,
        default="BFloat16",
        help="Output dtype (also the default symmetric partials dtype): BFloat16/Float16/Float32.",
    )
    parser.add_argument(
        "--ws_dtype",
        type=str,
        default=None,
        help="Partials workspace dtype override (e.g. Float32); default = d_dtype",
    )
    parser.add_argument("--tolerance", type=float, default=3e-02, help="Tolerance for validation")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=30, help="Benchmark iterations")
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")
    args = parser.parse_args()
    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    return args


def run(args):
    init_distributed()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size > 1, "launch with torchrun --nproc_per_node > 1"
    sm_major = get_device_capacity(torch.device("cuda"))[0]
    assert sm_major in (10, 11), f"epi_reduce requires SM100 (B200/B300); got SM{sm_major}x"

    m, n, k, l = args.mnkl
    mode = args.mode
    d_dtype = cutlass.dtype(args.d_dtype)
    torch_d = cutlass_torch.dtype(d_dtype)
    torch_ws = cutlass_torch.dtype(cutlass.dtype(args.ws_dtype)) if args.ws_dtype else None
    blockscaled = args.ab_dtype == "mxfp8"
    tile_M, tile_N = args.tile_shape_mnk[:2]
    cluster_M, cluster_N = args.cluster_shape_mnk[:2]
    vec = 128 // d_dtype.width  # one 16B multimem vector
    assert m % world_size == 0, f"m ({m}) must be divisible by world_size ({world_size})"
    assert k % world_size == 0, f"k ({k}) must be divisible by world_size ({world_size})"
    assert n % vec == 0, f"n ({n}) must be divisible by {vec} (16B multimem vectors)"
    k_local = k // world_size
    m_per_rank = m // world_size

    if rank == 0:
        print(f"Running SM100 GEMM + {mode} with:")
        print(f"mnkl: {args.mnkl} (k_local: {k_local}), world_size: {world_size}")
        print(f"tile_shape_mnk: {args.tile_shape_mnk}, cluster_shape_mnk: {args.cluster_shape_mnk}")

    # A: (l, m, k_local), B: (l, n, k_local), rank-seeded, caller order.
    torch.manual_seed(1111 + rank)
    bs_kwargs = {}
    if blockscaled:
        from quack.blockscaled.utils import (
            create_blockscaled_operand_quantized,
            scale_view_for_kernel,
        )

        sf_vec = 32
        assert k_local % (4 * sf_vec) == 0, (
            f"k_local ({k_local}) must be divisible by {4 * sf_vec} (SF tile granularity)"
        )
        a_ref_mkl, aq, a_sc = create_blockscaled_operand_quantized(l, m, k_local, False, sf_vec)
        b_ref_mkl, bq, b_sc = create_blockscaled_operand_quantized(l, n, k_local, False, sf_vec)
        a_gpu, b_gpu = aq.permute(2, 0, 1), bq.permute(2, 0, 1)
        bs_kwargs = dict(
            SFA=scale_view_for_kernel(a_sc, m, k_local // sf_vec, l),
            SFB=scale_view_for_kernel(b_sc, n, k_local // sf_vec, l),
            bs_format_a="mxfp8_e4m3",
            bs_format_b="mxfp8_e4m3",
        )
        # Baseline/ref runs on the bf16 dequants (canary pipeline stays bf16).
        A = a_ref_mkl.permute(2, 0, 1).to(torch_d).contiguous()
        B = b_ref_mkl.permute(2, 0, 1).to(torch_d).contiguous()
    else:
        torch_ab = torch.bfloat16
        a_gpu = (
            torch.empty(l, m, k_local, dtype=torch.float32)
            .normal_()
            .mul_(1.0 / (k**0.5))
            .to(torch_ab)
            .cuda()
        )
        b_gpu = (
            torch.empty(l, n, k_local, dtype=torch.float32)
            .normal_()
            .mul_(1.0 / (k**0.5))
            .to(torch_ab)
            .cuda()
        )
        A = a_gpu.to(torch_d)
        B = b_gpu.to(torch_d)
    # D output + padded symmetric partials workspace + semaphores: RS D is a plain
    # (l, m/world, n) tensor, AR D this rank's view of full symmetric (l, m, n).
    d_arg, epi_reduce_args = make_epi_reduce_args(
        mode, torch_d, m, n, l, tile_M, tile_N, cluster_M, world_size, ws_dtype=torch_ws
    )

    # No host-side barriers in the loop: the kernel owns cross-invocation sync
    # (PDL-gated reducer warps + the spin-lock exit barrier).
    def fn():
        gemm(
            a_gpu,
            b_gpu,
            d_arg,
            None,  # C
            None,  # tile_count_semaphore
            tile_M,
            tile_N,
            cluster_M,
            cluster_N,
            epi_reduce_mode=mode,
            epi_reduce_args=epi_reduce_args,
            **bs_kwargs,
        )

    D_full = torch.empty(l, m, n, dtype=torch_d, device="cuda")
    D_rs = torch.empty(l, m_per_rank, n, dtype=torch_d, device="cuda")

    def fn_baseline():
        torch.bmm(A, B.mT, out=D_full)
        if mode == "reduce_scatter":
            # Per-batch: reduce_scatter_tensor splits dim 0, and the slab dim is m.
            for i in range(l):
                dist.reduce_scatter_tensor(D_rs[i], D_full[i])
        else:
            dist.all_reduce(D_full)

    if not args.skip_ref_check:
        fn()
        torch.cuda.synchronize()
        dist.barrier()
        fn_baseline()
        torch.cuda.synchronize()
        # d_arg is the output itself: RS the (l, m/world, n) slab; AR the full D.
        out = d_arg
        ref = D_rs if mode == "reduce_scatter" else D_full
        torch.testing.assert_close(out, ref, atol=args.tolerance, rtol=1e-3)
        if rank == 0:
            print("Ref check PASSED")

    flops = 2 * m * n * k_local * l
    dist.barrier()
    t_base = do_bench_all(fn_baseline, warmup=args.warmup_iterations, rep=args.iterations)
    dist.barrier()
    t_quack = do_bench_all(fn, warmup=args.warmup_iterations, rep=args.iterations)
    if rank == 0:
        print(f"cuBLAS+NCCL: {t_base:.3f} ms,  {flops / (t_base * 1e9):7.1f} TFLOP/s")
        print(f"quack      : {t_quack:.3f} ms,  {flops / (t_quack * 1e9):7.1f} TFLOP/s")
        print(f"  (quack speedup vs cuBLAS+NCCL: {t_base / t_quack:.2f}x)")

    dist.barrier()
    clean_distributed()


if __name__ == "__main__":
    run(parse_arguments())
    if int(os.environ.get("RANK", "0")) == 0:
        print("PASS")
