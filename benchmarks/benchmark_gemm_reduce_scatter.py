"""GEMM + fused reduce-scatter benchmark (SM100, epi_reduce_mode="reduce_scatter").

Each rank computes A_local @ B_local^T over its K-shard (k_local = k / world_size); the
epi_reduce warps multimem-reduce the partial D across ranks and each rank keeps its
M-slab (m / world_size rows). Baseline: torch.bmm + dist.reduce_scatter_tensor.

Usage:
    torchrun --nproc_per_node=8 benchmarks/benchmark_gemm_reduce_scatter.py \
        --mnkl 8192,4096,4096,1 --tile_shape_mnk 256,256 --cluster_shape_mnk 2,1
"""

import argparse
import os

import torch
import torch.distributed as dist

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as cutlass_utils
from cutlass.cute.runtime import from_dlpack

from quack.bench.bench_utils_dist import do_bench_all
from quack.dist_utils import (
    torchrun_init_nvshmem,
    torchrun_finalize_nvshmem,
    create_multicast_tensor,
    make_barrier_flags,
)
from quack.cute_dsl_utils import get_device_capacity
from quack.epi_reduce import EpiReduceArguments
from quack.gemm_default_epi import GemmDefaultSm100
from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args

ab_dtype = cutlass.BFloat16
acc_dtype = cutlass.Float32


def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        ) from e


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GEMM + fused reduce-scatter benchmark")
    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(8192, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated); k is global (sharded k/world per rank)",
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
        "--d_dtype",
        type=str,
        default="BFloat16",
        help="Output dtype (also the two_shot partials dtype): BFloat16/Float16/Float32.",
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
    torchrun_init_nvshmem()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size > 1, "launch with torchrun --nproc_per_node > 1"
    sm_major = get_device_capacity(torch.device("cuda"))[0]
    assert sm_major in (10, 11), f"GEMM+RS requires SM100 (B200/B300); got SM{sm_major}x"

    m, n, k, l = args.mnkl
    d_dtype = cutlass.dtype(args.d_dtype)
    tile_M, tile_N = args.tile_shape_mnk[:2]
    cluster_M, cluster_N = args.cluster_shape_mnk[:2]
    vec = 128 // d_dtype.width  # one 16B multimem vector
    assert m % world_size == 0, f"m ({m}) must be divisible by world_size ({world_size})"
    assert n % vec == 0, f"n ({n}) must be divisible by {vec} (16B multimem vectors)"
    k_local = k // world_size
    m_per_rank = m // world_size

    if rank == 0:
        print("Running SM100 GEMM + reduce-scatter with:")
        print(f"mnkl: {args.mnkl} (k_local: {k_local}), world_size: {world_size}")
        print(f"tile_shape_mnk: {args.tile_shape_mnk}, cluster_shape_mnk: {args.cluster_shape_mnk}")

    # A: (m, k_local, l) k-major, B: (n, k_local, l) k-major, both rank-seeded.
    # D: (m, n, l) n-major in symmetric memory (the multimem reduce reads partials there).
    torch.manual_seed(1111 + rank)
    torch_ab = cutlass_torch.dtype(ab_dtype)
    a_torch_cpu = (
        cutlass_torch.matrix(l, m, k_local, False, ab_dtype)
        .to(torch.float32)
        .normal_()
        .mul_(1.0 / (k**0.5))
        .to(torch_ab)
    )
    b_torch_cpu = (
        cutlass_torch.matrix(l, n, k_local, False, ab_dtype)
        .to(torch.float32)
        .normal_()
        .mul_(1.0 / (k**0.5))
        .to(torch_ab)
    )
    a_tensor, _ = cutlass_torch.cute_tensor_like(
        a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, _ = cutlass_torch.cute_tensor_like(
        b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    # D is (m, n, l) n-major: an (l, m, n)-contiguous source viewed as (m, n, l) gives that
    # layout, which create_multicast_tensor preserves into symmetric memory (write-only here).
    d_cpu = torch.empty(l, m, n, dtype=cutlass_torch.dtype(d_dtype)).permute(1, 2, 0)
    d_tensor, d_tensor_mc, d_torch_gpu, _, _, d_peer_tensors = create_multicast_tensor(
        d_cpu, d_dtype, leading_dim=1
    )

    # Per-CTA-tile producer flags, the per-SM sync barrier (own tensor: shape-independent),
    # and per-consumer-tile counters (local-only); see EpiReduceArguments.
    use_2cta = cluster_M % 2 == 0 and tile_M in (128, 256)
    cta_m = tile_M // (2 if use_2cta else 1)
    n_tiles = (n + tile_N - 1) // tile_N
    num_tiles = ((m + cta_m - 1) // cta_m) * n_tiles * l
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tf_torch, tf_torch_mc, tile_flags, tile_flags_mc = make_barrier_flags(num_tiles)
    sb_torch, sb_torch_mc, sync_barrier, sync_barrier_mc = make_barrier_flags(num_sms)
    slab_tiles_m = (m_per_rank + cta_m - 1) // cta_m
    counters_torch = torch.zeros(slab_tiles_m * n_tiles * l, dtype=torch.int32, device="cuda")
    counters = from_dlpack(counters_torch).mark_layout_dynamic()
    epi_reduce_args = EpiReduceArguments(
        mD_mc=d_tensor_mc,
        mD_peers=tuple(d_peer_tensors),
        tile_flags=tile_flags,
        tile_flags_mc=tile_flags_mc,
        sync_barrier=sync_barrier,
        sync_barrier_mc=sync_barrier_mc,
        consumer_counters=counters,
    )

    gemm = GemmDefaultSm100(
        acc_dtype=acc_dtype,
        a_dtype=ab_dtype,
        mma_tiler_mnk=(tile_M, tile_N),
        cluster_shape_mnk=(cluster_M, cluster_N, 1),
        epi_reduce_mode="reduce_scatter",
    )
    epi_args = GemmDefaultSm100.EpilogueArguments()
    max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
        cluster_M * cluster_N
    )
    sched_args = make_scheduler_args(max_active_clusters, 8, None, None)
    varlen_args = make_varlen_args(None, None, None)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        d_tensor,
        None,
        epi_args,
        sched_args,
        varlen_args,
        current_stream,
        None,
        None,
        epi_reduce_args,
    )

    # No host-side barriers in the loop: the kernel owns cross-invocation sync
    # (PDL-gated epi_reduce warps + the spin-lock exit barrier).
    def fn():
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled_gemm(
            a_tensor,
            b_tensor,
            d_tensor,
            None,
            epi_args,
            sched_args,
            varlen_args,
            stream,
            None,
            None,
            epi_reduce_args,
        )

    torch_d = cutlass_torch.dtype(d_dtype)
    A = a_torch_cpu.permute(2, 0, 1).contiguous().cuda().to(torch_d)
    B = b_torch_cpu.permute(2, 0, 1).contiguous().cuda().to(torch_d)
    D_full = torch.empty(l, m, n, dtype=torch_d, device="cuda")
    D_rs = torch.empty(l, m_per_rank, n, dtype=torch_d, device="cuda")

    def fn_baseline():
        torch.bmm(A, B.mT, out=D_full)
        # Per-batch: reduce_scatter_tensor splits dim 0, and the slab dim is m.
        for i in range(l):
            dist.reduce_scatter_tensor(D_rs[i], D_full[i])

    if not args.skip_ref_check:
        fn()
        torch.cuda.synchronize()
        dist.barrier()
        fn_baseline()
        torch.cuda.synchronize()
        # d_torch_gpu is the (m, n, l) view; own slab is on the m dim, permute to (l, m, n).
        slab = d_torch_gpu[rank * m_per_rank : (rank + 1) * m_per_rank].permute(2, 0, 1)
        torch.testing.assert_close(slab, D_rs, atol=args.tolerance, rtol=1e-3)
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
    # create_multicast_tensor / make_barrier_flags registered their frees via on_finalize;
    # this runs them (reverse order), then nvshmem.core.finalize() + destroy_process_group().
    torchrun_finalize_nvshmem()


if __name__ == "__main__":
    run(parse_arguments())
    if int(os.environ.get("RANK", "0")) == 0:
        print("PASS")
