"""GEMM + fused reduce-scatter benchmark (SM100, use_epi_reduce="reduce_scatter").

Each rank computes A_local @ B_local^T over its K-shard (k_local = k / world_size); the
epi_reduce warps multimem-reduce the partial D across ranks and each rank keeps its
M-slab (m / world_size rows). Baseline: torch.bmm + dist.reduce_scatter_tensor.

Usage:
    torchrun --nproc_per_node=8 quack/bench/benchmark_gemm_reduce_scatter.py \
        --mnkl 8192,4096,4096,1 --tile_shape_mnk 256,256 --cluster_shape_mnk 2,1
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

import cuda.bindings.driver as cuda
import nvshmem.core
from cuda.core.experimental import Device

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as cutlass_utils
from cutlass.cute.runtime import from_dlpack

from quack.bench.bench_utils import do_bench_all
from quack.cute_dsl_utils import get_device_capacity
from quack.gemm_default_epi import GemmDefaultSm100
from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args

ab_dtype = cutlass.BFloat16
d_dtype = cutlass.BFloat16
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
    parser.add_argument("--tolerance", type=float, default=3e-02, help="Tolerance for validation")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=30, help="Benchmark iterations")
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")
    args = parser.parse_args()
    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    return args


def init_distributed_nvshmem():
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = Device(rank)
    dev.set_current()
    uid = nvshmem.core.get_unique_id(empty=(rank != 0))
    uid_tensor = torch.from_numpy(uid._data.view(np.uint8).copy()).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)
    nvshmem.core.init(device=dev, uid=uid, rank=rank, nranks=world_size, initializer_method="uid")
    return rank, world_size


def make_symmetric_mc_tensor(shape, torch_dtype, dtype, leading_dim):
    """Symmetric tensor + multicast view + per-rank peer views, as cute tensors."""
    torch_gpu = nvshmem.core.tensor(shape, dtype=torch_dtype)
    torch_gpu_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, torch_gpu)
    peer_torch = [nvshmem.core.get_peer_tensor(torch_gpu, r) for r in range(dist.get_world_size())]
    tensor = from_dlpack(torch_gpu, assumed_align=16)
    tensor.element_type = dtype
    tensor = tensor.mark_layout_dynamic(leading_dim=leading_dim)
    tensor = cutlass_torch.convert_cute_tensor(torch_gpu, tensor, dtype, is_dynamic_layout=True)
    tensor_mc = from_dlpack(torch_gpu_mc, assumed_align=16).mark_layout_dynamic(
        leading_dim=leading_dim
    )
    peer_tensors = [from_dlpack(t) for t in peer_torch]
    return tensor, tensor_mc, peer_tensors, torch_gpu, torch_gpu_mc


def run(args):
    rank, world_size = init_distributed_nvshmem()
    assert world_size > 1, "launch with torchrun --nproc_per_node > 1"
    sm_major = get_device_capacity(torch.device("cuda"))[0]
    assert sm_major in (10, 11), f"GEMM+RS requires SM100 (B200/B300); got SM{sm_major}x"

    m, n, k, l = args.mnkl
    assert l == 1, "GEMM+RS benchmark supports l=1 only"
    tile_M, tile_N = args.tile_shape_mnk[:2]
    cluster_M, cluster_N = args.cluster_shape_mnk[:2]
    assert m % (tile_M * world_size) == 0, (
        f"m ({m}) must be divisible by mma_tile_M * world_size ({tile_M} * {world_size}): "
        "output ownership is MMA-tile-granular in the slab scheduler"
    )
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
        .to(torch.float32).normal_().mul_(1.0 / (k**0.5)).to(torch_ab)
    )
    b_torch_cpu = (
        cutlass_torch.matrix(l, n, k_local, False, ab_dtype)
        .to(torch.float32).normal_().mul_(1.0 / (k**0.5)).to(torch_ab)
    )
    a_tensor, _ = cutlass_torch.cute_tensor_like(
        a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, _ = cutlass_torch.cute_tensor_like(
        b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    d_tensor, d_tensor_mc, d_peer_tensors, d_torch_gpu, d_torch_gpu_mc = make_symmetric_mc_tensor(
        (m, n, l), cutlass_torch.dtype(d_dtype), d_dtype, leading_dim=1
    )

    # Per-tile producer flags (one per CTA tile) + per-SM exit-barrier flags.
    use_2cta = cluster_M % 2 == 0 and tile_M in (128, 256)
    cta_m = tile_M // (2 if use_2cta else 1)
    num_tiles = (m // cta_m) * (n // tile_N)
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    bf_torch = nvshmem.core.tensor((num_tiles + num_sms,), dtype=torch.int32)
    bf_torch.fill_(0)
    bf_torch_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, bf_torch)
    bf = from_dlpack(bf_torch).mark_layout_dynamic()
    bf_mc = from_dlpack(bf_torch_mc).mark_layout_dynamic()

    gemm = GemmDefaultSm100(
        acc_dtype=acc_dtype,
        a_dtype=ab_dtype,
        mma_tiler_mnk=(tile_M, tile_N),
        cluster_shape_mnk=(cluster_M, cluster_N, 1),
        use_epi_reduce="reduce_scatter",
    )
    epi_args = GemmDefaultSm100.EpilogueArguments()
    max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
        cluster_M * cluster_N
    )
    sched_args = make_scheduler_args(max_active_clusters, 8, None, None)
    varlen_args = make_varlen_args(None, None, None)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_gemm = cute.compile(
        gemm, a_tensor, b_tensor, d_tensor, None, epi_args, sched_args, varlen_args,
        current_stream, None, None, d_tensor_mc, d_peer_tensors, bf, bf_mc,
    )

    # No host-side barriers in the loop: the kernel owns cross-invocation sync
    # (PDL-gated epi_reduce warps + the spin-lock exit barrier).
    def fn():
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled_gemm(
            a_tensor, b_tensor, d_tensor, None, epi_args, sched_args, varlen_args,
            stream, None, None, d_tensor_mc, d_peer_tensors, bf, bf_mc,
        )

    A = a_torch_cpu.permute(2, 0, 1).contiguous().cuda()
    B = b_torch_cpu.permute(2, 0, 1).contiguous().cuda()
    D_full = torch.empty(l, m, n, dtype=torch_ab, device="cuda")
    D_rs = torch.empty(l, m_per_rank, n, dtype=torch_ab, device="cuda")

    def fn_baseline():
        torch.bmm(A, B.mT, out=D_full)
        dist.reduce_scatter_tensor(D_rs, D_full)

    if not args.skip_ref_check:
        fn()
        torch.cuda.synchronize()
        dist.barrier()
        fn_baseline()
        torch.cuda.synchronize()
        slab = d_torch_gpu.permute(2, 0, 1)[:, rank * m_per_rank : (rank + 1) * m_per_rank]
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
    for t in (bf_torch_mc, bf_torch, d_torch_gpu_mc, d_torch_gpu):
        nvshmem.core.free_tensor(t)
    nvshmem.core.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    run(parse_arguments())
    if int(os.environ.get("RANK", "0")) == 0:
        print("PASS")
