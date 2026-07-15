"""Distributed correctness tests for fused GEMM + reduce-scatter epilogues.

Run this file on its own, not under pytest-xdist. The conftest xdist hook
narrows CUDA_VISIBLE_DEVICES to one GPU per worker, while this test needs one
pytest process to launch multiple torchrun ranks.
"""

import argparse
import os
import subprocess
import sys

import pytest
import torch


WORLD = int(os.environ.get("QUACK_DIST_WORLD_SIZE", "2"))
CASES = [(4096, 4096, 4096), (8192, 4096, 4096), (4104, 4096, 4096)]
TOLERANCES = {
    torch.bfloat16: (3e-2, 1e-3),
    torch.float16: (3e-2, 1e-3),
    torch.float32: (1e-4, 1e-4),
}


def _dist_skip_reason():
    if os.environ.get("QUACK_RUN_DIST_TESTS") != "1":
        return "set QUACK_RUN_DIST_TESTS=1 to run distributed tests"
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return "distributed torchrun tests must run outside pytest-xdist"
    if not torch.cuda.is_available():
        return "CUDA required"
    if torch.cuda.device_count() < WORLD:
        return f"requires >= {WORLD} visible GPUs"
    if torch.cuda.get_device_capability(0)[0] not in (10, 11):
        return "requires SM100/SM110"
    return None


_SKIP_REASON = _dist_skip_reason()
pytestmark = [pytest.mark.skip(reason=_SKIP_REASON)] if _SKIP_REASON else []


def _run_gemm_epi_reduce(m, n, k, l=1):
    import cuda.bindings.driver as cuda
    import torch.distributed as dist

    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as cutlass_utils
    from cutlass.cute.runtime import from_dlpack

    from quack.cute_dsl_utils import get_device_capacity
    from quack.dist_utils import (
        torchrun_init_nvshmem,
        torchrun_finalize_nvshmem,
        create_multicast_tensor,
        make_barrier_flags,
    )
    from quack.gemm_default_epi import GemmDefaultSm100
    from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args

    torchrun_init_nvshmem()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size > 1, "launch with torchrun --nproc_per_node > 1"
    sm_major = get_device_capacity(torch.device("cuda"))[0]
    assert sm_major in (10, 11), f"GEMM+RS requires SM100/SM110; got SM{sm_major}x"
    assert m % world_size == 0, f"m ({m}) must be divisible by world_size ({world_size})"
    assert k % world_size == 0, f"k ({k}) must be divisible by world_size ({world_size})"

    ab_dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32
    d_dtype = cutlass.BFloat16
    tile_m, tile_n = 256, 256
    cluster_m, cluster_n = 2, 1
    vec = 128 // d_dtype.width
    assert n % vec == 0, f"n ({n}) must be divisible by {vec} (16B multimem vectors)"

    k_local = k // world_size
    m_per_rank = m // world_size
    torch_ab = cutlass_torch.dtype(ab_dtype)
    torch_d = cutlass_torch.dtype(d_dtype)
    atol, rtol = TOLERANCES[torch_d]

    torch.manual_seed(1111 + rank)
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
    d_cpu = torch.empty(l, m, n, dtype=torch_d).permute(1, 2, 0)
    d_tensor, d_tensor_mc, d_torch_gpu, _, _, d_peer_tensors = create_multicast_tensor(
        d_cpu, d_dtype, leading_dim=1
    )

    use_2cta = cluster_m % 2 == 0 and tile_m in (128, 256)
    cta_m = tile_m // (2 if use_2cta else 1)
    n_tiles = (n + tile_n - 1) // tile_n
    num_tiles = ((m + cta_m - 1) // cta_m) * n_tiles * l
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tf_torch, tf_torch_mc, tile_flags, tile_flags_mc = make_barrier_flags(num_tiles)
    sb_torch, sb_torch_mc, sync_barrier, sync_barrier_mc = make_barrier_flags(num_sms)
    slab_tiles_m = (m_per_rank + cta_m - 1) // cta_m
    counters_torch = torch.zeros(slab_tiles_m * n_tiles * l, dtype=torch.int32, device="cuda")
    counters = from_dlpack(counters_torch).mark_layout_dynamic()
    epi_reduce_args = GemmDefaultSm100.EpiReduceArguments(
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
        mma_tiler_mnk=(tile_m, tile_n),
        cluster_shape_mnk=(cluster_m, cluster_n, 1),
        use_epi_reduce="reduce_scatter",
    )
    epi_args = GemmDefaultSm100.EpilogueArguments()
    max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
        cluster_m * cluster_n
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
    torch.cuda.synchronize()
    dist.barrier()

    a_ref = a_torch_cpu.permute(2, 0, 1).contiguous().cuda().to(torch_d)
    b_ref = b_torch_cpu.permute(2, 0, 1).contiguous().cuda().to(torch_d)
    d_full = torch.empty(l, m, n, dtype=torch_d, device="cuda")
    d_rs = torch.empty(l, m_per_rank, n, dtype=torch_d, device="cuda")
    torch.bmm(a_ref, b_ref.mT, out=d_full)
    for i in range(l):
        dist.reduce_scatter_tensor(d_rs[i], d_full[i])
    torch.cuda.synchronize()

    # d_torch_gpu is the (m, n, l) view; own slab is on the m dim, permute to (l, m, n).
    slab = d_torch_gpu[rank * m_per_rank : (rank + 1) * m_per_rank].permute(2, 0, 1)
    torch.testing.assert_close(slab, d_rs, atol=atol, rtol=rtol)
    if rank == 0:
        print("Ref check PASSED")

    dist.barrier()
    torchrun_finalize_nvshmem()


@pytest.mark.parametrize("m,n,k", CASES)
def test_gemm_epi_reduce(m, n, k):
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(WORLD),
        __file__,
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
    ]
    result = subprocess.run(
        cmd,
        env=env,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout
    assert "Ref check PASSED" in result.stdout, result.stdout


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--l", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _run_gemm_epi_reduce(args.m, args.n, args.k, args.l)
