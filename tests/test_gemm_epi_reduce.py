"""Distributed correctness tests for fused GEMM + reduce epilogues (RS / AR).

Marked `dist`: deselected by default, run with `pytest --dist-only` (never
under pytest-xdist — one pytest process launches multiple torchrun ranks).
"""

import argparse
import os
import subprocess
import sys

import pytest
import torch


WORLD_SIZES = [4]  # 4 is good default, 2 ranks can miss numerical/ordering bugs, 8 occupies full node
# (m, n, k, l, ab_dtype, d_dtype); dtypes per quack gemm test convention (bf16
# baseline, fp16 inputs, fp32 out), each non-bf16 pair on its own hard shape
CASES = [
    (4096, 4096, 4096, 1, "bfloat16", "bfloat16"),  # baseline
    (488, 1024, 1024, 2, "bfloat16", "bfloat16"),  # m_per_rank < cta_m: tiny grid, batched
    (528, 4104, 736, 3, "bfloat16", "bfloat16"),  # partial M/N tiles + K residue, batched
    (1032, 2056, 928, 2, "float16", "float16"),  # partial M/N tiles + K residue, batched
    (520, 1028, 672, 3, "bfloat16", "float32"),  # fp32 vec=4 path
]
pytestmark = pytest.mark.dist

def _run_gemm_epi_reduce(
    m, n, k, l=1, ab_dtype="bfloat16", d_dtype="bfloat16", use_epi_reduce="reduce_scatter"
):
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

    dtype_map = {
        "bfloat16": cutlass.BFloat16,
        "float16": cutlass.Float16,
        "float32": cutlass.Float32,
    }
    ab_dtype, d_dtype = dtype_map[ab_dtype], dtype_map[d_dtype]
    acc_dtype = cutlass.Float32
    tile_m, tile_n = 256, 256
    cluster_m, cluster_n = 2, 1
    vec = 128 // d_dtype.width
    assert n % vec == 0, f"n ({n}) must be divisible by {vec} (16B multimem vectors)"
    ab_vec = 128 // ab_dtype.width
    assert (k // world_size) % ab_vec == 0, (
        f"k_local ({k // world_size}) must be divisible by {ab_vec} (16B TMA alignment on A/B)"
    )

    k_local = k // world_size
    m_per_rank = m // world_size
    torch_ab = cutlass_torch.dtype(ab_dtype)
    torch_d = cutlass_torch.dtype(d_dtype)

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
    a_tensor, a_gpu = cutlass_torch.cute_tensor_like(
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
    tf_torch, _, tile_flags, tile_flags_mc = make_barrier_flags(num_tiles)
    _, _, sync_barrier, sync_barrier_mc = make_barrier_flags(num_sms)
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
        use_epi_reduce=use_epi_reduce,
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
    launch = lambda: compiled_gemm(
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
    # d_torch_gpu is the (m, n, l) view, permuted to (l, m, n): RS owns its m-slab;
    # AR holds the full reduced D on every rank after the multicast broadcast.
    if use_epi_reduce == "reduce_scatter":
        out = d_torch_gpu[rank * m_per_rank : (rank + 1) * m_per_rank].permute(2, 0, 1)
    else:
        out = d_torch_gpu.permute(2, 0, 1)

    # r2r: relaunch reuses tile_flags/sync_barrier/counters in place (stale-flag bugs
    # are invisible to a single launch); identical inputs must be bit-identical.
    runs = []
    for _ in range(2):
        launch()
        torch.cuda.synchronize()
        dist.barrier()
        runs.append(out.clone())
    assert torch.equal(runs[0], runs[1]), "r2r: relaunch not bit-identical"

    # Mutation loop: negate A each iter (bit-exact in every dtype) so stale/raced
    # values differ from expected; rotate a forced-skew straggler; no barrier between iters.
    expected = runs[1].clone()
    for it in range(4 * world_size):
        a_gpu.neg_()
        expected.neg_()
        if it % world_size == rank:
            torch.cuda._sleep(50_000_000)  # ~25 ms straggler: dwarfs one launch
        launch()
        torch.cuda.synchronize()
        assert torch.equal(out, expected), f"mutation loop iter {it}: stale or raced value"
    dist.barrier()

    # Flag wrap: flags/counters are monotonic (never reset), so int32 wrap is reachable;
    # seed just below the wrap and relaunch across it (A is restored, output = run 1).
    wrap_seed = torch.iinfo(torch.int32).max - world_size
    tf_torch.fill_(wrap_seed)
    counters_torch.fill_(wrap_seed)
    dist.barrier()
    for it in range(3):
        launch()
        torch.cuda.synchronize()
        assert torch.equal(out, runs[1]), f"flag-wrap launch {it}: mismatch"
    dist.barrier()

    # quack convention: fp32 ref is ground truth; kernel error < 2x same-dtype ref error.
    a_ref = a_torch_cpu.permute(2, 0, 1).contiguous().cuda()
    b_ref = b_torch_cpu.permute(2, 0, 1).contiguous().cuda()

    def epilogue_ref(dtype):
        d_full = torch.bmm(a_ref.to(dtype), b_ref.to(dtype).mT)
        if use_epi_reduce == "reduce_scatter":
            d_red = torch.empty(l, m_per_rank, n, dtype=dtype, device="cuda")
            for i in range(l):
                dist.reduce_scatter_tensor(d_red[i], d_full[i])
            return d_red.float()
        dist.all_reduce(d_full)
        return d_full.float()

    d_ref = epilogue_ref(torch.float32)
    d_pt = epilogue_ref(torch_d)
    torch.cuda.synchronize()

    d_err = (out.float() - d_ref).abs().max()
    d_base = (d_pt.to(torch_d).float() - d_ref).abs().max()
    if rank == 0:
        print(f"D err {d_err:.3e} base {d_base:.3e}")
    assert d_err < 2 * d_base + 1e-5, f"D err {d_err}, baseline {d_base}"
    if rank == 0:
        print("Ref check PASSED")

    dist.barrier()
    torchrun_finalize_nvshmem()


@pytest.mark.parametrize("world_size", WORLD_SIZES, ids=lambda w: f"world{w}")
@pytest.mark.parametrize("mode", ["reduce_scatter", "all_reduce"], ids=["rs", "ar"])
@pytest.mark.parametrize("m,n,k,l,ab_dtype,d_dtype", CASES)
def test_gemm_epi_reduce(m, n, k, l, ab_dtype, d_dtype, mode, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(world_size),
        __file__,
        *["--m", str(m), "--n", str(n), "--k", str(k), "--l", str(l)],
        *["--ab_dtype", ab_dtype, "--d_dtype", d_dtype, "--mode", mode],
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
    dtypes = ["bfloat16", "float16", "float32"]
    parser.add_argument("--ab_dtype", choices=dtypes, default="bfloat16")
    parser.add_argument("--d_dtype", choices=dtypes, default="bfloat16")
    parser.add_argument("--mode", choices=["reduce_scatter", "all_reduce"], default="reduce_scatter")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _run_gemm_epi_reduce(
        args.m, args.n, args.k, args.l, args.ab_dtype, args.d_dtype, args.mode
    )
