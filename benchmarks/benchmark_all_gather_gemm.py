# Copyright (c) 2026, QuACK team.
"""8-rank AllGather+GEMM benchmark: quack overlapped AG+GEMM vs baselines.

Methods:
  gemm_only   local persistent GEMM on the pre-gathered A (compute roof)
  quack_ag    this work: reverse-ring CE push + flag-gated shard-rotated persistent GEMM
  nccl_seq    NCCL all_gather_into_tensor, then the same quack GEMM (exposed comm)
  torch_fused torch.ops.symm_mem.fused_all_gather_matmul (if available)

Protocol (shared-node rules): methods interleaved per round, per-launch CUDA
events, median over rounds, cross-rank max reported. Run:

  torchrun --nproc_per_node=8 benchmarks/benchmark_all_gather_gemm.py
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from quack.distributed import AllGatherRunner
from quack.gemm import AllGatherArguments, gemm as quack_gemm


def bench_shape(m_total, n_local, k, dtype, rank, world_size, device, rounds, args):
    shard_m = m_total // world_size
    torch.manual_seed(1234)
    b = (torch.randn(n_local, k, dtype=dtype, device=device) / 8).contiguous()
    torch.manual_seed(1000 + rank)
    a_shard = (torch.randn(shard_m, k, dtype=dtype, device=device) / 8).contiguous()
    a_full = torch.empty(m_total, k, dtype=dtype, device=device)
    dist.all_gather_into_tensor(a_full, a_shard)
    d = torch.empty(m_total, n_local, dtype=dtype, device=device)

    ag = AllGatherRunner(shard_m, k, dtype, device=device)
    gemm_cfg = dict(
        tile_M=args.tile_m, tile_N=args.tile_n, cluster_M=args.cluster_m, cluster_N=args.cluster_n
    )

    def run_gemm_only():
        quack_gemm(
            a_full,
            b,
            d,
            None,
            None,
            gemm_cfg["tile_M"],
            gemm_cfg["tile_N"],
            gemm_cfg["cluster_M"],
            gemm_cfg["cluster_N"],
            max_swizzle_size=args.swizzle,
        )

    def ag_gemm(a):
        with ag.gather(a) as (a_buf, ag_args):
            quack_gemm(
                a_buf,
                b,
                d,
                None,
                None,
                gemm_cfg["tile_M"],
                gemm_cfg["tile_N"],
                gemm_cfg["cluster_M"],
                gemm_cfg["cluster_N"],
                max_swizzle_size=args.swizzle,
                ag_args=ag_args,
            )

    def run_quack_ag():
        ag_gemm(a_shard)

    # Zero-copy variant: the producer wrote A directly into the symmetric
    # buffer (both double-buffer slots prefilled below), matching PK's
    # methodology where local staging is outside the timed loop.
    for _ in range(len(ag.bufs)):
        ag.next_local_slot().copy_(a_shard)
        ag.last_parity ^= 1  # two flips = restored; visits both slots
    torch.cuda.synchronize(device)

    def run_quack_ag_zc():
        ag_gemm(ag.next_local_slot())

    def run_nccl_seq():
        dist.all_gather_into_tensor(a_full, a_shard)
        run_gemm_only()

    # AG schedule + gating with flags pre-satisfied and zero comm: isolates the
    # kernel-side cost of the rotated shard-major schedule + flag checks.
    preflags = torch.full((world_size,), 2**30, dtype=torch.int32, device=device)
    # preset epoch <= preset flags: the gate (flag - epoch >= 0, modular) is
    # always satisfied.
    pre_epoch = torch.ones(1, dtype=torch.int32, device=device)

    def run_quack_ag_noflags():
        quack_gemm(
            a_full,
            b,
            d,
            None,
            None,
            gemm_cfg["tile_M"],
            gemm_cfg["tile_N"],
            gemm_cfg["cluster_M"],
            gemm_cfg["cluster_N"],
            max_swizzle_size=args.swizzle,
            ag_args=AllGatherArguments(
                flags=preflags, epoch=pre_epoch, num_shards=world_size, first_shard=rank
            ),
        )

    methods = {
        "gemm_only": run_gemm_only,
        "quack_ag_noflags": run_quack_ag_noflags,
        "quack_ag": run_quack_ag,
        "quack_ag_zc": run_quack_ag_zc,
        "nccl_seq": run_nccl_seq,
    }

    if args.torch_fused:
        group_name = dist.group.WORLD.group_name
        a_shard_symm = symm_mem.empty(shard_m, k, dtype=dtype, device=device)
        symm_mem.rendezvous(a_shard_symm, group_name)
        a_shard_symm.copy_(a_shard)

        def run_torch_fused():
            torch.ops.symm_mem.fused_all_gather_matmul(a_shard_symm, [b.T], 0, group_name)

        methods["torch_fused"] = run_torch_fused

    # correctness spot-check before timing
    ref = a_full.to(dtype).float() @ b.float().T
    run_quack_ag()
    torch.cuda.synchronize(device)
    err = (d.float() - ref).abs().max().item()
    d2 = torch.empty_like(d)
    quack_gemm(
        a_full,
        b,
        d2,
        None,
        None,
        gemm_cfg["tile_M"],
        gemm_cfg["tile_N"],
        gemm_cfg["cluster_M"],
        gemm_cfg["cluster_N"],
        max_swizzle_size=args.swizzle,
    )
    torch.cuda.synchronize(device)
    tol = (d2.float() - ref).abs().max().item() + 1e-4
    assert err <= 2 * tol, f"rank{rank} quack_ag mismatch: {err} vs tol {tol}"

    # warmup (compiles, symm rendezvous, clock ramp)
    for fn in methods.values():
        for _ in range(3):
            fn()
    torch.cuda.synchronize(device)

    # --- CUDA-graph modes: capture GRAPH_CALLS calls (even count), replay.
    # quack_ag_graph: current design — gather()'s capture join restores the
    #   dep set, so side branches are graph LEAVES; interior calls keep the
    #   full 2-buffer slack and the barrier hits the critical path only at
    #   the replay boundary (whole-graph completion).
    # quack_ag_graph_lockstep: emulates a plain per-call join (barrier_i ->
    #   compute_{i+1} edges after every captured call) — the 1-buffer
    #   schedule a naive capture join produces. A/B for the dep-restore.
    GRAPH_CALLS = 4

    def _capture(lockstep):
        dist.barrier()
        ag.capture_lockstep = lockstep  # read at capture time, bakes per graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(GRAPH_CALLS):
                ag_gemm(a_shard)
        ag.capture_lockstep = False
        return g

    for _ in range(2):
        ag_gemm(a_shard)
    torch.cuda.synchronize(device)
    g_leaf = _capture(lockstep=False)
    g_lockstep = _capture(lockstep=True)
    methods["quack_ag_graph"] = g_leaf.replay
    methods["quack_ag_graph_lockstep"] = g_lockstep.replay
    calls_per_inv = {name: 1 for name in methods}
    calls_per_inv["quack_ag_graph"] = GRAPH_CALLS
    calls_per_inv["quack_ag_graph_lockstep"] = GRAPH_CALLS
    for _ in range(3):
        g_leaf.replay()
        g_lockstep.replay()
    torch.cuda.synchronize(device)

    # Steady-state protocol: one barrier per (method, round), then BURST
    # back-to-back iterations timed as a block. This measures the pipelined
    # regime (cross-iteration overlap allowed — the training-loop reality)
    # instead of injecting per-iteration cross-rank skew into every
    # distributed method. Medians over rounds, cross-rank max.
    BURST = 10
    times = {name: [] for name in methods}
    ev = {
        name: [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(rounds)
        ]
        for name in methods
    }
    for r in range(rounds):
        for name, fn in methods.items():
            dist.barrier()
            torch.cuda.synchronize(device)
            s_ev, e_ev = ev[name][r]
            fn()  # one un-timed iteration absorbs the post-barrier skew
            s_ev.record()
            for _ in range(BURST):
                fn()
            e_ev.record()
            torch.cuda.synchronize(device)
    for name in methods:
        times[name] = [s.elapsed_time(e) / (BURST * calls_per_inv[name]) for s, e in ev[name]]

    flops = 2 * m_total * n_local * k
    results = {}
    for name, ts in times.items():
        med = statistics.median(ts)
        t = torch.tensor([med], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        results[name] = t.item()
    if rank == 0:
        roof = results["gemm_only"]
        print(
            f"\nM={m_total} N_local={n_local} K={k} {dtype} TP={world_size} "
            f"(shard {shard_m}x{k}, {2 * shard_m * k / 1e6:.0f} MB/shard... "
            f"{a_shard.numel() * a_shard.element_size() / 1e6:.1f} MB)"
        )
        for name, med in results.items():
            tflops = flops / (med * 1e-3) / 1e12
            print(
                f"  {name:<12} {med:8.3f} ms  {tflops:7.0f} TFLOPS  "
                f"overhead vs roof {100 * (med / roof - 1):6.1f}%"
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=str,
        default="8192,2048,8192;16384,2048,8192;16384,4096,8192;32768,2048,8192",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--cluster-m", type=int, default=2)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--swizzle", type=int, default=8)
    parser.add_argument("--torch-fused", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    dist.init_process_group("nccl")
    dtype = getattr(torch, args.dtype)

    for shape in args.shapes.split(";"):
        m_total, n_local, k = map(int, shape.split(","))
        bench_shape(m_total, n_local, k, dtype, rank, world_size, device, args.rounds, args)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
