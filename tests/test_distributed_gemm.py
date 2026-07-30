# Copyright (c) 2026, QuACK team.
"""Multi-rank correctness tests for AllGatherRunner (quack/distributed/).

Runs under torchrun (one rank per GPU); the pytest entry point spawns
torchrun as a subprocess on the available GPUs.

    torchrun --nproc_per_node=4 tests/test_distributed_gemm.py
    pytest tests/test_distributed_gemm.py -x
"""

import os
import subprocess
import sys

import pytest
import torch

NUM_ITERS = 6  # exercises epoch monotonicity + double-buffer reuse


def _ag_gemm(runner, a_shard, b, d=None):
    """Plain AG+GEMM (D = A_full @ B^T) through the gather() context."""
    from quack.gemm import gemm as quack_gemm

    if d is None:
        d = torch.empty(runner.m_total, b.shape[0], dtype=runner.dtype, device=runner.device)
    with runner.gather(a_shard) as (a_full, ag_args):
        quack_gemm(a_full, b, d, None, None, 128, 256, 2, 1, ag_args=ag_args)
    return d


def _run_rank():
    import torch.distributed as dist

    from quack.distributed import AllGatherRunner

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group("nccl")
    device = torch.device("cuda", rank % torch.cuda.device_count())

    failures = []

    def check_plain_result(label, runner, a_shard, b, d):
        """Check both the GEMM result and the runner's last gathered buffer."""
        a_full_ref = torch.empty(runner.m_total, b.shape[1], dtype=b.dtype, device=device)
        dist.all_gather_into_tensor(a_full_ref, a_shard)
        ref = a_full_ref.float() @ b.float().T
        baseline = ref.to(b.dtype).float()
        err = (d.float() - ref).abs().max().item()
        tol = (baseline - ref).abs().max().item() + 1e-4
        gathered_err = (runner.gathered_a() - a_full_ref).abs().max().item()
        if err > 2 * tol or gathered_err != 0:
            failures.append(f"rank{rank} {label}: err={err} tol={tol} gathered_err={gathered_err}")

    for dtype, shard_m, n, k, delay_comm, chunks in [
        (torch.bfloat16, 1024, 2048, 4096, False, 1),
        (torch.bfloat16, 512, 1024, 8192, True, 1),  # late sends: gate must hold
        (torch.bfloat16, 1024, 2048, 4096, False, 4),  # chunked sends
        (torch.bfloat16, 2048, 1024, 4096, True, 8),  # chunked + late sends
        (torch.float16, 2048, 3072, 2048, False, 2),
    ]:
        torch.manual_seed(1234)  # same B on all ranks
        b = (torch.randn(n, k, dtype=dtype, device=device) / 8).contiguous()
        ag = AllGatherRunner(shard_m, k, dtype, device=device, arrival_chunks=chunks)
        m_total = shard_m * world_size
        for it in range(NUM_ITERS):
            torch.manual_seed(1000 * it + rank)  # per-rank, per-iter shard
            a_shard = (torch.randn(shard_m, k, dtype=dtype, device=device) / 8).contiguous()

            # Reference: real all-gather + fp32 matmul
            a_full_ref = torch.empty(m_total, k, dtype=dtype, device=device)
            dist.all_gather_into_tensor(a_full_ref, a_shard)
            ref = a_full_ref.float() @ b.float().T
            baseline = (a_full_ref.float() @ b.float().T).to(dtype).float()

            if delay_comm:
                # Hold the TRANSPORT stream back so data arrival (and its flag
                # writes) is provably late and the in-kernel gate must wait.
                # (An earlier version slept copy_stream, which only delays the
                # post-GEMM reuse barrier — it never stressed the gate.)
                with torch.cuda.stream(ag.push_stream):
                    torch.cuda._sleep(int(50e6))  # ~25 ms at 2 GHz

            d = _ag_gemm(ag, a_shard, b)
            torch.cuda.synchronize(device)

            err = (d.float() - ref).abs().max().item()
            tol = (baseline - ref).abs().max().item() + 1e-4
            gathered_err = (ag.gathered_a() - a_full_ref).abs().max().item()
            if err > 2 * tol or gathered_err != 0:
                failures.append(
                    f"rank{rank} dtype={dtype} shard_m={shard_m} n={n} k={k} chunks={chunks} "
                    f"delay={delay_comm} iter={it}: err={err} tol={tol} "
                    f"gathered_err={gathered_err}"
                )
    # --- Abstraction check: AllGather + GEMM+SiLU epilogue via the
    # runner.gather() CONTEXT — proves any epilogue-mod GEMM overlaps by
    # calling it inline in the with-body with ONE ag_args kwarg (same route
    # serves rope etc.).
    from quack.gemm_act import gemm_act

    dtype, shard_m, n, k = torch.bfloat16, 1024, 2048, 4096
    torch.manual_seed(4321)
    b = (torch.randn(n, k, dtype=dtype, device=device) / 8).contiguous()
    runner = AllGatherRunner(shard_m, k, dtype, device=device)
    m_total = shard_m * world_size
    postact = torch.empty(m_total, n, dtype=dtype, device=device)

    for it in range(3):
        torch.manual_seed(77 * it + rank)
        a_shard = (torch.randn(shard_m, k, dtype=dtype, device=device) / 8).contiguous()
        a_full_ref = torch.empty(m_total, k, dtype=dtype, device=device)
        dist.all_gather_into_tensor(a_full_ref, a_shard)
        acc = a_full_ref.float() @ b.float().T
        ref = torch.nn.functional.silu(acc)
        baseline = torch.nn.functional.silu(acc.to(dtype).float())
        with runner.gather(a_shard) as (a_full, ag_args):
            gemm_act(a_full, b, None, None, postact, None, "silu", 128, 256, 2, 1, ag_args=ag_args)
        torch.cuda.synchronize(device)
        err = (postact.float() - ref).abs().max().item()
        tol = (baseline - ref).abs().max().item() + 1e-4
        if err > 2 * tol:
            failures.append(f"rank{rank} AG+gemm_act(silu) iter={it}: err={err} tol={tol}")

    # --- Outer-capture smoke: the epoch is device-resident, so a caller may
    # capture any number of whole calls and replay.
    dtype, shard_m, n, k = torch.bfloat16, 1024, 1024, 4096
    torch.manual_seed(99)
    b = (torch.randn(n, k, dtype=dtype, device=device) / 8).contiguous()
    ag = AllGatherRunner(shard_m, k, dtype, device=device)
    m_total = shard_m * world_size
    a_in = torch.zeros(shard_m, k, dtype=dtype, device=device)
    d1 = torch.empty(m_total, n, dtype=dtype, device=device)
    d2 = torch.empty(m_total, n, dtype=dtype, device=device)
    for _ in range(2):  # warmup outside capture (also allocator warmup)
        _ag_gemm(ag, a_in, b, d1)
        _ag_gemm(ag, a_in, b, d2)
    torch.cuda.synchronize(device)
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    # THREE calls — deliberately ODD, the hard case: replaying it puts two
    # same-parity calls back to back across the replay boundary, which the
    # old chained per-parity epoch bump silently corrupted (reused epoch
    # value + same buffer => gates release on stale data); the global-row
    # epoch stays monotone. Also exercises a real intra-capture reuse edge
    # (call 3 waits ev_reuse recorded by call 1's barrier — the i -> i+2
    # edge only appears in captures of >= 3 calls).
    with torch.cuda.graph(g):
        _ag_gemm(ag, a_in, b, d1)
        _ag_gemm(ag, a_in, b, d2)
        _ag_gemm(ag, a_in, b, d1)
    g_parity = ag.last_capture_parity  # baked final buffer, constant across replays
    for it in range(3):
        torch.manual_seed(500 * it + rank)
        a_in.copy_((torch.randn(shard_m, k, dtype=dtype, device=device) / 8))
        g.replay()
        torch.cuda.synchronize(device)
        a_full_ref = torch.empty(m_total, k, dtype=dtype, device=device)
        dist.all_gather_into_tensor(a_full_ref, a_in)
        ref = a_full_ref.float() @ b.float().T
        for name, dd in (("d1", d1), ("d2", d2)):
            err = (dd.float() - ref).abs().max().item()
            baseline = (a_full_ref.float() @ b.float().T).to(dtype).float()
            tol = (baseline - ref).abs().max().item() + 1e-4
            if err > 2 * tol:
                failures.append(f"rank{rank} outer-capture {name} it={it}: err={err} tol={tol}")

    # --- Mode-mixing through quiesce(): after ODD-count replays, quiesce()
    # must (a) fence the replay work so an eager call is safe and (b) resync
    # last_parity to the graph's BAKED final buffer (without the resync,
    # gathered_a() was observed selecting the WRONG buffer after odd
    # replays). Uses the SYNC-FREE resync (caller passes the capture-parity
    # token); the bare syncing variant is cross-checked against it below.
    ag.quiesce(last_parity=g_parity)
    torch.manual_seed(4242 + rank)
    a_in.copy_((torch.randn(shard_m, k, dtype=dtype, device=device) / 8))
    d_eager = _ag_gemm(ag, a_in, b)
    torch.cuda.synchronize(device)
    # bare quiesce() derives the buffer from the device snapshots — must
    # agree with the token path (validates both resync variants)
    shadow = ag.last_parity
    ag.quiesce()
    if ag.last_parity != shadow:
        failures.append(
            f"rank{rank} quiesce resync mismatch: device={ag.last_parity} token={shadow}"
        )

    # --- Cross-stream replays: quiesce(replay_stream=...) must order itself
    # after replays launched on a NON-current stream (without the edge the
    # device-epoch read misses the replay and lands one call behind).
    replay_stream = torch.cuda.Stream(device)
    torch.manual_seed(777 + rank)
    a_in.copy_((torch.randn(shard_m, k, dtype=dtype, device=device) / 8))
    torch.cuda.synchronize(device)
    with torch.cuda.stream(replay_stream):
        g.replay()
    ag.quiesce(replay_stream=replay_stream)
    shadow_before = ag.last_parity
    d_eager = _ag_gemm(ag, a_in, b)
    torch.cuda.synchronize(device)
    a_full_ref = torch.empty(m_total, k, dtype=dtype, device=device)
    dist.all_gather_into_tensor(a_full_ref, a_in)
    ref = a_full_ref.float() @ b.float().T
    baseline = (a_full_ref.float() @ b.float().T).to(dtype).float()
    tol = (baseline - ref).abs().max().item() + 1e-4
    err = (d_eager.float() - ref).abs().max().item()
    gathered_err = (ag.gathered_a() - a_full_ref).abs().max().item()
    if err > 2 * tol or gathered_err != 0 or shadow_before != g_parity:
        failures.append(
            f"rank{rank} cross-stream quiesce: err={err} tol={tol} "
            f"gathered_err={gathered_err} parity={shadow_before} (baked {g_parity})"
        )
    a_full_ref = torch.empty(m_total, k, dtype=dtype, device=device)
    dist.all_gather_into_tensor(a_full_ref, a_in)
    ref = a_full_ref.float() @ b.float().T
    baseline = (a_full_ref.float() @ b.float().T).to(dtype).float()
    tol = (baseline - ref).abs().max().item() + 1e-4
    err = (d_eager.float() - ref).abs().max().item()
    gathered_err = (ag.gathered_a() - a_full_ref).abs().max().item()
    if err > 2 * tol or gathered_err != 0:
        failures.append(
            f"rank{rank} quiesce mode-mix: err={err} tol={tol} gathered_err={gathered_err}"
        )

    # --- Odd graph x even replay count: graph buffer addresses are baked, so
    # a one-call graph always finishes in the SAME physical buffer. Two
    # replays advance the global epoch by an even count but do not change that
    # final physical buffer. Neither arithmetic call-count parity nor global
    # device-epoch parity may be used to select gathered_a(). Distinct eager
    # and replay inputs make a wrong physical-buffer choice observable.
    # capture_lockstep=True also covers the paced captured schedule
    # (per-call join edges left in the graph) end to end.
    parity_ag = AllGatherRunner(shard_m, k, dtype, device=device, capture_lockstep=True)
    parity_eager = torch.zeros(shard_m, k, dtype=dtype, device=device)
    parity_in = torch.zeros_like(parity_eager)
    parity_d = torch.empty(m_total, n, dtype=dtype, device=device)
    for _ in range(2):
        _ag_gemm(parity_ag, parity_eager, b, parity_d)
    torch.cuda.synchronize(device)
    dist.barrier()

    parity_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(parity_graph):
        _ag_gemm(parity_ag, parity_in, b, parity_d)
    parity_token = parity_ag.last_capture_parity

    torch.manual_seed(8100 + rank)
    parity_in.copy_(torch.randn_like(parity_in) / 8)
    parity_graph.replay()
    torch.manual_seed(8200 + rank)
    parity_in.copy_(torch.randn_like(parity_in) / 8)
    parity_graph.replay()

    parity_ag.quiesce(last_parity=parity_token)
    check_plain_result("quiesce token physical parity", parity_ag, parity_in, b, parity_d)
    parity_ag.quiesce()
    check_plain_result("quiesce device physical parity", parity_ag, parity_in, b, parity_d)

    # --- Repeated graph <-> eager transitions through the sync-free quiesce
    # path. Rank skew keeps one peer's eager GEMM live while faster ranks reach
    # the boundary; omitting the eager->graph fence corrupts the next replay.
    mix_ag = AllGatherRunner(shard_m, k, dtype, device=device)
    mix_graph_in = torch.zeros(shard_m, k, dtype=dtype, device=device)
    mix_eager_in = torch.zeros_like(mix_graph_in)
    mix_graph_d = torch.empty(m_total, n, dtype=dtype, device=device)
    mix_eager_d = torch.empty_like(mix_graph_d)
    for _ in range(2):
        _ag_gemm(mix_ag, mix_graph_in, b, mix_graph_d)
    torch.cuda.synchronize(device)
    dist.barrier()

    mix_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(mix_graph):
        _ag_gemm(mix_ag, mix_graph_in, b, mix_graph_d)
    mix_token = mix_ag.last_capture_parity

    for it in range(2):
        torch.manual_seed(9000 + 100 * it + rank)
        mix_graph_in.copy_(torch.randn_like(mix_graph_in) / 8)
        mix_graph.replay()
        mix_ag.quiesce(last_parity=mix_token)
        check_plain_result(f"quiesce graph->eager iter={it}", mix_ag, mix_graph_in, b, mix_graph_d)

        torch.manual_seed(9050 + 100 * it + rank)
        mix_eager_in.copy_(torch.randn_like(mix_eager_in) / 8)
        if rank == world_size - 1:
            torch.cuda._sleep(int(50e6))
        _ag_gemm(mix_ag, mix_eager_in, b, mix_eager_d)
        # pure fence before the next replay: no graph work since the last
        # eager call, so the runner's own parity is already correct
        mix_ag.quiesce(last_parity=mix_ag.last_parity)
        check_plain_result(f"quiesce eager->graph iter={it}", mix_ag, mix_eager_in, b, mix_eager_d)

    # --- world_size == 1 degenerate path: the fast path skips transport,
    # flags, and gate, but must STILL bump the device parity snapshots, or
    # bare quiesce()'s buffer discovery cannot see replays (measured
    # bare_error=5.19 before the fix; token path was always fine).
    solo_group = None
    for r in range(world_size):  # new_group is collective: all ranks, every group
        group_r = dist.new_group([r])
        if r == rank:
            solo_group = group_r
    # (rendezvous on a 1-rank group makes torch log a scary-looking but
    # benign "fail to export multicast handle ... Gracefully skipping"
    # warning trace — the driver refuses 1-device multicast objects)
    solo_ag = AllGatherRunner(shard_m, k, dtype, group=solo_group, device=device)
    solo_in = torch.zeros(shard_m, k, dtype=dtype, device=device)
    solo_d = torch.empty(shard_m, n, dtype=dtype, device=device)
    for _ in range(2):
        _ag_gemm(solo_ag, solo_in, b, solo_d)
    torch.cuda.synchronize(device)
    solo_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(solo_graph):
        _ag_gemm(solo_ag, solo_in, b, solo_d)
    for seed in (6100, 6200):  # two replays: baked parity vs count parity diverge
        torch.manual_seed(seed + rank)
        solo_in.copy_(torch.randn_like(solo_in) / 8)
        solo_graph.replay()
    solo_ag.quiesce()  # bare: device discovery must find the baked buffer
    torch.cuda.synchronize(device)
    ref = solo_in.float() @ b.float().T
    baseline = ref.to(dtype).float()
    tol = (baseline - ref).abs().max().item() + 1e-4
    err = (solo_d.float() - ref).abs().max().item()
    gathered_err = (solo_ag.gathered_a() - solo_in).abs().max().item()
    if err > 2 * tol or gathered_err != 0:
        failures.append(
            f"rank{rank} ws==1 bare quiesce: err={err} tol={tol} gathered_err={gathered_err}"
        )

    fail_count = torch.tensor([len(failures)], device=device)
    dist.all_reduce(fail_count)
    if failures:
        print("\n".join(failures), file=sys.stderr, flush=True)
    dist.destroy_process_group()
    if int(fail_count.item()) > 0:
        sys.exit(1)
    if rank == 0:
        print("ALL AG+GEMM CHECKS PASSED", flush=True)


def _visible_gpus() -> str:
    """All GPUs, minus any with a large foreign allocation (shared node)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        free = [
            line.split(",")[0].strip()
            for line in out.strip().splitlines()
            if int(line.split(",")[1]) < 20000
        ]
        return ",".join(free)
    except Exception:
        return ",".join(str(i) for i in range(torch.cuda.device_count()))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_all_gather_gemm_multirank():
    gpus = _visible_gpus()
    nproc = min(len(gpus.split(",")), 8)
    assert nproc >= 2, f"not enough free GPUs: {gpus}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.setdefault("PYTHONPATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--standalone",
            os.path.abspath(__file__),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if res.returncode != 0:
        print(res.stdout[-4000:])
        print(res.stderr[-4000:], file=sys.stderr)
    assert res.returncode == 0
    assert "ALL AG+GEMM CHECKS PASSED" in res.stdout


if __name__ == "__main__":
    _run_rank()
