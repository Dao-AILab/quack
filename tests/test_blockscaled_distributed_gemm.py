# Copyright (c) 2026, QuACK team.
"""Multi-rank correctness tests for BlockScaledAllGatherRunner
(quack/distributed/block_scaled_allgather_gemm.py).

Runs under torchrun (one rank per GPU); the pytest entry point spawns
torchrun as a subprocess on the available GPUs.

    torchrun --nproc_per_node=4 tests/test_blockscaled_distributed_gemm.py
    pytest tests/test_blockscaled_distributed_gemm.py -x

Every gated result is compared BITWISE against the dense blockscaled GEMM
on NCCL-gathered operands (same tile config): the AG path must be exact,
not approximately right.
"""

import os
import subprocess
import sys

import pytest
import torch

SF_VEC = 32
TILE = (128, 128, 2, 1)
FMTS = ["mxfp8_e4m3", "mxfp8_e5m2", "mxfp4"]


def _run_rank():
    import torch.distributed as dist

    from quack.blockscaled.operand import BlockScaledOperand
    from quack.blockscaled.quantize import (
        pack_scale_2d_to_blocked_contig,
        to_mx,
        to_mxfp4,
    )
    from quack.distributed import BlockScaledAllGatherRunner
    from quack.gemm import gemm as quack_gemm

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group("nccl")
    device = torch.device("cuda", rank % torch.cuda.device_count())
    tm, tn, cm, cn = TILE

    failures = []
    a_dtypes = {
        "mxfp8_e4m3": torch.float8_e4m3fn,
        "mxfp8_e5m2": torch.float8_e5m2,
        "mxfp4": torch.float4_e2m1fn_x2,
    }

    def quantize(fmt, x):
        """(rows, k) bf16 -> (qdata in the format dtype, (rows, sf_k) e8m0)."""
        if fmt == "mxfp8_e4m3":
            q, sc = to_mx(x, SF_VEC)
        elif fmt == "mxfp8_e5m2":
            q, sc = to_mx(x, SF_VEC, elem_dtype=torch.float8_e5m2)
        else:  # mxfp4: (uint8 (rows, k/2), e8m0 (rows, k/32))
            q, sc = to_mxfp4(x, SF_VEC)
        return q.view(a_dtypes[fmt]), sc.contiguous()

    def shard_operand(fmt, a_q, a_sc, shard_m, k):
        scale = pack_scale_2d_to_blocked_contig(a_sc.view(1, shard_m, k // SF_VEC))
        return BlockScaledOperand(a_q, scale, fmt)

    def nccl_gathered(fmt, a_q, a_sc, m_total, k):
        """Reference gathered operand pieces via a real all-gather. Also pins
        the layout fact the transport rests on: packed shard SFAs concatenate
        into the packed full SFA (128-row M-blocks are the outermost dim)."""
        a_q_full_u8 = torch.empty(m_total, a_q.shape[1], dtype=torch.uint8, device=device)
        dist.all_gather_into_tensor(a_q_full_u8, a_q.view(torch.uint8).contiguous())
        sc_full = torch.empty(m_total, k // SF_VEC, dtype=torch.uint8, device=device)
        dist.all_gather_into_tensor(sc_full, a_sc.view(torch.uint8).contiguous())
        sfa_full = pack_scale_2d_to_blocked_contig(
            sc_full.view(torch.float8_e8m0fnu).view(1, m_total, k // SF_VEC)
        )
        packed_shard = pack_scale_2d_to_blocked_contig(
            a_sc.view(1, a_sc.shape[0], k // SF_VEC)
        )
        sfa_cat = torch.empty(sfa_full.numel(), dtype=torch.uint8, device=device)
        dist.all_gather_into_tensor(sfa_cat, packed_shard.view(torch.uint8).view(-1))
        if not torch.equal(sfa_cat, sfa_full.view(torch.uint8).view(-1)):
            failures.append(f"rank{rank} {fmt}: blocked-SF shard concat property violated")
        return a_q_full_u8.view(a_dtypes[fmt]), sfa_full

    def gated(runner, fmt_a, fmt_b, shard_op, b_q, sfb, d):
        with runner.gather(shard_op) as (a_op, ag_args):
            quack_gemm(
                a_op.qdata, b_q, d, None, None, tm, tn, cm, cn,
                SFA=a_op.scale, SFB=sfb,
                bs_format_a=fmt_a, bs_format_b=fmt_b, ag_args=ag_args,
            )
        return a_op

    # --- Full A x B format matrix, gated bitwise == dense on the same
    # operands; gathered bytes == the NCCL references. 3 iterations on the
    # A-format's runner exercise the epoch/parity rotation.
    shard_m, n, k = 1024, 2048, 4096
    m_total = shard_m * world_size
    runners = {fmt: BlockScaledAllGatherRunner(shard_m, k, fmt, device=device) for fmt in FMTS}
    for fmt_a in FMTS:
        for fmt_b in FMTS:
            torch.manual_seed(1234)  # same B on all ranks
            b_hp = (torch.randn(n, k, dtype=torch.bfloat16, device=device) / 8).contiguous()
            b_q, b_sc = quantize(fmt_b, b_hp)
            sfb = pack_scale_2d_to_blocked_contig(b_sc.view(1, n, k // SF_VEC))
            for it in range(3):
                torch.manual_seed(1000 * it + 10 * hash((fmt_a, fmt_b)) % 7919 + rank)
                a_hp = (torch.randn(shard_m, k, dtype=torch.bfloat16, device=device) / 8).contiguous()
                a_q, a_sc = quantize(fmt_a, a_hp)
                a_q_full, sfa_full = nccl_gathered(fmt_a, a_q, a_sc, m_total, k)
                d_ref = torch.empty(m_total, n, dtype=torch.bfloat16, device=device)
                quack_gemm(
                    a_q_full, b_q, d_ref, None, None, tm, tn, cm, cn,
                    SFA=sfa_full, SFB=sfb, bs_format_a=fmt_a, bs_format_b=fmt_b,
                )
                d = torch.empty_like(d_ref)
                a_op = gated(
                    runners[fmt_a], fmt_a, fmt_b,
                    shard_operand(fmt_a, a_q, a_sc, shard_m, k), b_q, sfb, d,
                )
                torch.cuda.synchronize(device)
                dist.barrier()  # peers' pushes into MY buffers must have landed
                torch.cuda.synchronize(device)
                label = f"rank{rank} {fmt_a} x {fmt_b} iter={it}"
                if not torch.equal(a_op.qdata.view(torch.uint8), a_q_full.view(torch.uint8)):
                    failures.append(f"{label}: gathered qdata bytes != NCCL reference")
                if not torch.equal(a_op.scale.view(torch.uint8), sfa_full.view(torch.uint8)):
                    failures.append(f"{label}: gathered SFA bytes != NCCL reference")
                if not torch.equal(d, d_ref):
                    failures.append(f"{label}: gated D != dense D (bitwise)")

    # --- Late sends: hold the transport stream back so SFA+qdata arrival is
    # provably late and the in-kernel gate must wait for both payloads.
    fmt = "mxfp8_e4m3"
    ag = runners[fmt]
    torch.manual_seed(1234)
    b_hp = (torch.randn(n, k, dtype=torch.bfloat16, device=device) / 8).contiguous()
    b_q, b_sc = quantize(fmt, b_hp)
    sfb = pack_scale_2d_to_blocked_contig(b_sc.view(1, n, k // SF_VEC))
    torch.manual_seed(4321 + rank)
    a_hp = (torch.randn(shard_m, k, dtype=torch.bfloat16, device=device) / 8).contiguous()
    a_q, a_sc = quantize(fmt, a_hp)
    a_q_full, sfa_full = nccl_gathered(fmt, a_q, a_sc, m_total, k)
    d_ref = torch.empty(m_total, n, dtype=torch.bfloat16, device=device)
    quack_gemm(a_q_full, b_q, d_ref, None, None, tm, tn, cm, cn,
               SFA=sfa_full, SFB=sfb, bs_format_a=fmt, bs_format_b=fmt)
    with torch.cuda.stream(ag.push_stream):
        torch.cuda._sleep(int(50e6))  # ~25 ms at 2 GHz
    d = torch.empty_like(d_ref)
    gated(ag, fmt, fmt, shard_operand(fmt, a_q, a_sc, shard_m, k), b_q, sfb, d)
    torch.cuda.synchronize(device)
    if not torch.equal(d, d_ref):
        failures.append(f"rank{rank} late-sends: gated D != dense D (bitwise)")

    # --- Plain-tensor shard: type-symmetric fall-through to the parent's
    # byte-only gather (SFA supplied statically here).
    d = torch.empty_like(d_ref)
    with ag.gather(a_q.view(torch.uint8).contiguous()) as (a_full_u8, ag_args):
        quack_gemm(
            a_full_u8.view(a_dtypes[fmt]), b_q, d, None, None, tm, tn, cm, cn,
            SFA=sfa_full, SFB=sfb, bs_format_a=fmt, bs_format_b=fmt, ag_args=ag_args,
        )
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.synchronize(device)
    if not torch.equal(a_full_u8, a_q_full.view(torch.uint8)):
        failures.append(f"rank{rank} plain-tensor: gathered bytes != NCCL reference")
    if not torch.equal(d, d_ref):
        failures.append(f"rank{rank} plain-tensor: D != dense D (bitwise)")

    # --- Outer-capture smoke: THREE whole calls (odd — the parity-hard case,
    # see test_distributed_gemm.py), replayed with mutated inputs.
    in_q = torch.zeros_like(a_q.view(torch.uint8))
    in_sf = torch.zeros(shard_m // 128, k // SF_VEC // 4, 32, 4, 4,
                        dtype=torch.uint8, device=device)
    cap_op = BlockScaledOperand(in_q.view(a_dtypes[fmt]), in_sf, fmt)
    d1 = torch.empty_like(d_ref)
    d2 = torch.empty_like(d_ref)
    for _ in range(2):  # warm every kernel the captured path touches
        gated(ag, fmt, fmt, cap_op, b_q, sfb, d1)
        gated(ag, fmt, fmt, cap_op, b_q, sfb, d2)
    torch.cuda.synchronize(device)
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gated(ag, fmt, fmt, cap_op, b_q, sfb, d1)
        gated(ag, fmt, fmt, cap_op, b_q, sfb, d2)
        gated(ag, fmt, fmt, cap_op, b_q, sfb, d1)
    for it in range(3):
        torch.manual_seed(500 * it + rank)
        a_hp = (torch.randn(shard_m, k, dtype=torch.bfloat16, device=device) / 8).contiguous()
        a_q_it, a_sc_it = quantize(fmt, a_hp)
        in_q.copy_(a_q_it.view(torch.uint8))
        in_sf.copy_(
            pack_scale_2d_to_blocked_contig(
                a_sc_it.view(1, shard_m, k // SF_VEC)
            ).view(torch.uint8).squeeze(0)
        )
        g.replay()
        torch.cuda.synchronize(device)
        dist.barrier()
        a_q_full, sfa_full = nccl_gathered(fmt, a_q_it, a_sc_it, m_total, k)
        quack_gemm(a_q_full, b_q, d_ref, None, None, tm, tn, cm, cn,
                   SFA=sfa_full, SFB=sfb, bs_format_a=fmt, bs_format_b=fmt)
        torch.cuda.synchronize(device)
        for name, dd in (("d1", d1), ("d2", d2)):
            if not torch.equal(dd, d_ref):
                failures.append(f"rank{rank} outer-capture {name} it={it}: D != dense (bitwise)")

    fail_count = torch.tensor([len(failures)], device=device)
    dist.all_reduce(fail_count)
    if failures:
        print("\n".join(failures), file=sys.stderr, flush=True)
    dist.destroy_process_group()
    if int(fail_count.item()) > 0:
        sys.exit(1)
    if rank == 0:
        print("ALL BLOCKSCALED AG+GEMM CHECKS PASSED", flush=True)


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
def test_blockscaled_all_gather_gemm_multirank():
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
    assert "ALL BLOCKSCALED AG+GEMM CHECKS PASSED" in res.stdout


if __name__ == "__main__":
    _run_rank()
