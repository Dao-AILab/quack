# Copyright (c) 2026, QuACK team.
"""Blockscaled (MXFP8) AllGather+GEMM benchmark: quack staged-SFA transport vs
torch-native and async-TP baselines.

Methods (all forward AG+GEMM, D = all_gather_M(A) @ B^T):

    bf16_roof        dense bf16 quack GEMM on pre-gathered A (compute roof)
    mx_roof          dense mxfp8 quack GEMM on pre-gathered A/SFA (compute roof;
                     tile config picked by a small built-in sweep per shape)
    quack_bf16_ag    AllGatherRunner + gated bf16 GEMM
    quack_mx_staged  this work: BlockScaledAllGatherRunner — fp8 bytes AND
                     blocked scale factors ride the CE push under one flag set
    quack_mx_nccl    quack byte transport for A + exposed NCCL gather for the
                     packed SFA (isolates the staged-SFA win)
    torch_bf16       dist.all_gather_into_tensor + torch.matmul
    torch_mx         dist.all_gather_into_tensor (bytes + packed SFA) +
                     torch._scaled_mm with e8m0 blocked scales (skipped if the
                     torch build rejects mx scales)
    asynctp_bf16     torch.ops.symm_mem.fused_all_gather_matmul (async-TP)
    asynctp_mx       torch.ops.symm_mem.fused_all_gather_scaled_matmul with mx
                     scales — SKIPS on current torch: the op validates the A
                     scale against the shard rowcount, which only composes with
                     per-row scales; blocked 128x4-atom scales are rejected.
                     (At K=4096 the blocked-scale view coincidentally has M
                     rows and appears to work — do not be fooled.)
    asynctp_fp8row   async-TP's native fp8 recipe: rowwise scales through
                     fused_all_gather_scaled_matmul (a coarser numerics
                     contract than mxfp8, shown for context)

Protocol: every method is CUDA-graph captured (1 call) and burst-timed in
interleaved rounds — one barrier per (method, round), one un-timed replay to
absorb post-barrier skew, then an event pair over BURST back-to-back replays;
medians over rounds, cross-rank max. Methods that refuse capture fall back to
eager bursts and are marked [eager]. Every method's output is checked against
a reference before it is timed (fp32-dequant product for fp8 methods) and is
skipped loudly if wrong — a broken baseline must not be timed.

Reporting includes the 100% speed-of-light wall clock (2*M*N*K / dense
datasheet peak for the method's MMA dtype) and each method's attained %SOL.

`torch._scaled_mm` mxfp8 recipe used by torch_mx: pass the cuBLAS-blocked SFA
bytes viewed as float8_e8m0fnu with shape (-1, 128); mat2 column-major.

Run (WORLD_SIZE = TP), e.g.:

  torchrun --nproc_per_node=4 benchmarks/benchmark_blockscaled_all_gather_gemm.py

On hosts with more GPUs than the TP you want, pass --tp N: ranks >= N park at
a barrier while the first N form the benchmark group.
"""

import argparse
import os
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist

from quack.blockscaled.operand import BlockScaledOperand
from quack.blockscaled.quantize import pack_scale_2d_to_blocked_contig, to_mx
from quack.distributed import AllGatherRunner, BlockScaledAllGatherRunner
from quack.gemm import gemm as quack_gemm

SF_VEC = 32
ROUNDS, BURST = 10, 10
_PG = None  # benchmark process group (TP subgroup or WORLD), set in main()
MX = dict(bs_format_a="mxfp8_e4m3", bs_format_b="mxfp8_e4m3")
# Power-of-2 roundings of large-model tensor-parallel GEMM shapes (M, N, K).
DEFAULT_SHAPES = [
    (16384, 16384, 8192),
    (16384, 8192, 8192),
    (16384, 4096, 8192),
    (8192, 16384, 8192),
]
MX_ROOF_CFGS = [(256, 128, 2, 1), (256, 192, 2, 1), (128, 256, 2, 1)]

# Dense tensor-core datasheet peaks (TFLOP/s); fp8 is 2x bf16 on these parts.
_PEAK_TFLOPS_BF16 = {
    "NVIDIA H100 80GB HBM3": 989.5,
    "NVIDIA B200": 2250.0,
    "NVIDIA GB200": 2500.0,
    "NVIDIA GB300": 2500.0,
}


def _peak_tflops(dtype):
    peak = _PEAK_TFLOPS_BF16.get(torch.cuda.get_device_name())
    if peak is None:
        return None
    return 2 * peak if dtype.itemsize == 1 else peak


def _ag_into(dst, src):
    dist.all_gather_into_tensor(dst, src, group=_PG)


def _print0(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def _tflops(M, N, K, time_us):
    return 2 * M * N * K / (time_us * 1e-6) / 1e12


def _method_math_dtype(name):
    """The MMA dtype a method's FLOPs run at (its SOL reference)."""
    return torch.float8_e4m3fn if ("mx" in name or "fp8" in name) else torch.bfloat16


def _sol_us(M, N, K, name):
    """100%-speed-of-light wall clock for the method's math dtype."""
    peak = _peak_tflops(_method_math_dtype(name))
    if not peak:
        return float("nan")
    return 2 * M * N * K / (peak * 1e12) * 1e6


class MethodTimer:
    """Interleaved graph-replay (or eager-fallback) burst timing."""

    def __init__(self, device, rank):
        self.device = device
        self.rank = rank
        self.entries = []  # (name, replay_fn, mode)

    def add(self, name, fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=_PG)
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize(self.device)
            self.entries.append((name, g.replay, "graph"))
        except Exception as e:
            torch.cuda.synchronize(self.device)
            dist.barrier(group=_PG)
            _print0(
                self.rank,
                f"  [{name}] graph capture failed "
                f"({type(e).__name__}: {str(e)[:120]}) — timing eagerly",
            )
            self.entries.append((name, fn, "eager"))

    def run(self):
        times = {name: [] for name, _, _ in self.entries}
        for _ in range(ROUNDS):
            for name, replay, _ in self.entries:
                dist.barrier(group=_PG)
                torch.cuda.synchronize(self.device)
                replay()  # one un-timed iteration absorbs post-barrier skew
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                for _ in range(BURST):
                    replay()
                e_ev.record()
                torch.cuda.synchronize(self.device)
                times[name].append(s_ev.elapsed_time(e_ev) / BURST)
        out = {}
        for name, _, mode in self.entries:
            med = statistics.median(times[name])
            t = torch.tensor([med], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=_PG)
            out[name] = (t.item() * 1000, mode)
        return out


def _check(rank, name, d, ref, tol) -> bool:
    err = (d.float() - ref).abs().max().item()
    flag = torch.tensor([1.0 if err > tol else 0.0], device=d.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=_PG)
    if flag.item() > 0:
        _print0(
            rank,
            f"  [{name}] SKIP: correctness check failed (err={err:.3e} "
            f"tol={tol:.3e}) — not timing a wrong result",
        )
        return False
    _print0(rank, f"  [{name}] correctness OK (err={err:.2e})")
    return True


def run_shape(rank, ws, device, M, N, K):
    m = M // ws
    _print0(rank, f"\n#### TP={ws} M={M} N={N} K={K} (shard_m={m})")

    torch.manual_seed(1234)
    b_hp = (torch.randn(N, K, device=device, dtype=torch.bfloat16) * K**-0.5).contiguous()
    torch.manual_seed(1000 + rank)
    a_hp = (torch.randn(m, K, device=device, dtype=torch.bfloat16) * K**-0.5).contiguous()
    a_hp_full = torch.empty(M, K, dtype=torch.bfloat16, device=device)
    _ag_into(a_hp_full, a_hp)

    b_q, b_sc = to_mx(b_hp, SF_VEC)
    sfb = pack_scale_2d_to_blocked_contig(b_sc.view(1, N, K // SF_VEC))
    a_q, a_sc = to_mx(a_hp, SF_VEC)
    a_q_u8 = a_q.view(torch.uint8).contiguous()
    a_sc = a_sc.contiguous()
    a_q_full = torch.empty(M, K, dtype=torch.uint8, device=device)
    _ag_into(a_q_full, a_q_u8)
    a_q_full = a_q_full.view(torch.float8_e4m3fn)
    a_sc_full = torch.empty(M, K // SF_VEC, dtype=a_sc.dtype, device=device)
    _ag_into(a_sc_full.view(torch.uint8), a_sc.view(torch.uint8))
    sfa_static = pack_scale_2d_to_blocked_contig(a_sc_full.view(1, M, K // SF_VEC))

    # fp32 dequant reference for loose correctness checks
    ref = (a_q_full.float() * a_sc_full.float().repeat_interleave(SF_VEC, dim=-1)) @ (
        b_q.float() * b_sc.float().repeat_interleave(SF_VEC, dim=-1)
    ).T
    ref_bf16 = a_hp_full.float() @ b_hp.float().T
    mx_tol = ref.abs().max().item() * 5e-2 + 1e-2
    bf16_tol = ref_bf16.abs().max().item() * 5e-2 + 1e-2

    d = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    # --- mx roof tile sweep -> best cfg
    timer = MethodTimer(device, rank)
    cfgs = [c for c in MX_ROOF_CFGS if m % (c[0] * c[2]) == 0]

    def mk_roof(cfg):
        def fn():
            quack_gemm(a_q_full, b_q, d, None, None, *cfg, SFA=sfa_static, SFB=sfb, **MX)

        return fn

    for c in cfgs:
        timer.add(f"mx_roof_{c[0]}x{c[1]}x{c[2]}", mk_roof(c))
    roof = timer.run()
    best_i = torch.tensor(
        [
            min(
                range(len(cfgs)),
                key=lambda i: roof[f"mx_roof_{cfgs[i][0]}x{cfgs[i][1]}x{cfgs[i][2]}"][0],
            )
        ],
        device=device,
        dtype=torch.int64,
    )
    dist.broadcast(best_i, src=0, group=_PG)
    CFG = cfgs[int(best_i.item())]
    _print0(rank, f"  mx roof cfg sweep -> {CFG}")

    runner_mx = BlockScaledAllGatherRunner(m, K, "mxfp8_e4m3", group=_PG, device=device)
    runner_bf16 = AllGatherRunner(m, K, torch.bfloat16, group=_PG, device=device)

    methods = {}

    def bf16_roof():
        quack_gemm(a_hp_full, b_hp, d, None, None, 256, 256, 2, 1)

    def mx_roof():
        quack_gemm(a_q_full, b_q, d, None, None, *CFG, SFA=sfa_static, SFB=sfb, **MX)

    def quack_bf16_ag():
        with runner_bf16.gather(a_hp) as (a_full, ag_args):
            quack_gemm(a_full, b_hp, d, None, None, 256, 256, 2, 1, ag_args=ag_args)

    def quack_mx_staged():
        packed = pack_scale_2d_to_blocked_contig(a_sc.view(1, m, K // SF_VEC))
        op = BlockScaledOperand(a_q, packed, "mxfp8_e4m3")
        with runner_mx.gather(op) as (a_op, ag_args):
            quack_gemm(
                a_op.qdata, b_q, d, None, None, *CFG,
                SFA=a_op.scale, SFB=sfb, **MX, ag_args=ag_args,
            )

    sfa_nccl = torch.empty_like(sfa_static)

    def quack_mx_nccl():
        packed = pack_scale_2d_to_blocked_contig(a_sc.view(1, m, K // SF_VEC))
        dist.all_gather_into_tensor(
            sfa_nccl.view(torch.uint8).view(-1),
            packed.view(torch.uint8).view(-1),
            group=_PG,
        )
        with runner_mx.gather(a_q_u8) as (a_full, ag_args):
            quack_gemm(
                a_full.view(torch.float8_e4m3fn), b_q, d, None, None, *CFG,
                SFA=sfa_nccl, SFB=sfb, **MX, ag_args=ag_args,
            )

    ag_bf16_dst = torch.empty(M, K, dtype=torch.bfloat16, device=device)

    def torch_bf16():
        _ag_into(ag_bf16_dst, a_hp)
        torch.matmul(ag_bf16_dst, b_hp.T, out=d)

    ag_q_dst = torch.empty(M, K, dtype=torch.uint8, device=device)
    sfa_torch = torch.empty_like(sfa_static)

    def torch_mx():
        _ag_into(ag_q_dst, a_q_u8)
        packed = pack_scale_2d_to_blocked_contig(a_sc.view(1, m, K // SF_VEC))
        dist.all_gather_into_tensor(
            sfa_torch.view(torch.uint8).view(-1),
            packed.view(torch.uint8).view(-1),
            group=_PG,
        )
        torch._scaled_mm(
            ag_q_dst.view(torch.float8_e4m3fn),
            b_q.t(),
            scale_a=sfa_torch.view(torch.float8_e8m0fnu).view(-1, 4 * SF_VEC),
            scale_b=sfb.view(torch.float8_e8m0fnu).view(-1, 4 * SF_VEC),
            out_dtype=torch.bfloat16,
            out=d,
        )

    group_name = _PG.group_name

    # async-TP's native fp8 recipe: ROWWISE scales (its scaled op slices the
    # A scale per gathered chunk, which only composes with per-row scales —
    # mx blocked scales are rejected, see asynctp_mx's probe).
    a_rs = (a_hp.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    a_fp8r = (a_hp.float() / a_rs).clamp(-448, 448).to(torch.float8_e4m3fn)
    b_cs = (b_hp.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    b_fp8r = (b_hp.float() / b_cs).clamp(-448, 448).to(torch.float8_e4m3fn)
    ref_fp8row = (a_fp8r.float() * a_rs) @ (b_fp8r.float() * b_cs).T
    ref_fp8row_full = torch.empty(M, N, dtype=torch.float32, device=device)
    _ag_into(ref_fp8row_full, ref_fp8row.contiguous())
    del ref_fp8row
    fp8row_tol = ref_fp8row_full.abs().max().item() * 5e-2 + 1e-2

    def asynctp_fp8row():
        _, (out,) = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            a_fp8r, [b_fp8r.t()], a_rs, [b_cs.view(1, N)],
            gather_dim=0, group_name=group_name,
            biases=[None], result_scales=[None],
            out_dtypes=[torch.bfloat16], use_fast_accum=[False],
        )
        d.copy_(out)

    def asynctp_bf16():
        _, (out,) = torch.ops.symm_mem.fused_all_gather_matmul(
            a_hp, [b_hp.t()], gather_dim=0, group_name=group_name
        )
        d.copy_(out)

    def asynctp_mx():
        packed = pack_scale_2d_to_blocked_contig(a_sc.view(1, m, K // SF_VEC))
        dist.all_gather_into_tensor(
            sfa_torch.view(torch.uint8).view(-1),
            packed.view(torch.uint8).view(-1),
            group=_PG,
        )
        _, (out,) = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            a_q_u8.view(torch.float8_e4m3fn), [b_q.t()],
            sfa_torch.view(torch.float8_e8m0fnu).view(-1, 4 * SF_VEC),
            [sfb.view(torch.float8_e8m0fnu).view(-1, 4 * SF_VEC)],
            gather_dim=0, group_name=group_name,
            biases=[None], result_scales=[None],
            out_dtypes=[torch.bfloat16], use_fast_accum=[False],
        )
        d.copy_(out)

    methods["bf16_roof"] = (bf16_roof, ref_bf16, bf16_tol)
    methods["mx_roof"] = (mx_roof, ref, mx_tol)
    methods["quack_bf16_ag"] = (quack_bf16_ag, ref_bf16, bf16_tol)
    methods["quack_mx_staged"] = (quack_mx_staged, ref, mx_tol)
    methods["quack_mx_nccl"] = (quack_mx_nccl, ref, mx_tol)
    methods["torch_bf16"] = (torch_bf16, ref_bf16, bf16_tol)
    methods["torch_mx"] = (torch_mx, ref, mx_tol)
    methods["asynctp_bf16"] = (asynctp_bf16, ref_bf16, bf16_tol)
    methods["asynctp_mx"] = (asynctp_mx, ref, mx_tol)
    methods["asynctp_fp8row"] = (asynctp_fp8row, ref_fp8row_full, fp8row_tol)

    timer = MethodTimer(device, rank)
    for name, (fn, method_ref, tol) in methods.items():
        # Probe once: a method this torch build does not support is skipped,
        # on all ranks together (collective probes must stay rank-uniform).
        ok = torch.tensor([1.0], device=device)
        try:
            fn()
            torch.cuda.synchronize(device)
        except Exception as e:
            ok[0] = 0.0
            torch.cuda.synchronize(device)
            _print0(rank, f"  [{name}] SKIP ({type(e).__name__}: {str(e)[:160]})")
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=_PG)
        if ok.item() < 1:
            continue
        d.zero_()
        fn()
        torch.cuda.synchronize(device)
        dist.barrier(group=_PG)
        if not _check(rank, name, d, method_ref, tol):
            continue
        timer.add(name, fn)

    results = timer.run()
    _print0(
        rank,
        f"  {'method':<18} {'us/iter':>10} {'TFLOP/s':>9} {'sol_us':>8} {'%sol':>6}  mode",
    )
    for name, (us, mode) in results.items():
        sol = _sol_us(M, N, K, name)
        _print0(
            rank,
            f"RESULT, TP={ws}, M={M}, N={N}, K={K}, {name:<18} "
            f"{us:10.1f} {_tflops(M, N, K, us):9.0f} {sol:8.1f} {100 * sol / us:5.0f}%"
            f"  [{mode}]",
        )

    # returning drops the runners' symmetric buffers before the next shape
    torch.cuda.synchronize(device)
    dist.barrier(group=_PG)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, nargs="+", default=None)
    parser.add_argument("--N", type=int, nargs="+", default=None)
    parser.add_argument("--K", type=int, nargs="+", default=None)
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="TP group size; ranks >= tp idle at a barrier (for hosts with "
        "more GPUs than the TP under test).",
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(hours=3))
    world = dist.get_world_size()
    tp = args.tp or world
    assert 1 <= tp <= world
    global _PG
    _PG = (
        dist.group.WORLD
        if tp == world
        else dist.new_group(list(range(tp)), timeout=timedelta(hours=3))
    )
    if rank >= tp:
        dist.barrier()  # WORLD: park until the TP ranks finish
        dist.destroy_process_group()
        return
    ws = tp

    if args.M and args.N and args.K:
        shapes = [(M, N, K) for M in args.M for N in args.N for K in args.K]
    else:
        shapes = DEFAULT_SHAPES

    for M, N, K in shapes:
        if (M // ws) % 128 != 0:
            _print0(rank, f"skipping M={M}: shard not a multiple of 128 at TP={ws}")
            continue
        run_shape(rank, ws, device, M, N, K)

    if tp != world:
        dist.barrier()  # WORLD: release the parked ranks
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
