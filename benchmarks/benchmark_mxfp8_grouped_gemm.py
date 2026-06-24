"""Regime sweep benchmark for the QuACK MXFP8 grouped GEMM (SM100 / B200).

For each MoE-style regime (uniform compute-bound, memory-bound small groups,
128-aligned skew, empty experts, non-128 ragged, many-expert) this measures the
achieved TFLOPS and per-call latency of the prepared ``MXFP8GroupedGemm``
(B-scale pre-packed once, A-scale packed per call). Each row is validated for
numerical correctness against a per-group dequantized matmul reference, and
compared to two PyTorch baselines: ``torch._scaled_grouped_mm`` (mxfp8) and
``torch._grouped_mm`` (bf16).

The printed table shows, per (regime, K, N) and routing path, the quack latency
and TFLOPS, the baseline TFLOPS, the quack/baseline ratios, and a correctness flag.
"""
import itertools

import torch

from quack.mx_utils import to_mx_compiled
from quack.blockscaled_gemm_utils import pack_scale_2d_to_blocked_contig
from quack.mxfp8_grouped_gemm import MXFP8GroupedGemm

SF_VEC = 32
dev = torch.device("cuda")
PEAK_BF16 = None  # filled empirically from the biggest bf16 run


def make(group_sizes, k, n, seed=0):
    torch.manual_seed(seed)
    e = len(group_sizes)
    total_m = sum(group_sizes)
    sf_k = k // SF_VEC
    std = k**-0.5
    a_hp = (torch.randn(max(total_m, 1), k, dtype=torch.bfloat16, device=dev) * std).contiguous()
    qa, sa = to_mx_compiled(a_hp, SF_VEC)
    b_hp = (torch.randn(e, n, k, dtype=torch.bfloat16, device=dev) * std).contiguous()
    qb_flat, sb = to_mx_compiled(b_hp.view(e * n, k), SF_VEC)
    qb = qb_flat.view(e, n, k)
    sb = sb.view(e, n, sf_k)
    b_disp = qb.transpose(1, 2)  # (E,K,N)
    a_ref = qa.float() * sa.float().repeat_interleave(SF_VEC, dim=-1)
    b_ref = qb_flat.float().view(e, n, k) * sb.float().repeat_interleave(SF_VEC, dim=-1)
    offs = torch.tensor(list(itertools.accumulate(group_sizes)), dtype=torch.int32, device=dev)
    return dict(qa=qa, b_disp=b_disp, offs=offs, sa=sa, sb=sb, a_ref=a_ref, b_ref=b_ref,
               a_hp=a_hp, b_hp=b_hp, total_m=total_m, e=e, k=k, n=n, sf_k=sf_k, gs=group_sizes)


def ref_grouped(a_ref, b_ref, group_sizes):
    outs, start = [], 0
    for gi, gs in enumerate(group_sizes):
        outs.append(a_ref[start : start + gs] @ b_ref[gi].T)
        start += gs
    return torch.cat(outs) if outs else torch.empty(0)


def timed(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def flops(group_sizes, k, n):
    return 2.0 * sum(g * n * k for g in group_sizes)


def bench_quack(d, route):
    """route in {uniform, varlen, offs}. Returns (ms, tflops, correct, err)."""
    gemm = MXFP8GroupedGemm(d["b_disp"], d["sb"])
    qa, offs, sa = d["qa"], d["offs"], d["sa"]
    if route == "uniform":
        call = lambda: gemm(qa, offs, sa, uniform=True)
    elif route == "varlen":
        call = lambda: gemm(qa, offs, sa, varlen=True)
    else:
        call = lambda: gemm(qa, offs, sa)  # general offs-routed (host sync for non-128)
    out = call()  # compile
    torch.cuda.synchronize()
    ref = ref_grouped(d["a_ref"], d["b_ref"], d["gs"])
    err = (out.float() - ref).abs().max().item() if ref.numel() else 0.0
    ms = timed(call)
    return ms, flops(d["gs"], d["k"], d["n"]) / (ms * 1e-3) / 1e12, err < 5e-2, err


def bench_torch_bf16(d):
    a_hp, b_hp, offs = d["a_hp"], d["b_hp"], d["offs"]
    bt = b_hp.transpose(1, 2).contiguous()  # (E,K,N)
    try:
        call = lambda: torch._grouped_mm(a_hp, bt, offs=offs, out_dtype=torch.bfloat16)
        call()
        torch.cuda.synchronize()
        ms = timed(call)
        return ms, flops(d["gs"], d["k"], d["n"]) / (ms * 1e-3) / 1e12
    except Exception as ex:
        return None, repr(ex)[:60]


def bench_torch_mxfp8(d):
    """Best-effort: torch._scaled_grouped_mm with cutlass blocked mxfp8 scales."""
    qa, b_disp, offs = d["qa"], d["b_disp"], d["offs"]
    e, k, n, sf_k = d["e"], d["k"], d["n"], d["sf_k"]
    try:
        # scale_b: per-group blocked pack of (E,N,sf_k) -> (E, rmN*rk*512) flattened
        sb_blk = pack_scale_2d_to_blocked_contig(d["sb"])  # (E, rmN, rk, 512)
        sb_t = sb_blk.reshape(e, -1)
        # scale_a: blocked pack -> 2D (M_pad, Kb_pad) (swizzled buffer reshaped; torch's layout)
        sa_blk = pack_scale_2d_to_blocked_contig(d["sa"].view(1, d["total_m"], sf_k))  # (1,rmM,rk,512)
        rmM, rk = sa_blk.shape[1], sa_blk.shape[2]
        sa_t = sa_blk.reshape(rmM * 128, rk * 4)
        call = lambda: torch._scaled_grouped_mm(qa, b_disp, sa_t, sb_t, offs=offs,
                                                out_dtype=torch.bfloat16)
        out = call()
        torch.cuda.synchronize()
        ref = ref_grouped(d["a_ref"], d["b_ref"], d["gs"])
        err = (out.float() - ref).abs().max().item() if ref.numel() else 1e9
        if err > 5e-2:
            return None, None, f"layout-wrong err={err:.2f}"
        ms = timed(call)
        return ms, flops(d["gs"], d["k"], d["n"]) / (ms * 1e-3) / 1e12, "ok"
    except Exception as ex:
        return None, None, repr(ex)[:70]


# ---------------- regime matrix ----------------
KN = [(2048, 2048), (4096, 4096), (8192, 4096)]  # (K, N): small / medium / large

CASES = []  # (regime, route, group_sizes_fn(E))
def uni(g, e): return (g,) * e

# 1) compute-bound uniform large g
for g in (512, 1024, 2048):
    for e in (8, 16):
        CASES.append((f"compute-uniform g={g} E={e}", "uniform", uni(g, e)))
# 2) memory-bound small g uniform
for g in (128, 256):
    for e in (8, 32, 128):
        CASES.append((f"membound-uniform g={g} E={e}", "uniform", uni(g, e)))
# 3) skewed 128-aligned -> varlen route
CASES.append(("skew-128 (1792,128,128,128) E=4", "varlen", (1792, 128, 128, 128)))
CASES.append(("skew-128 (3584,128,128,128,128,128,128,128) E=8", "varlen",
              (3584, 128, 128, 128, 128, 128, 128, 128)))
CASES.append(("balanced-128 (640,)*8 E=8 [skew control]", "varlen", (640,) * 8))
# 4) empty experts (128-aligned -> varlen)
CASES.append(("empty-mid (256,0,256,256) E=4", "varlen", (256, 0, 256, 256)))
CASES.append(("empty-many (512,0,0,0,512,0,0,0) E=8", "varlen", (512, 0, 0, 0, 512, 0, 0, 0)))
# 5) non-128 ragged (general offs route)
CASES.append(("nonaligned (100,300,200,424) E=4", "offs", (100, 300, 200, 424)))
CASES.append(("nonaligned-skew (500,12,8,4) E=4", "offs", (500, 12, 8, 4)))
# 6) many-expert MoE (tokens/expert ~ 64-256 avg)
CASES.append(("moe E=128 g=128 (uniform)", "uniform", uni(128, 128)))
CASES.append(("moe E=256 g=128 (uniform)", "uniform", uni(128, 256)))


def main():
    global PEAK_BF16
    print(f"{'regime':<46}{'KN':<13}{'route':<8}{'q_ms':>8}{'q_TFLOPS':>10}"
          f"{'bf16_TF':>9}{'fp8_TF':>9}{'q/bf16':>7}{'q/fp8':>7}{'ok':>4}")
    print("-" * 130)
    rows = []
    for (regime, route, gs) in CASES:
        for (k, n) in KN:
            try:
                d = make(gs, k, n)
            except Exception as ex:
                print(f"{regime:<46}{f'{k}x{n}':<13} make FAIL {repr(ex)[:40]}")
                continue
            try:
                q_ms, q_tf, ok, err = bench_quack(d, route)
            except Exception as ex:
                print(f"{regime:<46}{f'{k}x{n}':<13}{route:<8} quack FAIL {repr(ex)[:50]}")
                continue
            bf_ms, bf_tf = bench_torch_bf16(d)
            fp_ms, fp_tf, fp_msg = bench_torch_mxfp8(d)
            if isinstance(bf_tf, float):
                PEAK_BF16 = max(PEAK_BF16 or 0, bf_tf)
            spdup = (bf_ms / q_ms) if (isinstance(bf_ms, float) and q_ms) else float("nan")
            qfp = (fp_ms / q_ms) if (isinstance(fp_ms, float) and q_ms) else float("nan")
            bf_s = f"{bf_tf:.0f}" if isinstance(bf_tf, float) else "-"
            fp_s = f"{fp_tf:.0f}" if isinstance(fp_tf, float) else "-"
            qfp_s = f"{qfp:.2f}" if qfp == qfp else "-"
            print(f"{regime:<46}{f'{k}x{n}':<13}{route:<8}{q_ms:>8.3f}{q_tf:>10.0f}"
                  f"{bf_s:>9}{fp_s:>9}{spdup:>7.2f}{qfp_s:>7}{('Y' if ok else 'N!'):>4}")
            rows.append((regime, k, n, route, q_ms, q_tf, bf_tf, fp_tf, ok, err, fp_msg))
    print("-" * 122)
    print(f"empirical bf16 peak (max observed) = {PEAK_BF16:.0f} TFLOPS  "
          f"-> implied fp8 peak ~ {2*PEAK_BF16:.0f} TFLOPS")
    # mxfp8 baseline status
    msgs = set(r[10] for r in rows if r[7] is None)
    print("torch mxfp8 baseline status samples:", list(msgs)[:4])


if __name__ == "__main__":
    main()
