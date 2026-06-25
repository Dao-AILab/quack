"""CUDA-graph comparison: quack MXFP8 grouped GEMM vs torch._scaled_grouped_mm (cuBLAS mxfp8).

Captures each op in a CUDA graph and times replay. Only the sync-free quack paths
are graph-capturable:
  * uniform=True  -> dense batched-L (fixed-capacity MoE)
  * varlen=True   -> varlen-M, natural SFA (capacity-padded, 128-aligned MoE)
The general offs-routed / non-128 path does a host .cpu() sync and is NOT capturable.
"""
import itertools
import torch

from quack.mx_utils import to_mx_compiled
from quack.blockscaled_gemm_utils import pack_scale_2d_to_blocked_contig
from quack.mxfp8_grouped_gemm import MXFP8GroupedGemm

SF_VEC = 32
dev = torch.device("cuda")


def make(group_sizes, k, n, seed=0):
    torch.manual_seed(seed)
    e = len(group_sizes); total_m = sum(group_sizes); sf_k = k // SF_VEC; std = k**-0.5
    a_hp = (torch.randn(total_m, k, dtype=torch.bfloat16, device=dev) * std).contiguous()
    qa, sa = to_mx_compiled(a_hp, SF_VEC)
    b_hp = (torch.randn(e, n, k, dtype=torch.bfloat16, device=dev) * std).contiguous()
    qb_flat, sb = to_mx_compiled(b_hp.view(e * n, k), SF_VEC)
    sb = sb.view(e, n, sf_k)
    b_disp = qb_flat.view(e, n, k).transpose(1, 2)  # (E,K,N)
    a_ref = qa.float() * sa.float().repeat_interleave(SF_VEC, -1)
    b_ref = qb_flat.float().view(e, n, k) * sb.float().repeat_interleave(SF_VEC, -1)
    offs = torch.tensor(list(itertools.accumulate(group_sizes)), dtype=torch.int32, device=dev)
    return qa, b_disp, offs, sa, sb, a_ref, b_ref, total_m, e, sf_k


def ref_grouped(a_ref, b_ref, gs):
    outs, s = [], 0
    for i, g in enumerate(gs):
        outs.append(a_ref[s:s + g] @ b_ref[i].T); s += g
    return torch.cat(outs)


def torch_scales(sa, sb, total_m, sf_k, e):
    sb_blk = pack_scale_2d_to_blocked_contig(sb)                       # (E,rmN,rk,512)
    sb_t = sb_blk.reshape(e, -1)
    sa_blk = pack_scale_2d_to_blocked_contig(sa.view(1, total_m, sf_k))  # (1,rmM,rk,512)
    rmM, rk = sa_blk.shape[1], sa_blk.shape[2]
    sa_t = sa_blk.reshape(rmM * 128, rk * 4)
    return sa_t, sb_t


def capture_and_time(fn, n_warmup=3, n_replay=50):
    # fn must already have been called once (to JIT-compile) before this.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(n_replay):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / n_replay  # ms/replay


def flops(gs, k, n):
    return 2.0 * sum(g * n * k for g in gs)


# regimes that are graph-relevant (fixed / padded capacity MoE)
UNIFORM = [
    ("uniform g=2048 E=8", (2048,) * 8),
    ("uniform g=1024 E=8", (1024,) * 8),
    ("uniform g=512 E=16", (512,) * 16),
    ("uniform g=256 E=32", (256,) * 32),
    ("uniform g=128 E=128", (128,) * 128),
    ("uniform g=128 E=256", (128,) * 256),
]
VARLEN = [
    ("varlen balanced (640)*8", (640,) * 8),
    ("varlen skew (1792,128*3)", (1792, 128, 128, 128)),
    ("varlen skew8 (3584,128*7)", (3584, 128, 128, 128, 128, 128, 128, 128)),
]
KN = [(4096, 4096), (8192, 4096)]


def run(label, gs, k, n, mode):
    qa, b_disp, offs, sa, sb, a_ref, b_ref, total_m, e, sf_k = make(gs, k, n)
    gemm = MXFP8GroupedGemm(b_disp, sb)
    if mode == "uniform":
        qfn = lambda: gemm(qa, offs, sa, uniform=True)
    else:
        qfn = lambda: gemm(qa, offs, sa, varlen=True)
    # correctness (eager, once) + JIT compile
    out = qfn(); torch.cuda.synchronize()
    ok = (out.float() - ref_grouped(a_ref, b_ref, gs)).abs().max().item() < 5e-2
    q_ms = capture_and_time(qfn)

    # torch baseline (graphed)
    sa_t, sb_t = torch_scales(sa, sb, total_m, sf_k, e)
    tfn = lambda: torch._scaled_grouped_mm(
        qa, b_disp, sa_t, sb_t, offs=offs, out_dtype=torch.bfloat16
    )
    o = tfn(); torch.cuda.synchronize()
    t_err = (o.float() - ref_grouped(a_ref, b_ref, gs)).abs().max().item()
    assert t_err < 5e-2, f"torch mxfp8 baseline wrong: err={t_err:.2f}"
    t_ms = capture_and_time(tfn)

    fl = flops(gs, k, n)
    q_tf = fl / (q_ms * 1e-3) / 1e12
    t_tf = fl / (t_ms * 1e-3) / 1e12
    ratio = t_ms / q_ms  # >1 => quack faster
    print(f"{label:<28}{f'{k}x{n}':<11}{q_ms*1e3:>9.1f}{q_tf:>9.0f}{t_ms*1e3:>9.1f}{t_tf:>9.0f}"
          f"{ratio:>8.2f}{('Y' if ok else 'N!'):>4}")


def main():
    print("CUDA-graph replay: quack MXFP8 grouped GEMM vs torch._scaled_grouped_mm (cuBLAS mxfp8)")
    print(f"{'regime':<28}{'KN':<11}{'q_us':>9}{'q_TF':>9}{'torch_us':>9}{'tch_TF':>9}"
          f"{'tch/q':>8}{'ok':>4}")
    print("-" * 92)
    for k, n in KN:
        for label, gs in UNIFORM:
            run(label, gs, k, n, "uniform")
        for label, gs in VARLEN:
            run(label, gs, k, n, "varlen")
        print("-" * 92)
    print("q_us/torch_us = per-replay microseconds; tch/q > 1 => quack faster than cuBLAS")


if __name__ == "__main__":
    main()
