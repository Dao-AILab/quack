"""Benchmark quack SM90 mxfp8 GEMM vs deep_gemm fp8 GEMM across modes.

Modes (selectable via --mode):
  dense    : standard GEMM (FFN-up shape). Quack vs deep_gemm.fp8_gemm_nt.
  batched  : grouped GEMM with G experts × M_per_expert tokens (same M per expert).
             Quack 3D batched vs deep_gemm.m_grouped_fp8_gemm_nt_masked.
  varlen   : grouped GEMM with G experts × variable M per expert.
             Quack varlen (cu_seqlens_m) vs deep_gemm.m_grouped_fp8_gemm_nt_contiguous.
  all      : run all three (default).

Models are MoE configurations from real systems; per-expert GEMM shape is
(M_per_expert, hidden, expert_dim). FFN gating is omitted (single matmul only),
so reported TFLOPS counts the up-proj only.

Run on H100:
    python benchmarks/benchmark_quack_vs_deepgemm_fp8.py
    python benchmarks/benchmark_quack_vs_deepgemm_fp8.py --mode varlen --batch 4096
"""

import argparse
import math
from typing import List, Tuple

import torch
from triton.testing import do_bench

import deep_gemm
from deep_gemm.utils import get_m_alignment_for_contiguous_layout
from deep_gemm.utils.math import per_token_cast_to_fp8, per_block_cast_to_fp8

from quack.gemm_blockscaled_interface import (
    mxfp8_gemm_act,
    quantize_act_sm90,
    quantize_weight_sm90,
)


# MoE model shapes (name, hidden, expert_dim, num_experts, active_per_token)
# Active counts are typical top-k values for the model family.
MOE_MODELS: List[Tuple[str, int, int, int, int]] = [
    ("Qwen3 3a30b",     2048,  768, 128, 8),
    ("Qwen3 22a235b",   4096, 1536, 128, 8),
    ("Qwen3.5 3a35b",   2048,  512, 256, 8),
    ("Qwen3.5 17a397b", 4096, 1024, 512, 8),
    # DeepSeek-V4: hidden=7168, expert_dim=3072, num_experts=384. Disabled by
    # default because the (384, 3072, 7168) weight stack needs ~35 GB peak
    # during quantization (bf16 + contiguous copy) and doesn't fit on a single
    # 80 GB H100 alongside other allocations. Re-enable once we have a
    # per-expert quantization path that avoids the bf16 working copy.
    # ("DeepSeek-V4",     7168, 3072, 384, 8),
]

# Standalone dense shapes (FFN up/down at common sizes).
DENSE_SHAPES: List[Tuple[int, int, int]] = [
    (8192,  4096, 14336),
    (8192, 14336,  4096),
]


def _tflops(m_total: int, n: int, k: int, ms: float) -> float:
    return 2.0 * m_total * n * k / (ms * 1e-3) / 1e12


def _pt_cast_3d(x_3d: torch.Tensor):
    """deep_gemm's per_token_cast_to_fp8 asserts 2D; loop over batch dim."""
    qs, sfs = zip(*[
        per_token_cast_to_fp8(x_3d[i].contiguous(), use_ue8m0=True, gran_k=128)
        for i in range(x_3d.shape[0])
    ])
    return torch.stack(qs), torch.stack(sfs)


def _pb_cast_3d(x_3d: torch.Tensor):
    """deep_gemm's per_block_cast_to_fp8 asserts 2D; loop over batch dim."""
    qs, sfs = zip(*[
        per_block_cast_to_fp8(x_3d[i].contiguous(), use_ue8m0=True, gran_k=128)
        for i in range(x_3d.shape[0])
    ])
    return torch.stack(qs), torch.stack(sfs)


# ---------------------------------------------------------------------------
# Dense GEMM (single matmul)
# ---------------------------------------------------------------------------


def bench_dense_quack(M: int, K: int, N: int, repeats: int) -> Tuple[float, float]:
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / math.sqrt(K)
    A_q, A_sc = quantize_act_sm90(A)
    W_q, W_sc = quantize_weight_sm90(W)
    B_q, B_sc = W_q.mT, W_sc.mT
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def fn():
        mxfp8_gemm_act(
            A_q, B_q, A_sc, B_sc,
            activation=None,
            preact_out=None,
            postact_out=out,
            store_preact=False,
            tuned=False,
        )

    fn()
    ms = do_bench(fn, warmup=5, rep=repeats)
    return ms, _tflops(M, N, K, ms)


def bench_dense_dgemm(M: int, K: int, N: int, repeats: int) -> Tuple[float, float]:
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / math.sqrt(K)
    A_fp8, A_sf = per_token_cast_to_fp8(A, use_ue8m0=True, gran_k=128)
    B_fp8, B_sf = per_block_cast_to_fp8(W, use_ue8m0=True, gran_k=128)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def fn():
        deep_gemm.fp8_gemm_nt((A_fp8, A_sf), (B_fp8, B_sf), out)

    fn()
    ms = do_bench(fn, warmup=5, rep=repeats)
    return ms, _tflops(M, N, K, ms)


# ---------------------------------------------------------------------------
# Batched-grouped GEMM (G experts, same M per expert)
# ---------------------------------------------------------------------------


def bench_batched_quack(G: int, M_per_expert: int, K: int, N: int, repeats: int) -> Tuple[float, float]:
    # 3D (L=G, M, K) and (L=G, N, K).
    A = torch.randn(G, M_per_expert, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) / math.sqrt(K)
    A_q, A_sc = quantize_act_sm90(A)
    W_q, W_sc = quantize_weight_sm90(W)
    del A, W  # free bf16 — quantization is done
    B_q, B_sc = W_q.mT, W_sc.mT
    out = torch.empty(G, M_per_expert, N, dtype=torch.bfloat16, device="cuda")

    def fn():
        mxfp8_gemm_act(
            A_q, B_q, A_sc, B_sc,
            activation=None,
            preact_out=None,
            postact_out=out,
            store_preact=False,
            tuned=False,
        )

    fn()
    ms = do_bench(fn, warmup=5, rep=repeats)
    M_total = G * M_per_expert
    return ms, _tflops(M_total, N, K, ms)


def bench_batched_dgemm(G: int, M_per_expert: int, K: int, N: int, repeats: int) -> Tuple[float, float]:
    # deep_gemm masked: (G, M_max, K) tokens with per-group valid count.
    A = torch.randn(G, M_per_expert, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) / math.sqrt(K)
    A_fp8, A_sf = _pt_cast_3d(A)
    B_fp8, B_sf = _pb_cast_3d(W)
    del A, W  # free bf16; only fp8 needed from here
    masked_m = torch.full((G,), M_per_expert, dtype=torch.int32, device="cuda")
    out = torch.empty(G, M_per_expert, N, dtype=torch.bfloat16, device="cuda")

    def fn():
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            (A_fp8, A_sf), (B_fp8, B_sf), out, masked_m, M_per_expert,
        )

    fn()
    ms = do_bench(fn, warmup=5, rep=repeats)
    M_total = G * M_per_expert
    return ms, _tflops(M_total, N, K, ms)


# ---------------------------------------------------------------------------
# Varlen-grouped GEMM (G experts, variable M per expert)
# ---------------------------------------------------------------------------


def _make_seqlens(G: int, M_total_target: int, seed: int = 0) -> List[int]:
    """Generate G per-expert M values summing to ~M_total_target.

    Uses a mildly imbalanced distribution: ~30% variance around the mean,
    aligned to multiples of 8 (quack constraint per-segment) and 128
    (deep_gemm padding alignment) — we pick 128 as the common multiple.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mean = M_total_target / G
    raw = torch.empty(G).uniform_(0.7, 1.3, generator=gen) * mean
    align = 128
    seqlens = [max(align, (int(x) // align) * align) for x in raw.tolist()]
    return seqlens


def bench_varlen_quack(seqlens: List[int], K: int, N: int, repeats: int) -> Tuple[float, float]:
    G = len(seqlens)
    M_total = sum(seqlens)
    A = torch.randn(M_total, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) / math.sqrt(K)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
                      dtype=torch.int32, device="cuda")
    A_q, A_sc = quantize_act_sm90(A)
    W_q, W_sc = quantize_weight_sm90(W)
    del A, W
    B_q, B_sc = W_q.mT, W_sc.mT
    out = torch.empty(M_total, N, dtype=torch.bfloat16, device="cuda")

    def fn():
        mxfp8_gemm_act(
            A_q, B_q, A_sc, B_sc,
            activation=None,
            preact_out=None,
            postact_out=out,
            store_preact=False,
            cu_seqlens_m=cu,
            tuned=False,
        )

    fn()
    ms = do_bench(fn, warmup=5, rep=repeats)
    return ms, _tflops(M_total, N, K, ms)


def bench_varlen_dgemm(seqlens: List[int], K: int, N: int, repeats: int) -> Tuple[float, float]:
    """deep_gemm contiguous: flat 128-row-padded A, per-row grouped_layout with -1 padding."""
    G = len(seqlens)
    align = get_m_alignment_for_contiguous_layout()
    aligned = [((m + align - 1) // align) * align for m in seqlens]
    M_total_padded = sum(aligned)
    # Construct padded A in expert-permuted order; the unused rows hold zeros.
    A_padded = torch.zeros(M_total_padded, K, device="cuda", dtype=torch.bfloat16)
    grouped_layout = torch.empty(M_total_padded, device="cuda", dtype=torch.int32)
    row = 0
    for g, (a_m, p_m) in enumerate(zip(seqlens, aligned)):
        A_padded[row : row + a_m] = torch.randn(a_m, K, device="cuda", dtype=torch.bfloat16)
        grouped_layout[row : row + a_m] = g
        grouped_layout[row + a_m : row + p_m] = -1
        row += p_m

    W = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) / math.sqrt(K)
    A_fp8, A_sf = per_token_cast_to_fp8(A_padded, use_ue8m0=True, gran_k=128)
    B_fp8, B_sf = _pb_cast_3d(W)
    del A_padded, W
    out = torch.empty(M_total_padded, N, dtype=torch.bfloat16, device="cuda")

    def fn():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (A_fp8, A_sf), (B_fp8, B_sf), out, grouped_layout,
        )

    fn()
    ms = do_bench(fn, warmup=5, rep=repeats)
    # Report TFLOPS on the *unpadded* work — the padded rows do compute but aren't useful.
    M_total = sum(seqlens)
    return ms, _tflops(M_total, N, K, ms)


# ---------------------------------------------------------------------------
# Main / printing
# ---------------------------------------------------------------------------


def _print_header(extra: str) -> None:
    print(f"  {extra:<40}  {'quack ms':>10}  {'quack TF':>9}   "
          f"{'dgemm ms':>10}  {'dgemm TF':>9}   {'speedup':>8}")
    print("-" * 100)


def _print_row(label: str, q_ms, q_tf, d_ms, d_tf) -> None:
    speedup = d_ms / q_ms if q_ms > 0 else 0.0
    print(f"  {label:<40}  {q_ms:>10.4f}  {q_tf:>9.1f}   "
          f"{d_ms:>10.4f}  {d_tf:>9.1f}   {speedup:>7.2f}x")


def run_dense(repeats: int) -> None:
    print("\n=== Dense GEMM (no grouping) ===")
    _print_header(f"{'M':>6} {'K':>6} {'N':>6}")
    for M, K, N in DENSE_SHAPES:
        q_ms, q_tf = bench_dense_quack(M, K, N, repeats)
        d_ms, d_tf = bench_dense_dgemm(M, K, N, repeats)
        _print_row(f"{M:>6} {K:>6} {N:>6}", q_ms, q_tf, d_ms, d_tf)


def _safe_run(label: str, fn):
    """Run a bench fn; return (ms, tflops) or (None, None) on OOM/error, after logging."""
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        print(f"  {label}  SKIP: OOM")
        torch.cuda.empty_cache()
        return None, None
    except Exception as e:
        print(f"  {label}  SKIP: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()
        return None, None


def run_batched(repeats: int, m_per_expert: int) -> None:
    print(f"\n=== Batched-grouped GEMM (G experts × M_per_expert={m_per_expert}) ===")
    _print_header(f"{'model':<25}{'G':>5} {'M_per_expert':>13}")
    for name, hidden, expert_dim, G, _active in MOE_MODELS:
        M_per_expert = m_per_expert
        K, N = hidden, expert_dim
        label = f"{name:<25}{G:>5} {M_per_expert:>13}"
        q_ms, q_tf = _safe_run(label + "  (quack)", lambda: bench_batched_quack(G, M_per_expert, K, N, repeats))
        if q_ms is None:
            torch.cuda.empty_cache()
            continue
        d_ms, d_tf = _safe_run(label + "  (dgemm)", lambda: bench_batched_dgemm(G, M_per_expert, K, N, repeats))
        if d_ms is None:
            torch.cuda.empty_cache()
            continue
        _print_row(label, q_ms, q_tf, d_ms, d_tf)
        torch.cuda.empty_cache()


def run_varlen(repeats: int, m_per_expert: int) -> None:
    print(f"\n=== Varlen-grouped GEMM (G experts × variable M, mean={m_per_expert}) ===")
    _print_header(f"{'model':<25}{'G':>5} {'M_tot':>8}")
    for name, hidden, expert_dim, G, _active in MOE_MODELS:
        M_total_target = G * m_per_expert
        seqlens = _make_seqlens(G, M_total_target)
        K, N = hidden, expert_dim
        label = f"{name:<25}{G:>5} {sum(seqlens):>8}"
        q_ms, q_tf = _safe_run(label + "  (quack)", lambda: bench_varlen_quack(seqlens, K, N, repeats))
        if q_ms is None:
            torch.cuda.empty_cache()
            continue
        d_ms, d_tf = _safe_run(label + "  (dgemm)", lambda: bench_varlen_dgemm(seqlens, K, N, repeats))
        if d_ms is None:
            torch.cuda.empty_cache()
            continue
        _print_row(label, q_ms, q_tf, d_ms, d_tf)
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["dense", "batched", "varlen", "all"],
    )
    parser.add_argument(
        "--m-per-expert", type=int, default=1024,
        help="Per-expert M (tokens routed to each expert) for grouped GEMM benches",
    )
    args = parser.parse_args()

    cap = torch.cuda.get_device_properties(0).major
    if cap != 9:
        raise SystemExit(f"requires SM90 (H100); current device is sm_{cap}0")

    torch.manual_seed(0)
    if args.mode in ("dense", "all"):
        run_dense(args.repeats)
    if args.mode in ("batched", "all"):
        run_batched(args.repeats, args.m_per_expert)
    if args.mode in ("varlen", "all"):
        run_varlen(args.repeats, args.m_per_expert)


if __name__ == "__main__":
    main()
