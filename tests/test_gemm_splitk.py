"""Split-K GEMM (SM100 only).

Each output tile is computed by `split_k` work units covering disjoint K ranges.
Two reduction modes, both run-to-run deterministic:
- "parallel" (default): every split stores fp32 partials to its own workspace slice
  with no inter-CTA synchronization; a separate reduce kernel sums the slices in
  fixed ascending order and applies the epilogue (cuBLAS splitKreduce-style).
- "serial": fused in-kernel turnstile reduction; non-final splits serialize their
  partials into a shared slot (k-ascending) and the final split runs the epilogue.
"""

import math
import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm as quack_gemm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (10, 11),
    reason="split-K GEMM is SM100 only",
)

ATOL = {torch.bfloat16: 3e-2, torch.float16: 1e-2}
RTOL = 1e-3
MODES = ["parallel", "serial"]


def _run_gemm(A, B, D, split_k, mode, cluster_mn=(1, 1), **kwargs):
    quack_gemm(
        A,
        B,
        D,
        C=kwargs.pop("C", None),
        tile_count_semaphore=None,
        tile_M=128,
        tile_N=128,
        cluster_M=cluster_mn[0],
        cluster_N=cluster_mn[1],
        persistent=True,
        split_k=split_k,
        split_k_mode=mode,
        **kwargs,
    )


def _make_inputs(l, m, n, k, dtype=torch.bfloat16):
    torch.manual_seed(0)
    A = torch.randn(l, m, k, dtype=dtype, device="cuda") / math.sqrt(k)
    B = torch.randn(l, n, k, dtype=dtype, device="cuda") / math.sqrt(k)
    D = torch.empty(l, m, n, dtype=dtype, device="cuda")
    return A, B, D


# ── Correctness vs fp32 reference ────────────────────────────────────────────
# Shapes cover: aligned single tile; ragged k-tile count per split (4160) + batch;
# edge output tiles (192, 320 not multiples of 128) + batch -- OOB lanes round-trip the
# workspace and must be predicated away by the epilogue (serial) / reduce kernel (parallel).
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("split_k", [2, 4])
@pytest.mark.parametrize("m,n,k,l", [(128, 128, 16384, 1), (128, 256, 4160, 3), (192, 320, 8192, 2)])
def test_gemm_splitk(m, n, k, l, split_k, mode):
    A, B, D = _make_inputs(l, m, n, k)
    _run_gemm(A, B, D, split_k, mode)
    ref = torch.bmm(A.float(), B.float().mT).to(D.dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[D.dtype], rtol=RTOL)


# split_k larger than the number of k tiles: some splits own zero k tiles -> contribute zeros.
@pytest.mark.parametrize("mode", MODES)
def test_gemm_splitk_more_splits_than_k_tiles(mode):
    A, B, D = _make_inputs(1, 128, 128, 128)
    _run_gemm(A, B, D, 8, mode)
    ref = torch.bmm(A.float(), B.float().mT).to(D.dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[D.dtype], rtol=RTOL)


# Parallel split count can scale past the reduce's pre-stage window (stage_slots): split_k=64
# exercises the ordered remainder loop in the reduce kernel.
def test_gemm_splitk_parallel_large_split():
    A, B, D = _make_inputs(1, 128, 128, 16384)
    _run_gemm(A, B, D, 64, "parallel")
    ref = torch.bmm(A.float(), B.float().mT).to(D.dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[D.dtype], rtol=RTOL)


# Epilogue (alpha, beta*C, bias) must apply to the fully reduced accumulator, not per-split.
@pytest.mark.parametrize("mode", MODES)
def test_gemm_splitk_epilogue(mode):
    l, m, n, k = 2, 128, 256, 8192
    A, B, D = _make_inputs(l, m, n, k)
    C = torch.randn(l, m, n, dtype=D.dtype, device="cuda")
    bias = torch.randn(l, n, dtype=D.dtype, device="cuda")
    alpha, beta = 0.5, 0.7
    _run_gemm(A, B, D, 4, mode, C=C, alpha=alpha, beta=beta, rowvec_bias=bias)
    ref = (
        alpha * torch.bmm(A.float(), B.float().mT) + beta * C.float() + bias.float().unsqueeze(1)
    ).to(D.dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[D.dtype], rtol=RTOL)


# Determinism: the point of split-K with an ordered reduction.
@pytest.mark.parametrize("mode", MODES)
def test_gemm_splitk_deterministic(mode):
    A, B, D1 = _make_inputs(1, 128, 256, 16384)
    D2 = torch.empty_like(D1)
    _run_gemm(A, B, D1, 8, mode)
    for _ in range(5):
        _run_gemm(A, B, D2, 8, mode)
        assert torch.equal(D1, D2), "split-K reduction must be run-to-run deterministic"


# ── Table-driven reduce: variable contributors per tile (Stream-K precursor) ──
# Drive the reduce kernel DIRECTLY with a NON-uniform contributor layout to prove the
# per-tile (first_slot, count) prefix-sum indirection and the (m_idx, n_idx, l) tile
# enumeration are correct independently of the GEMM. The uniform split-K path uses
# first_slot = tile_idx * split_k (order-independent) and so cannot exercise this.
@pytest.mark.parametrize("vec_width", [1, 2, 4])
@pytest.mark.parametrize("d_dtype", [torch.bfloat16, torch.float16])
def test_splitk_reduce_variable_contributors(d_dtype, vec_width):
    from quack.gemm_splitk_reduce import splitk_reduce

    torch.manual_seed(0)
    l = 2
    tile_m = tile_n = 128
    ntile_m, ntile_n = 2, 3  # multi-tile, non-square raster (exact multiples, no edges)
    M, N = ntile_m * tile_m, ntile_n * tile_n
    num_tiles = ntile_m * ntile_n * l

    # Distinct per-tile contributor counts (>=1) so the prefix-sum offset actually matters.
    counts = torch.tensor([1, 5, 2, 4, 3, 1, 6, 2, 1, 3, 5, 2][:num_tiles], dtype=torch.int32)
    first = torch.zeros(num_tiles, dtype=torch.int32)
    first[1:] = torch.cumsum(counts.to(torch.int64), 0)[:-1].to(torch.int32)
    total = int(counts.sum().item())
    tile_first_slot, tile_count = first.cuda(), counts.cuda()

    ws = torch.randn(total, tile_m, tile_n, dtype=torch.float32, device="cuda")
    # n-major (M, N, L) layout, matching what perm3d hands the kernel in gemm().
    D = torch.empty(l, M, N, dtype=d_dtype, device="cuda").permute(1, 2, 0)
    C = torch.randn(l, M, N, dtype=d_dtype, device="cuda").permute(1, 2, 0)
    alpha, beta = 0.5, 0.7

    # Reference: sum each tile's own slots, then the kernel's epilogue order.
    ref = torch.zeros(M, N, l, dtype=torch.float32, device="cuda")
    for t in range(num_tiles):
        li, rem = t % l, t // l
        ni, mi = rem % ntile_n, rem // ntile_n
        f, c = int(first[t]), int(counts[t])
        part = ws[f : f + c].sum(0) * alpha
        rs, re, cs, ce = mi * tile_m, (mi + 1) * tile_m, ni * tile_n, (ni + 1) * tile_n
        ref[rs:re, cs:ce, li] = part + beta * C[rs:re, cs:ce, li].float()

    splitk_reduce(
        ws.reshape(-1), D, C, alpha, beta, None, None,
        tile_first_slot, tile_count, tile_m, tile_n, vec_width,
    )
    torch.testing.assert_close(D.float(), ref, atol=ATOL[d_dtype], rtol=RTOL)


# ── Intra-CTA K-split reduce (kgroups > 1): fills the GPU for few-tile reduces by having
# `kgroups` threads cooperate per output element (each reduces a contiguous group of
# contributors, combined in smem). Output is a fixed blocked-ascending sum: must stay
# correct (within bf16 tolerance) AND run-to-run deterministic. Variable per-tile counts
# (incl. counts < kgroups, leaving empty groups that must contribute 0) exercise the
# spg = ceil(count/R) grouping and the empty-group guard.
@pytest.mark.parametrize("kgroups", [2, 4, 8])
def test_splitk_reduce_kgroups(kgroups):
    from quack.gemm_splitk_reduce import splitk_reduce

    torch.manual_seed(1)
    l = 1
    tile_m = tile_n = 128
    ntile_m, ntile_n = 1, 2
    M, N = ntile_m * tile_m, ntile_n * tile_n
    num_tiles = ntile_m * ntile_n * l

    # Include a count < kgroups (=> some k-groups own zero contributors -> add 0) and a
    # count not divisible by kgroups (=> last group is short).
    counts = torch.tensor([3, 17][:num_tiles], dtype=torch.int32)
    first = torch.zeros(num_tiles, dtype=torch.int32)
    first[1:] = torch.cumsum(counts.to(torch.int64), 0)[:-1].to(torch.int32)
    total = int(counts.sum().item())
    tile_first_slot, tile_count = first.cuda(), counts.cuda()

    ws = torch.randn(total, tile_m, tile_n, dtype=torch.float32, device="cuda")
    D = torch.empty(l, M, N, dtype=torch.bfloat16, device="cuda").permute(1, 2, 0)

    # Reference: blocked-ascending sum over R contiguous groups (matches the kernel's
    # smem combine order), in fp32, then cast to the output dtype.
    ref = torch.zeros(M, N, l, dtype=torch.float32, device="cuda")
    for t in range(num_tiles):
        li, rem = t % l, t // l
        ni, mi = rem % ntile_n, rem // ntile_n
        f, c = int(first[t]), int(counts[t])
        spg = (c + kgroups - 1) // kgroups
        acc = torch.zeros(tile_m, tile_n, dtype=torch.float32, device="cuda")
        for g in range(kgroups):
            s0, s1 = g * spg, min((g + 1) * spg, c)
            for s in range(s0, s1):
                acc += ws[f + s]
        rs, re, cs, ce = mi * tile_m, (mi + 1) * tile_m, ni * tile_n, (ni + 1) * tile_n
        ref[rs:re, cs:ce, li] = acc

    def run(out):
        splitk_reduce(
            ws.reshape(-1), out, None, 1.0, 1.0, None, None,
            tile_first_slot, tile_count, tile_m, tile_n, 1, kgroups,
        )

    run(D)
    torch.testing.assert_close(D.float(), ref, atol=ATOL[torch.bfloat16], rtol=RTOL)

    D2 = torch.empty_like(D)
    run(D2)
    assert torch.equal(D, D2), "intra-CTA K-split reduce must be run-to-run deterministic"


# ── split_k=0 -> auto: gemm picks the split; output must be correct regardless ─────
# One shape where auto splits (single tile, large K) and one where it doesn't (GPU already full).
@pytest.mark.parametrize("m,n,k,l", [(128, 128, 16384, 1), (4096, 4096, 4096, 1)])
def test_gemm_splitk_auto(m, n, k, l):
    A, B, D = _make_inputs(l, m, n, k)
    _run_gemm(A, B, D, 0, "parallel")  # 0 = auto
    ref = torch.bmm(A.float(), B.float().mT).to(D.dtype)
    torch.testing.assert_close(D, ref, atol=ATOL[D.dtype], rtol=RTOL)


def test_auto_split_k_values():
    # Wave-aware heuristic (see auto_split_k docstring). Single tile: ~round(1.2*sqrt(k_tiles))
    # (balance GEMM latency vs reduce cost). Multi-tile: floor(0.70*num_sms/tiles) (fill ~0.7 of
    # a wave -- NOT a full wave, which spilled a near-empty 2nd wave / wasted reduce work).
    from quack.gemm_splitk_reduce import auto_split_k

    sms = 148  # B200
    assert auto_split_k(128, 128, 16384, 1, 128, 128, sms) == 19  # 1 tile, 256 k-tiles: ~1.2*16
    # 16 tiles: floor(0.70*148/16)=6 (the old fill cap was 10 -> 1.08 waves -> 0.69x of cuBLAS)
    assert auto_split_k(512, 512, 16384, 1, 128, 128, sms) == 6
    assert auto_split_k(256, 256, 32768, 1, 128, 128, sms) == 25  # 4 tiles: floor(0.70*148/4)
    assert auto_split_k(4096, 4096, 4096, 1, 128, 128, sms) == 1  # 1024 tiles already fill GPU


# ── CUDA graph capture ───────────────────────────────────────────────────────
# These shapes are launch-bound (per-call host work > GPU work), so the way to hide the
# overhead is CUDA-graph capture (replay drops it to ~0). Guards that the parallel split-K
# path stays capturable AND replays correctly -- the GEMM must overwrite every workspace
# slot the reduce reads, since capture reuses one workspace.
def test_gemm_splitk_cuda_graph():
    A, B, D = _make_inputs(1, 128, 128, 16384)
    ref = torch.bmm(A.float(), B.float().mT).to(D.dtype)

    for _ in range(3):  # warm caches (compile, uniform tables, device props) before capture
        _run_gemm(A, B, D, 8, "parallel")
    torch.cuda.synchronize()

    s = torch.cuda.Stream()  # CUDA-graph protocol: warm up on a side stream, then capture
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _run_gemm(A, B, D, 8, "parallel")
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _run_gemm(A, B, D, 8, "parallel")

    D.zero_()
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(D, ref, atol=ATOL[D.dtype], rtol=RTOL)
