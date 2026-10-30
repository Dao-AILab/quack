"""Mixed preferred/fallback cluster launches for SM100 CLC-persistent GEMMs.

Fallback support is on by default whenever CLC dynamic scheduling is enabled
(``is_dynamic_persistent=True``): one kernel body carries both cluster shapes'
load TMA atoms and selects atoms, multicast masks and barrier arrive counts at
runtime from the cluster's actual shape, and the driver may launch any mix of
preferred- and fallback-shaped clusters. The driver only falls back under
resource pressure, so the mixed-launch tests validate "correct for ANY mix",
while the forced tests deterministically execute the fallback path by shrinking
the preferred launch shape to the fallback shape (every cluster then launches
fallback-shaped and the runtime predicate selects the fallback atoms/masks/counts
of the production kernel).

Problem sizes are chosen to overfill the GPU (grid > SM count, counted at the
FALLBACK shape for the forced tests) so the CLC pending pool is non-empty and
work stealing actually happens — under-filled grids retire on their initial
tile and never exercise the steal decode.

Mixed-launch (preferred-shape) tests are limited to cluster size <= 2: CLC
scheduling with cluster sizes >= 4 — e.g. (2,2), (4,1), (4,2), (2,4) — hangs
under GPU contention on unmodified main, independent of fallback support. The
forced tests launch at the fallback shape (size <= 2), so they additionally
cover a 2-CTA preferred (2,2) whose fallback is the intact (2,1) pair.

The retirement drain (``cancel_pending_tail``) runs in preferred-shaped clusters
only, so no forced test exercises it; the mixed-launch tests do, for whichever
clusters the driver launches preferred-shaped.
"""

import math

import pytest
import torch

from cutlass import BFloat16, Float32

import quack.gemm as gemm_module
from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm as quack_gemm
from quack.gemm_sm100 import GemmSm100

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (10, 11),
    reason="Mixed (preferred/fallback) cluster launch is SM100/SM110 only",
)

DTYPE = torch.bfloat16
ATOL, RTOL = 3e-2, 1e-3

# (tile_M, tile_N, cluster_M, cluster_N): upstream-healthy CLC cluster shapes
# with a differing fallback. (1,2) exercises the N-remainder of the preferred
# footprint, (2,1) with tile_M=64 (1-CTA MMA) the M-remainder.
CONFIGS = [
    (128, 256, 1, 2),  # 1-CTA MMA, fallback (1,1), rem_n path
    (64, 256, 2, 1),  # 1-CTA MMA, fallback (1,1), rem_m path
]
# Forced-path only (launches at the size-2 fallback, never a size-4/8 cluster):
# 2-CTA MMA whose fallback keeps the pair — per-shape A multicast atoms, the
# pair's SFB/B 2SM atoms, and the pair-intact sched pipeline. The size-8 shapes
# are in the SM100 autotune sweep (gemm_config._get_sm100_configs, CLC only).
FORCED_CONFIGS = CONFIGS + [
    (128, 256, 2, 2),  # 2-CTA MMA, fallback (2,1), rem_n path
    (256, 128, 4, 1),  # 2-CTA MMA, fallback (2,1), rem_m path (pair offset 2)
    (256, 256, 4, 2),  # 2-CTA MMA, fallback (2,1), rem_m + rem_n (size 8)
    (256, 256, 2, 4),  # 2-CTA MMA, fallback (2,1), rem_n path (size 8)
]


def _run_gemm(A, B, D, tile_M, tile_N, cluster_M, cluster_N, **kwargs):
    quack_gemm(
        A,
        B,
        D,
        C=kwargs.pop("C", None),
        tile_count_semaphore=None,
        tile_M=tile_M,
        tile_N=tile_N,
        cluster_M=cluster_M,
        cluster_N=cluster_N,
        persistent=True,
        is_dynamic_persistent=True,  # CLC scheduling: fallback support is on
        **kwargs,
    )


@pytest.fixture
def force_fallback(monkeypatch):
    """Route every compile in the test through the fallback-forcing hook,
    isolated from every cache layer: the force flag is invisible to the
    arg-derived jit-cache key, so (a) disable the disk cache, (b) clear the
    in-memory compile cache, and (c) swap in a throwaway plan cache
    (monkeypatch reverts (a) and (c)). The forced kernels land in the in-memory
    compile cache under normal keys; teardown scrubs them — pass or fail — so
    later tests in this process don't pick up a forced kernel."""
    monkeypatch.setattr("quack.cache.CACHE_ENABLED", False)
    monkeypatch.setattr(gemm_module, "_gemm_plan_cache", {})
    gemm_module._compile_gemm.cache_clear()

    orig_compile = gemm_module.compile_gemm_kernel

    def compile_forced(*args, **kwargs):
        chained = kwargs.get("post_init")

        def post_init(gemm_obj):
            if chained is not None:
                chained(gemm_obj)
            assert gemm_obj.fallback_cluster_shape_mnk is not None
            gemm_obj._force_fallback_branch = True

        kwargs["post_init"] = post_init
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(gemm_module, "compile_gemm_kernel", compile_forced)
    yield
    gemm_module._compile_gemm.cache_clear()


def test_fallback_cluster_defaults():
    """Derivation rules: (2,1) with 2-CTA MMA (pair stays intact), (1,1)
    otherwise; disabled when the preferred cluster already is the fallback
    shape or when CLC persistence is off."""

    def mk(mma_tiler, cluster, **kw):
        return GemmSm100(Float32, BFloat16, mma_tiler, cluster, **kw)

    assert mk((256, 128), (2, 1, 1)).fallback_cluster_shape_mnk is None  # 2cta, equal
    assert mk((256, 128), (4, 2, 1)).fallback_cluster_shape_mnk == (2, 1, 1)  # 2cta
    assert mk((128, 128), (2, 2, 1)).fallback_cluster_shape_mnk == (2, 1, 1)  # 2cta (tile_m 128)
    assert mk((128, 128), (1, 2, 1)).fallback_cluster_shape_mnk == (1, 1, 1)  # 1cta
    assert mk((128, 128), (1, 1, 1)).fallback_cluster_shape_mnk is None  # equal
    clc_off = mk((256, 128), (4, 2, 1), use_clc_persistence=False)
    assert clc_off.fallback_cluster_shape_mnk is None


@pytest.mark.parametrize("tile_M,tile_N,cluster_M,cluster_N", CONFIGS)
def test_gemm_mixed_cluster_launch(tile_M, tile_N, cluster_M, cluster_N):
    """Numerics under a real mixed launch with active CLC stealing: the driver
    picks the preferred/fallback mix at runtime, so the result must be correct
    for any mix (all-preferred on an idle GPU included)."""
    torch.manual_seed(0)
    l, m, n, k = 4, 2048, 1536, 512  # grid >> SM count: steals guaranteed
    A = torch.randn(l, m, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    B = torch.randn(l, n, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    D = torch.empty(l, m, n, dtype=DTYPE, device="cuda")
    _run_gemm(A, B, D, tile_M, tile_N, cluster_M, cluster_N)
    ref = torch.bmm(A.float(), B.float().mT).to(DTYPE)
    torch.testing.assert_close(D, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("has_C", [False, True])
@pytest.mark.parametrize("tile_M,tile_N,cluster_M,cluster_N", FORCED_CONFIGS)
def test_gemm_forced_fallback_branch(force_fallback, tile_M, tile_N, cluster_M, cluster_N, has_C):
    """Deterministically execute the FALLBACK path of the production kernel
    with CLC steals active. This is the regression test for the
    CLC scheduler's preferred-unit remainder decode: reverting it (adding the
    runtime block_in_cluster_idx instead of the CTA remainder modulo the
    preferred shape) makes fallback clusters at non-zero offsets inside a
    preferred footprint compute duplicate tiles and skip others — caught here
    as a numerical mismatch. has_C additionally routes the epi-load warp
    through the runtime-selected sched-pipeline arrive counts.
    """
    torch.manual_seed(1)
    l, m, n, k = 4, 2048, 1536, 512  # >= 384 CTAs: > SM count even as (2,1) pairs
    A = torch.randn(l, m, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    B = torch.randn(l, n, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    C = torch.randn(l, m, n, dtype=DTYPE, device="cuda") if has_C else None
    D = torch.empty(l, m, n, dtype=DTYPE, device="cuda")
    kwargs = dict(C=C, alpha=1.0, beta=0.5) if has_C else {}
    _run_gemm(A, B, D, tile_M, tile_N, cluster_M, cluster_N, **kwargs)
    ref = torch.bmm(A.float(), B.float().mT)
    if has_C:
        ref = ref + 0.5 * C.float()
    torch.testing.assert_close(D, ref.to(DTYPE), atol=ATOL, rtol=RTOL)


def test_gemm_forced_fallback_split_k(force_fallback):
    """Forced fallback path with serial split-K: the split index rides the
    work-id z decode, which must stay consistent with the remainder decode.
    8 x 6 tiles x 4 batches x 2 splits = 384 CTAs at the (1,1) fallback, well
    over the SM count, so the stolen-work decode runs too."""
    torch.manual_seed(3)
    l, m, n, k = 4, 1024, 1536, 4096  # K-heavy: split_k's home turf
    A = torch.randn(l, m, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    B = torch.randn(l, n, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    D = torch.empty(l, m, n, dtype=DTYPE, device="cuda")
    _run_gemm(A, B, D, tile_M=128, tile_N=256, cluster_M=1, cluster_N=2, split_k=2)
    ref = torch.bmm(A.float(), B.float().mT).to(DTYPE)
    torch.testing.assert_close(D, ref, atol=ATOL, rtol=RTOL)


def test_gemm_forced_fallback_varlen_m(force_fallback):
    """Forced fallback path on the varlen_m scheduler: exercises the
    remainder decode in VarlenMTileScheduler and the forward-window scan reset
    when a stolen work id regresses (fallback grids raster across rows)."""
    torch.manual_seed(2)
    num_groups, n, k = 16, 1024, 512
    lens = [384, 1024, 128, 896, 640, 512, 768, 256] * 2  # ragged, not tile-aligned
    cu_seqlens_m = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32).cuda()
    total_m = sum(lens)
    A = torch.randn(total_m, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    B = torch.randn(num_groups, n, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    D = torch.empty(total_m, n, dtype=DTYPE, device="cuda")
    _run_gemm(A, B, D, tile_M=128, tile_N=256, cluster_M=1, cluster_N=2, cu_seqlens_m=cu_seqlens_m)
    ref_parts = []
    for i in range(num_groups):
        s, e = cu_seqlens_m[i].item(), cu_seqlens_m[i + 1].item()
        ref_parts.append(A[s:e].float() @ B[i].float().T)
    ref = torch.cat(ref_parts).to(DTYPE)
    torch.testing.assert_close(D, ref, atol=ATOL, rtol=RTOL)
