# Copyright (c) 2026, Tri Dao.
"""CPU-only checks of the Blackwell RMSNorm bwd analytical heuristic.

Wide rows: the per-thread fragment must stay <= 128 B. At 256 B/thread
(bf16 N=128k at cluster_n=8, N=256k at cluster_n=16) the kernel spills and
drops to ~1 TB/s on B200; capping the fragment recovers 3.4-3.6x."""

import pytest

from quack.rmsnorm_config import RmsNormBwdConfig, _max_cluster_for


@pytest.fixture(autouse=True)
def _blackwell_arch(monkeypatch):
    # Keep the heuristic GPU-blind (smem budget lookup goes through QUACK_ARCH).
    monkeypatch.setenv("QUACK_ARCH", "100")


def _bwd(N, width, **kw):
    return RmsNormBwdConfig.from_analytical_heuristic(
        N, width, width, arch_major=10, T_hint=kw.pop("T_hint", 16384), **kw
    )


def _frag_bytes(cfg, N, width):
    return (N // cfg.cluster_n) // cfg.threads_per_row * (width // 8)


@pytest.mark.parametrize("width", [16, 32])
@pytest.mark.parametrize("N", [4096, 8192, 16384, 32768, 65536, 98304, 131072, 262144])
def test_bwd_fragment_capped_at_128_bytes(N, width):
    if width == 32 and N > 128 * 1024:
        pytest.skip("fp32 rows above 128k are rejected by RMSNormBackward")
    cfg = _bwd(N, width)
    assert cfg.cluster_n <= _max_cluster_for(10)
    assert cfg.num_threads % cfg.threads_per_row == 0
    assert _frag_bytes(cfg, N, width) <= 128, cfg
    # one row per CTA in the cluster regime: sX + sdO for 2 stages must fit
    if cfg.cluster_n > 1:
        tile_bytes = (N // cfg.cluster_n) * (width // 8)
        assert cfg.num_threads == cfg.threads_per_row
        assert 2 * cfg.smem_stages * tile_bytes <= 220 * 1024, cfg


def test_bwd_wide_bf16_rows_widen_cluster_then_threads():
    # 128k bf16: cluster_n 8 -> 16 keeps the fragment at 128 B with 128 threads.
    cfg = _bwd(131072, 16)
    assert (cfg.cluster_n, cfg.threads_per_row) == (16, 128)
    # 256k bf16: the cluster is already maximal, so the row gets 256 threads.
    cfg = _bwd(262144, 16)
    assert (cfg.cluster_n, cfg.num_threads, cfg.threads_per_row) == (16, 256, 256)


def test_bwd_cluster_tma_only_for_small_fragments():
    # 32k bf16: 64 B/thread, the TMA path stays (unchanged from before).
    assert _bwd(32768, 16).use_tma
    # 64k..256k bf16: 128 B/thread, cp.async is 1.2-2.2x faster than TMA on B200.
    for N in (65536, 131072, 262144):
        assert not _bwd(N, 16).use_tma, N


def test_bwd_narrow_rows_unchanged():
    # The single-CTA ladder below 48 KB rows is untouched.
    cfg = _bwd(8192, 16)
    assert (cfg.cluster_n, cfg.num_threads, cfg.threads_per_row, cfg.use_tma) == (
        1,
        128,
        128,
        False,
    )
    cfg = _bwd(16384, 16)
    assert (cfg.cluster_n, cfg.num_threads, cfg.threads_per_row, cfg.use_tma) == (
        1,
        256,
        256,
        False,
    )
