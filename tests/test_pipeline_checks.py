# Copyright (c) 2025-2026, Tri Dao.
"""Host-side tests for quack.pipeline_checks (no GPU required).

The rule functions encode mbarrier arrive-count protocol semantics; these tests pin the
arithmetic against hand-derived counts for the real kernel configurations, and check
that violations raise with the facts in the message while dynamic legs are skipped.
"""

import pytest

from quack import pipeline_checks as pc


class FakeInt32:
    """Stands in for a dynamic (trace-time) value; must disable checking."""


# ── Rule arithmetic, pinned against hand-derived counts ─────────────────────


def test_async_thread_arrives():
    # All-thread arrives: one cp.async producer warp = 32 lanes.
    assert pc.async_thread_arrives(1, per_warp=False) == 32
    # Elected-lane arrives: 4 epilogue warps release once each.
    assert pc.async_thread_arrives(4, per_warp=True) == 4
    # 2-CTA MMA accumulator: both CTAs' 4 epi warps route arrives to the leader.
    assert pc.async_thread_arrives(4, per_warp=True, ctas_routed=2) == 8
    # Scheduler barrier, (2,1) cluster: 7 warps per CTA x 2 CTAs all arrive at CTA 0.
    assert pc.async_thread_arrives(7, per_warp=True, ctas_routed=2) == 14


def test_tma_producer_arrives():
    # Plain TMA load: single elected arrive_and_expect_tx.
    assert pc.tma_producer_arrives() == 1
    # gather_A on sm90: B via TMA (1) + 4 cp.async warps x 32 lanes.
    assert pc.tma_producer_arrives(num_tma_warps=1, cpasync_warps=4) == 129
    # Peer-CTA forwarding warp contributes a single extra arrive.
    assert pc.tma_producer_arrives(cpasync_warps=4, extra_unit_arrives=1) == 130


def test_mcast_peer_ctas():
    # No multicast: only self.
    assert pc.mcast_peer_ctas(num_mcast_ctas_a=1, num_mcast_ctas_b=1) == 1
    # (2,1)/(1,2) clusters: A multicast across 2 CTAs, B not (or vice versa).
    assert pc.mcast_peer_ctas(num_mcast_ctas_a=2, num_mcast_ctas_b=1) == 2
    assert pc.mcast_peer_ctas(num_mcast_ctas_a=1, num_mcast_ctas_b=2) == 2
    # 2x2 cluster, both operands multicast: 2 + 2 - self.
    assert pc.mcast_peer_ctas(num_mcast_ctas_a=2, num_mcast_ctas_b=2) == 3


def test_umma_producer_arrives():
    assert pc.umma_producer_arrives() == 1


def test_sm90_ab_consumer_composition():
    # 2 mma warpgroups (8 warps), (1,2) cluster with B multicast: per-warp releases
    # delivered to both CTAs in the peer set -> 16 arrives. This is the count that was
    # silently wrong in the (1,2) warp-layout corruption bug class.
    peers = pc.mcast_peer_ctas(num_mcast_ctas_a=1, num_mcast_ctas_b=2)
    assert pc.async_thread_arrives(8, per_warp=True, ctas_routed=peers) == 16


# ── check_arrive_count behavior ─────────────────────────────────────────────


def test_check_passes_on_match():
    pc.check_arrive_count("t", 4, pc.async_thread_arrives(4, per_warp=True), num_warps=4)


def test_check_raises_with_facts():
    with pytest.raises(pc.PipelineArriveCountError, match=r"expected 4.*num_epi_warps=4"):
        pc.check_arrive_count(
            "acc.consumer",
            8,
            pc.async_thread_arrives(4, per_warp=True),
            num_epi_warps=4,
        )


def test_check_skips_dynamic_actual():
    # A leader-dependent Int32 count must not be checked (and must not raise TypeError).
    pc.check_arrive_count("t", FakeInt32(), 1)


def test_rules_skip_dynamic_facts():
    # Dynamic inputs disable the rule (return None), which disables the check.
    assert pc.async_thread_arrives(FakeInt32(), per_warp=True) is None
    assert pc.tma_producer_arrives(cpasync_warps=FakeInt32()) is None
    assert pc.mcast_peer_ctas(num_mcast_ctas_a=FakeInt32(), num_mcast_ctas_b=1) is None
    pc.check_arrive_count("t", 123, None)  # expected=None -> no-op
