"""Non-GPU contracts for FlyDSL RMSNorm benchmark timing."""

import pytest

from benchmarks import benchmark_rmsnorm_flydsl as benchmark


def test_dispatch_latency_is_batched_synchronized_wall_time(monkeypatch):
    calls = []
    synchronizations = []
    timestamps = iter((10.0, 10.004))

    monkeypatch.setattr(
        benchmark.torch.cuda,
        "synchronize",
        lambda: synchronizations.append("sync"),
    )
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(timestamps))

    latency_us = benchmark._dispatch_latency_us(
        lambda: calls.append("call"),
        warmup=2,
        iterations=4,
    )

    assert latency_us == pytest.approx(1000.0)
    assert calls == ["call"] * 6
    assert synchronizations == ["sync", "sync"]


def test_dispatch_metrics_are_explicitly_labeled():
    results = []
    benchmark._record(
        results,
        "flydsl",
        "hot_cache_forward_dispatch_latency",
        1.25,
        "us",
    )
    assert results == [
        {
            "provider": "flydsl",
            "metric": "hot_cache_forward_dispatch_latency",
            "value": 1.25,
            "unit": "us",
            "timing": "batched_synchronized_wall_clock",
        }
    ]
