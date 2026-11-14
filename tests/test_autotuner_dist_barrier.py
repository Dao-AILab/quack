import pytest
import torch
import torch.distributed as dist

import quack.autotuner as autotuner


class _FakeDist:
    def __init__(self):
        self.initialized = True
        self.backend = "gloo"
        self.calls = []
        self.compiling = []

    def is_available(self):
        return True

    def is_initialized(self):
        return self.initialized

    def get_backend(self):
        return self.backend

    def get_rank(self):
        return 0

    def barrier(self, **kwargs):
        self.calls.append(kwargs)
        self.compiling.append(torch.compiler.is_compiling())


@pytest.fixture
def fake_dist(monkeypatch):
    fake = _FakeDist()
    for name in ("is_available", "is_initialized", "get_backend", "get_rank", "barrier"):
        monkeypatch.setattr(dist, name, getattr(fake, name))
    return fake


def test_no_barrier_by_default(monkeypatch, fake_dist):
    monkeypatch.delenv("QUACK_AUTOTUNE_DIST_BARRIER", raising=False)
    autotuner._sync_ranks_after_autotune()
    assert fake_dist.calls == []


def test_no_barrier_without_process_group(monkeypatch, fake_dist):
    monkeypatch.setenv("QUACK_AUTOTUNE_DIST_BARRIER", "1")
    fake_dist.initialized = False
    autotuner._sync_ranks_after_autotune()
    assert fake_dist.calls == []


def test_barrier_when_enabled(monkeypatch, fake_dist):
    monkeypatch.setenv("QUACK_AUTOTUNE_DIST_BARRIER", "1")
    autotuner._sync_ranks_after_autotune()
    assert fake_dist.calls == [{}]


def test_barrier_runs_eagerly_under_torch_compile(monkeypatch, fake_dist):
    monkeypatch.setenv("QUACK_AUTOTUNE_DIST_BARRIER", "1")
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    @torch.compile(backend=backend)
    def fn(x):
        y = x * 2
        autotuner._sync_ranks_after_autotune()
        return y + 1

    for x in (torch.arange(8, dtype=torch.float32), torch.randn(8)):
        torch.testing.assert_close(fn(x), x * 2 + 1)

    assert graphs
    assert fake_dist.calls == [{}, {}]
    assert fake_dist.compiling == [False, False]
