import pytest
import torch.distributed as dist

import quack.autotuner as autotuner


class _FakeDist:
    def __init__(self):
        self.initialized = True
        self.backend = "gloo"
        self.calls = []

    def is_available(self):
        return True

    def is_initialized(self):
        return self.initialized

    def get_backend(self):
        return self.backend

    def barrier(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture
def fake_dist(monkeypatch):
    fake = _FakeDist()
    for name in ("is_available", "is_initialized", "get_backend", "barrier"):
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
