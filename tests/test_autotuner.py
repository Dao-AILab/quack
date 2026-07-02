import pytest

from quack.autotuner import AutotuneConfig


def test_autotune_config_supports_multi_kwarg_hash_and_equality():
    config_a = AutotuneConfig(block_m=128, num_warps=4)
    config_b = AutotuneConfig(block_m=128, num_warps=4)
    config_c = AutotuneConfig(block_m=64, num_warps=4)

    assert config_a == config_b
    assert hash(config_a) == hash(config_b)
    assert config_a != config_c

    timings = {config_a: 1.25, config_c: 2.5}
    assert timings[config_b] == 1.25
    assert len({config_a, config_b, config_c}) == 2


# ---------------------------------------------------------------------------
# Bench loop: defer-and-retry over configs via the async compile pool
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_gpu_warmup needs a GPU")
def test_autotune_bench_loop_defers_and_retries(monkeypatch):
    """A config whose kernel raises CompilePending is rotated to the back and
    retried once its sha polls done; all configs end up benchmarked exactly
    as if they had been warm.

    This encodes the compile-only rip-out contract: the autotuner no longer
    precompiles via fake tensors — the bench loop discovers cold keys with
    the real tensors in-process and overlaps compilation via the pool.
    """
    from quack.autotuner import Autotuner, AutotuneConfig
    from quack.cache import async_compile
    from quack.cache.async_compile import CompilePending

    class _StubPool:
        """poll() reports 'pending' once per sha, then 'done'."""

        def __init__(self):
            self.polls = {}

        def poll(self, sha):
            n = self.polls.get(sha, 0) + 1
            self.polls[sha] = n
            return ("pending" if n == 1 else "done"), None

    stub = _StubPool()
    monkeypatch.setattr(async_compile, "_active_pool", stub)

    bench_order = []
    raised_once = set()

    def kernel(x, block: int = 0):
        # config block=1 is "cold": its first invocation defers.
        if block == 1 and 1 not in raised_once:
            raised_once.add(1)
            raise CompilePending("f" * 64, "fake._compile_kernel")
        bench_order.append(block)

    def do_bench(fn, quantiles=None, **kw):
        fn()
        return [1.0 + bench_order[-1], 1.0, 1.0]  # block=0 fastest

    tuner = Autotuner(
        kernel,
        key=[],
        configs=[AutotuneConfig(block=b) for b in (0, 1, 2)],
        do_bench=do_bench,
    )
    import torch

    x = torch.empty(4, device="cuda")
    tuner(x)

    # Bench order: block=1 deferred, so it benched AFTER block 2 (exactly
    # once). The trailing 0 is __call__'s real invocation with the winner.
    assert bench_order == [0, 2, 1, 0], bench_order
    assert stub.polls == {"f" * 64: 2}  # one rotation, one release
    assert len(tuner.configs_timings) == 3
    best = tuner.cache[next(iter(tuner.cache))]
    assert best.kwargs["block"] == 0  # timings intact despite the deferral


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_gpu_warmup needs a GPU")
def test_autotune_wedged_pool_falls_back_in_process(monkeypatch):
    """A sha that never resolves must not hang the sweep: past the attempt
    cap the config is benched with the pool suppressed (in-process compile),
    so autotuning always terminates.
    """
    import quack.autotuner as at
    from quack.autotuner import Autotuner, AutotuneConfig
    from quack.cache import async_compile
    from quack.cache.async_compile import CompilePending, get_active_pool

    class _WedgedPool:
        def poll(self, sha):
            return "pending", None  # never completes

    monkeypatch.setattr(async_compile, "_active_pool", _WedgedPool())
    monkeypatch.setattr(at, "_POOL_WEDGE_TIMEOUT_S", 0.2)

    benched = []

    def kernel(x, block: int = 0):
        # Defer as long as a pool is visible; succeed once suppressed.
        if block == 1 and get_active_pool() is not None:
            raise CompilePending("e" * 64, "fake._compile_kernel")
        benched.append(block)

    tuner = Autotuner(
        kernel,
        key=[],
        configs=[AutotuneConfig(block=b) for b in (0, 1)],
        do_bench=lambda fn, quantiles=None, **kw: (fn(), [1.0, 1.0, 1.0])[1],
    )
    import torch

    tuner(torch.empty(4, device="cuda"))
    assert benched.count(1) == 1  # eventually ran, via suppress_pool
    assert len(tuner.configs_timings) == 2
