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
    try:
        tuner(x)
    except CompilePending:  # the quack pytest plugin would report it as a pass
        pytest.fail("CompilePending escaped the autotuner")

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

    try:
        tuner(torch.empty(4, device="cuda"))
    except CompilePending:  # the quack pytest plugin would report it as a pass
        pytest.fail("CompilePending escaped the autotuner")
    assert benched.count(1) == 1  # eventually ran, via suppress_pool
    assert len(tuner.configs_timings) == 2


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_gpu_warmup needs a GPU")
@pytest.mark.parametrize("requires_grad", [False, True])
def test_autotune_restore_value_forced_clone_failure(monkeypatch, requires_grad):
    """When the L2-cold clone sets do not fit, the legacy bench runs the trials
    on the caller's tensor and restores it after each one, so delta is added to
    it exactly once. With requires_grad, the restore hooks must run under no_grad.
    """
    import torch

    from quack import autotuner
    from quack.autotuner import Autotuner, AutotuneConfig

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("forced clone failure")

    monkeypatch.setattr(autotuner, "_clone_l2_rotate_inputs", fail)
    monkeypatch.delenv("QUACK_CACHE_AUTOTUNING", raising=False)  # a disk-cached pick skips trials

    launches = []

    def kernel(acc, x, block: int = 0):
        launches.append(acc.data_ptr())
        # A raw kernel's write: autograd does not see it.
        with torch.no_grad():
            acc.add_(x)

    tuner = Autotuner(
        kernel,
        key=[],
        configs=[AutotuneConfig(block=b) for b in (0, 1, 2)],
        restore_value=["acc"],
    )
    torch.manual_seed(0)
    buf0 = torch.randn(4, dtype=torch.float32, device="cuda")
    delta = torch.randn_like(buf0)
    # A view from chunk(): after an in-place write, grad mode rejects any op on it, even
    # the pre-hook's clone. With requires_grad, the test fails unless both hooks use no_grad.
    buf = torch.cat([buf0, buf0], dim=0).requires_grad_(requires_grad).chunk(2, dim=0)[0]
    tuner(buf, delta)

    assert launches.count(buf.data_ptr()) > 3  # the trials ran on the caller's tensor
    assert all(t[0] != float("inf") for t in tuner.configs_timings.values())  # no trial raised
    assert torch.equal(buf, buf0 + delta)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_gpu_warmup needs a GPU")
def test_autotune_restore_value_keeps_l2_cold_bench(monkeypatch):
    """restore_value does not switch off the L2-cold bench: every config is
    timed on clones, and the caller's tensor sees only the real call.
    """
    import torch

    from quack import autotuner
    from quack.autotuner import Autotuner, AutotuneConfig

    l2_cold_blocks = []
    bench_l2_cold = autotuner._bench_cuda_graph_l2_rotate

    def recording_bench_l2_cold(*args, extra_kwargs, **kwargs):
        l2_cold_blocks.append(extra_kwargs["block"])
        return bench_l2_cold(*args, extra_kwargs=extra_kwargs, **kwargs)

    monkeypatch.setattr(autotuner, "_bench_cuda_graph_l2_rotate", recording_bench_l2_cold)
    monkeypatch.delenv("QUACK_CACHE_AUTOTUNING", raising=False)  # a disk-cached pick skips trials

    launches = []

    def kernel(acc, x, block: int = 0):
        launches.append(acc.data_ptr())
        acc.add_(x)

    tuner = Autotuner(
        kernel,
        key=[],
        configs=[AutotuneConfig(block=b) for b in (0, 1, 2)],
        restore_value=["acc"],
    )
    torch.manual_seed(0)
    buf0 = torch.randn(4, dtype=torch.float32, device="cuda")
    delta = torch.randn_like(buf0)
    buf = buf0.clone()
    tuner(buf, delta)

    assert sorted(l2_cold_blocks) == [0, 1, 2]  # every config, on clones
    assert launches.count(buf.data_ptr()) == 1  # only the real call
    assert all(t[0] != float("inf") for t in tuner.configs_timings.values())  # no trial raised
    assert torch.equal(buf, buf0 + delta)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_gpu_warmup needs a GPU")
def test_autotune_restore_value_on_compile_pending(monkeypatch):
    """A config whose kernel writes and then raises CompilePending (a
    BaseException) has its write undone before it is rotated to the back and
    retried, so delta is added to the caller's tensor exactly once.
    """
    import torch

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
    monkeypatch.delenv("QUACK_CACHE_AUTOTUNING", raising=False)  # a disk-cached pick skips trials

    raised_once = set()

    def kernel(acc, x, block: int = 0):
        # Every config is "cold": its first invocation writes, then defers, like a
        # fn whose first kernel ran while its second is still compiling.
        acc.add_(x)
        if block not in raised_once:
            raised_once.add(block)
            raise CompilePending(str(block) * 64, "fake._compile_kernel")

    # A custom do_bench selects the legacy bench, which runs on the caller's tensor.
    def do_bench(fn, quantiles=None, **kw):
        fn()
        return [1.0, 1.0, 1.0]

    tuner = Autotuner(
        kernel,
        key=[],
        configs=[AutotuneConfig(block=b) for b in (0, 1, 2)],
        restore_value=["acc"],
        do_bench=do_bench,
    )
    torch.manual_seed(0)
    buf0 = torch.randn(4, dtype=torch.float32, device="cuda")
    delta = torch.randn_like(buf0)
    buf = buf0.clone()
    try:
        tuner(buf, delta)
    except CompilePending:  # the quack pytest plugin would report it as a pass
        pytest.fail("CompilePending escaped the autotuner")

    assert stub.polls == {str(b) * 64: 2 for b in (0, 1, 2)}  # one rotation, one release each
    assert len(tuner.configs_timings) == 3
    assert torch.equal(buf, buf0 + delta)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_gpu_warmup needs a GPU")
def test_autotune_restore_value_gemm_add_c_is_out(monkeypatch):
    """gemm_add with C = out takes the add_to_output path, which reads out: when
    the L2-cold clone sets do not fit, the legacy bench runs the trials on out
    itself and restores it, so A @ B is added to out exactly once.
    """
    import torch

    from quack import autotuner
    from quack.gemm_interface import gemm_add, gemm_tuned, prune_invalid_gemm_configs

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("forced clone failure")

    monkeypatch.setattr(autotuner, "_clone_l2_rotate_inputs", fail)

    launches = []
    gemm = gemm_tuned.fn

    def recording_gemm(A, B, out, *args, **kwargs):
        launches.append(out.data_ptr())
        return gemm(A, B, out, *args, **kwargs)

    monkeypatch.setattr(gemm_tuned, "fn", recording_gemm)

    # A cached pick (in memory or on disk) would skip the trials: force a fresh tune.
    monkeypatch.setattr(gemm_tuned, "cache", {})
    monkeypatch.setattr(gemm_tuned, "cache_results", False)

    m, n, k = 512, 384, 256
    torch.manual_seed(0)
    A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
    out = torch.randn(m, n, dtype=torch.float32, device="cuda")
    # Three configs the prune keeps, for a short tune.
    pruned_configs = prune_invalid_gemm_configs(gemm_tuned.configs, {"A": A})[:3]
    monkeypatch.setattr(gemm_tuned, "configs", pruned_configs)
    ref = out + A.float() @ B.float()
    gemm_add(A=A, B=B, C=out, out=out, tuned=True)

    assert launches.count(out.data_ptr()) > 3  # the trials ran on out itself
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-3)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_gpu_warmup needs a GPU")
def test_autotune_disallow_raises_on_cache_miss(monkeypatch, tmp_path):
    """With QUACK_DISALLOW_AUTOTUNING=1, a cached pick (on disk or in memory) still
    runs, and a cache miss raises instead of autotuning.
    """
    import torch

    from quack.autotuner import Autotuner, AutotuneConfig

    monkeypatch.setenv("QUACK_CACHE_DIR", str(tmp_path))  # hermetic disk cache
    monkeypatch.delenv("QUACK_CACHE_AUTOTUNING", raising=False)  # it turns the disk cache on
    monkeypatch.delenv("QUACK_FORCE_CACHE_UPDATE", raising=False)  # it skips the disk read

    launches = []

    def kernel(x, block: int = 0):
        launches.append(block)

    def do_bench(fn, quantiles=None, **kw):
        fn()
        return [3.0 - launches[-1], 1.0, 1.0]  # block=2 fastest, not the first config

    # Built before QUACK_DISALLOW_AUTOTUNING is set, like an @autotune tuner at import time.
    tuner = Autotuner(
        kernel,
        key=[],
        configs=[AutotuneConfig(block=b) for b in (0, 1, 2)],
        do_bench=do_bench,
        cache_results=False,
    )
    x = torch.empty(4, device="cuda")
    y = torch.empty(8, device="cuda")

    # A miss raises before any trial, also without a disk cache.
    monkeypatch.setenv("QUACK_DISALLOW_AUTOTUNING", "1")
    with pytest.raises(RuntimeError, match="QUACK_DISALLOW_AUTOTUNING=1"):
        tuner(x)
    assert launches == []

    # Only "1" disallows: every config is tried, the pick is saved to disk, then it runs.
    monkeypatch.setenv("QUACK_DISALLOW_AUTOTUNING", "0")
    tuner.cache_results = True
    tuner(x)
    assert launches == [0, 1, 2, 2]

    # As in a new process: the pick is read from disk, with no trials.
    monkeypatch.setenv("QUACK_DISALLOW_AUTOTUNING", "1")
    tuner.cache.clear()
    launches.clear()
    tuner(x)
    assert launches == [2]

    # A shape never tuned is a miss.
    with pytest.raises(RuntimeError, match="QUACK_DISALLOW_AUTOTUNING=1"):
        tuner(y)

    # The pick is in memory now: it runs with no disk cache to read.
    monkeypatch.setenv("QUACK_CACHE_DIR", str(tmp_path / "empty"))
    tuner(x)
    assert launches == [2, 2]
@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_autotuning_does_not_pollute_selective_checkpoint_cache():
    import torch
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        checkpoint,
        create_selective_checkpoint_contexts,
    )

    from quack.autotuner import Autotuner

    def matmul(a, b, block=0):
        return a @ b

    tuner = Autotuner(
        matmul,
        key=[],
        configs=[AutotuneConfig(block=0), AutotuneConfig(block=1)],
    )
    saved_matmuls = []

    def policy(context, op, *args, **kwargs):
        if op == torch.ops.aten.mm.default:
            if not context.is_recompute:
                saved_matmuls.append(tuple(args[0].shape))
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    a = torch.randn(32, 32, device="cuda", requires_grad=True)
    b = torch.randn(32, 32, device="cuda", requires_grad=True)
    expected = a @ b
    expected_grads = torch.autograd.grad(expected.sum(), (a, b))
    actual = checkpoint(
        tuner,
        a,
        b,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(policy),
    )
    assert saved_matmuls == [(32, 32)]
    actual_grads = torch.autograd.grad(actual.sum(), (a, b))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
