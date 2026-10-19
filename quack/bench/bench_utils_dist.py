"""Distributed benchmarking helpers (collective-safe timing).

Kept separate from bench_utils.py so the single-GPU helpers stay import-light and
match upstream; anything that assumes torch.distributed lives here.
"""

import torch


def do_bench_all(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """Adapted from Triton do_bench. Issues with do_bench for multi-GPU scenarios:
      (1) Triton interprets warmup/rep as ms, and derives the count per-rank from a local estimate,
          so under skew ranks can disagree on the estimate and a collective fn() deadlocks.
      (2) Using fixed warmup/rep counts across ranks can cause throttling for longer vs shorter kernels.
    Solution: interpret warmup/rep as ms, derive the count per-rank from a local estimate,
    then sync the counts across ranks."""
    import torch.distributed as dist
    import triton.runtime as triton_runtime
    from triton.testing import _summarize_statistics

    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = triton_runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = triton_runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        triton_runtime.driver.active.clear_cache(cache)
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # sync counts across ranks so a collective fn() can't diverge and deadlock
    if dist.is_initialized():
        c = torch.tensor([n_warmup, n_repeat], device="cuda", dtype=torch.int64)
        dist.all_reduce(c, op=dist.ReduceOp.MAX)
        n_warmup, n_repeat = int(c[0].item()), int(c[1].item())

    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        triton_runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)
