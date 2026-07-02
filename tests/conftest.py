# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""pytest configuration for quack kernel tests.

Kernel-compile workflow (implemented by the reusable
`quack.testing.pytest_plugin` plugin loaded below — downstream projects opt
in with the same `pytest_plugins =` line):

  --async-compile[=N]  Single-pass workflow: on a kernel-compile cache miss,
                       the compile is shipped to a pool of N CPU workers
                       (forkserver sidecar: ~0.1 s/worker, GPU-blind), the
                       test is deferred, other tests run meanwhile, and the
                       deferred test retries once its .o is exported.
                       Zero overhead when the cache is warm. Works with and
                       without xdist.

Single-pass workflow (cold or warm cache, after changing kernel source):
  pytest tests/test_softmax.py --async-compile=16      # compiles overlap tests
  pytest tests/ -n 8 --async-compile=32

Multi-GPU with xdist:
  pytest tests/ -n 4                                   # workers round-robin across GPUs
"""

import os
import subprocess
import json
import logging
import tempfile
from pathlib import Path
from getpass import getuser

import pytest


def _get_gpu_ids():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        gpu_ids = [g.strip() for g in visible.split(",") if g.strip()]
        if gpu_ids:
            return gpu_ids
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if gpu_ids:
                return gpu_ids
    except (FileNotFoundError,):
        pass
    logging.warning("Failed to get gpu ids, use default '0'")
    return ["0"]


def _setup_worker_logging(worker_id, tmp):
    """Configure per-worker file logging for easier debugging of parallel runs."""
    log_file = tmp / f"tests_{worker_id}.log"
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.info("Worker %s logging to %s", worker_id, log_file)


def _assign_xdist_worker_gpu():
    """Narrow each xdist worker to one GPU before any CUDA-touching imports.

    Importing the reusable plugin as ``quack.testing.pytest_plugin`` first
    imports ``quack.__init__`` and CuTe/CUTLASS modules; those imports can call
    ``torch.cuda.is_available()``. If ``CUDA_VISIBLE_DEVICES`` still contains
    the full free-GPU list at that point, later narrowing inside
    ``pytest_configure`` is too late: CUDA has already cached the larger device
    set and workers can all default to logical GPU 0.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker_id:
        return None
    worker_num = int(worker_id.replace("gw", ""))
    gpu_ids = _get_gpu_ids()
    assigned_gpu = gpu_ids[worker_num % len(gpu_ids)]
    os.environ.setdefault(
        "QUACK_XDIST_ORIGINAL_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "")
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = assigned_gpu
    return worker_id, assigned_gpu, gpu_ids


_PRECONFIGURED_WORKER_GPU = _assign_xdist_worker_gpu()

# The `--async-compile` pool and defer-and-retry loop
# live in the reusable plugin. We defer to it only after xdist workers
# have been narrowed to a single GPU, because importing the `quack` package can
# touch CUDA via CUTLASS/PyTorch.
pytest_plugins = ["quack.testing.pytest_plugin"]


# Per-session bookkeeping for the xdist worker-crash retry hook below.
_crash_retried: set[str] = set()


def pytest_handlecrashitem(crashitem, report, sched):
    """Re-queue a worker-crashed test on a fresh worker.

    Background. Sometimes a worker crahses (idk why, still investigating).
    --max-worker-restart spawns a fresh
    replacement worker, but xdist still marks the test that was in flight
    when the death happened as 'failed' (see xdist.dsession.handle_crashitem,
    which synthesises a TestReport with longrepr only and no excinfo).

    pytest-rerunfailures' --only-rerun matches against excinfo and so cannot
    see worker-crash failures. This hook is the xdist-specific entry point:
    it re-queues the crashed test via the scheduler's mark_test_pending API
    and downgrades the synthetic crash report from 'failed' to 'skipped' so
    the run isn't marked red on the leak alone. The retry produces its own
    pass/fail report; if a test deterministically crashes any worker, the
    second crash isn't retried and the failure stands.
    """
    if crashitem in _crash_retried:
        # Already gave this nodeid one retry. If we're here again the bug is
        # in the test (or in code the test exercises), not in cumulative RSS.
        # Let xdist log the failure normally.
        return
    _crash_retried.add(crashitem)

    try:
        sched.mark_test_pending(crashitem)
    except (NotImplementedError, AttributeError, ValueError):
        # loadscope / each / unexpected scheduler shape: don't try to be
        # clever, just let the failure go through.
        return

    # Mutate the synthetic crash report so it isn't counted as a failure.
    # outcome='skipped' is the standard pytest signal; longrepr for skipped
    # tests is conventionally a (path, lineno, reason) tuple.
    fspath = report.location[0] if getattr(report, "location", None) else ""
    report.outcome = "skipped"
    report.longrepr = (
        fspath,
        0,
        f"worker crashed; re-queued for one retry on a fresh worker "
        f"({crashitem}). See cutlass#3062 background in tests/conftest.py.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # Compile-only context lifecycle is owned by quack.testing.pytest_plugin.
    # This hook handles only project-specific concerns: xdist GPU assignment
    # logging. The actual assignment happens at conftest import time above so
    # it beats CUDA-touching imports from the reusable plugin.
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id:
        tmp = Path(tempfile.gettempdir()) / getuser() / "quack_tests"
        tmp.mkdir(parents=True, exist_ok=True)
        assignment = _PRECONFIGURED_WORKER_GPU or _assign_xdist_worker_gpu()
        _setup_worker_logging(worker_id, tmp)
        if assignment is not None:
            assigned_worker, assigned_gpu, gpu_ids = assignment
            logging.info(
                "Worker %s assigned CUDA_VISIBLE_DEVICES=%s from visible GPUs %s",
                assigned_worker,
                assigned_gpu,
                gpu_ids,
            )


def pytest_collection_finish(session):
    """Print a summary of collected tests grouped by file and function."""
    if not session.items:
        return
    from collections import defaultdict

    counts = defaultdict(lambda: defaultdict(int))
    for item in session.items:
        file_name = item.location[0]
        func_name = item.originalname if hasattr(item, "originalname") else item.name
        counts[file_name][func_name] += 1
    summary = {f: dict(funcs) for f, funcs in sorted(counts.items())}
    total = len(session.items)
    session.config.pluginmanager.get_plugin("terminalreporter").write_line(
        f"Collected {total} tests: {json.dumps(summary, indent=2)}"
    )


# Compile-only error-swallow hooks live in quack.testing.pytest_plugin. The
# only normal-mode hook we keep here is the OOM-retry, which is QuACK-specific.


def _is_oom(exc_type, exc_val):
    """Check if an exception is a CUDA out-of-memory error."""
    import torch

    if issubclass(exc_type, torch.OutOfMemoryError):
        return True
    if issubclass(exc_type, RuntimeError) and "out of memory" in str(exc_val).lower():
        return True
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Retry once on CUDA OOM after freeing GPU memory."""
    outcome = yield
    if outcome.excinfo is not None and _is_oom(*outcome.excinfo[:2]):
        import gc
        import torch

        logging.warning("OOM in %s, freeing GPU memory and retrying once", item.nodeid)
        gc.collect()
        torch.cuda.empty_cache()
        outcome.force_result(None)
        item.runtest()
