# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""pytest configuration for quack kernel tests.

Supports:
  --compile-only    Compile all kernels (populating .o cache), skip actual execution.
                    Uses FakeTensorMode (no GPU memory) so you can use many xdist workers.
                    Works without a GPU if QUACK_ARCH and CUTE_DSL_ARCH are set.

Two-pass workflow (after changing kernel source):
  pytest tests/test_softmax.py --compile-only -n 64   # parallel compile, no GPU memory
  pytest tests/test_softmax.py                         # instant .o loads

CPU-only compilation (no GPU needed):
  QUACK_ARCH=90 CUTE_DSL_ARCH=sm_90a pytest tests/ --compile-only -n 64

Single-pass workflow (cache already warm):
  pytest tests/test_softmax.py                         # all .o cache hits

Multi-GPU with xdist:
  pytest tests/ -n 4                                   # workers round-robin across GPUs
"""

import os
import subprocess
import json
import time
import logging
import tempfile
from pathlib import Path
from getpass import getuser

import pytest


_compile_only = False
_fake_mode = None


def pytest_addoption(parser):
    parser.addoption(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile all kernels and export .o cache, skip actual kernel execution. "
        "Use with -n N (pytest-xdist) for parallel compilation.",
    )


def _get_gpu_ids():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [g.strip() for g in visible.split(",")]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()
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


def pytest_configure(config):
    global _compile_only, _fake_mode

    try:
        _compile_only = config.getoption("--compile-only", default=False)
    except (ValueError, AttributeError):
        _compile_only = False

    # Assign GPUs to xdist workers round-robin (skip for CPU-only compile)
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id and not (_compile_only and not _has_gpu()):
        tmp = Path(tempfile.gettempdir()) / getuser() / "quack_tests"
        tmp.mkdir(parents=True, exist_ok=True)
        worker_num = int(worker_id.replace("gw", ""))
        cached_gpu_ids = tmp / "gpu_ids.json"
        if worker_num == 0:
            gpu_ids = _get_gpu_ids()
            with cached_gpu_ids.open(mode="w") as f:
                json.dump(gpu_ids, f)
        else:
            while not cached_gpu_ids.exists():
                time.sleep(0.1)
            with cached_gpu_ids.open() as f:
                gpu_ids = json.load(f)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_num % len(gpu_ids)]
        _setup_worker_logging(worker_id, tmp)

    if _compile_only:
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        import quack.cache_utils

        quack.cache_utils.COMPILE_ONLY = True
        if torch.cuda.is_available():
            torch.cuda.init()
        _fake_mode = FakeTensorMode()
        _fake_mode.__enter__()


def _has_gpu():
    """Check for GPU without initializing CUDA."""
    import torch

    return torch.cuda.is_available()


def pytest_unconfigure(config):
    global _fake_mode
    if _fake_mode is not None:
        _fake_mode.__exit__(None, None, None)
        _fake_mode = None


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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """In --compile-only mode, swallow setup errors (e.g. fixtures allocating CUDA tensors)."""
    if not _compile_only:
        yield
        return
    outcome = yield
    if outcome.excinfo is not None:
        outcome.force_result(None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """In --compile-only mode, swallow all errors — we only care about compilation."""
    if not _compile_only:
        yield
        return
    outcome = yield
    if outcome.excinfo is not None:
        outcome.force_result(None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """In --compile-only mode, swallow teardown errors."""
    if not _compile_only:
        yield
        return
    outcome = yield
    if outcome.excinfo is not None:
        outcome.force_result(None)
