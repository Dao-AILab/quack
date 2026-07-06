# Copyright (c) 2026, Tri Dao.
"""Regression tests for the async compile pool (``--async-compile``).

Each test encodes a specific failure mode discovered while building the
feature — see quack/cache/async_compile.py and the defer-and-retry loop in
quack/testing/pytest_plugin.py.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from quack.cache.async_compile import CompilePending, CompilePool, _flock_held_exclusively


# ---------------------------------------------------------------------------
# CompilePending semantics
# ---------------------------------------------------------------------------


def test_compile_pending_is_base_exception():
    """CompilePending must NOT be catchable by ``except Exception``.

    Regression: test bodies with broad ``try/except Exception`` (or
    ``pytest.raises(Exception)``) around a kernel call would otherwise
    swallow the deferral signal and turn a never-executed test into a false
    pass.
    """
    assert issubclass(CompilePending, BaseException)
    assert not issubclass(CompilePending, Exception)

    with pytest.raises(BaseException):
        try:
            raise CompilePending("0" * 64, "some._compile_fn")
        except Exception:  # must NOT catch it
            pytest.fail("except Exception swallowed CompilePending")

    exc = CompilePending("ab" * 32, "quack.foo._compile_bar")
    assert exc.sha == "ab" * 32
    assert "quack.foo._compile_bar" in str(exc)


# ---------------------------------------------------------------------------
# Pool bookkeeping: poll states, external-compile detection
# ---------------------------------------------------------------------------


def test_flock_probe_detects_exclusive_holder(tmp_path):
    """The external-compile probe: exclusive flock held => True, free => False.

    This is what lets one xdist worker defer on a key that another worker's
    pool is already compiling, instead of burning a pool slot on a duplicate
    compile that would just block on the same flock.
    """
    import fcntl

    lock_path = tmp_path / "key.lock"
    assert not _flock_held_exclusively(str(lock_path))  # nonexistent -> free

    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert _flock_held_exclusively(str(lock_path))
        fcntl.flock(fd, fcntl.LOCK_UN)
        assert not _flock_held_exclusively(str(lock_path))
    finally:
        os.close(fd)


def test_pool_poll_external_lifecycle(tmp_path):
    """poll() on an externally-compiled key: pending while locked, done once
    the .o appears, back to new if the external compiler died lockless.

    Regression: without the died-without-.o transition, a crashed foreign
    compiler would leave every waiter deferring forever (until the wedge
    deadline force-synced them one by one).
    """
    import fcntl

    pool = CompilePool.__new__(CompilePool)  # skip executor spawn: bookkeeping only
    pool._futures = {}
    pool._external = {}
    pool.n_submitted = 0

    sha = "e" * 64
    o_path = tmp_path / f"{sha}.o"
    lock_path = tmp_path / f"{sha}.lock"

    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        pool.mark_external(sha, str(o_path), str(lock_path))
        assert pool.poll(sha) == ("pending", None)  # locked, no .o yet

        o_path.write_bytes(b"fake object file")
        assert pool.poll(sha) == ("done", None)  # .o landed
        assert sha not in pool._external  # consumed

        # Crash case: locked then released without producing a .o.
        sha2 = "f" * 64
        o2 = tmp_path / f"{sha2}.o"
        pool.mark_external(sha2, str(o2), str(lock_path))
        assert pool.poll(sha2) == ("pending", None)
        fcntl.flock(fd, fcntl.LOCK_UN)
        assert pool.poll(sha2) == ("new", None)  # forgotten -> resubmittable
        assert sha2 not in pool._external
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# End-to-end: defer-and-retry through a real pytest session
# ---------------------------------------------------------------------------

_INNER_TEST_SRC = textwrap.dedent(
    """
    import torch
    from quack.rmsnorm import rmsnorm_fwd

    def _run(N):
        x = torch.randn(32, N, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, device="cuda", dtype=torch.float32)
        out, *_ = rmsnorm_fwd(x, w)
        ref = torch.nn.functional.rms_norm(x.float(), (N,), weight=w).to(x.dtype)
        torch.testing.assert_close(out, ref, atol=3e-2, rtol=1e-2)

    def test_a():
        _run({N_A})

    def test_b():
        _run({N_B})
    """
)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="end-to-end defer needs a GPU"
)
def test_defer_and_retry_cold_session(tmp_path):
    """Cold keys + --async-compile: tests defer, retry, and verify numerics.

    Two test *modules* are used so the deferred retry crosses a module
    boundary — the regression that produced pytest's "previous item was not
    torn down properly" SetupState assert before the misprediction guard in
    ``_run_protocol`` (fixture teardown is scoped to a *predicted* next item
    and deferral breaks the prediction chain).
    """
    suite = tmp_path / "suite"
    suite.mkdir()
    # Unusual N values so the keys are cold in the throwaway cache dir.
    (suite / "test_defer_mod_a.py").write_text(_INNER_TEST_SRC.format(N_A=704, N_B=1216))
    (suite / "test_defer_mod_b.py").write_text(_INNER_TEST_SRC.format(N_A=1728, N_B=2240))
    (suite / "conftest.py").write_text('pytest_plugins = ["quack.testing.pytest_plugin"]\n')

    env = dict(os.environ)
    env["QUACK_CACHE_DIR"] = str(tmp_path / "cache")
    env.pop("PYTEST_XDIST_WORKER", None)  # inner session is single-proc
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-q",
            "-p",
            "no:cacheprovider",
            "--async-compile=4",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        cwd=str(suite),
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"inner pytest failed:\n{out}"
    assert "4 passed" in out, out
    # The session must actually have exercised the pool + deferral path.
    assert "keys submitted" in out, out
    n_submitted = int(out.split("async-compile: ")[1].split(" keys")[0])
    assert n_submitted >= 4, out  # one fwd key per distinct N


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="end-to-end defer needs a GPU"
)
def test_warm_session_with_flag_submits_nothing(tmp_path):
    """Warm cache + --async-compile must be a no-op (no submits, no defers).

    Guards the lazy-pool property: enabling the flag unconditionally (e.g.
    in an alias or CI default) must cost nothing when the cache is warm.
    """
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_warm.py").write_text(_INNER_TEST_SRC.format(N_A=704, N_B=1216))
    (suite / "conftest.py").write_text('pytest_plugins = ["quack.testing.pytest_plugin"]\n')

    env = dict(os.environ)
    env["QUACK_CACHE_DIR"] = str(tmp_path / "cache")
    env.pop("PYTEST_XDIST_WORKER", None)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(suite),
        "-q",
        "-p",
        "no:cacheprovider",
        "--async-compile=4",
    ]
    kw = dict(capture_output=True, text=True, env=env, timeout=600, cwd=str(suite))
    first = subprocess.run(cmd, **kw)
    assert first.returncode == 0, first.stdout + first.stderr
    second = subprocess.run(cmd, **kw)
    out = second.stdout + second.stderr
    assert second.returncode == 0, out
    assert "0 keys submitted" in out, out
    assert "0 test deferrals" in out, out


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="end-to-end defer needs a GPU"
)
def test_oom_retry_compile_pending_defers_not_warns(tmp_path):
    """OOM-retry hookwrapper + CompilePending on the retry must defer cleanly.

    Regression: tests/conftest.py's OOM-retry hook re-invokes
    ``item.runtest()`` inside an old-style hookwrapper teardown. When the
    retry hit a cold jit_cache miss (--async-compile), the resulting
    ``CompilePending`` escaped the teardown: pluggy emitted
    ``PluggyTeardownRaisedWarning``, the plugin's defer machinery (which had
    already run for this phase) never saw it, and the test reported as a
    CompilePending *failure* instead of deferring (or, worse, passing from a
    half-run attempt). Full analysis: the ``pytest_runtest_call`` docstring
    in tests/conftest.py.

    Uses the real repo conftest so the actual hook code is exercised. The
    inner test scripts the exact sequence: attempt 1 raises a synthetic OOM,
    the in-teardown retry raises CompilePending, and the defer-loop retry
    passes (the sha is unknown to the pool, so it polls "new" and re-runs
    immediately).
    """
    suite = tmp_path / "suite"
    suite.mkdir()
    shutil.copy(Path(__file__).parent / "conftest.py", suite / "conftest.py")
    (suite / "test_oom_retry.py").write_text(
        textwrap.dedent(
            """
            import torch
            from quack.cache.async_compile import CompilePending

            CALLS = {"n": 0}

            def test_oom_then_cold_compile():
                CALLS["n"] += 1
                if CALLS["n"] == 1:
                    raise torch.OutOfMemoryError("synthetic CUDA out of memory")
                if CALLS["n"] == 2:
                    # This is the conftest OOM-retry's item.runtest() call:
                    # simulate the retry hitting a cold kernel compile.
                    raise CompilePending("a" * 64, "fake._compile_rmsnorm_bwd")
                assert CALLS["n"] == 3  # defer-loop retry succeeds
            """
        )
    )

    env = dict(os.environ)
    env["QUACK_CACHE_DIR"] = str(tmp_path / "cache")
    env.pop("PYTEST_XDIST_WORKER", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-q",
            "-p",
            "no:cacheprovider",
            "--async-compile=2",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        cwd=str(suite),
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"inner pytest failed:\n{out}"
    assert "1 passed" in out, out
    assert "PluggyTeardownRaisedWarning" not in out, out
    assert "1 test deferrals" in out, out
