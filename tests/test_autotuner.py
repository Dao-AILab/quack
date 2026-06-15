import os
import subprocess
import sys
import textwrap
import threading
import time

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
# _PrecompileHandle: async submit + crash tracking (gaps 2 + 3)
# ---------------------------------------------------------------------------


def test_precompile_handle_noop_is_safe():
    """Empty handle (no workers spawned): wait_for / is_failed / shutdown
    are all no-ops. The bench loop relies on this when ``_precompile``
    short-circuits (cache disabled, only 1 config, cache hit)."""
    from quack.autotuner import _PrecompileHandle

    h = _PrecompileHandle()
    t0 = time.time()
    h.wait_for(0)
    h.wait_for(99)
    assert time.time() - t0 < 0.1, "wait_for on missing index must be instant"
    assert not h.is_failed(0)
    h.shutdown()  # idempotent + safe on empty handle


def test_precompile_handle_reader_thread_signals_events_in_order(monkeypatch):
    """Reader thread sets per-config events as replies arrive.

    Simulates a worker by feeding ``_recv_from_worker`` two replies and
    asserts the events fire in order. This isolates the threading /
    bookkeeping from the actual subprocess + cute.compile machinery.
    """
    import quack.autotuner as at

    # Fake worker: stub stdout that yields two ``"OK"`` replies then EOF.
    class _FakeStdout:
        def __init__(self, replies):
            import pickle
            import struct

            self._buffer = b"".join(
                struct.pack("<I", len(pickle.dumps(r))) + pickle.dumps(r) for r in replies
            )
            self._pos = 0

        def read(self, n):
            chunk = self._buffer[self._pos : self._pos + n]
            self._pos += len(chunk)
            return chunk

    class _FakeWorker:
        def __init__(self, replies):
            self.stdout = _FakeStdout(replies)

    handle = at._PrecompileHandle()
    handle.events[0] = threading.Event()
    handle.events[1] = threading.Event()
    worker = _FakeWorker(["OK", "OK"])

    t = threading.Thread(
        target=at._reader_thread_main,
        args=(handle, worker, [0, 1]),
        daemon=True,
    )
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), "reader thread did not terminate after replies exhausted"

    assert handle.events[0].is_set()
    assert handle.events[1].is_set()
    assert not handle.is_failed(0)
    assert not handle.is_failed(1)


def test_precompile_handle_truncated_body_does_not_deadlock():
    """Worker SIGKILL'd between writing header and body must NOT deadlock.

    Regression for the reviewer-flagged blocker: ``_recv_from_worker`` used
    to call ``pickle.loads(stream.read(length))`` without checking the
    actual read length. A worker SIGKILL'd by its parent-death watchdog
    (or by OOM-killer) between writing the 4-byte header and writing the
    full body raised ``UnpicklingError: pickle data was truncated``
    inside the reader thread; the thread died without setting events;
    ``handle.wait_for(i)`` had no timeout and hung the bench loop forever.

    This test feeds a fake stdout with a valid header but a truncated body
    and asserts the reader thread treats it as EOF (= worker crashed), not
    as a fatal exception.
    """
    import quack.autotuner as at

    class _TruncatedStdout:
        """4-byte header says length=N, but body returns only ``N // 2`` bytes."""

        def __init__(self, declared_length: int, actual_body: bytes):
            import struct

            self._buffer = struct.pack("<I", declared_length) + actual_body
            self._pos = 0

        def read(self, n):
            chunk = self._buffer[self._pos : self._pos + n]
            self._pos += len(chunk)
            return chunk

    class _FakeWorker:
        def __init__(self):
            # Claim length=100, but only provide 5 bytes of body. Worker died.
            self.stdout = _TruncatedStdout(declared_length=100, actual_body=b"hello")

    handle = at._PrecompileHandle()
    for i in range(2):
        handle.events[i] = threading.Event()
    worker = _FakeWorker()

    t = threading.Thread(
        target=at._reader_thread_main,
        args=(handle, worker, [0, 1]),
        daemon=True,
    )
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), (
        "reader thread did not terminate on truncated body; likely raised "
        "UnpicklingError and died with events unset (the original bug)"
    )

    # Both events set, both marked failed, no deadlock.
    assert handle.events[0].is_set()
    assert handle.events[1].is_set()
    assert handle.is_failed(0)
    assert handle.is_failed(1)


def test_precompile_handle_worker_crash_marks_remaining_failed():
    """If the worker dies mid-stream, all *remaining* assigned configs are
    marked failed (and their events set) so callers don't deadlock on
    ``wait_for`` and the bench loop's in-process retry path kicks in.

    Regression: the old ``_precompile`` silently dropped configs assigned
    to a crashed worker. Their bench timings included compile-on-demand
    cost from jit_cache, biasing the autotune pick toward configs that
    happened to land on healthy workers.
    """
    import quack.autotuner as at

    # Fake worker: one ``"OK"`` reply, then EOF (= crashed mid-second-task).
    class _FakeStdout:
        def __init__(self):
            import pickle
            import struct

            data = pickle.dumps("OK")
            self._buffer = struct.pack("<I", len(data)) + data
            self._pos = 0

        def read(self, n):
            chunk = self._buffer[self._pos : self._pos + n]
            self._pos += len(chunk)
            return chunk

    class _FakeWorker:
        def __init__(self):
            self.stdout = _FakeStdout()

    handle = at._PrecompileHandle()
    for i in range(3):
        handle.events[i] = threading.Event()
    worker = _FakeWorker()

    t = threading.Thread(
        target=at._reader_thread_main,
        args=(handle, worker, [0, 1, 2]),
        daemon=True,
    )
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), "reader thread did not terminate after EOF"

    # First config completed normally
    assert handle.events[0].is_set()
    assert not handle.is_failed(0)
    # Configs 1 and 2 were in-flight on the crashed worker — marked failed
    # and their events set so wait_for() doesn't deadlock.
    assert handle.events[1].is_set()
    assert handle.events[2].is_set()
    assert handle.is_failed(1)
    assert handle.is_failed(2)
    assert "crashed" in handle.failures[1]
    assert "crashed" in handle.failures[2]


# ---------------------------------------------------------------------------
# Watchdog: orphan worker self-SIGKILLs after parent dies (gap 1)
# ---------------------------------------------------------------------------


def test_compile_worker_watchdog_terminates_orphan(tmp_path):
    """Spawn a worker that arms the watchdog and busy-sleeps; verify the
    worker exits after the parent dies, *and* that the only way it can
    exit is via the watchdog.

    The previous version of this test went through the real worker's
    ``cw.main()`` loop and connected stdin to a ``subprocess.PIPE``. When
    the parent (the inner ``parent.py`` script) exited, the OS closed the
    pipe's write end, the worker's ``_recv(stdin)`` returned ``None``, and
    the worker exited via the normal "stdin EOF -> break" path — *without*
    ever needing the watchdog. That made the test a false positive: it
    passed even with the watchdog disabled.

    To genuinely exercise the watchdog, the spawned 'worker' here is a
    minimal inline script that:
      * imports ``quack._compile_worker``,
      * sets ``_WATCHDOG_POLL_SECS`` to a small value for test speed,
      * sets ``QUACK_COMPILE_WORKER_PARENT_PID`` to its parent's pid,
      * calls ``_install_parent_watchdog()``, then
      * blocks in ``time.sleep(120)``.

    This script does NOT go through ``cw.main()``. There is no stdin read
    that could complete on EOF. The only thing that can terminate the
    sleep is ``os.kill(self, SIGKILL)`` from the watchdog thread.

    Sequence:
      1. Test process T spawns ``parent.py`` (process P).
      2. P spawns the busy-sleep worker W with
         ``QUACK_COMPILE_WORKER_PARENT_PID = P.pid`` and stdin=DEVNULL
         (so even if anything ever read stdin, it wouldn't be a P-owned
         pipe).
      3. P writes W's pid to disk, waits until W reports the watchdog is
         installed, then exits.
      4. T polls for W's death. The watchdog must fire within ~10 s.
    """
    if sys.platform.startswith("win"):
        pytest.skip("watchdog uses os.getppid() semantics that differ on Windows")

    parent_script = tmp_path / "parent.py"
    pid_path = tmp_path / "worker.pid"
    ready_path = tmp_path / "worker.ready"

    # The 'worker' subprocess here is a busy-sleep loop with watchdog armed.
    # Critically: it does NOT call cw.main(), so the only thing that can
    # kill it is the watchdog SIGKILL'ing itself.
    worker_inline_src = textwrap.dedent(
        f"""
        import os
        import sys
        import time

        import quack._compile_worker as cw

        cw._WATCHDOG_POLL_SECS = float(os.environ["_QUACK_TEST_WATCHDOG_POLL_SECS"])
        cw._install_parent_watchdog()
        with open({str(ready_path)!r}, "w") as f:
            f.write("ready")
        # Sleep well past the watchdog poll * a safety margin. If the
        # watchdog doesn't fire, the test's outer 10s poll budget will
        # fail before this sleep returns.
        time.sleep(120)
        # If we reach here the watchdog didn't fire and the sleep finished;
        # exit with a recognizable nonzero code so the test can distinguish
        # "watchdog failed" from "watchdog SIGKILL'd" (SIGKILL gives -9).
        sys.exit(42)
        """
    )

    parent_script.write_text(
        textwrap.dedent(
            f"""
            import os, subprocess, sys

            env = os.environ.copy()
            env["QUACK_COMPILE_WORKER_PARENT_PID"] = str(os.getpid())
            env["_QUACK_TEST_WATCHDOG_POLL_SECS"] = "0.5"

            p = subprocess.Popen(
                [sys.executable, "-c", {worker_inline_src!r}],
                # DEVNULL on stdin so worker can't accidentally exit via
                # EOF read; the only exit path is the watchdog SIGKILL.
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            with open({str(pid_path)!r}, "w") as f:
                f.write(str(p.pid))

            # Wait until the worker has actually imported quack._compile_worker
            # and installed the watchdog before letting this parent exit. This
            # avoids a slow-import race on loaded CI machines, while still
            # testing the orphan path after the ready file appears.
            import time
            deadline = time.time() + 30.0
            while not os.path.exists({str(ready_path)!r}):
                rc = p.poll()
                if rc is not None:
                    raise SystemExit(f"worker exited before installing watchdog: rc={{rc}}")
                if time.time() > deadline:
                    raise SystemExit("worker did not install watchdog within 30s")
                time.sleep(0.05)
            sys.exit(0)
            """
        )
    )

    parent = subprocess.run(
        [sys.executable, str(parent_script)],
        capture_output=True,
        timeout=45,
    )
    assert parent.returncode == 0, (
        f"parent script failed: stdout={parent.stdout!r} stderr={parent.stderr!r}"
    )
    assert pid_path.exists(), "parent script did not write worker pid"
    worker_pid = int(pid_path.read_text())

    def _worker_has_stopped(pid: int) -> bool:
        if sys.platform.startswith("linux"):
            try:
                stat = open(f"/proc/{pid}/stat").read()
            except FileNotFoundError:
                return True
            except OSError:
                stat = ""
            if stat:
                # /proc/<pid>/stat field 2 is parenthesized comm (which may
                # contain spaces); field 3 after the final ')' is process state.
                state_fields = stat[stat.rfind(")") + 2 :].split()
                if state_fields and state_fields[0] == "Z":
                    return True
        try:
            os.kill(pid, 0)  # signal 0 = exists check
        except ProcessLookupError:
            return True
        return False

    # Poll for worker death. Budget: 10 s. Watchdog polls every 0.5 s
    # (the inline script sets _WATCHDOG_POLL_SECS=0.5), so we expect the
    # SIGKILL within ~0.5 s of the parent exit + a small grace period.
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if _worker_has_stopped(worker_pid):
            return  # worker is gone/zombie — watchdog fired as expected
        time.sleep(0.2)

    # Worker still running after 10 s; the watchdog didn't fire. Try to
    # clean up so we don't leak a process from the test.
    if not _worker_has_stopped(worker_pid):
        try:
            os.kill(worker_pid, 9)
        except ProcessLookupError:
            pass
    pytest.fail(
        f"worker pid {worker_pid} still alive 10s after parent exit; "
        f"watchdog did not fire (the inline worker would have exited with "
        f"rc=42 if its time.sleep(120) returned naturally — only watchdog "
        f"SIGKILL gives rc=-9)"
    )
