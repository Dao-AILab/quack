# Copyright (c) 2026, Tri Dao.
"""Unit tests for ``quack.cache``.

These are deliberately *small and synchronous*: they exercise the package's
mutable-state plumbing without launching real CuTe kernels, so they catch
import-order, mutation-visibility, and exception-propagation regressions
fast.
"""

from __future__ import annotations

import os


# Note: ``quack.cache`` is intentionally re-imported inside each test below;
# the inside-function import makes the dependency explicit at the test
# boundary.


# ---------------------------------------------------------------------------
# Package layout / public-API surface
# ---------------------------------------------------------------------------


def test_public_api_symbols_resolve():
    """Every name advertised in ``quack.cache.__all__`` must actually exist
    on the package and be importable as ``from quack.cache import X``.

    This is the regression test for the brittle import order in
    ``__init__.py``: if a maintainer reorders flag defs vs. submodule
    imports, the package will fail to initialize and this test fires.
    """
    import quack.cache

    for name in quack.cache.__all__:
        assert hasattr(quack.cache, name), (
            f"quack.cache advertises {name!r} in __all__ but it's missing"
        )


def test_static_config_flags_live_on_package_init():
    """Static-config flags must be direct attributes of the ``quack.cache``
    package object so that callers can set them once (before importing
    submodules that read them).

    """
    import quack.cache

    for name in ("CACHE_ENABLED", "CACHE_DIR", "EXTRA_SOURCE_DIRS"):
        assert name in vars(quack.cache), (
            f"static-config flag {name!r} should be a direct attribute of quack.cache, not a re-export"
        )


def test_jit_module_sees_live_flags():
    """``quack.cache.jit`` reads flags via ``_state`` (the partially-imported
    package). A regression that breaks this would surface as the disk cache
    silently ignoring ``CACHE_ENABLED=0``.
    """
    import quack.cache
    import quack.cache.jit as jit_module

    # `_state` should be the package object itself.
    assert jit_module._state is quack.cache


# ---------------------------------------------------------------------------
# Regression: lock-before-compile prevents the cold-cache convoy
# ---------------------------------------------------------------------------


def test_jit_cache_lock_serializes_redundant_compiles(tmp_path):
    """N concurrent processes hitting the same cold key compile exactly once.

    The pre-fix behavior of ``jit_cache.wrapper`` was: check disk under a
    shared lock; on miss, release the shared lock, run ``fn(*args, **kwargs)``
    *unlocked*, then take an exclusive lock to export. Under N-way racing
    that meant **N processes compiled the same key in parallel** — wall
    time was bounded by one compile, but compile-CPU pressure scaled with
    N, starving other compiles when many keys were cold at once.

    Post-fix: the compile body runs under the exclusive lock, with a
    re-check inside the lock. N-1 of the racing processes wait on the
    lock, then see the ``.o`` and load it.

    We verify by spawning N subprocesses that all call the same
    ``@jit_cache``-decorated stub. Each invocation of the compile body
    writes a marker file; we count markers. Without the fix: ~N markers.
    With the fix: 1.
    """
    import subprocess
    import sys
    import textwrap

    cache_dir = tmp_path / "cache"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()

    # Inline child script. We monkey-patch ``cute.runtime.load_module`` before
    # importing ``quack.cache`` so the load path returns a harmless stub
    # (the ``.o`` written by export is a placeholder, not a real CuTe object).
    # The actual thing under test is purely the lock + recheck logic in
    # ``jit_cache.wrapper``; no GPU or real CuTe involvement is needed.
    # Barrier file: subprocess imports are slow (~1-3 s for cute.compile),
    # so we can't rely on launching them to make them race. Each child does
    # all its imports / monkeypatching first, then spins on a barrier file
    # until the parent writes it. That guarantees all N children enter
    # ``_compile_stub(...)`` within a tight window.
    barrier = tmp_path / "GO"
    child_script = tmp_path / "child.py"
    child_script.write_text(
        textwrap.dedent(
            f"""
            import os, sys, time
            from pathlib import Path

            # Stub cute.runtime.load_module BEFORE importing quack.cache so
            # downstream readers (post-compile load path) don't choke on the
            # placeholder .o we'll write.
            import cutlass.cute as _cute
            class _StubMod:
                def __getitem__(self, name):
                    return lambda *a, **k: None
            _cute.runtime.load_module = lambda path, enable_tvm_ffi=True: _StubMod()

            import quack.cache

            MARKER_DIR = Path({str(marker_dir)!r})
            BARRIER = Path({str(barrier)!r})

            @quack.cache.jit_cache
            def _compile_stub(key):
                # Atomically record that THIS process ran the compile body.
                # Filename includes pid + ns clock so concurrent writes never
                # collide on the same marker name.
                marker = MARKER_DIR / f"compile_{{os.getpid()}}_{{time.time_ns()}}.marker"
                marker.touch()
                # Hold long enough that the *next* worker, if it slipped
                # through the unlocked-compile bug, would also be inside
                # this body simultaneously. We test with 1.5 s headroom;
                # post-fix the exclusive lock serializes us, so subsequent
                # workers wait, see the .o, and load without re-entering.
                time.sleep(1.5)

                class _Compiled:
                    def export_to_c(self, object_file_path, function_name):
                        Path(object_file_path).write_bytes(b"placeholder-o")

                    def __call__(self, *a, **k):
                        pass

                return _Compiled()

            # Spin on the barrier so all N children enter the wrapper
            # within a few ms of each other. Bound the wait so a stuck
            # parent doesn't hang the test.
            deadline = time.time() + 60
            while not BARRIER.exists():
                if time.time() > deadline:
                    sys.exit("barrier timeout")
                time.sleep(0.01)

            _compile_stub("shared-cold-key")
            """
        )
    )

    env = {
        **os.environ,
        "QUACK_CACHE_DIR": str(cache_dir),
        "QUACK_CACHE_ENABLED": "1",
    }

    N = 8
    procs = [
        subprocess.Popen(
            [sys.executable, str(child_script)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(N)
    ]
    # Give all N children time to finish their imports and reach the barrier
    # spin-loop. A few seconds is plenty even for the slow cute imports;
    # children that haven't reached the barrier by the time we flip it will
    # just enter slightly later, which still tests the race we care about.
    import time as _time

    _time.sleep(5.0)
    barrier.touch()

    # Wait for all; collect output for diagnostics if anything failed.
    outputs = [(p.wait(timeout=120), p.stdout.read(), p.stderr.read()) for p in procs]
    for rc, out, err in outputs:
        assert rc == 0, (
            f"child subprocess exited non-zero ({rc})\n"
            f"stdout:\n{out.decode(errors='replace')}\n"
            f"stderr:\n{err.decode(errors='replace')}"
        )

    markers = list(marker_dir.glob("compile_*.marker"))
    assert len(markers) == 1, (
        f"expected exactly 1 compile invocation across {N} racing processes "
        f"(lock-before-compile should serialize), saw {len(markers)}: {markers!r}"
    )


# ---------------------------------------------------------------------------
# R4: register_fake schema-split is auto-derived from the eager signature
# ---------------------------------------------------------------------------


def test_parse_concat_layout_roundtrip():
    """``_parse_concat_layout`` is the shared str↔tuple helper used by both
    eager bodies and the fake registration. A regression that breaks one
    side without the other reintroduces the 7acaadd compile-key drift.
    """
    from quack.gemm_interface import _parse_concat_layout

    assert _parse_concat_layout(None) is None
    assert _parse_concat_layout("") is None  # empty string → None
    assert _parse_concat_layout("a") == ("a",)
    assert _parse_concat_layout("a,b") == ("a", "b")
    # Idempotency: pre-converted tuples pass through unchanged so callers can
    # chain through the helper without checking the input type first.
    assert _parse_concat_layout(("a", "b")) == ("a", "b")
    assert _parse_concat_layout(()) == ()


def test_merge_tensor_helper():
    """Shared ``Union[scalar, Tensor]`` schema-split merge."""
    from quack.gemm_interface import _merge_tensor

    assert _merge_tensor(1.0, None) == 1.0
    assert _merge_tensor(1.0, "T") == "T"  # non-None tensor_value wins
    assert _merge_tensor(None, "T") == "T"
    assert _merge_tensor(None, None) is None  # both None → stay None


# ---------------------------------------------------------------------------
# Smoke: module imports / re-imports stay consistent
# ---------------------------------------------------------------------------
