# Copyright (c) 2026, Tri Dao.
"""Unit tests for ``quack.cache`` and its compile-only helpers.

These are deliberately *small and synchronous*: they exercise the package's
mutable-state plumbing without launching real CuTe kernels, so they catch
import-order, mutation-visibility, and exception-propagation regressions
fast.
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

# Note: ``quack.cache`` is intentionally re-imported inside each test below.
# Each test exercises a different slice of the package (legacy proxy, jit
# disk cache, FakeTensorMode, etc.) and the inside-function import makes the
# dependency explicit at the test boundary; ruff's F811 redefinition warnings
# are silenced by avoiding a module-level import we don't need at module
# scope anyway.

# ---------------------------------------------------------------------------
# Skip this whole module under ``pytest --compile-only``.
#
# These are unit tests of the compile-only plumbing itself. They enter and
# exit ``compile_only_mode()`` to exercise its semantics, so running them
# *inside* the plugin's session-wide compile-only context would nest two
# contexts and exercise nothing the nested tests below don't already cover.
# Phase 1 (cache warming) doesn't need them either — these tests don't warm
# a real CuTe kernel cache.
#
# Use ``pytest.mark.compile_only_skip`` (registered by the plugin) rather
# than ``pytest.mark.skipif(quack.cache.COMPILE_ONLY, ...)``: the marker is
# evaluated at test-setup time (post-``pytest_configure``), so it is robust
# to xdist worksteal item-fetch ordering. See ``quack.testing.pytest_plugin``
# for the marker registration.
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.compile_only_skip(
    "compile-only plumbing unit tests; not relevant to phase-1 cache warming"
)


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

    ``COMPILE_ONLY`` is intentionally NOT in this list — it's now resolved via
    module ``__getattr__`` from the ``_COMPILE_ONLY_DEPTH`` ContextVar, and
    direct writes are forbidden (see ``test_compile_only_direct_write_forbidden``).
    """
    import quack.cache

    for name in ("CACHE_ENABLED", "CACHE_DIR", "EXTRA_SOURCE_DIRS"):
        assert name in vars(quack.cache), (
            f"static-config flag {name!r} should be a direct attribute of quack.cache, not a re-export"
        )


def test_compile_only_direct_write_forbidden():
    """Direct assignment to ``quack.cache.COMPILE_ONLY`` must raise.

    This is the structural fix for the legacy leak-on-reset bug: callers used
    to do ``COMPILE_ONLY = True; ... finally: COMPILE_ONLY = False`` which
    clobbered the *outer* session value to ``False`` on exit, breaking the
    plugin's session-wide invariant for every downstream test on the same
    xdist worker. The custom module ``__setattr__`` now raises so the bug
    cannot resurface in either tests or downstream code.
    """
    import quack.cache

    with pytest.raises(AttributeError, match="read-only stack-backed flag"):
        quack.cache.COMPILE_ONLY = True
    with pytest.raises(AttributeError, match="read-only stack-backed flag"):
        quack.cache.COMPILE_ONLY = False


def test_jit_module_sees_live_flags():
    """``quack.cache.jit`` reads flags via ``_state`` (the partially-imported
    package). A regression that breaks this would surface as the disk cache
    silently ignoring ``CACHE_ENABLED=0`` or as ``COMPILE_ONLY`` mutations
    not affecting ``jit_cache``.
    """
    import quack.cache
    import quack.cache.jit as jit_module

    # `_state` should be the package object itself.
    assert jit_module._state is quack.cache


# ---------------------------------------------------------------------------
# compile_only_mode() / is_compile_only() semantics
# ---------------------------------------------------------------------------


def test_compile_only_mode_round_trip():
    """Enter, then exit, then assert the legacy proxy reflects each phase."""
    import quack.cache

    assert not quack.cache.is_compile_only()
    assert quack.cache.COMPILE_ONLY is False  # legacy proxy is live
    with quack.cache.compile_only_mode():
        assert quack.cache.is_compile_only()
        assert quack.cache.COMPILE_ONLY is True
    assert not quack.cache.is_compile_only()
    assert quack.cache.COMPILE_ONLY is False


def test_compile_only_mode_restores_on_exception():
    """ContextVar token reset must run even when the body raises."""
    import quack.cache

    assert not quack.cache.is_compile_only()
    with pytest.raises(RuntimeError, match="boom"):
        with quack.cache.compile_only_mode():
            assert quack.cache.is_compile_only()
            raise RuntimeError("boom")
    assert not quack.cache.is_compile_only()


def test_compile_only_mode_nested():
    """Each ``with`` pushes the depth counter one level deeper; the inner
    exit must not clobber the outer scope's ``True``. This is the structural
    fix for the old leak-on-reset bug — ContextVar token semantics make it
    impossible for an inner block to reset the outer depth to a value other
    than what it was when the inner entered.
    """
    import quack.cache

    assert not quack.cache.is_compile_only()
    with quack.cache.compile_only_mode():
        assert quack.cache.is_compile_only()
        with quack.cache.compile_only_mode():
            assert quack.cache.is_compile_only()
        # Outer still active after inner exit — the inner token reset only
        # popped one level, not all the way back to 0.
        assert quack.cache.is_compile_only()
    assert not quack.cache.is_compile_only()


def test_compile_only_depth_token_pops_to_prior_value():
    """Sanity check on the underlying ContextVar token semantics.

    This is the actual invariant that prevents the leak-on-reset bug: nested
    ``with`` blocks observe depths 1, 2, 1, 0 not 1, 2, 0, 0. A regression
    that switched the ContextVar to a plain assignment (no token) would fail
    this test even if ``is_compile_only`` still happened to look right by
    luck in the simpler nested test above.
    """
    import quack.cache

    observed = []
    observed.append(quack.cache._COMPILE_ONLY_DEPTH.get())
    with quack.cache.compile_only_mode():
        observed.append(quack.cache._COMPILE_ONLY_DEPTH.get())
        with quack.cache.compile_only_mode():
            observed.append(quack.cache._COMPILE_ONLY_DEPTH.get())
        observed.append(quack.cache._COMPILE_ONLY_DEPTH.get())
    observed.append(quack.cache._COMPILE_ONLY_DEPTH.get())
    assert observed == [0, 1, 2, 1, 0]


# ---------------------------------------------------------------------------
# CompileOnlyFakeTensorMode dispatch override
# ---------------------------------------------------------------------------


def test_fake_mode_intercepts_item_int():
    from quack.cache import CompileOnlyFakeTensorMode
    from quack.cache.compile_only import _INT_SENTINEL

    with CompileOnlyFakeTensorMode():
        t = torch.zeros(3, dtype=torch.int32, device="cuda")
        # Stock FakeTensorMode would raise DataDependentOutputException here.
        assert t.sum().item() == _INT_SENTINEL


def test_fake_mode_intercepts_item_float():
    from quack.cache import CompileOnlyFakeTensorMode
    from quack.cache.compile_only import _FLOAT_SENTINEL

    with CompileOnlyFakeTensorMode():
        t = torch.randn(3, dtype=torch.float32, device="cuda")
        assert t.sum().item() == _FLOAT_SENTINEL


def test_fake_mode_intercepts_tolist():
    from quack.cache import CompileOnlyFakeTensorMode
    from quack.cache.compile_only import _INT_SENTINEL

    with CompileOnlyFakeTensorMode():
        t = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
        # tolist() dispatches aten._local_scalar_dense per element.
        assert t.tolist() == [_INT_SENTINEL] * 4


def test_fake_mode_intercepts_int_cast():
    from quack.cache import CompileOnlyFakeTensorMode
    from quack.cache.compile_only import _INT_SENTINEL

    with CompileOnlyFakeTensorMode():
        t = torch.tensor(5, dtype=torch.int32, device="cuda")
        assert int(t) == _INT_SENTINEL
        # int-dtype FakeTensor → int sentinel → Python casts to float
        assert float(t) == float(_INT_SENTINEL)


def test_fake_mode_passes_through_other_ops():
    """Only ``aten._local_scalar_dense`` is intercepted; everything else
    must keep flowing through ``super().dispatch``."""
    from quack.cache import CompileOnlyFakeTensorMode

    with CompileOnlyFakeTensorMode():
        a = torch.randn(3, 4, device="cuda")
        b = a + 1
        c = a.transpose(0, 1)
        d = a @ a.T
        assert b.shape == (3, 4)
        assert c.shape == (4, 3)
        assert d.shape == (3, 3)


# ---------------------------------------------------------------------------
# jit_cache + COMPILE_ONLY interaction
# ---------------------------------------------------------------------------


def test_jit_cache_compile_only_returns_noop():
    """Under ``COMPILE_ONLY`` ``jit_cache`` must return the no-op kernel so
    downstream call sites can ``compiled_fn(...)`` safely on FakeTensors."""
    import quack.cache
    from quack.cache.jit import _noop_kernel

    calls = []

    @quack.cache.jit_cache
    def _compile_stub(key):
        # Pretend to compile by appending. The real implementation calls
        # cute.compile(); for the test we just track that the body ran.
        calls.append(key)

        # Return a "compiled" function that records its own call.
        def _launched(*a, **kw):
            calls.append(("launched", key))

        return _launched

    # Normal mode: returns the real compiled fn, which records calls.
    real_fn = _compile_stub("k1")
    real_fn("arg")
    assert calls == ["k1", ("launched", "k1")]

    # Compile-only mode: returns _noop_kernel which is a no-op.
    calls.clear()
    with quack.cache.compile_only_mode():
        noop = _compile_stub("k2")
        noop("arg")
        assert noop is _noop_kernel
        assert calls == ["k2"]  # body ran (compile happened) but no launch


# ---------------------------------------------------------------------------
# CompileOnlyStrictError plumbing
# ---------------------------------------------------------------------------


def test_strict_error_is_a_runtime_error():
    from quack.cache import CompileOnlyStrictError

    assert issubclass(CompileOnlyStrictError, RuntimeError)


def test_strict_mode_wraps_precompile_failures(monkeypatch):
    """``QUACK_COMPILE_ONLY_STRICT=1`` causes ``_precompile_default_config``
    to wrap raised exceptions in ``CompileOnlyStrictError``."""
    monkeypatch.setenv("QUACK_COMPILE_ONLY_STRICT", "1")

    import quack.cache
    from quack.cache import CompileOnlyStrictError
    from quack.gemm_interface import _precompile_default_config

    class _BadFn:
        __name__ = "bad_fn"

        @staticmethod
        def fn(*a, **kw):
            raise TypeError("simulated schema drift")

    with quack.cache.compile_only_mode():
        A = torch.randn(8, 8, device="cuda")
        with pytest.raises(CompileOnlyStrictError, match="bad_fn"):
            _precompile_default_config(_BadFn(), A)


def test_non_strict_mode_swallows_precompile_failures(monkeypatch):
    monkeypatch.delenv("QUACK_COMPILE_ONLY_STRICT", raising=False)

    import quack.cache
    from quack.gemm_interface import _precompile_default_config

    class _BadFn:
        __name__ = "bad_fn"

        @staticmethod
        def fn(*a, **kw):
            raise TypeError("would-be schema drift")

    with quack.cache.compile_only_mode():
        A = torch.randn(8, 8, device="cuda")
        # Must not raise; the swallow inside _precompile_default_config
        # is responsible for keeping non-strict runs quiet.
        _precompile_default_config(_BadFn(), A)


# ---------------------------------------------------------------------------
# Plugin _should_swallow classification
# ---------------------------------------------------------------------------


def test_plugin_should_swallow_classification():
    from quack.cache import CompileOnlyStrictError
    from quack.testing.pytest_plugin import _should_swallow

    # Skip: never swallowed (tests that meant to skip should keep skipping).
    assert not _should_swallow(pytest.skip.Exception)
    # Strict-mode failures: never swallowed (defeating the whole point).
    assert not _should_swallow(CompileOnlyStrictError)
    # Everything else: swallowed under --compile-only.
    assert _should_swallow(RuntimeError)
    assert _should_swallow(TypeError)
    assert _should_swallow(AssertionError)


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


def test_derive_tensor_split_pairs_matches_eager_signature():
    """Every ``*_tensor`` parameter on a custom_op signature maps to its
    unified twin via the auto-derived split spec.

    Regression: commits 7acaadd and 290a6a4 hand-maintained per-op rewrite
    lists that drifted from the eager signature (a forgotten ``alpha_tensor``
    merge, a ``concat_layout`` str-vs-tuple mismatch). The auto-derivation
    eliminates the drift surface entirely; this test pins that invariant so
    a future change that breaks the convention (e.g. ``alpha_t`` instead of
    ``alpha_tensor``) surfaces immediately.
    """
    import inspect
    import quack.gemm_interface as gi

    custom_ops = [
        gi.gemm_out,
        gi.gemm_add_out,
        gi.gemm_act_out,
        gi.gemm_dact_out,
        gi.gemm_gated_out,
        gi.gemm_add_inplace_op,
        gi.gemm_symmetric_out,
    ]
    for op in custom_ops:
        sig = inspect.signature(op._init_fn)
        params = list(sig.parameters)
        derived = dict(gi._derive_tensor_split_pairs(op._init_fn))
        # Every parameter ending in ``_tensor`` must (a) be derived as a
        # split, and (b) have its unified counterpart also in the signature.
        for name in params:
            if not name.endswith("_tensor"):
                continue
            unified = name[: -len("_tensor")]
            assert name in derived, f"{op._opname} schema has {name!r} but derived spec missed it"
            assert derived[name] == unified, (
                f"{op._opname} schema-split derived wrong unified name: "
                f"{derived[name]!r} != {unified!r}"
            )
            assert unified in params, (
                f"{op._opname} has split arg {name!r} but no unified "
                f"counterpart {unified!r} in signature; "
                f"register_fake would set a kwarg the eager fn rejects"
            )


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


def test_repeated_import_is_idempotent():
    """Reloading a submodule must not reset the compile-only depth counter.

    The ContextVar lives on the package object; reloading
    ``quack.cache.jit`` rebinds module-level names in jit.py but leaves
    ``quack.cache._COMPILE_ONLY_DEPTH`` (owned by ``__init__.py``)
    untouched.
    """
    import quack.cache

    with quack.cache.compile_only_mode():
        importlib.reload(quack.cache.jit)  # reload a submodule
        # The depth counter must still be > 0; reloading a submodule
        # shouldn't reset state owned by __init__.py.
        assert quack.cache.is_compile_only()
