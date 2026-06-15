# Copyright (c) 2026, Tri Dao.
"""Reusable pytest plugin for QuACK ``--compile-only`` cache warming.

Adds the ``--compile-only`` CLI flag, sets up a session-scoped
:class:`~quack.testing.CompileOnlyFakeTensorMode`, and swallows
test/setup/teardown errors under ``--compile-only`` so the run still finishes
even when individual tests stumble on FakeTensor-incompatible APIs that aren't
on the critical path to a kernel dispatch.

To opt in, add this line to your ``conftest.py``::

    pytest_plugins = ["quack.testing.pytest_plugin"]

After that, ``pytest --compile-only`` populates the persistent ``.o`` cache
without launching kernels (no GPU memory, parallelizable across many CPU
workers). A subsequent normal pytest run hits the disk cache for every kernel
signature warmed by ``--compile-only``.

See :mod:`quack.cache.compile_only` for the underlying mechanism.
"""

from __future__ import annotations

import pytest

from quack.cache import CompileOnlyFakeTensorMode, CompileOnlyStrictError


_fake_mode: CompileOnlyFakeTensorMode | None = None
# Token from ``_COMPILE_ONLY_DEPTH.set(...)`` so ``pytest_unconfigure`` pops
# back to the exact prior depth. ContextVar token semantics guarantee leak-free
# restoration even on exception, replacing the old save/restore-a-bool pattern
# that could clobber an outer caller's value to False.
_compile_only_token = None

# Saved originals so ``pytest_unconfigure`` can restore pytest internals we
# monkey-patched in ``pytest_configure``. Set to ``None`` when the
# corresponding patch was skipped (e.g. pytest internals didn't match what
# we expected, or the env-var opt-out was set).
_orig_compat_getfuncargnames = None
_orig_fixtures_getfuncargnames = None


def _should_swallow(exc_type) -> bool:
    """Should a ``--compile-only`` runtime error be force-passed?

    No for:
    * :class:`pytest.skip.Exception` — explicit skips must keep skipping.
    * :class:`quack.cache.CompileOnlyStrictError` — strict-mode precompile
      failures must surface as test failures, otherwise
      ``QUACK_COMPILE_ONLY_STRICT=1`` would be silently defeated by the
      blanket swallow.
    Yes for everything else: by the time we're under ``--compile-only``,
    the only thing that matters is that the kernel dispatched; downstream
    FakeTensor-incompatible APIs (``.numpy()``, ``assert_close``, etc.)
    are expected to fail and don't represent a regression.
    """
    if issubclass(exc_type, pytest.skip.Exception):
        return False
    if issubclass(exc_type, CompileOnlyStrictError):
        return False
    return True


def pytest_addoption(parser):
    parser.addoption(
        "--compile-only",
        action="store_true",
        default=False,
        help=(
            "Compile all kernels and export the .o cache, skip actual kernel "
            "execution. Uses FakeTensorMode (no GPU memory) so you can run "
            "many xdist workers in parallel. A subsequent normal pytest run "
            "hits the disk cache for every kernel signature warmed here."
        ),
    )


def _is_compile_only(config) -> bool:
    try:
        return bool(config.getoption("--compile-only", default=False))
    except (ValueError, AttributeError):
        return False


def _install_getfuncargnames_cache() -> None:
    """Cache ``_pytest.compat.getfuncargnames`` by stable function identity.

    Performance background
    ----------------------
    Pytest's fixture resolution calls ``getfuncargnames`` ~20 times per test
    (once per active fixture / parametrize axis). Each call runs
    ``inspect.signature(function)`` from scratch — a deep AST/Signature build
    with no caching upstream. For a 2k-test file this is ~40k
    ``inspect.signature`` invocations and ~1 s of pure CPU per worker. See
    pytest-dev/pytest#11284 (open since 2023) for the underlying root cause:
    pytest probes the entire fixture closure for every test instead of just
    the test's ``initialnames``.

    Caching by ``id(function)`` does NOT work: the ``request`` fixture is
    rebuilt per test, producing thousands of distinct function objects with
    identical signatures. Cache by ``(qualname, co_filename, co_firstlineno)``
    instead — that triple is stable across the per-test wrappers and uniquely
    identifies a Python function definition.

    Deliberate tradeoffs
    --------------------
    This is a monkey-patch of pytest's private API (``_pytest.compat`` and
    ``_pytest.fixtures``). We accept this because (a) the upstream fix
    (#11284) has been stuck in deprecation cycles for years, and (b) the
    speedup is ~1 s wall per file for parametrize-heavy suites. To minimize
    risk we:

    * Validate that ``getfuncargnames`` has the expected ``(function, *, name,
      cls)`` signature before patching, and skip silently with a warning if
      pytest's internals have changed.
    * Stash the originals so ``pytest_unconfigure`` restores them.
    * Honor ``QUACK_PYTEST_NO_GETFUNCARGNAMES_CACHE=1`` to opt out at runtime.

    Subtlety: ``_pytest.fixtures`` does ``from .compat import getfuncargnames``
    at import time, so we must rebind the name on both modules.
    """
    global _orig_compat_getfuncargnames, _orig_fixtures_getfuncargnames

    import os
    import warnings

    if os.environ.get("QUACK_PYTEST_NO_GETFUNCARGNAMES_CACHE"):
        return

    try:
        import _pytest.compat as _compat
        import _pytest.fixtures as _fixtures
    except ImportError:
        return  # pytest internals not where we expect them

    orig = getattr(_compat, "getfuncargnames", None)
    if orig is None or getattr(_fixtures, "getfuncargnames", None) is not orig:
        warnings.warn(
            "quack.testing.pytest_plugin: skipping getfuncargnames cache; "
            "pytest internals (_pytest.compat / _pytest.fixtures) do not "
            "match expected shape. Tests still run, just without the "
            "~1 s/file fixture-resolution speedup.",
            stacklevel=2,
        )
        return

    # Verify the signature is still `(function, *, name, cls)`. If pytest bumps
    # the API we'd rather skip than silently miscache.
    import inspect

    try:
        sig = inspect.signature(orig)
        params = sig.parameters
        expected = ("function", "name", "cls")
        if not all(p in params for p in expected):
            raise ValueError(f"unexpected params {tuple(params)!r}")
    except (TypeError, ValueError) as e:
        warnings.warn(
            f"quack.testing.pytest_plugin: skipping getfuncargnames cache; "
            f"signature check failed ({e!r}). Tests still run normally.",
            stacklevel=2,
        )
        return

    cache: dict = {}

    def _identity_key(function):
        code = getattr(function, "__code__", None)
        if code is None:
            return ("__obj__", function)
        return (
            getattr(function, "__qualname__", None) or function.__name__,
            code.co_filename,
            code.co_firstlineno,
        )

    def _patched(function, *, name="", cls=None):
        try:
            key = (_identity_key(function), name, cls)
        except (AttributeError, TypeError):
            return orig(function, name=name, cls=cls)
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = orig(function, name=name, cls=cls)
        cache[key] = result
        return result

    _orig_compat_getfuncargnames = orig
    _orig_fixtures_getfuncargnames = _fixtures.getfuncargnames  # == orig
    _compat.getfuncargnames = _patched
    # Captured via `from .compat import getfuncargnames` at import time.
    _fixtures.getfuncargnames = _patched


def _restore_getfuncargnames_cache() -> None:
    """Undo ``_install_getfuncargnames_cache``. No-op if not installed."""
    global _orig_compat_getfuncargnames, _orig_fixtures_getfuncargnames
    if _orig_compat_getfuncargnames is None:
        return
    try:
        import _pytest.compat as _compat
        import _pytest.fixtures as _fixtures
    except ImportError:
        return
    _compat.getfuncargnames = _orig_compat_getfuncargnames
    _fixtures.getfuncargnames = _orig_fixtures_getfuncargnames
    _orig_compat_getfuncargnames = None
    _orig_fixtures_getfuncargnames = None


def _disable_unused_accelerator_lazy_call() -> None:
    """No-op ``_lazy_call`` for accelerators that aren't available.

    Every ``torch.manual_seed(seed)`` fans out across CUDA, MPS, XPU, MTIA, and
    any custom device. For each *uninitialized* backend, ``_lazy_call`` takes
    a slow path that calls ``traceback.format_stack()`` to record where the
    seed was queued from. Under pytest the call stack is deep, so each
    ``format_stack`` costs ~1 ms; across thousands of tests this is several
    seconds of pure CPU overhead per worker, even though no XPU/MTIA work is
    ever submitted.

    Subtlety: ``torch.xpu/random.py`` and ``torch.cuda/random.py`` do
    ``from . import _lazy_call`` at import time, so we must replace the name
    on the submodule too — patching only the package attribute is not enough.
    """
    import torch

    nop = lambda callable, **kwargs: None  # noqa: E731
    if not torch.xpu.is_available():
        torch.xpu._lazy_call = nop
        torch.xpu.random._lazy_call = nop  # captured via `from . import _lazy_call`
    if not torch.mtia.is_available():
        torch.mtia._lazy_call = nop


def pytest_configure(config):
    """Enter the compile-only context for the duration of the test session."""
    global _fake_mode, _compile_only_token

    # Register markers regardless of --compile-only, so a test file using
    # ``pytestmark = pytest.mark.compile_only_skip("...")`` doesn't get a
    # ``PytestUnknownMarkWarning`` even when the user runs without the flag.
    config.addinivalue_line(
        "markers",
        "compile_only_skip(reason): skip this test when --compile-only is active. "
        "The skip is evaluated at test-setup time (not collection or import "
        "time), so it is xdist-worksteal-safe.",
    )
    config.addinivalue_line(
        "markers",
        "compile_only_only(reason): skip this test when --compile-only is NOT "
        "active. Use for tests that exercise the compile-only path itself.",
    )

    # Speed up manual_seed in real (non-compile-only) runs by short-circuiting
    # the queued-seed path on unavailable accelerators. Harmless under
    # --compile-only too (FakeTensorMode swallows the seed call anyway).
    _disable_unused_accelerator_lazy_call()

    # Cache inspect.signature() lookups behind _pytest.compat.getfuncargnames.
    # Pytest re-builds signatures ~20x per test during fixture resolution; on a
    # 2k-test file this is several seconds of pure CPU we can avoid.
    _install_getfuncargnames_cache()

    if not _is_compile_only(config):
        return

    import torch

    import quack.cache

    _compile_only_token = quack.cache._COMPILE_ONLY_DEPTH.set(
        quack.cache._COMPILE_ONLY_DEPTH.get() + 1
    )
    if torch.cuda.is_available():
        torch.cuda.init()
    _fake_mode = CompileOnlyFakeTensorMode()
    _fake_mode.__enter__()


def pytest_unconfigure(config):
    """Exit the compile-only context and undo any pytest-internal patches."""
    global _fake_mode, _compile_only_token

    # Always undo the global monkey-patches we installed, even if the
    # --compile-only branch below early-returns. This keeps the process clean
    # for downstream callers (e.g. notebook hosts that pytest-main multiple
    # times in the same interpreter).
    _restore_getfuncargnames_cache()

    if _fake_mode is None:
        return
    _fake_mode.__exit__(None, None, None)
    _fake_mode = None
    import quack.cache

    if _compile_only_token is not None:
        quack.cache._COMPILE_ONLY_DEPTH.reset(_compile_only_token)
        _compile_only_token = None


# --- compile_only_skip / compile_only_only marker evaluation ---------------
#
# Marker evaluation runs in ``pytest_runtest_setup`` at test-setup time, which
# is unambiguously *after* the plugin's ``pytest_configure`` has pushed the
# compile-only depth counter. This is the structural fix for the long-standing
# xdist worksteal gotcha that affected ``pytest.mark.skipif(quack.cache.COMPILE_ONLY, ...)``:
# ``skipif`` evaluates its condition at decorator-application = module-import
# time, and on worksteal workers the test module can import *before* the
# plugin's configure runs, so the captured value was ``False`` and the skip
# never fired. The marker below has no captured argument — the plugin reads
# the live ContextVar at setup time — so the race is gone.
#
# ``tryfirst=True`` is important: it runs before the hookwrapper below that
# swallows exceptions under --compile-only. ``pytest.skip.Exception`` is in
# ``_should_swallow``'s explicit don't-swallow list, so even if the order
# inverted the skip would still propagate; ``tryfirst`` is belt-and-suspenders
# and also keeps the swallow path from running for legitimate skips.


@pytest.hookimpl(tryfirst=True)
def _apply_compile_only_markers(item):
    """Evaluate ``compile_only_skip`` / ``compile_only_only`` at setup time."""
    compile_only = _is_compile_only(item.config)
    for marker in item.iter_markers(name="compile_only_skip"):
        if compile_only:
            reason = (
                marker.kwargs.get("reason")
                or (marker.args[0] if marker.args else None)
                or "compile_only_skip: skipped under --compile-only"
            )
            pytest.skip(reason)
    for marker in item.iter_markers(name="compile_only_only"):
        if not compile_only:
            reason = (
                marker.kwargs.get("reason")
                or (marker.args[0] if marker.args else None)
                or "compile_only_only: requires --compile-only"
            )
            pytest.skip(reason)


# --- Error swallowing under --compile-only --------------------------------
#
# Compile-only runs only care that kernels reach their compile path. Test
# bodies will hit FakeTensor-incompatible APIs further down (numerical
# comparisons via assert_close, .cpu()/.numpy() round-trips, etc.) that we
# don't care about for the cache-warming goal. Swallow those errors so the
# run completes; the real (non-compile-only) pass exercises the assertions.
#
# pytest.skip.Exception is *not* swallowed: tests that genuinely want to skip
# under --compile-only should keep skipping.


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    # Evaluate compile_only markers first. If they call ``pytest.skip(...)``
    # the raised exception propagates through ``yield`` below; ``_should_swallow``
    # explicitly does not swallow ``pytest.skip.Exception``, so the skip wins.
    _apply_compile_only_markers(item)
    if not _is_compile_only(item.config):
        yield
        return
    outcome = yield
    if outcome.excinfo is not None and _should_swallow(outcome.excinfo[0]):
        outcome.force_result(None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if not _is_compile_only(item.config):
        yield
        return
    outcome = yield
    if outcome.excinfo is not None and _should_swallow(outcome.excinfo[0]):
        outcome.force_result(None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    if not _is_compile_only(item.config):
        yield
        return
    outcome = yield
    if outcome.excinfo is not None and _should_swallow(outcome.excinfo[0]):
        outcome.force_result(None)
