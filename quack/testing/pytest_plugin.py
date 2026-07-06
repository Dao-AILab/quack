# Copyright (c) 2026, Tri Dao.
"""Reusable pytest plugin for QuACK async kernel compilation.

Adds the ``--async-compile[=N]`` CLI flag: on a kernel-compile ``.o``-cache
miss, the compile is shipped to a pool of N CPU workers (forkserver sidecar,
see :mod:`quack.cache.async_compile`), the test is deferred while other tests
run, and it is retried once the ``.o`` is exported. Zero overhead when the
cache is warm. Works single-process and under xdist (both ``load`` and
``worksteal`` schedulers).

To opt in, add this line to your ``conftest.py``::

    pytest_plugins = ["quack.testing.pytest_plugin"]
"""

from __future__ import annotations

import pytest


# Saved originals so ``pytest_unconfigure`` can restore pytest internals we
# monkey-patched in ``pytest_configure``. Set to ``None`` when the
# corresponding patch was skipped (e.g. pytest internals didn't match what
# we expected, or the env-var opt-out was set).
_orig_compat_getfuncargnames = None
_orig_fixtures_getfuncargnames = None


def pytest_addoption(parser):
    parser.addoption(
        "--async-compile",
        type=int,
        nargs="?",
        const=32,
        default=None,
        metavar="N",
        help=(
            "On a kernel-compile cache miss, submit the compile to a pool of "
            "N CPU workers, defer the test, and retry it once the .o is "
            "ready. Zero overhead when the cache is warm."
        ),
    )


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
    """Set up the async compile pool and pytest-internal speedup patches."""
    jobs = config.getoption("--async-compile", default=None)
    if jobs is not None:
        import os as _os

        worker = _os.environ.get("PYTEST_XDIST_WORKER")
        is_xdist_master = worker is None and getattr(config.option, "numprocesses", None)
        if not is_xdist_master:
            # Real test-running process (single-proc pytest or xdist worker).
            from quack.cache.async_compile import activate

            n_workers = int(_os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
            pool = activate(max(2, jobs // n_workers))
            pool.prewarm()  # sidecar import overlaps collection, not the first miss
            if worker is not None:
                config.pluginmanager.register(_XdistWorkerDefer(pool), "quack-xdist-defer")
            else:
                config.pluginmanager.register(_SingleProcDeferLoop(pool), "quack-defer-loop")

    # Speed up manual_seed by short-circuiting the queued-seed path on
    # unavailable accelerators.
    _disable_unused_accelerator_lazy_call()

    # Cache inspect.signature() lookups behind _pytest.compat.getfuncargnames.
    # Pytest re-builds signatures ~20x per test during fixture resolution; on a
    # 2k-test file this is several seconds of pure CPU we can avoid.
    _install_getfuncargnames_cache()


def pytest_unconfigure(config):
    """Tear down the compile pool and undo any pytest-internal patches."""
    from quack.cache.async_compile import get_active_pool, deactivate

    pool = get_active_pool()
    if pool is not None:
        stats = pool.stats()
        defer_plugin = config.pluginmanager.get_plugin(
            "quack-defer-loop"
        ) or config.pluginmanager.get_plugin("quack-xdist-defer")
        defers = defer_plugin.defer_count if defer_plugin else 0
        print(
            f"\nasync-compile: {stats['submitted']} keys submitted, "
            f"{stats['failed']} failed, {defers} test deferrals"
        )
        for sha, err in stats["errors"][:5]:
            print(f"  pool compile error [{sha[:12]}]: {err}")
        deactivate()

    # Always undo the global monkey-patches we installed. This keeps the
    # process clean for downstream callers (e.g. notebook hosts that
    # pytest-main multiple times in the same interpreter).
    _restore_getfuncargnames_cache()


# --- CompilePending deferral hooks ------------------------------------------


def _defer_if_compile_pending(item, outcome, force_pass: bool) -> bool:
    """If the phase raised CompilePending, flag the item for deferral.

    ``force_pass=True`` (call phase): convert the outcome to a pass. The
    defer loop discards all reports of a deferred attempt anyway, and
    letting the exception stand would make pytest build a full failure
    ``longrepr`` (source-loading traceback format, ~100 ms) for every
    deferral — measured to nearly double a cold run's in-session time.

    ``force_pass=False`` (setup phase): leave the exception so the phase
    reports as failed and pytest skips the call phase — half-built fixtures
    must not run the test body. Setup-phase compiles are rare, so the
    longrepr cost is negligible there.
    """
    if outcome.excinfo is None:
        return False
    from quack.cache.async_compile import CompilePending

    if not issubclass(outcome.excinfo[0], CompilePending):
        return False
    item._quack_pending_sha = outcome.excinfo[1].sha
    if force_pass:
        outcome.force_result(None)
    return True


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    outcome = yield
    _defer_if_compile_pending(item, outcome, force_pass=False)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    _defer_if_compile_pending(item, outcome, force_pass=True)


# --- Async compile pool: defer-and-retry run loop ---------------------------
#
# With --async-compile, jit_cache misses raise CompilePending after shipping
# the key to a CPU subprocess pool. A deferred test's reports are discarded
# (no logging) and the test is retried once its pending sha completes;
# everything else runs and reports normally. Tests that keep deferring past
# _MAX_ATTEMPTS run one final attempt with the pool suppressed (in-process
# compile) so persistent failures surface as ordinary test failures.
#
# Two execution modes:
#
# * Single-process pytest: ``_SingleProcDeferLoop`` replaces the default
#   ``pytest_runtestloop`` with a deque that rotates deferred items to the
#   back.
#
# * xdist worker: xdist's ``WorkerInteractor`` owns the runtestloop, but it
#   invokes ``pytest_runtest_protocol`` as a hook, so ``_XdistWorkerDefer``
#   takes over the protocol (tryfirst + firstresult). Deferred items are
#   stashed; the master receives ``runtest_protocol_complete`` immediately
#   (its ``mark_test_complete`` is a plain order-independent ``.remove()``)
#   and keeps streaming new items to the worker -- skip-ahead for free.
#   Stashed items are opportunistically re-run between incoming items as
#   their compiles finish, and fully drained in a ``pytest_runtestloop``
#   hookwrapper after xdist's inner loop exits (channel still open, so late
#   reports flow to the master normally).

# A test needing K distinct cold kernels defers K times (keys are discovered
# serially: each attempt stops at the first missing kernel). Attempts are
# cheap — an item is only re-run once its awaited sha resolves — so the cap
# just needs to exceed the realistic kernels-per-test count; past it, the
# final attempt compiles in-process (pool suppressed) so a wedged pool can't
# defer a test forever.
_MAX_ATTEMPTS = 20

# The item that the most recent runtestprotocol call *predicted* would run
# next (its ``nextitem``). Fixture teardown is scoped to that prediction
# (``SetupState.teardown_exact``), and pytest asserts the prediction was
# right at the next setup ("previous item was not torn down properly").
# Deferral breaks the chain — a drained item can run while the fixture stack
# is scoped for a different module — so ``_run_protocol`` detects the
# misprediction and forces a full teardown first.
_LAST_PREDICTED_NEXT = [None]


def _run_protocol(item, nextitem, force_sync: bool):
    """Run one test protocol without logging. Returns (pending_sha, reports)."""
    from _pytest.runner import runtestprotocol

    from quack.cache.async_compile import suppress_pool

    if _LAST_PREDICTED_NEXT[0] is not item:
        ss = item.session._setupstate
        if ss.stack:
            ss.teardown_exact(None)
    _LAST_PREDICTED_NEXT[0] = nextitem

    item._quack_pending_sha = None
    if force_sync:
        with suppress_pool():
            reports = runtestprotocol(item, nextitem=nextitem, log=False)
    else:
        reports = runtestprotocol(item, nextitem=nextitem, log=False)
    return getattr(item, "_quack_pending_sha", None), reports


def _log_reports(item, reports):
    ihook = item.ihook
    ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    for rep in reports:
        ihook.pytest_runtest_logreport(report=rep)
    ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)


def _log_deferred_teardown_errors(item, reports, log_fn):
    """Surface real teardown failures from an otherwise-discarded attempt.

    A deferred attempt's reports are dropped (the retry produces the ones
    that count), but the attempt's teardown *did run* — an error there is a
    genuine bug (leaked fixture state) that the retry may not reproduce.
    Forward just the failed teardown report so it isn't silently hidden.
    """
    for rep in reports:
        if rep.when == "teardown" and rep.failed:
            log_fn(item, [rep])


class _SingleProcDeferLoop:
    """Custom runtestloop for non-xdist runs: defer = rotate to back of deque."""

    def __init__(self, pool):
        self.pool = pool
        self.defer_count = 0

    #: How long a deferred test may wait on its pool compile before the loop
    #: stops trusting the pool and re-runs it with the pool suppressed
    #: (in-process compile). Guards against a permanently-"pending" sha
    #: (wedged worker, foreign flock holder that never produces the .o) —
    #: without it a rotation-only item would spin forever, since attempts
    #: increment only on actual runs.
    _WEDGE_TIMEOUT_S = 600.0

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtestloop(self, session):
        if session.config.option.collectonly:
            return None
        import time
        from collections import Counter, deque

        queue = deque(session.items)
        attempts: Counter = Counter()
        deadline: dict = {}  # nodeid -> wedge deadline for the awaited sha
        spins = 0  # consecutive queue rotations without running anything
        while queue:
            item = queue.popleft()
            awaiting = getattr(item, "_quack_await_sha", None)
            wedged = awaiting is not None and time.monotonic() > deadline.get(
                item.nodeid, float("inf")
            )
            if awaiting is not None and not wedged:
                state, _ = self.pool.poll(awaiting)
                if state == "pending":
                    queue.append(item)
                    spins += 1
                    if spins >= len(queue):
                        time.sleep(0.2)  # everything is blocked on the pool
                        spins = 0
                    continue
            spins = 0
            attempts[item.nodeid] += 1
            nextitem = queue[0] if queue else None
            pending_sha, reports = _run_protocol(
                item, nextitem, force_sync=wedged or attempts[item.nodeid] > _MAX_ATTEMPTS
            )
            if pending_sha:
                item._quack_await_sha = pending_sha
                deadline.setdefault(item.nodeid, time.monotonic() + self._WEDGE_TIMEOUT_S)
                _log_deferred_teardown_errors(item, reports, _log_reports)
                queue.append(item)
                self.defer_count += 1
                continue
            item._quack_await_sha = None
            _log_reports(item, reports)
            if session.shouldstop:
                raise session.Interrupted(session.shouldstop)
            if session.shouldfail:
                raise session.Failed(session.shouldfail)
        return True


class _XdistWorkerDefer:
    """Defer-and-retry inside an xdist worker.

    The worker's ``run_one_test`` sends ``runtest_protocol_complete`` to the
    master right after our protocol hook returns -- including for deferred
    attempts. That is what we want: the master stops tracking the item and
    keeps the worker's queue full. The item is then entirely this worker's
    responsibility; we re-run it once its compile lands and forward the
    reports late (the master's report handling is order-independent).
    """

    def __init__(self, pool):
        self.pool = pool
        self.deferred = []  # list of (item, awaited_sha)
        self.attempts = {}  # nodeid -> int
        self.defer_count = 0
        self._interactor = None

    def _get_interactor(self, item):
        """Find xdist's WorkerInteractor (it registers itself as a plugin)."""
        if self._interactor is None:
            for p in item.config.pluginmanager.get_plugins():
                if type(p).__name__ == "WorkerInteractor":
                    self._interactor = p
                    break
        return self._interactor

    def _log_reports_as(self, item, reports) -> None:
        """Forward reports with the interactor's item_index pointing at *item*.

        ``WorkerInteractor.pytest_runtest_logreport`` asserts (and serializes)
        ``session.items[self.item_index].nodeid == report.nodeid``. For the
        incoming item that index is already correct, but reports for a
        drained deferred item are forwarded while a *different* item is
        current — so temporarily repoint the index.
        """
        interactor = self._get_interactor(item)
        if interactor is None:
            _log_reports(item, reports)
            return
        saved = interactor.item_index
        try:
            interactor.item_index = item.session.items.index(item)
            _log_reports(item, reports)
        finally:
            interactor.item_index = saved

    def _attempt(self, item, nextitem) -> None:
        """Run item; either log its reports or stash it as deferred."""
        n = self.attempts.get(item.nodeid, 0) + 1
        self.attempts[item.nodeid] = n
        pending_sha, reports = _run_protocol(item, nextitem, force_sync=n > _MAX_ATTEMPTS)
        if pending_sha:
            self.deferred.append((item, pending_sha))
            self.defer_count += 1
            _log_deferred_teardown_errors(item, reports, self._log_reports_as)
        else:
            self._log_reports_as(item, reports)

    def _drain_ready(self, nextitem) -> None:
        """Re-run any deferred items whose compile finished (or failed)."""
        pending = self.deferred
        self.deferred = []
        for item, sha in pending:
            state, _ = self.pool.poll(sha)
            if state == "pending":
                self.deferred.append((item, sha))
            else:
                self._attempt(item, nextitem)  # may re-append to self.deferred

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        # Retry any ready deferred items first. nextitem correctness is
        # handled centrally by _run_protocol's misprediction guard.
        self._drain_ready(nextitem=item)
        self._attempt(item, nextitem)
        return True  # protocol handled; suppress the default impl

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtestloop(self, session):
        outcome = yield  # xdist WorkerInteractor loop: all assigned items
        if outcome.excinfo is not None:
            return
        if session.shouldstop or session.shouldfail:
            # -x / --maxfail / master-initiated stop: the session is being
            # aborted; running deferred tests now would both delay shutdown
            # and report tests "after" the stop point. Drop them (the master
            # already considers them complete; the run is failing anyway).
            self.deferred.clear()
            return
        import time

        # Drain remaining deferred items, blocking on the pool. The channel
        # to the master is still open, so reports flow normally.
        deadline = time.monotonic() + 600
        while self.deferred:
            before = len(self.deferred)
            self._drain_ready(nextitem=None)
            if self.deferred and len(self.deferred) == before:
                if time.monotonic() > deadline:
                    # Pool wedged: force remaining items through in-process.
                    for item, _ in self.deferred:
                        self.attempts[item.nodeid] = _MAX_ATTEMPTS + 1
                    deadline = float("inf")
                time.sleep(0.2)


# --- Session-end report-integrity check --------------------------------------
#
# The defer machinery reports an xdist item's protocol as complete *before*
# its reports are sent (that's what keeps the master streaming new items to
# the worker). The one integrity hole this opens: a worker that crashes with
# deferred items still stashed loses them SILENTLY — the master already
# counted them complete, so nothing else will notice the missing reports.
# This check closes the hole: at session end, any collected-but-unreported,
# non-deselected test flips the exit status to failure.

_reported_nodeids: set = set()
_deselected_nodeids: set = set()


def pytest_deselected(items):
    for item in items:
        _deselected_nodeids.add(item.nodeid)


def pytest_runtest_logreport(report):
    _reported_nodeids.add(report.nodeid)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    import os

    config = session.config
    if config.getoption("--async-compile", default=None) is None:
        return  # integrity risk only exists with the defer machinery
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return  # workers see the full collection but run a subset
    if getattr(config.option, "collectonly", False):
        return
    if exitstatus not in (0, 1):
        return  # interrupted (-x, ^C, internal error): missing reports expected
    missing = {item.nodeid for item in session.items} - _reported_nodeids - _deselected_nodeids
    if not missing:
        return
    tr = config.pluginmanager.get_plugin("terminalreporter")
    lines = [
        f"async-compile INTEGRITY ERROR: {len(missing)} collected test(s) produced no "
        "report (deferred tests lost to a worker crash?):"
    ] + [f"  {nodeid}" for nodeid in sorted(missing)[:20]]
    for line in lines:
        (tr.write_line if tr else print)(line)
    session.exitstatus = 1
