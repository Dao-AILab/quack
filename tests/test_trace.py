"""Tests for the intra-kernel trace profiler (quack.trace).

Run with:  QUACK_TRACE=1 pytest tests/test_trace.py -x -v
"""

import json
import os

import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64

# The trace tests are the one place where tracing should be enabled by
# default: CI runs this file as part of ``pytest tests/`` without setting
# QUACK_TRACE. The unconditional ``setdefault`` is safe because the
# ``compile_only_skip`` marker (see ``pytestmark`` below) skips every test
# in this module under ``--compile-only`` anyway — the env-var is set but
# nothing in this module reads it during a compile-only run. We avoid the
# old ``if not quack.cache.COMPILE_ONLY: ...`` guard because at *import*
# time on xdist worksteal workers the depth counter is still 0 (the plugin
# pushes it in ``pytest_configure``, which can run after module import on
# worksteal), so the guard never did what its comment claimed.
os.environ.setdefault("QUACK_TRACE", "1")

from quack.trace import TraceContext, TraceSession, enabled

# QUACK_TRACE-not-set check is a static environment-variable check evaluated
# at import time, which is correct for that condition. The compile-only skip
# uses the dedicated ``compile_only_skip`` marker (registered by
# ``quack.testing.pytest_plugin``) and is evaluated at test-setup time so it
# is robust to xdist worksteal item-fetch ordering.
pytestmark = [
    pytest.mark.skipif(not enabled(), reason="QUACK_TRACE=1 not set"),
    pytest.mark.compile_only_skip("raw CuTe JIT trace kernels do not warm quack's jit_cache"),
]


# ---------------------------------------------------------------------------
# Kernel definitions (module-level so CuTe-DSL can inspect source)
# ---------------------------------------------------------------------------

# --- All-warps kernel (no sampling) ---
ALL_WARPS_BLOCK = 128
ALL_WARPS_WPB = ALL_WARPS_BLOCK // 32
ALL_WARPS_GRID = 2
ALL_WARPS_REGIONS = ("inner", "outer")


@cute.kernel
def _kernel_all_warps(trace_ptr: Int64, iters: Int32):
    ctx = TraceContext.create(trace_ptr)
    ctx.b("outer")
    for i in cutlass.range(iters):
        ctx.b("inner")
        ctx.e("inner")
    ctx.e("outer")
    ctx.flush()


@cute.jit
def _launch_all_warps(trace_ptr: Int64, iters: Int32):
    _kernel_all_warps(trace_ptr, iters).launch(
        grid=(ALL_WARPS_GRID, 1, 1),
        block=(ALL_WARPS_BLOCK, 1, 1),
    )


# --- Warp-sampled kernel ---
SAMPLED_BLOCK = 128
SAMPLED_GRID = 4
SAMPLED_WARP_IDS = (0, 3)
SAMPLED_REGIONS = ("work",)


@cute.kernel
def _kernel_sampled(trace_ptr: Int64, iters: Int32):
    ctx = TraceContext.create(trace_ptr, warp_ids=SAMPLED_WARP_IDS)
    for i in cutlass.range(iters):
        ctx.b("work")
        ctx.e("work")
    ctx.flush()


@cute.jit
def _launch_sampled(trace_ptr: Int64, iters: Int32):
    _kernel_sampled(trace_ptr, iters).launch(
        grid=(SAMPLED_GRID, 1, 1),
        block=(SAMPLED_BLOCK, 1, 1),
    )


# --- Integer-ID kernel ---
INTID_BLOCK = 32
INTID_GRID = 1


@cute.kernel
def _kernel_intid(trace_ptr: Int64, iters: Int32):
    ctx = TraceContext.create(trace_ptr)
    ctx.record_b(0)
    for i in cutlass.range(iters):
        ctx.record_m(1)
    ctx.record_e(0)
    ctx.flush()


@cute.jit
def _launch_intid(trace_ptr: Int64, iters: Int32):
    _kernel_intid(trace_ptr, iters).launch(
        grid=(INTID_GRID, 1, 1),
        block=(INTID_BLOCK, 1, 1),
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _read_trace(path):
    with open(path) as f:
        return json.load(f)


def _complete_events(trace):
    return [e for e in trace["traceEvents"] if e.get("ph") == "X"]


def _instant_events(trace):
    return [e for e in trace["traceEvents"] if e.get("ph") == "i"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAllWarps:
    """Profile all warps, named regions, context manager."""

    ITERS = 500

    def test_event_count(self, tmp_path):
        path = str(tmp_path / "trace.json")
        with TraceSession(
            path,
            grid_size=ALL_WARPS_GRID,
            block_size=ALL_WARPS_BLOCK,
        ) as sess:
            _launch_all_warps(sess.ptr, self.ITERS)

        trace = _read_trace(path)
        events = _complete_events(trace)
        total_warps = ALL_WARPS_GRID * ALL_WARPS_WPB
        # inner: ITERS per warp, outer: 1 per warp
        assert len(events) == total_warps * (self.ITERS + 1)

    def test_region_names(self, tmp_path):
        path = str(tmp_path / "trace.json")
        with TraceSession(
            path,
            grid_size=ALL_WARPS_GRID,
            block_size=ALL_WARPS_BLOCK,
        ) as sess:
            _launch_all_warps(sess.ptr, self.ITERS)

        events = _complete_events(_read_trace(path))
        names = {e["name"] for e in events}
        assert names == {"inner", "outer"}

    def test_all_warps_present(self, tmp_path):
        path = str(tmp_path / "trace.json")
        with TraceSession(
            path,
            grid_size=ALL_WARPS_GRID,
            block_size=ALL_WARPS_BLOCK,
        ) as sess:
            _launch_all_warps(sess.ptr, self.ITERS)

        events = _complete_events(_read_trace(path))
        warps = {e["args"]["warp"] for e in events}
        assert warps == set(range(ALL_WARPS_WPB))

    def test_all_blocks_present(self, tmp_path):
        path = str(tmp_path / "trace.json")
        with TraceSession(
            path,
            grid_size=ALL_WARPS_GRID,
            block_size=ALL_WARPS_BLOCK,
        ) as sess:
            _launch_all_warps(sess.ptr, self.ITERS)

        events = _complete_events(_read_trace(path))
        blocks = {e["args"]["block"] for e in events}
        assert blocks == set(range(ALL_WARPS_GRID))

    def test_durations_positive(self, tmp_path):
        path = str(tmp_path / "trace.json")
        with TraceSession(
            path,
            grid_size=ALL_WARPS_GRID,
            block_size=ALL_WARPS_BLOCK,
        ) as sess:
            _launch_all_warps(sess.ptr, self.ITERS)

        events = _complete_events(_read_trace(path))
        outer = [e for e in events if e["name"] == "outer"]
        assert all(e["dur"] > 0 for e in outer)

    def test_summary_json(self, tmp_path):
        path = str(tmp_path / "trace.json")
        with TraceSession(
            path,
            grid_size=ALL_WARPS_GRID,
            block_size=ALL_WARPS_BLOCK,
        ) as sess:
            _launch_all_warps(sess.ptr, self.ITERS)

        summary_path = str(tmp_path / "trace_summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        region_names = {r["name"] for r in summary["regions"]}
        assert region_names == {"inner", "outer"}
        for r in summary["regions"]:
            assert r["count"] > 0
            assert r["mean_dur"] >= 0


class TestNoResetNeeded:
    """Running the kernel twice on the same session without reset should work."""

    ITERS = 500  # must match TestAllWarps.ITERS to reuse cached kernel

    def test_second_run_overwrites(self, tmp_path):
        """Second run produces a valid trace without calling reset()."""
        sess = TraceSession(
            grid_size=ALL_WARPS_GRID,
            block_size=ALL_WARPS_BLOCK,
        )

        # First run
        _launch_all_warps(sess.ptr, self.ITERS)
        torch.cuda.synchronize()

        # Second run — NO reset()
        _launch_all_warps(sess.ptr, self.ITERS)
        torch.cuda.synchronize()

        path = str(tmp_path / "trace.json")
        sess.write_trace(path)

        events = _complete_events(_read_trace(path))
        total_warps = ALL_WARPS_GRID * ALL_WARPS_WPB
        # Should have exactly the second run's events (create + flush overwrite metadata)
        assert len(events) == total_warps * (self.ITERS + 1)

    def test_second_run_correct_names(self, tmp_path):
        """Region names are correct on second run without reset."""
        sess = TraceSession(
            grid_size=ALL_WARPS_GRID,
            block_size=ALL_WARPS_BLOCK,
        )

        _launch_all_warps(sess.ptr, self.ITERS)
        torch.cuda.synchronize()
        _launch_all_warps(sess.ptr, self.ITERS)
        torch.cuda.synchronize()

        path = str(tmp_path / "trace.json")
        sess.write_trace(path)

        events = _complete_events(_read_trace(path))
        names = {e["name"] for e in events}
        assert names == {"inner", "outer"}


class TestWarpSampling:
    """Only selected warps should appear in the trace."""

    ITERS = 200

    def test_only_sampled_warps(self, tmp_path):
        path = str(tmp_path / "trace.json")
        with TraceSession(
            path,
            grid_size=SAMPLED_GRID,
            block_size=SAMPLED_BLOCK,
            warp_ids=list(SAMPLED_WARP_IDS),
        ) as sess:
            _launch_sampled(sess.ptr, self.ITERS)

        events = _complete_events(_read_trace(path))
        warps = {e["args"]["warp"] for e in events}
        assert warps == set(SAMPLED_WARP_IDS)

    def test_sampled_event_count(self, tmp_path):
        path = str(tmp_path / "trace.json")
        with TraceSession(
            path,
            grid_size=SAMPLED_GRID,
            block_size=SAMPLED_BLOCK,
            warp_ids=list(SAMPLED_WARP_IDS),
        ) as sess:
            _launch_sampled(sess.ptr, self.ITERS)

        events = _complete_events(_read_trace(path))
        active_warps = len(SAMPLED_WARP_IDS) * SAMPLED_GRID
        assert len(events) == active_warps * self.ITERS

    def test_all_blocks_present(self, tmp_path):
        path = str(tmp_path / "trace.json")
        with TraceSession(
            path,
            grid_size=SAMPLED_GRID,
            block_size=SAMPLED_BLOCK,
            warp_ids=list(SAMPLED_WARP_IDS),
        ) as sess:
            _launch_sampled(sess.ptr, self.ITERS)

        events = _complete_events(_read_trace(path))
        blocks = {e["args"]["block"] for e in events}
        assert blocks == set(range(SAMPLED_GRID))


class TestIntegerIDs:
    """Test integer region IDs and mark events."""

    ITERS = 100

    def test_mark_events(self, tmp_path):
        path = str(tmp_path / "trace.json")
        with TraceSession(path, grid_size=INTID_GRID, block_size=INTID_BLOCK) as sess:
            _launch_intid(sess.ptr, self.ITERS)

        trace = _read_trace(path)
        complete = _complete_events(trace)
        instants = _instant_events(trace)
        # 1 complete event (region 0 begin/end), ITERS instant marks (region 1)
        assert len(complete) == 1
        assert len(instants) == self.ITERS

    def test_numeric_region_names(self, tmp_path):
        """Integer IDs with no named regions show as string numbers."""
        path = str(tmp_path / "trace.json")
        with TraceSession(path, grid_size=INTID_GRID, block_size=INTID_BLOCK) as sess:
            _launch_intid(sess.ptr, self.ITERS)

        trace = _read_trace(path)
        complete = _complete_events(trace)
        assert complete[0]["name"] == "0"
        instants = _instant_events(trace)
        assert all(e["name"] == "1" for e in instants)


class TestDisabled:
    """When QUACK_TRACE != 1, session should be a no-op."""

    def test_no_buffer_allocated(self):
        """This test runs even without QUACK_TRACE=1 because it tests the disabled path."""
        import importlib
        import quack.trace as trace_mod

        orig = os.environ.get("QUACK_TRACE")
        try:
            os.environ["QUACK_TRACE"] = "0"
            importlib.reload(trace_mod)
            sess = trace_mod.TraceSession(per_warp_cap=1024, grid_size=1, block_size=32)
            assert sess.d_buf is None
            assert sess.ptr is None
        finally:
            if orig is not None:
                os.environ["QUACK_TRACE"] = orig
            else:
                os.environ.pop("QUACK_TRACE", None)
            importlib.reload(trace_mod)
