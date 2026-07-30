# Copyright (c) 2026, Tri Dao.
"""Run trace-time checks under the DSL-managed MLIR context.

Tests must NOT create their own ``with ir.Context():`` to build cute IR
(identity tensors, layout algebra, ``mlir_type`` queries, ...). Building cute
IR inside a user-created raw ``ir.Context`` leaves dangling references in the
DSL's process-global state; the next sizeable ``cute.compile`` then writes
through them and corrupts the heap — glibc aborts with ``malloc(): unaligned
tcache chunk detected`` (or SIGSEGV) in whatever kernel happens to compile
next. This was the cause of the intermittent xdist worker crashes in CI
(worker dies during the cross-entropy backward compile). Upstream
nvidia-cutlass-dsl bug, reproduced with pure-cutlass code on 4.6.0.dev0.

Use :func:`run_traced` instead: the check runs at ``cute.jit`` trace time,
inside a context whose lifecycle the DSL owns.
"""

import cutlass
import cutlass.cute as cute

__all__ = ["run_traced"]


@cute.jit
def _traced_runner(fn: cutlass.Constexpr):
    fn()


def run_traced(fn) -> None:
    """Call ``fn()`` at ``cute.jit`` trace time under the DSL-managed context.

    ``fn`` is invoked as a compile-time constexpr while tracing a trivial
    host-only jit function, so a live MLIR context (with all cute dialects
    registered) is current for the duration of the call. Assertion failures
    inside ``fn`` propagate to the caller like any Python exception.
    """
    cute.compile(_traced_runner, fn)
