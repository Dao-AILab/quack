# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""Unit tests for ``quack.dsl.torch_library_op.cute_op``.

The decorator registers its backend ``fn`` as both the CUDA impl and the
fake/meta impl. Under ``FakeTensorMode`` the body runs only in the
``quack.cache.COMPILE_ONLY`` scenario (``pytest --compile-only`` / the
``_compile_worker`` subprocess); under regular ``torch.compile`` the fake
must be a no-op.

Regression: when the fake was gated on ``torch.compiler.is_compiling()``
the body ran during Dynamo's ``_get_fake_value_impl`` (because
``_is_compiling_flag`` is only set during ``torch.export``, never during
``torch.compile``). Configs whose ``_compile_*`` raises — e.g. RMSNorm
backward with N > 128k and dtype >= 32 bits — surfaced as
``TorchRuntimeError: RuntimeError when making fake tensor call`` even when
the user never invoked ``.backward()``.
"""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import quack.cache
from quack.dsl.torch_library_op import _has_symint, cute_op


# Use a unique op namespace per test module to avoid cross-test collisions
# in the global torch.library registry.
_NS = "quack_test_cute_op"


def _make_op(op_name: str, *, unsupported_n: int | None = None):
    """Define a cute_op that records every body invocation.

    If ``unsupported_n`` is given the body raises ``ValueError`` for that
    last-dim size, mirroring how ``RMSNormBackward.__init__`` rejects
    ``N > 128 * 1024`` with dtype >= 32 bits.

    Returns ``(op, call_log)`` where each ``call_log`` entry is a
    ``(kind, shape)`` tuple with ``kind`` in ``{"fake", "eager"}``.
    """
    from torch._subclasses.fake_tensor import FakeTensor

    call_log: list[tuple] = []

    @cute_op(
        f"{_NS}::{op_name}",
        mutates_args={"out"},
        schema="(Tensor x, Tensor(a1!) out) -> ()",
    )
    def _impl(x: torch.Tensor, out: torch.Tensor) -> None:
        kind = "fake" if isinstance(x, FakeTensor) else "eager"
        call_log.append((kind, tuple(x.shape)))
        if unsupported_n is not None and x.shape[-1] == unsupported_n:
            raise ValueError(f"unsupported: N={x.shape[-1]} (mirrors RMSNormBackward N>128k fp32)")
        if not isinstance(x, FakeTensor):
            out.copy_(x)

    op = getattr(getattr(torch.ops, _NS), op_name).default
    return op, call_log


@pytest.mark.compile_only_skip(
    "test precondition requires COMPILE_ONLY=False; under outer "
    "--compile-only the FakeTensorMode also blocks Dynamo from compiling the frame"
)
def test_fake_is_noop_under_torch_compile():
    """torch.compile tracing must NOT execute the cute_op body.

    Before the fix the body ran during Dynamo's ``_get_fake_value_impl``
    (because the gate ``torch.compiler.is_compiling()`` returns False
    there — ``_is_compiling_flag`` is only set during ``torch.export``,
    never during ``torch.compile``). For configs whose body raises —
    e.g. RMSNorm backward with ``N > 128k`` fp32 — this surfaced as a
    ``TorchRuntimeError`` even when the user never invoked
    ``.backward()`` (AOT autograd traces backward anyway).

    With the fix, the fake stays a no-op under ``torch.compile`` and the
    trace completes; the body only runs when the eager backend is
    dispatched at runtime.
    """
    op, call_log = _make_op("noop_under_compile", unsupported_n=262144)

    @torch.compile(fullgraph=True)
    def f(x):
        out = torch.empty_like(x)
        op(x, out)
        return out

    assert not quack.cache.COMPILE_ONLY, "test precondition: COMPILE_ONLY must be False"

    # Use a shape that would have raised the (broken) fake body. After
    # the fix the trace completes; at runtime the eager backend then
    # raises the same ValueError directly (NOT wrapped as a
    # TorchRuntimeError from Dynamo). That direct ValueError is the
    # marker that the fix is in place.
    x = torch.empty(37, 262144)
    with pytest.raises(ValueError, match="unsupported: N=262144"):
        f(x)

    # The fake must not have run — only the eager backend.
    fake_calls = [shape for kind, shape in call_log if kind == "fake"]
    assert fake_calls == [], (
        f"fake body must not run under torch.compile, saw fake calls: {fake_calls}"
    )
    eager_calls = [shape for kind, shape in call_log if kind == "eager"]
    assert eager_calls == [(37, 262144)], (
        f"eager backend must run exactly once with the input shape, saw: {eager_calls}"
    )


def test_fake_runs_body_when_compile_only():
    """Inside ``compile_only_mode()`` the fake body runs.

    This is the path the ``_compile_worker`` subprocess and
    ``pytest --compile-only`` rely on to populate the .o cache.
    """
    op, call_log = _make_op("runs_under_compile_only")

    with quack.cache.compile_only_mode():
        with FakeTensorMode():
            x = torch.empty(8, 1024)
            out = torch.empty(8, 1024)
            op(x, out)

    assert call_log == [("fake", (8, 1024))], (
        f"fake body must run when compile-only mode is active; saw {call_log}"
    )


def test_has_symint_unit():
    """Unit-level coverage for ``_has_symint``.

    The integration tests below drive SymInts through ``torch.compile``,
    but the tensor-shape branch always fires first, leaving the scalar
    ``SymInt``, nested-container, and stride branches uncovered in
    practice. Cover them here so a regression that breaks any single
    branch is caught even when no real-world caller exercises it.
    """
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    sym = shape_env.create_unbacked_symint()
    assert isinstance(sym, torch.SymInt)

    # Direct scalar SymInt — the branch codex flagged.
    assert _has_symint(sym) is True
    assert _has_symint(7) is False

    # Nested containers.
    assert _has_symint((1, 2, sym)) is True
    assert _has_symint([1, [2, [sym]]]) is True
    assert _has_symint({"a": 1, "b": {"c": sym}}) is True
    assert _has_symint((1, 2, 3)) is False

    # Concrete tensor.
    t = torch.empty(4, 8)
    assert _has_symint(t) is False

    # Mixed dict (e.g. kwargs).
    assert _has_symint({"x": t, "n": sym}) is True
    assert _has_symint({"x": t, "n": 5}) is False


@pytest.mark.compile_only_skip(
    "requires Dynamo to actually compile f() with dynamic=True to propagate "
    "SymInts through the op call; the outer FakeTensorMode that --compile-only "
    "installs causes Dynamo to skip the frame, so no SymInts are ever produced "
    "and the test's invariant cannot be exercised"
)
def test_fake_skips_symint_under_compile_only_strict():
    """SymInts in tensor shape OR scalar args must bypass the body.

    Stricter than a passive shape check: the body raises ``AssertionError``
    if a SymInt ever reaches it (in ``x.shape`` or in the scalar ``n``
    arg). A passing test therefore proves the wrapper actually skipped
    every symbolic fake invocation — both the tensor-shape path and the
    scalar-arg path codex flagged.

    PyTorch's runtime schema check rejects passing a free ``torch.SymInt``
    to an ``int`` arg from regular Python, so the only realistic attack
    vector is ``torch.compile`` tracing. Under ``dynamic=True`` the
    compile uses its own ``FakeTensorMode`` with a ``ShapeEnv``; ``n =
    x.shape[0]`` carries a ``SymInt`` through the op call. The wrapper
    must early-return; if it does not, the asserts inside the body fire
    and surface as a ``TorchRuntimeError`` graph break.
    """
    from torch._subclasses.fake_tensor import FakeTensor

    fake_calls: list[tuple] = []

    @cute_op(
        f"{_NS}::strict_symint",
        mutates_args={"out"},
        schema="(Tensor x, Tensor(a1!) out, int n) -> ()",
    )
    def _impl(x: torch.Tensor, out: torch.Tensor, n: int) -> None:
        kind = "fake" if isinstance(x, FakeTensor) else "eager"
        if kind == "fake":
            fake_calls.append((tuple(x.shape), n))
            # Contract: the wrapper must NEVER let a SymInt reach the
            # body. A regression in ``_has_symint`` trips these.
            for s in x.shape:
                assert not isinstance(s, torch.SymInt), (
                    f"SymInt leaked into fake body via x.shape={x.shape}"
                )
            assert not isinstance(n, torch.SymInt), (
                f"SymInt leaked into fake body via scalar arg n={n!r}"
            )
        else:
            out.copy_(x)

    op = getattr(getattr(torch.ops, _NS), "strict_symint").default

    with quack.cache.compile_only_mode():
        # Tie the scalar ``n`` to a dynamic dim so it propagates as a
        # SymInt on the dynamic-shape compile. This exercises BOTH the
        # tensor-shape SymInt path (via ``x``) and the scalar-arg SymInt
        # path (via ``n``) in a single op invocation — a regression in
        # ``_has_symint`` would trip the body asserts on either.
        @torch.compile(dynamic=True, fullgraph=True)
        def f(x):
            n = x.shape[0]
            out = torch.empty_like(x)
            op(x, out, n)
            return out

        f(torch.randn(8, 1024))
        f(torch.randn(16, 2048))

    # If any symbolic invocation slipped through, the body's asserts
    # would have raised and surfaced before reaching here. Records of
    # body invocations (if any) must all be concrete.
    for shape, n in fake_calls:
        assert all(isinstance(s, int) for s in shape) and isinstance(n, int), (
            f"fake body ran with SymInts: shape={shape}, n={n!r}"
        )
