# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""Unit tests for ``quack.dsl.torch_library_op.cute_op``.

The decorator registers the backend ``fn`` as the CUDA impl and a pure
no-op as the fake/meta impl: our ops only mutate their inputs, so tracing
needs no shape effect and the body must never run under ``FakeTensorMode``
(kernel compilation is owned by jit_cache + the async compile pool at real
execution time).

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
from torch.utils._python_dispatch import TorchDispatchMode

from quack.dsl.torch_library_op import cute_op


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


def test_fake_never_runs_body_under_fake_mode():
    """Under ``FakeTensorMode`` the fake must be a pure no-op.

    This is the new contract after the compile-only rip-out: the body used
    to run under ``compile_only_mode()`` to warm the .o cache; that job now
    belongs to the async compile pool (which calls the tensor-free
    ``_compile_*`` functions directly), so there is no scenario left where
    the op body may execute on fake tensors.
    """
    op, call_log = _make_op("noop_under_fake_mode")

    with FakeTensorMode():
        x = torch.empty(8, 1024)
        out = torch.empty(8, 1024)
        op(x, out)

    assert call_log == [], f"fake body must never run; saw {call_log}"


class _OpCounter(TorchDispatchMode):
    """Count how many times a specific op overload reaches the dispatcher."""

    def __init__(self, needle: str):
        self.needle = needle
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.needle in str(func):
            self.count += 1
        return func(*args, **(kwargs or {}))


def test_wrapper_exposes_underlying_op():
    """cute_op returns a dispatch wrapper that still exposes the raw body and
    the registered custom op, and the op stays reachable via torch.ops."""
    _, _ = _make_op("wrapper_introspect")  # register under torch.ops

    @cute_op(
        f"{_NS}::wrapper_attrs", mutates_args={"out"}, schema="(Tensor x, Tensor(a1!) out) -> ()"
    )
    def _impl(x: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(x)

    assert callable(_impl)
    assert hasattr(_impl, "_init_fn") and hasattr(_impl, "_custom_op")
    assert hasattr(torch.ops, _NS) and hasattr(getattr(torch.ops, _NS), "wrapper_attrs")


def test_eager_call_bypasses_custom_op_boundary():
    """In eager, calling the wrapper must run the raw body directly and skip the
    ~60us torch.library dispatch/functionalization boundary — the registered op
    overload must NOT reach the dispatcher — while still mutating correctly.
    Calling the underlying op does go through the dispatcher (control)."""
    call_log: list[tuple] = []

    @cute_op(
        f"{_NS}::eager_bypass", mutates_args={"out"}, schema="(Tensor x, Tensor(a1!) out) -> ()"
    )
    def _impl(x: torch.Tensor, out: torch.Tensor) -> None:
        call_log.append(tuple(x.shape))
        out.copy_(x)

    x = torch.randn(8, 16)

    # Wrapper in eager: body runs, but the custom op does not hit the dispatcher.
    out = torch.empty_like(x)
    with _OpCounter("eager_bypass") as c:
        _impl(x, out)
    assert torch.equal(out, x), "eager bypass must still mutate out"
    assert call_log == [(8, 16)], f"body must run exactly once, saw {call_log}"
    assert c.count == 0, f"custom-op boundary must be bypassed in eager, dispatched {c.count}x"

    # Underlying op: goes through the dispatcher (control for the counter).
    out2 = torch.empty_like(x)
    with _OpCounter("eager_bypass") as c2:
        _impl._custom_op(x, out2)
    assert torch.equal(out2, x)
    assert c2.count >= 1, "underlying custom op must reach the dispatcher"


def test_wrapper_uses_real_op_under_compile():
    """Under torch.compile the wrapper must route to the real op (captured as a
    graph node with the no-op fake), not inline the body — so the body never runs
    on fake tensors and runs exactly once eagerly at runtime."""
    from torch._subclasses.fake_tensor import FakeTensor

    call_log: list[tuple] = []

    @cute_op(
        f"{_NS}::compile_wrapper", mutates_args={"out"}, schema="(Tensor x, Tensor(a1!) out) -> ()"
    )
    def _impl(x: torch.Tensor, out: torch.Tensor) -> None:
        call_log.append(("fake" if isinstance(x, FakeTensor) else "eager", tuple(x.shape)))
        if not isinstance(x, FakeTensor):
            out.copy_(x)

    @torch.compile(fullgraph=True)
    def f(x):
        out = torch.empty_like(x)
        _impl(x, out)
        return out

    x = torch.randn(4, 8)
    out = f(x)
    assert torch.equal(out, x)
    assert [k for k, _ in call_log] == ["eager"], (
        f"fake body must not run under compile; body must run once eagerly, saw {call_log}"
    )


def test_fake_is_noop_under_dynamic_shapes():
    """Dynamic-shape torch.compile (SymInt shapes + SymInt scalar args) must
    trace through the op without running the body.

    Historical hazard: the old compile-only fake ran the body and needed a
    ``_has_symint`` guard because SymInt shapes poisoned ``@jit_cache``'s
    ``lru_cache`` keys. The pure no-op fake makes the hazard structurally
    impossible; this test pins that.
    """
    op, call_log = _make_op("noop_dynamic")

    @torch.compile(dynamic=True, fullgraph=True)
    def f(x):
        out = torch.empty_like(x)
        op(x, out)
        return out

    x = torch.empty(16, 64)
    f(x)
    fake_calls = [shape for kind, shape in call_log if kind == "fake"]
    assert fake_calls == [], f"fake body must not run under dynamic compile: {fake_calls}"
