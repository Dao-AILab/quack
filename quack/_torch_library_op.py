# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""Backend-neutral opaque ``torch.library.custom_op`` helper.

The decorated operation mutates caller-allocated tensors. Its fake
implementation is therefore a no-op, which prevents Dynamo and AOTAutograd
from tracing into backend compilers or launchers. Eager calls bypass the
dispatcher to avoid its fixed overhead; compiled calls retain the opaque op.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Iterable, Optional, Union

import torch

__all__ = ["torch_library_op"]


def torch_library_op(
    name: str,
    *,
    mutates_args: Union[str, Iterable[str]],
    schema: Optional[str] = None,
    device_types: Optional[Union[str, Iterable[str]]] = None,
) -> Callable:
    """Register a mutating backend operation with an opaque fake implementation."""

    def dec(fn: Callable) -> Any:
        kwargs: dict[str, Any] = {"mutates_args": mutates_args}
        if schema is not None:
            kwargs["schema"] = schema
        if device_types is not None:
            kwargs["device_types"] = device_types
        op = torch.library.custom_op(name, fn, **kwargs)

        @op.register_fake
        def _fake(*args, **kwargs):
            return None

        return _make_eager_bypass(op, fn)

    return dec


def _make_eager_bypass(op, fn):
    """Return a Dynamo-inlineable eager bypass for ``op``.

    A callable object works for direct compiled calls, but Dynamo cannot wrap
    that object when it is referenced from inside ``autograd.Function``. A
    plain function is inlineable in both cases, so Dynamo can constant-fold
    ``is_compiling()`` and retain the custom-op node.
    """

    op_overload = op._opoverload

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if torch.compiler.is_compiling():
            # Dynamo natively understands OpOverload, including when this
            # wrapper is inlined inside autograd.Function. Capturing the
            # higher-level CustomOpDef here triggers SourcelessBuilder errors.
            return op_overload(*args, **kwargs)
        return fn(*args, **kwargs)

    wrapper._custom_op = op
    wrapper._init_fn = fn
    # Preserve post-registration hooks used by existing CuTe operators
    # (register_effect/register_autograd/_opoverload) and introspection tests.
    for attribute_name in dir(op):
        if attribute_name.startswith("__") or attribute_name in wrapper.__dict__:
            continue
        try:
            setattr(wrapper, attribute_name, getattr(op, attribute_name))
        except (AttributeError, TypeError):
            pass
    return wrapper
