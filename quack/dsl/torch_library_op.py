# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""``cute_op``: ``torch.library.custom_op`` for CuTe DSL kernels.

Same trick as ``torch.library.triton_op`` (register the impl as the fake/meta
kernel too), specialized for our setup:

* Under ``torch.compile`` we stay a complete no-op. Dynamo / AOT autograd
  only need to know the op's shape effect, and our ops only mutate inputs,
  so the fake has nothing to compute. Running the body here would also
  pay compile latency at dynamo trace time and (more importantly) crash
  for configs whose ``_compile_*`` constructors raise on unsupported
  shape/dtype combinations.
* Under ``FakeTensorMode`` with SymInt shapes (dynamic-shape tracing), skip:
  ``@jit_cache`` is an ``lru_cache`` and SymInts are unhashable.
* In the ``COMPILE_ONLY`` scenario (``pytest --compile-only`` or the
  ``_compile_worker`` subprocess) ``quack.cache.COMPILE_ONLY`` is already
  True on entry, so ``@jit_cache`` returns ``_noop_kernel`` for every
  ``_compile_*(...)`` it populates. The body runs end-to-end, the .o
  cache is filled, and no kernel is actually launched.

This removes the need for hand-written ``_*_fake`` twins on each op.

Note: we deliberately do NOT gate on ``torch.compiler.is_compiling()`` —
that flag's underlying ``_is_compiling_flag`` is only set during
``torch.export``, never during ``torch.compile``. Dynamo's
``_get_fake_value_impl`` would otherwise run the body and surface
any ``_compile_*`` ``ValueError`` as a ``TorchRuntimeError`` graph break.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

import torch

import quack.cache


__all__ = ["cute_op"]


def _has_symint(value: Any) -> bool:
    """Return True if ``value`` carries any ``torch.SymInt`` that would poison
    ``@jit_cache`` keys or ``_compile_*`` SymInt-hostile paths downstream.

    Walks direct scalar SymInt args, tensor ``.shape``/``.stride()`` SymInts,
    and nested ``tuple``/``list``/``dict``. We deliberately do not gate on
    tensor identity alone: a scalar ``int`` schema arg (e.g. ``sm_count``,
    ``max_seqlen``, ``num_heads_q``) can arrive as a SymInt computed from a
    fake tensor that is not in this op's args.
    """
    if isinstance(value, torch.SymInt):
        return True
    if isinstance(value, torch.Tensor):
        if any(isinstance(s, torch.SymInt) for s in value.shape):
            return True
        try:
            strides = value.stride()
        except (RuntimeError, NotImplementedError):
            # Some fake/meta tensors may not expose strides; shape is enough.
            return False
        return any(isinstance(s, torch.SymInt) for s in strides)
    if isinstance(value, (tuple, list)):
        return any(_has_symint(v) for v in value)
    if isinstance(value, dict):
        return any(_has_symint(v) for v in value.values())
    return False


def cute_op(
    name: str,
    *,
    mutates_args: Union[str, Iterable[str]],
    schema: Optional[str] = None,
    device_types: Optional[Union[str, Iterable[str]]] = None,
) -> Callable:
    """Like ``torch.library.triton_op``, but for CuTe DSL kernels.

    Args:
        name: ``"namespace::op_name"``.
        mutates_args: Names of mutated tensor args.
        schema: Optional explicit schema. Required when mutating an
            ``Optional[Tensor]`` arg (PyTorch can't infer those).
        device_types: Optional device-type restriction.
    """

    def dec(fn: Callable) -> Any:
        kwargs: dict[str, Any] = {"mutates_args": mutates_args}
        if schema is not None:
            kwargs["schema"] = schema
        if device_types is not None:
            kwargs["device_types"] = device_types
        op = torch.library.custom_op(name, fn, **kwargs)

        @op.register_fake
        def _fake(*args, **kw):
            # Only populate the .o cache in the explicit COMPILE_ONLY scenario
            # (pytest --compile-only or quack._compile_worker). Under regular
            # torch.compile / AOT autograd tracing the body must stay a no-op:
            # the op only mutates inputs (no fake output to produce) and the
            # body would otherwise raise for shape/dtype combos that the
            # kernel intentionally rejects.
            if not quack.cache.is_compile_only():
                return
            if _has_symint(args) or _has_symint(kw):
                return
            fn(*args, **kw)

        return op

    return dec
