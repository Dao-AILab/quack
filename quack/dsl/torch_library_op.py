# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""``cute_op``: ``torch.library.custom_op`` for CuTe DSL kernels.

Same trick as ``torch.library.triton_op`` (register the impl as the fake/meta
kernel too), specialized for our setup: the fake is a pure no-op. Our ops
only mutate their inputs, so Dynamo / AOT autograd need no shape effect from
the fake, and kernel compilation is owned entirely by ``jit_cache`` (plus
the async compile pool) at real execution time.

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

__all__ = ["cute_op"]


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
            # Pure no-op: our ops only mutate their input tensors, so under
            # torch.compile / AOT autograd tracing there is no fake output to
            # produce, and running the body would pay compile latency at
            # dynamo trace time (or crash for shape/dtype combos the kernel
            # intentionally rejects). Kernel compilation is handled by
            # jit_cache + the async compile pool at real execution time.
            return

        return op

    return dec
