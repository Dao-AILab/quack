# Copyright (c) 2026, Tri Dao.
"""The quack torch.library custom ops must register on every supported Python.

torch.library infers the op schema from the Python signature and renders scalar
defaults with ``str()``. On Python 3.10 ``str(IntEnum)`` is ``"RoundingMode.RN"``
(3.11+ prints the integer), so an ``IntEnum`` default makes the schema
unparseable and ``import quack.gemm_interface`` fails. CPU-only."""

import enum
import inspect

import torch
from torch._library.custom_ops import CustomOpDef

import quack.gemm_interface as gemm_interface


def _custom_ops(module):
    return [(n, o) for n, o in vars(module).items() if isinstance(o, CustomOpDef)]


def test_custom_op_scalar_defaults_are_not_enums():
    ops = _custom_ops(gemm_interface)
    assert len(ops) >= 4, [n for n, _ in ops]
    for name, op in ops:
        fn = op._init_fn
        for param in inspect.signature(fn).parameters.values():
            default = param.default
            if default is inspect.Parameter.empty:
                continue
            assert not isinstance(default, enum.Enum), (
                f"{name}({param.name}={default!r}): custom-op defaults must be plain "
                "scalars; str(IntEnum) is not a schema literal on Python 3.10"
            )


def test_custom_op_schemas_parse():
    for _, op in _custom_ops(gemm_interface):
        schema = str(getattr(torch.ops.quack, op._name).default._schema)
        assert "Mode." not in schema, schema
