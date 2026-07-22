# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""Compatibility export for CuTe callers.

The implementation is backend-neutral and lives in
``quack._torch_library_op`` so optional backends do not import ``quack.dsl``.
"""

from quack._torch_library_op import torch_library_op as cute_op

__all__ = ["cute_op"]
