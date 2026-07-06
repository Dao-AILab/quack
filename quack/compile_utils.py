# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

from typing import Optional

import cutlass.cute as cute


def make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1) -> Optional[cute.Tensor]:
    """Build a fake CuTe tensor with dynamic (sym) strides for tensor-free compilation.

    ``leading_dim`` selects the dim whose stride is statically 1 (matching
    ``from_dlpack(...).mark_layout_dynamic(leading_dim=...)``). Pass
    ``leading_dim=None`` for a fully-dynamic layout with no static stride-1 dim
    (matching ``mark_layout_dynamic()`` on a tensor without a contiguous dim).

    ``divisibility`` is in elements; ``assumed_align`` (bytes) is
    ``divisibility * dtype.width // 8``.
    """
    if dtype is None:
        return None
    if leading_dim is not None and leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8
    )


def make_fake_stream():
    """Fake CUDA stream for tensor-free compilation (real stream comes from the TVM FFI env)."""
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
