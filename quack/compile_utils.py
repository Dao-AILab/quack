# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

from typing import Optional

import cutlass.cute as cute


def make_fake_tensor(dtype, shape, divisibility=1) -> Optional[cute.Tensor]:
    if dtype is None:
        return None
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=(*[cute.sym_int64(divisibility=divisibility)] * (len(shape) - 1), 1),
        assumed_align=divisibility * dtype.width // 8,
    )
