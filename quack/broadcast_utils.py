# Copyright (c) 2025, Tri Dao.
from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from quack.layout_utils import make_acc_tensor_mn_view


@cute.jit
def vec_op(tCrC: cute.Tensor, tCrVec: cute.Tensor, op: Callable, is_colvec: bool) -> None:
    if const_expr(tCrC.element_type != Float32):  # Convert to f32
        tCrC_f32 = cute.make_rmem_tensor(tCrC.shape, Float32)
        tCrC_f32.store(tCrC.load().to(Float32))
    else:
        tCrC_f32 = tCrC
    # this happens to work for frgA layout too, not just acc layout
    tCrC_f32_mn = make_acc_tensor_mn_view(tCrC_f32)
    if const_expr(is_colvec):
        assert cute.size(tCrC_f32_mn, mode=[0]) == cute.size(tCrVec)
        for r in cutlass.range(cute.size(tCrC_f32_mn, mode=[0]), unroll_full=True):
            tCrC_f32_mn[r, None].store(op(tCrC_f32_mn[r, None].load(), tCrVec[r]))
    else:
        assert cute.size(tCrC_f32_mn, mode=[1]) == cute.size(tCrVec)
        for c in cutlass.range(cute.size(tCrC_f32_mn, mode=[1]), unroll_full=True):
            tCrC_f32_mn[None, c].store(op(tCrC_f32_mn[None, c].load(), tCrVec[c]))
    if const_expr(tCrC.element_type != Float32):  # Convert back to original dtype
        tCrC.store(tCrC_f32.load().to(tCrC.element_type))
