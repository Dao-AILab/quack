import cutlass.cute as cute
from cutlass import Float32, const_expr

import quack.utils as utils


@cute.jit
def compare_and_swap(
    arr: cute.Tensor, i: int, j: int, ascending: bool = True, use_selection: bool = False
) -> None:
    """Compare and swap elements at indices i and j in ascending or descending order."""
    if const_expr(use_selection):
        a, b = arr[i], arr[j]
        if (a > b) ^ (not ascending):
            arr[i] = b
            arr[j] = a
        # if const_expr(ascending):
        #     if a > b:
        #         arr[i] = b
        #         arr[j] = a
        # else:
        #     if a < b:
        #         arr[i] = b
        #         arr[j] = a
    else:
        min_fn = min if const_expr(arr.element_type != Float32) else utils.fmin
        max_fn = max if const_expr(arr.element_type != Float32) else cute.arch.fmax
        if const_expr(ascending):
            arr[i], arr[j] = min_fn(arr[i], arr[j]), max_fn(arr[i], arr[j])
        else:
            arr[i], arr[j] = max_fn(arr[i], arr[j]), min_fn(arr[i], arr[j])
