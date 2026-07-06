"""JAX bindings for QuACK softmax kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from quack.jax_utils import (
    TvmFfiKernel,
    check_rank,
    cutlass_dtype,
    shape_dtype_like,
)
from quack.softmax import Softmax, SoftmaxBackward


def _check_2d(name: str, x) -> None:
    check_rank(name, x, 2)


_SOFTMAX_FWD = TvmFfiKernel(
    "quack_softmax_fwd",
    lambda dtype, n_cols: Softmax.compile(
        cutlass_dtype(dtype),
        cutlass_dtype(dtype),
        n_cols,
    ),
)

_SOFTMAX_BWD = TvmFfiKernel(
    "quack_softmax_bwd",
    lambda dtype, n_cols: SoftmaxBackward.compile(
        cutlass_dtype(dtype),
        cutlass_dtype(dtype),
        cutlass_dtype(dtype),
        n_cols,
    ),
)


def _softmax_fwd(x):
    _check_2d("x", x)
    cutlass_dtype(x.dtype)
    if 0 in x.shape:
        return jnp.empty_like(x)
    return _SOFTMAX_FWD(
        x,
        key=(jnp.dtype(x.dtype), x.shape[1]),
        output_shape_dtype=shape_dtype_like(x),
    )


def _softmax_bwd(dy, y):
    _check_2d("dy", dy)
    _check_2d("y", y)
    if dy.shape != y.shape:
        raise ValueError(f"dy and y must have the same shape, got {dy.shape} and {y.shape}")
    if dy.dtype != y.dtype:
        raise TypeError(f"dy and y must have the same dtype, got {dy.dtype} and {y.dtype}")
    cutlass_dtype(dy.dtype)
    if 0 in dy.shape:
        return jnp.empty_like(dy)
    return _SOFTMAX_BWD(
        dy,
        y,
        key=(jnp.dtype(dy.dtype), dy.shape[1]),
        output_shape_dtype=shape_dtype_like(dy),
    )


@jax.custom_vjp
def softmax(x):
    """Apply QuACK softmax with a custom JAX VJP."""
    return _softmax_fwd(x)


def _softmax_rule_fwd(x):
    y = _softmax_fwd(x)
    return y, y


def _softmax_rule_bwd(y, dy):
    return (_softmax_bwd(dy, y),)


softmax.defvjp(_softmax_rule_fwd, _softmax_rule_bwd)


__all__ = ["softmax"]
