"""Shared helpers for optional JAX bindings."""

from __future__ import annotations

from typing import Any, Callable

import cutlass
import jax
import jax.numpy as jnp
import numpy as np


JAX_TO_CUTLASS_DTYPE: dict[jnp.dtype, type[cutlass.Numeric]] = {
    jnp.dtype(jnp.float16): cutlass.Float16,
    jnp.dtype(jnp.bfloat16): cutlass.BFloat16,
    jnp.dtype(jnp.float32): cutlass.Float32,
}


def require_jax_tvm_ffi():
    try:
        import jax_tvm_ffi
    except ImportError as e:
        raise ImportError(
            "This QuACK JAX TVM-FFI path requires jax-tvm-ffi. Install it with "
            "`pip install jax-tvm-ffi`."
        ) from e
    return jax_tvm_ffi


def dtype_name(dtype: jnp.dtype) -> str:
    return str(jnp.dtype(dtype)).replace("float", "f").replace("bfloat", "bf")


def cutlass_dtype(dtype: jnp.dtype) -> type[cutlass.Numeric]:
    dtype = jnp.dtype(dtype)
    if dtype not in JAX_TO_CUTLASS_DTYPE:
        raise TypeError(f"Unsupported dtype: {dtype}")
    return JAX_TO_CUTLASS_DTYPE[dtype]


def check_rank(name: str, x: Any, rank: int) -> None:
    if len(x.shape) != rank:
        raise ValueError(f"{name} must be {rank}D, got shape {x.shape}")


def shape_dtype_like(x: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


def tvm_ffi_call(
    target: str,
    *args,
    output_shape_dtype,
    vmap_method: str = "broadcast_all",
    **kwargs,
):
    call = jax.ffi.ffi_call(
        target,
        output_shape_dtype,
        vmap_method=vmap_method,
        **kwargs,
    )
    return call(*args)


def _tvm_ffi_key_part(value: Any) -> str:
    if isinstance(value, np.dtype):
        return dtype_name(value)
    if isinstance(value, type) and issubclass(value, cutlass.Numeric):
        return value.__name__.lower()
    return str(value).replace(" ", "").replace("/", "_").replace(".", "p")


class TvmFfiKernel:
    """Lazy JAX FFI wrapper for a family of TVM-FFI compiled kernels."""

    def __init__(
        self,
        name: str,
        compile_fn: Callable[..., Any],
        *,
        target_name: Callable[..., str] | None = None,
        platform: str = "gpu",
        arg_spec: list[str] | None = None,
        allow_cuda_graph: bool = False,
        pass_owned_tensor: bool = False,
        use_last_output_for_alloc_workspace: bool = False,
    ) -> None:
        self.name = name
        self.compile_fn = compile_fn
        self.target_name = target_name
        self.platform = platform
        self.arg_spec = arg_spec
        self.allow_cuda_graph = allow_cuda_graph
        self.pass_owned_tensor = pass_owned_tensor
        self.use_last_output_for_alloc_workspace = use_last_output_for_alloc_workspace
        self._targets: dict[tuple[Any, ...], str] = {}
        self._compiled: dict[tuple[Any, ...], Any] = {}

    def _target_for_key(self, key: tuple[Any, ...]) -> str:
        if self.target_name is not None:
            return self.target_name(*key)
        if not key:
            return self.name
        suffix = "_".join(_tvm_ffi_key_part(part) for part in key)
        return f"{self.name}_{suffix}"

    def target(self, key: tuple[Any, ...]) -> str:
        key = tuple(key)
        if key in self._targets:
            return self._targets[key]
        target = self._target_for_key(key)
        compiled = self.compile_fn(*key)
        require_jax_tvm_ffi().register_ffi_target(
            target,
            compiled,
            arg_spec=self.arg_spec,
            platform=self.platform,
            allow_cuda_graph=self.allow_cuda_graph,
            pass_owned_tensor=self.pass_owned_tensor,
            use_last_output_for_alloc_workspace=self.use_last_output_for_alloc_workspace,
        )
        self._targets[key] = target
        self._compiled[key] = compiled
        return target

    def __call__(
        self,
        *args,
        key: tuple[Any, ...],
        output_shape_dtype,
        vmap_method: str = "broadcast_all",
        **kwargs,
    ):
        return tvm_ffi_call(
            self.target(key),
            *args,
            output_shape_dtype=output_shape_dtype,
            vmap_method=vmap_method,
            **kwargs,
        )
