"""Quack's backend-aware public package surface.

CUDA builds preserve the original eager CuTe initialization. ROCm builds keep
both compiler backends unloaded until a public operator is requested.
"""

from __future__ import annotations

import os
from importlib import import_module

__version__ = "0.6.1"

__all__ = ["rmsnorm", "softmax", "cross_entropy", "RoundingMode"]

_LAZY_EXPORTS = {
    "softmax": ("quack.softmax", "softmax"),
    "cross_entropy": ("quack.cross_entropy", "cross_entropy"),
    "RoundingMode": ("quack.rounding", "RoundingMode"),
}


def __getattr__(name: str):
    if name == "rmsnorm":
        import torch

        if torch.compiler.is_compiling():
            raise RuntimeError("resolve `from quack import rmsnorm` before entering torch.compile")
        from quack.backends.registry import REGISTRY
        from quack.backends.target import vendor_from_torch_build

        vendor = vendor_from_torch_build()
        if vendor not in {"amd", "nvidia"}:
            raise RuntimeError("quack.rmsnorm requires a CUDA or ROCm PyTorch build")
        value = REGISTRY.load_provider("rmsnorm", vendor).load_op()
        globals()[name] = value
        return value
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def _initialize_backend_exports() -> None:
    """Restore pre-existing CUDA initialization and callable exports.

    CUDA builds install CuTe tensor helpers and bind callable package exports
    during ``import quack``. ROCm builds leave exports lazy so importing Quack
    never requires CuTe or FlyDSL.
    """

    import torch

    if torch.version.hip is not None:
        return
    if torch.version.cuda is None:
        return

    import_module("quack.dsl")
    if os.environ.get("CUTE_DSL_PTXAS_PATH") is not None:
        from quack.dsl import cute_dsl_ptxas as _cute_dsl_ptxas

        _cute_dsl_ptxas.patch()

    exports = {
        "rmsnorm": ("quack.rmsnorm", "rmsnorm"),
        "softmax": ("quack.softmax", "softmax"),
        "cross_entropy": ("quack.cross_entropy", "cross_entropy"),
        "RoundingMode": ("quack.rounding", "RoundingMode"),
    }
    for name, (module_name, attr_name) in exports.items():
        globals()[name] = getattr(import_module(module_name), attr_name)


_initialize_backend_exports()
