__version__ = "0.6.4"

import os

from quack._platform import IS_ROCM_BUILD

_CUTE_ONLY_EXPORTS = frozenset({"RoundingMode", "cross_entropy", "rmsnorm", "softmax"})

# FlyDSL is a backend of quack's RMSNorm forward, not a separate library: the
# two implementations take the same arguments and return the same triple, so
# the caller picks a device, not a backend. Resolved on first attribute access
# rather than at import so `import quack` does not pull in a backend nobody
# asked for.
_FORWARD_BACKENDS = {"rmsnorm_fwd": "quack.rmsnorm_flydsl" if IS_ROCM_BUILD else "quack.rmsnorm"}


def __getattr__(name):
    backend = _FORWARD_BACKENDS.get(name)
    if backend is not None:
        import importlib

        value = getattr(importlib.import_module(backend), name)
        globals()[name] = value
        return value
    if IS_ROCM_BUILD and name in _CUTE_ONLY_EXPORTS:
        raise ImportError(
            f"quack.{name} requires the CUDA/CuTe backend; "
            "quack.rmsnorm_fwd is available on both backends"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# CuTe and FlyDSL bundle incompatible MLIR runtimes. Keep the package bootstrap
# CuTe-free on ROCm so optional FlyDSL modules can be imported safely.
if not IS_ROCM_BUILD:
    import quack.dsl as _quack_dsl  # noqa: F401

    if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
        from quack.dsl import cute_dsl_ptxas as _cute_dsl_ptxas

        # Patch before importing any modules that instantiate CuTeDSL. The patch
        # forces PTX dumping so the CUDA library loader can replace CUTLASS DSL's
        # embedded ptxas-library cubin with one assembled by system ptxas.
        _cute_dsl_ptxas.patch()

    # Pythonic CuTe tensor indexing (`:` / `...` sugar) is installed as a side effect
    # of importing `quack.dsl`, which imports `quack.dsl.cute_tensor_indexing` and
    # monkey-patches CuTe's tensor classes process-wide.
    from quack.cross_entropy import cross_entropy
    from quack.rmsnorm import rmsnorm
    from quack.rounding import RoundingMode
    from quack.softmax import softmax

    __all__ = [
        "RoundingMode",
        "cross_entropy",
        "rmsnorm",
        "rmsnorm_fwd",
        "softmax",
    ]
else:
    __all__ = ["rmsnorm_fwd"]
