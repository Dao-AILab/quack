"""Lazy adapter for the existing CuTe RMSNorm implementation."""

from importlib.util import find_spec

from quack.backends.protocol import Support


def is_available() -> bool:
    return find_spec("cutlass") is not None and find_spec("cuda") is not None


def supports(spec, target) -> Support:
    if target.vendor != "nvidia":
        return Support.no(f"CuTe RMSNorm requires an NVIDIA target, got {target.vendor!r}")
    return Support.yes()


def load_op():
    from quack.rmsnorm import rmsnorm

    return rmsnorm
