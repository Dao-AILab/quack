"""Lazy operator-backend selection primitives."""

from quack.backends.protocol import BackendProvider, Support
from quack.backends.registry import REGISTRY, BackendRegistry
from quack.backends.rmsnorm_spec import RMSNormSpec, normalize_rmsnorm_spec
from quack.backends.target import Target, target_from_tensor, vendor_from_torch_build

__all__ = [
    "BackendProvider",
    "BackendRegistry",
    "REGISTRY",
    "RMSNormSpec",
    "Support",
    "Target",
    "normalize_rmsnorm_spec",
    "target_from_tensor",
    "vendor_from_torch_build",
]
