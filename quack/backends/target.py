"""Backend target identification.

Build-vendor detection is safe to bind before Dynamo tracing. Exact device and
architecture detection must be called from an opaque backend-op body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Target:
    """An exact device target used for provider support and cache selection."""

    vendor: str
    arch: str
    device_index: int


def vendor_from_torch_build() -> str:
    """Return ``"amd"``, ``"nvidia"``, or ``"cpu"`` for this PyTorch build."""

    import torch

    if torch.version.hip is not None:
        return "amd"
    if torch.version.cuda is not None:
        return "nvidia"
    return "cpu"


def target_from_tensor(tensor: Any) -> Target:
    """Resolve an exact target from a real tensor.

    This performs a device-property query and must not run inside a Dynamo
    graph. Backend custom-op bodies are the intended call site.
    """

    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"target_from_tensor expects a torch.Tensor, got {type(tensor)!r}")
    if tensor.device.type != "cuda":
        raise ValueError(f"Quack GPU backends require a CUDA/HIP tensor, got {tensor.device}")

    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    vendor = vendor_from_torch_build()
    if vendor == "amd":
        arch = (getattr(props, "gcnArchName", "") or "").split(":", 1)[0]
        if not arch:
            raise RuntimeError(f"could not determine AMD gfx architecture for cuda:{device_index}")
    elif vendor == "nvidia":
        major = getattr(props, "major", None)
        minor = getattr(props, "minor", None)
        if major is None or minor is None:
            major, minor = torch.cuda.get_device_capability(device_index)
        arch = f"sm_{major}{minor}"
    else:
        raise RuntimeError("this PyTorch build has neither CUDA nor ROCm support")
    return Target(vendor=vendor, arch=arch, device_index=device_index)
