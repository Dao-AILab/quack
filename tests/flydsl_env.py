# Copyright (c) 2026, Tri Dao.

"""Environment gate shared by the FlyDSL test modules.

Not a conftest fixture: each module needs these at import time to build its own
``pytestmark`` and to guard the ``quack.*`` imports, which pull in flydsl.
"""

import importlib.util

import torch

HAS_ROCM_GPU = torch.version.hip is not None and torch.cuda.is_available()
DEVICE = torch.device("cuda", torch.cuda.current_device()) if HAS_ROCM_GPU else torch.device("cuda")
ARCH = (
    torch.cuda.get_device_properties(DEVICE).gcnArchName.split(":", 1)[0] if HAS_ROCM_GPU else None
)
HAS_FLYDSL = importlib.util.find_spec("flydsl") is not None
CAN_RUN = HAS_ROCM_GPU and ARCH == "gfx950" and HAS_FLYDSL

if not HAS_ROCM_GPU:
    SKIP_REASON = "requires a ROCm GPU"
elif ARCH != "gfx950":
    SKIP_REASON = "FlyDSL kernels currently require gfx950"
elif not HAS_FLYDSL:
    SKIP_REASON = "requires flydsl"
else:
    SKIP_REASON = ""
