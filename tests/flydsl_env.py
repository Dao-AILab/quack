# Copyright (c) 2026, Tri Dao.

"""Which rmsnorm_fwd backend this machine can test, decided at import time.

Not a conftest fixture: the test module needs the verdict before ``pytestmark``
and before it picks which quack module to import.

Nothing here imports quack or cutlass. ``torch.version.hip`` already separates
the two builds, and importing cutlass on ROCm loads an MLIR runtime that is
incompatible with FlyDSL's.
"""

import importlib.util

import torch

_HAS_GPU = torch.cuda.is_available()
_IS_ROCM = torch.version.hip is not None

DEVICE = torch.device("cuda", torch.cuda.current_device()) if _HAS_GPU else torch.device("cuda")
ARCH = (
    torch.cuda.get_device_properties(DEVICE).gcnArchName.split(":", 1)[0]
    if _HAS_GPU and _IS_ROCM
    else None
)

# BACKEND is what quack.rmsnorm_fwd will resolve to on this machine -- see the
# _FORWARD_BACKENDS table in quack/__init__.py, which keys off the same build flag.
if not _HAS_GPU:
    BACKEND, SKIP_REASON = None, "requires a GPU"
elif not _IS_ROCM:
    BACKEND, SKIP_REASON = "cute", ""
elif importlib.util.find_spec("flydsl") is None:
    BACKEND, SKIP_REASON = None, "requires flydsl"
elif ARCH != "gfx950":
    BACKEND, SKIP_REASON = None, "FlyDSL kernels currently require gfx950"
else:
    BACKEND, SKIP_REASON = "flydsl", ""

CAN_RUN = BACKEND is not None
IS_FLYDSL = BACKEND == "flydsl"
