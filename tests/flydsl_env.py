# Copyright (c) 2026, Tri Dao.

"""Which rmsnorm_fwd backend this machine can test, decided at import time.

Not a conftest fixture: the test module needs the verdict before ``pytestmark``
and before it picks a quack module to import.

Recomputes ``torch.version.hip`` instead of importing :data:`IS_ROCM_BUILD`,
because reaching ``quack._platform`` executes ``quack/__init__.py``, which on
CUDA pulls in the whole CuTe stack -- a skip verdict must not depend on the
stack under test importing cleanly.
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

# Selection is a build fact; running also needs a device and, on ROCm, gfx950
# plus the optional package.
SELECTED_BACKEND = "flydsl" if _IS_ROCM else "cute"

if not _HAS_GPU:
    CAN_RUN, SKIP_REASON = False, "requires a GPU"
elif not _IS_ROCM:
    CAN_RUN, SKIP_REASON = True, ""
elif ARCH != "gfx950":
    CAN_RUN, SKIP_REASON = False, "FlyDSL kernels currently require gfx950"
elif importlib.util.find_spec("flydsl") is None:
    CAN_RUN, SKIP_REASON = False, "requires flydsl"
else:
    CAN_RUN, SKIP_REASON = True, ""

IS_FLYDSL = CAN_RUN and SELECTED_BACKEND == "flydsl"
