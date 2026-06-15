__version__ = "0.5.0"

import os

# Try the binary shim first (it patches both libnvvm and ptxas inside the
# CUTLASS DSL `.so` with a single mprotect+trampoline pass, no per-compile
# Python overhead). The shim is off by default and activates only when
# QUACK_CUTE_DSL_SHIM=1 is set.
#
# If the shim activates, the legacy Python `cute_dsl_ptxas` hook is skipped
# — otherwise it falls back when CUTE_DSL_PTXAS_PATH is set.
from quack.dsl import cute_dsl_shim as _cute_dsl_shim

_shim_active = _cute_dsl_shim.try_activate()

if not _shim_active and os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from quack.dsl import cute_dsl_ptxas as _cute_dsl_ptxas

    # Patch before importing any modules that instantiate CuTeDSL. The patch
    # forces PTX dumping so the CUDA library loader can replace CUTLASS DSL's
    # embedded ptxas-library cubin with one assembled by system ptxas.
    _cute_dsl_ptxas.patch()

# Pythonic CuTe tensor indexing (`:` / `...` sugar) is installed as a side effect
# of `from quack.dsl import cute_dsl_shim` above — that import runs
# `quack/dsl/__init__.py`, which imports `quack.dsl.cute_tensor_indexing` and
# monkey-patches CuTe's tensor classes process-wide. No explicit import is
# needed here; do not re-add one.
from quack.rmsnorm import rmsnorm  # noqa: E402
from quack.softmax import softmax  # noqa: E402
from quack.cross_entropy import cross_entropy  # noqa: E402
from quack.rounding import RoundingMode  # noqa: E402


__all__ = [
    "rmsnorm",
    "softmax",
    "cross_entropy",
    "RoundingMode",
]
