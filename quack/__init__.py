__version__ = "0.6.1"

import os

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
