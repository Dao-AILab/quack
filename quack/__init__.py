__version__ = "0.3.10"

import os

from quack.rmsnorm import rmsnorm
from quack.softmax import softmax
from quack.cross_entropy import cross_entropy
from quack.rounding import RoundingMode
from quack.gemm_interface import gemm, gemm_grouped


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    import quack.cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    quack.cute_dsl_ptxas.patch()


__all__ = [
    "rmsnorm",
    "softmax",
    "cross_entropy",
    "gemm",
    "gemm_grouped",
    "RoundingMode",
]
