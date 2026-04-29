__version__ = "0.4.0"

import os

# Two CuTeDSL workarounds, both must run before the first cute.compile call:
#   - cutlass#3161: duplicate .text section flags break MCJIT in multi-process
#     loads (see quack.cute_dsl_elf_fix).
#   - cutlass#3062: ir.Context spawns LLVM thread pools that leak across
#     compiles, eventually exhausting pthreads (see quack.cute_dsl_mlir_threading).
import quack.cute_dsl_elf_fix
import quack.cute_dsl_mlir_threading

quack.cute_dsl_elf_fix.patch()
quack.cute_dsl_mlir_threading.patch()

from quack.rmsnorm import rmsnorm  # noqa: E402
from quack.softmax import softmax  # noqa: E402
from quack.cross_entropy import cross_entropy  # noqa: E402
from quack.rounding import RoundingMode  # noqa: E402
from quack.sm120_tma_utils import (  # noqa: E402
    assert_sm120_direct_tma_2d,
    has_sm120_direct_tma_2d,
)


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    import quack.cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    quack.cute_dsl_ptxas.patch()


__all__ = [
    "rmsnorm",
    "softmax",
    "cross_entropy",
    "RoundingMode",
    "assert_sm120_direct_tma_2d",
    "has_sm120_direct_tma_2d",
]
