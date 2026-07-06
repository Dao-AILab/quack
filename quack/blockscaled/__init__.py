# Copyright (c) 2026, Tri Dao.
"""Blockscaled (MXFP8 / MXFP4 / NVFP4) GEMM support.

- :mod:`quack.blockscaled.quantize` — pure-PyTorch quantizers (ported from
  torchao) with torch.compile'd fast paths.
- :mod:`quack.blockscaled.utils` — scale-factor packing/unpacking, operand
  builders for tests/benchmarks, and the kernel-level compile path.

The GEMM entry points live in :mod:`quack.gemm_interface` (pass ``(A, SFA)`` /
``(B, SFB)`` tuples).
"""

from quack.blockscaled.quantize import (  # noqa: F401
    nvfp4_per_tensor_scale,
    to_blocked,
    to_mx,
    to_mx_compiled,
    to_mxfp4,
    to_mxfp4_compiled,
    to_nvfp4,
    to_nvfp4_compiled,
)
from quack.blockscaled.utils import (  # noqa: F401
    BLOCKSCALED_FORMATS,
    blockscaled_gemm_reference,
    blockscaled_quantize,
    dequant_operand,
    pack_scale_2d_to_blocked_contig,
    scale_blocked_for_cublas,
    unpack_scale_blocked_to_2d,
)
