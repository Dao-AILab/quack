# Copyright (c) 2026, QuACK team.
"""Distributed (multi-GPU overlapped) GEMMs.

Package layout:
- all_gather_gemm.py: AllGatherRunner (ce-push transport, gather() context);
  the module docstring is the full design record
- block_scaled_allgather_gemm.py: BlockScaledAllGatherRunner (adds a
  scale-factor lane under the same arrival flags, for blockscaled GEMMs)
"""

from quack.distributed.all_gather_gemm import AllGatherRunner
from quack.distributed.block_scaled_allgather_gemm import BlockScaledAllGatherRunner

__all__ = ["AllGatherRunner", "BlockScaledAllGatherRunner"]
