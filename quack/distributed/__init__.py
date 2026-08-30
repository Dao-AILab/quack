# Copyright (c) 2026, QuACK team.
"""Distributed (multi-GPU overlapped) GEMMs.

Package layout:
- all_gather_gemm.py: AllGatherRunner (ce-push transport, gather() context;
  the module docstring is the full design record) and
  BlockScaledAllGatherRunner (adds a scale-factor lane under the same
  arrival flags, for blockscaled GEMMs)
"""

from quack.distributed.all_gather_gemm import (
    AllGatherRunner,
    BlockScaledAllGatherRunner,
)

__all__ = ["AllGatherRunner", "BlockScaledAllGatherRunner"]
