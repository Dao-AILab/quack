# Copyright (c) 2026, QuACK team.
"""Distributed (multi-GPU overlapped) GEMMs.

Package layout:
- all_gather_gemm.py: AllGatherRunner (ce-push transport, gather() context);
  the module docstring is the full design record
"""

from quack.distributed.all_gather_gemm import AllGatherRunner

__all__ = ["AllGatherRunner"]
