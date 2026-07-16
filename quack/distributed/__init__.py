# Copyright (c) 2026, QuACK team.
"""Distributed (multi-GPU overlapped) GEMMs.

Package layout:
- all_gather_gemm.py: AllGatherRunner (ce-push transport, gather() context);
  design doc AI/allgather_gemm_design.md
"""

from quack.distributed.all_gather_gemm import AllGatherRunner

__all__ = ["AllGatherRunner"]
