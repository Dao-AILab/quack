# Copyright (c) 2026, Tri Dao.

"""Dependency-free constants shared by FlyDSL host and config modules."""

# GFX950 MUBUF's per-lane copy ceiling, not a mandatory transaction width.
# Each operand picks a supported width up to this cap.
MAX_ACCESS_BITS = 128

# CDNA's subgroup width, the unit a shuffle reduction closes over before it has
# to go through LDS. CuTe's cute.arch.WARP_SIZE counterpart.
WAVE_SIZE = 64

__all__ = ["MAX_ACCESS_BITS", "WAVE_SIZE"]
