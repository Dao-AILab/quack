"""gfx942/gfx950 RMSNorm launch and reduction selection."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Literal

import torch

from quack.backends.target import Target

ATOMIC = "atomic"
TWO_STAGE = "two_stage"
MAX_BWD_PROGRAMS = 512
TWO_STAGE_MIN_ROWS = 512
TWO_STAGE_MAX_N = 8192


@dataclass(frozen=True, slots=True)
class RMSNormFwdConfig:
    block_threads: int


@dataclass(frozen=True, slots=True)
class RMSNormBwdConfig:
    path: Literal["atomic", "two_stage"]
    block_threads: int
    num_programs: int = 0


def get_bwd_workspace_rows(
    m: int,
    n: int,
    input_dtype: torch.dtype,
) -> int:
    """Return a safe workspace-row bound for runtime backward selection."""

    min_rows = TWO_STAGE_MIN_ROWS * (2 if input_dtype == torch.float32 else 1)
    if m < min_rows or n > TWO_STAGE_MAX_N:
        return 0
    return min(m, MAX_BWD_PROGRAMS)


def select_fwd_config(
    m: int,
    n: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    target: Target,
    cu_count: int,
) -> RMSNormFwdConfig:
    if weight_dtype != torch.float32:
        raise ValueError(f"FlyDSL RMSNorm requires FP32 weight, got {weight_dtype}")
    if n <= 256:
        return RMSNormFwdConfig(block_threads=64)
    if n <= 1024 and (
        input_dtype == torch.float32 or target.arch == "gfx942" or m < max(1, 2 * cu_count)
    ):
        return RMSNormFwdConfig(block_threads=128)
    return RMSNormFwdConfig(block_threads=256)


def select_bwd_config(
    m: int,
    n: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    target: Target,
    cu_count: int,
) -> RMSNormBwdConfig:
    if weight_dtype != torch.float32:
        raise ValueError(f"FlyDSL RMSNorm requires FP32 weight, got {weight_dtype}")
    block_threads = 64 if n <= 256 else 128 if n <= 1024 else 256
    workspace_rows = get_bwd_workspace_rows(m, n, input_dtype)
    if workspace_rows == 0:
        return RMSNormBwdConfig(path=ATOMIC, block_threads=block_threads)
    large_m_threshold = 2048 if target.arch == "gfx950" else 4096
    occupancy_multiplier = 1 if m < large_m_threshold else 2
    num_programs = min(workspace_rows, max(1, occupancy_multiplier * cu_count))
    return RMSNormBwdConfig(
        path=TWO_STAGE,
        block_threads=block_threads,
        num_programs=num_programs,
    )


_HIP_LIB = None
_HIP_LIB_TRIED = False


def _hip_context_identity() -> int:
    """Return the current HIP context handle for Quack's compiled-cache key."""

    global _HIP_LIB, _HIP_LIB_TRIED
    if not _HIP_LIB_TRIED:
        _HIP_LIB_TRIED = True
        for soname in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
            try:
                lib = ctypes.CDLL(soname)
                lib.hipCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
                lib.hipCtxGetCurrent.restype = ctypes.c_int
                _HIP_LIB = lib
                break
            except (AttributeError, OSError):
                continue
    if _HIP_LIB is None:
        return 0
    context = ctypes.c_void_p()
    if _HIP_LIB.hipCtxGetCurrent(ctypes.byref(context)) != 0:
        return 0
    return int(context.value or 0)


def cache_target_identity(target: Target) -> tuple[str, int, int]:
    """Architecture, device, and active context identity for compiled caches."""

    # Materialize the target device's primary context before asking HIP for it.
    torch.cuda.current_stream(target.device_index)
    return target.arch, target.device_index, _hip_context_identity()


__all__ = [
    "ATOMIC",
    "MAX_BWD_PROGRAMS",
    "RMSNormBwdConfig",
    "RMSNormFwdConfig",
    "TWO_STAGE",
    "TWO_STAGE_MAX_N",
    "get_bwd_workspace_rows",
    "select_bwd_config",
    "select_fwd_config",
]
