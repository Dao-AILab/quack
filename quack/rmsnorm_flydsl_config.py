# Copyright (c) 2026, Tri Dao.

"""Pure-Python launch heuristic for the eager FlyDSL RMSNorm forward kernel."""

import math
from dataclasses import dataclass

ACCESS_BITS = 128
WAVE_SIZE = 64
MIN_NUM_THREADS = WAVE_SIZE
TARGET_BLOCK_THREADS = 256
MAX_WIDE_ROW_THREADS = 1024
REGISTER_CACHE_ELEMS = 32
MAX_N = 262144


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length() if value > 1 else 1


@dataclass(frozen=True, slots=True)
class RmsNormFwdConfig:
    """Vector and thread geometry for one normalized row."""

    vecsize: int
    num_threads: int
    num_tiles: int
    num_vecs: int

    @property
    def needs_predicate(self) -> bool:
        return self.num_vecs != self.num_tiles * self.num_threads

    @property
    def reload_from_gmem(self) -> bool:
        return self.num_tiles * self.vecsize > REGISTER_CACHE_ELEMS

    @classmethod
    def for_forward(cls, n: int, dtype_width: int) -> "RmsNormFwdConfig":
        """Choose the stable analytical row geometry."""
        vecsize = _vector_size(n, dtype_width)
        num_vecs = n // vecsize
        if num_vecs < MIN_NUM_THREADS:
            num_threads = _next_power_of_two(num_vecs)
        else:
            vectors_per_thread = max(1, REGISTER_CACHE_ELEMS // vecsize)
            required_threads = _next_power_of_two(-(-num_vecs // vectors_per_thread))
            num_threads = min(
                max(required_threads, MIN_NUM_THREADS),
                MAX_WIDE_ROW_THREADS,
            )
            if TARGET_BLOCK_THREADS <= num_threads < MAX_WIDE_ROW_THREADS:
                num_threads = min(num_threads * 2, MAX_WIDE_ROW_THREADS)
        return cls(
            vecsize=vecsize,
            num_threads=num_threads,
            num_tiles=-(-num_vecs // num_threads),
            num_vecs=num_vecs,
        )


def _vector_size(n: int, dtype_width: int) -> int:
    if dtype_width not in (16, 32):
        raise ValueError(f"unsupported element width: {dtype_width} bits")
    return math.gcd(n, ACCESS_BITS // dtype_width)


def rows_per_block(config: RmsNormFwdConfig) -> int:
    """Pack short single-pass rows into a 256-thread block."""
    if config.num_vecs >= MIN_NUM_THREADS:
        return 1
    return max(1, TARGET_BLOCK_THREADS // config.num_threads)


__all__ = [
    "ACCESS_BITS",
    "MAX_N",
    "WAVE_SIZE",
    "RmsNormFwdConfig",
    "rows_per_block",
]
