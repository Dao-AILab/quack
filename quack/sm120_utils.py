# Copyright (c) 2026, QuACK team.
"""Small public SM120 helper facade.

The SM120 NVFP4 kernel implementation uses ``quack._sm120_nvfp4_utils`` directly.
Keep this module limited to stable inspection helpers used by tests and callers.
"""

from quack import _sm120_nvfp4_utils as _sm120


def get_ab_tma_tx_bytes(
    tile_mn: int = 128,
    tile_k: int = 128,
    *,
    smem_format: str = "packed",
) -> int:
    return _sm120.mxf4nvf4_ab_tma_tx_bytes(tile_mn, tile_k, smem_format=smem_format)


def get_scale_tma_tx_bytes(
    tile_mn: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = _sm120.MXF4NVF4_SCALE_VEC_SIZE,
) -> int:
    return _sm120.mxf4nvf4_scale_tma_tx_bytes(tile_mn, tile_k, sf_vec_size)


def get_full_tma_tx_bytes(
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = _sm120.MXF4NVF4_SCALE_VEC_SIZE,
    *,
    ab_smem_format: str = "packed",
) -> int:
    return _sm120.mxf4nvf4_full_tma_tx_bytes(
        tile_m,
        tile_n,
        tile_k,
        sf_vec_size,
        ab_smem_format=ab_smem_format,
    )


__all__ = [
    "get_ab_tma_tx_bytes",
    "get_full_tma_tx_bytes",
    "get_scale_tma_tx_bytes",
]
