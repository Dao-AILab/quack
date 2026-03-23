# Copyright (C) 2025, Fri Dao.
import itertools
from typing import Optional, List
from functools import partial
from dataclasses import dataclass


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 192
    pingpong: bool = True
    # by default, we use dynamic persistent tile scheduler on SM100 but not on SM90
    is_dynamic_persistent: bool = True
    cluster_m: int = 2
    cluster_n: int = 1
    swap_ab: bool = False
    # raster_order: int = 1
    max_swizzle_size: int = 8
    device_capacity: int = 9


def _get_sm90_configs(
    epilogue: Optional[str] = None,
    tune_coop: bool = True,
) -> List[GemmConfig]:
    tile_n_vals = [128, 160, 192, 208]
    tile_mn_coop_vals = [(256, tile_n) for tile_n in tile_n_vals] + [
        (128, 224),
        (128, 256),
        # (192, 256),  # Getting IOT instruction (core dumped) in the bwd
    ]
    tile_mn_pingpong_vals = [(128, tile_n) for tile_n in tile_n_vals] + [(192, 128)]
    if epilogue in ["gated"]:
        tile_mn_coop_vals = [(m, n) for m, n in tile_mn_coop_vals if n % 32 == 0 and m != 192]
        tile_mn_pingpong_vals = [(m, n) for m, n in tile_mn_pingpong_vals if n % 32 == 0]
    elif epilogue in ["lse"]:
        tile_mn_coop_vals = [(m, n) for m, n in tile_mn_coop_vals if m != 192]
    tile_mn_vals = []
    if tune_coop:
        tile_mn_vals += [(m, n, False) for m, n in tile_mn_coop_vals]
    tile_mn_vals += [(m, n, True) for m, n in tile_mn_pingpong_vals]
    cluster = [(1, 2), (2, 1)]
    # cluster = [(1, 1), (1, 2), (2, 1)]
    if epilogue in ["lse"]:
        cluster = [(1, 2), (2, 1)]
    swap_ab_vals = [False, True]
    if epilogue in ["lse", "gated"]:
        swap_ab_vals = [False]

    return [
        GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            pingpong=pingpong,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            swap_ab=swap_ab,
            device_capacity=9,
            is_dynamic_persistent=False,  # default to not use dynamic persistent on SM90
        )
        for (tile_m, tile_n, pingpong), (cluster_m, cluster_n), swap_ab in itertools.product(
            tile_mn_vals,
            cluster,
            swap_ab_vals,
        )
    ]


def _get_sm100_configs(
    epilogue: Optional[str] = None,
) -> List[GemmConfig]:
    tile_n_vals = [64, 128, 160, 192, 224, 256]
    tile_mn_cluster_vals = (
        [(128, tile_n, (1, 1)) for tile_n in tile_n_vals]
        + [(128, tile_n, (1, 2)) for tile_n in tile_n_vals]
        + [(128, tile_n, (2, 1)) for tile_n in tile_n_vals]
        + [(128, tile_n, (2, 2)) for tile_n in tile_n_vals]
        + [(256, tile_n, (2, 1)) for tile_n in tile_n_vals]
        + [(256, tile_n, (2, 2)) for tile_n in tile_n_vals]
    )
    swap_ab_vals = [False, True]
    if epilogue in ["lse", "gated"]:
        swap_ab_vals = [False]
    GemmConfigCls = partial(
        GemmConfig, pingpong=False, device_capacity=10
    )  # There's no pingpong on Sm100
    use_clc_vals = [True, False]
    return [
        GemmConfigCls(
            tile_m=m,
            tile_n=n,
            cluster_m=cm,
            cluster_n=cn,
            swap_ab=sab,
            max_swizzle_size=8,
            is_dynamic_persistent=use_clc,
        )
        for (m, n, (cm, cn)), sab, use_clc in itertools.product(
            tile_mn_cluster_vals, swap_ab_vals, use_clc_vals
        )
    ]


def get_all_configs(
    epilogue: Optional[str] = None,
    tune_coop: bool = True,
) -> List[GemmConfig]:
    """Return autotuning configs for all supported device capabilities (sm90 + sm100).

    Each GemmConfig is tagged with its target device_capacity, so the caller can
    filter at runtime based on the actual device. This avoids querying the device
    (and initializing a CUDA context) at import time.
    """
    return _get_sm90_configs(epilogue, tune_coop) + _get_sm100_configs(epilogue)
