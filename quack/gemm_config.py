# Copyright (C) 2025, Tri Dao.
import itertools
from typing import Optional, List
from dataclasses import dataclass


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 192
    cluster_m: int = 2
    cluster_n: int = 1
    swap_ab: bool = False
    pingpong: bool = True
    # raster_order: int = 2
    # max_swizzle_size: int = 1


def get_all_configs(
    epilogue: Optional[str] = None,
    # tune_raster_order=True,
) -> List[GemmConfig]:
    tile_n_vals = [128, 144, 160, 176, 192, 208]
    tile_mn_coop_vals = [(256, tile_n) for tile_n in tile_n_vals] + [
        (128, 224),
        (128, 256),
        (192, 256),
    ]
    tile_mn_pingpong_vals = [(128, tile_n) for tile_n in tile_n_vals] + [(192, 128)]
    if epilogue in ["swiglu"]:
        tile_mn_coop_vals = [(m, n) for m, n in tile_mn_coop_vals if n % 32 == 0 and m != 192]
        tile_mn_pingpong_vals = [(m, n) for m, n in tile_mn_pingpong_vals if n % 32 == 0]
    elif epilogue in ["lse"]:
        tile_mn_coop_vals = [(m, n) for m, n in tile_mn_coop_vals if m != 192]
    tile_mn_vals = [(m, n, False) for m, n in tile_mn_coop_vals] + [
        (m, n, True) for m, n in tile_mn_pingpong_vals
    ]
    # cluster = [(1, 2), (2, 1)]
    cluster = [(1, 1), (1, 2), (2, 1)]
    if epilogue in ["lse"]:
        cluster = [(1, 2), (2, 1)]
    swap_ab_vals = [False, True]
    if epilogue in ["lse", "swiglu"]:
        swap_ab_vals = [False]
    # raster_swizzle = (
    #     [(0, 1)]
    #     if not tune_raster_order
    #     else [(1, 1), (1, 2), (1, 4), (1, 8), (2, 1), (2, 2), (2, 4), (2, 8)]
    # )
    return [
        GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            swap_ab=swap_ab,
            pingpong=pingpong,
            # raster_order=raster_order,
            # max_swizzle_size=max_swizzle_size,
        )
        for (tile_m, tile_n, pingpong), (cluster_m, cluster_n), swap_ab in itertools.product(
            tile_mn_vals,
            cluster,
            swap_ab_vals,
            # raster_swizzle,
        )
    ]
