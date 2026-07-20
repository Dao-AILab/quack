# Copyright (C) 2025, Tri Dao.
import itertools
from typing import Optional, List
from functools import partial
from dataclasses import dataclass


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 192
    tile_k: int | None = None
    # Epilogue sub-tile N extent (SM90 blockscaled). The epilogue drains the output in
    # ceil_div(tile_n, epi_tile_n) N sub-tiles, each costing 2 CTA bar.sync (the dominant
    # epilogue stall). Widening epi_tile_n toward tile_n collapses those sub-tiles (fewer
    # barriers) at the cost of a larger epilogue smem buffer (fewer AB stages). None keeps
    # the heuristic default; must divide tile_n. Swept by the autotuner.
    # NOTE: the M axis is deliberately NOT a lever — widening epi_tile_m forces a full-tile
    # (64KB) output buffer that starves the AB pipeline to 1 stage (~2x slower); the 49KB/
    # stage AB tiles leave no room for it. See _get_sm90_blockscaled_configs.
    epi_tile_n: int | None = None
    num_warps: int | None = None
    pingpong: bool = True
    # by default, we use dynamic persistent tile scheduler on SM100 but not on SM90
    is_dynamic_persistent: bool = True
    cluster_m: int = 2
    cluster_n: int = 1
    cluster_k: int = 1
    swap_ab: bool = False
    # raster_order: int = 1
    max_swizzle_size: int = 8
    device_capacity: int = 9
    # whether to use TMA gather (vs normal cp.async) for gather_A on SM100
    use_tma_gather: bool = False


def _get_sm90_configs(
    epilogue: Optional[str] = None,
    tune_coop: bool = True,
) -> List[GemmConfig]:
    tile_n_vals = [128, 160, 192, 208]
    tile_mn_vals_coop = [(256, tile_n) for tile_n in tile_n_vals] + [
        (128, 224),
        (128, 256),
        # (192, 256),  # Getting IOT instruction (core dumped) in the bwd
    ]
    tile_mn_vals_pingpong = [(128, tile_n) for tile_n in tile_n_vals] + [(192, 128)]
    if epilogue in ["gated"]:
        tile_mn_vals_coop = [(m, n) for m, n in tile_mn_vals_coop if n % 32 == 0 and m != 192]
        tile_mn_vals_pingpong = [(m, n) for m, n in tile_mn_vals_pingpong if n % 32 == 0]
    elif epilogue in ["lse"]:
        tile_mn_vals_coop = [(m, n) for m, n in tile_mn_vals_coop if m != 192]
    tile_mn_vals = []
    if tune_coop:
        tile_mn_vals += [(m, n, False) for m, n in tile_mn_vals_coop]
    tile_mn_vals += [(m, n, True) for m, n in tile_mn_vals_pingpong]
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
            use_tma_gather=False,  # TMA gather not supported on SM90
        )
        for (tile_m, tile_n, pingpong), (cluster_m, cluster_n), swap_ab in itertools.product(
            tile_mn_vals,
            cluster,
            swap_ab_vals,
        )
    ]


def _get_sm90_blockscaled_configs(epilogue: Optional[str] = None) -> List[GemmConfig]:
    """Autotuning configs for the SM90 MXFP8 blockscaled GEMM (software-applied f32 scales).

    Cooperative (non-pingpong) only for now. Correctness constraints:
      - ``tile_n in {128, 256}``: the per-tile weight-scale (``scale_b``) indexing assumes a
        tile spans exactly one or two of the 128-wide weight-scale N-blocks. ``tile_n`` of
        160/192/208 silently produce wrong results (they compile and are fast, so they must
        NOT be offered to the autotuner, which selects on speed alone).
      - ``tile_m in {128, 256}`` (must be a multiple of 128).
      - Pingpong is register-capped to a 128x128 accumulator per warpgroup, so it can neither
        hold ``tile_n=256`` nor serve the gated (concat-B) epilogue's ``tile_n>128`` requirement
        -> cooperative only.
      - no ``swap_ab`` (SFA/SFB would have to swap too; untested).
    ``(256, 256)`` is dropped: its 256-reg/thread accumulator spills (~300 vs ~1350 TFLOPS).
    """
    # tile_n=192 is enabled by the per-N-core predicate scale in mma_blockscaled (a tile
    # straddles a 128-wide SFB block). (256, 192) is excluded: acc = 192 regs/thread spills.
    tile_mn_vals = [(128, 128), (128, 256), (256, 128), (128, 192)]
    if epilogue in ("gated", "dgated"):
        # concat-B gated needs tile_n > 128 AND a multiple of 128: its gate/up parity scale
        # path (unlike the non-gated predicate path) still assumes one 128-wide SFB block per
        # tile half, so it can't straddle a boundary (tile_n=192 would).
        tile_mn_vals = [(m, n) for m, n in tile_mn_vals if n > 128 and n % 128 == 0]
    cluster = [(1, 2), (2, 1)]

    # Epilogue N granularity lever: None = heuristic default (~32-wide sub-tiles, many
    # CTA barriers); larger values collapse sub-tiles -> fewer barriers. Only offer
    # divisors of tile_n and cap at 128 so the epilogue smem buffer never starves the
    # AB pipeline stages (128*128*2B per stage stays within budget for every config).
    # The M axis is intentionally left alone: widening epi_tile_m to a single M pass needs
    # a 64KB full-tile buffer that drops AB stages to 1 (measured ~2x slower), so it is not
    # swept — epi_tile_n at the 4-barrier point (epi_tile_n=tile_n) is the sweet spot.
    def _epi_tile_n_choices(tile_n: int) -> List[Optional[int]]:
        return [None] + [v for v in (64, 128) if tile_n % v == 0 and v <= tile_n]

    return [
        GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            epi_tile_n=epi_tile_n,
            pingpong=False,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            swap_ab=False,
            device_capacity=9,
            is_dynamic_persistent=False,
            use_tma_gather=False,
        )
        for (tile_m, tile_n), (cluster_m, cluster_n) in itertools.product(tile_mn_vals, cluster)
        for epi_tile_n in _epi_tile_n_choices(tile_n)
    ]


def _get_sm80_configs() -> List[GemmConfig]:
    tile_mn_warps_vals = [
        (128, 128, 4),
        (128, 128, 8),
        (128, 160, 4),
        # TODO: Make 128x160 work with 8 warps. It currently makes the accumulator
        # N layout odd and fails epilogue retile.
        (128, 192, 4),
        (128, 192, 8),
        (128, 256, 8),
        (128, 64, 4),
        (64, 128, 4),
    ]
    return [
        GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            num_warps=num_warps,
            pingpong=False,
            cluster_m=1,
            cluster_n=1,
            swap_ab=swap_ab,
            device_capacity=8,
            is_dynamic_persistent=False,
            use_tma_gather=False,
        )
        for (tile_m, tile_n, num_warps), tile_k, swap_ab in itertools.product(
            tile_mn_warps_vals, [32, 64], [False, True]
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
        + [(256, 512, (2, 1))]
    )
    swap_ab_vals = [False, True]
    if epilogue in ["lse", "gated"]:
        swap_ab_vals = [False]
    GemmConfigCls = partial(
        GemmConfig, pingpong=False, device_capacity=10
    )  # There's no pingpong on Sm100
    use_clc_vals = [True, False]
    use_tma_gather_vals = [True, False]
    return [
        GemmConfigCls(
            tile_m=m,
            tile_n=n,
            cluster_m=cm,
            cluster_n=cn,
            swap_ab=sab,
            max_swizzle_size=8,
            is_dynamic_persistent=use_clc,
            use_tma_gather=use_tma_gather,
        )
        for (m, n, (cm, cn)), sab, use_clc, use_tma_gather in itertools.product(
            tile_mn_cluster_vals, swap_ab_vals, use_clc_vals, use_tma_gather_vals
        )
    ]


def _get_sm120_configs(
    epilogue: Optional[str] = None,
    tune_coop: bool = True,
) -> List[GemmConfig]:
    tile_mn_vals_coop = [(128, 128), (128, 64), (64, 128), (128, 160), (128, 192)]
    tile_mn_vals_pingpong = [(128, 128), (128, 64), (64, 128), (128, 160)]
    tile_mn_vals = []
    if tune_coop:
        tile_mn_vals += [(m, n, False) for m, n in tile_mn_vals_coop]
    tile_mn_vals += [(m, n, True) for m, n in tile_mn_vals_pingpong]
    swap_ab_vals = [False, True]
    if epilogue in ["lse", "gated"]:
        swap_ab_vals = [False]
    return [
        GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            pingpong=pingpong,
            cluster_m=1,
            cluster_n=1,
            swap_ab=swap_ab,
            device_capacity=12,
            is_dynamic_persistent=True,
            use_tma_gather=False,  # TMA gather not supported on SM120
        )
        for (tile_m, tile_n, pingpong), swap_ab in itertools.product(tile_mn_vals, swap_ab_vals)
    ]


def get_all_configs(
    epilogue: Optional[str] = None,
    tune_coop: bool = True,
) -> List[GemmConfig]:
    """Return autotuning configs for all supported device capabilities.

    Each GemmConfig is tagged with its target device_capacity, so the caller can
    filter at runtime based on the actual device. This avoids querying the device
    (and initializing a CUDA context) at import time.
    """
    return (
        _get_sm80_configs()
        + _get_sm90_configs(epilogue, tune_coop)
        + _get_sm100_configs(epilogue)
        + _get_sm120_configs(epilogue, tune_coop)
    )
