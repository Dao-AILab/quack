# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""SM120 MXF4NVF4 warp-GEMM helper API."""

from typing import Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import const_expr
from cutlass._mlir import ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import nvvm as _nvvm_ir
from cutlass.base_dsl.typing import Numeric
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import dsl_user_op, for_generate, yield_out
from cutlass.utils import blockscaled_layout
from cutlass.utils.blackwell_helpers import (
    get_layoutSFA_TV,
    partition_fragment_SFA,
    partition_fragment_SFB,
    thrfrg_SFA,
    thrfrg_SFB,
)
from cutlass.utils.smem_allocator import SmemAllocator
from cutlass.utils.static_persistent_tile_scheduler import (
    PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
)
from quack.sm120_pipeline import PipelineTmaWarpMma


MXF4NVF4_CTA_SHAPE_MNK = (128, 128, 128)
MXF4NVF4_MMA_SHAPE_MNK = (16, 8, 64)
MXF4NVF4_SCALE_VEC_SIZE = 16
MXF4NVF4_SCALE_K = MXF4NVF4_CTA_SHAPE_MNK[2] // MXF4NVF4_SCALE_VEC_SIZE
MXF4NVF4_SCALE_TMA_MIN_L = 2
MXF4NVF4_AB_PACKED_TMA_BYTES = MXF4NVF4_CTA_SHAPE_MNK[0] * MXF4NVF4_CTA_SHAPE_MNK[2] // 2
MXF4NVF4_AB_UNPACK_TMA_BYTES = MXF4NVF4_AB_PACKED_TMA_BYTES
MXF4NVF4_AB_TMA_BYTES = MXF4NVF4_AB_PACKED_TMA_BYTES
MXF4NVF4_AB_SMEM_BYTES = MXF4NVF4_CTA_SHAPE_MNK[0] * MXF4NVF4_CTA_SHAPE_MNK[2]
MXF4NVF4_SCALE_TMA_BYTES = MXF4NVF4_CTA_SHAPE_MNK[0] * MXF4NVF4_SCALE_K
MXF4NVF4_FULL_TMA_BYTES = 2 * MXF4NVF4_AB_TMA_BYTES + 2 * MXF4NVF4_SCALE_TMA_BYTES
MXF4NVF4_FULL_UNPACK_TMA_BYTES = MXF4NVF4_FULL_TMA_BYTES
MXF4NVF4_COOPERATIVE_PRODUCER_REGS = 40
MXF4NVF4_COOPERATIVE_CONSUMER_REGS = 232
MXF4NVF4_COOPERATIVE_THREADS_PER_WARPGROUP = 128
MXF4NVF4_PINGPONG_MMA_BARRIER_ID = 3
MXF4NVF4_PINGPONG_EPI_BARRIER_ID = 5
MXF4NVF4_PINGPONG_BARRIER_THREADS = 2 * MXF4NVF4_COOPERATIVE_THREADS_PER_WARPGROUP


def _check_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"`{name}` must be positive, but got {value}")


def _check_default_tile(tile_mn: int, tile_k: int, sf_vec_size: int) -> None:
    _check_positive("tile_mn", tile_mn)
    _check_positive("tile_k", tile_k)
    _check_positive("sf_vec_size", sf_vec_size)
    if tile_k != 128:
        raise ValueError("SM120 MXF4NVF4 helpers currently support tile_k=128")
    if sf_vec_size != MXF4NVF4_SCALE_VEC_SIZE:
        raise ValueError("SM120 MXF4NVF4 helpers currently support sf_vec_size=16")


def _normalize_mxf4nvf4_ab_smem_format(smem_format: str) -> str:
    normalized = smem_format.replace("-", "_").lower()
    if normalized in ("packed", "align8", "16u4_align8b"):
        return "packed"
    if normalized in (
        "unpack",
        "unpacked",
        "unpack_smem",
        "unpacksmem",
        "align16",
        "16u4_align16b",
    ):
        return "unpack"
    raise ValueError(
        "`smem_format` must be 'packed'/'16u4_align8b' or "
        f"'unpack'/'16u4_align16b', but got {smem_format!r}"
    )


def _mxf4nvf4_ab_tma_internal_type(smem_format: str) -> Optional[Type[Numeric]]:
    if _normalize_mxf4nvf4_ab_smem_format(smem_format) == "unpack":
        return cutlass.Uint8
    return None


def _require_zero_major_offset(name: str, value: cutlass.Int32 | int) -> None:
    raw_value = getattr(value, "value", value)
    if raw_value != 0:
        raise ValueError(
            f"`{name}` is not supported by this helper; encode the global major "
            "tile in the TMA descriptor coordinates and stage the local 128-major tile"
        )


def _require_zero_scale_major_offset(name: str, value: cutlass.Int32 | int) -> None:
    _require_zero_major_offset(name, value)


def _check_tuple(name: str, value: tuple[int, ...], rank: int) -> None:
    if len(value) != rank:
        raise ValueError(f"`{name}` must have rank {rank}, but got {value}")


def _mxf4nvf4_contiguous_alignment(dtype: Type[Numeric]) -> int:
    return 16 * 8 // dtype.width


def _mxf4nvf4_gemm_config_errors(
    *,
    m: int = 128,
    n: int = 128,
    k: int = 128,
    l_extent: int = 1,
    a_dtype: Type[Numeric] = cutlass.Float4E2M1FN,
    b_dtype: Type[Numeric] = cutlass.Float4E2M1FN,
    sf_dtype: Type[Numeric] = cutlass.Float8E4M3FN,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    c_dtype: Type[Numeric] = cutlass.BFloat16,
    acc_dtype: Type[Numeric] = cutlass.Float32,
    tile_shape_mnk: tuple[int, int, int] = MXF4NVF4_CTA_SHAPE_MNK,
    cluster_shape_mnk: tuple[int, int, int] = (1, 1, 1),
    a_major: str = "k",
    b_major: str = "k",
    c_major: str = "n",
    ab_smem_format: str = "packed",
) -> list[str]:
    errors: list[str] = []
    for name, value in (("m", m), ("n", n), ("k", k), ("l_extent", l_extent)):
        if value <= 0:
            errors.append(f"`{name}` must be positive")

    try:
        _check_tuple("tile_shape_mnk", tile_shape_mnk, 3)
    except ValueError as exc:
        errors.append(str(exc))
    try:
        _check_tuple("cluster_shape_mnk", cluster_shape_mnk, 3)
    except ValueError as exc:
        errors.append(str(exc))

    if a_dtype != cutlass.Float4E2M1FN:
        errors.append("A dtype must be Float4E2M1FN")
    if b_dtype != cutlass.Float4E2M1FN:
        errors.append("B dtype must be Float4E2M1FN")
    if sf_dtype != cutlass.Float8E4M3FN:
        errors.append("scale dtype must be Float8E4M3FN")
    if sf_vec_size != MXF4NVF4_SCALE_VEC_SIZE:
        errors.append(f"sf_vec_size must be {MXF4NVF4_SCALE_VEC_SIZE}")
    if acc_dtype != cutlass.Float32:
        errors.append("accumulator dtype must be Float32")
    if c_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.BFloat16}:
        errors.append("output dtype must be Float32, Float16, or BFloat16")

    if tile_shape_mnk != MXF4NVF4_CTA_SHAPE_MNK:
        errors.append(f"tile_shape_mnk must be {MXF4NVF4_CTA_SHAPE_MNK}")
    if cluster_shape_mnk != (1, 1, 1):
        errors.append("cluster_shape_mnk must be (1, 1, 1)")
    if a_major != "k":
        errors.append("A layout must be K-major")
    if b_major != "k":
        errors.append("B layout must be K-major")
    if c_major not in {"n", "m"}:
        errors.append("output layout must be N-major or M-major")

    try:
        normalized_ab_smem_format = _normalize_mxf4nvf4_ab_smem_format(ab_smem_format)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if normalized_ab_smem_format != "packed":
            errors.append("native SM120 MXF4NVF4 GEMM currently supports only packed A/B TMA")

    if len(tile_shape_mnk) == 3:
        tile_m, tile_n, tile_k = tile_shape_mnk
        if m % tile_m != 0:
            errors.append("m must be divisible by tile_shape_mnk[0]")
        if n % tile_n != 0:
            errors.append("n must be divisible by tile_shape_mnk[1]")
        if k % tile_k != 0:
            errors.append("k must be divisible by tile_shape_mnk[2]")

    if a_dtype == cutlass.Float4E2M1FN and k % _mxf4nvf4_contiguous_alignment(a_dtype):
        errors.append("K-major A requires k to be 16-byte aligned")
    if b_dtype == cutlass.Float4E2M1FN and k % _mxf4nvf4_contiguous_alignment(b_dtype):
        errors.append("K-major B requires k to be 16-byte aligned")
    if c_dtype in {cutlass.Float32, cutlass.Float16, cutlass.BFloat16}:
        c_contiguous_extent = m if c_major == "m" else n
        if c_contiguous_extent % _mxf4nvf4_contiguous_alignment(c_dtype):
            errors.append("output contiguous dimension must be 16-byte aligned")

    return errors


def mxf4nvf4_can_implement(
    *,
    m: int = 128,
    n: int = 128,
    k: int = 128,
    l_extent: int = 1,
    a_dtype: Type[Numeric] = cutlass.Float4E2M1FN,
    b_dtype: Type[Numeric] = cutlass.Float4E2M1FN,
    sf_dtype: Type[Numeric] = cutlass.Float8E4M3FN,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    c_dtype: Type[Numeric] = cutlass.BFloat16,
    acc_dtype: Type[Numeric] = cutlass.Float32,
    tile_shape_mnk: tuple[int, int, int] = MXF4NVF4_CTA_SHAPE_MNK,
    cluster_shape_mnk: tuple[int, int, int] = (1, 1, 1),
    a_major: str = "k",
    b_major: str = "k",
    c_major: str = "n",
    ab_smem_format: str = "packed",
) -> bool:
    """Return whether the public packed native-TMA SM120 NVFP4 path supports a GEMM.

    This is a conservative public contract for the currently validated SM120
    Blackwell GeForce path. It describes the supported building block for
    downstream kernels instead of implying that experimental descriptor or
    unpack-SMEM probes are production-supported.
    """
    return not _mxf4nvf4_gemm_config_errors(
        m=m,
        n=n,
        k=k,
        l_extent=l_extent,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        tile_shape_mnk=tile_shape_mnk,
        cluster_shape_mnk=cluster_shape_mnk,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_smem_format=ab_smem_format,
    )


def validate_mxf4nvf4_gemm_config(
    *,
    m: int = 128,
    n: int = 128,
    k: int = 128,
    l_extent: int = 1,
    a_dtype: Type[Numeric] = cutlass.Float4E2M1FN,
    b_dtype: Type[Numeric] = cutlass.Float4E2M1FN,
    sf_dtype: Type[Numeric] = cutlass.Float8E4M3FN,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    c_dtype: Type[Numeric] = cutlass.BFloat16,
    acc_dtype: Type[Numeric] = cutlass.Float32,
    tile_shape_mnk: tuple[int, int, int] = MXF4NVF4_CTA_SHAPE_MNK,
    cluster_shape_mnk: tuple[int, int, int] = (1, 1, 1),
    a_major: str = "k",
    b_major: str = "k",
    c_major: str = "n",
    ab_smem_format: str = "packed",
) -> None:
    """Raise if a GEMM is outside the public SM120 NVFP4 native-TMA contract."""
    errors = _mxf4nvf4_gemm_config_errors(
        m=m,
        n=n,
        k=k,
        l_extent=l_extent,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        tile_shape_mnk=tile_shape_mnk,
        cluster_shape_mnk=cluster_shape_mnk,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_smem_format=ab_smem_format,
    )
    if errors:
        raise ValueError("Unsupported SM120 MXF4NVF4 GEMM configuration: " + "; ".join(errors))


def mxf4nvf4_native_tma_tile_coords(
    m_tile: cutlass.Int32 | int = 0,
    n_tile: cutlass.Int32 | int = 0,
    k_tile: cutlass.Int32 | int = 0,
    l_tile: cutlass.Int32 | int = 0,
) -> dict[str, tuple[cutlass.Int32 | int, ...]]:
    """Map one GEMM tile coordinate to native SM120 A/B/SFA/SFB TMA coords."""
    scale_k_tile = k_tile % 2
    scale_page = k_tile // 2
    return {
        "ab_tile_coord_a": (m_tile, k_tile, l_tile),
        "ab_tile_coord_b": (n_tile, k_tile, l_tile),
        "scale_tile_coord_sfa": (m_tile, scale_k_tile, scale_page, l_tile),
        "scale_tile_coord_sfb": (n_tile, scale_k_tile, scale_page, l_tile),
    }


def mxf4nvf4_scheduler_tile_tma_coords(
    tile_mnl: tuple[cutlass.Int32 | int, ...],
    k_tile: cutlass.Int32 | int = 0,
) -> dict[str, tuple[cutlass.Int32 | int, ...]]:
    """Map a persistent scheduler tile coordinate to native SM120 TMA coords."""
    _check_tuple("tile_mnl", tile_mnl, 3)
    return mxf4nvf4_native_tma_tile_coords(
        tile_mnl[0],
        tile_mnl[1],
        k_tile,
        tile_mnl[2],
    )


def mxf4nvf4_tiled_problem_shape(
    *,
    m: int = 128,
    n: int = 128,
    k: int = 128,
    l_extent: int = 1,
    a_dtype: Type[Numeric] = cutlass.Float4E2M1FN,
    b_dtype: Type[Numeric] = cutlass.Float4E2M1FN,
    sf_dtype: Type[Numeric] = cutlass.Float8E4M3FN,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    c_dtype: Type[Numeric] = cutlass.BFloat16,
    acc_dtype: Type[Numeric] = cutlass.Float32,
    tile_shape_mnk: tuple[int, int, int] = MXF4NVF4_CTA_SHAPE_MNK,
    cluster_shape_mnk: tuple[int, int, int] = (1, 1, 1),
    a_major: str = "k",
    b_major: str = "k",
    c_major: str = "n",
    ab_smem_format: str = "packed",
) -> dict[str, tuple[int, ...] | int]:
    """Return host-side tiling metadata for the public SM120 NVFP4 path."""
    validate_mxf4nvf4_gemm_config(
        m=m,
        n=n,
        k=k,
        l_extent=l_extent,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        tile_shape_mnk=tile_shape_mnk,
        cluster_shape_mnk=cluster_shape_mnk,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_smem_format=ab_smem_format,
    )

    tile_m, tile_n, tile_k = tile_shape_mnk
    num_ctas_mnl = (m // tile_m, n // tile_n, l_extent)
    cluster_shape_mnl = (cluster_shape_mnk[0], cluster_shape_mnk[1], 1)
    return {
        "problem_shape_mnkl": (m, n, k, l_extent),
        "tile_shape_mnk": tile_shape_mnk,
        "cluster_shape_mnk": cluster_shape_mnk,
        "cluster_shape_mnl": cluster_shape_mnl,
        "num_ctas_mnl": num_ctas_mnl,
        "k_tile_count": k // tile_k,
        "logical_grid_shape": num_ctas_mnl,
    }


@dsl_user_op
def make_mxf4nvf4_static_tile_scheduler_params(
    *,
    m: int = 128,
    n: int = 128,
    k: int = 128,
    l_extent: int = 1,
    max_active_clusters: int = 1,
    swizzle_size: int = 1,
    raster_along_m: bool = True,
    a_dtype: Type[Numeric] = cutlass.Float4E2M1FN,
    b_dtype: Type[Numeric] = cutlass.Float4E2M1FN,
    sf_dtype: Type[Numeric] = cutlass.Float8E4M3FN,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    c_dtype: Type[Numeric] = cutlass.BFloat16,
    acc_dtype: Type[Numeric] = cutlass.Float32,
    tile_shape_mnk: tuple[int, int, int] = MXF4NVF4_CTA_SHAPE_MNK,
    cluster_shape_mnk: tuple[int, int, int] = (1, 1, 1),
    a_major: str = "k",
    b_major: str = "k",
    c_major: str = "n",
    ab_smem_format: str = "packed",
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> tuple[PersistentTileSchedulerParams, Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]]:
    """Return static persistent scheduler params and launch grid for SM120 NVFP4."""
    problem_shape = mxf4nvf4_tiled_problem_shape(
        m=m,
        n=n,
        k=k,
        l_extent=l_extent,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        tile_shape_mnk=tile_shape_mnk,
        cluster_shape_mnk=cluster_shape_mnk,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_smem_format=ab_smem_format,
    )
    tile_sched_params = PersistentTileSchedulerParams(
        problem_shape["num_ctas_mnl"],
        problem_shape["cluster_shape_mnl"],
        swizzle_size=swizzle_size,
        raster_along_m=raster_along_m,
        loc=loc,
        ip=ip,
    )
    grid = StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params,
        cutlass.Int32(max_active_clusters),
        loc=loc,
        ip=ip,
    )
    return tile_sched_params, grid


@dsl_user_op
def make_mxf4nvf4_static_tile_scheduler(
    tile_sched_params: PersistentTileSchedulerParams,
    block_idx: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    grid_dim: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> StaticPersistentTileScheduler:
    """Create the static persistent tile scheduler for an SM120 NVFP4 kernel."""
    return StaticPersistentTileScheduler.create(
        tile_sched_params,
        block_idx,
        grid_dim,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_a_gmem_layout(
    m: int = 128,
    k: int = 128,
    l_extent: int = 1,
    major: str = "k",
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """Return the public logical A GMEM layout for the SM120 NVFP4 path."""
    _check_positive("m", m)
    _check_positive("k", k)
    _check_positive("l_extent", l_extent)
    if major != "k":
        raise ValueError("SM120 MXF4NVF4 A GMEM layout currently requires major='k'")
    return cute.make_layout(
        (m, k, l_extent),
        stride=(k, 1, m * k),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_b_gmem_layout(
    n: int = 128,
    k: int = 128,
    l_extent: int = 1,
    major: str = "k",
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """Return the public logical B GMEM layout for the SM120 NVFP4 path."""
    _check_positive("n", n)
    _check_positive("k", k)
    _check_positive("l_extent", l_extent)
    if major != "k":
        raise ValueError("SM120 MXF4NVF4 B GMEM layout currently requires major='k'")
    return cute.make_layout(
        (n, k, l_extent),
        stride=(k, 1, n * k),
        loc=loc,
        ip=ip,
    )


def _preserve_mxf4nvf4_ab_tma_l_mode(gmem_tensor: cute.Tensor) -> cute.Tensor:
    """Keep A/B tensor maps rank-3 even for logical L=1.

    This mirrors the 79a C++ path, which builds A/B tensor maps over
    ``(M,K,L)`` / ``(N,K,L)`` and keeps the L coordinate in the TMA
    instruction stream.
    """
    if const_expr(cute.size(gmem_tensor, mode=[2]) != 1):
        return gmem_tensor
    return cute.make_tensor(
        gmem_tensor.iterator,
        cute.make_layout(
            (gmem_tensor.shape[0], gmem_tensor.shape[1], MXF4NVF4_SCALE_TMA_MIN_L),
            stride=gmem_tensor.layout.stride,
        ),
    )


@dsl_user_op
def make_mxf4nvf4_d_gmem_layout(
    m: int = 128,
    n: int = 128,
    l_extent: int = 1,
    major: str = "n",
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """Return the public logical D GMEM layout for the SM120 NVFP4 path."""
    _check_positive("m", m)
    _check_positive("n", n)
    _check_positive("l_extent", l_extent)
    if major == "n":
        stride = (n, 1, m * n)
    elif major == "m":
        stride = (1, m, m * n)
    else:
        raise ValueError("SM120 MXF4NVF4 D GMEM layout requires major='n' or 'm'")
    return cute.make_layout(
        (m, n, l_extent),
        stride=stride,
        loc=loc,
        ip=ip,
    )


def mxf4nvf4_ab_tma_tx_bytes(
    tile_mn: int = 128,
    tile_k: int = 128,
    *,
    smem_format: str = "packed",
) -> int:
    """Return bytes completed by one A or B full-tile TMA transaction.

    The unpack-SMEM FP4 tensor-map format expands the destination SMEM footprint
    to 16 KiB for a 128x128 tile, but the transaction barrier count follows the
    logical packed FP4 payload bytes, matching the SM120 C++ collectives.
    """
    _check_default_tile(tile_mn, tile_k, MXF4NVF4_SCALE_VEC_SIZE)
    _normalize_mxf4nvf4_ab_smem_format(smem_format)
    return tile_mn * tile_k // 2


def mxf4nvf4_ab_packed_tma_tx_bytes(tile_mn: int = 128, tile_k: int = 128) -> int:
    """Return A/B TMA bytes for the normal packed FP4 ALIGN8B format."""
    return mxf4nvf4_ab_tma_tx_bytes(tile_mn, tile_k, smem_format="packed")


def mxf4nvf4_ab_unpack_tma_tx_bytes(tile_mn: int = 128, tile_k: int = 128) -> int:
    """Return A/B barrier bytes for the FP4 unpack-SMEM ALIGN16B format."""
    return mxf4nvf4_ab_tma_tx_bytes(tile_mn, tile_k, smem_format="unpack")


def mxf4nvf4_ab_physical_smem_bytes(tile_mn: int = 128, tile_k: int = 128) -> int:
    """Return bytes reserved for one A or B physical-SMEM tile."""
    _check_default_tile(tile_mn, tile_k, MXF4NVF4_SCALE_VEC_SIZE)
    return tile_mn * tile_k


def mxf4nvf4_scale_tma_tx_bytes(
    tile_mn: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
) -> int:
    """Return bytes completed by one SFA or SFB full-tile TMA transaction."""
    _check_default_tile(tile_mn, tile_k, sf_vec_size)
    return tile_mn * (tile_k // sf_vec_size)


def mxf4nvf4_scale_physical_smem_bytes(
    tile_mn: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
) -> int:
    """Return bytes reserved for one SFA or SFB physical-SMEM tile."""
    _check_default_tile(tile_mn, tile_k, sf_vec_size)
    return max(tile_mn, 128) * (tile_k // sf_vec_size)


def mxf4nvf4_full_tma_tx_bytes(
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    *,
    ab_smem_format: str = "packed",
) -> int:
    """Return the barrier transaction byte count for A/B/SFA/SFB TMA."""
    return (
        mxf4nvf4_ab_tma_tx_bytes(tile_m, tile_k, smem_format=ab_smem_format)
        + mxf4nvf4_ab_tma_tx_bytes(tile_n, tile_k, smem_format=ab_smem_format)
        + mxf4nvf4_scale_tma_tx_bytes(tile_m, tile_k, sf_vec_size)
        + mxf4nvf4_scale_tma_tx_bytes(tile_n, tile_k, sf_vec_size)
    )


def mxf4nvf4_full_packed_tma_tx_bytes(
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
) -> int:
    """Return full-tile TMA bytes for packed FP4 A/B plus SFA/SFB."""
    return mxf4nvf4_full_tma_tx_bytes(
        tile_m,
        tile_n,
        tile_k,
        sf_vec_size,
        ab_smem_format="packed",
    )


def mxf4nvf4_full_unpack_tma_tx_bytes(
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
) -> int:
    """Return full-tile TMA bytes for unpack-SMEM FP4 A/B plus SFA/SFB."""
    return mxf4nvf4_full_tma_tx_bytes(
        tile_m,
        tile_n,
        tile_k,
        sf_vec_size,
        ab_smem_format="unpack",
    )


@dsl_user_op
def make_mxf4nvf4_sfa_gmem_layout(
    m: int = 128,
    k: int = 128,
    l_extent: int = 1,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """Return the C++ 79a-style GMEM layout for SFA scale tensors."""
    _check_positive("m", m)
    _check_positive("k", k)
    _check_positive("l_extent", l_extent)
    _check_positive("sf_vec_size", sf_vec_size)
    if sf_vec_size != MXF4NVF4_SCALE_VEC_SIZE:
        raise ValueError("SM120 MXF4NVF4 scale GMEM layout requires sf_vec_size=16")
    return blockscaled_layout.tile_atom_to_shape_SF(
        (m, k, l_extent),
        sf_vec_size,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_sfb_gmem_layout(
    n: int = 128,
    k: int = 128,
    l_extent: int = 1,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """Return the C++ 79a-style GMEM layout for SFB scale tensors."""
    _check_positive("n", n)
    _check_positive("k", k)
    _check_positive("l_extent", l_extent)
    _check_positive("sf_vec_size", sf_vec_size)
    if sf_vec_size != MXF4NVF4_SCALE_VEC_SIZE:
        raise ValueError("SM120 MXF4NVF4 scale GMEM layout requires sf_vec_size=16")
    return blockscaled_layout.tile_atom_to_shape_SF(
        (n, k, l_extent),
        sf_vec_size,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_scale_tma_physical_gmem_layout(
    major_extent: int = 128,
    scale_k_extent: int = MXF4NVF4_SCALE_K * 2,
    tile_extent: int = 1,
    l_extent: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """Return the compact GMEM layout consumed by native SM120 scale TMA.

    The native scale TMA atom keeps the public scale tensor typed as FP8 and
    exposes a layout with major as the contiguous mode. This is distinct from a
    row-major Torch view of the same storage and from the logical blockscaled
    SFA/SFB layout used by fragments.
    """
    _check_positive("major_extent", major_extent)
    _check_positive("scale_k_extent", scale_k_extent)
    _check_positive("tile_extent", tile_extent)
    _check_positive("l_extent", l_extent)
    return cute.make_layout(
        (major_extent, scale_k_extent, tile_extent, l_extent),
        stride=(
            1,
            major_extent,
            major_extent * scale_k_extent,
            major_extent * scale_k_extent * tile_extent,
        ),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_scale_interleaved_gmem_layout(
    major_extent: int = 128,
    logical_k_extent: int = 128,
    l_extent: int = 1,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """Return the compact 4D FP8 scale layout consumed by SM120 TMA."""
    _check_positive("major_extent", major_extent)
    _check_positive("logical_k_extent", logical_k_extent)
    _check_positive("l_extent", l_extent)
    _check_positive("sf_vec_size", sf_vec_size)
    if sf_vec_size != MXF4NVF4_SCALE_VEC_SIZE:
        raise ValueError("SM120 MXF4NVF4 scale layout requires sf_vec_size=16")
    if major_extent % 128 != 0:
        raise ValueError("SM120 scale interleaved layout requires major_extent % 128 == 0")
    if logical_k_extent % sf_vec_size != 0:
        raise ValueError(
            "SM120 scale interleaved layout requires logical_k_extent % sf_vec_size == 0"
        )
    logical_scale_k = cute.ceil_div(logical_k_extent, sf_vec_size)
    if logical_scale_k % 4 != 0:
        raise ValueError("SM120 scale interleaved layout requires scale_k % 4 == 0")
    major_tiles = major_extent // 128
    scale_tiles = logical_scale_k // 4
    l_stride = major_tiles * scale_tiles * 512
    return cute.make_layout(
        (((32, 4), major_tiles), 4, scale_tiles, l_extent),
        stride=(((16, 4), 512), 1, major_tiles * 512, l_stride),
        loc=loc,
        ip=ip,
    )


def mxf4nvf4_padded_scale_k_extent(logical_scale_k_extent: int) -> int:
    """Return the padded physical scale-K extent for SM120 NVFP4 scale TMA."""
    _check_positive("logical_scale_k_extent", logical_scale_k_extent)
    granularity = MXF4NVF4_SCALE_K * 2
    return ((logical_scale_k_extent + granularity - 1) // granularity) * granularity


def mxf4nvf4_scale_tma_physical_k_extent(
    k: int,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
) -> int:
    """Return the physical scale-K extent needed to back a logical K extent."""
    _check_positive("k", k)
    _check_positive("sf_vec_size", sf_vec_size)
    if sf_vec_size != MXF4NVF4_SCALE_VEC_SIZE:
        raise ValueError(f"SM120 MXF4NVF4 scale TMA requires sf_vec_size={MXF4NVF4_SCALE_VEC_SIZE}")
    if k % sf_vec_size != 0:
        raise ValueError("SM120 MXF4NVF4 K extent must be divisible by sf_vec_size")
    return mxf4nvf4_padded_scale_k_extent(k // sf_vec_size)


def mxf4nvf4_scale_tma_physical_l_extent(logical_l_extent: int) -> int:
    """Return the physical scale-L extent used by native SM120 scale TMA.

    Keeping the scale tensor-map rank-4 even for logical L=1 preserves the
    compact scale TMA path used by the native SM120 MXF4/NVFP4 mainloop.
    """
    _check_positive("logical_l_extent", logical_l_extent)
    return max(logical_l_extent, MXF4NVF4_SCALE_TMA_MIN_L)


def mxf4nvf4_cooperative_launch_kwargs(
    *,
    producer_warpgroups: int = 1,
    consumer_warpgroups: int = 2,
    min_ctas_per_sm: int = 1,
) -> dict[str, tuple[int, int, int] | int]:
    """Return launch metadata required for SM120 dynamic register allocation.

    PTX `setmaxnreg` only lowers to SASS `USETMAXREG` when ptxas sees an entry
    metadata context such as `.maxntid` plus `.minnctapersm`. This helper keeps
    that contract close to the cooperative SM120 schedule shape instead of
    requiring each caller to spell the metadata by hand.
    """
    _check_positive("producer_warpgroups", producer_warpgroups)
    _check_positive("consumer_warpgroups", consumer_warpgroups)
    _check_positive("min_ctas_per_sm", min_ctas_per_sm)
    warpgroups = producer_warpgroups + consumer_warpgroups
    threads = warpgroups * MXF4NVF4_COOPERATIVE_THREADS_PER_WARPGROUP
    return {
        "block": (threads, 1, 1),
        "max_number_threads": (threads, 1, 1),
        "min_blocks_per_mp": min_ctas_per_sm,
    }


def mxf4nvf4_cooperative_sass_count_targets(
    *,
    tma_issue_groups: int = 3,
    consumer_issue_groups: int = 2,
) -> dict[str, int]:
    """Return expected static SASS count targets for SM120 schedule probes."""
    _check_positive("tma_issue_groups", tma_issue_groups)
    _check_positive("consumer_issue_groups", consumer_issue_groups)
    return {
        "USETMAXREG": 2,
        "UTMALDG": 4 * tma_issue_groups,
        "OMMA.SF": 32 * consumer_issue_groups,
        "LDSM": 12 * consumer_issue_groups,
    }


def _mxf4nvf4_pingpong_barrier_base_id(stage: str) -> int:
    if stage == "mma":
        return MXF4NVF4_PINGPONG_MMA_BARRIER_ID
    if stage == "epi":
        return MXF4NVF4_PINGPONG_EPI_BARRIER_ID
    raise ValueError("SM120 ping-pong barrier stage must be 'mma' or 'epi'")


@dsl_user_op
def mxf4nvf4_pingpong_barrier_arrive(
    warpgroup_idx: cutlass.Int32 | int,
    stage: str,
    *,
    number_of_threads: cutlass.Int32 | int = MXF4NVF4_PINGPONG_BARRIER_THREADS,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Arrive on the SM120 two-warpgroup ping-pong named barrier."""
    cute.arch.barrier_arrive(
        barrier_id=_mxf4nvf4_pingpong_barrier_base_id(stage) + warpgroup_idx,
        number_of_threads=number_of_threads,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mxf4nvf4_pingpong_barrier_sync(
    warpgroup_idx: cutlass.Int32 | int,
    stage: str,
    *,
    number_of_threads: cutlass.Int32 | int = MXF4NVF4_PINGPONG_BARRIER_THREADS,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Arrive and wait on the SM120 two-warpgroup ping-pong named barrier."""
    cute.arch.barrier(
        barrier_id=_mxf4nvf4_pingpong_barrier_base_id(stage) + warpgroup_idx,
        number_of_threads=number_of_threads,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mxf4nvf4_mma_warpgroup_barrier_sync(
    *,
    barrier_id: cutlass.Int32 | int = MXF4NVF4_PINGPONG_MMA_BARRIER_ID,
    number_of_threads: cutlass.Int32 | int = MXF4NVF4_PINGPONG_BARRIER_THREADS,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Arrive and wait on the SM120 MXF4/NVFP4 MMA warpgroup barrier."""
    cute.arch.barrier(
        barrier_id=barrier_id,
        number_of_threads=number_of_threads,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mxf4nvf4_split_tma_consumer_wait(
    pipe_mk: PipelineTmaWarpMma,
    pipe_nk: PipelineTmaWarpMma,
    consumer_state_mk: pipeline.PipelineState,
    consumer_state_nk: pipeline.PipelineState,
    *,
    join_split_tma_barrier: bool = True,
    try_wait_token_mk: Optional[cutlass.Boolean] = None,
    try_wait_token_nk: Optional[cutlass.Boolean] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Wait for the SM120 split-TMA MK/NK consumer stages.

    When ``join_split_tma_barrier`` is true, the MK and NK TMA streams share one
    full barrier and only the MK pipe is waited on.  Callers may pass tokens
    from an earlier ``consumer_try_wait`` to separate mbarrier probing from the
    actual wait, matching the 79a-style ping-pong handoff.
    """
    if const_expr(try_wait_token_mk is None):
        try_wait_token_mk = pipe_mk.consumer_try_wait(consumer_state_mk, loc=loc, ip=ip)
    pipe_mk.consumer_wait(
        consumer_state_mk,
        try_wait_token_mk,
        loc=loc,
        ip=ip,
    )
    if const_expr(not join_split_tma_barrier):
        if const_expr(try_wait_token_nk is None):
            try_wait_token_nk = pipe_nk.consumer_try_wait(consumer_state_nk, loc=loc, ip=ip)
        pipe_nk.consumer_wait(
            consumer_state_nk,
            try_wait_token_nk,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def mxf4nvf4_split_tma_consumer_release(
    pipe_mk: PipelineTmaWarpMma,
    pipe_nk: PipelineTmaWarpMma,
    consumer_state_mk: pipeline.PipelineState,
    consumer_state_nk: pipeline.PipelineState,
    *,
    join_split_tma_barrier: bool = True,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Release the SM120 split-TMA MK/NK consumer stages."""
    pipe_mk.consumer_release(consumer_state_mk, loc=loc, ip=ip)
    if const_expr(not join_split_tma_barrier):
        pipe_nk.consumer_release(consumer_state_nk, loc=loc, ip=ip)


def make_mxf4nvf4_native_tma_pipeline(
    barrier_storage: cute.Tensor,
    *,
    num_stages: int = 1,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    ab_smem_format: str = "packed",
    producer_group=None,
    consumer_group=None,
):
    """Create the SM120 native A/B/SFA/SFB TMA load pipeline."""
    _check_positive("num_stages", num_stages)
    if producer_group is None:
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    if consumer_group is None:
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
    return PipelineTmaWarpMma.create(
        num_stages=num_stages,
        producer_group=producer_group,
        consumer_group=consumer_group,
        tx_count=mxf4nvf4_full_tma_tx_bytes(
            tile_m,
            tile_n,
            tile_k,
            sf_vec_size,
            ab_smem_format=ab_smem_format,
        ),
        barrier_storage=barrier_storage,
    )


@dsl_user_op
def producer_acquire_native_tma_already_elected(
    pipe: PipelineTmaWarpMma,
    state: pipeline.PipelineState,
    try_acquire_token: Optional[cutlass.Boolean] = None,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Acquire a native-TMA pipeline stage from inside an elected producer lane."""
    pipe.producer_acquire_already_elected(
        state,
        try_acquire_token,
        loc=loc,
        ip=ip,
    )


def _as_i32_ir_value(value, *, loc=None, ip=None):
    if hasattr(value, "ir_value"):
        return cutlass.Int32(value).ir_value(loc=loc, ip=ip)
    return cutlass.Int32(value).ir_value(loc=loc, ip=ip)


def _flatten_coord_values(coord) -> list:
    if isinstance(coord, tuple):
        values = []
        for item in coord:
            values.extend(_flatten_coord_values(item))
        return values
    return [coord]


@dsl_user_op
def _issue_native_tma_load_already_elected(
    tma_atom: cute.CopyAtom,
    src: cute.Tensor,
    dst: cute.Tensor,
    tma_bar_ptr: cute.Pointer,
    *,
    cache_policy: Optional[cutlass.Int64] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one non-multicast native TMA load from an already elected lane."""
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    tma_desc_ptr_type = ir.Type.parse(
        "!cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<128>>"
    )
    tma_desc_ptr = _cute_nvgpu_ir.get_tma_desc_addr(tma_desc_ptr_type, exec_atom, loc=loc, ip=ip)
    coords = [
        _as_i32_ir_value(coord, loc=loc, ip=ip) for coord in _flatten_coord_values(src.iterator)
    ]
    _nvvm_ir.CpAsyncBulkTensorGlobalToSharedCTAOp(
        dstMem=dst.iterator.llvm_ptr,
        tmaDescriptor=tma_desc_ptr.llvm_ptr,
        mbar=tma_bar_ptr.llvm_ptr,
        coordinates=coords,
        l2CacheHint=cache_policy.ir_value(loc=loc, ip=ip) if cache_policy is not None else None,
        loc=loc,
        ip=ip,
    )


class Mxf4Nvf4CooperativeSchedule:
    """Composable SM120 MXF4NVF4 cooperative schedule contract.

    This is intentionally small: it owns the launch metadata and dynamic
    register-allocation role split that are easy to get wrong, while leaving
    descriptor routing, pipelines, fragment movement, MMA issue order, and
    epilogue composition to the caller.
    """

    def __init__(
        self,
        *,
        producer_warpgroups: int = 1,
        consumer_warpgroups: int = 2,
        producer_warpgroup_start: int | None = None,
        regs_producer: int = MXF4NVF4_COOPERATIVE_PRODUCER_REGS,
        regs_consumer: int = MXF4NVF4_COOPERATIVE_CONSUMER_REGS,
        min_ctas_per_sm: int = 1,
        tma_issue_groups: int = 3,
        consumer_issue_groups: int = 2,
    ) -> None:
        _check_positive("producer_warpgroups", producer_warpgroups)
        _check_positive("consumer_warpgroups", consumer_warpgroups)
        _check_positive("regs_producer", regs_producer)
        _check_positive("regs_consumer", regs_consumer)
        _check_positive("min_ctas_per_sm", min_ctas_per_sm)
        _check_positive("tma_issue_groups", tma_issue_groups)
        _check_positive("consumer_issue_groups", consumer_issue_groups)
        if producer_warpgroup_start is None:
            producer_warpgroup_start = consumer_warpgroups
        if producer_warpgroup_start < 0:
            raise ValueError("`producer_warpgroup_start` must be non-negative")
        self.producer_warpgroups = producer_warpgroups
        self.consumer_warpgroups = consumer_warpgroups
        self.producer_warpgroup_start = producer_warpgroup_start
        self.regs_producer = regs_producer
        self.regs_consumer = regs_consumer
        self.min_ctas_per_sm = min_ctas_per_sm
        self.tma_issue_groups = tma_issue_groups
        self.consumer_issue_groups = consumer_issue_groups

    @property
    def threads_per_cta(self) -> int:
        warpgroups = self.producer_warpgroups + self.consumer_warpgroups
        return warpgroups * MXF4NVF4_COOPERATIVE_THREADS_PER_WARPGROUP

    @property
    def warps_per_warpgroup(self) -> int:
        return MXF4NVF4_COOPERATIVE_THREADS_PER_WARPGROUP // 32

    @property
    def producer_warps(self) -> int:
        return self.producer_warpgroups * self.warps_per_warpgroup

    @property
    def consumer_warps(self) -> int:
        return self.consumer_warpgroups * self.warps_per_warpgroup

    @property
    def consumer_warp_start(self) -> int:
        return 0

    @property
    def consumer_warp_end(self) -> int:
        return self.consumer_warp_start + self.consumer_warps

    @property
    def producer_warp_start(self) -> int:
        return self.producer_warpgroup_start * self.warps_per_warpgroup

    @property
    def producer_warp_end(self) -> int:
        return self.producer_warp_start + self.producer_warps

    @property
    def producer_issue_warp(self) -> int:
        return self.producer_warp_start

    def is_consumer_warp(self, warp_idx: cutlass.Int32 | int) -> cutlass.Boolean:
        return cutlass.Boolean(warp_idx >= self.consumer_warp_start) & cutlass.Boolean(
            warp_idx < self.consumer_warp_end
        )

    def is_producer_warp(self, warp_idx: cutlass.Int32 | int) -> cutlass.Boolean:
        return cutlass.Boolean(warp_idx >= self.producer_warp_start) & cutlass.Boolean(
            warp_idx < self.producer_warp_end
        )

    def is_producer_issue_warp(self, warp_idx: cutlass.Int32 | int) -> cutlass.Boolean:
        return cutlass.Boolean(warp_idx == self.producer_issue_warp)

    def launch_kwargs(self) -> dict[str, tuple[int, int, int] | int]:
        return mxf4nvf4_cooperative_launch_kwargs(
            producer_warpgroups=self.producer_warpgroups,
            consumer_warpgroups=self.consumer_warpgroups,
            min_ctas_per_sm=self.min_ctas_per_sm,
        )

    def sass_count_targets(self) -> dict[str, int]:
        return mxf4nvf4_cooperative_sass_count_targets(
            tma_issue_groups=self.tma_issue_groups,
            consumer_issue_groups=self.consumer_issue_groups,
        )

    def setmaxregister_role(self, warp_idx: cutlass.Int32) -> None:
        setmaxregister_mxf4nvf4_cooperative_role(
            warp_idx,
            producer_warpgroup_start=self.producer_warpgroup_start,
            producer_warpgroups=self.producer_warpgroups,
            regs_producer=self.regs_producer,
            regs_consumer=self.regs_consumer,
        )

    def setmaxregister_producer(self) -> None:
        setmaxregister_mxf4nvf4_producer(self.regs_producer)

    def setmaxregister_consumer(self) -> None:
        setmaxregister_mxf4nvf4_consumer(self.regs_consumer)


def make_mxf4nvf4_cooperative_schedule(
    **kwargs,
) -> Mxf4Nvf4CooperativeSchedule:
    """Create an SM120 MXF4NVF4 cooperative schedule facade."""
    return Mxf4Nvf4CooperativeSchedule(**kwargs)


@dsl_user_op
def setmaxregister_mxf4nvf4_producer(
    regs_producer: int = MXF4NVF4_COOPERATIVE_PRODUCER_REGS,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Apply the SM120 MXF4NVF4 producer-side register deallocation."""
    _check_positive("regs_producer", regs_producer)
    cute.arch.setmaxregister_decrease(regs_producer, loc=loc, ip=ip)


@dsl_user_op
def setmaxregister_mxf4nvf4_consumer(
    regs_consumer: int = MXF4NVF4_COOPERATIVE_CONSUMER_REGS,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Apply the SM120 MXF4NVF4 consumer-side register allocation."""
    _check_positive("regs_consumer", regs_consumer)
    cute.arch.setmaxregister_increase(regs_consumer, loc=loc, ip=ip)


@cute.jit(preprocess=True)
def setmaxregister_mxf4nvf4_cooperative_role(
    warp_idx: cutlass.Int32,
    producer_warpgroup_start: int = 0,
    producer_warpgroups: int = 1,
    regs_producer: int = 40,
    regs_consumer: int = 232,
) -> None:
    """Apply SM120 cooperative producer/consumer dynamic register allocation.

    The role is selected at warpgroup granularity. Callers should launch with
    `max_number_threads` and `min_blocks_per_mp` metadata, for example via
    `mxf4nvf4_cooperative_launch_kwargs()`, otherwise ptxas may keep PTX
    `setmaxnreg` text but omit SASS `USETMAXREG`.
    """
    if const_expr(producer_warpgroups <= 0):
        raise ValueError("`producer_warpgroups` must be positive")
    if const_expr(regs_producer <= 0):
        raise ValueError("`regs_producer` must be positive")
    if const_expr(regs_consumer <= 0):
        raise ValueError("`regs_consumer` must be positive")
    if const_expr(producer_warpgroup_start < 0):
        raise ValueError("`producer_warpgroup_start` must be non-negative")
    warpgroup_idx = cute.arch.make_warp_uniform(warp_idx // 4)
    producer_warpgroup_end = producer_warpgroup_start + producer_warpgroups
    if warpgroup_idx < producer_warpgroup_start:
        cute.arch.setmaxregister_increase(regs_consumer)
    else:
        if warpgroup_idx < producer_warpgroup_end:
            cute.arch.setmaxregister_decrease(regs_producer)
        else:
            cute.arch.setmaxregister_increase(regs_consumer)


@dsl_user_op
def issue_mxf4nvf4_native_tma_stage(
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    tma_tensor_sfa: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    tma_tensor_sfb: cute.Tensor,
    sA: cute.Tensor,
    sB: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    tma_bar_ptr: cute.Pointer,
    stage_idx: cutlass.Int32 | int = 0,
    *,
    batch_idx: cutlass.Int32 | int = 0,
    already_elected: cutlass.Constexpr[bool] = False,
    cache_policy=None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one native TMA stage for A/B and SFA/SFB atoms.

    The TMA tensors are expected to be already tiled/sliced to the CTA tile the
    caller wants to load. This helper owns the repetitive SM120 partition and
    copy plumbing for the descriptor-free atom path.
    """
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1, loc=loc, ip=ip),
        cute.group_modes(sA, 0, 2, loc=loc, ip=ip),
        cute.group_modes(tma_tensor_a, 0, 2, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1, loc=loc, ip=ip),
        cute.group_modes(sB, 0, 2, loc=loc, ip=ip),
        cute.group_modes(tma_tensor_b, 0, 2, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    tSFAs, tSFAg = cpasync.tma_partition(
        tma_atom_sfa,
        0,
        cute.make_layout(1, loc=loc, ip=ip),
        sSFA,
        tma_tensor_sfa,
        loc=loc,
        ip=ip,
    )
    tSFBs, tSFBg = cpasync.tma_partition(
        tma_atom_sfb,
        0,
        cute.make_layout(1, loc=loc, ip=ip),
        sSFB,
        tma_tensor_sfb,
        loc=loc,
        ip=ip,
    )

    if cutlass.const_expr(already_elected):
        _issue_native_tma_load_already_elected(
            tma_atom_a,
            tAgA[(None, batch_idx)],
            tAsA[(None, stage_idx)],
            tma_bar_ptr,
            cache_policy=cache_policy,
            loc=loc,
            ip=ip,
        )
        _issue_native_tma_load_already_elected(
            tma_atom_b,
            tBgB[(None, batch_idx)],
            tBsB[(None, stage_idx)],
            tma_bar_ptr,
            cache_policy=cache_policy,
            loc=loc,
            ip=ip,
        )
        _issue_native_tma_load_already_elected(
            tma_atom_sfa,
            tSFAg[(None, 0, 0, batch_idx)],
            tSFAs[(None, 0, 0, stage_idx)],
            tma_bar_ptr,
            cache_policy=cache_policy,
            loc=loc,
            ip=ip,
        )
        _issue_native_tma_load_already_elected(
            tma_atom_sfb,
            tSFBg[(None, 0, 0, batch_idx)],
            tSFBs[(None, 0, 0, stage_idx)],
            tma_bar_ptr,
            cache_policy=cache_policy,
            loc=loc,
            ip=ip,
        )
        return

    cute.copy(
        tma_atom_a,
        tAgA[(None, batch_idx)],
        tAsA[(None, stage_idx)],
        tma_bar_ptr=tma_bar_ptr,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tma_atom_b,
        tBgB[(None, batch_idx)],
        tBsB[(None, stage_idx)],
        tma_bar_ptr=tma_bar_ptr,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tma_atom_sfa,
        tSFAg[(None, 0, 0, batch_idx)],
        tSFAs[(None, 0, 0, stage_idx)],
        tma_bar_ptr=tma_bar_ptr,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tma_atom_sfb,
        tSFBg[(None, 0, 0, batch_idx)],
        tSFBs[(None, 0, 0, stage_idx)],
        tma_bar_ptr=tma_bar_ptr,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def local_tile_mxf4nvf4_native_tma_tensors(
    tma_tensor_a: cute.Tensor,
    tma_tensor_b: cute.Tensor,
    tma_tensor_sfa: cute.Tensor,
    tma_tensor_sfb: cute.Tensor,
    tile_mnl: tuple[cutlass.Int32 | int, ...],
    k_tile: cutlass.Int32 | int,
    *,
    ab_cta_tiler: cute.Tile = (128, 128, 1),
    scale_cta_tiler: cute.Tile = (128, 8, 1, 1),
    scale_smem_format: str = "physical",
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
    """Local-tile native SM120 TMA tensors for one scheduler work tile."""
    tile_coords = mxf4nvf4_scheduler_tile_tma_coords(tile_mnl, k_tile)
    if scale_smem_format == "interleaved":
        scale_cta_tiler = (
            scale_cta_tiler[0],
            4,
            MXF4NVF4_CTA_SHAPE_MNK[2] // (MXF4NVF4_SCALE_VEC_SIZE * 4),
            1,
        )
        scale_tile_coord_sfa = (tile_mnl[0], 0, k_tile, tile_mnl[2])
        scale_tile_coord_sfb = (tile_mnl[1], 0, k_tile, tile_mnl[2])
    elif scale_smem_format == "physical":
        scale_tile_coord_sfa = tile_coords["scale_tile_coord_sfa"]
        scale_tile_coord_sfb = tile_coords["scale_tile_coord_sfb"]
    else:
        raise ValueError("scale_smem_format must be either 'physical' or 'interleaved'")
    return (
        cute.local_tile(
            tma_tensor_a,
            ab_cta_tiler,
            tile_coords["ab_tile_coord_a"],
            loc=loc,
            ip=ip,
        ),
        cute.local_tile(
            tma_tensor_b,
            ab_cta_tiler,
            tile_coords["ab_tile_coord_b"],
            loc=loc,
            ip=ip,
        ),
        cute.local_tile(
            tma_tensor_sfa,
            scale_cta_tiler,
            scale_tile_coord_sfa,
            loc=loc,
            ip=ip,
        ),
        cute.local_tile(
            tma_tensor_sfb,
            scale_cta_tiler,
            scale_tile_coord_sfb,
            loc=loc,
            ip=ip,
        ),
    )


@dsl_user_op
def issue_mxf4nvf4_native_tma_stage_for_tile(
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    tma_tensor_sfa: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    tma_tensor_sfb: cute.Tensor,
    sA: cute.Tensor,
    sB: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    tma_bar_ptr: cute.Pointer,
    tile_mnl: tuple[cutlass.Int32 | int, ...],
    k_tile: cutlass.Int32 | int = 0,
    stage_idx: cutlass.Int32 | int = 0,
    *,
    batch_idx: cutlass.Int32 | int = 0,
    ab_cta_tiler: cute.Tile = (128, 128, 1),
    scale_cta_tiler: cute.Tile = (128, 8, 1, 1),
    scale_smem_format: str = "physical",
    already_elected: cutlass.Constexpr[bool] = False,
    cache_policy=None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Local-tile native TMA tensors for one scheduler tile and issue the stage."""
    (
        tiled_tma_tensor_a,
        tiled_tma_tensor_b,
        tiled_tma_tensor_sfa,
        tiled_tma_tensor_sfb,
    ) = local_tile_mxf4nvf4_native_tma_tensors(
        tma_tensor_a,
        tma_tensor_b,
        tma_tensor_sfa,
        tma_tensor_sfb,
        tile_mnl,
        k_tile,
        ab_cta_tiler=ab_cta_tiler,
        scale_cta_tiler=scale_cta_tiler,
        scale_smem_format=scale_smem_format,
        loc=loc,
        ip=ip,
    )
    issue_mxf4nvf4_native_tma_stage(
        tma_atom_a,
        tiled_tma_tensor_a,
        tma_atom_b,
        tiled_tma_tensor_b,
        tma_atom_sfa,
        tiled_tma_tensor_sfa,
        tma_atom_sfb,
        tiled_tma_tensor_sfb,
        sA,
        sB,
        sSFA,
        sSFB,
        tma_bar_ptr,
        stage_idx,
        batch_idx=batch_idx,
        already_elected=already_elected,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def issue_mxf4nvf4_native_tma_full_tile_consumer_group(
    tiled_mma: cute.TiledMma,
    sA_consumer: cute.Tensor,
    sB_consumer: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    acc: cute.Tensor,
    tidx: cutlass.Int32,
    stage_idx: cutlass.Int32 | int = 0,
    *,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    ab_smem_format: str = "packed",
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one K128 native-TMA consumer group for a full 128x128 tile."""
    _check_default_tile(tile_m, tile_k, sf_vec_size)
    _check_default_tile(tile_n, tile_k, sf_vec_size)
    ab_smem_format = _normalize_mxf4nvf4_ab_smem_format(ab_smem_format)
    if ab_smem_format != "packed":
        raise ValueError(
            "SM120 native full-tile consumer group currently supports only ab_smem_format='packed'"
        )

    a_frag = cute.make_rmem_tensor(
        tiled_mma.partition_shape_A((tile_m, tile_k), loc=loc, ip=ip),
        cutlass.Float4E2M1FN,
        loc=loc,
        ip=ip,
    )
    b_frag = cute.make_rmem_tensor(
        tiled_mma.partition_shape_B((tile_n, tile_k), loc=loc, ip=ip),
        cutlass.Float4E2M1FN,
        loc=loc,
        ip=ip,
    )
    copy_atom_a, copy_atom_b = make_mxf4nvf4_ab_smem_copy_atoms(loc=loc, ip=ip)
    tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma, loc=loc, ip=ip)
    tiled_copy_b = cute.make_tiled_copy_B(copy_atom_b, tiled_mma, loc=loc, ip=ip)
    thr_copy_a = tiled_copy_a.get_slice(tidx)
    thr_copy_b = tiled_copy_b.get_slice(tidx)
    sA_src = cute.as_position_independent_swizzle_tensor(sA_consumer, loc=loc, ip=ip)
    sB_src = cute.as_position_independent_swizzle_tensor(sB_consumer, loc=loc, ip=ip)
    tCsA = thr_copy_a.partition_S(sA_src, loc=loc, ip=ip)
    tCsB = thr_copy_b.partition_S(sB_src, loc=loc, ip=ip)
    tCrA = thr_copy_a.retile_D(a_frag, loc=loc, ip=ip)
    tCrB = thr_copy_b.retile_D(b_frag, loc=loc, ip=ip)

    sfa_frag, sfb_frag = make_mxf4nvf4_direct_tma_scale_fragments(
        tiled_mma,
        sSFA,
        sSFB,
        tidx,
        tile_shape_mnk=(tile_m, tile_n, tile_k),
        sf_vec_size=sf_vec_size,
        loc=loc,
        ip=ip,
    )

    issue_mxf4nvf4_direct_tma_consumer_group(
        tiled_mma,
        tiled_copy_a,
        tiled_copy_b,
        tCsA,
        tCsB,
        tCrA,
        tCrB,
        a_frag,
        b_frag,
        sSFA,
        sSFB,
        sfa_frag,
        sfb_frag,
        acc,
        tidx,
        stage_idx,
        major_extent_sfa=tile_m,
        major_extent_sfb=tile_n,
        tile_k=tile_k,
        sf_vec_size=sf_vec_size,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_tiled_mma(
    atom_layout_mnk: Tuple[int, int, int] = (1, 1, 1),
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.TiledMma:
    """Create the SM120 warp-level MXF4NVF4 tiled MMA."""
    mma_op = warp.MmaMXF4NVF4Op(
        cutlass.Float4E2M1FN,
        cutlass.Float32,
        cutlass.Float8E4M3FN,
    )
    return cute.make_tiled_mma(mma_op, atom_layout_mnk=atom_layout_mnk, loc=loc, ip=ip)


@dsl_user_op
def make_mxf4nvf4_79a_tiled_mma(
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.TiledMma:
    """Create the 79a-style SM120 128x128 ping-pong tiled MMA.

    This is the compact 4-warpgroup-local layout used by the fast SM120
    NVFP4 path: a (2,2,1) MMA atom layout with the N-major permutation needed
    by the STSM epilogue schedule.
    """
    mma_op = warp.MmaMXF4NVF4Op(
        cutlass.Float4E2M1FN,
        cutlass.Float32,
        cutlass.Float8E4M3FN,
    )
    return cute.make_tiled_mma(
        mma_op,
        atom_layout_mnk=cute.make_layout((2, 2, 1), stride=(1, 2, 0), loc=loc, ip=ip),
        permutation_mnk=(
            128,
            cute.make_layout((8, 2, 2), stride=(1, 16, 8), loc=loc, ip=ip),
            64,
        ),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def convert_mxf4nvf4_acc_layout_for_epilogue_stmatrix(
    acc_layout: cute.Layout,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """View SM120 accumulator registers in the fragment order used by STSM."""
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):
        div = 2 if const_expr(acc_layout.shape[0][2] % 2 == 0) else 1
        divided = cute.logical_divide(acc_layout, ((None, None, div), None, None), loc=loc, ip=ip)
        return cute.make_layout(
            (
                (divided.shape[0][0], divided.shape[0][1], divided.shape[0][2][0]),
                divided.shape[1],
                (divided.shape[0][2][1], divided.shape[2]),
            ),
            stride=(
                (
                    divided.stride[0][0],
                    divided.stride[0][1],
                    divided.stride[0][2][0],
                ),
                divided.stride[1],
                (divided.stride[0][2][1], divided.stride[2]),
            ),
            loc=loc,
            ip=ip,
        )
    if acc_layout.shape[2] % 2 != 0:
        raise ValueError("SM120 epilogue STSM accumulator view requires even N modes")
    divided = cute.logical_divide(acc_layout, (None, None, 2), loc=loc, ip=ip)
    return cute.make_layout(
        (
            (divided.shape[0][0], divided.shape[0][1], divided.shape[2][0]),
            divided.shape[1],
            divided.shape[2][1],
        ),
        stride=(
            (divided.stride[0][0], divided.stride[0][1], divided.stride[2][0]),
            divided.stride[1],
            divided.stride[2][1],
        ),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def retile_mxf4nvf4_accumulator_for_epilogue_stmatrix(
    acc: cute.Tensor,
    tRS_rD: cute.Tensor,
    tiled_copy_r2s: cute.TiledCopy,
    *,
    epi_tile_shape: cute.Tile = (2, 1),
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Tensor:
    """Retile a full SM120 accumulator fragment for one epilogue SMEM tile."""
    return tiled_copy_r2s.retile(acc, loc=loc, ip=ip)


@dsl_user_op
def load_mxf4nvf4_accumulator_epilogue_subtile(
    tRS_rAcc: cute.Tensor,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Load one accumulator epilogue subtile into the STSM source registers."""
    tRS_rD_flat = cute.coalesce(tRS_rD, loc=loc, ip=ip)
    for mma_n_in_epi in range(2):
        for mma_m_in_epi in range(2):
            idx = mma_n_in_epi * 2 + mma_m_in_epi
            tRS_rAcc_flat = cute.coalesce(
                tRS_rAcc[
                    None,
                    epi_coord[0] * 2 + mma_m_in_epi,
                    epi_coord[1] * 2 + mma_n_in_epi,
                ],
                loc=loc,
                ip=ip,
            )
            for epi_v in range(4):
                tRS_rD_flat[idx * 4 + epi_v] = tRS_rAcc_flat[epi_v].to(tRS_rD.element_type)


@dsl_user_op
def copy_mxf4nvf4_epilogue_registers_to_smem(
    tiled_copy_r2s: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Convert epilogue registers to the SMEM type and issue the STSM copy."""
    if const_expr(src.element_type != dst.element_type):
        src_cvt = cute.make_rmem_tensor_like(src, dst.element_type, loc=loc, ip=ip)
        src_cvt.store(src.load().to(dst.element_type), loc=loc, ip=ip)
        src = src_cvt
    cute.copy(tiled_copy_r2s, src, dst, loc=loc, ip=ip)
    cute.arch.fence_view_async_shared()


@dsl_user_op
def make_mxf4nvf4_epilogue_stmatrix_views(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    sD_tile: cute.Tensor,
    tidx: cutlass.Int32,
    *,
    epi_tile_shape: cute.Tile = (2, 1),
    num_matrices: int = 2,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
    """Create the SM120 STSM epilogue copy views for a BF16 SMEM tile."""
    if num_matrices != 2:
        raise ValueError("SM120 MXF4NVF4 epilogue STSM helper currently requires x2")
    copy_atom_c = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=False, num_matrices=num_matrices),
        cutlass.Float16,
        loc=loc,
        ip=ip,
    )
    tiled_copy_c_atom = cute.make_tiled_copy_C_atom(copy_atom_c, tiled_mma, loc=loc, ip=ip)
    copy_atom_r2s = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=False, num_matrices=num_matrices),
        sD_tile.element_type,
        loc=loc,
        ip=ip,
    )
    tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_c_atom, loc=loc, ip=ip)
    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
    tRS_sD = thr_copy_r2s.partition_D(sD_tile, loc=loc, ip=ip)
    tRS_rD_shape = thr_copy_r2s.partition_S(
        cute.make_identity_tensor(sD_tile.shape, loc=loc, ip=ip), loc=loc, ip=ip
    ).shape
    tRS_rD = cute.make_rmem_tensor(tRS_rD_shape, acc.element_type, loc=loc, ip=ip)
    tRS_rAcc = retile_mxf4nvf4_accumulator_for_epilogue_stmatrix(
        acc,
        tRS_rD,
        tiled_copy_r2s,
        epi_tile_shape=epi_tile_shape,
        loc=loc,
        ip=ip,
    )
    return tiled_copy_r2s, tRS_rD, tRS_sD, tRS_rAcc


def mxf4nvf4_79a_epilogue_tile(
    tile_m: int = 128,
    tile_n: int = 128,
) -> tuple[int, int]:
    """Return the 79a-style SM120 NVFP4 epilogue TMA-store subtile."""
    if tile_m != 128 or tile_n != 128:
        raise ValueError("SM120 MXF4NVF4 79a epilogue tile currently requires a 128x128 CTA tile")
    return (64, 32)


@dsl_user_op
def make_mxf4nvf4_epilogue_smem_layout(
    *,
    epi_tile: cute.Tile = (64, 32),
    num_stages: int = 1,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """Create the BF16 epilogue SMEM layout used by the SM120 fast store path."""
    _check_tuple("epi_tile", epi_tile, 2)
    epi_m, epi_n = epi_tile
    _check_positive("epi_m", epi_m)
    _check_positive("epi_n", epi_n)
    _check_positive("num_stages", num_stages)
    return cute.make_layout(
        (epi_m, epi_n, num_stages),
        stride=(epi_n, 1, epi_m * epi_n),
        loc=loc,
        ip=ip,
    )


def make_mxf4nvf4_epilogue_tma_store_atom(
    gD: cute.Tensor,
    smem_layout,
    *,
    epi_tile: cute.Tile = (64, 32),
    op=None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Create the SM120 MXF4/NVFP4 BF16 epilogue S2G TMA atom and tensor."""
    _check_tuple("epi_tile", epi_tile, 2)
    if op is None:
        op = cpasync.CopyBulkTensorTileS2GOp()
    smem_rank = cute.rank(smem_layout)
    if smem_rank == cute.rank(epi_tile) + 1:
        smem_layout = cute.slice_(smem_layout, (None, None, 0), loc=loc, ip=ip)
    d_cta_v_layout = cute.composition(
        cute.make_identity_layout(gD.shape, loc=loc, ip=ip),
        epi_tile,
        loc=loc,
        ip=ip,
    )
    return cpasync.make_tiled_tma_atom(
        op,
        gD,
        smem_layout,
        d_cta_v_layout,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def partition_mxf4nvf4_epilogue_tma_store(
    tma_atom_d: cute.CopyAtom,
    tma_tensor_d: cute.Tensor,
    sD_epi: cute.Tensor,
    tile_mnl: tuple[cutlass.Int32 | int, ...],
    *,
    cta_tiler: cute.Tile = (128, 128, 1),
    epi_tile: cute.Tile = (64, 32),
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> tuple[cute.Tensor, cute.Tensor]:
    """Partition one scheduler-selected BF16 epilogue S2G TMA store tile."""
    _check_tuple("tile_mnl", tile_mnl, 3)
    _check_tuple("cta_tiler", cta_tiler, 3)
    _check_tuple("epi_tile", epi_tile, 2)
    tiled_d = cute.local_tile(
        tma_tensor_d,
        cta_tiler[:2],
        (None, None, None),
        loc=loc,
        ip=ip,
    )
    tile_d = tiled_d[(None, None, tile_mnl[0], tile_mnl[1], tile_mnl[2])]
    epi_d = cute.zipped_divide(tile_d, epi_tile, loc=loc, ip=ip)
    return cpasync.tma_partition(
        tma_atom_d,
        0,
        cute.make_layout(1),
        cute.group_modes(sD_epi, 0, 2, loc=loc, ip=ip),
        epi_d,
        loc=loc,
        ip=ip,
    )


def issue_mxf4nvf4_epilogue_tma_store(
    tma_atom_d: cute.CopyAtom,
    tDsD: cute.Tensor,
    tDgD: cute.Tensor,
    *,
    epi_m: int = 0,
    epi_n: int = 0,
    stage_idx: int = 0,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one selected SM120 MXF4/NVFP4 BF16 epilogue S2G TMA subtile."""
    cute.copy(
        tma_atom_d,
        tDsD[None, stage_idx],
        tDgD[None, (epi_m, epi_n)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stage_mxf4nvf4_accumulator_fragment_D_to_smem(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    sD_tile: cute.Tensor,
    tidx: cutlass.Int32,
    *,
    epi_m: int = 0,
    epi_n: int = 0,
    epi_tile_shape: cute.Tile = (2, 1),
    num_matrices: int = 2,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Stage one SM120 MXF4/NVFP4 accumulator epilogue tile with STSM.

    The default shape matches the 79a-style 128x128 CTA tile split into
    subtiles.  ``epi_m`` and ``epi_n`` select the epilogue subtile to stage.
    """
    tiled_copy_r2s, tRS_rD, tRS_sD, tRS_rAcc = make_mxf4nvf4_epilogue_stmatrix_views(
        tiled_mma,
        acc,
        sD_tile,
        tidx,
        epi_tile_shape=epi_tile_shape,
        num_matrices=num_matrices,
        loc=loc,
        ip=ip,
    )
    load_mxf4nvf4_accumulator_epilogue_subtile(tRS_rAcc, tRS_rD, (epi_m, epi_n), loc=loc, ip=ip)
    copy_mxf4nvf4_epilogue_registers_to_smem(tiled_copy_r2s, tRS_rD, tRS_sD, loc=loc, ip=ip)


@dsl_user_op
def store_mxf4nvf4_accumulator_fragment_D(
    thr_mma: cute.ThrMma,
    acc: cute.Tensor,
    gD: cute.Tensor,
    pred: Optional[cute.Tensor] = None,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Store one SM120 warp-MMA accumulator fragment directly to global D."""
    tDgD = thr_mma.partition_C(gD)
    rD = cute.make_rmem_tensor(acc.layout, gD.element_type)
    rD.store(acc.load().to(gD.element_type))
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gD.element_type,
        loc=loc,
        ip=ip,
    )
    if const_expr(pred is None):
        cute.copy(copy_atom, rD, tDgD, loc=loc, ip=ip)
    else:
        cute.copy(copy_atom, rD, tDgD, pred=pred, loc=loc, ip=ip)


@dsl_user_op
def local_tile_mxf4nvf4_d_tensor(
    gD: cute.Tensor,
    tile_mnl: tuple[cutlass.Int32 | int, ...],
    *,
    cta_tiler: cute.Tile = (128, 128, 1),
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Tensor:
    """Local-tile the SM120 output tensor for one scheduler work tile."""
    _check_tuple("tile_mnl", tile_mnl, 3)
    if const_expr(cute.rank(gD) == 2):
        return cute.local_tile(
            gD,
            (cta_tiler[0], cta_tiler[1]),
            (tile_mnl[0], tile_mnl[1]),
            loc=loc,
            ip=ip,
        )
    gD_tile = cute.local_tile(
        gD,
        cta_tiler,
        tile_mnl,
        loc=loc,
        ip=ip,
    )
    return gD_tile[(None, None, 0)]


@dsl_user_op
def store_mxf4nvf4_accumulator_fragment_D_for_tile(
    thr_mma: cute.ThrMma,
    acc: cute.Tensor,
    gD: cute.Tensor,
    tile_mnl: tuple[cutlass.Int32 | int, ...],
    pred: Optional[cute.Tensor] = None,
    *,
    cta_tiler: cute.Tile = (128, 128, 1),
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Store one accumulator fragment to the scheduler-selected D tile."""
    gD_tile = local_tile_mxf4nvf4_d_tensor(
        gD,
        tile_mnl,
        cta_tiler=cta_tiler,
        loc=loc,
        ip=ip,
    )
    store_mxf4nvf4_accumulator_fragment_D(
        thr_mma,
        acc,
        gD_tile,
        pred=pred,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_mxf4nvf4_accumulator_fragment_D_for_tiled_mma_tile(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    gD: cute.Tensor,
    tile_mnl: tuple[cutlass.Int32 | int, ...],
    tidx: cutlass.Int32,
    pred: Optional[cute.Tensor] = None,
    *,
    cta_tiler: cute.Tile = (128, 128, 1),
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Direct-store one SM120 accumulator tile from a tiled-MMA slice.

    Ping-pong mainloops commonly have two consumer warpgroups resident in one
    CTA. Each warpgroup can store a complete 128x128 accumulator tile when the
    caller gates ownership with a surrounding runtime branch. This helper keeps
    the selected warpgroup's direct global-store path compact so callers do not
    need to route through BF16 epilogue SMEM/TMA staging.
    """
    thr_mma = tiled_mma.get_slice(tidx)
    store_mxf4nvf4_accumulator_fragment_D_for_tile(
        thr_mma,
        acc,
        gD,
        tile_mnl,
        pred=pred,
        cta_tiler=cta_tiler,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_ab_tma_physical_layout_staged(
    major_extent: int = 128,
    tile_k: int = 128,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """Return the A/B physical SMEM byte layout populated by external TMA."""
    _check_default_tile(major_extent, tile_k, MXF4NVF4_SCALE_VEC_SIZE)
    _check_positive("num_stages", num_stages)
    return cute.make_layout(
        (major_extent, tile_k, num_stages),
        stride=(tile_k, 1, major_extent * tile_k),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_consumer_smem_layout_atom_ab(
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.ComposedLayout:
    """Return the SM120 packed-FP4 consumer SMEM atom layout.

    This mirrors the layout atom selected by the 79a C++ collective:
    `Sw<2,4,3> o smem_ptr[4b] o (_8,_128):(_128,_1)`.
    """
    return cute.make_composed_layout(
        cute.make_swizzle(2, 4, 3, loc=loc, ip=ip),
        0,
        cute.make_layout((8, 128), stride=(128, 1), loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_a_consumer_smem_layout_staged(
    major_extent: int = 128,
    tile_k: int = 128,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.ComposedLayout:
    """Return the staged 79a-style A consumer SMEM layout."""
    _check_default_tile(major_extent, tile_k, MXF4NVF4_SCALE_VEC_SIZE)
    _check_positive("num_stages", num_stages)
    return cute.tile_to_shape(
        make_mxf4nvf4_consumer_smem_layout_atom_ab(loc=loc, ip=ip),
        (major_extent, tile_k, num_stages),
        (0, 1, 2),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_b_consumer_smem_layout_staged(
    major_extent: int = 128,
    tile_k: int = 128,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.ComposedLayout:
    """Return the staged 79a-style B consumer SMEM layout."""
    _check_default_tile(major_extent, tile_k, MXF4NVF4_SCALE_VEC_SIZE)
    _check_positive("num_stages", num_stages)
    return cute.tile_to_shape(
        make_mxf4nvf4_consumer_smem_layout_atom_ab(loc=loc, ip=ip),
        (major_extent, tile_k, num_stages),
        (0, 1, 2),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_direct_tma_consumer_smem_layout_atom_ab(
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.ComposedLayout:
    """Return the packed-U4 direct-TMA consumer SMEM atom layout.

    This is the 79a-style `UMMA::Layout_K_SW128_Atom<uint8_t>` layout for
    loading A/B directly from TMA into the SMEM layout consumed by LDSM.
    """
    return cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3, loc=loc, ip=ip),
        0,
        cute.make_layout((8, 128), stride=(128, 1), loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_a_direct_tma_consumer_smem_layout_staged(
    major_extent: int = 128,
    tile_k: int = 128,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.ComposedLayout:
    """Return the staged A direct-TMA consumer SMEM layout."""
    _check_default_tile(major_extent, tile_k, MXF4NVF4_SCALE_VEC_SIZE)
    _check_positive("num_stages", num_stages)
    return cute.tile_to_shape(
        make_mxf4nvf4_direct_tma_consumer_smem_layout_atom_ab(loc=loc, ip=ip),
        (major_extent, tile_k, num_stages),
        (0, 1, 2),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_b_direct_tma_consumer_smem_layout_staged(
    major_extent: int = 128,
    tile_k: int = 128,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.ComposedLayout:
    """Return the staged B direct-TMA consumer SMEM layout."""
    _check_default_tile(major_extent, tile_k, MXF4NVF4_SCALE_VEC_SIZE)
    _check_positive("num_stages", num_stages)
    return cute.tile_to_shape(
        make_mxf4nvf4_direct_tma_consumer_smem_layout_atom_ab(loc=loc, ip=ip),
        (major_extent, tile_k, num_stages),
        (0, 1, 2),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_a_packed_direct_tma_consumer_smem_layout_staged(
    major_extent: int = 128,
    tile_k: int = 128,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.ComposedLayout:
    """Return the packed-FP4 A direct-TMA consumer SMEM layout.

    This is the compact FP4 consumer layout used by the native CuTe TMA atom
    path when A/B SMEM format is packed.
    """
    return make_mxf4nvf4_a_consumer_smem_layout_staged(
        major_extent, tile_k, num_stages, loc=loc, ip=ip
    )


@dsl_user_op
def make_mxf4nvf4_b_packed_direct_tma_consumer_smem_layout_staged(
    major_extent: int = 128,
    tile_k: int = 128,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.ComposedLayout:
    """Return the packed-FP4 B direct-TMA consumer SMEM layout."""
    return make_mxf4nvf4_b_consumer_smem_layout_staged(
        major_extent, tile_k, num_stages, loc=loc, ip=ip
    )


def make_mxf4nvf4_ab_direct_tma_consumer_smem_views(
    smem: SmemAllocator,
    *,
    num_stages: int = 1,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Allocate Uint8 A/B SMEM views for direct consumer-layout TMA."""
    layout_a = make_mxf4nvf4_a_direct_tma_consumer_smem_layout_staged(tile_m, tile_k, num_stages)
    layout_b = make_mxf4nvf4_b_direct_tma_consumer_smem_layout_staged(tile_n, tile_k, num_stages)
    return (
        smem.allocate_tensor(cutlass.Uint8, layout_a, byte_alignment=128),
        smem.allocate_tensor(cutlass.Uint8, layout_b, byte_alignment=128),
    )


def make_mxf4nvf4_ab_packed_direct_tma_consumer_smem_views(
    smem: SmemAllocator,
    *,
    num_stages: int = 1,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Allocate packed-FP4 A/B SMEM views for direct consumer-layout TMA."""
    return make_mxf4nvf4_ab_consumer_smem_views(
        smem, num_stages=num_stages, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k
    )


def make_mxf4nvf4_native_tma_smem_views(
    smem: SmemAllocator,
    *,
    tiled_mma: Optional[cute.TiledMma] = None,
    num_stages: int = 1,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    ab_smem_format: str = "packed",
    scale_smem_format: str = "physical",
) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
    """Allocate A/B/SFA/SFB SMEM views for native SM120 TMA atoms."""
    ab_smem_format = _normalize_mxf4nvf4_ab_smem_format(ab_smem_format)
    if tiled_mma is None:
        tiled_mma = make_mxf4nvf4_tiled_mma()
    if ab_smem_format == "unpack":
        sA, sB = make_mxf4nvf4_ab_direct_tma_consumer_smem_views(
            smem,
            num_stages=num_stages,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
        )
    else:
        sA, sB = make_mxf4nvf4_ab_packed_direct_tma_consumer_smem_views(
            smem,
            num_stages=num_stages,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
        )
    if scale_smem_format == "interleaved":
        sSFA, sSFB = allocate_mxf4nvf4_scale_tma_interleaved(
            smem,
            tiled_mma=tiled_mma,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            sf_vec_size=sf_vec_size,
            num_stages=num_stages,
        )
    elif scale_smem_format == "physical":
        sSFA = allocate_mxf4nvf4_scale_tma_physical(
            smem,
            major_extent=tile_m,
            tile_k=tile_k,
            sf_vec_size=sf_vec_size,
            num_stages=num_stages,
        )
        sSFB = allocate_mxf4nvf4_scale_tma_physical(
            smem,
            major_extent=tile_n,
            tile_k=tile_k,
            sf_vec_size=sf_vec_size,
            num_stages=num_stages,
        )
    else:
        raise ValueError("scale_smem_format must be either 'physical' or 'interleaved'")
    return (sA, sB, sSFA, sSFB)


@dsl_user_op
def make_mxf4nvf4_ab_packed_direct_tma_consumer_tma_views(
    sA_consumer: cute.Tensor,
    sB_consumer: cute.Tensor,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Reinterpret packed-FP4 direct consumer SMEM as byte TMA destinations."""
    return (
        cute.recast_tensor(sA_consumer, cutlass.Uint8, loc=loc, ip=ip),
        cute.recast_tensor(sB_consumer, cutlass.Uint8, loc=loc, ip=ip),
    )


@dsl_user_op
def make_mxf4nvf4_ab_direct_tma_consumer_fp4_views(
    sA_direct: cute.Tensor,
    sB_direct: cute.Tensor,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Reinterpret direct-TMA Uint8 A/B SMEM as packed FP4 consumer views."""
    return (
        cute.recast_tensor(sA_direct, cutlass.Float4E2M1FN, loc=loc, ip=ip),
        cute.recast_tensor(sB_direct, cutlass.Float4E2M1FN, loc=loc, ip=ip),
    )


def make_mxf4nvf4_ab_consumer_smem_views(
    smem: SmemAllocator,
    *,
    num_stages: int = 1,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Allocate A/B SMEM views for the 79a-style consumer LDSM path."""
    layout_a = make_mxf4nvf4_a_consumer_smem_layout_staged(tile_m, tile_k, num_stages)
    layout_b = make_mxf4nvf4_b_consumer_smem_layout_staged(tile_n, tile_k, num_stages)
    return (
        smem.allocate_tensor(cutlass.Float4E2M1FN, layout_a, byte_alignment=128),
        smem.allocate_tensor(cutlass.Float4E2M1FN, layout_b, byte_alignment=128),
    )


@dsl_user_op
def make_mxf4nvf4_ab_consumer_microtile_views(
    sA_consumer: cute.Tensor,
    sB_consumer: cute.Tensor,
    m_atom: cutlass.Int32 | int = 0,
    n_atom: cutlass.Int32 | int = 0,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Return local 16x8 MMA microtile views from a staged CTA consumer tile.

    Global CTA M/N selection belongs in the tensor-map descriptor coordinates.
    This helper only selects the local output atom within the already-staged
    128x128 CTA tile.
    """
    return (
        cute.domain_offset(
            (m_atom * MXF4NVF4_MMA_SHAPE_MNK[0], 0, 0),
            sA_consumer,
            loc=loc,
            ip=ip,
        ),
        cute.domain_offset(
            (n_atom * MXF4NVF4_MMA_SHAPE_MNK[1], 0, 0),
            sB_consumer,
            loc=loc,
            ip=ip,
        ),
    )


@dsl_user_op
def make_mxf4nvf4_sfa_smem_layout_staged(
    tiled_mma: Optional[cute.TiledMma] = None,
    tile_shape_mnk: cute.Tile = MXF4NVF4_CTA_SHAPE_MNK,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """Return the staged SFA SMEM layout for SM120 MXF4NVF4."""
    tiled_mma = make_mxf4nvf4_tiled_mma(loc=loc, ip=ip) if tiled_mma is None else tiled_mma
    return blockscaled_layout.sm120_make_smem_layout_sfa(
        tiled_mma,
        tile_shape_mnk,
        sf_vec_size,
        num_stages,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_sfb_smem_layout_staged(
    tiled_mma: Optional[cute.TiledMma] = None,
    tile_shape_mnk: cute.Tile = MXF4NVF4_CTA_SHAPE_MNK,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """Return the staged SFB SMEM layout for SM120 MXF4NVF4."""
    tiled_mma = make_mxf4nvf4_tiled_mma(loc=loc, ip=ip) if tiled_mma is None else tiled_mma
    return blockscaled_layout.sm120_make_smem_layout_sfb(
        tiled_mma,
        tile_shape_mnk,
        sf_vec_size,
        num_stages,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_scale_tma_physical_as_tiled_smem_layout_staged(
    major_extent: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.ComposedLayout:
    """Return a logical-K view over rank-4 native scale-TMA SMEM bytes.

    Native scale TMA writes compact FP8 scale columns through the tensor-map
    128B swizzle.  The consumer scale fragments are indexed by logical FP4 K
    coordinates, where all 16 FP4 values in one scale vector share one FP8
    scale.  This layout keeps that logical K surface while preserving the TMA
    physical byte mapping in shared memory.
    """
    _check_default_tile(major_extent, tile_k, sf_vec_size)
    _check_positive("num_stages", num_stages)
    if major_extent % 128 != 0:
        raise ValueError("SM120 scale TMA logical view requires major_extent % 128 == 0")
    scale_k = tile_k // sf_vec_size
    if scale_k % 4 != 0:
        raise ValueError("SM120 scale TMA logical view requires scale_k % 4 == 0")
    physical_major_extent = max(major_extent, 128)
    major_tiles = physical_major_extent // 128
    physical_bytes = physical_major_extent * scale_k
    layout = cute.make_layout(
        (((32, 4), major_tiles), ((sf_vec_size, 4), scale_k // 4), num_stages),
        stride=(
            ((1, 32), 128),
            ((0, physical_major_extent), physical_major_extent * 4),
            physical_bytes,
        ),
        loc=loc,
        ip=ip,
    )
    return cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3, loc=loc, ip=ip),
        0,
        layout,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_scale_tma_physical_layout_staged(
    major_extent: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.ComposedLayout:
    """Return the rank-4 scale TMA physical destination layout.

    The returned layout carries the 128B tensor-map swizzle used by the SM120
    native scale TMA path. Flattened byte consumers can still use the raw
    iterator with mxf4nvf4_scale_tma_physical_offset.
    """
    _check_default_tile(major_extent, tile_k, sf_vec_size)
    _check_positive("num_stages", num_stages)
    scale_k = tile_k // sf_vec_size
    physical_major_extent = max(major_extent, 128)
    physical_bytes = physical_major_extent * scale_k
    if major_extent < physical_major_extent:
        layout = cute.make_layout(
            (major_extent, scale_k, 1, num_stages),
            stride=(1, physical_major_extent, physical_bytes, physical_bytes),
            loc=loc,
            ip=ip,
        )
    else:
        layout = cute.make_layout(
            (physical_major_extent, scale_k, 1, num_stages),
            stride=(1, physical_major_extent, physical_bytes, physical_bytes),
            loc=loc,
            ip=ip,
        )
    return cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3, loc=loc, ip=ip),
        0,
        layout,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_scale_interleaved_tma_layout_staged(
    major_extent: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """Return compact interleaved FP8 scale TMA SMEM layout."""
    _check_default_tile(major_extent, tile_k, sf_vec_size)
    _check_positive("num_stages", num_stages)
    scale_k = tile_k // sf_vec_size
    if scale_k % 4 != 0:
        raise ValueError("SM120 scale interleaved SMEM layout requires scale_k % 4 == 0")
    major_tiles = major_extent // 128
    scale_tiles = scale_k // 4
    stage_stride = major_tiles * scale_tiles * 512
    return cute.make_layout(
        (((32, 4), major_tiles), 4, scale_tiles, num_stages),
        stride=(((16, 4), 512), 1, major_tiles * 512, stage_stride),
        loc=loc,
        ip=ip,
    )


def make_mxf4nvf4_tma_scale_layout_staged(
    major_extent: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    num_stages: int = 1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.ComposedLayout:
    """Compatibility alias for the scale TMA physical SMEM layout."""
    return make_mxf4nvf4_scale_tma_physical_layout_staged(
        major_extent, tile_k, sf_vec_size, num_stages, loc=loc, ip=ip
    )


def allocate_mxf4nvf4_scale_tma_physical(
    smem: SmemAllocator,
    major_extent: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    num_stages: int = 1,
) -> cute.Tensor:
    """Allocate padded SFA/SFB TMA physical storage and return its logical view."""
    view_layout = make_mxf4nvf4_scale_tma_physical_layout_staged(
        major_extent,
        tile_k,
        sf_vec_size,
        num_stages,
    )
    physical_bytes = mxf4nvf4_scale_physical_smem_bytes(major_extent, tile_k, sf_vec_size)
    backing = smem.allocate_tensor(
        cutlass.Uint8,
        cute.make_layout((physical_bytes, num_stages), stride=(1, physical_bytes)),
        byte_alignment=128,
    )
    return cute.make_tensor(backing.iterator, view_layout)


def allocate_mxf4nvf4_scale_tma_interleaved(
    smem: SmemAllocator,
    *,
    tiled_mma: Optional[cute.TiledMma] = None,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    num_stages: int = 1,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Allocate 79a/SM100-style interleaved scale TMA SMEM views."""
    if tiled_mma is None:
        tiled_mma = make_mxf4nvf4_tiled_mma()
    sfa_layout = make_mxf4nvf4_scale_interleaved_tma_layout_staged(
        tile_m,
        tile_k,
        sf_vec_size,
        num_stages,
    )
    sfb_layout = make_mxf4nvf4_scale_interleaved_tma_layout_staged(
        tile_n,
        tile_k,
        sf_vec_size,
        num_stages,
    )
    return (
        smem.allocate_tensor(cutlass.Uint8, sfa_layout, byte_alignment=128),
        smem.allocate_tensor(cutlass.Uint8, sfb_layout, byte_alignment=128),
    )


def _make_mxf4nvf4_tiled_tma_atom(
    gmem_tensor: cute.Tensor,
    smem_layout: cute.Layout,
    cta_tiler: cute.Tile,
    *,
    internal_type: Optional[Type[Numeric]] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    op = cpasync.CopyBulkTensorTileG2SOp()
    return cpasync.make_tiled_tma_atom(
        op,
        gmem_tensor,
        smem_layout,
        cta_tiler,
        internal_type=internal_type,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_tiled_tma_atom_a(
    gmem_tensor: cute.Tensor,
    smem_layout: Optional[cute.Layout] = None,
    cta_tiler: cute.Tile = (128, 128, 1),
    smem_format: str = "packed",
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Create the layout-aware TMA atom/tensor for one A tile.

    The default uses the runtime-validated packed FP4 tensor-map format. Keep
    the GMEM tensor logically FP4 and pass ``smem_format="unpack"`` or call
    ``make_mxf4nvf4_unpack_tiled_tma_atom_a`` explicitly for the experimental
    FP4 unpack-SMEM tensor-map path
    (``CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B``).
    """
    smem_format = _normalize_mxf4nvf4_ab_smem_format(smem_format)
    if const_expr(smem_layout is None):
        if const_expr(smem_format == "unpack"):
            smem_layout = make_mxf4nvf4_a_direct_tma_consumer_smem_layout_staged(loc=loc, ip=ip)
        else:
            smem_layout = make_mxf4nvf4_a_packed_direct_tma_consumer_smem_layout_staged(
                loc=loc, ip=ip
            )
    return _make_mxf4nvf4_tiled_tma_atom(
        gmem_tensor,
        smem_layout,
        cta_tiler,
        internal_type=_mxf4nvf4_ab_tma_internal_type(smem_format),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_tiled_tma_atom_b(
    gmem_tensor: cute.Tensor,
    smem_layout: Optional[cute.Layout] = None,
    cta_tiler: cute.Tile = (128, 128, 1),
    smem_format: str = "packed",
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Create the layout-aware TMA atom/tensor for one B tile."""
    smem_format = _normalize_mxf4nvf4_ab_smem_format(smem_format)
    if const_expr(smem_layout is None):
        if const_expr(smem_format == "unpack"):
            smem_layout = make_mxf4nvf4_b_direct_tma_consumer_smem_layout_staged(loc=loc, ip=ip)
        else:
            smem_layout = make_mxf4nvf4_b_packed_direct_tma_consumer_smem_layout_staged(
                loc=loc, ip=ip
            )
    return _make_mxf4nvf4_tiled_tma_atom(
        gmem_tensor,
        smem_layout,
        cta_tiler,
        internal_type=_mxf4nvf4_ab_tma_internal_type(smem_format),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_packed_tiled_tma_atom_a(
    gmem_tensor: cute.Tensor,
    smem_layout: Optional[cute.Layout] = None,
    cta_tiler: cute.Tile = (128, 128, 1),
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Create an A TMA atom for packed FP4 SMEM format."""
    return make_mxf4nvf4_tiled_tma_atom_a(
        gmem_tensor,
        smem_layout,
        cta_tiler,
        smem_format="packed",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_packed_tiled_tma_atom_b(
    gmem_tensor: cute.Tensor,
    smem_layout: Optional[cute.Layout] = None,
    cta_tiler: cute.Tile = (128, 128, 1),
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Create a B TMA atom for packed FP4 SMEM format."""
    return make_mxf4nvf4_tiled_tma_atom_b(
        gmem_tensor,
        smem_layout,
        cta_tiler,
        smem_format="packed",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_unpack_tiled_tma_atom_a(
    gmem_tensor: cute.Tensor,
    smem_layout: Optional[cute.Layout] = None,
    cta_tiler: cute.Tile = (128, 128, 1),
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Create an A TMA atom for FP4 unpack-SMEM format."""
    return make_mxf4nvf4_tiled_tma_atom_a(
        gmem_tensor,
        smem_layout,
        cta_tiler,
        smem_format="unpack",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_unpack_tiled_tma_atom_b(
    gmem_tensor: cute.Tensor,
    smem_layout: Optional[cute.Layout] = None,
    cta_tiler: cute.Tile = (128, 128, 1),
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Create a B TMA atom for FP4 unpack-SMEM format."""
    return make_mxf4nvf4_tiled_tma_atom_b(
        gmem_tensor,
        smem_layout,
        cta_tiler,
        smem_format="unpack",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_sfa_tiled_tma_atom(
    gmem_tensor: cute.Tensor,
    smem_layout: Optional[cute.Layout] = None,
    cta_tiler: cute.Tile = (128, 8, 1, 1),
    tiled_mma: Optional[cute.TiledMma] = None,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Create the layout-aware TMA atom/tensor for one SFA tile."""
    if const_expr(smem_layout is None):
        smem_layout = make_mxf4nvf4_tma_scale_layout_staged(loc=loc, ip=ip)
    return _make_mxf4nvf4_tiled_tma_atom(
        gmem_tensor,
        smem_layout,
        cta_tiler,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_sfb_tiled_tma_atom(
    gmem_tensor: cute.Tensor,
    smem_layout: Optional[cute.Layout] = None,
    cta_tiler: cute.Tile = (128, 8, 1, 1),
    tiled_mma: Optional[cute.TiledMma] = None,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Create the layout-aware TMA atom/tensor for one SFB tile."""
    if const_expr(smem_layout is None):
        smem_layout = make_mxf4nvf4_tma_scale_layout_staged(loc=loc, ip=ip)
    return _make_mxf4nvf4_tiled_tma_atom(
        gmem_tensor,
        smem_layout,
        cta_tiler,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_native_tma_atoms(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gSFA: cute.Tensor,
    gSFB: cute.Tensor,
    *,
    tiled_mma: Optional[cute.TiledMma] = None,
    ab_smem_format: str = "packed",
    ab_cta_tiler: cute.Tile = (128, 128, 1),
    ab_tile_coord: Optional[Tuple[int, int, int]] = None,
    ab_tile_coord_a: Optional[Tuple[int, int, int]] = None,
    ab_tile_coord_b: Optional[Tuple[int, int, int]] = None,
    scale_cta_tiler: cute.Tile = (128, 8, 1, 1),
    scale_tile_coord: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 0),
    scale_tile_coord_sfa: Optional[Tuple[int, int, int, int]] = None,
    scale_tile_coord_sfb: Optional[Tuple[int, int, int, int]] = None,
    scale_smem_format: str = "physical",
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Create A/B/SFA/SFB native TMA atoms for the SM120 NVFP4 path.

    `ab_tile_coord` is optional to preserve the legacy single-tile behavior.
    Tiled GEMM callers can pass independent `ab_tile_coord_a` and
    `ab_tile_coord_b` values because A is tiled by M while B is tiled by N.

    `scale_tile_coord` preserves the single-tile default. Tiled GEMM callers can
    pass independent `scale_tile_coord_sfa` and `scale_tile_coord_sfb` values
    because SFA is tiled by M while SFB is tiled by N.
    """
    ab_smem_format = _normalize_mxf4nvf4_ab_smem_format(ab_smem_format)
    gA = _preserve_mxf4nvf4_ab_tma_l_mode(gA)
    gB = _preserve_mxf4nvf4_ab_tma_l_mode(gB)
    if const_expr(ab_smem_format == "unpack"):
        tma_atom_a, tma_tensor_a = make_mxf4nvf4_unpack_tiled_tma_atom_a(
            gA, cta_tiler=ab_cta_tiler, loc=loc, ip=ip
        )
        tma_atom_b, tma_tensor_b = make_mxf4nvf4_unpack_tiled_tma_atom_b(
            gB, cta_tiler=ab_cta_tiler, loc=loc, ip=ip
        )
    else:
        tma_atom_a, tma_tensor_a = make_mxf4nvf4_packed_tiled_tma_atom_a(
            gA, cta_tiler=ab_cta_tiler, loc=loc, ip=ip
        )
        tma_atom_b, tma_tensor_b = make_mxf4nvf4_packed_tiled_tma_atom_b(
            gB, cta_tiler=ab_cta_tiler, loc=loc, ip=ip
        )
    if ab_tile_coord_a is None:
        ab_tile_coord_a = ab_tile_coord
    if ab_tile_coord_b is None:
        ab_tile_coord_b = ab_tile_coord
    if ab_tile_coord_a is not None:
        tma_tensor_a = cute.local_tile(
            tma_tensor_a,
            ab_cta_tiler,
            ab_tile_coord_a,
            loc=loc,
            ip=ip,
        )
    if ab_tile_coord_b is not None:
        tma_tensor_b = cute.local_tile(
            tma_tensor_b,
            ab_cta_tiler,
            ab_tile_coord_b,
            loc=loc,
            ip=ip,
        )
    if scale_smem_format == "interleaved":
        if tiled_mma is None:
            tiled_mma = make_mxf4nvf4_tiled_mma(loc=loc, ip=ip)
        scale_cta_tiler = (
            scale_cta_tiler[0],
            4,
            MXF4NVF4_CTA_SHAPE_MNK[2] // (MXF4NVF4_SCALE_VEC_SIZE * 4),
            1,
        )
        sfa_smem_layout = make_mxf4nvf4_scale_interleaved_tma_layout_staged(
            MXF4NVF4_CTA_SHAPE_MNK[0],
            MXF4NVF4_CTA_SHAPE_MNK[2],
            MXF4NVF4_SCALE_VEC_SIZE,
            1,
            loc=loc,
            ip=ip,
        )
        sfb_smem_layout = make_mxf4nvf4_scale_interleaved_tma_layout_staged(
            MXF4NVF4_CTA_SHAPE_MNK[1],
            MXF4NVF4_CTA_SHAPE_MNK[2],
            MXF4NVF4_SCALE_VEC_SIZE,
            1,
            loc=loc,
            ip=ip,
        )
        tma_atom_sfa, tma_tensor_sfa = make_mxf4nvf4_sfa_tiled_tma_atom(
            gSFA,
            smem_layout=sfa_smem_layout,
            cta_tiler=scale_cta_tiler,
            tiled_mma=tiled_mma,
            loc=loc,
            ip=ip,
        )
        tma_atom_sfb, tma_tensor_sfb = make_mxf4nvf4_sfb_tiled_tma_atom(
            gSFB,
            smem_layout=sfb_smem_layout,
            cta_tiler=scale_cta_tiler,
            tiled_mma=tiled_mma,
            loc=loc,
            ip=ip,
        )
    elif scale_smem_format == "physical":
        tma_atom_sfa, tma_tensor_sfa = make_mxf4nvf4_sfa_tiled_tma_atom(
            gSFA, cta_tiler=scale_cta_tiler, tiled_mma=tiled_mma, loc=loc, ip=ip
        )
        tma_atom_sfb, tma_tensor_sfb = make_mxf4nvf4_sfb_tiled_tma_atom(
            gSFB, cta_tiler=scale_cta_tiler, tiled_mma=tiled_mma, loc=loc, ip=ip
        )
    else:
        raise ValueError("scale_smem_format must be either 'physical' or 'interleaved'")
    if scale_tile_coord_sfa is None:
        scale_tile_coord_sfa = scale_tile_coord
    if scale_tile_coord_sfb is None:
        scale_tile_coord_sfb = scale_tile_coord
    if scale_tile_coord_sfa is not None:
        tma_tensor_sfa = cute.local_tile(
            tma_tensor_sfa,
            scale_cta_tiler,
            scale_tile_coord_sfa,
            loc=loc,
            ip=ip,
        )
    if scale_tile_coord_sfb is not None:
        tma_tensor_sfb = cute.local_tile(
            tma_tensor_sfb,
            scale_cta_tiler,
            scale_tile_coord_sfb,
            loc=loc,
            ip=ip,
        )
    return (
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_sfa,
        tma_tensor_sfa,
        tma_atom_sfb,
        tma_tensor_sfb,
    )


@dsl_user_op
def make_mxf4nvf4_native_tma_atoms_for_scheduler(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gSFA: cute.Tensor,
    gSFB: cute.Tensor,
    *,
    tiled_mma: Optional[cute.TiledMma] = None,
    ab_smem_format: str = "packed",
    ab_cta_tiler: cute.Tile = (128, 128, 1),
    scale_cta_tiler: cute.Tile = (128, 8, 1, 1),
    scale_smem_format: str = "physical",
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Create unlocalized native TMA atoms for scheduler-driven SM120 tiles."""
    return make_mxf4nvf4_native_tma_atoms(
        gA,
        gB,
        gSFA,
        gSFB,
        tiled_mma=tiled_mma,
        ab_smem_format=ab_smem_format,
        ab_cta_tiler=ab_cta_tiler,
        ab_tile_coord=None,
        ab_tile_coord_a=None,
        ab_tile_coord_b=None,
        scale_cta_tiler=scale_cta_tiler,
        scale_smem_format=scale_smem_format,
        scale_tile_coord=None,
        scale_tile_coord_sfa=None,
        scale_tile_coord_sfb=None,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def partition_mxf4nvf4_native_tma_tensors_for_scheduler(
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    tma_tensor_sfa: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    tma_tensor_sfb: cute.Tensor,
    sA: cute.Tensor,
    sB: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    *,
    tile_shape_mnk: cute.Tile = MXF4NVF4_CTA_SHAPE_MNK,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    scale_group_rank_smem: int = 3,
    scale_smem_format: str = "physical",
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Partition native TMA tensors by scheduler M/N/K/L tile coordinates.

    The returned GMEM partitions are indexed as:
    A: ``(None, tile_m, k_tile, tile_l)``
    B: ``(None, tile_n, k_tile, tile_l)``
    SFA: ``(None, tile_m, k_tile % 2, k_tile // 2, tile_l)``
    SFB: ``(None, tile_n, k_tile % 2, k_tile // 2, tile_l)``
    """
    _check_tuple("tile_shape_mnk", tile_shape_mnk, 3)
    tile_m, tile_n, tile_k = tile_shape_mnk
    _check_default_tile(tile_m, tile_k, sf_vec_size)
    _check_default_tile(tile_n, tile_k, sf_vec_size)
    scale_k = tile_k // sf_vec_size
    gA_mkl = cute.local_tile(
        tma_tensor_a,
        (tile_m, tile_k, 1),
        (None, None, None),
        loc=loc,
        ip=ip,
    )
    gB_nkl = cute.local_tile(
        tma_tensor_b,
        (tile_n, tile_k, 1),
        (None, None, None),
        loc=loc,
        ip=ip,
    )
    if scale_smem_format == "interleaved":
        scale_tiles_per_tma = tile_k // (sf_vec_size * 4)
        gSFA_mkl = cute.local_tile(
            tma_tensor_sfa,
            (tile_m, 4, scale_tiles_per_tma, 1),
            (None, None, None, None),
            loc=loc,
            ip=ip,
        )
        gSFB_nkl = cute.local_tile(
            tma_tensor_sfb,
            (tile_n, 4, scale_tiles_per_tma, 1),
            (None, None, None, None),
            loc=loc,
            ip=ip,
        )
        scale_group_rank_smem = 3
        scale_group_rank_gmem = 4
    elif scale_smem_format == "physical":
        gSFA_mkl = cute.local_tile(
            tma_tensor_sfa,
            (tile_m, scale_k, 1, 1),
            (None, None, None, None),
            loc=loc,
            ip=ip,
        )
        gSFB_nkl = cute.local_tile(
            tma_tensor_sfb,
            (tile_n, scale_k, 1, 1),
            (None, None, None, None),
            loc=loc,
            ip=ip,
        )
        scale_group_rank_gmem = 4
    else:
        raise ValueError("scale_smem_format must be either 'physical' or 'interleaved'")
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1, loc=loc, ip=ip),
        cute.group_modes(sA, 0, 2, loc=loc, ip=ip),
        cute.group_modes(gA_mkl, 0, 3, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1, loc=loc, ip=ip),
        cute.group_modes(sB, 0, 2, loc=loc, ip=ip),
        cute.group_modes(gB_nkl, 0, 3, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    tSFAs, tSFAg = cpasync.tma_partition(
        tma_atom_sfa,
        0,
        cute.make_layout(1, loc=loc, ip=ip),
        cute.group_modes(sSFA, 0, scale_group_rank_smem, loc=loc, ip=ip),
        cute.group_modes(gSFA_mkl, 0, scale_group_rank_gmem, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    tSFBs, tSFBg = cpasync.tma_partition(
        tma_atom_sfb,
        0,
        cute.make_layout(1, loc=loc, ip=ip),
        cute.group_modes(sSFB, 0, scale_group_rank_smem, loc=loc, ip=ip),
        cute.group_modes(gSFB_nkl, 0, scale_group_rank_gmem, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return tAsA, tAgA, tBsB, tBgB, tSFAs, tSFAg, tSFBs, tSFBg


@dsl_user_op
def issue_mxf4nvf4_partitioned_native_tma_stage_for_tile(
    tma_atom_a: cute.CopyAtom,
    tAsA: cute.Tensor,
    tAgA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tBsB: cute.Tensor,
    tBgB: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    tSFAs: cute.Tensor,
    tSFAg: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    tSFBs: cute.Tensor,
    tSFBg: cute.Tensor,
    tma_bar_ptr: cute.Pointer,
    tile_mnl: tuple[cutlass.Int32 | int, ...],
    k_tile: cutlass.Int32 | int = 0,
    stage_idx: cutlass.Int32 | int = 0,
    *,
    already_elected: cutlass.Constexpr[bool] = False,
    scale_smem_format: str = "physical",
    cache_policy=None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one scheduler-selected stage from pre-partitioned native TMA tensors."""
    _check_tuple("tile_mnl", tile_mnl, 3)
    tile_m, tile_n, tile_l = tile_mnl
    scale_k_tile = k_tile % 2
    scale_page = k_tile // 2
    if scale_smem_format == "interleaved":
        sfa_coord = (None, tile_m, 0, k_tile, tile_l)
        sfb_coord = (None, tile_n, 0, k_tile, tile_l)
    elif scale_smem_format == "physical":
        sfa_coord = (None, tile_m, scale_k_tile, scale_page, tile_l)
        sfb_coord = (None, tile_n, scale_k_tile, scale_page, tile_l)
    else:
        raise ValueError("scale_smem_format must be either 'physical' or 'interleaved'")
    if cutlass.const_expr(already_elected):
        _issue_native_tma_load_already_elected(
            tma_atom_a,
            tAgA[(None, tile_m, k_tile, tile_l)],
            tAsA[(None, stage_idx)],
            tma_bar_ptr,
            cache_policy=cache_policy,
            loc=loc,
            ip=ip,
        )
        _issue_native_tma_load_already_elected(
            tma_atom_b,
            tBgB[(None, tile_n, k_tile, tile_l)],
            tBsB[(None, stage_idx)],
            tma_bar_ptr,
            cache_policy=cache_policy,
            loc=loc,
            ip=ip,
        )
        _issue_native_tma_load_already_elected(
            tma_atom_sfa,
            tSFAg[sfa_coord],
            tSFAs[(None, stage_idx)],
            tma_bar_ptr,
            cache_policy=cache_policy,
            loc=loc,
            ip=ip,
        )
        _issue_native_tma_load_already_elected(
            tma_atom_sfb,
            tSFBg[sfb_coord],
            tSFBs[(None, stage_idx)],
            tma_bar_ptr,
            cache_policy=cache_policy,
            loc=loc,
            ip=ip,
        )
        return
    cute.copy(
        tma_atom_a,
        tAgA[(None, tile_m, k_tile, tile_l)],
        tAsA[(None, stage_idx)],
        tma_bar_ptr=tma_bar_ptr,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tma_atom_b,
        tBgB[(None, tile_n, k_tile, tile_l)],
        tBsB[(None, stage_idx)],
        tma_bar_ptr=tma_bar_ptr,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tma_atom_sfa,
        tSFAg[sfa_coord],
        tSFAs[(None, stage_idx)],
        tma_bar_ptr=tma_bar_ptr,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tma_atom_sfb,
        tSFBg[sfb_coord],
        tSFBs[(None, stage_idx)],
        tma_bar_ptr=tma_bar_ptr,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def issue_mxf4nvf4_partitioned_native_tma_mk_stage_for_tile(
    tma_atom_a: cute.CopyAtom,
    tAsA: cute.Tensor,
    tAgA: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    tSFAs: cute.Tensor,
    tSFAg: cute.Tensor,
    tma_bar_ptr: cute.Pointer,
    tile_mnl: tuple[cutlass.Int32 | int, ...],
    k_tile: cutlass.Int32 | int = 0,
    stage_idx: cutlass.Int32 | int = 0,
    *,
    scale_smem_format: str = "physical",
    cache_policy=None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue the A/SFA half of one scheduler-selected native TMA stage."""
    _check_tuple("tile_mnl", tile_mnl, 3)
    tile_m, _, tile_l = tile_mnl
    scale_k_tile = k_tile % 2
    scale_page = k_tile // 2
    if scale_smem_format == "interleaved":
        sfa_coord = (None, tile_m, 0, k_tile, tile_l)
    elif scale_smem_format == "physical":
        sfa_coord = (None, tile_m, scale_k_tile, scale_page, tile_l)
    else:
        raise ValueError("scale_smem_format must be either 'physical' or 'interleaved'")
    cute.copy(
        tma_atom_a,
        tAgA[(None, tile_m, k_tile, tile_l)],
        tAsA[(None, stage_idx)],
        tma_bar_ptr=tma_bar_ptr,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tma_atom_sfa,
        tSFAg[sfa_coord],
        tSFAs[(None, stage_idx)],
        tma_bar_ptr=tma_bar_ptr,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def issue_mxf4nvf4_partitioned_native_tma_nk_stage_for_tile(
    tma_atom_b: cute.CopyAtom,
    tBsB: cute.Tensor,
    tBgB: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    tSFBs: cute.Tensor,
    tSFBg: cute.Tensor,
    tma_bar_ptr: cute.Pointer,
    tile_mnl: tuple[cutlass.Int32 | int, ...],
    k_tile: cutlass.Int32 | int = 0,
    stage_idx: cutlass.Int32 | int = 0,
    *,
    scale_smem_format: str = "physical",
    cache_policy=None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue the B/SFB half of one scheduler-selected native TMA stage."""
    _check_tuple("tile_mnl", tile_mnl, 3)
    _, tile_n, tile_l = tile_mnl
    scale_k_tile = k_tile % 2
    scale_page = k_tile // 2
    if scale_smem_format == "interleaved":
        sfb_coord = (None, tile_n, 0, k_tile, tile_l)
    elif scale_smem_format == "physical":
        sfb_coord = (None, tile_n, scale_k_tile, scale_page, tile_l)
    else:
        raise ValueError("scale_smem_format must be either 'physical' or 'interleaved'")
    cute.copy(
        tma_atom_b,
        tBgB[(None, tile_n, k_tile, tile_l)],
        tBsB[(None, stage_idx)],
        tma_bar_ptr=tma_bar_ptr,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tma_atom_sfb,
        tSFBg[sfb_coord],
        tSFBs[(None, stage_idx)],
        tma_bar_ptr=tma_bar_ptr,
        cache_policy=cache_policy,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_ldsm_copy_atom(
    *,
    transpose: bool = False,
    dtype: Type[Numeric] = cutlass.Uint8,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.CopyAtom:
    """Create the packed 16-bit LDSM atom used by SM120 MXF4NVF4 A/B loads."""
    return cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
        dtype,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_ab_smem_copy_atoms(
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[cute.CopyAtom, cute.CopyAtom]:
    """Return the non-transposed packed-FP4 A/B LDSM copy atoms."""
    return (
        make_mxf4nvf4_ldsm_copy_atom(transpose=False, dtype=cutlass.Float4E2M1FN, loc=loc, ip=ip),
        make_mxf4nvf4_ldsm_copy_atom(transpose=False, dtype=cutlass.Float4E2M1FN, loc=loc, ip=ip),
    )


@dsl_user_op
def make_mxf4nvf4_unpack_ldsm_copy_atom(
    *,
    dtype: Type[Numeric] = cutlass.Uint8,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.CopyAtom:
    """Create the 79a-style FP4 unpack-SMEM LDSM atom.

    This mirrors C++ ``SM100_SU4_DU8x16_x4_LDSM_N``:
    ``ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64``.
    """
    return cute.make_copy_atom(
        warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4, unpack_bits=4),
        dtype,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_ab_unpack_smem_copy_atoms(
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[cute.CopyAtom, cute.CopyAtom]:
    """Return A/B LDSM atoms for FP4 unpack-SMEM consumer tiles."""
    return (
        make_mxf4nvf4_unpack_ldsm_copy_atom(dtype=cutlass.Uint8, loc=loc, ip=ip),
        make_mxf4nvf4_unpack_ldsm_copy_atom(dtype=cutlass.Uint8, loc=loc, ip=ip),
    )


@dsl_user_op
def make_mxf4nvf4_ab_ldsm_copy_views_from_consumer_smem(
    tiled_mma: cute.TiledMma,
    sA_consumer: cute.Tensor,
    sB_consumer: cute.Tensor,
    a_frag: cute.Tensor,
    b_frag: cute.Tensor,
    lane_idx: cutlass.Int32,
    *,
    m_atom: cutlass.Int32 | int = 0,
    n_atom: cutlass.Int32 | int = 0,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Return 79a-style A/B tiled-copy views from consumer SMEM.

    Input tensors must use the consumer SMEM layouts produced by
    `make_mxf4nvf4_ab_consumer_smem_views()` or the corresponding layout
    helpers. Do not pass raw external-TMA physical SMEM to this helper.
    `lane_idx` is the warp-local lane index, not the CTA thread index.
    """
    sA_consumer, sB_consumer = make_mxf4nvf4_ab_consumer_microtile_views(
        sA_consumer,
        sB_consumer,
        m_atom=m_atom,
        n_atom=n_atom,
        loc=loc,
        ip=ip,
    )
    copy_atom_a, copy_atom_b = make_mxf4nvf4_ab_smem_copy_atoms(loc=loc, ip=ip)
    tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma, loc=loc, ip=ip)
    tiled_copy_b = cute.make_tiled_copy_B(copy_atom_b, tiled_mma, loc=loc, ip=ip)
    thr_copy_a = tiled_copy_a.get_slice(lane_idx)
    thr_copy_b = tiled_copy_b.get_slice(lane_idx)
    sA_src = cute.as_position_independent_swizzle_tensor(sA_consumer, loc=loc, ip=ip)
    sB_src = cute.as_position_independent_swizzle_tensor(sB_consumer, loc=loc, ip=ip)
    tCsA = thr_copy_a.partition_S(sA_src, loc=loc, ip=ip)
    tCsB = thr_copy_b.partition_S(sB_src, loc=loc, ip=ip)
    tCrA = thr_copy_a.retile_D(a_frag, loc=loc, ip=ip)
    tCrB = thr_copy_b.retile_D(b_frag, loc=loc, ip=ip)
    return tiled_copy_a, tCsA, tCrA, tiled_copy_b, tCsB, tCrB


@dsl_user_op
def make_mxf4nvf4_ab_unpack_ldsm_copy_views_from_consumer_smem(
    tiled_mma: cute.TiledMma,
    sA_consumer: cute.Tensor,
    sB_consumer: cute.Tensor,
    a_frag: cute.Tensor,
    b_frag: cute.Tensor,
    lane_idx: cutlass.Int32,
    *,
    m_atom: cutlass.Int32 | int = 0,
    n_atom: cutlass.Int32 | int = 0,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
):
    """Return 79a-style unpack-SMEM A/B tiled-copy views.

    Input tensors must be Uint8 direct-consumer SMEM views populated by a
    logical-FP4 TMA atom built with ``internal_type=cutlass.Uint8``. The
    LDSM source pointer is aligned to 16 bytes because the unpack form loads
    128-bit source rows.
    """
    sA_consumer, sB_consumer = make_mxf4nvf4_ab_consumer_microtile_views(
        sA_consumer,
        sB_consumer,
        m_atom=m_atom,
        n_atom=n_atom,
        loc=loc,
        ip=ip,
    )
    copy_atom_a, copy_atom_b = make_mxf4nvf4_ab_unpack_smem_copy_atoms(loc=loc, ip=ip)
    tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma, loc=loc, ip=ip)
    tiled_copy_b = cute.make_tiled_copy_B(copy_atom_b, tiled_mma, loc=loc, ip=ip)
    thr_copy_a = tiled_copy_a.get_slice(lane_idx)
    thr_copy_b = tiled_copy_b.get_slice(lane_idx)
    sA_src = cute.as_position_independent_swizzle_tensor(sA_consumer, loc=loc, ip=ip)
    sB_src = cute.as_position_independent_swizzle_tensor(sB_consumer, loc=loc, ip=ip)
    tCsA = thr_copy_a.partition_S(sA_src, loc=loc, ip=ip)
    tCsB = thr_copy_b.partition_S(sB_src, loc=loc, ip=ip)
    tCsA = cute.make_tensor(tCsA.iterator.align(16), tCsA.layout, loc=loc, ip=ip)
    tCsB = cute.make_tensor(tCsB.iterator.align(16), tCsB.layout, loc=loc, ip=ip)
    tCrA = thr_copy_a.retile_D(a_frag, loc=loc, ip=ip)
    tCrB = thr_copy_b.retile_D(b_frag, loc=loc, ip=ip)
    return tiled_copy_a, tCsA, tCrA, tiled_copy_b, tCsB, tCrB


@dsl_user_op
def make_mxf4nvf4_ab_fragments_from_consumer_smem(
    tiled_mma: cute.TiledMma,
    sA_consumer: cute.Tensor,
    sB_consumer: cute.Tensor,
    lane_idx: cutlass.Int32,
    *,
    m_atom: cutlass.Int32 | int = 0,
    n_atom: cutlass.Int32 | int = 0,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Allocate A/B fragments for one local output atom.

    `lane_idx` is the warp-local lane index. Use `m_atom`/`n_atom` to select
    the local 16x8 atom inside the staged 128x128 CTA tile.
    """
    sA_consumer, sB_consumer = make_mxf4nvf4_ab_consumer_microtile_views(
        sA_consumer,
        sB_consumer,
        m_atom=m_atom,
        n_atom=n_atom,
        loc=loc,
        ip=ip,
    )
    thread_mma = tiled_mma.get_slice(lane_idx)
    tCsA_mma = thread_mma.partition_A(sA_consumer, loc=loc, ip=ip)
    tCsB_mma = thread_mma.partition_B(sB_consumer, loc=loc, ip=ip)
    return (
        tiled_mma.make_fragment_A(tCsA_mma[None, None, None, 0], loc=loc, ip=ip),
        tiled_mma.make_fragment_B(tCsB_mma[None, None, None, 0], loc=loc, ip=ip),
    )


@dsl_user_op
def shift_mxf4nvf4_post_ldsm_fp4_fragment(
    fragment: cute.Tensor,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Validate the FP4 post-LDSM transform point for MXF4NVF4 MMA.

    The C++ path applies an explicit nibble shift after its packed LDSM copy.
    Python CuTe's typed `Float4E2M1FN` LDSM copy already materializes
    MMA-ready fragments, so this hook is intentionally a no-op after dtype
    validation. Keeping the hook explicit makes the consumer path match the C++
    mainloop structure without corrupting the Python fragment encoding.
    """
    if fragment.element_type is not cutlass.Float4E2M1FN:
        raise TypeError(
            "SM120 MXF4NVF4 post-LDSM shift expects a Float4E2M1FN fragment, "
            f"got {fragment.element_type}"
        )


@dsl_user_op
def fp4_shift_mxf4nvf4_a(
    fragment: cute.Tensor,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Apply the A-fragment FP4 post-LDSM shift."""
    shift_mxf4nvf4_post_ldsm_fp4_fragment(fragment, loc=loc, ip=ip)


@dsl_user_op
def fp4_shift_mxf4nvf4_b(
    fragment: cute.Tensor,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Apply the B-fragment FP4 post-LDSM shift."""
    shift_mxf4nvf4_post_ldsm_fp4_fragment(fragment, loc=loc, ip=ip)


@dsl_user_op
def load_mxf4nvf4_ab_fragments_from_consumer_smem(
    tiled_mma: cute.TiledMma,
    sA_consumer: cute.Tensor,
    sB_consumer: cute.Tensor,
    a_frag: cute.Tensor,
    b_frag: cute.Tensor,
    lane_idx: cutlass.Int32,
    k_block_idx: int,
    consumer_stage_idx: int = 0,
    *,
    m_atom: cutlass.Int32 | int = 0,
    n_atom: cutlass.Int32 | int = 0,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Load one K64 A/B block through the 79a-style consumer copy path.

    `lane_idx` is the warp-local lane index. Use `m_atom`/`n_atom` to select
    the local 16x8 atom inside the staged 128x128 CTA tile.
    """
    tiled_copy_a, tCsA, tCrA, tiled_copy_b, tCsB, tCrB = (
        make_mxf4nvf4_ab_ldsm_copy_views_from_consumer_smem(
            tiled_mma,
            sA_consumer,
            sB_consumer,
            a_frag,
            b_frag,
            lane_idx,
            m_atom=m_atom,
            n_atom=n_atom,
            loc=loc,
            ip=ip,
        )
    )
    tCsA_stage = tCsA[(None, None, None, consumer_stage_idx)]
    tCsB_stage = tCsB[(None, None, None, consumer_stage_idx)]
    cute.copy(
        tiled_copy_a,
        tCsA_stage[(None, None, k_block_idx)],
        tCrA[(None, None, k_block_idx)],
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tiled_copy_b,
        tCsB_stage[(None, None, k_block_idx)],
        tCrB[(None, None, k_block_idx)],
        loc=loc,
        ip=ip,
    )
    fp4_shift_mxf4nvf4_a(tCrA[(None, None, k_block_idx)], loc=loc, ip=ip)
    fp4_shift_mxf4nvf4_b(tCrB[(None, None, k_block_idx)], loc=loc, ip=ip)


@dsl_user_op
def make_mxf4nvf4_ab_unpack_fragments_from_consumer_smem(
    tiled_mma: cute.TiledMma,
    sA_consumer: cute.Tensor,
    sB_consumer: cute.Tensor,
    lane_idx: cutlass.Int32,
    *,
    m_atom: cutlass.Int32 | int = 0,
    n_atom: cutlass.Int32 | int = 0,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Allocate logical FP4 A/B fragments for the unpack-SMEM LDSM path."""
    return make_mxf4nvf4_ab_fragments_from_consumer_smem(
        tiled_mma,
        sA_consumer,
        sB_consumer,
        lane_idx,
        m_atom=m_atom,
        n_atom=n_atom,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_mxf4nvf4_ab_unpack_fragments_from_consumer_smem(
    tiled_mma: cute.TiledMma,
    sA_consumer: cute.Tensor,
    sB_consumer: cute.Tensor,
    a_frag: cute.Tensor,
    b_frag: cute.Tensor,
    lane_idx: cutlass.Int32,
    k_block_idx: int,
    consumer_stage_idx: int = 0,
    *,
    m_atom: cutlass.Int32 | int = 0,
    n_atom: cutlass.Int32 | int = 0,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Load one K64 block through the 79a-style unpack-SMEM LDSM path."""
    a_copy_frag = cute.recast_tensor(a_frag, cutlass.Uint8, loc=loc, ip=ip)
    b_copy_frag = cute.recast_tensor(b_frag, cutlass.Uint8, loc=loc, ip=ip)
    tiled_copy_a, tCsA, tCrA, tiled_copy_b, tCsB, tCrB = (
        make_mxf4nvf4_ab_unpack_ldsm_copy_views_from_consumer_smem(
            tiled_mma,
            sA_consumer,
            sB_consumer,
            a_copy_frag,
            b_copy_frag,
            lane_idx,
            m_atom=m_atom,
            n_atom=n_atom,
            loc=loc,
            ip=ip,
        )
    )
    tCsA_stage = tCsA[(None, None, None, consumer_stage_idx)]
    tCsB_stage = tCsB[(None, None, None, consumer_stage_idx)]
    cute.copy(
        tiled_copy_a,
        tCsA_stage[(None, None, k_block_idx)],
        tCrA[(None, None, k_block_idx)],
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tiled_copy_b,
        tCsB_stage[(None, None, k_block_idx)],
        tCrB[(None, None, k_block_idx)],
        loc=loc,
        ip=ip,
    )


def stage_mxf4nvf4_a_tma_physical_to_consumer_smem(
    sA_tma_physical: cute.Tensor,
    sA_consumer: cute.Tensor,
    *,
    a_major_tile: cutlass.Int32 = cutlass.Int32(0),
    consumer_stage_idx: int = 0,
    tile_m: int = 128,
    tile_k: int = 128,
    lane_idx: Optional[cutlass.Int32] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Stage one physical TMA A tile into the SM120 consumer SMEM layout.

    `consumer_stage_idx` selects the destination consumer stage. Pass a
    physical-stage view as `sA_tma_physical` when staging from a nonzero
    physical stage.
    """
    _require_zero_major_offset("a_major_tile", a_major_tile)
    smem_bytes = mxf4nvf4_ab_physical_smem_bytes(tile_m, tile_k)
    tma_bytes = mxf4nvf4_ab_tma_tx_bytes(tile_m, tile_k)
    k_bytes = tile_k // 2
    loop_start = lane_idx if lane_idx is not None else 0
    loop_step = 32 if lane_idx is not None else 1
    src = cute.make_tensor(sA_tma_physical.iterator, cute.make_layout(smem_bytes))
    dst = cute.recast_tensor(sA_consumer, cutlass.Uint8, loc=loc, ip=ip)
    for i in for_generate(loop_start, tma_bytes, loop_step, loc=loc, ip=ip):
        major = i // k_bytes
        k_byte = i % k_bytes
        payload_byte = major * k_bytes + k_byte
        payload_chunk = payload_byte // 8
        payload_byte_in_chunk = payload_byte % 8
        physical_chunk = payload_chunk ^ ((payload_chunk >> 3) & 0x7)
        dst[(major, k_byte, consumer_stage_idx)] = src[physical_chunk * 16 + payload_byte_in_chunk]
        yield_out()


def stage_mxf4nvf4_b_tma_physical_to_consumer_smem(
    sB_tma_physical: cute.Tensor,
    sB_consumer: cute.Tensor,
    *,
    b_major_tile: cutlass.Int32 = cutlass.Int32(0),
    consumer_stage_idx: int = 0,
    tile_n: int = 128,
    tile_k: int = 128,
    lane_idx: Optional[cutlass.Int32] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Stage one physical TMA B tile into the SM120 consumer SMEM layout.

    `consumer_stage_idx` selects the destination consumer stage. Pass a
    physical-stage view as `sB_tma_physical` when staging from a nonzero
    physical stage.
    """
    _require_zero_major_offset("b_major_tile", b_major_tile)
    smem_bytes = mxf4nvf4_ab_physical_smem_bytes(tile_n, tile_k)
    tma_bytes = mxf4nvf4_ab_tma_tx_bytes(tile_n, tile_k)
    k_bytes = tile_k // 2
    loop_start = lane_idx if lane_idx is not None else 0
    loop_step = 32 if lane_idx is not None else 1
    src = cute.make_tensor(sB_tma_physical.iterator, cute.make_layout(smem_bytes))
    dst = cute.recast_tensor(sB_consumer, cutlass.Uint8, loc=loc, ip=ip)
    for i in for_generate(loop_start, tma_bytes, loop_step, loc=loc, ip=ip):
        major = i // k_bytes
        k_byte = i % k_bytes
        payload_byte = major * k_bytes + k_byte
        payload_chunk = payload_byte // 8
        payload_byte_in_chunk = payload_byte % 8
        physical_chunk = payload_chunk ^ ((payload_chunk >> 3) & 0x7)
        dst[(major, k_byte, consumer_stage_idx)] = src[physical_chunk * 16 + payload_byte_in_chunk]
        yield_out()


def stage_mxf4nvf4_ab_tma_physical_to_consumer_smem(
    sA_tma_physical: cute.Tensor,
    sB_tma_physical: cute.Tensor,
    sA_consumer: cute.Tensor,
    sB_consumer: cute.Tensor,
    *,
    a_major_tile: cutlass.Int32 = cutlass.Int32(0),
    b_major_tile: cutlass.Int32 = cutlass.Int32(0),
    consumer_stage_idx: int = 0,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    lane_idx: Optional[cutlass.Int32] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Stage physical TMA A/B tiles into the SM120 consumer SMEM layouts."""
    stage_mxf4nvf4_a_tma_physical_to_consumer_smem(
        sA_tma_physical,
        sA_consumer,
        a_major_tile=a_major_tile,
        consumer_stage_idx=consumer_stage_idx,
        tile_m=tile_m,
        tile_k=tile_k,
        lane_idx=lane_idx,
        loc=loc,
        ip=ip,
    )
    stage_mxf4nvf4_b_tma_physical_to_consumer_smem(
        sB_tma_physical,
        sB_consumer,
        b_major_tile=b_major_tile,
        consumer_stage_idx=consumer_stage_idx,
        tile_n=tile_n,
        tile_k=tile_k,
        lane_idx=lane_idx,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_mxf4nvf4_scale_smem_fragment_views(
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    k_block_idx: int,
    stage_idx: int = 0,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Return SFA/SFB source views for one K64 block from staged scale SMEM."""
    scale_k_offset = k_block_idx * (MXF4NVF4_MMA_SHAPE_MNK[2] // MXF4NVF4_SCALE_VEC_SIZE)
    stage_offset = stage_idx * MXF4NVF4_SCALE_TMA_BYTES
    sfa_f8 = cute.recast_tensor(sSFA, cutlass.Float8E4M3FN, loc=loc, ip=ip)
    sfb_f8 = cute.recast_tensor(sSFB, cutlass.Float8E4M3FN, loc=loc, ip=ip)
    sfa_ptr = sfa_f8.iterator + scale_k_offset + stage_offset
    sfb_ptr = sfb_f8.iterator + scale_k_offset + stage_offset
    return (
        cute.make_tensor(sfa_ptr, warp.make_mxf4nvf4_sfa_layout(loc=loc, ip=ip), loc=loc, ip=ip),
        cute.make_tensor(sfb_ptr, warp.make_mxf4nvf4_sfb_layout(loc=loc, ip=ip), loc=loc, ip=ip),
    )


def _mxf4nvf4_scale_tma_physical_offset_const(
    major: int,
    scale_col: int,
    major_extent: int,
) -> int:
    physical_major_extent = max(major_extent, 128)
    payload_idx = scale_col * physical_major_extent + major
    payload_chunk = payload_idx // 16
    payload_byte_in_chunk = payload_idx % 16
    physical_chunk = payload_chunk ^ ((payload_chunk >> 3) & 0x7)
    return physical_chunk * 16 + payload_byte_in_chunk


@dsl_user_op
def make_mxf4nvf4_scale_fragment_views_from_direct_tma(
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    k_block_idx: int,
    stage_idx: int = 0,
    major_extent_sfa: int = 128,
    major_extent_sfb: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Return SFA/SFB fragment views from compact direct scale TMA storage."""
    _check_default_tile(major_extent_sfa, tile_k, sf_vec_size)
    _check_default_tile(major_extent_sfb, tile_k, sf_vec_size)
    scale_col = k_block_idx * (MXF4NVF4_MMA_SHAPE_MNK[2] // sf_vec_size)
    sfa_stage_offset = stage_idx * mxf4nvf4_scale_physical_smem_bytes(
        major_extent_sfa, tile_k, sf_vec_size
    )
    sfb_stage_offset = stage_idx * mxf4nvf4_scale_physical_smem_bytes(
        major_extent_sfb, tile_k, sf_vec_size
    )
    sfa_offset = sfa_stage_offset + _mxf4nvf4_scale_tma_physical_offset_const(
        0, scale_col, major_extent_sfa
    )
    sfb_offset = sfb_stage_offset + _mxf4nvf4_scale_tma_physical_offset_const(
        0, scale_col, major_extent_sfb
    )
    sfa_f8 = cute.recast_tensor(sSFA, cutlass.Float8E4M3FN, loc=loc, ip=ip)
    sfb_f8 = cute.recast_tensor(sSFB, cutlass.Float8E4M3FN, loc=loc, ip=ip)
    return (
        cute.make_tensor(
            sfa_f8.iterator + sfa_offset,
            warp.make_mxf4nvf4_sfa_layout(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        ),
        cute.make_tensor(
            sfb_f8.iterator + sfb_offset,
            warp.make_mxf4nvf4_sfb_layout(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        ),
    )


make_mxf4nvf4_scale_fragment_views_from_compact_smem = make_mxf4nvf4_scale_smem_fragment_views


@dsl_user_op
def mxf4nvf4_scale_tma_physical_offset(
    major: cutlass.Int32,
    scale_col: cutlass.Int32,
    major_extent: int = 128,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cutlass.Int32:
    """Return rank-4 scale TMA physical byte offset for one logical scale."""
    _check_positive("major_extent", major_extent)
    physical_major_extent = max(major_extent, 128)
    payload_idx = scale_col * physical_major_extent + major
    chunk = payload_idx // 16
    byte = payload_idx % 16
    phys_chunk = chunk ^ (chunk >> 3)
    return phys_chunk * 16 + byte


def stage_mxf4nvf4_sfa_tma_physical_to_tiled_smem(
    tiled_mma: cute.TiledMma,
    sSFA_tma_physical: cute.Tensor,
    sSFA_tiled_smem: cute.Tensor,
    *,
    major_extent: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    lane_idx: Optional[cutlass.Int32] = None,
    thread_idx: Optional[cutlass.Int32] = None,
    thread_count: int = 32,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Stage SFA physical TMA SMEM into the SM120 tiled scale SMEM layout."""
    _check_default_tile(major_extent, tile_k, sf_vec_size)
    scale_k = tile_k // sf_vec_size
    tma_bytes = mxf4nvf4_scale_tma_tx_bytes(major_extent, tile_k, sf_vec_size)
    physical_bytes = mxf4nvf4_scale_physical_smem_bytes(major_extent, tile_k, sf_vec_size)
    src_u8 = cute.make_tensor(
        sSFA_tma_physical.iterator,
        cute.make_layout(physical_bytes, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    dst_u8 = cute.make_tensor(
        sSFA_tiled_smem.iterator,
        make_mxf4nvf4_sfa_smem_layout_staged(
            tiled_mma,
            (major_extent, tile_n, tile_k),
            sf_vec_size,
            1,
            loc=loc,
            ip=ip,
        ),
        loc=loc,
        ip=ip,
    )
    if thread_idx is not None:
        loop_start = thread_idx
        loop_step = thread_count
    else:
        loop_start = lane_idx if lane_idx is not None else 0
        loop_step = 32 if lane_idx is not None else 1
    for i in for_generate(loop_start, tma_bytes, loop_step, loc=loc, ip=ip):
        local_major = i // scale_k
        scale_col = i % scale_k
        phys = mxf4nvf4_scale_tma_physical_offset(
            local_major, scale_col, major_extent, loc=loc, ip=ip
        )
        dst_u8[(local_major, scale_col * sf_vec_size, 0)] = src_u8[phys]
        yield_out()


def stage_mxf4nvf4_sfb_tma_physical_to_tiled_smem(
    tiled_mma: cute.TiledMma,
    sSFB_tma_physical: cute.Tensor,
    sSFB_tiled_smem: cute.Tensor,
    *,
    tile_m: int = 128,
    major_extent: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    lane_idx: Optional[cutlass.Int32] = None,
    thread_idx: Optional[cutlass.Int32] = None,
    thread_count: int = 32,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Stage SFB physical TMA SMEM into the SM120 tiled scale SMEM layout."""
    _check_default_tile(major_extent, tile_k, sf_vec_size)
    scale_k = tile_k // sf_vec_size
    tma_bytes = mxf4nvf4_scale_tma_tx_bytes(major_extent, tile_k, sf_vec_size)
    physical_bytes = mxf4nvf4_scale_physical_smem_bytes(major_extent, tile_k, sf_vec_size)
    src_u8 = cute.make_tensor(
        sSFB_tma_physical.iterator,
        cute.make_layout(physical_bytes, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    dst_u8 = cute.make_tensor(
        sSFB_tiled_smem.iterator,
        make_mxf4nvf4_sfb_smem_layout_staged(
            tiled_mma,
            (tile_m, major_extent, tile_k),
            sf_vec_size,
            1,
            loc=loc,
            ip=ip,
        ),
        loc=loc,
        ip=ip,
    )
    if thread_idx is not None:
        loop_start = thread_idx
        loop_step = thread_count
    else:
        loop_start = lane_idx if lane_idx is not None else 0
        loop_step = 32 if lane_idx is not None else 1
    for i in for_generate(loop_start, tma_bytes, loop_step, loc=loc, ip=ip):
        local_major = i // scale_k
        scale_col = i % scale_k
        phys = mxf4nvf4_scale_tma_physical_offset(
            local_major, scale_col, major_extent, loc=loc, ip=ip
        )
        dst_u8[(local_major, scale_col * sf_vec_size, 0)] = src_u8[phys]
        yield_out()


def stage_mxf4nvf4_scale_tma_physical_to_tiled_smem(
    tiled_mma: cute.TiledMma,
    sSFA_tma_physical: cute.Tensor,
    sSFB_tma_physical: cute.Tensor,
    sSFA_tiled_smem: cute.Tensor,
    sSFB_tiled_smem: cute.Tensor,
    *,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    lane_idx: Optional[cutlass.Int32] = None,
    thread_idx: Optional[cutlass.Int32] = None,
    thread_count: int = 32,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Stage physical SFA/SFB TMA storage into tiled SM120 scale SMEM."""
    stage_mxf4nvf4_sfa_tma_physical_to_tiled_smem(
        tiled_mma,
        sSFA_tma_physical,
        sSFA_tiled_smem,
        major_extent=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        sf_vec_size=sf_vec_size,
        lane_idx=lane_idx,
        thread_idx=thread_idx,
        thread_count=thread_count,
        loc=loc,
        ip=ip,
    )
    stage_mxf4nvf4_sfb_tma_physical_to_tiled_smem(
        tiled_mma,
        sSFB_tma_physical,
        sSFB_tiled_smem,
        tile_m=tile_m,
        major_extent=tile_n,
        tile_k=tile_k,
        sf_vec_size=sf_vec_size,
        lane_idx=lane_idx,
        thread_idx=thread_idx,
        thread_count=thread_count,
        loc=loc,
        ip=ip,
    )


def copy_mxf4nvf4_tiled_smem_scale_fragments(
    tiled_mma: cute.TiledMma,
    sSFA_tiled_smem: cute.Tensor,
    sSFB_tiled_smem: cute.Tensor,
    sfa_frag_dst: cute.Tensor,
    sfb_frag_dst: cute.Tensor,
    tidx: cutlass.Int32,
    k_block_idx: int,
    stage_idx: int = 0,
    *,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Copy one K64 block from tiled scale SMEM into SFA/SFB fragments."""
    _check_default_tile(tile_m, tile_k, sf_vec_size)
    _check_default_tile(tile_n, tile_k, sf_vec_size)
    stage_stride_sfa = mxf4nvf4_scale_tma_tx_bytes(tile_m, tile_k, sf_vec_size)
    stage_stride_sfb = mxf4nvf4_scale_tma_tx_bytes(tile_n, tile_k, sf_vec_size)
    scale_copy_tile_k = MXF4NVF4_MMA_SHAPE_MNK[2]
    scale_copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float8E4M3FN,
        loc=loc,
        ip=ip,
    )
    tiled_copy_sfa = cute.make_tiled_copy(
        scale_copy_atom,
        get_layoutSFA_TV(tiled_mma),
        (tile_m, scale_copy_tile_k),
        loc=loc,
        ip=ip,
    )
    sSFA_f8 = cute.recast_tensor(
        cute.make_tensor(
            sSFA_tiled_smem.iterator + stage_idx * stage_stride_sfa,
            make_mxf4nvf4_sfa_smem_layout_staged(
                tiled_mma,
                (tile_m, tile_n, tile_k),
                sf_vec_size,
                1,
                loc=loc,
                ip=ip,
            ),
            loc=loc,
            ip=ip,
        ),
        cutlass.Float8E4M3FN,
        loc=loc,
        ip=ip,
    )
    sSFB_f8 = cute.recast_tensor(
        cute.make_tensor(
            sSFB_tiled_smem.iterator + stage_idx * stage_stride_sfb,
            make_mxf4nvf4_sfb_smem_layout_staged(
                tiled_mma,
                (tile_m, tile_n, tile_k),
                sf_vec_size,
                1,
                loc=loc,
                ip=ip,
            ),
            loc=loc,
            ip=ip,
        ),
        cutlass.Float8E4M3FN,
        loc=loc,
        ip=ip,
    )
    thr_copy_sfa = tiled_copy_sfa.get_slice(tidx)
    tCsSFA = thr_copy_sfa.partition_S(
        cute.as_position_independent_swizzle_tensor(sSFA_f8, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    tCrSFA = thr_copy_sfa.retile(sfa_frag_dst, loc=loc, ip=ip)
    thr_mma = tiled_mma.get_slice(tidx)
    sfb_source_layout = thrfrg_SFB(sSFB_f8[(None, None, 0)].layout, thr_mma)
    sfb_source = cute.make_tensor(sSFB_f8.iterator, sfb_source_layout, loc=loc, ip=ip)
    thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
    thr_vnk = (thr_vmnk[0], (thr_vmnk[2], thr_vmnk[3]))
    sfb_source = sfb_source[thr_vnk, (None, None)]
    sfb_source = cute.group_modes(cute.flatten(sfb_source), 0, 2)
    sfb_source = cute.group_modes(sfb_source, 1, 3)
    cute.copy(
        tiled_copy_sfa,
        tCsSFA[(None, None, k_block_idx, 0)],
        tCrSFA[(None, None, k_block_idx)],
        loc=loc,
        ip=ip,
    )
    sfb_source_k = sfb_source[(None, None, k_block_idx)]
    sfb_dst_k = sfb_frag_dst[(None, None, k_block_idx)]
    sfb_dst_k.store(sfb_source_k.load(loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def make_mxf4nvf4_scale_fragments(
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Create SFA and SFB register fragments for bundled SM120 MXF4NVF4 MMA."""
    return (
        warp.make_mxf4nvf4_sfa_fragment(loc=loc, ip=ip),
        warp.make_mxf4nvf4_sfb_fragment(loc=loc, ip=ip),
    )


@dsl_user_op
def make_mxf4nvf4_direct_tma_scale_fragments(
    tiled_mma: cute.TiledMma,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    tidx: cutlass.Int32 | int,
    *,
    tile_shape_mnk: cute.Tile = MXF4NVF4_CTA_SHAPE_MNK,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Create bundled K128 scale fragments for direct native scale TMA SMEM."""
    _check_tuple("tile_shape_mnk", tile_shape_mnk, 3)
    tile_m, tile_n, tile_k = tile_shape_mnk
    _check_default_tile(tile_m, tile_k, sf_vec_size)
    _check_default_tile(tile_n, tile_k, sf_vec_size)
    sSFA_f8 = cute.recast_tensor(sSFA, cutlass.Float8E4M3FN, loc=loc, ip=ip)
    sSFB_f8 = cute.recast_tensor(sSFB, cutlass.Float8E4M3FN, loc=loc, ip=ip)
    sSFA_logical = cute.make_tensor(
        sSFA_f8.iterator,
        make_mxf4nvf4_sfa_smem_layout_staged(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            1,
            loc=loc,
            ip=ip,
        ),
        loc=loc,
        ip=ip,
    )
    sSFB_logical = cute.make_tensor(
        sSFB_f8.iterator,
        make_mxf4nvf4_sfb_smem_layout_staged(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            1,
            loc=loc,
            ip=ip,
        ),
        loc=loc,
        ip=ip,
    )
    thr_mma = tiled_mma.get_slice(tidx)
    return (
        partition_fragment_SFA(sSFA_logical[(None, None, 0)], thr_mma, tidx),
        partition_fragment_SFB(sSFB_logical[(None, None, 0)], thr_mma, tidx),
    )


@dsl_user_op
def make_mxf4nvf4_direct_tma_scale_fragment_source_views(
    tiled_mma: cute.TiledMma,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    tidx: cutlass.Int32 | int,
    *,
    stage_idx: int = 0,
    tile_shape_mnk: cute.Tile = MXF4NVF4_CTA_SHAPE_MNK,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    scale_smem_format: str = "physical",
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[cute.Tensor, cute.Tensor]:
    """Return per-thread source views over native physical scale-TMA SMEM."""
    _check_tuple("tile_shape_mnk", tile_shape_mnk, 3)
    tile_m, tile_n, tile_k = tile_shape_mnk
    _check_default_tile(tile_m, tile_k, sf_vec_size)
    _check_default_tile(tile_n, tile_k, sf_vec_size)
    if scale_smem_format == "interleaved":
        stage_stride_sfa = mxf4nvf4_scale_tma_tx_bytes(tile_m, tile_k, sf_vec_size)
        stage_stride_sfb = mxf4nvf4_scale_tma_tx_bytes(tile_n, tile_k, sf_vec_size)
        sfa_layout = make_mxf4nvf4_sfa_smem_layout_staged(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            1,
            loc=loc,
            ip=ip,
        )
        sfb_layout = make_mxf4nvf4_sfb_smem_layout_staged(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            1,
            loc=loc,
            ip=ip,
        )
    elif scale_smem_format == "physical":
        stage_stride_sfa = mxf4nvf4_scale_physical_smem_bytes(tile_m, tile_k, sf_vec_size)
        stage_stride_sfb = mxf4nvf4_scale_physical_smem_bytes(tile_n, tile_k, sf_vec_size)
        sfa_layout = make_mxf4nvf4_scale_tma_physical_as_tiled_smem_layout_staged(
            tile_m,
            tile_k,
            sf_vec_size,
            1,
            loc=loc,
            ip=ip,
        )
        sfb_layout = make_mxf4nvf4_scale_tma_physical_as_tiled_smem_layout_staged(
            tile_n,
            tile_k,
            sf_vec_size,
            1,
            loc=loc,
            ip=ip,
        )
    else:
        raise ValueError("scale_smem_format must be either 'physical' or 'interleaved'")
    sSFA_f8 = cute.recast_tensor(
        cute.make_tensor(
            sSFA.iterator + stage_idx * stage_stride_sfa,
            sfa_layout,
            loc=loc,
            ip=ip,
        ),
        cutlass.Float8E4M3FN,
        loc=loc,
        ip=ip,
    )
    sSFB_f8 = cute.recast_tensor(
        cute.make_tensor(
            sSFB.iterator + stage_idx * stage_stride_sfb,
            sfb_layout,
            loc=loc,
            ip=ip,
        ),
        cutlass.Float8E4M3FN,
        loc=loc,
        ip=ip,
    )
    sSFA_src = cute.as_position_independent_swizzle_tensor(sSFA_f8, loc=loc, ip=ip)
    sSFB_src = cute.as_position_independent_swizzle_tensor(sSFB_f8, loc=loc, ip=ip)
    thr_mma = tiled_mma.get_slice(tidx)
    thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)

    sfa_source_layout = thrfrg_SFA(sSFA_src[(None, None, 0)].layout, thr_mma)
    sfa_source = cute.make_tensor(
        sSFA_src.iterator,
        sfa_source_layout,
        loc=loc,
        ip=ip,
    )
    thr_vmk = (thr_vmnk[0], (thr_vmnk[1], thr_vmnk[3]))
    sfa_source = sfa_source[thr_vmk, (None, None)]
    sfa_source = cute.group_modes(cute.flatten(sfa_source), 0, 2)

    sfb_source_layout = thrfrg_SFB(sSFB_src[(None, None, 0)].layout, thr_mma)
    sfb_source = cute.make_tensor(
        sSFB_src.iterator,
        sfb_source_layout,
        loc=loc,
        ip=ip,
    )
    thr_vnk = (thr_vmnk[0], (thr_vmnk[2], thr_vmnk[3]))
    sfb_source = sfb_source[thr_vnk, (None, None)]
    sfb_source = cute.group_modes(cute.flatten(sfb_source), 0, 2)
    sfb_source = cute.group_modes(sfb_source, 1, 3)
    return sfa_source, sfb_source


@dsl_user_op
def load_mxf4nvf4_sfa_fragment(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Load a prepartitioned SFA scale view into its register fragment."""
    dst.store(src.load(loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def load_mxf4nvf4_sfb_fragment(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Load a prepartitioned SFB scale view into its register fragment."""
    dst.store(src.load(loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def issue_mxf4nvf4_native_tma_consumer_group(
    tiled_mma: cute.TiledMma,
    sA_consumer: cute.Tensor,
    sB_consumer: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    acc: cute.Tensor,
    lane_idx: cutlass.Int32,
    stage_idx: cutlass.Int32 | int = 0,
    *,
    m_atom: cutlass.Int32 | int = 0,
    n_atom: cutlass.Int32 | int = 0,
    major_extent_sfa: int = 128,
    major_extent_sfb: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    ab_smem_format: str = "packed",
    sync_between_k_blocks: bool = True,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one K128 consumer group from native-TMA staged SMEM.

    This is the compact descriptor-free path for a single local SM120
    MXF4/NVFP4 warp-MMA output atom. It loads both K64 A/B halves from the
    consumer SMEM layout, pairs each half with the matching direct scale TMA
    fragment views, and accumulates into ``acc``.
    """
    ab_smem_format = _normalize_mxf4nvf4_ab_smem_format(ab_smem_format)
    if ab_smem_format != "packed":
        raise ValueError(
            "SM120 native TMA consumer group currently supports only "
            "ab_smem_format='packed'; use the lower-level unpack LDSM helpers "
            "for unpack-SMEM experiments"
        )
    a_frag, b_frag = make_mxf4nvf4_ab_fragments_from_consumer_smem(
        tiled_mma,
        sA_consumer,
        sB_consumer,
        lane_idx,
        m_atom=m_atom,
        n_atom=n_atom,
        loc=loc,
        ip=ip,
    )
    sfa, sfb = make_mxf4nvf4_scale_fragments(loc=loc, ip=ip)
    for k_block_idx in range(2):
        load_mxf4nvf4_ab_fragments_from_consumer_smem(
            tiled_mma,
            sA_consumer,
            sB_consumer,
            a_frag,
            b_frag,
            lane_idx,
            k_block_idx,
            consumer_stage_idx=stage_idx,
            m_atom=m_atom,
            n_atom=n_atom,
            loc=loc,
            ip=ip,
        )
        sfa_src, sfb_src = make_mxf4nvf4_scale_fragment_views_from_direct_tma(
            sSFA,
            sSFB,
            k_block_idx,
            stage_idx=stage_idx,
            major_extent_sfa=major_extent_sfa,
            major_extent_sfb=major_extent_sfb,
            tile_k=tile_k,
            sf_vec_size=sf_vec_size,
            loc=loc,
            ip=ip,
        )
        load_mxf4nvf4_sfa_fragment(sfa_src, sfa, loc=loc, ip=ip)
        load_mxf4nvf4_sfb_fragment(sfb_src, sfb, loc=loc, ip=ip)
        cute.gemm(
            tiled_mma,
            acc,
            (a_frag[(None, 0, k_block_idx)], sfa),
            (b_frag[(None, 0, k_block_idx)], sfb),
            acc,
            loc=loc,
            ip=ip,
        )
        if const_expr(sync_between_k_blocks):
            cute.arch.sync_threads()


@dsl_user_op
def copy_mxf4nvf4_direct_tma_scale_fragments(
    tiled_mma: cute.TiledMma,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    sfa_frag_dst: cute.Tensor,
    sfb_frag_dst: cute.Tensor,
    tidx: cutlass.Int32,
    k_block_idx: int,
    stage_idx: int = 0,
    major_extent_sfa: int = 128,
    major_extent_sfb: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    scale_smem_format: str = "physical",
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Copy one K64 block from native scale-TMA SMEM into MMA fragments."""
    _check_default_tile(major_extent_sfa, tile_k, sf_vec_size)
    _check_default_tile(major_extent_sfb, tile_k, sf_vec_size)
    if const_expr(major_extent_sfa != major_extent_sfb):
        raise ValueError("direct scale fragment copy currently requires square CTA tiles")

    sfa_source, sfb_source = make_mxf4nvf4_direct_tma_scale_fragment_source_views(
        tiled_mma,
        sSFA,
        sSFB,
        tidx,
        stage_idx=stage_idx,
        tile_shape_mnk=(major_extent_sfa, major_extent_sfb, tile_k),
        sf_vec_size=sf_vec_size,
        scale_smem_format=scale_smem_format,
        loc=loc,
        ip=ip,
    )
    sfa_source_k = sfa_source[(None, None, k_block_idx)]
    sfb_source_k = sfb_source[(None, None, k_block_idx)]
    sfa_dst_k = sfa_frag_dst[(None, None, k_block_idx)]
    sfb_dst_k = sfb_frag_dst[(None, None, k_block_idx)]
    sfa_src_compact = cute.filter_zeros(sfa_source_k, loc=loc, ip=ip)
    sfb_src_compact = cute.filter_zeros(sfb_source_k, loc=loc, ip=ip)
    sfa_dst_compact = cute.filter_zeros(sfa_dst_k, loc=loc, ip=ip)
    sfb_dst_compact = cute.filter_zeros(sfb_dst_k, loc=loc, ip=ip)
    sfa_dst_compact.store(sfa_src_compact.load(loc=loc, ip=ip), loc=loc, ip=ip)
    sfb_dst_compact.store(sfb_src_compact.load(loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def load_mxf4nvf4_direct_tma_k_block_fragments(
    tiled_mma: cute.TiledMma,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    sfa_frag: cute.Tensor,
    sfb_frag: cute.Tensor,
    tidx: cutlass.Int32,
    k_block_idx: int,
    stage_idx: int = 0,
    *,
    major_extent_sfa: int = 128,
    major_extent_sfb: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    scale_first: bool = False,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Load one SM120 direct-TMA K64 A/B/scale block into MMA fragments."""
    if const_expr(scale_first):
        copy_mxf4nvf4_direct_tma_scale_fragments(
            tiled_mma,
            sSFA,
            sSFB,
            sfa_frag,
            sfb_frag,
            tidx,
            k_block_idx,
            stage_idx=stage_idx,
            major_extent_sfa=major_extent_sfa,
            major_extent_sfb=major_extent_sfb,
            tile_k=tile_k,
            sf_vec_size=sf_vec_size,
            loc=loc,
            ip=ip,
        )
    cute.copy(
        tiled_copy_a,
        tCsA[(None, None, k_block_idx, stage_idx)],
        tCrA[(None, None, k_block_idx)],
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tiled_copy_b,
        tCsB[(None, None, k_block_idx, stage_idx)],
        tCrB[(None, None, k_block_idx)],
        loc=loc,
        ip=ip,
    )
    if const_expr(not scale_first):
        copy_mxf4nvf4_direct_tma_scale_fragments(
            tiled_mma,
            sSFA,
            sSFB,
            sfa_frag,
            sfb_frag,
            tidx,
            k_block_idx,
            stage_idx=stage_idx,
            major_extent_sfa=major_extent_sfa,
            major_extent_sfb=major_extent_sfb,
            tile_k=tile_k,
            sf_vec_size=sf_vec_size,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def load_mxf4nvf4_direct_tma_k_block_a_scale_fragments(
    tiled_mma: cute.TiledMma,
    tiled_copy_a: cute.TiledCopy,
    tCsA: cute.Tensor,
    tCrA: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    sfa_frag: cute.Tensor,
    sfb_frag: cute.Tensor,
    tidx: cutlass.Int32,
    k_block_idx: int,
    stage_idx: int = 0,
    *,
    major_extent_sfa: int = 128,
    major_extent_sfb: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    scale_first: bool = False,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Load one K64 A fragment and matching direct-TMA scale fragments."""
    if const_expr(scale_first):
        copy_mxf4nvf4_direct_tma_scale_fragments(
            tiled_mma,
            sSFA,
            sSFB,
            sfa_frag,
            sfb_frag,
            tidx,
            k_block_idx,
            stage_idx=stage_idx,
            major_extent_sfa=major_extent_sfa,
            major_extent_sfb=major_extent_sfb,
            tile_k=tile_k,
            sf_vec_size=sf_vec_size,
            loc=loc,
            ip=ip,
        )
    cute.copy(
        tiled_copy_a,
        tCsA[(None, None, k_block_idx, stage_idx)],
        tCrA[(None, None, k_block_idx)],
        loc=loc,
        ip=ip,
    )
    if const_expr(not scale_first):
        copy_mxf4nvf4_direct_tma_scale_fragments(
            tiled_mma,
            sSFA,
            sSFB,
            sfa_frag,
            sfb_frag,
            tidx,
            k_block_idx,
            stage_idx=stage_idx,
            major_extent_sfa=major_extent_sfa,
            major_extent_sfb=major_extent_sfb,
            tile_k=tile_k,
            sf_vec_size=sf_vec_size,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def load_mxf4nvf4_direct_tma_k_block_b_group_fragment(
    tiled_copy_b: cute.TiledCopy,
    tCsB: cute.Tensor,
    tCrB: cute.Tensor,
    k_block_idx: int,
    b_group_idx: int,
    stage_idx: int = 0,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Load one B LDSM group for a K64 block."""
    cute.copy(
        tiled_copy_b,
        tCsB[(None, b_group_idx, k_block_idx, stage_idx)],
        tCrB[(None, b_group_idx, k_block_idx)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def issue_mxf4nvf4_direct_tma_eager_consumer_group(
    tiled_mma: cute.TiledMma,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    a_frag: cute.Tensor,
    b_frag: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    sfa_frag: cute.Tensor,
    sfb_frag: cute.Tensor,
    acc: cute.Tensor,
    tidx: cutlass.Int32,
    stage_idx: cutlass.Int32,
    *,
    major_extent_sfa: int = 128,
    major_extent_sfb: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one staged consumer group with eager K-block fragment loads.

    The helper loads both K64 halves from staged A/B consumer SMEM, copies the
    matching direct-TMA scale fragments, and issues the two bundled warp MMAs
    that cover one K128 CTA stage. This is retained as a comparison path; the
    primary helper uses the 79a-style copy-next / compute-current schedule.
    """
    cute.copy(
        tiled_copy_a,
        tCsA[(None, None, 0, stage_idx)],
        tCrA[(None, None, 0)],
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tiled_copy_b,
        tCsB[(None, None, 0, stage_idx)],
        tCrB[(None, None, 0)],
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tiled_copy_a,
        tCsA[(None, None, 1, stage_idx)],
        tCrA[(None, None, 1)],
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tiled_copy_b,
        tCsB[(None, None, 1, stage_idx)],
        tCrB[(None, None, 1)],
        loc=loc,
        ip=ip,
    )
    for k_block_idx in range(2):
        copy_mxf4nvf4_direct_tma_scale_fragments(
            tiled_mma,
            sSFA,
            sSFB,
            sfa_frag,
            sfb_frag,
            tidx,
            k_block_idx,
            stage_idx=stage_idx,
            major_extent_sfa=major_extent_sfa,
            major_extent_sfb=major_extent_sfb,
            tile_k=tile_k,
            sf_vec_size=sf_vec_size,
            loc=loc,
            ip=ip,
        )
        cute.gemm(
            tiled_mma,
            acc,
            (
                a_frag[(None, None, k_block_idx)],
                sfa_frag[(None, None, k_block_idx)],
            ),
            (
                b_frag[(None, None, k_block_idx)],
                sfb_frag[(None, None, k_block_idx)],
            ),
            acc,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def issue_mxf4nvf4_direct_tma_consumer_group(
    tiled_mma: cute.TiledMma,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    a_frag: cute.Tensor,
    b_frag: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    sfa_frag: cute.Tensor,
    sfb_frag: cute.Tensor,
    acc: cute.Tensor,
    tidx: cutlass.Int32,
    stage_idx: cutlass.Int32,
    *,
    major_extent_sfa: int = 128,
    major_extent_sfb: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one direct-TMA consumer group with a 79a-style K-block schedule.

    The first K64 block is loaded before compute starts. The second K64 block is
    then loaded before issuing the first MMA group, matching the copy-next /
    compute-current shape used by the C++ SM120 blockscaled mainloop.
    """
    cute.copy(
        tiled_copy_a,
        tCsA[(None, None, 0, stage_idx)],
        tCrA[(None, None, 0)],
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tiled_copy_b,
        tCsB[(None, None, 0, stage_idx)],
        tCrB[(None, None, 0)],
        loc=loc,
        ip=ip,
    )
    copy_mxf4nvf4_direct_tma_scale_fragments(
        tiled_mma,
        sSFA,
        sSFB,
        sfa_frag,
        sfb_frag,
        tidx,
        0,
        stage_idx=stage_idx,
        major_extent_sfa=major_extent_sfa,
        major_extent_sfb=major_extent_sfb,
        tile_k=tile_k,
        sf_vec_size=sf_vec_size,
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tiled_copy_a,
        tCsA[(None, None, 1, stage_idx)],
        tCrA[(None, None, 1)],
        loc=loc,
        ip=ip,
    )
    cute.copy(
        tiled_copy_b,
        tCsB[(None, None, 1, stage_idx)],
        tCrB[(None, None, 1)],
        loc=loc,
        ip=ip,
    )
    copy_mxf4nvf4_direct_tma_scale_fragments(
        tiled_mma,
        sSFA,
        sSFB,
        sfa_frag,
        sfb_frag,
        tidx,
        1,
        stage_idx=stage_idx,
        major_extent_sfa=major_extent_sfa,
        major_extent_sfb=major_extent_sfb,
        tile_k=tile_k,
        sf_vec_size=sf_vec_size,
        loc=loc,
        ip=ip,
    )
    gemm_mxf4nvf4_direct_tma_k_block(
        tiled_mma,
        acc,
        a_frag,
        b_frag,
        sfa_frag,
        sfb_frag,
        0,
        loc=loc,
        ip=ip,
    )
    gemm_mxf4nvf4_direct_tma_k_block(
        tiled_mma,
        acc,
        a_frag,
        b_frag,
        sfa_frag,
        sfb_frag,
        1,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def gemm_mxf4nvf4_direct_tma_k_block(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    a_frag: cute.Tensor,
    b_frag: cute.Tensor,
    sfa_frag: cute.Tensor,
    sfb_frag: cute.Tensor,
    k_block_idx: int,
    *,
    ab_smem_format: str = "packed",
    n_major: bool = False,
    sync_warp_before: bool = False,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one SM120 MXF4/NVFP4 K64 bundled-MMA block.

    This helper is the composable per-K-block compute primitive used by the
    higher-level SM120 direct-TMA schedules.  It keeps the logical K128 fragment
    contract intact while letting callers choose the local MMA traversal order
    independently from TMA staging.
    """
    ab_smem_format = _normalize_mxf4nvf4_ab_smem_format(ab_smem_format)
    if ab_smem_format == "unpack":
        a_frag = cute.recast_tensor(a_frag, cutlass.Int8, loc=loc, ip=ip)
        b_frag = cute.recast_tensor(b_frag, cutlass.Int8, loc=loc, ip=ip)
    if const_expr(sync_warp_before):
        cute.arch.sync_warp()
    if const_expr(n_major):
        a_block = a_frag[(None, None, k_block_idx)]
        b_block = b_frag[(None, None, k_block_idx)]
        sfa_block = sfa_frag[(None, None, k_block_idx)]
        sfb_block = sfb_frag[(None, None, k_block_idx)]
        a_tile_size = 16 if const_expr(ab_smem_format == "unpack") else 32
        b_tile_size = 8 if const_expr(ab_smem_format == "unpack") else 16
        a_tiles = cute.size(a_block) // a_tile_size
        b_tiles = cute.size(b_block) // b_tile_size
        for n_idx in range(b_tiles):
            for m_idx in range(a_tiles):
                warp.mma_mxf4nvf4(
                    tiled_mma,
                    acc[(None, m_idx, n_idx)],
                    (a_block[(None, m_idx)], sfa_block[(None, m_idx)]),
                    (b_block[(None, n_idx)], sfb_block[(None, n_idx)]),
                    acc[(None, m_idx, n_idx)],
                    loc=loc,
                    ip=ip,
                )
    else:
        cute.gemm(
            tiled_mma,
            acc,
            (
                a_frag[(None, None, k_block_idx)],
                sfa_frag[(None, None, k_block_idx)],
            ),
            (
                b_frag[(None, None, k_block_idx)],
                sfb_frag[(None, None, k_block_idx)],
            ),
            acc,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def gemm_mxf4nvf4_direct_tma_k_block_b_group(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    a_frag: cute.Tensor,
    b_frag: cute.Tensor,
    sfa_frag: cute.Tensor,
    sfb_frag: cute.Tensor,
    k_block_idx: int,
    b_group_idx: int,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue the two N-major MMA columns covered by one packed B load group."""
    a_block = a_frag[(None, None, k_block_idx)]
    b_block = b_frag[(None, None, k_block_idx)]
    sfa_block = sfa_frag[(None, None, k_block_idx)]
    sfb_block = sfb_frag[(None, None, k_block_idx)]
    a_tiles = cute.size(a_block) // 32
    n_start = b_group_idx * 2
    for n_offset in range(2):
        n_idx = n_start + n_offset
        for m_idx in range(a_tiles):
            warp.mma_mxf4nvf4(
                tiled_mma,
                acc[(None, m_idx, n_idx)],
                (a_block[(None, m_idx)], sfa_block[(None, m_idx)]),
                (b_block[(None, n_idx)], sfb_block[(None, n_idx)]),
                acc[(None, m_idx, n_idx)],
                loc=loc,
                ip=ip,
            )


@dsl_user_op
def issue_mxf4nvf4_direct_tma_pingpong_consumer_group(
    tiled_mma: cute.TiledMma,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    a_frag: cute.Tensor,
    b_frag: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    sfa_frag: cute.Tensor,
    sfb_frag: cute.Tensor,
    acc: cute.Tensor,
    tidx: cutlass.Int32,
    stage_idx: cutlass.Int32,
    *,
    major_extent_sfa: int = 128,
    major_extent_sfb: int = 128,
    tile_k: int = 128,
    sf_vec_size: int = MXF4NVF4_SCALE_VEC_SIZE,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Compatibility alias for the primary pingpong consumer group helper."""
    issue_mxf4nvf4_direct_tma_consumer_group(
        tiled_mma,
        tiled_copy_a,
        tiled_copy_b,
        tCsA,
        tCsB,
        tCrA,
        tCrB,
        a_frag,
        b_frag,
        sSFA,
        sSFB,
        sfa_frag,
        sfb_frag,
        acc,
        tidx,
        stage_idx,
        major_extent_sfa=major_extent_sfa,
        major_extent_sfb=major_extent_sfb,
        tile_k=tile_k,
        sf_vec_size=sf_vec_size,
        loc=loc,
        ip=ip,
    )


__all__: tuple[str, ...] = ()
