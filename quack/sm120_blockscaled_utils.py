# Copyright (c) 2026, QuACK team.
"""Host-side helpers for SM120 NVFP4 blockscaled GEMM."""

from __future__ import annotations

from typing import Tuple

import torch


_FP4_E2M1FN_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def _validate_cuda_tensor(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")


def validate_sm120_nvfp4_ab_storage(
    packed_tensor: torch.Tensor,
    *,
    logical_k: int,
    major_extent: int,
    batch_extent: int,
) -> None:
    _validate_cuda_tensor(packed_tensor, "packed_tensor")
    if packed_tensor.dtype != torch.float4_e2m1fn_x2:
        raise TypeError(f"packed_tensor must be torch.float4_e2m1fn_x2, got {packed_tensor.dtype}")
    if logical_k <= 0 or logical_k % 128 != 0:
        raise ValueError(f"logical_k must be a positive multiple of 128, got {logical_k}")
    expected_shape = (major_extent, logical_k // 2, batch_extent)
    if tuple(packed_tensor.shape) != expected_shape:
        raise ValueError(
            f"packed_tensor shape must be {expected_shape}, got {tuple(packed_tensor.shape)}"
        )
    expected_stride = (logical_k // 2, 1, major_extent * logical_k // 2)
    stride = tuple(packed_tensor.stride())
    stride_ok = stride == expected_stride or (
        batch_extent == 1 and stride[:2] == expected_stride[:2]
    )
    if not stride_ok:
        raise ValueError(
            "packed_tensor must be K-major contiguous packed storage with stride "
            f"{expected_stride}, got {stride}"
        )
    if packed_tensor.data_ptr() % 32 != 0:
        raise ValueError("SM120 NVFP4 A/B base pointer must be 32B aligned")


def validate_sm120_nvfp4_d_storage(
    d: torch.Tensor,
    *,
    m: int,
    n: int,
    l: int,
) -> None:
    _validate_cuda_tensor(d, "D")
    if d.dtype != torch.bfloat16:
        raise TypeError(f"D must be torch.bfloat16, got {d.dtype}")
    expected_shape = (m, n, l)
    if tuple(d.shape) != expected_shape:
        raise ValueError(f"D shape must be {expected_shape}, got {tuple(d.shape)}")
    expected_stride = (n, 1, m * n)
    stride = tuple(d.stride())
    stride_ok = stride == expected_stride or (l == 1 and stride[:2] == expected_stride[:2])
    if not stride_ok:
        raise ValueError(f"D must be N-major with stride {expected_stride}, got {stride}")
    if d.data_ptr() % 16 != 0:
        raise ValueError("SM120 NVFP4 D base pointer must be at least 16B aligned")


def _check_major_tile(major_extent: int, major_tile: int, tile_major: int) -> int:
    major_offset = major_tile * tile_major
    if major_tile < 0 or major_offset + tile_major > major_extent:
        raise ValueError(f"major_tile={major_tile} is outside major_extent={major_extent}")
    return major_offset


def sm120_nvfp4_scale_pages(logical_k: int, sf_vec_size: int = 16) -> tuple[int, int, int]:
    if sf_vec_size != 16:
        raise ValueError(f"SM120 NVFP4 requires sf_vec_size=16, got {sf_vec_size}")
    if logical_k <= 0 or logical_k % 128 != 0:
        raise ValueError(f"logical_k must be a positive multiple of 128, got {logical_k}")
    logical_scale_cols = ceil_div(logical_k, sf_vec_size)
    physical_scale_cols = round_up(logical_scale_cols, 16)
    physical_scale_pages = max(physical_scale_cols // 16, 2)
    return logical_scale_cols, physical_scale_cols, physical_scale_pages


def sm120_nvfp4_scale_physical_offset(
    major: int,
    scale_col: int,
    major_extent: int,
) -> int:
    physical_major_extent = max(major_extent, 128)
    payload_idx = scale_col * physical_major_extent + major
    payload_chunk = payload_idx // 16
    payload_byte = payload_idx % 16
    physical_chunk = payload_chunk ^ ((payload_chunk >> 3) & 0x7)
    return physical_chunk * 16 + payload_byte


def sm120_nvfp4_scale_interleaved_size(
    logical_k: int,
    major_extent: int,
    batch_extent: int,
) -> tuple[int, int, int]:
    if logical_k <= 0 or logical_k % 16 != 0:
        raise ValueError(f"logical_k must be a positive multiple of 16, got {logical_k}")
    if major_extent % 128 != 0:
        raise ValueError(f"major_extent must be a multiple of 128, got {major_extent}")
    if batch_extent <= 0:
        raise ValueError(f"batch_extent must be positive, got {batch_extent}")
    logical_cols = logical_k // 16
    if logical_cols % 4 != 0:
        raise ValueError(f"logical_k / 16 must be a multiple of 4, got {logical_cols}")
    major_tiles = major_extent // 128
    scale_tiles = ceil_div(logical_cols, 4)
    return logical_cols, scale_tiles, major_tiles * scale_tiles * 512 * batch_extent


def sm120_nvfp4_scale_interleaved_offset(
    major: int,
    scale_col: int,
    *,
    logical_k: int,
    major_extent: int,
    batch_idx: int = 0,
) -> int:
    logical_cols, scale_tiles, _ = sm120_nvfp4_scale_interleaved_size(
        logical_k, major_extent, batch_idx + 1
    )
    if scale_col < 0 or scale_col >= logical_cols:
        raise ValueError(f"scale_col={scale_col} is outside logical_cols={logical_cols}")
    if major < 0 or major >= major_extent:
        raise ValueError(f"major={major} is outside major_extent={major_extent}")
    major_tiles = major_extent // 128
    major_tile = major // 128
    major_in_tile = major - major_tile * 128
    major_row = major_in_tile % 32
    major_quad = major_in_tile // 32
    scale_tile = scale_col // 4
    scale_quad = scale_col - scale_tile * 4
    l_stride = scale_tiles * major_tiles * 512
    return (
        batch_idx * l_stride
        + scale_tile * major_tiles * 512
        + major_tile * 512
        + major_row * 16
        + major_quad * 4
        + scale_quad
    )


def validate_sm120_nvfp4_scale_storage(
    scale_tensor: torch.Tensor,
    *,
    logical_k: int,
    major_extent: int,
    batch_extent: int,
) -> tuple[int, int, int]:
    _validate_cuda_tensor(scale_tensor, "scale_tensor")
    if scale_tensor.dtype != torch.float8_e4m3fn:
        raise TypeError(f"scale_tensor must be torch.float8_e4m3fn, got {scale_tensor.dtype}")
    logical_cols, physical_cols, pages = sm120_nvfp4_scale_pages(logical_k)
    _logical_cols, _scale_tiles, interleaved_size = sm120_nvfp4_scale_interleaved_size(
        logical_k, major_extent, batch_extent
    )
    if scale_tensor.ndim != 1:
        raise ValueError("SM120 NVFP4 scale storage must be compact 1D interleaved FP8 storage")
    if scale_tensor.numel() != interleaved_size:
        raise ValueError(
            "interleaved scale_tensor storage must have "
            f"{interleaved_size} elements, got {scale_tensor.numel()}"
        )
    if scale_tensor.stride() != (1,):
        raise ValueError(
            f"interleaved scale_tensor must be contiguous, got {scale_tensor.stride()}"
        )
    if scale_tensor.data_ptr() % 16 != 0:
        raise ValueError("SM120 NVFP4 scale base pointer must be 16B aligned")
    return logical_cols, physical_cols, pages


def copy_sm120_nvfp4_scale_blocks_to_storage(
    scale_tensor: torch.Tensor,
    block_values: torch.Tensor,
    *,
    logical_k: int,
) -> None:
    if not block_values.is_cuda:
        raise ValueError("block_values must be a CUDA tensor")
    major_extent, logical_cols, batch_extent = block_values.shape
    expected_cols = ceil_div(logical_k, 16)
    if logical_cols != expected_cols:
        raise ValueError(f"block_values shape[1] must be {expected_cols}, got {logical_cols}")
    validate_sm120_nvfp4_scale_storage(
        scale_tensor,
        logical_k=logical_k,
        major_extent=major_extent,
        batch_extent=batch_extent,
    )
    ref_u8 = block_values.to(torch.float8_e4m3fn).view(torch.uint8)
    storage_u8 = scale_tensor.view(torch.uint8)
    storage_u8.zero_()
    _logical_cols, scale_tiles, _storage_size = sm120_nvfp4_scale_interleaved_size(
        logical_k, major_extent, batch_extent
    )
    major_tiles = major_extent // 128
    storage_view = storage_u8.view(batch_extent, scale_tiles, major_tiles, 32, 4, 4)
    for batch_idx in range(batch_extent):
        for col in range(logical_cols):
            scale_tile = col // 4
            scale_quad = col - scale_tile * 4
            for major_tile in range(major_tiles):
                major_start = major_tile * 128
                src = ref_u8[major_start : major_start + 128, col, batch_idx]
                storage_view[batch_idx, scale_tile, major_tile, :, :, scale_quad].copy_(
                    src.view(4, 32).transpose(0, 1)
                )


def make_sm120_nvfp4_ab_metadata_tensor(
    *, device, logical_k: int = 128, major_extent: int = 128, batch_extent: int = 2
) -> torch.Tensor:
    """Create rank-preserving dummy metadata for SM120 A/B TMA atom construction."""
    return torch.empty(
        (logical_k, major_extent, batch_extent),
        dtype=torch.uint8,
        device=device,
    )


def make_sm120_nvfp4_scale_metadata_tensor(*, device) -> torch.Tensor:
    """Create rank-preserving dummy metadata for SM120 scale TMA atom construction."""
    return torch.empty((1024,), dtype=torch.uint8, device=device)


def create_sm120_nvfp4_ab_tensor(
    l: int, major: int, k: int, *, fill_byte: int | None = None
) -> torch.Tensor:
    """Create SM120 packed-K NVFP4 A/B storage shaped ``(major, k / 2, l)``."""
    if l <= 0:
        raise ValueError(f"l must be positive, got {l}")
    if major <= 0:
        raise ValueError(f"major must be positive, got {major}")
    if k <= 0 or k % 128 != 0:
        raise ValueError(f"k must be a positive multiple of 128, got {k}")
    storage = torch.empty((l, major, k // 2), dtype=torch.float4_e2m1fn_x2, device="cuda")
    packed = storage.permute(1, 2, 0)
    if fill_byte is not None:
        if fill_byte < 0 or fill_byte > 255:
            raise ValueError(f"fill_byte must fit in uint8, got {fill_byte}")
        packed.view(torch.uint8).fill_(fill_byte)
    return packed


def create_sm120_nvfp4_tensorfill_like_ab_tensor(
    l: int, major: int, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create bounded random non-zero FP4 A/B data close to 79a TensorFillRandomUniform.

    79a uses TensorFillRandomUniform with the FP4 input scope [-2, 2].  Keep the
    same value range but avoid FP4 zero codes so performance checks cannot turn
    into accidental zero-skipping checks.
    """
    packed = create_sm120_nvfp4_ab_tensor(l, major, k)
    magnitudes = torch.randint(1, 5, (major, k, l), device="cuda", dtype=torch.uint8)
    signs = torch.randint(0, 2, (major, k, l), device="cuda", dtype=torch.uint8) << 3
    codes = magnitudes | signs
    packed.view(torch.uint8).copy_(codes[:, 0::2, :] | (codes[:, 1::2, :] << 4))
    table = torch.tensor(_FP4_E2M1FN_VALUES, device="cuda", dtype=torch.float32)
    return table[codes.long()], packed


def create_sm120_nvfp4_scale_tensor(l: int, mn: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create random SM120 interleaved scale storage and expanded FP32 reference."""
    logical_cols, _scale_tiles, storage_size = sm120_nvfp4_scale_interleaved_size(k, mn, l)
    ref_blocks = torch.randint(1, 4, (mn, logical_cols, l), device="cuda").float()
    storage = torch.zeros(storage_size, dtype=torch.float8_e4m3fn, device="cuda")
    copy_sm120_nvfp4_scale_blocks_to_storage(storage, ref_blocks, logical_k=k)
    ref = (
        ref_blocks.permute(0, 2, 1)
        .unsqueeze(-1)
        .expand(mn, l, logical_cols, 16)
        .reshape(mn, l, logical_cols * 16)
        .permute(0, 2, 1)
    )[:, :k, :]
    return ref, storage


def create_sm120_nvfp4_tensorfill_like_scale_tensor(
    l: int, mn: int, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create bounded random positive non-zero E4M3 scale data."""
    logical_cols, _scale_tiles, storage_size = sm120_nvfp4_scale_interleaved_size(k, mn, l)
    choices = torch.tensor([0.5, 1.0], device="cuda", dtype=torch.float32)
    indices = torch.randint(0, choices.numel(), (mn, logical_cols, l), device="cuda")
    ref_blocks = choices[indices]
    storage = torch.zeros(storage_size, dtype=torch.float8_e4m3fn, device="cuda")
    copy_sm120_nvfp4_scale_blocks_to_storage(storage, ref_blocks, logical_k=k)
    ref = (
        ref_blocks.permute(0, 2, 1)
        .unsqueeze(-1)
        .expand(mn, l, logical_cols, 16)
        .reshape(mn, l, logical_cols * 16)
        .permute(0, 2, 1)
    )[:, :k, :]
    return ref, storage
