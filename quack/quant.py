# Copyright (c) 2025, Quack authors.

"""Blockwise FP8 quantization.

For each block of ``block_size`` elements, the quant kernel computes amax, derives a
power-of-2 scale, and converts to FP8. Input is logically reshaped to
``(num_blocks, block_size)`` so each "row" is one quant block, reusing the
``ReductionBase`` tiled_copy + ``row_reduce`` pattern. Dequantization is just a per-block
scale broadcast, so it's plain torch (``blockwise_dequant``) — no kernel.
"""

import math
from typing import Optional, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass._mlir.dialects import arith, llvm
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import T

import torch
from torch import Tensor

import quack.copy_utils as copy_utils
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.reduce import row_reduce
from quack.reduction_base import ReductionBase
from quack.fast_math import FastDivmod
from quack.autotuner import autotune, AutotuneConfig
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.cache import jit_cache

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


class BlockwiseQuant(ReductionBase):
    """Blockwise FP8 quantization kernel.

    For each block of `block_size` elements, computes amax, derives a power-of-2 scale,
    and converts to FP8. Input is logically reshaped to (num_blocks, block_size) so each
    "row" is one quant block, reusing the ReductionBase tiled_copy + row_reduce pattern.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        block_size: int,
        threads_per_row: Optional[int] = None,
        num_threads: Optional[int] = None,
    ):
        super().__init__(dtype, block_size, stage=1)
        self.cluster_n = 1
        self.fp8_max = FP8_MAX
        self._threads_per_row_val = threads_per_row
        self._num_threads_val = num_threads

    def _threads_per_row(self):
        if self._threads_per_row_val is not None:
            return self._threads_per_row_val
        N = self.N
        # Cap at WARP_SIZE so the entire reduction stays within a warp (no smem needed)
        for limit, threads in [(64, 8), (128, 16)]:
            if N <= limit:
                return threads
        return 32

    def _num_threads(self):
        if self._num_threads_val is not None:
            return self._num_threads_val
        return 128 if self.N <= 16384 else 256

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        mScale: cute.Tensor,
        mScaleRowIdx: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        largest_dtype_width = const_expr(max(mX.element_type.width, mO.element_type.width))
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        # mScale is 2D (dst_rows, num_blocks_N). Each flat block row i is mapped to
        # (m, bj) = divmod(i, num_blocks_N) and written as mScale[m, bj], so the
        # tensor's gmem strides decide the layout: row-major or col-major
        # (transposed) scales are both handled with no kernel change. When mScaleRowIdx
        # is given, the block-row's scale is scattered to mScale[mScaleRowIdx[m], bj]
        # (e.g. dQaccum-padded destination) instead of mScale[m, bj].
        bn_divmod = FastDivmod(mScale.shape[1])
        self.kernel(
            mX, mO, mScale, mScaleRowIdx, tiler_mn, tiled_copy, threads_per_row, bn_divmod
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        mScale: cute.Tensor,
        mScaleRowIdx: cute.Tensor | None,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
        bn_divmod: FastDivmod,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, gO, cX = [cute.local_tile(mT, tiler_mn, (bidx, 0)) for mT in (mX, mO, idX)]

        thr_copy_X = tiled_copy.get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXgO = thr_copy_X.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrX, tXrO = [cute.make_rmem_tensor_like(t) for t in (tXgX, tXgO)]

        is_even_N = const_expr(shape[1] == tiler_mn[1])
        tXpX = (
            copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        copy = partial(copy_utils.copy, pred=tXpX)

        # Direct gmem -> rmem (no smem)
        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXrX)
        x = tXrX.load().to(cute.Float32)

        # Compute abs in rmem, then single MAX reduction
        tXrAbs = cute.make_rmem_tensor_like(tXrX, cute.Float32)
        for i in cutlass.range_constexpr(const_expr(cute.size(tXrAbs))):
            tXrAbs[i] = mlir_math.absf(Float32(tXrX[i]))
        abs_x = tXrAbs.load()
        # Warp-only reduction (no smem buffer needed, threads_per_row <= WARP_SIZE)
        amax = row_reduce(abs_x, cute.ReductionOp.MAX, threads_per_row, init_val=1e-5)

        # quant_scale = FP8_MAX / amax, truncated to power-of-2 (zero mantissa bits)
        quant_scale = self.fp8_max / amax
        scale_bits = llvm.bitcast(T.i32(), quant_scale)
        scale_bits = scale_bits & 0xFF800000
        quant_scale = arith.bitcast(T.f32(), scale_bits)
        dequant_scale = 1.0 / quant_scale
        y = x * quant_scale
        tXrO.store(y.to(tXrO.element_type))
        # Quantize and store FP8 output
        if row < shape[0]:
            copy(tXrO, tXgO)

        if tXcX[0][1] == 0 and row < shape[0]:
            m, bj = divmod(row, bn_divmod)
            if const_expr(mScaleRowIdx is not None):
                m = mScaleRowIdx[m]
            mScale[m, bj] = dequant_scale


@jit_cache
def _compile_quant_fwd(
    dtype,
    out_dtype,
    block_size,
    scale_transpose=False,
    has_scatter=False,
    threads_per_row=None,
    num_threads=None,
):
    batch_sym = cute.sym_int()
    all_dtypes = [dt for dt in [dtype, out_dtype] if dt is not None]
    div = math.gcd(block_size, *(128 // dt.width for dt in all_dtypes))
    x_cute = fake_tensor(dtype, (batch_sym, block_size), div)
    o_cute = fake_tensor(out_dtype, (batch_sym, block_size), div)
    # Scale is (dst_rows, num_blocks_N). leading_dim picks the contiguous axis:
    # last dim (row-major) by default, first dim (col-major / transposed) otherwise.
    m_sym, bn_sym = cute.sym_int(), cute.sym_int()
    scale_cute = fake_tensor(Float32, (m_sym, bn_sym), leading_dim=0 if scale_transpose else 1)
    # scale_row_idx: (M,) int32 per-input-row destination row into the scale tensor, so the
    # kernel scatters each block's scale straight to its dQaccum-padded position.
    scale_row_idx_cute = fake_tensor(cutlass.Int32, (cute.sym_int(),)) if has_scatter else None
    return cute.compile(
        BlockwiseQuant(dtype, block_size, threads_per_row, num_threads),
        x_cute,
        o_cute,
        scale_cute,
        scale_row_idx_cute,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _quant_configs(block_size: int, vecsize: int) -> list[AutotuneConfig]:
    """Valid (threads_per_row, num_threads) pairs for a given block_size and vecsize."""
    return [AutotuneConfig(config=(4, 64))]
    configs = []
    for num_threads in [64, 128, 256]:
        for threads_per_row in [4, 8, 16, 32]:
            if threads_per_row > num_threads:
                continue
            # tiler_mn[1] must equal block_size
            elems_per_vec = block_size // vecsize
            if elems_per_vec % threads_per_row != 0:
                continue
            configs.append(AutotuneConfig(config=(threads_per_row, num_threads)))
    return configs


@autotune(
    configs=_quant_configs(block_size=128, vecsize=8),
    key=[],
    restore_value=["out", "scale"],
)
def _blockwise_quant_launch(x, out, scale, scale_row_idx, block_size, config):
    threads_per_row, num_threads = config
    dtype = torch2cute_dtype_map[x.dtype]
    out_dtype = torch2cute_dtype_map[out.dtype]
    # Col-major scale has its last dim non-contiguous (stride != 1).
    scale_transpose = scale.stride(-1) != 1
    has_scatter = scale_row_idx is not None
    compiled = _compile_quant_fwd(
        dtype, out_dtype, block_size, scale_transpose, has_scatter, threads_per_row, num_threads
    )
    compiled(x, out, scale, scale_row_idx)


@torch.library.custom_op(
    "cutedsl::_blockwise_quant",
    mutates_args=("out", "scale"),
    device_types="cuda",
    schema="(Tensor x, Tensor(a1!) out, Tensor(a2!) scale, Tensor? scale_row_idx, int block_size) -> ()",
)
def _blockwise_quant(
    x: Tensor,
    out: Tensor,
    scale: Tensor,
    scale_row_idx: Tensor | None,
    block_size: int,
) -> None:
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, f"Unsupported dtype: {x.dtype}"
    _, N = x.shape
    assert N % block_size == 0, f"N={N} must be divisible by block_size={block_size}"
    x_flat = x.reshape(-1, block_size)
    out_flat = out.reshape(-1, block_size)
    # scale stays 2D (dst_rows, N // block_size); its strides carry the layout (row/col-major).
    _blockwise_quant_launch(x_flat, out_flat, scale, scale_row_idx, block_size)


def blockwise_quant(
    src: Tensor,
    block_size: int = 128,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_transpose: bool = False,
    scale_row_idx: Tensor | None = None,
    scale_rows: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Blockwise FP8 quantization.

    Args:
        src: Input tensor of shape (M, N), N must be divisible by block_size.
        block_size: Number of elements per quantization block.
        quant_dtype: Target FP8 dtype.
        scale_transpose: If True, store the scale tensor column-major (M-contiguous,
            strides (1, M)) instead of the default row-major (N//block_size-contiguous).
            The logical shape is (M, N // block_size) either way; only the memory
            layout differs. Useful for feeding the SM90 blockscaled GEMM, which wants
            M-contiguous activation scales, without a follow-up `.contiguous()` copy.
        scale_row_idx: optional (M,) int32 destination-row map. When given, input row m's
            scale is written to scale-row ``scale_row_idx[m]`` instead of ``m`` (a fused
            scatter, e.g. straight into a dQaccum-padded SFA). Requires ``scale_rows``.
        scale_rows: number of rows in the (scattered) scale buffer. Required with
            ``scale_row_idx``; the buffer is allocated with this many rows and rows not
            targeted by ``scale_row_idx`` are left uninitialized.

    Returns:
        (quantized, scale). ``quantized`` has shape (M, N) in quant_dtype. Without
        ``scale_row_idx``, ``scale`` is (M, N // block_size); with it, (scale_rows,
        N // block_size). To dequantize: src ≈ quantized.float() * scale.unsqueeze(-1).

    For expert-grouped input needing a dQaccum-padded SFA, either pass ``scale_row_idx`` to
    scatter in-kernel, or quantize unpadded then use ``grouped_scale_to_dqaccum`` /
    ``permute_scale_to_dqaccum``.
    """
    N = src.shape[-1]
    orig_shape = src.shape
    src = src.view(-1, N)
    M = src.shape[0]
    assert N % block_size == 0
    if scale_row_idx is not None:
        assert scale_rows is not None, "scale_rows is required when scale_row_idx is given"
    rows = scale_rows if scale_row_idx is not None else M
    out = torch.empty_like(src, dtype=quant_dtype)
    if scale_transpose:
        # (rows, N // block_size) but column-major: each block-column is contiguous.
        scale = torch.empty(
            N // block_size, rows, device=src.device, dtype=torch.float32
        ).transpose(0, 1)
    else:
        scale = torch.empty(rows, N // block_size, device=src.device, dtype=torch.float32)
    _blockwise_quant(src, out, scale, scale_row_idx, block_size)
    return out.view(*orig_shape), scale.view(*orig_shape[:-1], N // block_size)


def dqaccum_total_padded_m(total_m: int, num_experts: int) -> int:
    """Row count of the dQaccum-padded SFA buffer: (ceil(total_m/128) + (E-1)) tiles.

    Proven sufficient in AI/varlen_blockscaled_sf_layout.md. Depends only on Python
    ints (tensor shapes / expert count), so it introduces no host<->device sync.
    """
    return ((total_m + 127) // 128 + (num_experts - 1)) * 128


@torch.compile(dynamic=True)
def _scatter_scale_to_dqaccum(
    dense_scale: Tensor,
    grouped_pos: Tensor,
    src_row: Tensor,
    cu_seqlens_m: Tensor,
    total_padded_M: int,
    m_contiguous: bool,
) -> Tensor:
    """Scatter block scales into the dQaccum-padded, expert-grouped SFA (fully on-device).

    For each source element i, ``sfa[dest[i]] = dense_scale[src_row[i]]`` where ``dest[i]``
    is the padded destination of unpadded grouped position ``grouped_pos[i]``: expert ``b``'s
    rows begin at ``(cu_seqlens_m[b]//128 + b)*128``, so ``dest = base_b + (g - cu[b])``.
    ``b`` is found with ``searchsorted`` — no host sync (unlike a per-expert Python loop).
    """
    cu = cu_seqlens_m.long()
    g = grouped_pos.long()
    b = torch.searchsorted(cu, g, right=True) - 1
    dest = (cu[b] // 128 + b) * 128 + (g - cu[b])
    SF_K = dense_scale.shape[1]
    shape = (SF_K, total_padded_M) if m_contiguous else (total_padded_M, SF_K)
    # torch.empty (no fill): padding rows are uninitialized. Their GEMM output rows are
    # discarded and rows don't cross-contaminate, so values there don't affect results.
    sfa = torch.empty(shape, device=dense_scale.device, dtype=dense_scale.dtype)
    if m_contiguous:
        sfa = sfa.transpose(0, 1)  # (total_padded_M, SF_K), M-contiguous
    sfa[dest] = dense_scale[src_row.long()]
    return sfa


def permute_scale_to_dqaccum(
    token_scale: Tensor,
    s_reverse_scatter_idx: Tensor,
    cu_seqlens_m: Tensor,
    topk: int,
    m_contiguous: bool = True,
) -> Tensor:
    """Scatter token-order block scales into the dQaccum-padded, expert-grouped SFA.

    For the MoE up-projection (gather_A): activations stay in token order and are
    index-gathered by the GEMM, so only the scales are physically permuted + padded and
    TMA-loaded per expert. Fully on-device (no host sync) — the padded destination is
    derived from ``cu_seqlens_m`` via ``searchsorted``. Companion to ``blockwise_quant``.

    Args:
        token_scale: ``(T, SF_K)`` token-order block scales (e.g. from ``blockwise_quant``).
        s_reverse_scatter_idx: ``(T*topk,)`` entry ``t*topk+k`` -> *unpadded* expert-grouped
            position (from the routing metadata).
        cu_seqlens_m: ``(E+1,)`` expert offsets (``expert_frequency_offset``).
        topk: experts per token.
        m_contiguous: store M-contiguous (stride ``(1, total_padded_M)``), the SFA TMA's preference.
    """
    TK = s_reverse_scatter_idx.shape[0]
    num_experts = cu_seqlens_m.shape[0] - 1
    total_padded_M = dqaccum_total_padded_m(TK, num_experts)
    token_of_entry = torch.arange(TK, device=token_scale.device) // topk
    return _scatter_scale_to_dqaccum(
        token_scale,
        s_reverse_scatter_idx,
        token_of_entry,
        cu_seqlens_m,
        total_padded_M,
        m_contiguous,
    )


def grouped_scale_to_dqaccum(
    grouped_scale: Tensor,
    cu_seqlens_m: Tensor,
    m_contiguous: bool = True,
) -> Tensor:
    """dQaccum-pad block scales that are already in expert-grouped order (MoE down-proj).

    ``grouped_scale`` is ``(TK, SF_K)`` with row ``g`` belonging to expert ``b``
    (``cu_seqlens_m[b] <= g < cu_seqlens_m[b+1]``); each row is scattered to its 128-aligned
    padded position. Fully on-device (no host sync).
    """
    TK = grouped_scale.shape[0]
    num_experts = cu_seqlens_m.shape[0] - 1
    total_padded_M = dqaccum_total_padded_m(TK, num_experts)
    g = torch.arange(TK, device=grouped_scale.device)
    return _scatter_scale_to_dqaccum(
        grouped_scale, g, g, cu_seqlens_m, total_padded_M, m_contiguous
    )


def quant_ref(
    src: Tensor,
    block_size: int = 128,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
    min_scale: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Reference implementation for blockwise FP8 quantization."""
    M, N = src.shape
    fp8_max = torch.finfo(quant_dtype).max
    src_f32 = src.reshape(-1, block_size).to(torch.float32)
    amax = torch.amax(torch.abs(src_f32), dim=1).clamp(min=min_scale)
    # Truncate scale to power-of-2
    quant_scale = fp8_max / amax
    scale_bits = quant_scale.view(torch.int32) & 0xFF800000
    quant_scale = scale_bits.view(torch.float32)
    dequant_scale = 1.0 / quant_scale
    quantized = (src_f32 * quant_scale.unsqueeze(1)).to(quant_dtype)
    return quantized.reshape(M, N), dequant_scale.reshape(M, N // block_size)


# ---------------------------------------------------------------------------
# Blockwise Dequantization
# ---------------------------------------------------------------------------


@torch.compile
def blockwise_dequant(
    src: Tensor,
    scale: Tensor,
    block_size: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Reference implementation for blockwise FP8 dequantization."""
    M, N = src.shape
    src_f32 = src.reshape(-1, block_size).to(torch.float32)
    dequantized = src_f32 * scale.reshape(-1, 1)
    return dequantized.to(out_dtype).reshape(M, N)
