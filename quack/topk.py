# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Tri Dao.

import math
import torch
from typing import Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass import const_expr

import quack.utils as utils
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.sort.bitonic_sort import bitonic_topk


class TopK:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, k: int):
        self.dtype = dtype
        self.N = N
        self.vecsize = 128 // dtype.width
        self.k = k
        assert N == 2 ** int(math.log2(N)), "N must be a power of 2"
        assert k == 2 ** int(math.log2(k)), "N must be a power of 2"
        assert k <= 128
        assert N <= 4096

    def _calculate_threads_per_row(self):
        # we want num_elems_per_thread >= self.k
        # and each thread can handle at most 64 elements
        N = self.N
        num_threads_per_row = max(min(N // self.k, 32, N // 64), 1)
        return num_threads_per_row

    def _get_tv_layout(self):
        N = self.N
        vecsize = self.vecsize
        num_threads = 128 if N <= 16384 else 256
        threads_per_row = self._calculate_threads_per_row()
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(min(N, 16384) // vecsize, threads_per_row)
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tv_layout = cute.make_layout(
            ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * threads_per_row),
            ),
        )
        return tiler_mn, tv_layout

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mValues.element_type == self.dtype
        assert mIndices.element_type == cutlass.Int32
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        self.kernel(mX, mValues, mIndices, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        # We use domain_offset_i64 to deal with tensors larger than 2^31 elements
        mX = utils.domain_offset_i64((bidx * tiler_mn[0], 0), mX)
        gX = cute.local_tile(mX, tiler_mn, (0, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        # declare the atoms which will be used later for memory copy
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=128
        )
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == tiler_mn[1])
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1]) if not is_even_N else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
        tXrX_f32 = cute.make_fragment(tXrX.shape, cutlass.Float32)
        tXrX_f32.store(tXrX.load().to(cutlass.Float32))

        # Encode the indices into the bottom bits of values.
        log_N = int(math.log2(self.N))
        idx_mask = (1 << log_N) - 1
        vecsize = cutlass.const_expr(tv_layout.shape[1][0])
        tXrX_u32 = cute.recast_tensor(tXrX_f32, cutlass.Uint32)
        # Encode indices into the last log_N bits of tXrX_u32
        for i in cutlass.range(cute.size(tXrX_u32), unroll_full=True):
            # tXcX only keeps track of the indices for every @vecsize elements
            col_idx = cutlass.Uint32(tXcX[i // vecsize][1] + i % vecsize)
            # If positive, invert the bits of the index, so that if there's a tie,
            # indices coming from a earlier column will win.
            encoded_idx = ~col_idx if tXrX_f32[i] >= 0 else col_idx
            # Mask to keep only the last log_N bits of the encoded index
            encoded_idx = encoded_idx & idx_mask
            # Clear the last log_N bits and set them to our encoded index
            tXrX_u32[i] = (tXrX_u32[i] & ~idx_mask) | encoded_idx

        # Fill OOB values with -inf for top-k
        if const_expr(not is_even_N):
            utils.fill_oob(tXrX_f32, tXpX, -tXrX_f32.element_type.inf)

        threads_per_row = tv_layout.shape[0][0]
        topk_vals = bitonic_topk(tXrX_f32, self.k, warp_width=threads_per_row)

        # Extract indices and clean values
        topk_vals_u32 = cute.recast_tensor(topk_vals, cutlass.Uint32)
        topk_indices = cute.make_fragment(self.k, cutlass.Int32)
        for i in cutlass.range(self.k):
            # Extract the encoded index from the last log_N bits
            encoded_idx = topk_vals_u32[i] & idx_mask
            # Check if original value was positive by looking at the cleaned value
            topk_vals_u32[i] = topk_vals_u32[i] & ~idx_mask  # Clear last log_N bits
            # If positive, we need to invert the bits back to get original index
            col_idx = ~encoded_idx if topk_vals[i] >= 0 else encoded_idx
            topk_indices[i] = cutlass.Int32(col_idx & idx_mask)

        # Convert cleaned values to output type
        topk_vals_out = cute.make_fragment_like(topk_vals, mValues.element_type)
        topk_vals_out.store(topk_vals.load().to(mValues.element_type))

        row = tXcX[0][0]
        # Only the 1st thread in this row writes the top-k values and indices
        if row < shape[0] and tXcX[0][1] == 0:
            # for i in cutlass.range(self.k):
            #     mValues[row, i] = topk_vals_out[i]
            #     mIndices[row, i] = topk_indices[i]
            # Vectorized write
            elems_per_store = const_expr(math.gcd(vecsize, self.k))
            mValues_store = cute.tiled_divide(mValues[row, None], (elems_per_store,))
            mIndices_store = cute.tiled_divide(mIndices[row, None], (elems_per_store,))
            topk_vals_out_store = cute.tiled_divide(topk_vals_out, (elems_per_store,))
            topk_indices_store = cute.tiled_divide(topk_indices, (elems_per_store,))
            for i in cutlass.range(cute.size(topk_vals_out_store.shape, [1]), unroll_full=True):
                cute.autovec_copy(topk_vals_out_store[None, i], mValues_store[None, i])
                cute.autovec_copy(topk_indices_store[None, i], mIndices_store[None, i])


@torch.library.custom_op("quack::_topk_fwd", mutates_args={"values", "indices"})
def _topk_fwd(x: torch.Tensor, k: int, values: torch.Tensor, indices: torch.Tensor) -> None:
    """Top-k forward pass.
    Args:
        x: Input tensor of shape (M, N)
        k: Number of top elements to return
    Returns:
        Tuple of (values tensor of shape (M, k), indices tensor of shape (M, k))
    """
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Tensor must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    assert k > 0 and k <= x.shape[1], "k must be positive and <= N"

    N = x.size(1)

    dtype = torch2cute_dtype_map[x.dtype]
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )

    x_tensor, values_tensor, indices_tensor = [
        convert_from_dlpack(tensor) for tensor in (x, values, indices)
    ]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, N, k)
    if compile_key not in _topk_fwd.compile_cache:
        topk_op = TopK(dtype, N, k)
        _topk_fwd.compile_cache[compile_key] = cute.compile(
            topk_op, x_tensor, values_tensor, indices_tensor, current_stream
        )
    _topk_fwd.compile_cache[compile_key](x_tensor, values_tensor, indices_tensor, current_stream)


_topk_fwd.compile_cache = {}


def topk(x: torch.Tensor, k: int):
    """Top-k operation.

    Args:
        x: Input tensor of shape (M, N)
        k: Number of top elements to return

    Returns:
        Tuple of (values tensor of shape (M, k), indices tensor of shape (M, k))
    """

    M = x.size(0)

    values = torch.empty((M, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=x.device)

    _topk_fwd(x, k, values, indices)

    return values, indices
