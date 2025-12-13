# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Tri Dao.

import math
from functools import partial
from typing import Type

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

import quack.utils as utils
import quack.copy_utils as copy_utils
from quack.compile_utils import make_fake_tensor as fake_tensor
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
        assert mIndices.element_type == Int32
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
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, 0)) for mT in (mX, idX)]

        num_copy_elems = tv_layout.shape[1][0]
        copy_atom = copy_utils.get_copy_atom(mX.element_type, num_copy_elems)
        thr_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn).get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == tiler_mn[1])
        tXpX = None if is_even_N else utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX, num_copy_elems=num_copy_elems)

        if tXcX[0][0] < shape[0]:
            copy(tXgX, tXrX)
        tXrX_f32 = cute.make_fragment(tXrX.shape, Float32)
        tXrX_f32.store(tXrX.load().to(Float32))

        # Encode the indices into the bottom bits of values.
        log_N = int(math.log2(self.N))
        idx_mask = (1 << log_N) - 1
        vecsize = const_expr(tv_layout.shape[1][0])
        tXrX_i32 = cute.recast_tensor(tXrX_f32, Int32)
        # Encode indices into the last log_N bits of tXrX_i32
        for i in cutlass.range(cute.size(tXrX_i32), unroll_full=True):
            # tXcX only keeps track of the indices for every @vecsize elements
            col_idx = Int32(tXcX[i // vecsize][1] + i % vecsize)
            # If positive, invert the bits of the index, so that if there's a tie,
            # indices coming from a earlier column will win.
            encoded_idx = ~col_idx if tXrX_f32[i] >= 0 else col_idx
            # Mask to keep only the last log_N bits of the encoded index
            encoded_idx = encoded_idx & idx_mask
            # Clear the last log_N bits and set them to our encoded index
            tXrX_i32[i] = (tXrX_i32[i] & ~idx_mask) | encoded_idx

        # Fill OOB values with -inf for top-k
        if const_expr(not is_even_N):
            utils.fill_oob(tXrX_f32, tXpX, -tXrX_f32.element_type.inf)

        threads_per_row = tv_layout.shape[0][0]
        topk_vals = bitonic_topk(tXrX_f32, self.k, warp_width=threads_per_row)

        # Extract indices and clean values
        topk_vals_i32 = cute.recast_tensor(topk_vals, Int32)
        topk_indices = cute.make_fragment(self.k, Int32)
        for i in cutlass.range(self.k):
            # Extract the encoded index from the last log_N bits
            encoded_idx = topk_vals_i32[i] & idx_mask
            # Check if original value was positive by looking at the cleaned value
            topk_vals_i32[i] = topk_vals_i32[i] & ~idx_mask  # Clear last log_N bits
            # If positive, we need to invert the bits back to get original index
            col_idx = ~encoded_idx if topk_vals[i] >= 0 else encoded_idx
            topk_indices[i] = Int32(col_idx & idx_mask)

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
    compile_key = (dtype, N, k)
    if compile_key not in _topk_fwd.compile_cache:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute = fake_tensor(dtype, (batch_sym, N), div)
        values_cute = fake_tensor(dtype, (batch_sym, k), div)
        indices_cute = fake_tensor(Int32, (batch_sym, k), div)
        topk_op = TopK(dtype, N, k)
        _topk_fwd.compile_cache[compile_key] = cute.compile(
            topk_op,
            x_cute,
            values_cute,
            indices_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _topk_fwd.compile_cache[compile_key](x, values, indices)


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
