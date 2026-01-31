# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Tri Dao.

import math
from functools import partial
from typing import Type, Optional

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

import quack.utils as utils
import quack.copy_utils as copy_utils
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.reduction_base import ReductionBase
from quack.reduce import row_reduce
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.sort.bitonic_sort import bitonic_topk


class TopK:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, k: int, softmax: bool = False):
        self.dtype = dtype
        self.N = N
        self.vecsize = 128 // dtype.width
        self.k = k
        self.softmax = softmax
        assert N == 2 ** int(math.log2(N)), "N must be a power of 2"
        assert k == 2 ** int(math.log2(k)), "N must be a power of 2"
        assert k <= 128
        assert N <= 4096

    def _threads_per_row(self):
        # we want num_elems_per_thread >= self.k
        # and each thread can handle at most 64 elements
        N = self.N
        num_threads_per_row = max(min(N // self.k, 32, N // 64), 1)
        return num_threads_per_row

    def _get_tiled_copy(self):
        N = self.N
        vecsize = self.vecsize
        num_threads = 128 if N <= 16384 else 256
        threads_per_row = self._threads_per_row()
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(min(N, 16384) // vecsize, threads_per_row)
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, num_threads, num_copy_elems=vecsize
        )
        return tiled_copy, tiler_mn, threads_per_row

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
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy()
        num_threads = tiled_copy.size
        self.kernel(mX, mValues, mIndices, tiler_mn, tiled_copy, threads_per_row).launch(
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
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, 0)) for mT in (mX, idX)]

        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == tiler_mn[1])
        tXpX = (
            None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        )
        copy = partial(copy_utils.copy, pred=tXpX)

        if tXcX[0][0] < shape[0]:
            copy(tXgX, tXrX)
        tXrX_f32 = cute.make_rmem_tensor(tXrX.shape, Float32)
        tXrX_f32.store(tXrX.load().to(Float32))

        # Encode the indices into the bottom bits of values.
        log_N = int(math.log2(self.N))
        idx_mask = (1 << log_N) - 1
        vecsize = const_expr(cute.size(tv_layout.shape[1]))
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

        topk_vals = bitonic_topk(tXrX_f32, self.k, warp_width=threads_per_row)

        # Thread 0 in each row contains all the top-k values, so we split those into multiple threads
        vecsize_out = const_expr(min(self.k, vecsize, 128 // mIndices.element_type.width))
        assert self.k % vecsize_out == 0
        nvec_per_thread = const_expr(cute.ceil_div(self.k, vecsize_out * threads_per_row))
        # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
        mask = cute.arch.WARP_SIZE - threads_per_row
        mask_and_clamp = mask << 8 | (cute.arch.WARP_SIZE - 1)
        topk_vals_split = cute.make_rmem_tensor((vecsize_out, nvec_per_thread), Float32)
        for i in cutlass.range(cute.ceil_div(self.k, vecsize_out), unroll_full=True):
            should_receive = tidx % threads_per_row == i % threads_per_row
            for v in cutlass.range(vecsize_out, unroll_full=True):
                if const_expr(threads_per_row > 1):
                    if i * vecsize_out + v < self.k:
                        val = cute.arch.shuffle_sync(
                            topk_vals[i * vecsize_out + v], offset=0, mask_and_clamp=mask_and_clamp
                        )
                        if should_receive:
                            topk_vals_split[v, i // threads_per_row] = val
                else:
                    topk_vals_split[v, i // threads_per_row] = topk_vals[i * vecsize_out + v]

        # Extract indices and clean values
        topk_vals_i32 = cute.recast_tensor(topk_vals_split, Int32)
        topk_indices = cute.make_rmem_tensor(topk_vals_i32.shape, Int32)
        for i in cutlass.range(cute.size(topk_vals_i32), unroll_full=True):
            # Extract the encoded index from the last log_N bits
            encoded_idx = topk_vals_i32[i] & idx_mask
            # Check if original value was positive by looking at the cleaned value
            topk_vals_i32[i] = topk_vals_i32[i] & ~idx_mask  # Clear last log_N bits
            # If positive, we need to invert the bits back to get original index
            col_idx = ~encoded_idx if topk_vals[i] >= 0 else encoded_idx
            topk_indices[i] = Int32(col_idx & idx_mask)

        # Compute softmax if requested
        if const_expr(self.softmax):
            # Need masking as some elements may be OOB
            for i in cutlass.range(cute.size(topk_vals_split, mode=[1]), unroll_full=True):
                col = i * threads_per_row + tidx % threads_per_row
                if col >= self.k // vecsize_out:
                    for v in cutlass.range(vecsize_out, unroll_full=True):
                        topk_vals_split[v, i] = -Float32.inf
            # Get max from thread 0 (topk_vals[0] is the max since sorted descending)
            max_val = cute.arch.shuffle_sync(topk_vals[0], offset=0, mask_and_clamp=mask_and_clamp)
            log2_e = math.log2(math.e)
            exp_x = cute.math.exp2(
                topk_vals_split.load() * log2_e - (max_val * log2_e), fastmath=True
            )
            denom = cute.arch.warp_reduction_sum(
                exp_x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
                threads_in_group=threads_per_row,
            )
            topk_vals_split.store(exp_x * cute.arch.rcp_approx(denom))

        # Convert cleaned values to output type
        topk_vals_out = cute.make_fragment_like(topk_vals_split, mValues.element_type)
        topk_vals_out.store(topk_vals_split.load().to(mValues.element_type))

        row = tXcX[0][0]
        # # Only the 1st thread in this row writes the top-k values and indices
        # if row < shape[0] and tXcX[0][1] == 0:
        #     # for i in cutlass.range(self.k):
        #     #     mValues[row, i] = topk_vals_out[i]
        #     #     mIndices[row, i] = topk_indices[i]
        #     # Vectorized write
        #     elems_per_store = const_expr(math.gcd(vecsize, self.k))
        #     mValues_store = cute.tiled_divide(mValues[row, None], (elems_per_store,))
        #     mIndices_store = cute.tiled_divide(mIndices[row, None], (elems_per_store,))
        #     topk_vals_out_store = cute.tiled_divide(topk_vals_out, (elems_per_store,))
        #     topk_indices_store = cute.tiled_divide(topk_indices, (elems_per_store,))
        #     for i in cutlass.range(cute.size(topk_vals_out_store.shape, [1]), unroll_full=True):
        #         cute.autovec_copy(topk_vals_out_store[None, i], mValues_store[None, i])
        #         cute.autovec_copy(topk_indices_store[None, i], mIndices_store[None, i])
        if tiler_mn[0] == 0 or row < shape[0]:
            # Vectorized write
            mValues_store = cute.tiled_divide(mValues[row, None], (vecsize_out,))
            mIndices_store = cute.tiled_divide(mIndices[row, None], (vecsize_out,))
            for i in cutlass.range(cute.size(topk_vals_out.shape, [1]), unroll_full=True):
                col = i * threads_per_row + tidx % threads_per_row
                if col < self.k // vecsize_out:
                    cute.autovec_copy(topk_vals_out[None, i], mValues_store[None, col])
                    cute.autovec_copy(topk_indices[None, i], mIndices_store[None, col])


@torch.library.custom_op("quack::_topk_fwd", mutates_args={"values", "indices"})
def _topk_fwd(
    x: torch.Tensor, k: int, softmax: bool, values: torch.Tensor, indices: torch.Tensor
) -> None:
    """Top-k forward pass.
    Args:
        x: Input tensor of shape (M, N)
        k: Number of top elements to return
        softmax: Whether to apply softmax to the top-k values
    Returns:
        Tuple of (values tensor of shape (M, k), indices tensor of shape (M, k))
    """
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Tensor must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    assert k > 0 and k <= x.shape[1], "k must be positive and <= N"

    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]
    compile_key = (dtype, N, k, softmax)
    if compile_key not in _topk_fwd.compile_cache:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute = fake_tensor(dtype, (batch_sym, N), div)
        values_cute = fake_tensor(dtype, (batch_sym, k), div)
        indices_cute = fake_tensor(Int32, (batch_sym, k), div)
        topk_op = TopK(dtype, N, k, softmax=softmax)
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


def topk_fwd(x: torch.Tensor, k: int, softmax: bool = False):
    """Top-k operation.

    Args:
        x: Input tensor of shape (M, N)
        k: Number of top elements to return
        softmax: Whether to apply softmax to the top-k values

    Returns:
        Tuple of (values tensor of shape (M, k), indices tensor of shape (M, k))
    """
    M = x.size(0)
    values = torch.empty((M, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=x.device)
    _topk_fwd(x, k, softmax, values, indices)
    return values, indices


class TopKBackward(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, k: int, softmax: bool = False):
        super().__init__(dtype, N, stage=1, reduction_dtype=Float32)
        self.dtype = dtype
        self.N = N
        self.k = k
        self.softmax = softmax
        assert k <= N
        assert k <= 32768

    def _num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _get_tiled_copy(self, N: int, vecsize: Optional[int] = None):
        if vecsize is None:
            vecsize = min(N, 128 // self.dtype.width)
        assert N % vecsize == 0, f"Input N {N} is not divisible by vector size {vecsize}"
        num_threads = self._num_threads()
        threads_per_row = min(N // vecsize, num_threads)
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row)
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, num_threads, num_copy_elems=vecsize
        )
        return tiled_copy, tiler_mn, threads_per_row

    @cute.jit
    def __call__(
        self,
        mdValues: cute.Tensor,  # (M, k)
        mValues: Optional[cute.Tensor],  # (M, k)
        mIndices: cute.Tensor,  # (M, k)
        mdX: cute.Tensor,  # (M, N)
        stream: cuda.CUstream,
    ):
        assert mdValues.element_type == self.dtype
        if const_expr(mValues is not None):
            assert mValues.element_type == self.dtype
        assert mIndices.element_type == Int32
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(
                *(t.element_type.width for t in [mdValues, mValues, mIndices, mdX] if t is not None)
            )
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(self.N, vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(
            mdValues,
            mValues,
            mIndices,
            mdX,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mdX.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdValues: cute.Tensor,  # (M, k)
        mValues: Optional[cute.Tensor],  # (M, k)
        mIndices: cute.Tensor,  # (M, k)
        mdX: cute.Tensor,  # (M, N)
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        tv_layout = tiled_copy.layout_tv_tiled
        shape = mdX.shape
        idX = cute.make_identity_tensor(shape)
        idTopK = cute.make_identity_tensor(mdValues.shape)
        # slice for CTAs
        gdX, cX = [cute.local_tile(mT, tiler_mn, (bidx, 0)) for mT in (mdX, idX)]
        gdVals, gVals, gIdx, cTopK = [
            cute.local_tile(mT, tiler_mn, (bidx, 0)) if mT is not None else None
            for mT in (mdValues, mValues, mIndices, idTopK)
        ]

        # Allocate smem for output gradients
        smem = cutlass.utils.SmemAllocator()
        sdX = smem.allocate_tensor(
            mdX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)

        tXgdV = thr_copy.partition_S(gdVals)
        tXgV = thr_copy.partition_S(gVals) if const_expr(gVals is not None) else None
        tXgI = thr_copy.partition_S(gIdx)
        tXrdV = cute.make_fragment_like(tXgdV)
        tXrV = cute.make_fragment_like(tXgV) if const_expr(tXgV is not None) else None
        tXrI = cute.make_fragment_like(tXgI)
        tXrdV.fill(tXrdV.element_type.zero)
        if const_expr(mValues is not None):
            tXrV.fill(tXrV.element_type.zero)
        tXrI.fill(0)

        tXsdX = thr_copy.partition_D(sdX)
        tXgdX = thr_copy.partition_D(gdX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXrdX = cute.make_fragment_like(tXgdX)

        is_even_N = const_expr(shape[1] == tiler_mn[1])
        tXpV = copy_utils.predicate_k(thr_copy.partition_S(cTopK), limit=mdValues.shape[1])
        tXpX = (
            None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        )
        copy_k = partial(copy_utils.copy, pred=tXpV)
        copy_dx = partial(copy_utils.copy, pred=tXpX)

        row = tXcX[0][0]
        tile_row_start = Int32(cute.arch.block_idx()[0] * tiler_mn[0])

        # Zero out smem
        utils.fill_oob(tXsdX, None, fill_value=mdX.element_type.zero)

        if row < shape[0]:
            copy_k(tXgdV, tXrdV)
            if const_expr(mValues is not None):
                copy_k(tXgV, tXrV)
            copy_k(tXgI, tXrI)

        cute.arch.barrier()

        dvals_f32 = tXrdV.load().to(Float32)
        if const_expr(self.softmax):
            vals_f32 = tXrV.load().to(Float32)
            dot = row_reduce(
                dvals_f32 * vals_f32,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
            )
            grads = vals_f32 * (dvals_f32 - dot)
        else:
            grads = dvals_f32
        grad_cvt = cute.make_rmem_tensor(tXrdV.shape, mdX.element_type)
        grad_cvt.store(grads.to(mdX.element_type))

        # Scatter values to smem
        if row < shape[0]:
            for rest_v in cutlass.range(tXrdV.shape[0][1], unroll_full=True):
                for n in cutlass.range(tXrdV.shape[2], unroll_full=True):
                    if tXpV[rest_v, 0, n]:
                        for v in cutlass.range(tXrdV.shape[0][0], unroll_full=True):
                            sdX[row - tile_row_start, tXrI[(v, rest_v), 0, n]] = grad_cvt[
                                (v, rest_v), 0, n
                            ]
        cute.arch.barrier()

        # Read from smem to rmem, then write to gmem
        cute.autovec_copy(tXsdX, tXrdX)
        if row < shape[0]:
            copy_dx(tXrdX, tXgdX)


@torch.library.custom_op("quack::_topk_bwd", mutates_args={"dx"})
def _topk_bwd(
    dvalues: torch.Tensor,
    values: Optional[torch.Tensor],
    indices: torch.Tensor,
    k: int,
    softmax: bool,
    dx: torch.Tensor,
) -> None:
    """Top-k backward pass.
    Args:
        dvalues: Upstream gradients tensor of shape (M, k)
        values: Forward top-k values tensor of shape (M, k)
        indices: Indices tensor of shape (M, k) from forward pass
        k: Number of top elements
        softmax: Whether softmax was applied in forward
        dx: Output gradient tensor of shape (M, N)
    """
    assert dvalues.dim() == 2, "dvalues must be 2D"
    if values is not None:
        assert values.dim() == 2, "values must be 2D"
    assert indices.dim() == 2, "indices must be 2D"
    assert dvalues.is_cuda and indices.is_cuda, "Tensors must be on CUDA device"
    assert dvalues.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"

    N = dx.size(1)
    dtype = torch2cute_dtype_map[dvalues.dtype]
    val_dtype = torch2cute_dtype_map[values.dtype] if values is not None else dtype
    dx_dtype = torch2cute_dtype_map[dx.dtype]
    compile_key = (dtype, val_dtype, dx_dtype, N, k, softmax)
    if compile_key not in _topk_bwd.compile_cache:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        dvalues_cute = fake_tensor(dtype, (batch_sym, k), div)
        values_cute = fake_tensor(val_dtype, (batch_sym, k), div) if values is not None else None
        indices_cute = fake_tensor(Int32, (batch_sym, k), div)
        dx_cute = fake_tensor(dx_dtype, (batch_sym, N), div)
        topk_bwd_op = TopKBackward(dtype, N, k, softmax=softmax)
        _topk_bwd.compile_cache[compile_key] = cute.compile(
            topk_bwd_op,
            dvalues_cute,
            values_cute,
            indices_cute,
            dx_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _topk_bwd.compile_cache[compile_key](dvalues, values, indices, dx)


_topk_bwd.compile_cache = {}


def topk_bwd(
    dvalues: torch.Tensor,
    values: Optional[torch.Tensor],
    indices: torch.Tensor,
    N: int,
    softmax: bool = False,
) -> torch.Tensor:
    """Top-k backward pass.

    Args:
        dvalues: Upstream gradients tensor of shape (M, k)
        values: Forward top-k values tensor of shape (M, k), required if softmax=True
        indices: Indices tensor of shape (M, k) from forward pass
        N: Size of the original input dimension
        softmax: Whether softmax was applied in forward

    Returns:
        Input gradients tensor of shape (M, N)
    """
    M, k = dvalues.shape
    dx = torch.zeros((M, N), dtype=dvalues.dtype, device=dvalues.device)
    _topk_bwd(dvalues, values, indices, k, softmax, dx)
    return dx


class TopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, k: int, softmax: bool = False):
        values, indices = topk_fwd(x, k, softmax=softmax)
        ctx.save_for_backward(values if softmax else None, indices)
        ctx.k = k
        ctx.N = x.shape[1]
        ctx.softmax = softmax
        ctx.mark_non_differentiable(indices)
        ctx.set_materialize_grads(False)
        return values, indices

    @staticmethod
    def backward(ctx, dvalues: torch.Tensor, dindices_: Optional[torch.Tensor] = None):
        values, indices = ctx.saved_tensors
        dx = topk_bwd(dvalues, values, indices, N=ctx.N, softmax=ctx.softmax)
        return dx, None, None


def topk(x: torch.Tensor, k: int, softmax: bool = False):
    """Top-k operation.

    Args:
        x: Input tensor of shape (M, N)
        k: Number of top elements to return
        softmax: Whether to apply softmax to the top-k values

    Returns:
        Tuple of (values tensor of shape (M, k), indices tensor of shape (M, k))
    """
    return TopKFunction.apply(x, k, softmax)
