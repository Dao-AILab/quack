# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import math
from typing import Type
from functools import partial

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int64, Float32, const_expr

import quack.utils as utils
import quack.copy_utils as copy_utils
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.reduce import row_reduce, online_softmax_reduce
from quack.reduction_base import ReductionBase
from quack.cute_dsl_utils import torch2cute_dtype_map


class Softmax(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, online_softmax: bool = True):
        # 2 stages: 1 for max, 1 for sum
        super().__init__(
            dtype,
            N,
            stage=2 if not online_softmax else 1,
            reduction_dtype=Float32 if not online_softmax else Int64,
        )
        self.online_softmax = online_softmax

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        N = self.N
        if const_expr(self.dtype.width == 16):
            thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
        else:
            thresholds = [(32 * 1024, 1), (64 * 1024, 2), (128 * 1024, 4), (256 * 1024, 8)]
        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mX, mO]))
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(
            vecsize=128 // largest_dtype_width
        )
        num_threads = tiled_copy.size
        self.kernel(mX, mO, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tv_layout = tiled_copy.layout_tv_tiled

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        gX, gO, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, mO, idX)]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy_X = tiled_copy.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_X.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            None
            if is_even_N
            else copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
        )
        # Each copy will use the same predicate
        copy = partial(copy_utils.copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        if tXcX[0][0] < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        # Fill OOB values with -inf
        if const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        if const_expr(not self.online_softmax):
            max_x = row_reduce(
                x,
                cute.ReductionOp.MAX,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
                init_val=-Float32.inf,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            )
            log2_e = math.log2(math.e)
            exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=True)
            denom = row_reduce(
                exp_x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
            )
        else:
            max_x, denom, exp_x = online_softmax_reduce(
                x,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
                return_exp_x=True,
            )
        # y = exp_x * (1.0 / denom)
        y = exp_x * cute.arch.rcp_approx(denom)
        tXrO.store(y.to(tXrO.element_type))
        if tXcX[0][0] < shape[0]:
            copy(tXrO, tXgO)


@torch.library.custom_op("quack::_softmax_fwd", mutates_args={"out"})
def _softmax_fwd(x: torch.Tensor, out: torch.Tensor) -> None:
    """Softmax forward pass.
    Args:
        x: Input tensor of shape (M, N)
    Returns:
        Softmax output tensor of same shape as x
    """
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Tensor must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    N = x.size(1)
    dtype, out_dtype = [torch2cute_dtype_map[t.dtype] for t in [x, out]]
    compile_key = (dtype, out_dtype, N)
    if compile_key not in _softmax_fwd.compile_cache:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute, out_cute = [fake_tensor(dt, (batch_sym, N), div) for dt in [dtype, out_dtype]]
        softmax_op = Softmax(dtype, N)
        _softmax_fwd.compile_cache[compile_key] = cute.compile(
            softmax_op,
            x_cute,
            out_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _softmax_fwd.compile_cache[compile_key](x, out)


_softmax_fwd.compile_cache = {}


def softmax_fwd(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    _softmax_fwd(x, out)
    return out


class SoftmaxBackward(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        # 1 stage for computing dot product
        super().__init__(dtype, N, stage=1, reduction_dtype=Float32)

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (8192, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        N = self.N
        if const_expr(self.dtype.width == 16):
            thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
        else:
            thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    def _num_threads(self):
        return 128 if self.N <= 8192 else 256

    @cute.jit
    def __call__(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mdY.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mdY, mY, mdX]))
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(
            vecsize=128 // largest_dtype_width
        )
        num_threads = tiled_copy.size
        self.kernel(mdY, mY, mdX, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mdY.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mdY.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        gdY, gY, gdX, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mdY, mY, mdX, idX)
        ]

        smem = cutlass.utils.SmemAllocator()
        sdY = smem.allocate_tensor(
            mdY.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        sY = smem.allocate_tensor(
            mY.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)

        tdYgdY = thr_copy.partition_S(gdY)
        tdYsdY = thr_copy.partition_D(sdY)
        tYgY = thr_copy.partition_S(gY)
        tYsY = thr_copy.partition_D(sY)
        tdXgdX = thr_copy.partition_D(gdX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tdYrdY, tYrY, tdXrdX = [cute.make_fragment_like(thr) for thr in (tdYgdY, tYgY, tdXgdX)]

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        )
        # Each copy will use the same predicate
        copy = partial(copy_utils.copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        if tXcX[0][0] < shape[0]:
            copy(tdYgdY, tdYsdY, is_async=True)
            copy(tYgY, tYsY, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        # Don't need fill_oob since cp.async will automatically fills OOB elements with zeros

        cute.autovec_copy(tdYsdY, tdYrdY)
        cute.autovec_copy(tYsY, tYrY)
        dy = tdYrdY.load().to(cute.Float32)
        y = tYrY.load().to(cute.Float32)

        # Compute dot product: dot = Σⱼ dy_j × y_j
        dot = row_reduce(
            dy * y,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )

        # Compute gradient: dx_i = y_i × (dy_i - dot)
        dx = y * (dy - dot)
        tdXrdX.store(dx.to(tdXrdX.element_type))
        if tXcX[0][0] < shape[0]:
            copy(tdXrdX, tdXgdX)


@torch.library.custom_op("quack::_softmax_backward", mutates_args={"dx"})
def _softmax_backward(dy: torch.Tensor, y: torch.Tensor, dx: torch.Tensor) -> None:
    """Softmax backward pass.
    Args:
        dy: Upstream gradients tensor of shape (M, N)
        y: Softmax output tensor of shape (M, N)
    Returns:
        Input gradients tensor of same shape as dy and y
    """
    assert dy.dim() == 2, "dy must be 2D"
    assert y.dim() == 2, "y must be 2D"
    assert dy.shape == y.shape, "dy and y must have same shape"
    assert dy.is_cuda and y.is_cuda, "Tensors must be on CUDA device"
    assert dy.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    assert y.dtype == dy.dtype, "dy and y must have same dtype"

    N = dy.size(1)
    dtype, y_dtype, dx_dtype = [torch2cute_dtype_map[t.dtype] for t in [dy, y, dx]]
    compile_key = (dtype, y_dtype, dx_dtype, N)
    if compile_key not in _softmax_backward.compile_cache:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        dy_cute, y_cute, dx_cute = [
            fake_tensor(dt, (batch_sym, N), div) for dt in [dtype, y_dtype, dx_dtype]
        ]
        softmax_backward_op = SoftmaxBackward(dtype, N)
        _softmax_backward.compile_cache[compile_key] = cute.compile(
            softmax_backward_op,
            dy_cute,
            y_cute,
            dx_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _softmax_backward.compile_cache[compile_key](dy, y, dx)


_softmax_backward.compile_cache = {}


def softmax_bwd(dy: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dx = torch.empty_like(dy)
    _softmax_backward(dy, y, dx)
    return dx


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = softmax_fwd(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        dx = softmax_bwd(dy, y)
        return dx


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax forward pass with automatic differentiation support.

    Args:
        x: Input tensor of shape (M, N)

    Returns:
        Softmax output tensor of same shape as x
    """
    return SoftmaxFunction.apply(x)
