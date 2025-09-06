import math
import torch
from typing import Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import quack.utils as utils
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
            reduction_dtype=cutlass.Float32 if not online_softmax else cutlass.Int64,
        )
        self.online_softmax = online_softmax

    def _calculate_threads_per_row(self):
        N = self.N
        return (
            8
            if N <= 64
            else (
                16
                if N <= 128
                else (32 if N <= 3072 else (64 if N <= 6144 else (128 if N <= 16384 else 256)))
            )
        )

    def _set_cluster_n(self):
        N = self.N
        if cutlass.const_expr(self.dtype.width == 16):
            cluster_n = (
                1
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                )
            )
        else:  # fp32
            cluster_n = (
                1
                if N <= 32 * 1024
                else (
                    2
                    if N <= 64 * 1024
                    else (4 if N <= 128 * 1024 else (8 if N <= 256 * 1024 else 16))
                )
            )
        self.cluster_n = cluster_n

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        self.kernel(mX, mO, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if cutlass.const_expr(self.cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        # We use domain_offset_i64 to deal with tensors larger than 2^31 elements
        mX, mO = [utils.domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mX, mO)]
        gX, gO = [cute.local_tile(mT, tiler_mn, (0, cluster_y)) for mT in (mX, mO)]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        # declare the atoms which will be used later for memory copy
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gO.element_type, num_bits_per_copy=128
        )

        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
        thr_copy_O = cute.make_tiled_copy(copy_atom_store_O, tv_layout, tiler_mn).get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1]) if not is_even_N else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        # Fill OOB values with -inf
        if cutlass.const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        threads_per_row = tv_layout.shape[0][0]
        if cutlass.const_expr(not self.online_softmax):
            max_x = row_reduce(
                x,
                cute.ReductionOp.MAX,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if cutlass.const_expr(self.cluster_n > 1) else None,
                init_val=-cutlass.Float32.inf,
                hook_fn=cute.arch.cluster_wait if cutlass.const_expr(self.cluster_n > 1) else None,
            )
            log2_e = math.log2(math.e)
            exp_x = cute.math.exp2((x - max_x) * log2_e, fastmath=True)
            denom = row_reduce(
                exp_x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if cutlass.const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
            )
        else:
            max_x, denom, exp_x = online_softmax_reduce(
                x,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                hook_fn=cute.arch.cluster_wait if cutlass.const_expr(self.cluster_n > 1) else None,
                return_exp_x=True,
            )
        y = exp_x * (1.0 / denom)
        tXrO.store(y.to(tXrO.element_type))
        tOpO = (
            utils.predicate_k(thr_copy_O.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)


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
    dtype = torch2cute_dtype_map[x.dtype]
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    x_tensor, out_tensor = [convert_from_dlpack(tensor) for tensor in (x, out)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, N)
    if compile_key not in _softmax_fwd.compile_cache:
        softmax_op = Softmax(dtype, N)
        _softmax_fwd.compile_cache[compile_key] = cute.compile(
            softmax_op, x_tensor, out_tensor, current_stream
        )
    _softmax_fwd.compile_cache[compile_key](x_tensor, out_tensor, current_stream)


_softmax_fwd.compile_cache = {}


def softmax_fwd(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    _softmax_fwd(x, out)
    return out


class SoftmaxBackward(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        # 1 stage for computing dot product
        super().__init__(dtype, N, stage=1, reduction_dtype=cutlass.Float32)

    def _calculate_threads_per_row(self):
        N = self.N
        return (
            8
            if N <= 64
            else (
                16
                if N <= 128
                else (32 if N <= 3072 else (64 if N <= 6144 else (128 if N <= 8192 else 256)))
            )
        )

    def _set_cluster_n(self):
        N = self.N
        if cutlass.const_expr(self.dtype.width == 16):
            cluster_n = (
                1
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                )
            )
        else:  # fp32
            cluster_n = (
                1
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                )
            )
        self.cluster_n = cluster_n

    def _get_num_threads(self):
        return 128 if self.N <= 8192 else 256

    def _smem_size_in_bytes(self, tiler_mn, num_warps):
        return (
            # Multiply by 2 since we need space for Y and dY
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * 2
            + self.stage * num_warps * self.cluster_n * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mdY.element_type == self.dtype
        assert mY.element_type == self.dtype
        assert mdX.element_type == self.dtype
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        self.kernel(mdY, mY, mdX, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(mdY.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if cutlass.const_expr(self.cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        shape = mdY.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        mdY, mY, mdX = [
            utils.domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mdY, mY, mdX)
        ]
        gdY, gY, gdX = [cute.local_tile(mT, tiler_mn, (0, cluster_y)) for mT in (mdY, mY, mdX)]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        smem = cutlass.utils.SmemAllocator()
        sdY = smem.allocate_tensor(
            mdY.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        sY = smem.allocate_tensor(
            mY.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        # declare the atoms which will be used later for memory copy
        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mdY.element_type, num_bits_per_copy=128
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gdX.element_type, num_bits_per_copy=128
        )

        thr_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn).get_slice(tidx)
        thr_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn).get_slice(tidx)

        tdYgdY = thr_copy_load.partition_S(gdY)
        tdYsdY = thr_copy_load.partition_D(sdY)
        tYgY = thr_copy_load.partition_S(gY)
        tYsY = thr_copy_load.partition_D(sY)
        tdXgdX = thr_copy_store.partition_D(gdX)
        tXcX = thr_copy_load.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tdYrdY, tYrY, tdXrdX = [cute.make_fragment_like(thr) for thr in (tdYgdY, tYgY, tdXgdX)]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tdYpdY = (
            utils.predicate_k(thr_copy_load.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load, tdYgdY, tdYsdY, pred=tdYpdY)
            cute.copy(copy_atom_load, tYgY, tYsY, pred=tdYpdY)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tdYsdY, tdYrdY)
        cute.autovec_copy(tYsY, tYrY)
        dy = tdYrdY.load().to(cute.Float32)
        y = tYrY.load().to(cute.Float32)

        # Compute dot product: dot = Σⱼ dy_j × y_j
        threads_per_row = tv_layout.shape[0][0]
        dot = row_reduce(
            dy * y,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr if cutlass.const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if cutlass.const_expr(self.cluster_n > 1) else None,
        )

        # Compute gradient: dx_i = y_i × (dy_i - dot)
        dx = y * (dy - dot)
        tdXrdX.store(dx.to(tdXrdX.element_type))
        tdXpdX = (
            utils.predicate_k(thr_copy_store.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store, tdXrdX, tdXgdX, pred=tdXpdX)


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
    dtype = torch2cute_dtype_map[dy.dtype]
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    dy_tensor, y_tensor, dx_tensor = [convert_from_dlpack(tensor) for tensor in (dy, y, dx)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (dtype, N)
    if compile_key not in _softmax_backward.compile_cache:
        softmax_backward_op = SoftmaxBackward(dtype, N)
        _softmax_backward.compile_cache[compile_key] = cute.compile(
            softmax_backward_op, dy_tensor, y_tensor, dx_tensor, current_stream
        )
    _softmax_backward.compile_cache[compile_key](dy_tensor, y_tensor, dx_tensor, current_stream)


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
