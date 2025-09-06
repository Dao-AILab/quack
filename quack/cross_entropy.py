# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import math
from typing import Optional, Type

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import quack.utils as utils
from quack.reduce import row_reduce, online_softmax_reduce
from quack.reduction_base import ReductionBase
from quack.cute_dsl_utils import torch2cute_dtype_map


class CrossEntropy(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, online_softmax: bool = True):
        # 2 stages: 1 for max, 1 for sum
        super().__init__(
            dtype,
            N,
            stage=2 if not online_softmax else 1,
            reduction_dtype=cutlass.Float32 if not online_softmax else cutlass.Int64,
        )
        self.online_softmax = online_softmax
        self.reload_from = None if N <= 16384 or online_softmax else "smem"

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
                if N <= 16 * 1024
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
        mTarget: cute.Tensor,
        mLoss: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        self.kernel(mX, mTarget, mLoss, mLSE, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=([1, self.cluster_n, 1] if cutlass.const_expr(self.cluster_n > 1) else None),
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,  # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mLoss: cute.Tensor,  # (M,)
        mLSE: Optional[cute.Tensor],  # (M,)
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        shape: cute.Shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        # We use domain_offset_i64 to deal with tensors larger than 2^31 elements
        mX_off = utils.domain_offset_i64((bidx * tiler_mn[0], 0), mX)
        gX = cute.local_tile(mX_off, tiler_mn, (0, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        # declare the atoms which will be used later for memory copy
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), gX.element_type, num_bits_per_copy=128
        )
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)

        #### Thread View
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_fragment_like(tXgX)

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]
        target = cute.Int32.zero
        if row < shape[0] and tXcX[0][1] == 0:
            target = cute.Int32(mTarget[row])

        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )
        if row < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        # Fill OOB values with -inf
        if cutlass.const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)

        target_logit = cute.Float32.zero
        if row < shape[0] and tXcX[0][1] == 0:
            # Use Int64 for indexing to deal with large tensors
            mX_off = utils.domain_offset_i64((row, 0), mX)
            target_logit = cute.Float32(mX_off[0, target])

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
            if cutlass.const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(cute.Float32)
            log2_e = math.log2(math.e)
            # exp_x = cute.math.exp2((x - max_x) * log2_e, fastmath=True)
            # a bit faster, probably because it's calling ex2.approx.ftz instead of ex2.approx?
            # exp_x = utils.exp2f((x - max_x) * log2_e)
            # This would use ffma instead of fadd then fmul
            exp_x = utils.exp2f(x * log2_e - (max_x * log2_e))
            denom = row_reduce(
                exp_x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if cutlass.const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
            )
        else:
            max_x, denom, _ = online_softmax_reduce(
                x,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                hook_fn=cute.arch.cluster_wait if cutlass.const_expr(self.cluster_n > 1) else None,
            )

        if (
            tXcX[0][1] == 0
            and row < shape[0]
            and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
        ):
            ln_2 = math.log(2.0)
            lse = max_x + utils.log2f(denom) * ln_2
            loss_val = lse - target_logit
            mLoss[row] = loss_val.to(mLoss.element_type)
            if cutlass.const_expr(mLSE is not None):
                mLSE[row] = lse


@torch.library.custom_op("quack::_cross_entropy", mutates_args={"loss", "lse"})
def _cross_entropy(
    x: torch.Tensor,
    target: torch.Tensor,
    loss: torch.Tensor,
    lse: torch.Tensor | None,
) -> None:
    """Cross entropy forward pass.

    Args:
        x: Input logits tensor of shape (M, N)
        target: Target class indices tensor of shape (M,)

    Returns:
        Cross entropy loss tensor of shape (M,)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert target.dim() == 1, "Target must be 1D"
    assert x.shape[0] == target.shape[0], "Batch dimensions must match"
    assert x.is_cuda and target.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"
    assert target.dtype in [torch.int32, torch.int64], "Target must be int32 or int64"
    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    x_tensor = convert_from_dlpack(x)
    loss_tensor = from_dlpack(loss.detach(), assumed_align=4).mark_compact_shape_dynamic(mode=0)
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_compact_shape_dynamic(mode=0)
        if lse is not None
        else None
    )
    target_tensor = from_dlpack(target.detach(), assumed_align=8).mark_compact_shape_dynamic(mode=0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (dtype, N, lse is not None)
    if compile_key not in _cross_entropy.compile_cache:
        cross_entropy_op = CrossEntropy(dtype, N)
        _cross_entropy.compile_cache[compile_key] = cute.compile(
            cross_entropy_op, x_tensor, target_tensor, loss_tensor, lse_tensor, stream
        )
    _cross_entropy.compile_cache[compile_key](
        x_tensor, target_tensor, loss_tensor, lse_tensor, stream
    )


_cross_entropy.compile_cache = {}


def cross_entropy_fwd(
    x: torch.Tensor,
    target: torch.Tensor,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor]:
    M = x.size(0)
    device = x.device
    loss = torch.empty(M, device=device, dtype=torch.float32)
    lse = torch.empty(M, device=device, dtype=torch.float32) if return_lse else None
    _cross_entropy(x, target, loss, lse)
    return loss if not return_lse else (loss, lse)


class CrossEntropyBackward:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        self.dtype = dtype
        self.N = N
        self.vecsize = 128 // dtype.width

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
        mTarget: cute.Tensor,
        mDLoss: cute.Tensor,
        mdX: cute.Tensor,
        mLSE: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mdX.element_type == self.dtype

        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])

        # (M,) -> (M, N) with stride 0 in the N dimension
        mDLoss, mTarget, mLSE = [
            cute.make_tensor(
                X.iterator, cute.append(X.layout, cute.make_layout((self.N,), stride=(0,)))
            )
            for X in (mDLoss, mTarget, mLSE)
        ]

        smem_size = cute.size_in_bytes(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0))
        )

        self.kernel(mX, mTarget, mDLoss, mdX, mLSE, mX.shape, tv_layout, tiler_mn).launch(
            grid=[
                cute.ceil_div(mX.shape[0], tiler_mn[0]),
                cute.ceil_div(mX.shape[1], tiler_mn[1]),
                1,
            ],
            block=[num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,  # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mDLoss: cute.Tensor,  # (M,)
        mdX: cute.Tensor,  # (M, N)
        mLSE: cute.Tensor,  # (M,)
        shape: cute.Shape,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        idX = cute.make_identity_tensor(shape)
        # We use domain_offset_i64 to deal with tensors larger than 2^31 elements
        mX, mdX = [utils.domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mX, mdX)]
        gX, gdX = [cute.local_tile(mT, tiler_mn, (0, bidy)) for mT in (mX, mdX)]
        cX = cute.local_tile(idX, tiler_mn, (bidx, bidy))

        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=128
        )
        copy_atom_load_X_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), gX.element_type, num_bits_per_copy=128
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gdX.element_type, num_bits_per_copy=128
        )

        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
        thr_copy_X_async = cute.make_tiled_copy(
            copy_atom_load_X_async, tv_layout, tiler_mn
        ).get_slice(tidx)
        thr_copy_O = cute.make_tiled_copy(copy_atom_store_O, tv_layout, tiler_mn).get_slice(tidx)

        #### Thread View
        tXgX = thr_copy_X_async.partition_S(gX)
        tXsX = thr_copy_X_async.partition_S(sX)

        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXcFull = thr_copy_X.partition_S(cX)  # improve

        tXgO = thr_copy_O.partition_D(gdX)

        # allocate fragments for gmem->rmem
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        is_even_N = cutlass.const_expr(shape[1] % tiler_mn[1] == 0)
        row = tXcX[0][0]

        tXpX = (
            utils.predicate_k(thr_copy_X_async.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )

        if row < shape[0]:
            cute.copy(copy_atom_load_X_async, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if cutlass.const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)

        label = cute.Int32.zero
        dloss = cute.Float32.zero
        lse = cute.Float32.zero
        if row < shape[0]:
            label = cute.Int32(mTarget[row])
            dloss = cute.Float32(mDLoss[row])
            lse = cute.Float32(mLSE[row])

        log2_e = math.log2(math.e)
        probs = utils.exp2f((x - lse) * log2_e)
        prob_shifted = probs - 1.0
        mask = cute.make_fragment_like(tXrX, cutlass.Boolean)
        for i in cutlass.range(cute.size(tXcFull), unroll_full=True):
            mask[i] = tXcFull[i][1] == label
        grad = cute.where(mask.load(), prob_shifted, probs)
        grad = grad * dloss

        tXrO.store(grad.to(tXrO.element_type))
        tOpO = (
            utils.predicate_k(thr_copy_O.partition_S(cX), limit=shape[1]) if not is_even_N else None
        )
        if row < shape[0]:
            cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)


def _cross_entropy_backward(
    x: torch.Tensor,
    target: torch.Tensor,
    dloss: torch.Tensor,
    lse: torch.Tensor,
    dx: torch.Tensor,
) -> None:
    """Cross entropy backward pass.
    Args:
        x: Input logits tensor of shape (M, N)
        target: Target class indices tensor of shape (M,)
        dloss: Upstream gradients tensor of shape (M,)
        lse: Log-sum-exp values tensor of shape (M,)
    Returns:
        Input gradients tensor of shape (M, N)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert target.dim() == 1, "Target must be 1D"
    assert dloss.dim() == 1, "dloss must be 1D"
    assert lse.dim() == 1, "lse must be 1D"
    assert x.shape[0] == target.shape[0], "Batch dimensions must match"
    assert x.shape[0] == dloss.shape[0], "Batch dimensions must match"
    assert x.shape[0] == lse.shape[0], "Batch dimensions must match"
    assert (
        x.is_cuda and target.is_cuda and dloss.is_cuda and lse.is_cuda
    ), "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"
    assert target.dtype in [torch.int32, torch.int64], "Target must be int32 or int64"

    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]

    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    x_tensor = convert_from_dlpack(x)
    dx_tensor = convert_from_dlpack(dx)
    dloss_tensor = from_dlpack(dloss.detach(), assumed_align=16).mark_compact_shape_dynamic(mode=0)
    lse_tensor = from_dlpack(lse.detach(), assumed_align=16).mark_compact_shape_dynamic(mode=0)
    target_tensor = from_dlpack(target.detach(), assumed_align=32).mark_compact_shape_dynamic(
        mode=0
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (dtype, N)
    if compile_key not in _cross_entropy_backward.compile_cache:
        cross_entropy_backward_op = CrossEntropyBackward(dtype, N)
        _cross_entropy_backward.compile_cache[compile_key] = cute.compile(
            cross_entropy_backward_op,
            x_tensor,
            target_tensor,
            dloss_tensor,
            dx_tensor,
            lse_tensor,
            stream,
        )
    _cross_entropy_backward.compile_cache[compile_key](
        x_tensor, target_tensor, dloss_tensor, dx_tensor, lse_tensor, stream
    )


_cross_entropy_backward.compile_cache = {}


@torch.library.custom_op("quack::_cross_entropy_bwd_no_inplace", mutates_args={"dx"})
def _cross_entropy_bwd_no_inplace(
    x: torch.Tensor,
    target: torch.Tensor,
    dloss: torch.Tensor,
    lse: torch.Tensor,
    dx: torch.Tensor,
) -> None:
    _cross_entropy_backward(x, target, dloss, lse, dx)


def cross_entropy_bwd(
    x: torch.Tensor,
    target: torch.Tensor,
    dloss: torch.Tensor,
    lse: torch.Tensor,
    inplace_backward: bool = False,
) -> None:
    if inplace_backward and not torch.compiler.is_compiling():
        dx = x
        _cross_entropy_backward(x=x, target=target, dloss=dloss, lse=lse, dx=x)
    else:
        dx = torch.empty_like(x)
        _cross_entropy_bwd_no_inplace(x=x, target=target, dloss=dloss, lse=lse, dx=dx)

    return dx


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, inplace_backward=False):
        loss, lse = cross_entropy_fwd(x, target, return_lse=True)
        ctx.save_for_backward(x, target, lse)
        ctx.inplace_backward = inplace_backward
        return loss

    @staticmethod
    def backward(ctx, dloss):
        x, target, lse = ctx.saved_tensors
        dx = cross_entropy_bwd(x, target, dloss, lse, inplace_backward=ctx.inplace_backward)
        return dx, None, None


def cross_entropy(
    x: torch.Tensor,
    target: torch.Tensor,
    inplace_backward: bool = True,
    reduction: str = "none",
) -> torch.Tensor:
    """Cross entropy loss with automatic differentiation support.

    Args:
        x: Input logits tensor of shape (M, N)
        target: Target class indices tensor of shape (M,)
        inplace_backward: Whether to perform backward pass in-place
        reduction: Specifies the reduction to apply to the output:
            'none': no reduction will be applied (default)
            'mean': the sum of the output will be divided by the number of elements
            'sum': the output will be summed

    Returns:
        Cross entropy loss tensor:
            - If reduction='none': tensor of shape (M,) with per-example losses
            - If reduction='mean': scalar tensor with mean loss
            - If reduction='sum': scalar tensor with sum of losses
    """
    loss = CrossEntropyFunction.apply(x, target, inplace_backward)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', or 'sum'"
        )
