# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import math
from typing import Optional, Type, Literal

import torch
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, Boolean, const_expr
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
            reduction_dtype=Float32 if not online_softmax else Int64,
        )
        self.online_softmax = online_softmax
        self.reload_from = None if N <= 16384 or online_softmax else "smem"

    def _calculate_threads_per_row(self):
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
            thresholds = [(16 * 1024, 1), (64 * 1024, 2), (128 * 1024, 4), (256 * 1024, 8)]
        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mTargetLogit: Optional[cute.Tensor],  # (M, K) or (M,). If None, we use mX
        mLoss: cute.Tensor,  # (M,)
        mLSE: Optional[cute.Tensor],  # (M,)
        mdX: Optional[cute.Tensor],  # (M, N) - if provided, compute gradient
        ignore_index: Int32,  # Index to ignore in loss computation
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        if const_expr(mTargetLogit is None):
            mTargetLogit = mX
        self._set_cluster_n()
        # e.g. if self.N isn't divisible by 8 for bf16, we might use 64 bits (4 elements) copy
        num_copy_bits = math.gcd(self.N, 128 // self.dtype.width) * self.dtype.width
        tiler_mn, tv_layout = self._get_tv_layout(num_copy_bits=num_copy_bits)
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        self.kernel(
            mX, mTarget, mTargetLogit, mLoss, mLSE, mdX, ignore_index, tv_layout, tiler_mn
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=([1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None),
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,  # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mTargetLogit: cute.Tensor,  # (M, K) or (M,)
        mLoss: cute.Tensor,  # (M,)
        mLSE: Optional[cute.Tensor],  # (M,)
        mdX: Optional[cute.Tensor],  # (M, N) - if provided, compute gradient
        ignore_index: Int32,  # Index to ignore in loss computation
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

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
        num_copy_elems_X = tv_layout.shape[1][0]
        num_copy_bits_X = mX.element_type.width * num_copy_elems_X
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), gX.element_type, num_bits_per_copy=num_copy_bits_X
        )
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)

        #### Partition to get thread view
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_fragment_like(tXgX)

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]
        target = Int32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )
        if row < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        # Fill OOB values with -inf
        if const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        target_logit = Float32.zero
        should_ignore = Boolean(target == ignore_index)
        if row < shape[0] and tXcX[0][1] == 0 and not should_ignore:
            # Only load target logit if not ignoring this index
            if const_expr(cute.rank(mTargetLogit.shape) == 2):
                # Use Int64 for indexing to deal with large tensors
                mTargetLogit_off = utils.domain_offset_i64((row, 0), mTargetLogit)
                target_logit = Float32(mTargetLogit_off[0, target])
            else:
                assert cute.rank(mTargetLogit.shape) == 1
                target_logit = Float32(mTargetLogit[row])

        threads_per_row = tv_layout.shape[0][0]
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
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(Float32)
            log2_e = math.log2(math.e)
            # This would use ffma instead of fadd then fmul
            exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=False)
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
                return_exp_x=const_expr(mdX is not None),
            )

        # Write loss and lse to gmem
        if (
            tXcX[0][1] == 0
            and row < shape[0]
            and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
        ):
            lse = max_x + cute.math.log(denom, fastmath=True)
            # Set loss to 0 if this index should be ignored, otherwise compute normally
            loss_val = (lse - target_logit) if not should_ignore else Float32.zero
            mLoss[row] = mLoss.element_type(loss_val)
            if const_expr(mLSE is not None):
                mLSE[row] = lse

        # Compute gradient if mdX is provided
        if const_expr(mdX is not None):
            # Compute probabilities: exp(x) / sum(exp(x))
            # If ignored, gradient should be zero
            denom_inv = (
                # 1.0 / denom
                cute.arch.rcp_approx(denom)
                if not (denom == 0.0 or denom != denom or should_ignore)
                else Float32.zero
            )
            probs = exp_x * denom_inv
            mdX_off = utils.domain_offset_i64((bidx * tiler_mn[0], 0), mdX)
            gdX = cute.local_tile(mdX_off, tiler_mn, (0, cluster_y))
            # Setup copy atom for storing gradient
            copy_atom_store = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), mdX.element_type, num_bits_per_copy=num_copy_bits_X
            )
            thr_copy_dX = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn).get_slice(tidx)
            tXgdX = thr_copy_dX.partition_D(gdX)
            tXrdX = cute.make_fragment_like(tXgdX)
            tXcFull = thr_copy_X.partition_S(cX)
            # Compute gradient: probs for all classes, (probs - 1) for target class
            # If ignored, gradient is already zero
            tXrdX_f32 = cute.make_fragment_like(tXrX, Float32)
            tXrdX_f32.store(probs)
            if not should_ignore:
                for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                    tXrdX_f32[i] = tXrdX_f32[i] if tXcFull[i][1] != target else tXrdX_f32[i] - 1.0
            tXrdX.store(tXrdX_f32.load().to(tXrdX.element_type))
            tXpdX = (
                utils.predicate_k(thr_copy_dX.partition_S(cX), limit=shape[1])
                if not is_even_N
                else None
            )
            if row < shape[0]:
                cute.copy(copy_atom_store, tXrdX, tXgdX, pred=tXpdX)


@torch.library.custom_op("quack::cross_entropy_fwd_out", mutates_args={"loss", "lse", "dx"})
def cross_entropy_fwd_out(
    x: Tensor,
    target: Tensor,
    target_logit: Optional[Tensor],
    loss: Tensor,
    lse: Optional[Tensor],
    dx: Optional[Tensor],
    ignore_index: int = -100,
) -> None:
    """Cross entropy forward pass.

    Args:
        x: Input logits tensor of shape (M, N)
        target: Target class indices tensor of shape (M,)
        target_logit: (M, K) or (M,).
            If provided, the target logit will be read from this tensor instead of x.
        loss: Output loss tensor of shape (M,)
        lse: Optional output log-sum-exp tensor of shape (M,)
        dx: Optional output gradient tensor of shape (M, N)
        ignore_index: Index to ignore in loss computation

    Returns:
        None (mutates loss, lse, and optionally dx in-place)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert target.dim() == 1, "Target must be 1D"
    assert x.shape[0] == target.shape[0], "Batch dimensions must match"
    assert x.is_cuda and target.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"
    assert target.dtype in [torch.int32, torch.int64], "Target must be int32 or int64"
    if target_logit is not None:
        assert target_logit.shape[0] == x.shape[0]
        assert target_logit.is_cuda, "Target logits must be on CUDA device"
        assert target_logit.dtype in [torch.float16, torch.bfloat16, torch.float32]
    if dx is not None:
        assert dx.shape == x.shape, "dx must have same shape as x"
        assert dx.is_cuda, "dx must be on CUDA device"
        assert dx.dtype == x.dtype, "dx must have same dtype as x"
    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    x_tensor = convert_from_dlpack(x)
    loss_tensor = from_dlpack(loss.detach(), assumed_align=4).mark_layout_dynamic()
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic()
        if lse is not None
        else None
    )
    target_tensor = from_dlpack(target.detach(), assumed_align=8).mark_layout_dynamic()
    target_logit_tensor = (
        from_dlpack(target_logit.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=target_logit.ndim - 1
        )
        if target_logit is not None
        else None
    )
    dx_tensor = convert_from_dlpack(dx) if dx is not None else None
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (
        dtype,
        N,
        target_logit.dtype if target_logit is not None else None,
        lse.dtype if lse is not None else None,
        dx is not None,
        loss.stride(),
        lse.stride() if lse is not None else None,
        target.stride(),
        target_logit.stride(-1) if target_logit is not None else None,
    )
    if compile_key not in cross_entropy_fwd_out.compile_cache:
        # If there's dx, it's faster to not use online softmax since we want the exp(x - max)
        cross_entropy_op = CrossEntropy(dtype, N, online_softmax=dx is None)
        cross_entropy_fwd_out.compile_cache[compile_key] = cute.compile(
            cross_entropy_op,
            x_tensor,
            target_tensor,
            target_logit_tensor,
            loss_tensor,
            lse_tensor,
            dx_tensor,
            Int32(ignore_index),
            stream,
        )
    cross_entropy_fwd_out.compile_cache[compile_key](
        x_tensor,
        target_tensor,
        target_logit_tensor,
        loss_tensor,
        lse_tensor,
        dx_tensor,
        Int32(ignore_index),
        stream,
    )


cross_entropy_fwd_out.compile_cache = {}


def cross_entropy_fwd(
    x: torch.Tensor,
    target: torch.Tensor,
    target_logit: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    return_lse: bool = False,
    return_dx: bool = False,
    inplace_backward: bool = False,
) -> torch.Tensor | tuple[torch.Tensor]:
    M = x.size(0)
    device = x.device
    loss = torch.empty(M, device=device, dtype=torch.float32)
    lse = torch.empty(M, device=device, dtype=torch.float32) if return_lse else None
    dx = (torch.empty_like(x) if not inplace_backward else x) if return_dx else None
    cross_entropy_fwd_out(x, target, target_logit, loss, lse, dx, ignore_index)
    if return_lse and return_dx:
        return loss, lse, dx
    elif return_lse:
        return loss, lse
    elif return_dx:
        return loss, dx
    else:
        return loss


class CrossEntropyBackward:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        self.dtype = dtype
        self.N = N
        self.vecsize = 128 // dtype.width

    def _calculate_threads_per_row(self):
        N = min(self.N, 16384)  # We split by blocks of 16k
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _get_tv_layout(self, num_copy_bits=128):
        vecsize = num_copy_bits // self.dtype.width
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        N = min(self.N, 16384)
        num_threads = 128 if N <= 16384 else 256
        threads_per_row = self._calculate_threads_per_row()
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row)
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
        ignore_index: Int32,  # Index to ignore in gradient computation
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mdX.element_type == self.dtype
        # e.g. if self.N isn't divisible by 8 for bf16, we might use 64 bits (4 elements) copy
        num_copy_bits = math.gcd(self.N, 128 // self.dtype.width) * self.dtype.width
        tiler_mn, tv_layout = self._get_tv_layout(num_copy_bits=num_copy_bits)
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
        self.kernel(
            mX, mTarget, mDLoss, mdX, mLSE, ignore_index, mX.shape, tv_layout, tiler_mn
        ).launch(
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
        ignore_index: Int32,  # Index to ignore in gradient computation
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

        num_copy_elems_X = tv_layout.shape[1][0]
        num_copy_bits_X = mX.element_type.width * num_copy_elems_X
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), gX.element_type, num_bits_per_copy=num_copy_bits_X
        )
        copy_atom_store_dX = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gdX.element_type, num_bits_per_copy=num_copy_bits_X
        )
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
        thr_copy_dX = cute.make_tiled_copy(copy_atom_store_dX, tv_layout, tiler_mn).get_slice(tidx)

        #### Partition to get thread view
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_S(sX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXcFull = thr_copy_X.partition_S(cX)
        tXgdX = thr_copy_dX.partition_D(gdX)
        # allocate fragments for gmem->rmem
        tXrX, tXrdX = [cute.make_fragment_like(thr) for thr in (tXgX, tXgdX)]

        is_even_N = const_expr(shape[1] % tiler_mn[1] == 0)
        row = tXcX[0][0]
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1]) if not is_even_N else None
        )
        if row < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        target = Int32.zero
        dloss = Float32.zero
        lse = Float32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])
            should_ignore = Boolean(target == ignore_index)
            # Set dloss to 0 if this index should be ignored
            dloss = Float32(mDLoss[row]) if not should_ignore else Float32.zero
            lse = Float32(mLSE[row])

        log2_e = math.log2(math.e)
        probs = cute.math.exp2(x * log2_e - (lse * log2_e), fastmath=True)
        prob_shifted = probs - 1.0
        mask = cute.make_fragment_like(tXrX, Boolean)
        for i in cutlass.range(cute.size(tXcFull), unroll_full=True):
            mask[i] = tXcFull[i][1] == target
        grad = cute.where(mask.load(), prob_shifted, probs)
        grad = grad * dloss

        tXrdX.store(grad.to(tXrdX.element_type))
        tXpdX = (
            utils.predicate_k(thr_copy_dX.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        if row < shape[0]:
            cute.copy(copy_atom_store_dX, tXrdX, tXgdX, pred=tXpdX)


def _cross_entropy_backward(
    x: torch.Tensor,
    target: torch.Tensor,
    dloss: torch.Tensor,
    lse: torch.Tensor,
    dx: torch.Tensor,
    ignore_index=-100,
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
    assert x.is_cuda and target.is_cuda and dloss.is_cuda and lse.is_cuda, (
        "Tensors must be on CUDA device"
    )
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
    dloss_tensor = from_dlpack(dloss.detach(), assumed_align=4).mark_layout_dynamic()
    lse_tensor = from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic()
    target_tensor = from_dlpack(target.detach(), assumed_align=8).mark_layout_dynamic()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (dtype, N, target.dtype, dloss.stride(), lse.stride(), target.stride())
    if compile_key not in _cross_entropy_backward.compile_cache:
        cross_entropy_backward_op = CrossEntropyBackward(dtype, N)
        _cross_entropy_backward.compile_cache[compile_key] = cute.compile(
            cross_entropy_backward_op,
            x_tensor,
            target_tensor,
            dloss_tensor,
            dx_tensor,
            lse_tensor,
            Int32(ignore_index),
            stream,
        )
    _cross_entropy_backward.compile_cache[compile_key](
        x_tensor, target_tensor, dloss_tensor, dx_tensor, lse_tensor, Int32(ignore_index), stream
    )


_cross_entropy_backward.compile_cache = {}


@torch.library.custom_op("quack::cross_entropy_bwd_out", mutates_args={"dx"})
def cross_entropy_bwd_out(
    x: torch.Tensor,
    target: torch.Tensor,
    dloss: torch.Tensor,
    lse: torch.Tensor,
    dx: torch.Tensor,
    ignore_index: int = -100,
) -> None:
    _cross_entropy_backward(x, target, dloss, lse, dx, ignore_index)


def cross_entropy_bwd(
    x: torch.Tensor,
    target: torch.Tensor,
    dloss: torch.Tensor,
    lse: torch.Tensor,
    ignore_index: int = -100,
    inplace_backward: bool = False,
) -> None:
    if inplace_backward and not torch.compiler.is_compiling():
        dx = x
        _cross_entropy_backward(
            x=x, target=target, dloss=dloss, lse=lse, dx=x, ignore_index=ignore_index
        )
    else:
        dx = torch.empty_like(x)
        cross_entropy_bwd_out(
            x=x, target=target, dloss=dloss, lse=lse, dx=dx, ignore_index=ignore_index
        )
    return dx


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, lse_partial=None, ignore_index=-100, inplace_backward=False):
        if lse_partial is None:
            loss, lse = cross_entropy_fwd(x, target, ignore_index=ignore_index, return_lse=True)
        else:
            # if we already compute partial lse, then to compute the final lse we treat
            # @lse_partial as @x and @x as @target_logit
            loss, lse = cross_entropy_fwd(
                lse_partial, target, target_logit=x, ignore_index=ignore_index, return_lse=True
            )
        ctx.save_for_backward(x, target, lse)
        ctx.ignore_index = ignore_index
        ctx.inplace_backward = inplace_backward
        return loss

    @staticmethod
    def backward(ctx, dloss):
        x, target, lse = ctx.saved_tensors
        dx = cross_entropy_bwd(
            x, target, dloss, lse, ctx.ignore_index, inplace_backward=ctx.inplace_backward
        )
        return dx, None, None, None, None


def cross_entropy(
    x: torch.Tensor,
    target: torch.Tensor,
    lse_partial: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
    inplace_backward: bool = False,
) -> torch.Tensor:
    """Cross entropy loss with automatic differentiation support.

    Args:
        x: Input logits tensor of shape (M, N)
        target: Target class indices tensor of shape (M,)
        lse_partial: Optional precomputed log-sum-exp partial results
        reduction: Specifies the reduction to apply to the output:
            'none': no reduction will be applied (default)
            'mean': the sum of the output will be divided by the number of elements
            'sum': the output will be summed
        inplace_backward: Whether to perform backward pass in-place
        ignore_index: Index to ignore in loss computation (loss will be 0 for these indices)

    Returns:
        Cross entropy loss tensor:
            - If reduction='none': tensor of shape (M,) with per-example losses
            - If reduction='mean': scalar tensor with mean loss
            - If reduction='sum': scalar tensor with sum of losses
    """
    loss = CrossEntropyFunction.apply(x, target, lse_partial, ignore_index, inplace_backward)
    if reduction == "mean":
        return loss.sum() / (target != ignore_index).sum().float()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', or 'sum'"
        )
