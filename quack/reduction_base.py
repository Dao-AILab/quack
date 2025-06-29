# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import torch
from typing import Type, Tuple, Optional

import cutlass
import cutlass.cute as cute

import quack.utils as utils


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


class ReductionBase:
    def __init__(
        self, dtype: Type[cutlass.Numeric], N: int, stage: int, reduction_dtype=cutlass.Float32
    ):
        self.dtype = dtype
        self.N = N
        self.stage = stage
        self.reduction_dtype = reduction_dtype

    def _calculate_threads_per_row(self):
        raise NotImplementedError()

    def _set_cluster_n(self):
        self.cluster_n = 1

    def _get_num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _get_tv_layout(self):
        copy_bits = 128
        vecsize = copy_bits // self.dtype.width
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        num_threads = self._get_num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0

        threads_per_row = self._calculate_threads_per_row()
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        cols_per_block = num_threads // threads_per_row

        # Example with `N = 512` and FP16, then each tile has `[4, 512]` elements.
        # - `vector_size = 8` (i.e., 16-bit x 8 = 128-bit)
        # - `num_threads = 128`
        # - `threads_per_row = 32`
        # - `cols_per_block = 4`
        # - `cluster_n = 1`
        # - `num_blocks_N = 2`
        # TV Layout:
        # - Shape  (4, 32), (8, 2)
        # - Stride (1, 32), (4, 1024)
        # Here's the memory layout:
        # - Row 0: [B0, T0 x8] [B0, T4 x8] ... [B0, T124 x8] [B1, T0 x8] [B1, T4 x8] ... [B1, T124 x8]
        # - Row 1: [B0, T1 x8] [B0, T5 x8] ... [B0, T125 x8] [B1, T1 x8] [B1, T5 x8] ... [B1, T125 x8]
        # - Row 2: [B0, T2 x8] [B0, T6 x8] ... [B0, T126 x8] [B1, T2 x8] [B1, T6 x8] ... [B1, T126 x8]
        # - Row 3: [B0, T3 x8] [B0, T7 x8] ... [B0, T127 x8] [B1, T3 x8] [B1, T7 x8] ... [B1, T127 x8]
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tv_layout = cute.make_layout(
            ((cols_per_block, threads_per_row), (vecsize, num_blocks_N)),
            stride=(
                (1, vecsize * cols_per_block),
                (cols_per_block, cols_per_block * vecsize * threads_per_row),
            ),
        )
        return tiler_mn, tv_layout

    def _smem_size_in_bytes(self, tiler_mn, num_warps):
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
            + self.stage * num_warps * self.cluster_n * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    def _get_reduction_buffer_layout(self, tv_layout: cute.Layout, cluster_n: int):
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        warps_per_row = utils.max_constexpr(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
            order=(1, 0, 2),
        )

    def _allocate_reduction_buffer_and_mbar(
        self, smem: cutlass.utils.SmemAllocator, tv_layout: cute.Layout
    ) -> Tuple[cute.Tensor, Optional[cute.Pointer]]:
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            self._get_reduction_buffer_layout(tv_layout, self.cluster_n),
            byte_alignment=4,
        )
        if cutlass.const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=self.stage)
        else:
            mbar_ptr = None
        return reduction_buffer, mbar_ptr

    @cute.jit
    def _initialize_cluster(self, tidx: cutlass.Int32, mbar_ptr: cute.Pointer, num_warps: int):
        if cutlass.const_expr(self.cluster_n > 1):
            if tidx < self.stage:
                cute.arch.mbarrier_init_arrive_cnt(mbar_ptr + tidx, 1)
            cute.arch.mbarrier_init_fence()
            if tidx < self.stage:
                cute.arch.mbarrier_init_tx_bytes(
                    mbar_ptr + tidx, num_warps * self.cluster_n * self.reduction_dtype.width // 8
                )
            # Cluster arrive after barrier init
            cute.arch.cluster_arrive_relaxed()
