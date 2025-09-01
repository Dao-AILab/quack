# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from typing import Optional, Type, Tuple, Union, Callable
from functools import partial

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass import Int32, const_expr

from quack.cute_dsl_utils import ParamsBase
from quack.tile_scheduler import (
    TileSchedulerArguments,
    TileScheduler,
    RasterOrderOption,
)

"""
A high-performance persistent batched dense GEMM example for the NVIDIA Blackwell SM100 architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp: Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp:
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Type convert C matrix to output type.
    - Optionally store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations,
      or directly store C matrix from registers (RMEM) to global memory (GMEM) without TMA operations.
    - Optionally accept an elementwise lambda function epilogue_op to apply to the output tensor:
      e.g., relu can set epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))

SM100 tcgen05.mma instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Input arguments to this example is same as dense_gemm.py.

.. code-block:: bash

    python examples/blackwell/dense_gemm_persistent.py                          \
      --ab_dtype Float16 --d_dtype Float16 --acc_dtype Float32                  \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
      --mnkl 8192,8192,8192,1                                                   \
      --use_2cta_instrs

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/dense_gemm_persistent.py                     \
      --ab_dtype Float16 --d_dtype Float16 --acc_dtype Float32                 \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,8192,1                                                  \
      --use_2cta_instrs                                        \
      --warmup_iterations 1 --iterations 10 --skip_ref_check


Constraints are same as dense_gemm.py:
* Supported input data types: fp16, bf16, tf32, int8, uint8, fp8 (e4m3fn, e5m2),
  see detailed valid dtype combinations in below PersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type
* Mma tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
* Mma tiler N must be 32-256, step 32
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if use_2cta_instrs=True
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 4, 8, and 16 for TFloat32,
  Float16/BFloat16, and Int8/Uint8/Float8, respectively.
* OOB tiles are not allowed when TMA store is disabled
"""


class PersistentDenseGemmKernel:
    """This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported A/B data types:
        - TFloat32
        - Float16/BFloat16
        - Int8/Uint8
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulator data types:
        - Float32 (for all floating point A/B data types)
        - Float16 (only for fp16 and fp8 A/B data types)
        - Int32 (only for uint8/int8 A/B data types)

    :note: Supported C data types:
        - Float32 (for float32 and int32 accumulator data types)
        - Int32 (for float32 and int32 accumulator data types)
        - Float16/BFloat16 (for fp16 and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16

    Example:
        >>> gemm = PersistentDenseGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     use_2cta_instrs=True,
        ...     mma_tiler_mn=(128, 128),
        ...     cluster_shape_mn=(2, 2)
        ... )
        >>> gemm(mA, mB, mD, max_active_clusters, stream)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        sf_vec_size: Optional[int] = None,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        """

        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.sf_vec_size = sf_vec_size
        self.blockscaled = sf_vec_size is not None

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.tma_epi_warp_id = 6
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, self.tma_epi_warp_id, *self.epilog_warp_id)
        )
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.epilog_load_bar_id = 3
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes("sm_100")

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Compute mma instruction shapes
        mma_inst_bits_k = 256
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_bits_k // self.a_dtype.width,
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk_sfb = (
            self.mma_inst_shape_mnk[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mnk[1], 128),
            self.mma_inst_shape_mnk[2],
        )

        # Configure tiled mma
        if const_expr(not self.blockscaled):
            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler[:2],
            )
            tiled_mma_sfb = None
        else:
            tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape_mnk[:2],
            )
            tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                self.mma_inst_shape_mnk_sfb[:2],
            )

        # Compute mma/cluster/tile shapes
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mnk[0],
            self.mma_inst_shape_mnk[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        if const_expr(self.blockscaled):
            self.mma_tiler_sfb = (
                self.mma_inst_shape_mnk_sfb[0],
                self.mma_inst_shape_mnk_sfb[1],
                self.mma_inst_shape_mnk_sfb[2] * mma_inst_tile_k,
            )
        else:
            self.mma_tiler_sfb = None
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        if const_expr(self.blockscaled):
            self.cluster_layout_sfb_vmnk = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)),
                (tiled_mma_sfb.thr_id.shape,),
            )
        else:
            self.cluster_layout_sfb_vmnk = None

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        if const_expr(self.blockscaled):
            self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
            self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.d_layout,
            self.d_dtype,
        )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_d_stage,
            self.num_c_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.d_dtype,
            self.c_dtype,
            self.d_layout,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.d_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.d_dtype, self.d_layout, self.epi_tile, self.num_d_stage
        )
        if const_expr(self.c_dtype is not None):
            self.epi_c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
            )
        else:
            self.epi_c_smem_layout_staged = None
        if const_expr(self.blockscaled):
            self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                self.num_ab_stage,
            )
            self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                self.num_ab_stage,
            )
        else:
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged = None, None

        # Compute the number of tensor memory allocation columns
        if const_expr(not self.blockscaled):
            self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
                tiled_mma, self.mma_tiler, self.num_acc_stage
            )
        else:
            SM100_TMEM_CAPACITY_COLUMNS = 512
            self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mC: Optional[cute.Tensor],
        tile_count_semaphore: Optional[cute.Pointer],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        mSFA: Optional[cute.Tensor] = None,
        mSFB: Optional[cute.Tensor] = None,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param mA: Input tensor A
        :type mA: cute.Tensor
        :param mB: Input tensor B
        :type mB: cute.Tensor
        :param mD: Output tensor D
        :type mD: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        :raises AssertionError: If OOB (Out-Of-Bounds) tiles are present when TMA store is disabled.
        """
        if const_expr(self.blockscaled):
            assert mSFA is not None and mSFB is not None
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = mA.element_type
        self.b_dtype: Type[cutlass.Numeric] = mB.element_type
        self.d_dtype: Type[cutlass.Numeric] = mD.element_type
        self.c_dtype = mC.element_type if mC is not None else None
        self.sf_dtype: Optional[Type[cutlass.Numeric]] = (
            mSFA.element_type if mSFA is not None else None
        )
        self.a_major_mode = cutlass.utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        self.b_major_mode = cutlass.utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        self.d_layout = cutlass.utils.LayoutEnum.from_tensor(mD)
        self.c_layout = cutlass.utils.LayoutEnum.from_tensor(mC) if mC is not None else None

        # Check if input data types are compatible with MMA instruction
        if const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        if const_expr(self.blockscaled):
            # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
            # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
            sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(mA.shape, self.sf_vec_size)
            mSFA = cute.make_tensor(mSFA.iterator, sfa_layout)
            # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(mB.shape, self.sf_vec_size)
            mSFB = cute.make_tensor(mSFB.iterator, sfb_layout)

        if const_expr(not self.blockscaled):
            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler[:2],
            )
            tiled_mma_sfb = None
        else:
            tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape_mnk[:2],
            )
            tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                self.mma_inst_shape_mnk_sfb[:2],
            )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            mA,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if mA.element_type is cutlass.Float32 else None),
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            mB,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if mB.element_type is cutlass.Float32 else None),
        )

        if const_expr(self.blockscaled):
            # Setup TMA load for SFA
            sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
            tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                mSFA,
                sfa_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            # Setup TMA load for SFB
            sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                mSFB,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )
        else:
            tma_atom_sfa, tma_tensor_sfa = None, None
            tma_atom_sfb, tma_tensor_sfb = None, None

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size
        if const_expr(self.blockscaled):
            sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            self.num_tma_load_bytes += (sfa_copy_size + sfb_copy_size) * atom_thr_size

        # Setup TMA store for D
        epi_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, 0))
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mD,
            epi_smem_layout,
            self.epi_tile,
        )
        if const_expr(mC is not None):
            epi_c_smem_layout = cute.slice_(self.epi_c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mC,
                epi_c_smem_layout,
                self.epi_tile,
            )
        else:
            tma_atom_c, tma_tensor_c = None, None

        problem_shape_ntile_mnl = cute.ceil_div(mD.shape[:2], self.cta_tile_shape_mnk[:2]) + (
            mD.shape[2],
        )
        TileSchedulerCls = TileScheduler
        tile_sched_args = TileSchedulerArguments(
            problem_shape_ntile_mnl=problem_shape_ntile_mnl,
            raster_order=RasterOrderOption.Heuristic,
            group_size=8,
            cluster_shape_mnk=(*self.cluster_shape_mn, 1),
            tile_count_semaphore=tile_count_semaphore,
            is_persistent=True,
        )
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
        grid = TileSchedulerCls.get_grid_shape(tile_sched_params, max_active_clusters)

        self.buffer_align_bytes = 1024

        epi_c_smem_size = cute.cosize(self.epi_c_smem_layout_staged) if mC is not None else 0
        sf_dtype = self.sf_dtype if const_expr(self.blockscaled) else cutlass.Float8E8M0FNU
        sfa_smem_size = (
            cute.cosize(self.sfa_smem_layout_staged) if const_expr(self.blockscaled) else 0
        )
        sfb_smem_size = (
            cute.cosize(self.sfb_smem_layout_staged) if const_expr(self.blockscaled) else 0
        )

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: Int32
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tile_count: cute.struct.MemRange[cutlass.Int32, 1]
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, cute.cosize(self.d_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype if self.c_dtype is not None else Int32, epi_c_smem_size
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[sf_dtype, sfa_smem_size],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[sf_dtype, sfb_smem_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.d_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
            TileSchedulerCls,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: Optional[cute.TiledMma],
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: Optional[cute.CopyAtom],
        mSFA_mkl: Optional[cute.Tensor],
        tma_atom_sfb: Optional[cute.CopyAtom],
        mSFB_nkl: Optional[cute.Tensor],
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: Optional[cute.Layout],
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: Optional[cute.Layout],
        sfb_smem_layout_staged: Optional[cute.Layout],
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: ParamsBase,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        epilogue_op: cutlass.Constexpr[Callable],
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if const_expr(self.blockscaled):
                cpasync.prefetch_descriptor(tma_atom_sfa)
                cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_d)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        if const_expr(self.blockscaled):
            block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )
        else:
            block_in_cluster_coord_sfb_vmnk = None
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        if const_expr(mC_mnl is not None):
            # Threads/warps participating in this pipeline
            epi_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            # Each warp will contribute 1 to the arrive count
            consumer_arrive_cnt = len(self.epilog_warp_id)
            epi_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, consumer_arrive_cnt
            )
            c_smem_layout = cute.slice_(epi_c_smem_layout_staged, (None, None, 0))
            tma_copy_c_bytes = cute.size_in_bytes(self.c_dtype, c_smem_layout)
            epi_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.epi_pipeline_array_ptr.data_ptr(),
                num_stages=self.num_c_stage,
                producer_group=epi_pipeline_producer_group,
                consumer_group=epi_pipeline_consumer_group,
                tx_count=tma_copy_c_bytes,
            )
        else:
            epi_pipeline = None

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # if const_expr(tile_sched_params.tile_count_semaphore is not None):
        #     # Dynamic persistent scheduler
        #     # Threads/warps participating in this pipeline
        #     sched_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        #     cluster_size = cute.size(cluster_layout_vmnk)
        #     # Each warp that are not the scheduler warp will contribute 1 to the arrive count
        #     consumer_arrive_cnt = (
        #         (self.mma_warp_groups if not self.pingpong else 1) * 4 + self.num_ab_load_warps
        #     ) * cluster_size - 1
        #     sched_pipeline_consumer_group = pipeline.CooperativeGroup(
        #         pipeline.Agent.Thread, consumer_arrive_cnt
        #     )
        #     sched_pipeline = pipeline.PipelineAsync.create(
        #         barrier_storage=storage.sched_pipeline_array_ptr.data_ptr(),
        #         num_stages=self.sched_stage,
        #         producer_group=sched_pipeline_producer_group,
        #         consumer_group=sched_pipeline_consumer_group,
        #         # If there's cluster, the consumers must arrive at the mbar of CTA 0 in the cluster.
        #         consumer_mask=None if const_expr(cute.size(cluster_layout_mnk) == 1) else 0,
        #     )
        #     tile_count = storage.tile_count.get_tensor((self.sched_stage,))
        # else:
        #     sched_pipeline = None
        #     tile_count = None

        # Setup smem tensor A/B/D
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        if const_expr(self.blockscaled):
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        else:
            sSFA, sSFB = None, None
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        if const_expr(mC_mnl is not None):
            sC = storage.sC.get_tensor(
                epi_c_smem_layout_staged.outer, swizzle=epi_c_smem_layout_staged.inner
            )
        else:
            sC = None

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = (
            tiled_mma_sfb.get_slice(mma_tile_coord_v) if const_expr(self.blockscaled) else None
        )

        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        tmem_ptr_read_threads = cute.arch.WARP_SIZE * len((self.mma_warp_id, *self.epilog_warp_id))
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_ptr_sync_bar_id, num_threads=tmem_ptr_read_threads
        )

        TileSchedulerCls = partial(TileSchedulerCls.create, tile_sched_params)
        k_tile_cnt = cute.ceil_div(cute.size(mA_mkl.shape[1]), self.mma_tiler[2])

        if const_expr(mC_mnl is not None):
            epi_load_barrier = pipeline.NamedBarrier(
                barrier_id=int(self.epilog_load_bar_id), num_threads=2 * cute.arch.WARP_SIZE
            )
        else:
            epi_load_barrier = None

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            # Compute multicast mask for A/B buffer full
            if const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
                a_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
                b_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
                )
                if const_expr(self.blockscaled):
                    sfa_mcast_mask = cpasync.create_tma_multicast_mask(
                        cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                    )
                    sfb_mcast_mask = cpasync.create_tma_multicast_mask(
                        cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
                    )
                else:
                    sfa_mcast_mask, sfb_mcast_mask = None, None
            else:
                a_mcast_mask, b_mcast_mask = None, None
                sfa_mcast_mask, sfb_mcast_mask = None, None

            # Persistent tile scheduling loop
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            do_epi_load_barrier_arrive = cutlass.Boolean(True)
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                tile_coord_mnkl = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    tile_coord_mnkl[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_coord_mnkl[1],
                    tile_coord_mnkl[3],
                )
                # Local_tile partition global tensors
                # (bM, bK, RestK)
                gA_mkl = cute.local_tile(
                    mA_mkl,
                    cute.slice_(self.mma_tiler, (None, 0, None)),
                    (mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]),
                )
                # (bN, bK, RestK)
                gB_nkl = cute.local_tile(
                    mB_nkl,
                    cute.slice_(self.mma_tiler, (0, None, None)),
                    (mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2]),
                )
                if const_expr(self.blockscaled):
                    # (bM, bK)
                    gSFA_mkl = cute.local_tile(
                        mSFA_mkl,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]),
                    )
                    # (bN, bK)
                    gSFB_nkl = cute.local_tile(
                        mSFB_nkl,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2]),
                    )
                # Partition global tensor for TiledMMA_A/B/D
                # (MMA, MMA_M, MMA_K, RestK)
                tCgA = thr_mma.partition_A(gA_mkl)
                # (MMA, MMA_N, MMA_K, RestK)
                tCgB = thr_mma.partition_B(gB_nkl)
                if const_expr(self.blockscaled):
                    # (MMA, MMA_M, MMA_K)
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)
                    # (MMA, MMA_N, MMA_K)
                    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
                # Partition global/shared tensor for TMA load A/B
                # TMA load A partition_S/D
                a_cta_layout = cute.make_layout(
                    cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
                )
                # ((atom_v, rest_v), STAGE)
                # ((atom_v, rest_v), RestK)
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(tCgA, 0, 3),
                )
                # TMA load B partition_S/D
                b_cta_layout = cute.make_layout(
                    cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                )
                # ((atom_v, rest_v), STAGE)
                # ((atom_v, rest_v), RestK)
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(tCgB, 0, 3),
                )
                if const_expr(self.blockscaled):
                    #  TMA load SFA partition_S/D
                    sfa_cta_layout = a_cta_layout
                    # ((atom_v, rest_v), STAGE)
                    # ((atom_v, rest_v), RestK)
                    tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_sfa,
                        block_in_cluster_coord_vmnk[2],
                        sfa_cta_layout,
                        cute.group_modes(sSFA, 0, 3),
                        cute.group_modes(tCgSFA, 0, 3),
                    )
                    tAsSFA = cute.filter_zeros(tAsSFA)
                    tAgSFA = cute.filter_zeros(tAgSFA)
                    # TMA load SFB partition_S/D
                    sfb_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
                    )
                    # ((atom_v, rest_v), STAGE)
                    # ((atom_v, rest_v), RestK)
                    tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_sfb,
                        block_in_cluster_coord_sfb_vmnk[1],
                        sfb_cta_layout,
                        cute.group_modes(sSFB, 0, 3),
                        cute.group_modes(tCgSFB, 0, 3),
                    )
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)
                else:
                    tAsSFA, tAgSFA = None, None
                    tBsSFB, tBgSFB = None, None
                ab_producer_state = self.load_AB(
                    ab_pipeline,
                    ab_producer_state,
                    tma_atom_a,
                    tAgA,
                    tAsA,
                    a_mcast_mask,
                    tma_atom_b,
                    tBgB,
                    tBsB,
                    b_mcast_mask,
                    tma_atom_sfa,
                    tAgSFA,
                    tAsSFA,
                    sfa_mcast_mask,
                    tma_atom_sfb,
                    tBgSFB,
                    tBsSFB,
                    sfb_mcast_mask,
                )
                if const_expr(epi_load_barrier is not None):
                    # In the first work tile, the epi load warp will wait for the signal
                    # from the mainloop load warp to start loading C, to avoid interfering
                    # with loading A and B.
                    if do_epi_load_barrier_arrive:
                        epi_load_barrier.arrive()
                        do_epi_load_barrier_arrive = cutlass.Boolean(False)
                # Advance to next tile
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            # Wait A/B buffer empty
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized TMA epi load warp
        #
        if const_expr(mC_mnl is not None):
            if warp_idx == self.tma_epi_warp_id:
                epi_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_c_stage
                )
                do_epi_load_barrier_wait = cutlass.Boolean(True)
                # Persistent tile scheduling loop
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    # Get tile coord from tile scheduler
                    tile_coord_mnkl = work_tile.tile_idx
                    mma_tile_coord_mnl = (
                        tile_coord_mnkl[0] // cute.size(tiled_mma.thr_id.shape),
                        tile_coord_mnkl[1],
                        tile_coord_mnkl[3],
                    )
                    # Local_tile partition global tensors
                    # (bM, bN)
                    gC_mnl = cute.local_tile(
                        mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), mma_tile_coord_mnl
                    )
                    # Partition global tensor for TiledMMA_A/B/D
                    # (MMA, MMA_M, MMA_N)
                    tCgC = thr_mma.partition_C(gC_mnl)
                    # bGS_gC has shape ((ATOM_V, REST_V), EPI_M, EPI_N)
                    bGS_sC, bGS_gC = self.epilog_gmem_copy_and_partition(
                        tma_atom_c, tCgC, epi_tile, sC
                    )
                    bGS_gC = cute.group_modes(bGS_gC, 1, cute.rank(bGS_gC))
                    if do_epi_load_barrier_wait:
                        epi_load_barrier.arrive_and_wait()
                        do_epi_load_barrier_wait = cutlass.Boolean(False)
                    epi_tile_num = const_expr(cute.size(bGS_gC, mode=[1]))
                    for subtile_idx in cutlass.range(epi_tile_num, unroll=1):
                        epi_pipeline.producer_acquire(epi_producer_state)
                        cute.copy(
                            tma_atom_c,
                            bGS_gC[None, subtile_idx],
                            bGS_sC[None, epi_producer_state.index],
                            tma_bar_ptr=epi_pipeline.producer_get_barrier(epi_producer_state),
                        )
                        # Epi pipeline's producer commit is a NOP
                        epi_pipeline.producer_commit(epi_producer_state)
                        epi_producer_state.advance()
                    # Advance to next tile
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                epi_pipeline.producer_tail(epi_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            tmem_alloc_barrier.arrive_and_wait()
            # Retrieving tensor memory ptr and make accumulator tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
            # Partition shared/tensor memory tensor for TiledMMA_A/B/D
            # (MMA, MMA_M, MMA_K, STAGE)
            tCrA = tiled_mma.make_fragment_A(sA)
            # (MMA, MMA_N, MMA_K, STAGE)
            tCrB = tiled_mma.make_fragment_B(sB)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            if const_expr(self.blockscaled):
                # Make SFA tmem tensor
                sfa_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                    dtype=self.sf_dtype,
                )
                # (MMA, MMA_M, MMA_K)
                tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

                # Make SFB tmem tensor
                sfb_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr
                    + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                    + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                    dtype=self.sf_dtype,
                )
                # (MMA, MMA_N, MMA_K)
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
                # Partition for S2T copy of SFA/SFB
                (
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t,
                    tCtSFA_compact_s2t,
                ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
                (
                    tiled_copy_s2t_sfb,
                    tCsSFB_compact_s2t,
                    tCtSFB_compact_s2t,
                ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            else:
                tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = None, None, None
                tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = None, None, None

            # Persistent tile scheduling loop
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                tile_coord_mnkl = work_tile.tile_idx
                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[None, None, None, acc_producer_state.index]
                ab_consumer_state, acc_producer_state, tiled_mma = self.mma(
                    ab_pipeline,
                    acc_pipeline,
                    ab_consumer_state,
                    acc_producer_state,
                    tiled_mma,
                    tCrA,
                    tCrB,
                    tCtAcc,
                    k_tile_cnt,
                    is_leader_cta,
                    tiled_copy_s2t_sfa,
                    tiled_copy_s2t_sfb,
                    tCsSFA_compact_s2t,
                    tCsSFB_compact_s2t,
                    tCtSFA_compact_s2t,
                    tCtSFB_compact_s2t,
                )
                # Advance to next tile
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            # Wait for accumulator buffer empty
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            # Alloc tensor memory buffer
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols, tmem_holding_buf, is_two_cta=use_2cta_instrs
                )
            # Bar sync for retrieve tensor memory ptr from shared memory
            tmem_alloc_barrier.arrive_and_wait()
            # Retrieving tensor memory ptr and make accumulator tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            epilog_threads = cute.arch.WARP_SIZE * len(self.epilog_warp_id)
            epilogue_barrier = pipeline.NamedBarrier(
                barrier_id=self.epilog_sync_bar_id, num_threads=epilog_threads
            )

            # Partition for epilogue
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, epi_tile, use_2cta_instrs
            )

            tTR_rD = cute.make_fragment(tTR_rAcc.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rD, epi_tidx, sD
            )
            if const_expr(mC_mnl is not None):
                tTR_rC = cute.make_fragment_like(tTR_rD, self.c_dtype)
                tiled_copy_s2r, tSR_rC, tSR_sC = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC
                )
                # TODO: for m major, D is being stored w STSM so we'd need LDSM here
                # tRS_rC = tSR_rC  # TODO: retile?
                tRS_rC = cute.make_fragment(tRS_rD.layout, self.c_dtype)
                tSR_rC = tiled_copy_s2r.get_slice(epi_tidx).retile(tRS_rC)

            # Persistent tile scheduling loop
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            # Threads/warps participating in tma store pipeline
            d_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
                32 * len(self.epilog_warp_id),
            )
            d_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_d_stage, producer_group=d_producer_group
            )
            epi_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_c_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                tile_coord_mnkl = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    tile_coord_mnkl[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_coord_mnkl[1],
                    tile_coord_mnkl[3],
                )
                # Local_tile partition global tensors
                # (bM, bN)
                gD_mnl = cute.local_tile(
                    mD_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), mma_tile_coord_mnl
                )
                # Partition global tensor for TiledMMA_A/B/D
                # (MMA, MMA_M, MMA_N)
                tDgD = thr_mma.partition_C(gD_mnl)
                # bSG_gD has shape ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_sD, bSG_gD = self.epilog_gmem_copy_and_partition(tma_atom_d, tDgD, epi_tile, sD)

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[None, None, None, None, None, acc_consumer_state.index]

                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))

                # Store accumulator to global memory in subtiles
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_scheduler.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    # Load accumulator from tensor memory buffer to register
                    tTR_tAcc_mn = tTR_tAcc[None, None, None, subtile_idx]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    # Convert to D type
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    acc_vec = epilogue_op(acc_vec)
                    if const_expr(mC_mnl is not None):
                        epi_pipeline.consumer_wait(epi_read_state)
                        cute.copy(
                            tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC
                        )
                        # Fence to make sure shared memory read is visible to TMA load
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                        )
                        cute.arch.sync_warp()
                        with cute.arch.elect_one():
                            epi_pipeline.consumer_release(epi_read_state)
                        epi_read_state.advance()
                        acc_vec = acc_vec + tRS_rC.load().to(self.acc_dtype)
                    tRS_rD.store(acc_vec.to(self.d_dtype))
                    # Store D to shared memory
                    d_buffer = (num_prev_subtiles + subtile_idx) % self.num_d_stage
                    cute.copy(tiled_copy_r2s, tRS_rD, tRS_sD[(None, None, None, d_buffer)])
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                    )
                    epilogue_barrier.arrive_and_wait()
                    # TMA store D to global memory
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(tma_atom_d, bSG_sD[None, d_buffer], bSG_gD[None, subtile_idx])
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        d_pipeline.producer_commit()
                        d_pipeline.producer_acquire()
                    epilogue_barrier.arrive_and_wait()

                # Async arrive accumulator buffer empty
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                # Advance to next tile
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            # Dealloc the tensor memory buffer
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            epilogue_barrier.arrive_and_wait()
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )

            # Wait for D store complete
            d_pipeline.producer_tail()

    @cute.jit
    def load_AB(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        tma_atom_a: cute.CopyAtom,
        tAgA: cute.Tensor,
        tAsA: cute.Tensor,
        a_mcast_mask: cutlass.Int16,
        tma_atom_b: cute.CopyAtom,
        tBgB: cute.Tensor,
        tBsB: cute.Tensor,
        b_mcast_mask: cutlass.Int16,
        tma_atom_sfa: Optional[cute.CopyAtom] = None,
        tAgSFA: Optional[cute.Tensor] = None,
        tAsSFA: Optional[cute.Tensor] = None,
        sfa_mcast_mask: Optional[cutlass.Int16] = None,
        tma_atom_sfb: Optional[cute.CopyAtom] = None,
        tBgSFB: Optional[cute.Tensor] = None,
        tBsSFB: Optional[cute.Tensor] = None,
        sfb_mcast_mask: Optional[cutlass.Int16] = None,
    ) -> cutlass.pipeline.PipelineState:
        blockscaled = const_expr(tma_atom_sfa is not None)
        if const_expr(blockscaled):
            assert all(x is not None for x in (tma_atom_sfa, tAgSFA, tAsSFA))
            assert all(x is not None for x in (tma_atom_sfb, tBgSFB, tBsSFB))
        k_tile_cnt = cute.size(tAgA, mode=[1])
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_ab_empty_status = cutlass.Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # /////////////////////////////////////////////////////////////////////////
        # TMA load
        # /////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            # Wait for A/B buffers to be empty before loading into them
            # Also sets the transaction barrier for the A/B buffers
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
            cute.copy(
                tma_atom_a,
                tAgA[None, k_tile],
                tAsA[None, ab_producer_state.index],
                tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                mcast_mask=a_mcast_mask,
            )
            cute.copy(
                tma_atom_b,
                tBgB[None, k_tile],
                tBsB[None, ab_producer_state.index],
                tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                mcast_mask=b_mcast_mask,
            )
            if const_expr(blockscaled):
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA[None, ab_producer_state.count],
                    tAsSFA[None, ab_producer_state.index],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    mcast_mask=sfa_mcast_mask,
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB[None, ab_producer_state.count],
                    tBsSFB[None, ab_producer_state.index],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    mcast_mask=sfb_mcast_mask,
                )
            # Mainloop pipeline's producer commit is a NOP
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
            peek_ab_empty_status = cutlass.Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        return ab_producer_state

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        acc_pipeline: cutlass.pipeline.PipelineAsync,
        ab_consumer_state: cutlass.pipeline.PipelineState,
        acc_producer_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        acc: cute.Tensor,
        k_tile_cnt: Int32,
        is_leader_cta: cutlass.Boolean,
        tiled_copy_s2t_sfa: Optional[cute.TiledCopy] = None,
        tiled_copy_s2t_sfb: Optional[cute.TiledCopy] = None,
        tCsSFA_compact_s2t: Optional[cute.Tensor] = None,
        tCsSFB_compact_s2t: Optional[cute.Tensor] = None,
        tCtSFA_compact_s2t: Optional[cute.Tensor] = None,
        tCtSFB_compact_s2t: Optional[cute.Tensor] = None,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState, cute.TiledMma]:
        blockscaled = const_expr(tiled_copy_s2t_sfa is not None)
        if const_expr(blockscaled):
            assert all(x is not None for x in (tiled_copy_s2t_sfa, tiled_copy_s2t_sfb))
            assert all(x is not None for x in (tCsSFA_compact_s2t, tCsSFB_compact_s2t))
            assert all(x is not None for x in (tCtSFA_compact_s2t, tCtSFB_compact_s2t))
        # Peek (try_wait) AB buffer full for k_tile = 0
        peek_ab_full_status = cutlass.Boolean(True)
        if 0 < k_tile_cnt and is_leader_cta:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        # Wait for accumulator buffer empty
        if is_leader_cta:
            acc_pipeline.producer_acquire(acc_producer_state)
        # Reset the ACCUMULATE field for each tile
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        # Mma mainloop
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            if is_leader_cta:
                # Conditionally wait for AB buffer full
                ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                #  Copy SFA/SFB from smem to tmem
                if const_expr(blockscaled):
                    s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                    tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                    tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                    cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged, tCtSFA_compact_s2t)
                    cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_staged, tCtSFB_compact_s2t)
                for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    k_blk_coord = (None, None, k_blk_idx, ab_consumer_state.index)
                    cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                # Async arrive AB buffer empty
                ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
            # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
            peek_ab_full_status = cutlass.Boolean(True)
            if k_tile + 1 < k_tile_cnt and is_leader_cta:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        # Async arrive accumulator buffer full
        if is_leader_cta:
            acc_pipeline.producer_commit(acc_producer_state)
        acc_producer_state.advance()
        # If we don't return the tiled_mma, we get compiler error
        # "operand #0 does not dominate this use"
        return ab_consumer_state, acc_producer_state, tiled_mma

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)
        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: Int32,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout,
            self.d_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        cAcc_epi = cute.flat_divide(cAcc, epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_cAcc = thr_copy_t2r.partition_D(cAcc_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_fragment(tTR_cAcc[None, None, None, 0, 0].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rD: cute.Tensor,
        tidx: Int32,
        sD: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rD: The partitioned accumulator tensor
        :type tTR_rD: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: Int32
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rD, tRS_sD) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rD: The partitioned tensor C (register source)
            - tRS_sD: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.d_layout, self.d_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        # (R2S, R2S_M, R2S_N)
        tRS_rD = tiled_copy_r2s.retile(tTR_rD)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    # def epilog_smem_load_copy_and_partition(
    #     self,
    #     tiled_copy_t2r: cute.TiledCopy,
    #     tTR_rC: cute.Tensor,
    #     tidx: Int32,
    #     sC: cute.Tensor,
    # ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
    #     copy_atom_s2r = cute.make_copy_atom(
    #         warp.LdMatrix8x8x16bOp(self.c_layout.is_m_major_c(), num_matrices=4),
    #         self.c_dtype,  # TODO: this probably only works for f16 for now?
    #     )
    #     # copy_atom_s2r = utils.sm90_get_smem_load_op(self.c_layout, self.c_dtype)
    #     tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
    #     # (R2S, R2S_M, R2S_N, PIPE_D)
    #     thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
    #     # (R2S, R2S_M, R2S_N)
    #     tSR_sC = thr_copy_s2r.partition_S(sC)
    #     return tiled_copy_s2r, tSR_sC

    def epilog_gmem_copy_and_partition(
        self,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gD_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gD_mnl: The global tensor C
        :type gD_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor

        :return: A tuple containing either:
            - For TMA store: (tma_atom_d, bSG_sD, bSG_gD) where:
                - tma_atom_d: The TMA copy atom
                - bSG_sD: The partitioned shared memory tensor C
                - bSG_gD: The partitioned global tensor C
            - For non-TMA store: (simt_atom, tTR_rD, tTR_gD) where:
                - simt_atom: The SIMT copy atom
                - tTR_rD: The register tensor C
                - tTR_gD: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        gD_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0)], epi_tile)
        sD_for_tma_partition = cute.group_modes(sD, 0, 2)
        gD_for_tma_partition = cute.group_modes(gD_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        bSG_sD, bSG_gD = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            sD_for_tma_partition,
            gD_for_tma_partition,
        )
        return bSG_sD, bSG_gD

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        d_dtype: Type[cutlass.Numeric],
        c_dtype: Optional[Type[cutlass.Numeric]],
        d_layout: cutlass.utils.LayoutEnum,
        c_layout: Optional[cutlass.utils.LayoutEnum],
        sf_dtype: Optional[Type[cutlass.Numeric]],
        sf_vec_size: Optional[int],
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param d_dtype: Data type of operand C (output).
        :type d_dtype: type[cutlass.Numeric]
        :param d_layout: Layout enum of operand C.
        :type d_layout: cutlass.utils.LayoutEnum
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        blockscaled = sf_dtype is not None
        # Default ACC stages
        if const_expr(not blockscaled):
            num_acc_stage = 2
        else:
            num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default D stages
        num_d_stage = 2
        num_c_stage = 2 if c_dtype is not None else 0

        # Calculate smem layout and size for one stage of A, B, and C
        a_smem_layout_staged_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        d_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)
        c_smem_layout_staged_one = (
            sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
            if c_dtype is not None
            else None
        )
        if const_expr(blockscaled):
            sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma,
                mma_tiler_mnk,
                sf_vec_size,
                1,  # a tmp 1 stage is provided
            )
            sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma,
                mma_tiler_mnk,
                sf_vec_size,
                1,  # a tmp 1 stage is provided
            )

        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_staged_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        if const_expr(blockscaled):
            ab_bytes_per_stage += cute.size_in_bytes(
                sf_dtype, sfa_smem_layout_staged_one
            ) + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        d_bytes_per_stage = cute.size_in_bytes(d_dtype, d_smem_layout_staged_one)
        epi_bytes = d_bytes_per_stage * num_d_stage
        if const_expr(c_dtype is not None):
            c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
            epi_bytes += c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_d_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + epi_bytes)
        ) // (occupancy * d_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_d_stage, num_c_stage

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]
        :param num_acc_stage: The stage of the accumulator tensor.
        :type num_acc_stage: int

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = cutlass.utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        return num_tmem_alloc_cols

    @staticmethod
    def is_valid_dtypes(
        ab_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes are valid

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if ab_dtype not in {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.TFloat32,
            cutlass.Uint8,
            cutlass.Int8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        if (
            acc_dtype not in {cutlass.Float32, cutlass.Float16, Int32}
            or acc_dtype == cutlass.Float16
            and ab_dtype not in {cutlass.Float16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}
            or acc_dtype == Int32
            and ab_dtype not in {cutlass.Uint8, cutlass.Int8}
        ):
            is_valid = False
        if (
            acc_dtype == cutlass.Float32
            and d_dtype
            not in {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
                Int32,
                cutlass.Int8,
                cutlass.Uint8,
            }
            or acc_dtype == cutlass.Float16
            and d_dtype
            not in {
                cutlass.BFloat16,
                cutlass.Float16,
            }
            or acc_dtype == Int32
            and d_dtype
            not in {
                cutlass.BFloat16,
                cutlass.Float16,
                cutlass.Float32,
                Int32,
                cutlass.Int8,
                cutlass.Uint8,
            }
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        d_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype not in {cutlass.Float4E2M1FN, cutlass.Float8E5M2, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid d_dtype
        if d_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param d_major: The major dimension of the C tensor
        :type d_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        blockscaled: bool,
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param use_2cta_instrs: Whether to use 2 CTA groups
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if not (
            (not use_2cta_instrs and mma_tiler_mn[0] in [64, 128])
            or (use_2cta_instrs and mma_tiler_mn[0] in [128, 256])
        ):
            is_valid = False
        if not blockscaled:
            if mma_tiler_mn[1] not in range(32, 257, 32):
                is_valid = False
        else:
            if mma_tiler_mn[1] not in [128, 256]:
                is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if use_2cta_instrs else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        if blockscaled:
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            if cluster_shape_mn[0] > 4 or cluster_shape_mn[1] > 4:
                is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        d_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param d_major: The major axis of the C tensor
        :type d_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(d_dtype, d_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        d_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param use_2cta_instrs: Whether to use 2 CTA groups
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param d_major: The major axis of the C tensor
        :type d_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not PersistentDenseGemmKernel.is_valid_dtypes(ab_dtype, acc_dtype, d_dtype):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not PersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, blockscaled=False
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not PersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, d_dtype, a_major, b_major, d_major
        ):
            can_implement = False
        return can_implement


def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
    c_dtype: Optional[Type[cutlass.Numeric]],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    d_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    use_2cta_instrs: bool = True,
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    dynamic_persistent: bool = False,
    **kwargs,
):
    """Execute a persistent batched dense GEMM operation on Blackwell architecture with performance benchmarking.

    This function prepares input tensors, configures and launches the persistent GEMM kernel,
    optionally performs reference validation, and benchmarks the execution performance.

    :param mnkl: Problem size (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param ab_dtype: Data type for input tensors A and B
    :type ab_dtype: Type[cutlass.Numeric]
    :param d_dtype: Data type for output tensor C
    :type d_dtype: Type[cutlass.Numeric]
    :param acc_dtype: Data type for accumulation during matrix multiplication
    :type acc_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/d_major: Memory layout of tensor A/B/C
    :type a_major/b_major/d_major: str
    :param mma_tiler_mn: MMA tiling size. If not specified in the decorator parameters, the autotuner will use the
        default value of (256, 256). Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type mma_tiler_mn: Tuple[int, int], optional
    :param cluster_shape_mn: Cluster shape. If not specified in the decorator parameters, the autotuner will use the
        default value of (2, 1). Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type cluster_shape_mn: Tuple[int, int], optional
    :param use_2cta_instrs: Whether to use 2CTA instructions. If not specified in the decorator parameters, the autotuner
        will use the default value of True. Otherwise, the autotuner will use the value specified in the decorator parameters.
    :type use_2cta_instrs: bool, optional
    :param tolerance: Tolerance value for reference validation comparison, defaults to 1e-01
    :type tolerance: float, optional
    :param warmup_iterations: Number of warmup iterations before benchmarking, defaults to 0
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations to run, defaults to 1
    :type iterations: int, optional
    :param skip_ref_check: Whether to skip reference result validation, defaults to False
    :type skip_ref_check: bool, optional
    :raises RuntimeError: If CUDA GPU is not available
    :raises ValueError: If the configuration is invalid or unsupported by the kernel
    :return: Execution time of the GEMM kernel
    :rtype: float
    """
    print("Running Blackwell Persistent Dense GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, C dtype: {d_dtype}, Acc dtype: {acc_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {d_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"2CTA MMA instructions: {'True' if use_2cta_instrs else 'False'}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")

    assert not dynamic_persistent, "Dynamic persistent mode is not supported yet."

    # Unpack parameters
    m, n, k, l = mnkl

    # Skip unsupported testcase
    if not PersistentDenseGemmKernel.can_implement(
        ab_dtype,
        acc_dtype,
        d_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        d_major,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {acc_dtype}, {d_dtype}, {use_2cta_instrs}, {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {d_major}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True):
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # else: (l, mode0, mode1) -> (mode0, mode1, l)
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        is_unsigned = dtype in {cutlass.Uint8}
        # Temporarily use uint8 as torch does not support fp8 type
        torch_dtype = cutlass_torch.dtype(dtype)
        gen_dtype = (
            torch_dtype
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch.bfloat16
        )

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            shape,
            gen_dtype,
            permute_order=permute_order,
            # init_type=cutlass.torch.TensorInitType.RANDOM,
            # init_config=cutlass.torch.RandomInitConfig(
            #     min_val=0 if is_unsigned else -2, max_val=4 if is_unsigned else 2
            # ),
            init_type=cutlass.torch.TensorInitType.GAUSSIAN,
            init_config=cutlass.torch.GaussianInitConfig(std=k ** (-0.5), scale=1),
        ).to(torch_dtype)
        # Create dtype torch tensor (gpu)
        torch_tensor = torch_tensor_cpu.cuda()

        # Create f32 torch tensor (cpu)
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # Create dtype cute tensor (gpu)
        torch_tensor_view = (
            torch_tensor
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch_tensor.view(torch.uint8)
        )
        cute_tensor = from_dlpack(torch_tensor_view, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=(0 if is_mode0_major else 1))
        cute_tensor = cutlass_torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor, torch_tensor_cpu

    a_ref, mA, a_torch, a_torch_cpu = create_and_permute_tensor(
        l, m, k, a_major == "m", ab_dtype, is_dynamic_layout=True
    )
    b_ref, mB, b_torch, b_torch_cpu = create_and_permute_tensor(
        l, n, k, b_major == "n", ab_dtype, is_dynamic_layout=True
    )
    _, mD, d_torch, d_torch_cpu = create_and_permute_tensor(
        l, m, n, d_major == "m", d_dtype, is_dynamic_layout=True
    )
    if c_dtype is not None:
        c, mC, c_torch, d_torch_cpu = create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)
    else:
        c, mC, c_torch = None, None, None

    # Configure gemm kernel
    gemm = PersistentDenseGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
    )

    # Compute max active clusters on current device
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    if dynamic_persistent:
        tile_count_semaphore = torch.zeros(1, dtype=torch.int32, device="cuda")
    else:
        tile_count_semaphore = None

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    # Compile gemm kernel
    compiled_gemm = cute.compile(
        gemm,
        mA,
        mB,
        mD,
        mC,
        make_ptr(Int32, tile_count_semaphore.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
        if tile_count_semaphore is not None
        else None,
        max_active_clusters,
        current_stream,
    )

    if not skip_ref_check:
        compiled_gemm(mA, mB, mD, mC, tile_count_semaphore, current_stream)
        if ab_dtype in {
            cutlass.Int8,
            cutlass.Uint8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            ref = torch.einsum("mkl,nkl->mnl", a_ref.cpu(), b_ref.cpu())
        else:
            ref = torch.einsum("mkl,nkl->mnl", a_ref, b_ref)
        if c is not None:
            ref = ref + c
        ref = ref.cpu()

        # Copy gpu result back
        gpu_d = d_torch.cpu()

        # Convert ref to c_type
        if d_dtype == cutlass.Float32:
            ref_d = ref
        elif d_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}:
            # m major: (l, n, m) -> (m, n, l)
            # n major: (l, m, n) -> (m, n, l)
            permute_order = (1, 2, 0) if d_major == "n" else (2, 1, 0)
            shape = (l, m, n) if d_major == "n" else (l, n, m)
            f8_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
                shape,
                torch.uint8,
                permute_order=permute_order,
                init_type=cutlass_torch.TensorInitType.SKIP,
            ).cuda()
            # Create dtype cute tensor (gpu)
            ref_d_tensor = from_dlpack(f8_torch_tensor, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if d_major == "n" else 0)
            )
            ref_d_tensor.element_type = d_dtype
            ref_d_tensor = cutlass_torch.convert_cute_tensor(
                ref,
                ref_d_tensor,
                d_dtype,
                is_dynamic_layout=True,
            )

            ref_d = f8_torch_tensor.cpu()
        else:
            ref_d = ref.to(cutlass_torch.dtype(d_dtype))

        # Reference checking ref_d and gpu_d
        torch.testing.assert_close(gpu_d, ref_d, atol=tolerance, rtol=1e-05)

    from triton.testing import do_bench

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    flops = 2 * m * n * k * l

    repeats = iterations
    warmup = warmup_iterations

    import time

    time.sleep(0.5)
    if ab_dtype.width == 8:
        assert l == 1
        scale_ab = torch.ones((1,), dtype=torch.float32, device="cuda")
        fn_cublas = lambda: torch._scaled_mm(
            a_torch[:, :, 0],
            b_torch[:, :, 0].mT,
            scale_a=scale_ab,
            scale_b=scale_ab,
            out_dtype=torch.bfloat16,
            # use_fast_accum=fp8_fast_accum,
        )
    else:
        if c_torch is None:
            fn_cublas = lambda: torch.matmul(a_torch.permute(2, 0, 1), b_torch.permute(2, 0, 1).mT)
        else:
            c_torch_convert = c_torch.to(a_torch.dtype)  # In case C is in FP32
            fn_cublas = lambda: torch.baddbmm(
                c_torch_convert.permute(2, 0, 1),
                a_torch.permute(2, 0, 1),
                b_torch.permute(2, 0, 1).mT,
            )
    timing_cublas = do_bench(fn_cublas, warmup=warmup, rep=repeats)
    tflops_cublas = flops / (timing_cublas * 1e9)  # Convert to TFlops
    print(f"CuBLAS Average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")

    time.sleep(0.5)
    fn = lambda: compiled_gemm(mA, mB, mD, mC, tile_count_semaphore, current_stream)
    timing = do_bench(fn, warmup=warmup, rep=repeats)
    tflops = flops / (timing * 1e9)  # Convert to TFlops
    print(f"Cute-DSL Average time: {timing:.3f} ms, TFLOPS: {tflops:.1f}")

    # time.sleep(0.5)
    # timing_cublas = do_bench(fn_cublas, warmup=warmup, rep=repeats)
    # tflops_cublas = flops / (timing_cublas * 1e9)  # Convert to TFlops
    # print(f"CuBLAS Average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")

    parser = argparse.ArgumentParser(description="Example of Dense Persistent GEMM on Blackwell.")

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(256, 256, 512, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--d_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=None)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument(
        "--use_2cta_instrs",
        action="store_true",
        help="Enable 2CTA MMA instructions feature",
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--d_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")

    parser.add_argument("--tolerance", type=float, default=3e-02, help="Tolerance for validation")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")
    parser.add_argument(
        "--dynamic_persistent", action="store_true", help="Dynamic persistent kernel"
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    run(
        args.mnkl,
        args.ab_dtype,
        args.d_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.d_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.use_2cta_instrs,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.dynamic_persistent,
    )
    print("PASS")
