# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py

import enum
from typing import Tuple, Type, Callable, Optional, Union
from functools import partial
import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32, Boolean, const_expr


from quack.cute_dsl_utils import ParamsBase, ArgumentsBase
from quack.tile_scheduler import (
    TileSchedulerArguments,
    TileScheduler,
    VarlenMTileSchedulerArguments,
    VarlenMTileScheduler,
    RasterOrderOption,
)
from quack.tensormap_manager import TensorMapManagerSm90

# return PipelineStateWAdvance instead of PipelineState
from quack.pipeline import make_pipeline_state, PipelineTmaCpAsync
import quack.utils as utils

"""
A high-performance batched dense GEMM (C = A * B) example for the NVIDIA Hopper architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using WGMMA instruction.
3. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Hopper WGMMA instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

Constraints:
* Supported input data types: fp16, fp8 (e4m3fn, e5m2)
* For fp16 types, A and B must have the same data type
* For fp8 types, A and B can have different types (e4m3fn or e5m2) but both must be 8-bit
* Fp8 types only support k-major layout
* Only fp32 accumulation is supported in this example
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* CTA tile shape K must be 64
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
"""


class NamedBarrierGemm(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    # For mainloop load warps to signal that the epilogue load warp can start.
    # This is to avoid loading C too early, interfering with loading A and B.
    EpilogueLoad = enum.auto()
    MmaWG0 = enum.auto()
    MmaWG1 = enum.auto()
    EpiWG0 = enum.auto()
    EpiWG1 = enum.auto()


class GemmSm90:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mnk: Shape of the CTA tile (M,N,K)
    :type tile_shape_mnk: Tuple[int, int, int]
    :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
    :type cluster_shape_mnk: Tuple[int, int, int]

    :note: Data type requirements:
        - For 16-bit types: A and B must have the same data type
        - For 8-bit types: A and B can have different types (Float8E4M3FN/Float8E5M2) as long as both are 8-bit
        - Float8 types only support k-major layout

    :note: Supported data types:
        - Float16
        - BFloat16
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulation types:
        - Float32 (for all floating point inputs)

    :note: Constraints:
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = GemmSm90(
        ...     acc_dtype=cutlass.Float32,
        ...     tile_shape_mnk=(128, 256, 64),
        ...     cluster_shape_mnk=(1, 1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    bytes_per_tensormap = 128
    num_tensormaps = 1  # For D only

    EpilogueArguments = ArgumentsBase
    EpilogueParams = ParamsBase

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = True,
        fp8_fast_accum: bool = False,
        gather_A: bool = False,
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mnk: Shape of the CTA tile (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
        :type cluster_shape_mnk: Tuple[int, int, int]
        """

        self.acc_dtype = acc_dtype
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"
        self.fp8_slow_accum = not fp8_fast_accum and a_dtype.width == 8
        self.gather_A = gather_A
        if gather_A:
            assert cluster_shape_mnk[1] == 1, "Cluster shape N must be 1 for gather A "
        self.tensormap_update_mode = cutlass.utils.TensorMapUpdateMode.SMEM

        self.cluster_shape_mnk = cluster_shape_mnk
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        tile_M, tile_N = tile_shape_mnk[0], tile_shape_mnk[1]
        # check the cta tile shape
        if not self.pingpong:
            if tile_M not in [64, 128, 192, 256, 320]:
                raise ValueError("CTA tile shape M must be 64/128/192/256/320")
            if tile_M in [192, 320]:  # special case
                tile_N_max = 256 if tile_M == 192 else 160
                if not (tile_N % 32 == 0 and tile_N <= tile_N_max):
                    raise ValueError(
                        f"If tile_m == {tile_M}, CTA tile shape N must be divisible by 32 and <= {tile_N_max}"
                    )
            else:
                if not (
                    (tile_N % 16 == 0 and tile_N <= 256) or (tile_N % 32 == 0 and tile_N <= 512)
                ):
                    raise ValueError(
                        "CTA tile shape N must be divisible by 16 and <= 256, or divisible by 32 and <= 512"
                    )
        else:
            if tile_M not in [64, 128, 192]:
                raise ValueError("CTA tile shape M must be 64/128/192 if pingpong")
            tile_N_max = 256 if tile_M == 64 else (208 if tile_M == 128 else 128)
            if not (tile_N % 16 == 0 and tile_N <= tile_N_max):
                raise ValueError(f"CTA tile shape N must be divisible by 16 and <= {tile_N_max}")
        if not self.tile_shape_mnk[2] % 16 == 0:
            raise ValueError("CTA tile shape K must be divisible by 16")

        if not self.pingpong:
            if tile_M == 320:  # tile_M / 64 is not even so we have to split along N
                atom_layout_m, atom_layout_n = 1, 2
            elif tile_M == 192:
                if tile_N <= 128:
                    atom_layout_m, atom_layout_n = 3, 1
                else:
                    atom_layout_m, atom_layout_n = 1, 2
            else:
                atom_layout_m = tile_shape_mnk[0] // 64 if tile_shape_mnk[0] < 256 else 2
                atom_layout_n = 1
            assert atom_layout_m in [1, 2, 3] and atom_layout_n in [1, 2]
        else:
            atom_layout_m, atom_layout_n = 1, 1
        self.atom_layout_mnk = (atom_layout_m, atom_layout_n, 1)

        self.num_mcast_ctas_a = self.cluster_shape_mnk[1] if not self.gather_A else 1
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk) * (1 if not self.pingpong else 2)
        if self.pingpong:
            assert self.mma_warp_groups == 2
        assert self.mma_warp_groups in [1, 2, 3]
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (self.mma_warp_groups + 1) * self.num_threads_per_warp_group
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes("sm_90")
        self.num_epi_threads = (
            self.mma_warp_groups if not self.pingpong else 1
        ) * self.num_threads_per_warp_group
        self.num_ab_load_warps = 1 if not self.gather_A else 4
        self.num_ab_load_threads = cute.arch.WARP_SIZE * self.num_ab_load_warps
        self.num_epi_load_threads = cute.arch.WARP_SIZE * 1
        self.ab_load_warp_id = self.mma_warp_groups * 4
        self.epi_load_warp_id = self.ab_load_warp_id + self.num_ab_load_warps

        regs_per_thread = math.prod(self.tile_shape_mnk[:2]) // (
            math.prod(self.atom_layout_mnk) * self.num_threads_per_warp_group
        )
        if self.fp8_slow_accum:
            regs_per_thread *= 2
        if not self.gather_A:
            if self.mma_warp_groups == 3:
                self.num_regs_load, self.num_regs_mma = 32, 160
            else:
                heavy_register_pressure = regs_per_thread >= 208
                self.num_regs_load, self.num_regs_mma = (
                    (40, 232) if not heavy_register_pressure else (24, 240)
                )
        else:
            if self.mma_warp_groups == 3:
                self.num_regs_load, self.num_regs_mma = 56, 152
            else:
                self.num_regs_load, self.num_regs_mma = (56, 224)

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def _setup_attributes(self, epilogue_args: EpilogueArguments):
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
        """

        self.cluster_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk,
            self.atom_layout_mnk,
            self.d_dtype,
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage, self.epi_c_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.d_dtype,
            self.c_dtype,
            epilogue_args,
            self.smem_capacity,
            self.occupancy,
            # epi_smem will reuse smem ab if not persistent.
            overlap_sD_sA=not self.is_persistent,
        )
        self.sched_stage = 2 if self.pingpong else 1

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.d_dtype,
            self.d_layout,
            self.epi_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_c_stage,
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mC: Optional[cute.Tensor],
        epilogue_args: EpilogueArguments,
        mAIdx: Optional[cute.Tensor],
        mCuSeqlensM: Optional[cute.Tensor],
        mTensormaps: Optional[cute.Tensor],
        tile_count_semaphore: Optional[cute.Pointer],
        max_active_clusters: Int32,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param mA: Input tensor A
        :type mA: cute.Tensor
        :param mB: Input tensor B
        :type mB: cute.Tensor
        :param mD: Output tensor D
        :type mD: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type
        self.c_dtype = mC.element_type if mC is not None else None
        self.a_layout = cutlass.utils.LayoutEnum.from_tensor(mA)
        self.b_layout = cutlass.utils.LayoutEnum.from_tensor(mB)
        self.d_layout = cutlass.utils.LayoutEnum.from_tensor(mD)
        self.c_layout = cutlass.utils.LayoutEnum.from_tensor(mC) if mC is not None else None

        if const_expr(self.a_dtype.width == 16 and self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}")
        if const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")
        assert (mAIdx is not None) == self.gather_A

        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: tuple(
            cute.assume(s, divby=128 // t.element_type.width) if not cute.is_static(s) else s
            for s in t.stride
        )
        mA, mD = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mA, mD)
        ]

        self._setup_attributes(epilogue_args)

        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1] // self.atom_layout_mnk[1]),
        )
        if const_expr(self.atom_layout_mnk[1] > 1):
            # If N dimension is split among 2 WGs, we need to permute the N dimension so
            # that in the epilogue, WG0 and WG1 can write to epi smem of size e.g. (64, 32)
            # containing accumulators that are next to each other in the N dimension.
            # Without permutation WG0 would write to epi smem of size (64, 16) and
            # WG1 would write to a separate epi smem of size (64, 16) that's far away.
            atom_n = self.atom_layout_mnk[1]
            permutation_n = cute.make_ordered_layout(
                (8, self.tile_shape_mnk[1] // atom_n // 8, atom_n), order=(0, 2, 1)
            )
            tiled_mma = cute.make_tiled_mma(
                cute.make_mma_atom(tiled_mma.op),
                self.atom_layout_mnk,
                permutation_mnk=(None, permutation_n, None),
            )

        if const_expr(not self.gather_A):
            tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
                mA,
                self.a_smem_layout_staged,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                self.cluster_shape_mnk[1],
            )
        else:
            tma_atom_a, tma_tensor_a = None, None

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            mB,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )

        tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
            mD, self.epi_smem_layout_staged, self.epi_tile, store_or_load="store"
        )

        if const_expr(mC is not None):
            tma_atom_c, tma_tensor_c = self._make_tma_epi_atoms_and_tensors(
                mC, self.epi_c_smem_layout_staged, self.epi_tile, store_or_load="load"
            )
        else:
            tma_atom_c, tma_tensor_c = None, None

        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)

        if const_expr(mCuSeqlensM is None):
            problem_shape_ntile_mnl = cute.ceil_div(mD.shape[:2], self.tile_shape_mnk[:2]) + (
                mD.shape[2],
            )
            TileSchedulerCls = TileScheduler
            tile_sched_args = TileSchedulerArguments(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                raster_order=RasterOrderOption.Heuristic,
                group_size=8,
                cluster_shape_mnk=self.cluster_shape_mnk,
                tile_count_semaphore=tile_count_semaphore,
                is_persistent=self.is_persistent,
            )
        else:
            assert mTensormaps is not None
            problem_shape_ntile_mnl = (
                None,
                cute.ceil_div(mD.shape[1], self.tile_shape_mnk[1]),
                mCuSeqlensM.shape[0] - 1,
            )
            TileSchedulerCls = VarlenMTileScheduler
            tile_sched_args = VarlenMTileSchedulerArguments(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                total_m=mD.shape[0],
                cu_seqlens_m=mCuSeqlensM,
                raster_order=RasterOrderOption.Heuristic,
                group_size=8,
                tile_shape_mnk=self.tile_shape_mnk,
                cluster_shape_mnk=self.cluster_shape_mnk,
                tile_count_semaphore=tile_count_semaphore,
                is_persistent=self.is_persistent,
            )
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
        grid = TileSchedulerCls.get_grid_shape(tile_sched_params, max_active_clusters)

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged) if self.is_persistent else 0
        epi_c_smem_size = cute.cosize(self.epi_c_smem_layout_staged) if mC is not None else 0

        size_tensormap_in_i64 = (
            0
            if mCuSeqlensM is None
            or self.tensormap_update_mode == cutlass.utils.TensorMapUpdateMode.GMEM
            else GemmSm90.num_tensormaps * GemmSm90.bytes_per_tensormap // 8
        ) * (1 if not self.pingpong else 2)

        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, size_tensormap_in_i64],
                64,
            ]
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            tile_count: cute.struct.MemRange[cutlass.Int32, self.sched_stage]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, epi_smem_size],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype if self.c_dtype is not None else Int32, epi_c_smem_size
                ],
                self.buffer_align_bytes,
            ]
            epi: self.epi_get_smem_struct(epilogue_params)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a if const_expr(not self.gather_A) else mA,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            epilogue_params,
            mAIdx,
            mCuSeqlensM,
            mTensormaps,
            tiled_mma,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            tile_sched_params,
            TileSchedulerCls,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: Optional[cute.CopyAtom],
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        epilogue_params: EpilogueParams,
        mAIdx: Optional[cute.Tensor],
        cu_seqlens_m: Optional[cute.Tensor],
        tensormaps: Optional[cute.Tensor],
        tiled_mma: cute.TiledMma,
        cluster_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_c_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: ParamsBase,
        TileSchedulerCls: cutlass.Constexpr[Callable],
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_d: TMA copy atom for D tensor
        :type tma_atom_d: cute.CopyAtom
        :param mD_mnl: Output tensor D
        :type mD_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cluster_layout_mnk: CTA layout
        :type cluster_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """

        varlen = const_expr(cu_seqlens_m is not None)
        has_C = const_expr(mC_mnl is not None)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.ab_load_warp_id:
            if const_expr(tma_atom_a is not None):
                cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_d)
            if const_expr(tma_atom_c is not None):
                cpasync.prefetch_descriptor(tma_atom_c)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        if const_expr(not self.gather_A):
            tma_copy_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Threads/warps participating in this pipeline
        producer_cnt = 1 if const_expr(not self.gather_A) else 1 + self.num_ab_load_threads
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, producer_cnt)
        # Each warp will contribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * (tiled_mma.size // cute.arch.WARP_SIZE)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cluster_layout_mnk.shape))
        pipeline_cls = pipeline.PipelineTmaAsync if not self.gather_A else PipelineTmaCpAsync
        ab_pipeline = pipeline_cls.create(
            barrier_storage=storage.ab_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if const_expr(has_C):
            # Threads/warps participating in this pipeline
            epi_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            # Each warp will contribute 1 to the arrive count
            consumer_arrive_cnt = self.num_epi_threads // cute.arch.WARP_SIZE
            epi_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, consumer_arrive_cnt
            )
            c_smem_layout = cute.slice_(epi_c_smem_layout_staged, (None, None, 0))
            tma_copy_c_bytes = cute.size_in_bytes(self.c_dtype, c_smem_layout)
            epi_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.epi_pipeline_array_ptr.data_ptr(),
                num_stages=self.epi_c_stage,
                producer_group=epi_pipeline_producer_group,
                consumer_group=epi_pipeline_consumer_group,
                tx_count=tma_copy_c_bytes,
            )
        else:
            epi_pipeline = None

        if const_expr(tile_sched_params.tile_count_semaphore is not None):
            # Dynamic persistent scheduler
            # Threads/warps participating in this pipeline
            sched_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            cluster_size = cute.size(cluster_layout_mnk)
            # Each warp that are not the scheduler warp will contribute 1 to the arrive count
            consumer_arrive_cnt = (
                (self.mma_warp_groups if not self.pingpong else 1) * 4 + self.num_ab_load_warps
            ) * cluster_size - 1
            sched_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, consumer_arrive_cnt
            )
            sched_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.sched_pipeline_array_ptr.data_ptr(),
                num_stages=self.sched_stage,
                producer_group=sched_pipeline_producer_group,
                consumer_group=sched_pipeline_consumer_group,
                # If there's cluster, the consumers must arrive at the mbar of CTA 0 in the cluster.
                consumer_mask=None if const_expr(cute.size(cluster_layout_mnk) == 1) else 0,
            )
            tile_count = storage.tile_count.get_tensor((self.sched_stage,))
        else:
            sched_pipeline = None
            tile_count = None

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        if const_expr(not self.is_persistent):
            sD_ptr = cute.recast_ptr(sA.iterator, epi_smem_layout_staged.inner, dtype=self.d_dtype)
            sD = cute.make_tensor(sD_ptr, epi_smem_layout_staged.outer)
        else:
            sD = storage.sD.get_tensor(
                epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
            )
        if const_expr(has_C):
            sC = storage.sC.get_tensor(
                epi_c_smem_layout_staged.outer, swizzle=epi_c_smem_layout_staged.inner
            )
        else:
            sC = None
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

        # Get tensormap buffer address
        if const_expr(varlen):
            grid_dim = cute.arch.grid_dim()
            bid = cute.arch.block_idx()
            tensormap_workspace_idx = (
                bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]
            )
            # TODO: this is only for D, not for A/B
            if const_expr(self.pingpong):
                tensormap_workspace_idx = tensormap_workspace_idx * 2 + warp_idx // 4
            tensormap_manager = TensorMapManagerSm90(
                self.tensormap_update_mode, GemmSm90.bytes_per_tensormap
            )
            tensormap_d_ptr = tensormap_manager.get_tensormap_ptr(
                tensormaps[tensormap_workspace_idx, None].iterator
            )
            if const_expr(self.tensormap_update_mode == cutlass.utils.TensorMapUpdateMode.SMEM):
                tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
                tensormap_d_smem_ptr = tensormap_smem_ptr + (warp_idx // 4) * (
                    GemmSm90.bytes_per_tensormap // 8
                )
                # Need this, otherwise "expected tma descriptor pointer to have alignment at least 64, but got 8"
                tensormap_d_smem_ptr = cute.make_ptr(
                    cutlass.Int64,
                    tensormap_d_smem_ptr.toint(),
                    cute.AddressSpace.smem,
                    assumed_align=64,
                )
                tensormap_d_init_ptr = tensormap_d_smem_ptr
            else:
                tensormap_d_smem_ptr = None
                tensormap_d_init_ptr = tensormap_d_ptr
        else:
            tensormap_d_smem_ptr = None
            tensormap_manager, tensormap_d_ptr, tensormap_d_init_ptr = None, None, None

        TileSchedulerCls = partial(
            TileSchedulerCls.create, tile_sched_params, tile_count, sched_pipeline
        )

        k_tile_cnt = cute.ceil_div(cute.size(mA_mkl.shape[1]), self.tile_shape_mnk[2])
        c_tile_cnt = cute.size(cute.ceil_div(self.tile_shape_mnk[:2], self.epi_tile))

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
            if (
                warp_idx >= self.ab_load_warp_id
                and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
                # ///////////////////////////////////////////////////////////////////////////////
                # Get mcast mask
                # ///////////////////////////////////////////////////////////////////////////////
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, cluster_coord_mnk, mode=1
                )
                b_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, cluster_coord_mnk, mode=0
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                # Persistent tile scheduling loop
                is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                if const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                tile_scheduler = TileSchedulerCls(is_scheduler_warp=is_scheduler_warp)
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    # ///////////////////////////////////////////////////////////////////////////
                    #  Local_tile partition global tensors
                    # ///////////////////////////////////////////////////////////////////////////
                    if const_expr(not self.gather_A):
                        if const_expr(cu_seqlens_m is not None):
                            mA_mk = cute.domain_offset((cu_seqlens_m[batch_idx], 0), mA_mkl)
                        else:
                            mA_mk = mA_mkl[None, None, batch_idx]
                        # (bM, bK, RestK)
                        gA_k = cute.local_tile(
                            mA_mk,
                            cute.select(self.tile_shape_mnk, [0, 2]),
                            (tile_coord_mnkl[0], None),
                        )
                    else:
                        mA_mk = mA_mkl
                        if const_expr(cu_seqlens_m is not None):
                            mAIdx_mk = cute.domain_offset((cu_seqlens_m[batch_idx],), mAIdx)
                        else:
                            mAIdx_mk = mAIdx[None, batch_idx]
                        gAIdx = cute.local_tile(
                            mAIdx_mk, (self.tile_shape_mnk[0],), (tile_coord_mnkl[0],)
                        )
                    # (bN, bK, RestK)
                    gB_k = cute.local_tile(
                        mB_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)
                    )
                    # //////////////////////////////////////////////////////////////////////////
                    #  Partition shared tensor for TMA load A/B
                    # //////////////////////////////////////////////////////////////////////////
                    #  TMA load A partition_S/D
                    a_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_mnk, (0, None, 0)).shape
                    )
                    a_cta_crd = cluster_coord_mnk[1]
                    if const_expr(not self.gather_A):
                        # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
                        tAsA, tAgA_k = cpasync.tma_partition(
                            tma_atom_a,
                            a_cta_crd,
                            a_cta_layout,
                            cute.group_modes(sA, 0, 2),
                            cute.group_modes(gA_k, 0, 2),
                        )
                        copy_A = partial(cute.copy, tma_atom_a, mcast_mask=a_mcast_mask)
                    else:
                        tiled_copy_A = self._make_gmem_tiled_copy_A(
                            mA_mkl.element_type, self.a_layout, self.num_ab_load_threads
                        )
                        tidx = (
                            cute.arch.thread_idx()[0]
                            - self.mma_warp_groups * self.num_threads_per_warp_group
                        )
                        thr_copy_A = tiled_copy_A.get_slice(tidx)
                        # (atom_v, CPY_M, 1, STAGE)
                        tAsA = thr_copy_A.partition_D(sA)
                        assert tAsA.shape[2] == 1
                        tAsA = cute.group_modes(cute.slice_(tAsA, (None, None, 0, None)), 0, 2)
                        copy_A = partial(cute.copy, tiled_copy_A)
                    # TMA load B partition_S/D
                    b_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape
                    )
                    b_cta_crd = cluster_coord_mnk[0]
                    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
                    tBsB, tBgB_k = cpasync.tma_partition(
                        tma_atom_b,
                        b_cta_crd,
                        b_cta_layout,
                        cute.group_modes(sB, 0, 2),
                        cute.group_modes(gB_k, 0, 2),
                    )
                    copy_B = partial(cute.copy, tma_atom_b, mcast_mask=b_mcast_mask)
                    if const_expr(not self.gather_A):
                        ab_producer_state = self.load_AB(
                            ab_pipeline,
                            ab_producer_state,
                            copy_A,
                            tAgA_k,
                            tAsA,
                            copy_B,
                            tBgB_k,
                            tBsB,
                        )
                    else:
                        limit_m = (
                            mAIdx.shape[0]
                            if const_expr(cu_seqlens_m is None)
                            else cu_seqlens_m[batch_idx + 1] - cu_seqlens_m[batch_idx]
                        )
                        ab_producer_state = self.load_AB_gather_A(
                            ab_pipeline,
                            ab_producer_state,
                            thr_copy_A,
                            mA_mk,
                            tAsA,
                            gAIdx,
                            copy_B,
                            tBgB_k,
                            tBsB,
                            limit_A=(
                                limit_m - tile_coord_mnkl[0] * self.tile_shape_mnk[0],
                                mA_mk.shape[1],
                            ),
                        )
                    tile_scheduler.fetch_next_work(is_scheduler_warp=is_scheduler_warp)
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                if const_expr(self.pingpong):
                    # Need to write the tile_idx to smem for the next WG in the pingpong mode
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

        if warp_idx < self.ab_load_warp_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)
            is_tma_warp = Boolean(
                (not self.pingpong and warp_idx == 0)
                or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            if const_expr(varlen):
                # initialize tensormap for D
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_d,
                    tensormap_d_init_ptr,
                    is_manager_warp=is_tma_warp,
                )
            # //////////////////////////////////////////////////////////////////////////////
            #  Partition global tensor for TiledMMA_A/B/C
            # //////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group
            warp_group_thread_layout = cute.make_layout(
                self.mma_warp_groups if not self.pingpong else 1,
                stride=self.num_threads_per_warp_group,
            )
            thr_mma = tiled_mma.get_slice(
                warp_group_thread_layout(warp_group_idx if not self.pingpong else 0)
            )

            # //////////////////////////////////////////////////////////////////////////////
            #  Make fragments
            # //////////////////////////////////////////////////////////////////////////////
            tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))

            acc_shape = tiled_mma.partition_shape_C(cute.select(self.tile_shape_mnk, mode=[0, 1]))
            acc = cute.make_fragment(acc_shape, self.acc_dtype)
            if const_expr(self.fp8_slow_accum):
                acc_slow = cute.make_fragment(acc_shape, self.acc_dtype)
            else:
                acc_slow = None

            if const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # WG0 needs a start signal at the very beginning
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            epi_read_state = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_c_stage
            )
            epi_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.epi_c_stage
            )
            tile_scheduler = TileSchedulerCls()
            if const_expr(self.pingpong):
                if warp_idx >= 4:
                    # Advance 2nd Math WG to the next work tile for the startup
                    tile_scheduler.advance_to_next_work()
                    # Advance 2nd Math WG pipeline states to the end of 1st Math WG
                    ab_read_state.advance_iters(k_tile_cnt)
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
            work_tile = tile_scheduler.initial_work_tile_info()
            if const_expr(varlen):
                # wait tensormap initialization complete before update
                tensormap_manager.fence_tensormap_initialization()
            # batch index of last tile
            last_batch_idx = cutlass.Int32(-1)
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                if const_expr(varlen):
                    is_group_changed = batch_idx != last_batch_idx
                    last_batch_idx = batch_idx
                    if is_group_changed:
                        # construct tensor D based on real address, shape and stride information
                        tensormap_manager.update_tensormap_shape(
                            ((tensormap_d_ptr),),
                            is_manager_warp=is_tma_warp,
                            tensormap_smem_ptr=(tensormap_d_smem_ptr,),
                            shapes=(cu_seqlens_m[batch_idx + 1],),
                            orders=(0 if const_expr(self.d_layout.is_m_major_c()) else 1,),
                        )

                ab_read_state, tiled_mma = self.mma(
                    ab_pipeline,
                    ab_read_state,
                    tiled_mma,
                    tCrA,
                    tCrB,
                    acc,
                    acc_slow,
                    k_tile_cnt,
                    warp_group_idx,
                )
                if const_expr(self.pingpong):
                    # Update starting mainloop pipeline state for the next tile
                    ab_read_state.advance_iters(k_tile_cnt)

                # /////////////////////////////////////////////////////////////////////////////
                #  EPILOGUE
                # /////////////////////////////////////////////////////////////////////////////
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")

                epilogue_barrier = pipeline.NamedBarrier(
                    barrier_id=int(NamedBarrierGemm.Epilogue), num_threads=self.num_epi_threads
                )

                if const_expr(varlen):
                    # ensure the update to tensormap has completed before using it
                    if is_group_changed:
                        if is_tma_warp:
                            tensormap_manager.fence_tensormap_update(tensormap_d_ptr)
                    tma_desc_ptr = tensormap_manager.get_tensormap_ptr(
                        tensormap_d_ptr,
                        cute.AddressSpace.generic,
                    )
                else:
                    tma_desc_ptr = None

                bSG_sD, bSG_gD = self.epilog_gmem_copy_and_partition(
                    tma_atom_d,
                    mD_mnl,
                    self.epi_tile,
                    sD,
                    tile_coord_mnkl,
                    cu_seqlens_m,
                )
                copy_D = partial(cute.copy, tma_atom_d, tma_desc_ptr=tma_desc_ptr)
                if const_expr(has_C):
                    bGS_sC, bGS_gC = self.epilog_gmem_copy_and_partition(
                        tma_atom_c,
                        mC_mnl,
                        self.epi_tile,
                        sC,
                        tile_coord_mnkl,
                        cu_seqlens_m,
                    )
                    copy_C = partial(cute.copy, tma_atom_c)
                    epi_load_g2s = partial(self.epi_load_g2s, epi_pipeline, copy_C, bGS_gC, bGS_sC)
                else:
                    epi_load_g2s = None

                tiled_copy_r2s, tRS_rAcc, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                    tiled_mma, self.d_layout, self.d_dtype, acc, sD, tidx
                )
                if const_expr(mC_mnl is not None):
                    tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                        tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
                    )
                else:
                    tiled_copy_s2r, tSR_sC, tRS_rC, tSR_rC = None, None, None, None

                # Wait for all warp groups in the thread block to finish, because smem for tensor
                # A in the mainloop is reused in the epilogue if not persistent.
                if const_expr(not self.is_persistent):
                    epilogue_barrier.arrive_and_wait()

                self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)

                epi_read_state, epi_producer_state = self.epilogue(
                    epilogue_params,
                    epi_smem_tensors,
                    epi_pipeline,
                    epi_read_state,
                    epi_producer_state,
                    tRS_rAcc,
                    tRS_rD,
                    tRS_rC,
                    tiled_copy_r2s,
                    tRS_sD,
                    tiled_copy_s2r,
                    tSR_rC,
                    tSR_sC,
                    copy_D,
                    bSG_sD,
                    bSG_gD,
                    epi_load_g2s,
                    tile_coord_mnkl,
                    cu_seqlens_m,
                    epilogue_barrier,
                    tile_scheduler,
                    tidx,
                    is_tma_warp,
                )

                if const_expr(self.pingpong):
                    # Update starting load/store pipeline states for the next tile
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    # With pingpong, 2 WGs write two different output tiles to the same smem,
                    # so we have to make sure the smem content is done reading before signaling
                    # the next WG's epilogue.
                    if is_tma_warp:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")

                tile_scheduler.advance_to_next_work(
                    advance_count=1 if not self.pingpong else self.mma_warp_groups
                )
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            if const_expr(not self.pingpong):
                if warp_idx == 0:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def load_AB(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        copy_A: Callable,
        tAgA: cute.Tensor,
        tAsA: cute.Tensor,
        copy_B: Callable,
        tBgB: cute.Tensor,
        tBsB: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        k_tile_cnt = cute.size(tAgA, mode=[1])
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # /////////////////////////////////////////////////////////////////////////
        # TMA load
        # /////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            # Wait for A/B buffers to be empty before loading into them
            # Also sets the transaction barrier for the A/B buffers
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            copy_A(
                tAgA[None, k_tile],
                tAsA[None, ab_producer_state.index],
                tma_bar_ptr=tma_bar_ptr,
            )
            copy_B(
                tBgB[None, k_tile],
                tBsB[None, ab_producer_state.index],
                tma_bar_ptr=tma_bar_ptr,
            )
            # Mainloop pipeline's producer commit is a NOP
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
            peek_ab_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        return ab_producer_state

    @cute.jit
    def load_AB_gather_A(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        thr_copy_A: cute.core.ThrCopy,
        mA: cute.Tensor,
        tAsA: cute.Tensor,
        gAIdx: cute.Tensor,
        copy_B: Callable,
        tBgB: cute.Tensor,
        tBsB: cute.Tensor,
        limit_A: Tuple[Int32, Int32],
    ) -> cutlass.pipeline.PipelineState:
        # (atom_v, CPY_M, 1, RestK)
        limit_m, limit_k = limit_A
        limit_m = min(limit_m, self.tile_shape_mnk[0])  # To avoid writing beyond smem limit
        cA = cute.make_identity_tensor(cute.select(self.tile_shape_mnk, [0, 2]))
        tAcA = thr_copy_A.partition_S(cA)
        t0AcA = thr_copy_A.get_slice(0).partition_S(cA)
        # Instead of comparing tAcA to limit_m, we instead compare t0AcA to limit_m - tAcA[0][0]
        # since we know that tAcA[m][0] = t0AcA[m][0] + tAcA[0][0].
        # This is so that when we do the comparison, t0AcA is known at compile time.
        limit_m = limit_m - tAcA[0][0]
        # Read indices for A
        rows_per_thread = const_expr(cute.size(tAcA.shape, mode=[1]))
        m_idx = cute.make_fragment(rows_per_thread, Int32)
        for m in cutlass.range(rows_per_thread):
            row_idx = tAcA[0, m, 0][0]
            if t0AcA[0, m, 0][0] < limit_m:
                m_idx[m] = gAIdx[row_idx]
            else:
                m_idx[m] = -1
        elems_per_load = cute.size(tAsA.shape[0][0])
        # (m, (bK, RestK))
        mA_k = cute.logical_divide(mA, (None, self.tile_shape_mnk[2]))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        k_tile_cnt = cute.size(tBgB, mode=[1])
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # /////////////////////////////////////////////////////////////////////////
        # TMA load on B and cp.async on A
        # /////////////////////////////////////////////////////////////////////////
        copy_A = partial(cute.copy, thr_copy_A)
        for k_tile in cutlass.range(k_tile_cnt - 1, unroll=1):
            # Wait for A/B buffers to be empty before loading into them
            # Also sets the transaction barrier for the A/B buffers
            ab_pipeline.producer_acquire(
                ab_producer_state,
                peek_ab_empty_status,
                # A tiny bit faster to rotate the warp that does TMA
                is_tma_warp=warp_idx == self.ab_load_warp_id + (k_tile % self.num_ab_load_warps),
            )
            # A bit faster to load B first while we calculate the predicate for A
            if warp_idx == self.ab_load_warp_id + (k_tile % self.num_ab_load_warps):
                copy_B(
                    tBgB[None, k_tile],
                    tBsB[None, ab_producer_state.index],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
            # (m, bK)
            mA_cur = mA_k[None, (None, k_tile)]
            for m in cutlass.range_constexpr(tAcA.shape[1]):
                # (elems_per_load, thread_per_row)
                mA_row = cute.tiled_divide(mA_cur[m_idx[m], None], (elems_per_load,))
                if t0AcA[0, m, 0][0] < limit_m:
                    # There's only 1 load per row
                    assert cute.size(tAcA.shape, mode=[2]) == 1
                    ki = tAcA[0, 0, 0][1] // elems_per_load
                    copy_A(mA_row[None, ki], tAsA[(None, m), ab_producer_state.index])
            # This tells mbarrier to track the completion of cp.async
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
            peek_ab_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # bound checking in the K dimension on the last k_tile
        if 0 < k_tile_cnt:
            k_tile = k_tile_cnt - 1
            ab_pipeline.producer_acquire(
                ab_producer_state,
                peek_ab_empty_status,
                is_tma_warp=warp_idx == self.ab_load_warp_id + (k_tile % self.num_ab_load_warps),
            )
            if warp_idx == self.ab_load_warp_id + (k_tile % self.num_ab_load_warps):
                copy_B(
                    tBgB[None, k_tile],
                    tBsB[None, ab_producer_state.index],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
            assert tAcA.shape[2] == 1  # there's only 1 load along the K dimension
            tApA = cute.make_fragment(1, Boolean)
            tApA[0] = tAcA[0, 0, 0][1] < limit_k
            # (m, bK)
            mA_cur = mA_k[None, (None, k_tile)]
            for m in cutlass.range_constexpr(tAcA.shape[1]):
                # (elems_per_load, thread_per_row)
                mA_row = cute.tiled_divide(mA_cur[m_idx[m], None], (elems_per_load,))
                if t0AcA[0, m, 0][0] < limit_m:
                    # There's only 1 load per row
                    assert cute.size(tAcA.shape, mode=[2]) == 1
                    ki = tAcA[0, 0, 0][1] // elems_per_load
                    # copy_A(mA_row[None, ki], tAsA[(None, m), ab_producer_state.index], pred=tApA)
                    # TODO
                    copy_A(mA_row[None, ki], tAsA[(None, m), ab_producer_state.index])
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
        return ab_producer_state

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        acc: cute.Tensor,
        acc_slow: Optional[cute.Tensor],
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
    ) -> Tuple[cutlass.pipeline.PipelineState, cute.TiledMma]:
        # /////////////////////////////////////////////////////////////////////////////
        #  Prologue MMAs
        # /////////////////////////////////////////////////////////////////////////////
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        if const_expr(self.pingpong):
            self.pingpong_barrier_sync(warp_group_idx, stage="mma")
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        # TODO: this is probably not correct if k_tile_cnt == 0
        for k_tile in cutlass.range(num_prologue_mma):
            # Wait for A/B buffer to be ready
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            warpgroup.fence()
            for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_blk_coord = (None, None, k_blk_idx, ab_read_state.index)
                cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        if const_expr(self.fp8_slow_accum):
            warpgroup.wait_group(0)
            acc_slow.store(acc.load())

        # /////////////////////////////////////////////////////////////////////////////
        #  MAINLOOP
        # /////////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            # Wait for TMA copies to complete
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            # WGMMA
            warpgroup.fence()
            if const_expr(self.fp8_slow_accum):
                tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
            for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_blk_coord = (None, None, k_blk_idx, ab_read_state.index)
                cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
            if const_expr(not self.fp8_slow_accum):
                warpgroup.wait_group(k_pipe_mmas)
            else:
                warpgroup.wait_group(0)
                acc_slow.store(acc_slow.load() + acc.load())
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        if const_expr(self.pingpong):
            # Cue for next WG's MMA to start
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        if const_expr(not self.fp8_slow_accum):
            # fp8_slow_accum would already called wait_group(0) inside the loop
            warpgroup.wait_group(0)
        for k_tile in cutlass.range(k_pipe_mmas, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        if const_expr(self.fp8_slow_accum):
            acc.store(acc_slow.load())
        # If we don't return the tiled_mma, we get compiler error
        # "operand #0 does not dominate this use"
        return ab_read_state, tiled_mma

    @cute.jit
    def epilogue(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        epi_pipeline: cutlass.pipeline.PipelineAsync,
        epi_read_state: cutlass.pipeline.PipelineState,
        epi_producer_state: cutlass.pipeline.PipelineState,
        tRS_rAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor],
        tiled_copy_r2s: cute.core.ThrCopy,
        tRS_sD: cute.Tensor,
        tiled_copy_s2r: Optional[cute.core.ThrCopy],
        tSR_rC: Optional[cute.Tensor],
        tSR_sC: Optional[cute.Tensor],
        copy_D: Callable,
        bSG_sD: cute.Tensor,
        bSG_gD: cute.Tensor,
        epi_load_g2s: Optional[Callable],
        tile_coord_mnkl: cute.Coord,
        cu_seqlens_m: Optional[cute.Tensor],
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler,
        tidx: Int32,
        is_tma_warp: Boolean,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        has_C = const_expr(tRS_rC is not None)
        # We iterate over epi tiles in the N dimension first before the M dimension
        epi_tile_layout = cute.make_layout(bSG_gD.shape[1], stride=(bSG_gD.shape[1][1], 1))
        epi_tile_num = cute.size(bSG_gD.shape[1])
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        if const_expr(has_C):
            for epi_idx in cutlass.range(min(epi_tile_num, self.epi_c_stage), unroll=1):
                epi_producer_state = epi_load_g2s(epi_producer_state, epi_idx, is_tma_warp)

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # Copy from acc to D registers
            for epi_v in cutlass.range_constexpr(cute.size(tRS_rD)):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * cute.size(tRS_rD) + epi_v]
            if const_expr(has_C):
                epi_pipeline.consumer_wait(epi_read_state)
                cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC)
                # Fence to make sure shared memory read is visible to TMA load
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    epi_pipeline.consumer_release(epi_read_state)
                epi_read_state.advance()
            if const_expr(has_C and epi_idx + self.epi_c_stage < epi_tile_num):
                epi_producer_state = epi_load_g2s(
                    epi_producer_state, epi_idx + self.epi_c_stage, is_tma_warp
                )
            tRS_rEpi = self.epi_visit_acc_subtile(params, tRS_rD, tRS_rC)
            # Type conversion
            tRS_rD_out = cute.make_fragment_like(tRS_rD, self.d_dtype)
            tRS_rD_out.store(tRS_rD.load().to(self.d_dtype))
            # Copy from D registers to shared memory
            epi_buffer = (num_prev_subtiles + epi_idx) % cute.size(tRS_sD, mode=[3])
            cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[None, None, None, epi_buffer])
            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            epilogue_barrier.arrive_and_wait()
            # Get the global memory coordinate for the current epi tile
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from shared memory to global memory
            if is_tma_warp:
                copy_D(bSG_sD[None, epi_buffer], bSG_gD[None, gmem_coord])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(self.epi_stage - 1, read=True)
            epilogue_barrier.arrive_and_wait()

        return epi_read_state, epi_producer_state

    @cute.jit
    def epi_load_g2s(
        self,
        epi_pipeline: cutlass.pipeline.PipelineAsync,
        copy_C: Callable,
        bGS_gC: cute.Tensor,
        bGS_sC: cute.Tensor,
        epi_producer_state: cutlass.pipeline.PipelineState,
        epi_idx: Int32,
        should_load: Boolean,
    ) -> cutlass.pipeline.PipelineState:
        # We iterate over epi tiles in the N dimension first before the M dimension
        epi_tile_layout = cute.make_layout(bGS_gC.shape[1], stride=(bGS_gC.shape[1][1], 1))
        if should_load:
            epi_pipeline.producer_acquire(epi_producer_state)
            # Get the global memory coordinate for the current epi tile
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            copy_C(
                bGS_gC[None, gmem_coord],
                bGS_sC[None, epi_producer_state.index],
                tma_bar_ptr=epi_pipeline.producer_get_barrier(epi_producer_state),
            )
            # Epi pipeline's producer commit is a NOP
            epi_pipeline.producer_commit(epi_producer_state)
        epi_producer_state.advance()
        return epi_producer_state

    def epi_visit_acc_subtile(
        self,
        params: EpilogueParams,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        if const_expr(tRS_rC is not None):
            tRS_rD.store(tRS_rD.load() + tRS_rC.load().to(tRS_rD.element_type))
        return None

    @cute.jit
    def epi_visit_acc(
        self,
        params: EpilogueParams,
        acc: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tile_coord_mnkl: cute.Coord,
        tidx: Int32,
    ) -> None:
        pass

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        return GemmSm90.EpilogueParams()

    @staticmethod
    def epi_smem_bytes_per_stage(
        args: EpilogueArguments,
        tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
    ) -> int:
        return 0

    def epi_get_smem_struct(self, params: EpilogueParams):
        return cute.struct.MemRange[cutlass.Int32, 0]  # Dummy struct

    def epi_get_smem_tensors(self, params: EpilogueParams, storage) -> Tuple[cute.Tensor, ...]:
        return tuple()

    def pingpong_barrier_sync(self, warp_group_idx: Int32, stage: str):
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
        cute.arch.barrier(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def pingpong_barrier_arrive(self, warp_group_idx: Int32, stage: str):
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
        cute.arch.barrier_arrive(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def epilog_smem_copy_atom(self, tiled_mma: cute.TiledMma) -> cute.TiledCopy:
        copy_atom_C = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(
                self.d_layout.is_m_major_c(),
                num_matrices=4 if self.epi_tile[1] % 16 == 0 else 2,
            ),
            cutlass.Float16,  # this is just to get the right source layout
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        return tiled_copy_C_atom

    def epilog_smem_store_and_partition(
        self,
        tiled_mma: cute.TiledMma,
        d_layout: cutlass.utils.LayoutEnum,
        dtype: Type[cutlass.Numeric],
        acc: cute.Tensor,
        sD: cute.Tensor,
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        # Doesn't work with tile_N % 8 == 0 but tile_n % 16 != since this always
        # get st.matrix with num_matrices=4
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            d_layout, elem_ty_d=dtype, elem_ty_acc=self.acc_dtype
        )
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        # (R2S, R2S_M, R2S_N)
        tRS_rAcc = tiled_copy_r2s.retile(acc)
        tRS_rD_layout = cute.make_layout(thr_copy_r2s.partition_S(sD).shape[:3])
        tRS_rD = cute.make_fragment(tRS_rD_layout, self.acc_dtype)
        return tiled_copy_r2s, tRS_rAcc, tRS_rD, tRS_sD

    def epilog_smem_load_and_partition(
        self,
        tiled_mma: cute.TiledMma,
        c_layout: cutlass.utils.LayoutEnum,
        dtype: Type[cutlass.Numeric],
        sC: cute.Tensor,
        tRS_rD_layout: cutlass.Layout,
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        copy_atom_s2r = utils.sm90_get_smem_load_op(c_layout, dtype)
        tiled_copy_s2r = cute.make_tiled_copy_S(copy_atom_s2r, tiled_copy_C_atom)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        tSR_sC = thr_copy_s2r.partition_S(sC)
        tRS_rC = cute.make_fragment(tRS_rD_layout, dtype)
        tSR_rC = thr_copy_s2r.retile(tRS_rC)
        return tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC

    def epilog_gmem_copy_and_partition(
        self,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        mD_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
        cu_seqlens_m: Optional[cute.Tensor] = None,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        batch_idx = tile_coord_mnkl[3]
        if const_expr(cu_seqlens_m is not None):
            mD_mn = cute.domain_offset((cu_seqlens_m[batch_idx], 0), mD_mnl)
        else:
            mD_mn = mD_mnl[None, None, batch_idx]
        # (bM, bN)
        gD = cute.local_tile(mD_mn, cute.select(self.tile_shape_mnk, [0, 1]), tile_coord_mnkl[:2])
        tDgD_for_tma_partition = cute.zipped_divide(gD, epi_tile)
        bSG_sD, bSG_gD = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sD, 0, 2),
            tDgD_for_tma_partition,
        )
        return bSG_sD, bSG_gD

    @classmethod
    def _compute_stages(
        cls,
        tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        c_dtype: Optional[Type[cutlass.Numeric]],
        epilogue_args: EpilogueArguments,
        smem_capacity: int,
        occupancy: int,
        overlap_sD_sA: bool,
    ) -> Tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: Tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: Tuple[int, int]
        """

        epi_stage = 2
        if overlap_sD_sA:
            epi_bytes = 0
        else:
            d_bytes_per_stage = cute.size(epi_tile) * d_dtype.width // 8
            epi_bytes_per_stage = d_bytes_per_stage + cls.epi_smem_bytes_per_stage(
                epilogue_args, tile_shape_mnk, epi_tile
            )
            epi_bytes = epi_bytes_per_stage * epi_stage
        epi_c_stage = 0 if c_dtype is None else 2
        if c_dtype is not None:
            epi_bytes += cute.size(epi_tile) * c_dtype.width // 8 * epi_c_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8 + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        remaining_bytes = (
            (smem_capacity - occupancy * 1024) // occupancy - mbar_helpers_bytes - epi_bytes
        )
        ab_stage = remaining_bytes // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        if not overlap_sD_sA:
            epi_stage += (remaining_bytes - ab_bytes_per_stage * ab_stage) // epi_bytes_per_stage
        return ab_stage, epi_stage, epi_c_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk: Tuple[int, int, int],
        atom_layout_mnk: Tuple[int, int, int],
        element_type: Type[cutlass.Numeric],
        epi_tile_override: Tuple[int, int] | None = None,
    ) -> Tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if tile_shape_mnk[0] % 128 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(tile_shape_mnk, mode=[1]))
        elif tile_shape_mnk[0] % 192 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(192, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(tile_shape_mnk, mode=[1]))
        else:
            # In the case of tile shape 128 x N but atom_layout 1 x 2, we need to set
            # epi_tile_m = 64. If epi_tile_m = 128, the epilogue would iterate along the
            # M dimension first, then move to the N dimension. But the accumulator in registers
            # iterate along the N dimension first, then move to the M dimension.
            # We could change the epilogue to accommodate this,
            # but it's easier to just set epi_tile_m = 64.
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = math.gcd(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(n_perf, cute.size(tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        a_layout: cutlass.utils.LayoutEnum,
        b_dtype: Type[cutlass.Numeric],
        b_layout: cutlass.utils.LayoutEnum,
        ab_stage: int,
        d_dtype: Type[cutlass.Numeric],
        d_layout: cutlass.utils.LayoutEnum,
        epi_stage: int,
        c_dtype: Optional[Type[cutlass.Numeric]],
        c_layout: Optional[cutlass.utils.LayoutEnum],
        epi_c_stage: int,
    ) -> Tuple[
        cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout, Optional[cute.ComposedLayout]
    ]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: cutlass.utils.LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: cutlass.utils.LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param d_dtype: Data type for output matrix C
        :type d_dtype: type[cutlass.Numeric]
        :param d_layout: Layout enum for the output matrix C
        :type d_layout: cutlass.utils.LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.K
        b_is_k_major = b_layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.K
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        d_smem_shape = epi_tile
        d_major_mode_size = epi_tile[1] if d_layout.is_n_major_c() else epi_tile[0]
        d_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(d_layout, d_dtype, d_major_mode_size),
            d_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            d_smem_layout_atom,
            cute.append(d_smem_shape, epi_stage),
            order=(1, 0, 2) if d_layout.is_m_major_c() else (0, 1, 2),
        )

        if c_dtype is not None:
            assert c_layout is not None
            c_smem_shape = epi_tile
            c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
            c_smem_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(c_layout, c_dtype, c_major_mode_size),
                c_dtype,
            )
            epi_c_smem_layout_staged = cute.tile_to_shape(
                c_smem_layout_atom,
                cute.append(c_smem_shape, epi_c_stage),
                order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
            )
        else:
            epi_c_smem_layout_staged = None

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
            epi_c_smem_layout_staged,
        )

    @staticmethod
    def _make_tma_epi_atoms_and_tensors(
        tensor_d: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: Tuple[int, int],
        store_or_load: str,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for storing D or loading C.

        :param tensor_d: Output tensor D
        :type tensor_d: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        assert store_or_load in ["load", "store"]
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        d_cta_v_layout = cute.composition(cute.make_identity_layout(tensor_d.shape), epi_tile)
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if store_or_load == "load"
            else cpasync.CopyBulkTensorTileS2GOp()
        )
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            op, tensor_d, epi_smem_layout, d_cta_v_layout
        )
        return tma_atom_d, tma_tensor_d

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: Tuple[int, int],
        mcast_dim: int,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    def _make_gmem_tiled_copy_A(self, dtype, major_mode, num_threads, copy_bits=128):
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=copy_bits,
        )
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.tile_shape_mnk[2]) // copy_elems
        # thread layout for copy
        thread_layout = cute.make_layout(
            (num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != cutlass.utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.tile_shape_mnk[0]) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        # Value layout for copy
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == cutlass.utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_async_copy, thread_layout, value_layout)

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if a_dtype not in {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # tested b_dtype
        if b_dtype not in {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        if acc_dtype not in {cutlass.Float32, cutlass.Float16}:
            is_valid = False
        # tested d_dtype
        if d_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        # make sure a_dtype.width == b_dtype.width (i.e, Float8E4M3FN or Float8E5M2)
        if a_dtype.width != b_dtype.width:
            is_valid = False

        # for Float8 types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (b_dtype.width == 8 and b_major != "k"):
            is_valid = False
        return is_valid
