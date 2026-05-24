# Copyright (c) 2025-2026, QuACK team.
# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py
# SM120-style GEMM using warp-level MMA (MmaF16BF16Op) + ldmatrix.
# Unlike SM90 WGMMA (which reads A/B from SMEM directly), warp-level MMA
# requires explicit SMEM→RMEM copies via ldmatrix before each MMA instruction.

# This is a work in progress and not very optimized.

import math
from typing import Tuple, Type, Callable, Optional, Union
from functools import partial

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warp
from cutlass import Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum
from quack import _sm120_nvfp4_utils as _sm120

from quack.varlen_utils import VarlenManager
from quack.sm120_pipeline import PipelineTmaWarpMma
from quack.tile_scheduler import (
    PersistenceMode,
    RasterOrderOption,
    TileScheduler,
    TileSchedulerArguments,
)
from quack.pipeline import make_pipeline_state
from quack import copy_utils
from quack.gemm_sm90 import GemmSm90, NamedBarrierGemm
from quack import sm80_utils


class GemmSm120(GemmSm90):
    """SM120-style GEMM using warp-level MMA instead of WGMMA.

    Key differences from SM90:
    - Uses MmaF16BF16Op (warp-level, 32 threads) instead of WGMMA (warp-group, 128 threads)
    - Requires explicit SMEM→RMEM copy via ldmatrix before MMA
    - Thread config: num_mma_warps regular warps + 1 DMA warp
    - Pingpong: 2 warp groups of (2,2,1), each processing alternating tiles
    - No fp8 support (warp-level MMA only supports fp16/bf16)
    """

    arch = 120

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Tuple[int, int] | Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = True,
        gather_A: bool = False,
        concat_layout: tuple | None = None,
        use_pdl: bool = True,
        sf_vec_size: Optional[int] = None,
        sf_dtype: Optional[Type[cutlass.Numeric]] = None,
        sm120_nvfp4_path: str = "validated",
    ):
        # Don't call super().__init__ — we set up our own config
        self.acc_dtype = acc_dtype
        self.sf_vec_size = sf_vec_size
        self.sf_dtype = sf_dtype
        self.blockscaled = sf_vec_size is not None
        if sm120_nvfp4_path not in ("validated", "fast"):
            raise ValueError("SM120 NVFP4 path must be 'validated' or 'fast'")
        if not self.blockscaled and sm120_nvfp4_path != "validated":
            raise ValueError("SM120 NVFP4 fast path requires blockscaled NVFP4")
        self.sm120_nvfp4_path = sm120_nvfp4_path
        if self.blockscaled:
            self._validate_blockscaled_nvfp4_config(
                acc_dtype,
                a_dtype,
                tile_shape_mnk,
                cluster_shape_mnk,
                sf_vec_size,
                sf_dtype,
                pingpong,
                is_persistent,
                gather_A,
            )
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        self.use_clc_persistence = False
        self.use_pdl = use_pdl
        self.fp8_slow_accum = False
        self.gather_A = gather_A
        self.concat_layout = concat_layout or ()
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"
        if gather_A:
            assert cluster_shape_mnk[1] == 1

        self.cluster_shape_mnk = cluster_shape_mnk
        assert len(tile_shape_mnk) in [2, 3], "CTA tile shape must be (M, N) or (M, N, K)"
        # K dimension: if user provides 3 values, use their K; otherwise default in _setup_tiled_mma.
        self.cta_tile_shape_mnk = (
            tuple(tile_shape_mnk) if len(tile_shape_mnk) == 3 else (*tile_shape_mnk, 0)
        )
        tile_M, tile_N = self.cta_tile_shape_mnk[:2]

        self.mma_inst_mnk = (16, 8, 16)
        if self.blockscaled:
            if self.pingpong:
                self.atom_layout_mnk = (2, 2, 1)
                self.num_mma_warps = 8
            else:
                self.atom_layout_mnk = (
                    (4, 2, 1)
                    if tile_M == 128 and tile_N == 128
                    else (tile_M // self.mma_inst_mnk[0], 1, 1)
                )
                self.num_mma_warps = tile_M // self.mma_inst_mnk[0]
        else:
            # Pingpong: 2 warp groups each with (2,2,1) atom layout
            # Non-pingpong: 1 group of 8 warps with (4,2,1) atom layout
            self.atom_layout_mnk = (4, 2, 1) if not self.pingpong else (2, 2, 1)
            # num_mma_warps = total warps doing MMA (both warp groups in pingpong)
            self.num_mma_warps = math.prod(self.atom_layout_mnk) * (1 if not self.pingpong else 2)
        # Keep the warp-group-sized thread count used by SM120 scheduling helpers.
        self.num_threads_per_warp_group = 128
        assert self.num_mma_warps % 4 == 0
        self.mma_warp_groups = self.num_mma_warps // 4
        if self.pingpong:
            assert self.mma_warp_groups == 2
        direct_128_pingpong = (
            self.blockscaled and self.pingpong and self.cta_tile_shape_mnk[:2] == (128, 128)
        )
        self.blockscaled_pingpong_split_tiles = direct_128_pingpong
        self.blockscaled_pingpong_full_tma_pipeline = False
        self.blockscaled_pingpong_elected_tma = False
        # threads_per_cta must be a multiple of 128 (warp group size) so that
        # the DMA warp's setmaxnreg.dec.sync has a complete warp group to sync with.
        self.threads_per_cta = (self.mma_warp_groups + 1) * self.num_threads_per_warp_group

        self.num_mcast_ctas_a = cluster_shape_mnk[1]
        if gather_A:
            assert self.num_mcast_ctas_a == 1
        self.num_mcast_ctas_b = cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.occupancy = 1
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}")

        # In pingpong, only 1 warp group (4 warps) participates in epilogue at a time
        split_epi_by_warpgroup = self.pingpong and (
            not self.blockscaled
            or self.blockscaled_pingpong_split_tiles
            or self.blockscaled_pingpong_full_tma_pipeline
            or self.blockscaled_pingpong_elected_tma
        )
        self.num_epi_warps = (self.mma_warp_groups if not split_epi_by_warpgroup else 1) * 4
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.Epilogue),
            num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )
        self.num_ab_load_warps = 1 if not self.gather_A else 4
        self.ab_load_warp_id = self.num_mma_warps
        self.mma_warp_id_start = 0
        if self.blockscaled and self.pingpong:
            self.ab_load_warp_id = 0
            self.mma_warp_id_start = 4
        self.mma_warp_id_end = self.mma_warp_id_start + self.num_mma_warps

        if not self.gather_A:
            self.num_regs_load = 40
            self.num_regs_mma = 232
        else:
            self.num_regs_load = 56
            self.num_regs_mma = 224

        self.ab_stage = None
        self.epi_stage = None
        self.epi_m_major = True
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024
        self.max_active_clusters = 1
        self.direct_producer_unroll = 1
        self.direct_consumer_unroll = 1
        self.direct_consumer_barrier = True
        self.direct_consumer_fence = True
        self.direct_consumer_warp_sync = False
        self.direct_release_before_sync = False
        self.direct_kblock_pipeline = False
        self.direct_kblock_barrier = False
        self.direct_ab_tma_layout = "packed"
        self.direct_unpack_shift = False
        direct_128_default = self.blockscaled and self.cta_tile_shape_mnk[:2] == (128, 128)
        self.direct_pre_mma_warp_sync = False
        ab_stage_override = 0
        self.direct_elected_tma = direct_128_default and self.blockscaled_pingpong_elected_tma
        self.direct_setmaxregister = True
        self.direct_cute_dsl_helpers = False
        self.direct_global_store = sm120_nvfp4_path == "validated"
        self.direct_global_store_probe = False
        self.direct_tile_scheduler = direct_128_default
        self.direct_cute_static_scheduler = sm120_nvfp4_path == "validated"
        self.direct_pipelined_consumer = direct_128_default
        self.direct_split_tma_pipelines = direct_128_default and not self.direct_elected_tma
        self.direct_full_tma_pipeline = False
        self.direct_single_tma_pipeline = False
        self.direct_join_split_tma_barrier = direct_128_default and self.direct_split_tma_pipelines
        if self.direct_split_tma_pipelines:
            if not self.direct_tile_scheduler or not self.direct_pipelined_consumer:
                raise ValueError("SM120 NVFP4 split TMA requires scheduler and consumer pipelines")
            self.num_ab_load_warps = 3
        elif self.direct_join_split_tma_barrier:
            raise ValueError("SM120 NVFP4 joined split barrier requires split TMA pipelines")
        if (
            self.blockscaled
            and self.pingpong
            and not (
                self.direct_split_tma_pipelines
                or self.direct_full_tma_pipeline
                or self.direct_single_tma_pipeline
            )
        ):
            raise ValueError("SM120 NVFP4 blockscaled pingpong requires split TMA pipelines")
        self.direct_skip_split_tma_tail = False
        if self.direct_skip_split_tma_tail and not self.direct_split_tma_pipelines:
            raise ValueError("SM120 NVFP4 skip split TMA tail requires split TMA pipelines")
        self.direct_scheduler_local_tma = False
        self.direct_sched_exclude_producer = False
        self.direct_skip_scheduler_tail = False
        self.direct_pingpong_barriers = True
        self.direct_try_wait_before_pingpong_barrier = (
            direct_128_default
            and self.direct_split_tma_pipelines
            and self.blockscaled_pingpong_split_tiles
        )
        self.direct_pingpong_split_tiles = (
            self.blockscaled_pingpong_split_tiles and not self.direct_elected_tma
        )
        if self.pingpong and self.direct_pingpong_split_tiles and not self.direct_pingpong_barriers:
            raise ValueError("SM120 NVFP4 pingpong split path requires pingpong barriers")
        if self.direct_try_wait_before_pingpong_barrier and not (
            self.direct_split_tma_pipelines
            and self.direct_pingpong_split_tiles
            and self.direct_pingpong_barriers
        ):
            raise ValueError("SM120 NVFP4 try-wait path requires split-TMA pingpong barriers")
        self.direct_scale_prefetch_first = False
        self.direct_scale_smem_format = "interleaved"
        direct_bgroup_pipeline_requested = False
        self.direct_epi_barrier_trim = direct_128_default
        if self.direct_epi_barrier_trim and not (
            self.direct_split_tma_pipelines or self.direct_full_tma_pipeline
        ):
            raise ValueError("SM120 NVFP4 epilogue barrier trim requires split/full TMA")
        self.direct_mma_n_major = direct_128_default and self.pingpong
        self.direct_bgroup_pipeline = (
            direct_128_default
            and direct_bgroup_pipeline_requested
            and self.direct_full_tma_pipeline
            and self.direct_ab_tma_layout == "packed"
            and self.direct_mma_n_major
        )
        self.direct_fragment_contract = "shape"
        self.direct_tma_scale_first = False
        self.direct_tma_prefetch = False
        self.direct_skip_tma_acquire = False
        self.direct_tma_policy = "zero"
        self.direct_epi_stsm_matrices = 2
        if self.direct_epi_stsm_matrices != 2:
            raise ValueError("SM120 NVFP4 epilogue STSM matrices must be 2")
        self.direct_epi_tile_m = 0
        if self.direct_epi_tile_m not in (0, 64, 128):
            raise ValueError("SM120 NVFP4 epilogue tile M must be 0, 64, or 128")
        self.direct_epi_tile_n = 0
        if self.direct_epi_tile_n not in (0, 32, 64, 128):
            raise ValueError("SM120 NVFP4 epilogue tile N must be 0, 32, 64, or 128")
        both_epi_dims_split = self.direct_epi_tile_m not in (
            0,
            128,
        ) and self.direct_epi_tile_n not in (0, 128)
        if both_epi_dims_split and (self.direct_epi_tile_m, self.direct_epi_tile_n) != (
            64,
            32,
        ):
            raise ValueError("Only the 79a-style SM120 NVFP4 (64, 32) epilogue split is supported")
        self.direct_epi_tma_rank3 = direct_128_default and self.pingpong

    @staticmethod
    def _validate_blockscaled_nvfp4_config(
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Tuple[int, int] | Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        sf_vec_size: Optional[int],
        sf_dtype: Optional[Type[cutlass.Numeric]],
        pingpong: bool,
        is_persistent: bool,
        gather_A: bool,
    ) -> None:
        if acc_dtype is not cutlass.Float32:
            raise ValueError("SM120 NVFP4 blockscaled requires Float32 accumulation")
        if a_dtype is not cutlass.Float4E2M1FN:
            raise ValueError("SM120 NVFP4 blockscaled requires Float4E2M1FN A/B operands")
        if sf_dtype is not cutlass.Float8E4M3FN:
            raise ValueError("SM120 NVFP4 blockscaled requires Float8E4M3FN scales")
        if sf_vec_size != 16:
            raise ValueError("SM120 NVFP4 blockscaled requires sf_vec_size=16")
        if tuple(tile_shape_mnk) != (128, 128, 128):
            raise ValueError("SM120 NVFP4 blockscaled supports CTA tile (128,128,128)")
        if tuple(cluster_shape_mnk) != (1, 1, 1):
            raise ValueError("SM120 NVFP4 blockscaled initially supports cluster (1,1,1)")
        if pingpong and (tuple(tile_shape_mnk) != (128, 128, 128) or not is_persistent):
            raise ValueError(
                "SM120 NVFP4 blockscaled pingpong requires persistent (128,128,128) tiles"
            )
        if gather_A:
            raise ValueError("SM120 NVFP4 blockscaled does not support gather_A")

    @staticmethod
    def can_implement_blockscaled(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        d_dtype: Type[cutlass.Numeric],
        mma_tiler_mnk: Tuple[int, int] | Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        d_major: str,
    ) -> bool:
        tile_shape = tuple(mma_tiler_mnk) if len(mma_tiler_mnk) == 3 else (*mma_tiler_mnk, 128)
        return (
            ab_dtype is cutlass.Float4E2M1FN
            and sf_dtype is cutlass.Float8E4M3FN
            and sf_vec_size == 16
            and d_dtype is cutlass.BFloat16
            and tile_shape == (128, 128, 128)
            and cluster_shape_mn == (1, 1)
            and m % tile_shape[0] == 0
            and n % tile_shape[1] == 0
            and k % 128 == 0
            and l >= 1
            and a_major == "k"
            and b_major == "k"
            and d_major == "n"
        )

    def epi_smem_warp_shape_mnk(self):
        return self.atom_layout_mnk

    def _setup_tiled_mma(self):
        """Set up warp-level MMA (MmaF16BF16Op) and tile K dimension."""
        op = warp.MmaF16BF16Op(self.a_dtype, self.acc_dtype, self.mma_inst_mnk)
        tC = cute.make_layout(self.atom_layout_mnk)
        atom_m, atom_n, atom_k = self.atom_layout_mnk
        # We want each warp to have 16 consecutive elements in the N direction, for STSM
        # and for gated epilogue.
        permutation_n = cute.make_ordered_layout((self.mma_inst_mnk[1], atom_n, 2), order=(0, 2, 1))
        permutation_mnk = (
            atom_m * self.mma_inst_mnk[0],
            permutation_n,
            atom_k * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)
        tile_k = (
            self.cta_tile_shape_mnk[2]
            if self.cta_tile_shape_mnk[2] > 0
            else self.mma_inst_mnk[2] * 4
        )
        assert tile_k > 0, "CTA tile K must be positive"
        assert tile_k % self.mma_inst_mnk[2] == 0, (
            f"CTA tile K ({tile_k}) must be divisible by MMA instruction K ({self.mma_inst_mnk[2]})"
        )
        self.cta_tile_shape_mnk = (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], tile_k)

    # __call__, _setup_attributes, make_ab_pipeline, make_epi_store_pipeline,
    # epilogue are all inherited from GemmSm90.

    def make_sm120_single_warp_epi_store_pipeline(self):
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                cute.arch.WARP_SIZE,
            ),
        )

    def make_sched_pipeline(
        self, cluster_layout_mnk: cute.Layout, sched_pipeline_mbar_ptr: cute.Pointer, varlen_k: bool
    ):
        if not (self.blockscaled and (self.pingpong or self.direct_sched_exclude_producer)):
            return super().make_sched_pipeline(
                cluster_layout_mnk, sched_pipeline_mbar_ptr, varlen_k
            )

        # The inherited SM90 pingpong scheduler counts one MMA warpgroup when
        # varlen_k is false. This SM120 NVFP4 non-split path has both consumer
        # warpgroups waiting on the scheduler tile.
        sched_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_layout_mnk)
        sched_mma_warp_groups = (
            1
            if const_expr(self.direct_full_tma_pipeline)
            else (self.mma_warp_groups if not self.direct_pingpong_split_tiles else 1)
        )
        sched_producer_warps = (
            0 if const_expr(self.direct_sched_exclude_producer) else self.num_ab_load_warps
        )
        consumer_arrive_cnt = (sched_mma_warp_groups * 4 + sched_producer_warps) * cluster_size
        sched_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return pipeline.PipelineAsync.create(
            barrier_storage=sched_pipeline_mbar_ptr,
            num_stages=self.sched_stage,
            producer_group=sched_pipeline_producer_group,
            consumer_group=sched_pipeline_consumer_group,
            consumer_mask=None if const_expr(cluster_size == 1) else 0,
            defer_sync=True,
        )

    @cute.jit
    def blockscaled_call(
        self,
        gA_storage: cute.Tensor,
        gB_storage: cute.Tensor,
        mD: cute.Tensor,
        gSFA_storage: cute.Tensor,
        gSFB_storage: cute.Tensor,
        problem_m: cutlass.Constexpr[int],
        problem_n: cutlass.Constexpr[int],
        problem_k: cutlass.Constexpr[int],
        problem_l: cutlass.Constexpr[int],
        epilogue_args,
        stream: cuda.CUstream,
    ):
        self.a_dtype = cutlass.Float4E2M1FN
        self.b_dtype = cutlass.Float4E2M1FN
        self.d_dtype = mD.element_type
        self.d_layout = LayoutEnum.from_tensor(mD)
        if const_expr(self.d_dtype is not cutlass.BFloat16):
            raise TypeError("SM120 NVFP4 blockscaled output must be BFloat16")
        if const_expr(not self.d_layout.is_n_major_c()):
            raise ValueError("SM120 NVFP4 blockscaled output must be N-major")
        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)

        tile_extent_m, tile_extent_n, tile_extent_k = self.cta_tile_shape_mnk
        gA = cute.make_tensor(
            gA_storage.iterator,
            _sm120.make_mxf4nvf4_a_gmem_layout(problem_m, problem_k, problem_l),
        )
        gB = cute.make_tensor(
            gB_storage.iterator,
            _sm120.make_mxf4nvf4_b_gmem_layout(problem_n, problem_k, problem_l),
        )
        gSFA = cute.make_tensor(
            gSFA_storage.iterator,
            _sm120.make_mxf4nvf4_scale_interleaved_gmem_layout(problem_m, problem_k, problem_l),
        )
        gSFB = cute.make_tensor(
            gSFB_storage.iterator,
            _sm120.make_mxf4nvf4_scale_interleaved_gmem_layout(problem_n, problem_k, problem_l),
        )
        k_tile_count = problem_k // tile_extent_k
        m_tile_count = cute.ceil_div(problem_m, tile_extent_m)
        n_tile_count = cute.ceil_div(problem_n, tile_extent_n)
        if const_expr(tile_extent_m == 128 and tile_extent_n == 128):
            if const_expr(self.direct_ab_tma_layout == "unpack"):
                self.ab_stage = 2
            else:
                self.ab_stage = 4
            self.sched_stage = 3
        else:
            self.ab_stage = 2
            self.sched_stage = 1
        direct_128_default = tile_extent_m == 128 and tile_extent_n == 128
        delay_tma_store = direct_128_default and self.pingpong
        use_default_79a_epi_tile = (
            self.pingpong
            and direct_128_default
            and self.direct_epi_tile_m == 0
            and self.direct_epi_tile_n == 0
        )
        effective_epi_tile_m = (
            64 if const_expr(use_default_79a_epi_tile) else self.direct_epi_tile_m
        )
        effective_epi_tile_n = (
            32 if const_expr(use_default_79a_epi_tile) else self.direct_epi_tile_n
        )
        if const_expr(effective_epi_tile_n == 32 and effective_epi_tile_m == 0):
            effective_epi_tile_m = tile_extent_m
        if const_expr(effective_epi_tile_n == 64 and effective_epi_tile_m == 0):
            effective_epi_tile_m = tile_extent_m
        if const_expr(effective_epi_tile_m == 64 and effective_epi_tile_n == 0):
            effective_epi_tile_n = tile_extent_n
        use_split_epi_tile = (
            tile_extent_m == 128
            and tile_extent_n == 128
            and effective_epi_tile_m != 0
            and effective_epi_tile_n != 0
            and (effective_epi_tile_m != tile_extent_m or effective_epi_tile_n != tile_extent_n)
        )
        self.epi_tile = (
            (effective_epi_tile_m, effective_epi_tile_n)
            if const_expr(use_split_epi_tile)
            else (tile_extent_m, tile_extent_n)
        )
        self.direct_epi_m_tiles = (
            tile_extent_m // effective_epi_tile_m if const_expr(use_split_epi_tile) else 1
        )
        self.direct_epi_n_tiles = (
            tile_extent_n // effective_epi_tile_n if const_expr(use_split_epi_tile) else 1
        )
        self.direct_epi_tiles = self.direct_epi_m_tiles * self.direct_epi_n_tiles
        self.epi_tile_shape = cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile)
        self.direct_delay_tma_store = delay_tma_store
        if const_expr(self.direct_delay_tma_store and self.direct_epi_tiles < 2):
            raise ValueError("SM120 NVFP4 delayed TMA store requires a split epilogue tile")
        self.epi_stage = 2 if const_expr(self.direct_delay_tma_store) else 1
        self.direct_epi_tma_rank3 = direct_128_default and self.pingpong
        self.epi_c_stage = 0
        epi_smem_m, epi_smem_n = self.epi_tile
        self.epi_smem_layout_staged = _sm120.make_mxf4nvf4_epilogue_smem_layout(
            epi_tile=(epi_smem_m, epi_smem_n),
            num_stages=self.epi_stage,
        )
        if const_expr(tile_extent_m == 128 and tile_extent_n == 128):
            # Match CUTLASS 79a's 128x128 NVFP4 tiled-MMA layout.
            if const_expr(self.pingpong):
                self.tiled_mma = _sm120.make_mxf4nvf4_79a_tiled_mma()
            else:
                mma_op = warp.MmaMXF4NVF4Op(
                    cutlass.Float4E2M1FN,
                    cutlass.Float32,
                    cutlass.Float8E4M3FN,
                )
                self.tiled_mma = cute.make_tiled_mma(
                    mma_op,
                    atom_layout_mnk=cute.make_layout((4, 2, 1), stride=(1, 4, 0)),
                    permutation_mnk=(
                        128,
                        cute.make_layout((8, 2, 2), stride=(1, 16, 8)),
                        64,
                    ),
                )
        else:
            self.tiled_mma = _sm120.make_mxf4nvf4_tiled_mma(
                atom_layout_mnk=self.atom_layout_mnk,
            )

        if const_expr(self.direct_epi_tma_rank3):
            mD_tma = mD
            if const_expr(cute.size(mD, mode=[2]) == 1):
                mD_tma = cute.make_tensor(
                    mD.iterator,
                    cute.make_layout(
                        (mD.shape[0], mD.shape[1], 2),
                        stride=mD.layout.stride,
                    ),
                )
            epi_tma_smem_layout = self.epi_smem_layout_staged
            if const_expr(self.epi_stage != 1):
                epi_tma_smem_layout = cute.slice_(
                    self.epi_smem_layout_staged,
                    (None, None, 0),
                )
            tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mD_tma,
                epi_tma_smem_layout,
                self.epi_tile,
            )
        else:
            tma_atom_d, tma_tensor_d = _sm120.make_mxf4nvf4_epilogue_tma_store_atom(
                mD,
                self.epi_smem_layout_staged,
                epi_tile=self.epi_tile,
            )

        grid = (
            cute.ceil_div(cute.size(mD, mode=[0]), tile_extent_m),
            cute.ceil_div(cute.size(mD, mode=[1]), tile_extent_n),
            cute.size(mD, mode=[2]),
        )
        if const_expr(tile_extent_m == 128 and tile_extent_n == 128):
            (
                tma_atom_a,
                tma_tensor_a,
                tma_atom_b,
                tma_tensor_b,
                tma_atom_sfa,
                tma_tensor_sfa,
                tma_atom_sfb,
                tma_tensor_sfb,
            ) = _sm120.make_mxf4nvf4_native_tma_atoms_for_scheduler(
                gA,
                gB,
                gSFA,
                gSFB,
                tiled_mma=self.tiled_mma,
                ab_smem_format=self.direct_ab_tma_layout,
                scale_smem_format=self.direct_scale_smem_format,
            )
            persistence_mode = PersistenceMode.CLC if self.pingpong else PersistenceMode.STATIC
            if const_expr(self.direct_cute_static_scheduler):
                static_scheduler_swizzle_size = 1
                if const_expr(static_scheduler_swizzle_size < 1):
                    raise ValueError("SM120 NVFP4 static scheduler swizzle must be >= 1")
                tile_sched_params, static_scheduler_grid = (
                    _sm120.make_mxf4nvf4_static_tile_scheduler_params(
                        m=problem_m,
                        n=problem_n,
                        k=problem_k,
                        l_extent=problem_l,
                        max_active_clusters=self.max_active_clusters,
                        swizzle_size=static_scheduler_swizzle_size,
                    )
                )
                direct_grid = static_scheduler_grid
            else:
                tile_sched_args = TileSchedulerArguments(
                    problem_shape_ntile_mnl=grid,
                    raster_order=RasterOrderOption.Heuristic,
                    group_size=Int32(8),
                    cluster_shape_mnk=self.cluster_shape_mnk,
                    tile_count_semaphore=None,
                    batch_idx_permute=None,
                    persistence_mode=persistence_mode,
                )
                tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
                direct_grid = (
                    TileScheduler.get_grid_shape(tile_sched_params, self.max_active_clusters)
                    if const_expr(self.direct_tile_scheduler)
                    else (1, 1, m_tile_count * n_tile_count * problem_l)
                )
            cooperative_schedule = _sm120.make_mxf4nvf4_cooperative_schedule(
                producer_warpgroup_start=self.mma_warp_groups,
                consumer_warpgroups=self.mma_warp_groups,
            )
            cooperative_launch_kwargs = cooperative_schedule.launch_kwargs()
            scheduler_smem_bytes = 0
            if const_expr(not self.direct_cute_static_scheduler):
                scheduler_data_rows = (
                    12 if const_expr(persistence_mode == PersistenceMode.CLC) else 4
                )
                scheduler_smem_bytes = (
                    8 * self.sched_stage * 2 + 4 * scheduler_data_rows * self.sched_stage
                )
            split_smem_bytes = 82688 + max(0, self.ab_stage - 4) * 9728 + scheduler_smem_bytes
            single_smem_bytes = max(82432, self.ab_stage * (18432 + 16)) + scheduler_smem_bytes
            launch_smem_bytes = (
                split_smem_bytes
                if const_expr(self.direct_split_tma_pipelines)
                else single_smem_bytes
            )
            if const_expr(launch_smem_bytes > 101376):
                raise ValueError(
                    "SM120 NVFP4 kernel exceeds the 128x128 shared-memory budget; "
                    "use fewer A/B stages or disable the tiled scale SMEM path"
                )
            self.blockscaled_kernel(
                self.tiled_mma,
                tma_atom_d,
                tma_tensor_d,
                mD,
                self.epi_smem_layout_staged,
                cute.size(mD, mode=[2]),
                tma_atom_a,
                tma_tensor_a,
                tma_atom_b,
                tma_tensor_b,
                tma_atom_sfa,
                tma_tensor_sfa,
                tma_atom_sfb,
                tma_tensor_sfb,
                epilogue_params,
                tile_sched_params,
                k_tile_count,
                m_tile_count,
                n_tile_count,
            ).launch(
                grid=direct_grid,
                block=cooperative_launch_kwargs["block"],
                max_number_threads=cooperative_launch_kwargs["max_number_threads"],
                min_blocks_per_mp=cooperative_launch_kwargs["min_blocks_per_mp"],
                cluster=(1, 1, 1),
                stream=stream,
                smem=launch_smem_bytes,
            )
        else:
            raise NotImplementedError("SM120 NVFP4 blockscaled requires tile (128,128,128)")

    @cute.kernel
    def blockscaled_kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_d: cute.CopyAtom,
        tma_tensor_d: cute.Tensor,
        gD: cute.Tensor,
        epi_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        l_tiles: Int32,
        tma_atom_a: cute.CopyAtom,
        tma_tensor_a: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        tma_tensor_b: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        tma_tensor_sfa: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        tma_tensor_sfb: cute.Tensor,
        epilogue_params,
        tile_sched_params,
        k_tile_count: cutlass.Constexpr[int],
        m_tile_count: cutlass.Constexpr[int],
        n_tile_count: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // 32)
        lane_idx = tidx % 32
        cta_m, cta_n, batch_idx = cute.arch.block_idx()
        tile_extent_m, tile_extent_n, tile_extent_k = self.cta_tile_shape_mnk
        m_atoms = tile_extent_m // 16
        n_atoms = tile_extent_n // 8

        if const_expr(self.direct_tma_prefetch):
            if warp_idx == self.ab_load_warp_id:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_sfa)
                cpasync.prefetch_descriptor(tma_atom_sfb)

        smem = cutlass.utils.SmemAllocator()
        sA_consumer, sB_consumer, sSFA, sSFB = _sm120.make_mxf4nvf4_native_tma_smem_views(
            smem,
            tiled_mma=tiled_mma,
            num_stages=self.ab_stage,
            tile_m=tile_extent_m,
            tile_n=tile_extent_n,
            tile_k=tile_extent_k,
            sf_vec_size=16,
            ab_smem_format=self.direct_ab_tma_layout,
            scale_smem_format=self.direct_scale_smem_format,
        )
        if const_expr(self.direct_ab_tma_layout == "unpack"):
            sA_direct = sA_consumer
            sB_direct = sB_consumer
        else:
            sA_direct, sB_direct = _sm120.make_mxf4nvf4_ab_packed_direct_tma_consumer_tma_views(
                sA_consumer,
                sB_consumer,
            )
        if const_expr(self.direct_split_tma_pipelines):
            barriers_mk = smem.allocate_array(cutlass.Int64, self.ab_stage * 2, byte_alignment=8)
            if const_expr(self.direct_join_split_tma_barrier):
                barriers_nk = barriers_mk
            else:
                barriers_nk = smem.allocate_array(
                    cutlass.Int64, self.ab_stage * 2, byte_alignment=8
                )
        else:
            barriers = smem.allocate_array(cutlass.Int64, self.ab_stage * 2, byte_alignment=8)
        sD_epi = cute.make_tensor(
            cute.recast_ptr(sA_direct.iterator, dtype=cutlass.BFloat16),
            epi_smem_layout_staged,
        )
        epi_store_pipeline = self.make_sm120_single_warp_epi_store_pipeline()

        if const_expr(self.direct_split_tma_pipelines):
            producer_arrive_count = 2 if const_expr(self.direct_join_split_tma_barrier) else 1
            tma_consumer_warps = (
                4 if const_expr(self.direct_pingpong_split_tiles) else self.num_mma_warps
            )
            pipe_mk = PipelineTmaWarpMma.create(
                num_stages=self.ab_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, producer_arrive_count
                ),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, tma_consumer_warps),
                tx_count=(
                    _sm120.mxf4nvf4_ab_tma_tx_bytes(tile_extent_m, tile_extent_k)
                    + _sm120.mxf4nvf4_scale_tma_tx_bytes(tile_extent_m, tile_extent_k, 16)
                ),
                barrier_storage=barriers_mk,
            )
            if const_expr(self.direct_join_split_tma_barrier):
                pipe_nk = pipe_mk
            else:
                pipe_nk = PipelineTmaWarpMma.create(
                    num_stages=self.ab_stage,
                    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                    consumer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, tma_consumer_warps
                    ),
                    tx_count=(
                        _sm120.mxf4nvf4_ab_tma_tx_bytes(tile_extent_n, tile_extent_k)
                        + _sm120.mxf4nvf4_scale_tma_tx_bytes(tile_extent_n, tile_extent_k, 16)
                    ),
                    barrier_storage=barriers_nk,
                )
        else:
            tma_consumer_warps = (
                4
                if const_expr(self.direct_full_tma_pipeline and self.pingpong)
                else self.num_mma_warps
            )
            pipe = _sm120.make_mxf4nvf4_native_tma_pipeline(
                barrier_storage=barriers,
                num_stages=self.ab_stage,
                tile_m=tile_extent_m,
                tile_n=tile_extent_n,
                tile_k=tile_extent_k,
                sf_vec_size=16,
                ab_smem_format=self.direct_ab_tma_layout,
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    tma_consumer_warps,
                ),
            )

        if const_expr(self.direct_tile_scheduler or self.direct_split_tma_pipelines):
            (
                tAsA,
                tAgA,
                tBsB,
                tBgB,
                tSFAs,
                tSFAg,
                tSFBs,
                tSFBg,
            ) = _sm120.partition_mxf4nvf4_native_tma_tensors_for_scheduler(
                tma_atom_a,
                tma_tensor_a,
                tma_atom_b,
                tma_tensor_b,
                tma_atom_sfa,
                tma_tensor_sfa,
                tma_atom_sfb,
                tma_tensor_sfb,
                sA_consumer,
                sB_consumer,
                sSFA,
                sSFB,
                tile_shape_mnk=(tile_extent_m, tile_extent_n, tile_extent_k),
                sf_vec_size=16,
                scale_smem_format=self.direct_scale_smem_format,
            )
        mn_tile_count = m_tile_count * n_tile_count
        total_work = mn_tile_count * l_tiles
        work_stride = cute.arch.grid_dim()[2]
        if const_expr(self.direct_tile_scheduler):
            cluster_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
            if const_expr(not self.direct_cute_static_scheduler):
                sched_barriers = smem.allocate_array(
                    cutlass.Int64, self.sched_stage * 2, byte_alignment=8
                )
                sched_data_rows = (
                    12
                    if const_expr(tile_sched_params.persistence_mode == PersistenceMode.CLC)
                    else 4
                )
                sched_data = smem.allocate_tensor(
                    Int32,
                    cute.make_layout((sched_data_rows, self.sched_stage)),
                    byte_alignment=16,
                )
                sched_pipeline = self.make_sched_pipeline(
                    cluster_layout_mnk,
                    sched_pipeline_mbar_ptr=sched_barriers,
                    varlen_k=False,
                )
                TileSchedulerCls = partial(
                    TileScheduler.create, tile_sched_params, sched_data, sched_pipeline
                )
            pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if (
            const_expr(self.direct_setmaxregister)
            and self.ab_load_warp_id <= warp_idx
            and warp_idx < self.ab_load_warp_id + 4
        ):
            _sm120.setmaxregister_mxf4nvf4_producer(self.num_regs_load)

        tma_cache_policy = None
        if const_expr(self.direct_tma_policy == "evict_last"):
            tma_cache_policy = cpasync.create_l2_evict_last_policy()

        if const_expr(self.direct_split_tma_pipelines):
            if warp_idx == self.ab_load_warp_id:
                if const_expr(
                    (not self.direct_cute_static_scheduler)
                    and self.use_pdl
                    and tile_sched_params.persistence_mode == PersistenceMode.CLC
                ):
                    cute.arch.griddepcontrol_wait()
                producer_state_mk = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                if const_expr(self.direct_cute_static_scheduler):
                    tile_scheduler = _sm120.make_mxf4nvf4_static_tile_scheduler(
                        tile_sched_params,
                        cute.arch.block_idx(),
                        cute.arch.grid_dim(),
                    )
                else:
                    tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    cta_m = tile_coord_mnkl[0]
                    batch_idx = (
                        tile_coord_mnkl[2]
                        if const_expr(self.direct_cute_static_scheduler)
                        else tile_coord_mnkl[3]
                    )
                    for k_tile in cutlass.range(k_tile_count, unroll=self.direct_producer_unroll):
                        pipe_mk.producer_acquire(
                            producer_state_mk,
                            pipe_mk.producer_try_acquire(producer_state_mk),
                        )
                        _sm120.issue_mxf4nvf4_partitioned_native_tma_mk_stage_for_tile(
                            tma_atom_a,
                            tAsA,
                            tAgA,
                            tma_atom_sfa,
                            tSFAs,
                            tSFAg,
                            pipe_mk.producer_get_barrier(producer_state_mk),
                            (cta_m, tile_coord_mnkl[1], batch_idx),
                            k_tile,
                            producer_state_mk.index,
                            scale_smem_format=self.direct_scale_smem_format,
                            cache_policy=tma_cache_policy,
                        )
                        pipe_mk.producer_commit(producer_state_mk)
                        producer_state_mk.advance()
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                if const_expr(not self.direct_skip_split_tma_tail):
                    pipe_mk.producer_tail(producer_state_mk)

            if const_expr(not self.direct_cute_static_scheduler):
                if warp_idx == self.ab_load_warp_id + 1:
                    if const_expr(
                        self.use_pdl and tile_sched_params.persistence_mode == PersistenceMode.CLC
                    ):
                        cute.arch.griddepcontrol_wait()
                    tile_scheduler = TileSchedulerCls(is_scheduler_warp=True)
                    work_tile = tile_scheduler.initial_work_tile_info()
                    while work_tile.is_valid_tile:
                        tile_scheduler.advance_to_next_work(is_scheduler_warp=True)
                        work_tile = tile_scheduler.get_current_work()
                    if const_expr(self.direct_pingpong_split_tiles):
                        tile_scheduler.write_work_tile_to_smem(work_tile)
                    if const_expr(not self.direct_skip_scheduler_tail):
                        tile_scheduler.producer_tail()

            if warp_idx == self.ab_load_warp_id + 2:
                if const_expr(
                    (not self.direct_cute_static_scheduler)
                    and self.use_pdl
                    and tile_sched_params.persistence_mode == PersistenceMode.CLC
                ):
                    cute.arch.griddepcontrol_wait()
                producer_state_nk = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                if const_expr(self.direct_cute_static_scheduler):
                    tile_scheduler = _sm120.make_mxf4nvf4_static_tile_scheduler(
                        tile_sched_params,
                        cute.arch.block_idx(),
                        cute.arch.grid_dim(),
                    )
                else:
                    tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    cta_n = tile_coord_mnkl[1]
                    batch_idx = (
                        tile_coord_mnkl[2]
                        if const_expr(self.direct_cute_static_scheduler)
                        else tile_coord_mnkl[3]
                    )
                    for k_tile in cutlass.range(k_tile_count, unroll=self.direct_producer_unroll):
                        pipe_nk.producer_acquire(
                            producer_state_nk,
                            pipe_nk.producer_try_acquire(producer_state_nk),
                        )
                        _sm120.issue_mxf4nvf4_partitioned_native_tma_nk_stage_for_tile(
                            tma_atom_b,
                            tBsB,
                            tBgB,
                            tma_atom_sfb,
                            tSFBs,
                            tSFBg,
                            pipe_nk.producer_get_barrier(producer_state_nk),
                            (tile_coord_mnkl[0], cta_n, batch_idx),
                            k_tile,
                            producer_state_nk.index,
                            scale_smem_format=self.direct_scale_smem_format,
                            cache_policy=tma_cache_policy,
                        )
                        pipe_nk.producer_commit(producer_state_nk)
                        producer_state_nk.advance()
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                if const_expr(not self.direct_skip_split_tma_tail):
                    pipe_nk.producer_tail(producer_state_nk)

        elif warp_idx == self.ab_load_warp_id:
            if const_expr(
                (not self.direct_cute_static_scheduler)
                and self.use_pdl
                and tile_sched_params.persistence_mode == PersistenceMode.CLC
            ):
                cute.arch.griddepcontrol_wait()
            producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            if const_expr(self.direct_tile_scheduler):
                if const_expr(self.direct_cute_static_scheduler):
                    tile_scheduler = _sm120.make_mxf4nvf4_static_tile_scheduler(
                        tile_sched_params,
                        cute.arch.block_idx(),
                        cute.arch.grid_dim(),
                    )
                else:
                    tile_scheduler = TileSchedulerCls(is_scheduler_warp=True)
                work_tile = tile_scheduler.initial_work_tile_info()
            else:
                work_idx = cute.arch.block_idx()[2]
            while (
                work_tile.is_valid_tile
                if const_expr(self.direct_tile_scheduler)
                else work_idx < total_work
            ):
                if const_expr(self.direct_tile_scheduler):
                    tile_coord_mnkl = work_tile.tile_idx
                    cta_m = tile_coord_mnkl[0]
                    cta_n = tile_coord_mnkl[1]
                    batch_idx = (
                        tile_coord_mnkl[2]
                        if const_expr(self.direct_cute_static_scheduler)
                        else tile_coord_mnkl[3]
                    )
                else:
                    batch_idx = work_idx // mn_tile_count
                    mn_tile = work_idx - batch_idx * mn_tile_count
                    cta_n = mn_tile // m_tile_count
                    cta_m = mn_tile - cta_n * m_tile_count
                if const_expr((not self.direct_tile_scheduler) and k_tile_count == 1):
                    pipe.producer_acquire(
                        producer_state,
                        pipe.producer_try_acquire(producer_state),
                    )
                    _sm120.issue_mxf4nvf4_native_tma_stage_for_tile(
                        tma_atom_a,
                        tma_tensor_a,
                        tma_atom_b,
                        tma_tensor_b,
                        tma_atom_sfa,
                        tma_tensor_sfa,
                        tma_atom_sfb,
                        tma_tensor_sfb,
                        sA_consumer,
                        sB_consumer,
                        sSFA,
                        sSFB,
                        pipe.producer_get_barrier(producer_state),
                        (cta_m, cta_n, batch_idx),
                        0,
                        producer_state.index,
                        scale_smem_format=self.direct_scale_smem_format,
                        cache_policy=tma_cache_policy,
                    )
                    pipe.producer_commit(producer_state)
                    producer_state.advance()
                else:
                    for k_tile in cutlass.range(k_tile_count, unroll=self.direct_producer_unroll):
                        if const_expr(self.direct_elected_tma):
                            with cute.arch.elect_one():
                                pipe.producer_acquire_already_elected(
                                    producer_state,
                                    pipe.producer_try_acquire(producer_state),
                                )
                                if const_expr(
                                    (not self.direct_tile_scheduler)
                                    or self.direct_scheduler_local_tma
                                ):
                                    _sm120.issue_mxf4nvf4_native_tma_stage_for_tile(
                                        tma_atom_a,
                                        tma_tensor_a,
                                        tma_atom_b,
                                        tma_tensor_b,
                                        tma_atom_sfa,
                                        tma_tensor_sfa,
                                        tma_atom_sfb,
                                        tma_tensor_sfb,
                                        sA_consumer,
                                        sB_consumer,
                                        sSFA,
                                        sSFB,
                                        pipe.producer_get_barrier(producer_state),
                                        (cta_m, cta_n, batch_idx),
                                        k_tile,
                                        producer_state.index,
                                        already_elected=True,
                                        scale_smem_format=self.direct_scale_smem_format,
                                        cache_policy=tma_cache_policy,
                                    )
                                else:
                                    _sm120.issue_mxf4nvf4_partitioned_native_tma_stage_for_tile(
                                        tma_atom_a,
                                        tAsA,
                                        tAgA,
                                        tma_atom_b,
                                        tBsB,
                                        tBgB,
                                        tma_atom_sfa,
                                        tSFAs,
                                        tSFAg,
                                        tma_atom_sfb,
                                        tSFBs,
                                        tSFBg,
                                        pipe.producer_get_barrier(producer_state),
                                        (cta_m, cta_n, batch_idx),
                                        k_tile,
                                        producer_state.index,
                                        already_elected=True,
                                        scale_smem_format=self.direct_scale_smem_format,
                                        cache_policy=tma_cache_policy,
                                    )
                                pipe.producer_commit(producer_state)
                        else:
                            pipe.producer_acquire(
                                producer_state,
                                pipe.producer_try_acquire(producer_state),
                            )
                            if const_expr(
                                self.direct_tile_scheduler and not self.direct_scheduler_local_tma
                            ):
                                _sm120.issue_mxf4nvf4_partitioned_native_tma_stage_for_tile(
                                    tma_atom_a,
                                    tAsA,
                                    tAgA,
                                    tma_atom_b,
                                    tBsB,
                                    tBgB,
                                    tma_atom_sfa,
                                    tSFAs,
                                    tSFAg,
                                    tma_atom_sfb,
                                    tSFBs,
                                    tSFBg,
                                    pipe.producer_get_barrier(producer_state),
                                    (cta_m, cta_n, batch_idx),
                                    k_tile,
                                    producer_state.index,
                                    cache_policy=tma_cache_policy,
                                    scale_smem_format=self.direct_scale_smem_format,
                                )
                            else:
                                _sm120.issue_mxf4nvf4_native_tma_stage_for_tile(
                                    tma_atom_a,
                                    tma_tensor_a,
                                    tma_atom_b,
                                    tma_tensor_b,
                                    tma_atom_sfa,
                                    tma_tensor_sfa,
                                    tma_atom_sfb,
                                    tma_tensor_sfb,
                                    sA_consumer,
                                    sB_consumer,
                                    sSFA,
                                    sSFB,
                                    pipe.producer_get_barrier(producer_state),
                                    (cta_m, cta_n, batch_idx),
                                    k_tile,
                                    producer_state.index,
                                    scale_smem_format=self.direct_scale_smem_format,
                                    cache_policy=tma_cache_policy,
                                )
                            pipe.producer_commit(producer_state)
                        producer_state.advance()
                if const_expr(self.direct_tile_scheduler):
                    if const_expr(self.direct_cute_static_scheduler):
                        tile_scheduler.advance_to_next_work()
                    else:
                        tile_scheduler.advance_to_next_work(is_scheduler_warp=True)
                    work_tile = tile_scheduler.get_current_work()
                else:
                    work_idx += work_stride
            if const_expr(self.direct_tile_scheduler):
                if const_expr(
                    (not self.direct_cute_static_scheduler)
                    and (not self.direct_skip_scheduler_tail)
                ):
                    tile_scheduler.producer_tail()

        if const_expr(tile_extent_m == 128 and tile_extent_n == 128):
            if self.mma_warp_id_start <= warp_idx and warp_idx < self.mma_warp_id_end:
                warp_group_idx = cute.arch.make_warp_uniform(
                    tidx // self.num_threads_per_warp_group
                )
                pingpong_group_idx = warp_group_idx
                tidx_mma = tidx
                warp_idx_mma = warp_idx - self.mma_warp_id_start
                if const_expr(self.pingpong):
                    pingpong_group_idx = warp_group_idx - self.mma_warp_id_start // 4
                if const_expr(self.direct_pingpong_split_tiles):
                    tidx_mma = tidx % self.num_threads_per_warp_group
                    warp_idx_mma = warp_idx % 4
                    if const_expr(self.direct_pingpong_barriers) and pingpong_group_idx == 0:
                        self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                        self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")
                elif const_expr(self.pingpong):
                    tidx_mma = tidx % self.num_threads_per_warp_group
                    warp_idx_mma = warp_idx % 4
                    if (
                        const_expr(self.direct_full_tma_pipeline and self.direct_pingpong_barriers)
                        and pingpong_group_idx == 0
                    ):
                        self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                        self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")
                if const_expr(self.direct_setmaxregister):
                    _sm120.setmaxregister_mxf4nvf4_consumer(self.num_regs_mma)
                if const_expr(self.direct_split_tma_pipelines):
                    consumer_state_mk = make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.ab_stage
                    )
                    consumer_state_nk = make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.ab_stage
                    )
                else:
                    consumer_state = make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.ab_stage
                    )
                if const_expr(self.direct_tile_scheduler):
                    if const_expr(self.direct_cute_static_scheduler):
                        tile_scheduler = _sm120.make_mxf4nvf4_static_tile_scheduler(
                            tile_sched_params,
                            cute.arch.block_idx(),
                            cute.arch.grid_dim(),
                        )
                    else:
                        tile_scheduler = TileSchedulerCls()
                    work_tile = tile_scheduler.initial_work_tile_info()
                    if const_expr(self.direct_pingpong_split_tiles):
                        if pingpong_group_idx == 1:
                            consumer_state_mk.advance_iters(k_tile_count)
                            consumer_state_nk.advance_iters(k_tile_count)
                            tile_scheduler.advance_to_next_work()
                            work_tile = tile_scheduler.get_current_work()
                    elif const_expr(self.direct_full_tma_pipeline):
                        if pingpong_group_idx == 1:
                            consumer_state.advance_iters(k_tile_count)
                            tile_scheduler.advance_to_next_work()
                            work_tile = tile_scheduler.get_current_work()
                else:
                    work_idx = cute.arch.block_idx()[2]
                    if const_expr(self.direct_full_tma_pipeline):
                        if pingpong_group_idx == 1:
                            consumer_state.advance_iters(k_tile_count)
                            work_idx += work_stride
                while (
                    work_tile.is_valid_tile
                    if const_expr(self.direct_tile_scheduler)
                    else work_idx < total_work
                ):
                    if const_expr(self.direct_tile_scheduler):
                        tile_coord_mnkl = work_tile.tile_idx
                        cta_m = tile_coord_mnkl[0]
                        cta_n = tile_coord_mnkl[1]
                        batch_idx = (
                            tile_coord_mnkl[2]
                            if const_expr(self.direct_cute_static_scheduler)
                            else tile_coord_mnkl[3]
                        )
                    else:
                        batch_idx = work_idx // mn_tile_count
                        mn_tile = work_idx - batch_idx * mn_tile_count
                        cta_n = mn_tile // m_tile_count
                        cta_m = mn_tile - cta_n * m_tile_count
                    if const_expr(
                        (self.direct_pingpong_split_tiles or self.direct_full_tma_pipeline)
                        and self.direct_pingpong_barriers
                    ):
                        if const_expr(self.direct_try_wait_before_pingpong_barrier):
                            initial_try_wait_mk = pipe_mk.consumer_try_wait(consumer_state_mk)
                            initial_try_wait_nk = initial_try_wait_mk
                            if const_expr(not self.direct_join_split_tma_barrier):
                                initial_try_wait_nk = pipe_nk.consumer_try_wait(consumer_state_nk)
                        self.pingpong_barrier_sync(pingpong_group_idx, stage="mma")
                    tDsD_work, tDgD_work = self._partition_blockscaled_epilogue_tma_store(
                        tma_atom_d,
                        tma_tensor_d,
                        sD_epi,
                        cta_m,
                        cta_n,
                        batch_idx,
                    )
                    if const_expr(self.direct_split_tma_pipelines):
                        (
                            consumer_state_mk,
                            consumer_state_nk,
                        ) = self._blockscaled_compute_store_full_tile_k_loop_split_tma(
                            tiled_mma,
                            pipe_mk,
                            pipe_nk,
                            consumer_state_mk,
                            consumer_state_nk,
                            sA_consumer,
                            sB_consumer,
                            sSFA,
                            sSFB,
                            epilogue_params,
                            sD_epi,
                            tma_atom_d,
                            tDsD_work,
                            tDgD_work,
                            gD,
                            (cta_m, cta_n, batch_idx),
                            epi_store_pipeline,
                            tidx_mma,
                            warp_idx_mma,
                            pingpong_group_idx,
                            lane_idx,
                            k_tile_count,
                            tile_extent_m,
                            tile_extent_n,
                            tile_extent_k,
                            (
                                initial_try_wait_mk
                                if const_expr(self.direct_try_wait_before_pingpong_barrier)
                                else None
                            ),
                            (
                                initial_try_wait_nk
                                if const_expr(self.direct_try_wait_before_pingpong_barrier)
                                else None
                            ),
                        )
                    else:
                        consumer_state = self._blockscaled_compute_store_full_tile_k_loop(
                            tiled_mma,
                            pipe,
                            consumer_state,
                            sA_consumer,
                            sB_consumer,
                            sSFA,
                            sSFB,
                            epilogue_params,
                            sD_epi,
                            tma_atom_d,
                            tDsD_work,
                            tDgD_work,
                            gD,
                            (cta_m, cta_n, batch_idx),
                            epi_store_pipeline,
                            tidx_mma,
                            warp_idx_mma,
                            pingpong_group_idx,
                            lane_idx,
                            k_tile_count,
                            tile_extent_m,
                            tile_extent_n,
                            tile_extent_k,
                        )
                    if const_expr(self.direct_tile_scheduler):
                        if const_expr(self.direct_pingpong_split_tiles):
                            consumer_state_mk.advance_iters(k_tile_count)
                            consumer_state_nk.advance_iters(k_tile_count)
                            tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        elif const_expr(self.direct_full_tma_pipeline):
                            consumer_state.advance_iters(k_tile_count)
                            tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        else:
                            tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        if const_expr(self.direct_full_tma_pipeline):
                            consumer_state.advance_iters(k_tile_count)
                            work_idx += work_stride * self.mma_warp_groups
                        else:
                            work_idx += work_stride
                if const_expr(
                    (not self.direct_cute_static_scheduler)
                    and self.use_pdl
                    and tile_sched_params.persistence_mode == PersistenceMode.CLC
                ):
                    cute.arch.griddepcontrol_launch_dependents()
        else:
            tDsD, tDgD = self._partition_blockscaled_epilogue_tma_store(
                tma_atom_d,
                tma_tensor_d,
                sD_epi,
                cta_m,
                cta_n,
                batch_idx,
            )
            for m_atom in cutlass.range_constexpr(m_atoms):
                if warp_idx == m_atom:
                    self._blockscaled_compute_store_m_atom_k_loop(
                        tiled_mma,
                        pipe,
                        sA_consumer,
                        sB_consumer,
                        sSFA,
                        sSFB,
                        epilogue_params,
                        sD_epi,
                        tma_atom_d,
                        tDsD,
                        tDgD,
                        epi_store_pipeline,
                        m_atom,
                        lane_idx,
                        k_tile_count,
                        n_atoms,
                        tile_extent_m,
                        tile_extent_n,
                        tile_extent_k,
                        0,
                        1,
                        m_atom == 0,
                    )

    @cute.jit
    def _blockscaled_store_full_tile_direct_global(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        gD: cute.Tensor,
        tile_mnl,
        tidx: Int32,
        warp_group_idx: Int32,
    ) -> None:
        if const_expr(
            (self.direct_full_tma_pipeline or self.direct_pingpong_split_tiles)
            and self.direct_pingpong_barriers
        ):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
            self.pingpong_barrier_sync(warp_group_idx, stage="epi")
        _sm120.store_mxf4nvf4_accumulator_fragment_D_for_tiled_mma_tile(
            tiled_mma,
            acc,
            gD,
            tile_mnl,
            tidx,
        )
        if const_expr(
            (self.direct_full_tma_pipeline or self.direct_pingpong_split_tiles)
            and self.direct_pingpong_barriers
        ):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")

    @cute.jit
    def _blockscaled_compute_store_full_tile_k_loop(
        self,
        tiled_mma: cute.TiledMma,
        pipe,
        consumer_state: cutlass.pipeline.PipelineState,
        sA_consumer: cute.Tensor,
        sB_consumer: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        epilogue_params,
        sD_epi: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        tDsD: cute.Tensor,
        tDgD: cute.Tensor,
        gD: Optional[cute.Tensor],
        tile_mnl,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        warp_idx: Int32,
        warp_group_idx: Int32,
        lane_idx: Int32,
        k_tile_count: Int32,
        tile_extent_m: cutlass.Constexpr[int],
        tile_extent_n: cutlass.Constexpr[int],
        tile_extent_k: cutlass.Constexpr[int],
    ) -> cutlass.pipeline.PipelineState:
        if const_expr(self.direct_pipelined_consumer):
            return self._blockscaled_compute_store_full_tile_k_loop_pipelined(
                tiled_mma,
                pipe,
                consumer_state,
                sA_consumer,
                sB_consumer,
                sSFA,
                sSFB,
                epilogue_params,
                sD_epi,
                tma_atom_d,
                tDsD,
                tDgD,
                gD,
                tile_mnl,
                epi_store_pipeline,
                tidx,
                warp_idx,
                warp_group_idx,
                lane_idx,
                k_tile_count,
                tile_extent_m,
                tile_extent_n,
                tile_extent_k,
            )

        a_frag = cute.make_rmem_tensor(
            tiled_mma.partition_shape_A((tile_extent_m, tile_extent_k)),
            cutlass.Float4E2M1FN,
        )
        b_frag = cute.make_rmem_tensor(
            tiled_mma.partition_shape_B((tile_extent_n, tile_extent_k)),
            cutlass.Float4E2M1FN,
        )
        if const_expr(self.direct_ab_tma_layout == "unpack"):
            a_copy_frag = cute.recast_tensor(a_frag, cutlass.Uint8)
            b_copy_frag = cute.recast_tensor(b_frag, cutlass.Uint8)
        else:
            a_copy_frag = a_frag
            b_copy_frag = b_frag
        acc = cute.make_rmem_tensor(
            tiled_mma.partition_shape_C((tile_extent_m, tile_extent_n)),
            cutlass.Float32,
        )
        acc.fill(0.0)
        if const_expr(self.direct_ab_tma_layout == "unpack"):
            copy_atom_a = cute.make_copy_atom(
                warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4, unpack_bits=4),
                cutlass.Uint8,
            )
            copy_atom_b = cute.make_copy_atom(
                warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4, unpack_bits=4),
                cutlass.Uint8,
            )
        else:
            copy_atom_a = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.Float4E2M1FN,
            )
            copy_atom_b = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.Float4E2M1FN,
            )
        tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma)
        tiled_copy_b = cute.make_tiled_copy_B(copy_atom_b, tiled_mma)
        thr_copy_a = tiled_copy_a.get_slice(tidx)
        thr_copy_b = tiled_copy_b.get_slice(tidx)
        tCsA = thr_copy_a.partition_S(cute.as_position_independent_swizzle_tensor(sA_consumer))
        tCsB = thr_copy_b.partition_S(cute.as_position_independent_swizzle_tensor(sB_consumer))
        if const_expr(self.direct_ab_tma_layout == "unpack"):
            tCsA = cute.make_tensor(tCsA.iterator.align(16), tCsA.layout)
            tCsB = cute.make_tensor(tCsB.iterator.align(16), tCsB.layout)
        tCrA = thr_copy_a.retile(a_copy_frag)
        tCrB = thr_copy_b.retile(b_copy_frag)

        sfa_frag, sfb_frag = _sm120.make_mxf4nvf4_direct_tma_scale_fragments(
            tiled_mma,
            sSFA,
            sSFB,
            tidx,
            tile_shape_mnk=(tile_extent_m, tile_extent_n, tile_extent_k),
            sf_vec_size=16,
        )
        scale_stage_thread_idx = None
        scale_stage_thread_count = cute.arch.WARP_SIZE
        scale_barrier_id = Int32(8)
        for _k_tile in cutlass.range(k_tile_count, unroll=self.direct_consumer_unroll):
            stage = consumer_state.index
            pipe.consumer_wait(consumer_state, pipe.consumer_try_wait(consumer_state))
            if const_expr(self.direct_cute_dsl_helpers):
                _sm120.issue_mxf4nvf4_direct_tma_consumer_group(
                    tiled_mma,
                    tiled_copy_a,
                    tiled_copy_b,
                    tCsA,
                    tCsB,
                    tCrA,
                    tCrB,
                    a_frag,
                    b_frag,
                    sSFA,
                    sSFB,
                    sfa_frag,
                    sfb_frag,
                    acc,
                    tidx,
                    stage,
                    major_extent_sfa=tile_extent_m,
                    major_extent_sfb=tile_extent_n,
                    tile_k=tile_extent_k,
                    sf_vec_size=16,
                )
            elif const_expr(self.direct_kblock_pipeline):
                cute.copy(
                    tiled_copy_a,
                    tCsA[(None, None, 0, stage)],
                    tCrA[(None, None, 0)],
                )
                cute.copy(
                    tiled_copy_b,
                    tCsB[(None, None, 0, stage)],
                    tCrB[(None, None, 0)],
                )
                self._blockscaled_load_scale_fragments(
                    tiled_mma,
                    sSFA,
                    sSFB,
                    sfa_frag,
                    sfb_frag,
                    tidx,
                    0,
                    stage,
                    tile_extent_m,
                    tile_extent_n,
                    tile_extent_k,
                )
                cute.copy(
                    tiled_copy_a,
                    tCsA[(None, None, 1, stage)],
                    tCrA[(None, None, 1)],
                )
                cute.copy(
                    tiled_copy_b,
                    tCsB[(None, None, 1, stage)],
                    tCrB[(None, None, 1)],
                )
                self._blockscaled_load_scale_fragments(
                    tiled_mma,
                    sSFA,
                    sSFB,
                    sfa_frag,
                    sfb_frag,
                    tidx,
                    1,
                    stage,
                    tile_extent_m,
                    tile_extent_n,
                    tile_extent_k,
                )
                for k_block_idx in cutlass.range_constexpr(2):
                    cute.gemm(
                        tiled_mma,
                        acc,
                        (
                            a_frag[(None, None, k_block_idx)],
                            sfa_frag[(None, None, k_block_idx)],
                        ),
                        (
                            b_frag[(None, None, k_block_idx)],
                            sfb_frag[(None, None, k_block_idx)],
                        ),
                        acc,
                    )
            else:
                cute.copy(
                    tiled_copy_a,
                    tCsA[(None, None, 0, stage)],
                    tCrA[(None, None, 0)],
                )
                cute.copy(
                    tiled_copy_b,
                    tCsB[(None, None, 0, stage)],
                    tCrB[(None, None, 0)],
                )
                cute.copy(
                    tiled_copy_a,
                    tCsA[(None, None, 1, stage)],
                    tCrA[(None, None, 1)],
                )
                cute.copy(
                    tiled_copy_b,
                    tCsB[(None, None, 1, stage)],
                    tCrB[(None, None, 1)],
                )
                for k_block_idx in cutlass.range_constexpr(2):
                    self._blockscaled_load_scale_fragments(
                        tiled_mma,
                        sSFA,
                        sSFB,
                        sfa_frag,
                        sfb_frag,
                        tidx,
                        k_block_idx,
                        stage,
                        tile_extent_m,
                        tile_extent_n,
                        tile_extent_k,
                    )
                cute.gemm(
                    tiled_mma,
                    acc,
                    (a_frag, sfa_frag),
                    (b_frag, sfb_frag),
                    acc,
                )
            if const_expr(self.direct_release_before_sync):
                pipe.consumer_release(consumer_state)
                consumer_state.advance()
            if const_expr(self.direct_consumer_barrier):
                self._direct_kblock_barrier()
            if const_expr(self.direct_consumer_fence):
                cute.arch.fence_view_async_shared()
            if const_expr(self.direct_consumer_warp_sync):
                cute.arch.sync_warp()
            if const_expr(not self.direct_release_before_sync):
                pipe.consumer_release(consumer_state)
                consumer_state.advance()

        if const_expr(self.direct_global_store or self.direct_global_store_probe):
            self._blockscaled_store_full_tile_direct_global(
                tiled_mma,
                acc,
                gD,
                tile_mnl,
                tidx,
                warp_group_idx,
            )
            return consumer_state

        if const_expr(self.direct_full_tma_pipeline and self.direct_pingpong_barriers):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
            self.pingpong_barrier_sync(warp_group_idx, stage="epi")
        if const_expr(self.direct_delay_tma_store):
            if warp_idx == 0:
                epi_store_pipeline.producer_acquire()
        for epi_tile_idx in cutlass.range_constexpr(self.direct_epi_tiles):
            epi_m = epi_tile_idx // self.direct_epi_n_tiles
            epi_n = epi_tile_idx - epi_m * self.direct_epi_n_tiles
            epi_stage_idx = epi_tile_idx % self.epi_stage
            prev_epi_tile_idx = epi_tile_idx - 1
            prev_epi_m = prev_epi_tile_idx // self.direct_epi_n_tiles
            prev_epi_n = prev_epi_tile_idx - prev_epi_m * self.direct_epi_n_tiles
            prev_epi_stage_idx = prev_epi_tile_idx % self.epi_stage
            acc_epi_m = epi_m
            if const_expr(self.direct_delay_tma_store and epi_tile_idx != 0):
                if warp_idx == 0:
                    self._copy_blockscaled_epilogue_tma_store(
                        tma_atom_d,
                        tDsD,
                        tDgD,
                        prev_epi_m,
                        prev_epi_n,
                        prev_epi_stage_idx,
                    )
                    epi_store_pipeline.producer_commit()
                    epi_store_pipeline.producer_acquire()
                self.epilogue_barrier.arrive_and_wait()
            elif const_expr(not self.direct_delay_tma_store) and warp_idx == 0:
                epi_store_pipeline.producer_acquire()
            if const_expr(not self.direct_epi_barrier_trim):
                self.epilogue_barrier.arrive_and_wait()
            if const_expr(
                (not self.pingpong)
                or self.direct_pingpong_split_tiles
                or self.direct_full_tma_pipeline
            ):
                self._blockscaled_stage_full_tile_to_epi_smem(
                    tiled_mma,
                    epilogue_params,
                    sD_epi,
                    acc,
                    tidx,
                    acc_epi_m,
                    epi_n,
                    epi_stage_idx,
                )
            elif const_expr(epi_tile_idx == 0):
                if warp_group_idx == 0:
                    self._blockscaled_stage_full_tile_to_epi_smem(
                        tiled_mma,
                        epilogue_params,
                        sD_epi,
                        acc,
                        tidx,
                        acc_epi_m,
                        epi_n,
                        epi_stage_idx,
                    )
            else:
                if warp_group_idx == 1:
                    self._blockscaled_stage_full_tile_to_epi_smem(
                        tiled_mma,
                        epilogue_params,
                        sD_epi,
                        acc,
                        tidx,
                        acc_epi_m,
                        epi_n,
                        epi_stage_idx,
                    )
            cute.arch.fence_view_async_shared()
            self.epilogue_barrier.arrive_and_wait()
            if const_expr(not self.direct_delay_tma_store):
                if warp_idx == 0:
                    self._copy_blockscaled_epilogue_tma_store(
                        tma_atom_d,
                        tDsD,
                        tDgD,
                        epi_m,
                        epi_n,
                        0,
                    )
                    epi_store_pipeline.producer_commit()
                self.epilogue_barrier.arrive_and_wait()
        if const_expr(self.direct_delay_tma_store):
            last_epi_tile_idx = self.direct_epi_tiles - 1
            last_epi_m = last_epi_tile_idx // self.direct_epi_n_tiles
            last_epi_n = last_epi_tile_idx - last_epi_m * self.direct_epi_n_tiles
            last_epi_stage_idx = last_epi_tile_idx % self.epi_stage
            if warp_idx == 0:
                self._copy_blockscaled_epilogue_tma_store(
                    tma_atom_d,
                    tDsD,
                    tDgD,
                    last_epi_m,
                    last_epi_n,
                    last_epi_stage_idx,
                )
                epi_store_pipeline.producer_commit()
            self.epilogue_barrier.arrive_and_wait()
        if warp_idx == 0:
            epi_store_pipeline.producer_tail()
        if const_expr(self.direct_full_tma_pipeline and self.direct_pingpong_barriers):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
        return consumer_state

    @cute.jit
    def _blockscaled_compute_store_full_tile_k_loop_pipelined(
        self,
        tiled_mma: cute.TiledMma,
        pipe,
        consumer_state: cutlass.pipeline.PipelineState,
        sA_consumer: cute.Tensor,
        sB_consumer: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        epilogue_params,
        sD_epi: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        tDsD: cute.Tensor,
        tDgD: cute.Tensor,
        gD: Optional[cute.Tensor],
        tile_mnl,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        warp_idx: Int32,
        warp_group_idx: Int32,
        lane_idx: Int32,
        k_tile_count: Int32,
        tile_extent_m: cutlass.Constexpr[int],
        tile_extent_n: cutlass.Constexpr[int],
        tile_extent_k: cutlass.Constexpr[int],
    ) -> cutlass.pipeline.PipelineState:
        a_frag = cute.make_rmem_tensor(
            tiled_mma.partition_shape_A((tile_extent_m, tile_extent_k)),
            cutlass.Float4E2M1FN,
        )
        b_frag = cute.make_rmem_tensor(
            tiled_mma.partition_shape_B((tile_extent_n, tile_extent_k)),
            cutlass.Float4E2M1FN,
        )
        if const_expr(self.direct_ab_tma_layout == "unpack"):
            a_copy_frag = cute.recast_tensor(a_frag, cutlass.Uint8)
            b_copy_frag = cute.recast_tensor(b_frag, cutlass.Uint8)
        else:
            a_copy_frag = a_frag
            b_copy_frag = b_frag
        acc = cute.make_rmem_tensor(
            tiled_mma.partition_shape_C((tile_extent_m, tile_extent_n)),
            cutlass.Float32,
        )
        acc.fill(0.0)

        if const_expr(self.direct_ab_tma_layout == "unpack"):
            copy_atom_a = cute.make_copy_atom(
                warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4, unpack_bits=4),
                cutlass.Uint8,
            )
            copy_atom_b = cute.make_copy_atom(
                warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4, unpack_bits=4),
                cutlass.Uint8,
            )
        else:
            copy_atom_a = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.Float4E2M1FN,
            )
            copy_atom_b = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.Float4E2M1FN,
            )
        tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma)
        tiled_copy_b = cute.make_tiled_copy_B(copy_atom_b, tiled_mma)
        thr_copy_a = tiled_copy_a.get_slice(tidx)
        thr_copy_b = tiled_copy_b.get_slice(tidx)
        tCsA = thr_copy_a.partition_S(cute.as_position_independent_swizzle_tensor(sA_consumer))
        tCsB = thr_copy_b.partition_S(cute.as_position_independent_swizzle_tensor(sB_consumer))
        if const_expr(self.direct_ab_tma_layout == "unpack"):
            tCsA = cute.make_tensor(tCsA.iterator.align(16), tCsA.layout)
            tCsB = cute.make_tensor(tCsB.iterator.align(16), tCsB.layout)
        tCrA = thr_copy_a.retile(a_copy_frag)
        tCrB = thr_copy_b.retile(b_copy_frag)

        sfa_frag, sfb_frag = _sm120.make_mxf4nvf4_direct_tma_scale_fragments(
            tiled_mma,
            sSFA,
            sSFB,
            tidx,
            tile_shape_mnk=(tile_extent_m, tile_extent_n, tile_extent_k),
            sf_vec_size=16,
        )
        scale_stage_thread_idx = None
        scale_stage_thread_count = cute.arch.WARP_SIZE
        scale_barrier_id = Int32(8)

        stage = consumer_state.index
        pipe.consumer_wait(consumer_state, pipe.consumer_try_wait(consumer_state))

        if const_expr(self.direct_bgroup_pipeline):
            self._blockscaled_load_direct_k_block_a_scale_fragments(
                tiled_mma,
                tiled_copy_a,
                tCsA,
                tCrA,
                sSFA,
                sSFB,
                sfa_frag,
                sfb_frag,
                tidx,
                0,
                stage,
                tile_extent_m,
                tile_extent_n,
                tile_extent_k,
                scale_stage_thread_idx,
                scale_stage_thread_count,
                scale_barrier_id,
            )
            for b_group_idx in cutlass.range_constexpr(2):
                self._blockscaled_load_direct_k_block_b_group_fragment(
                    tiled_copy_b,
                    tCsB,
                    tCrB,
                    0,
                    b_group_idx,
                    stage,
                )
        _sm120.load_mxf4nvf4_direct_tma_k_block_fragments(
            tiled_mma,
            tiled_copy_a,
            tiled_copy_b,
            tCsA,
            tCsB,
            tCrA,
            tCrB,
            sSFA,
            sSFB,
            sfa_frag,
            sfb_frag,
            tidx,
            0,
            stage_idx=stage,
            major_extent_sfa=tile_extent_m,
            major_extent_sfb=tile_extent_n,
            tile_k=tile_extent_k,
            sf_vec_size=16,
            scale_first=self.direct_scale_prefetch_first,
        )

        for _k_tile in cutlass.range(k_tile_count - 1, unroll=1):
            for k_block_idx in cutlass.range_constexpr(2):
                k_block_next = 0 if const_expr(k_block_idx == 1) else 1
                if const_expr(k_block_idx == 1):
                    if const_expr(self.direct_kblock_barrier):
                        self._direct_kblock_barrier()
                    pipe.consumer_release(consumer_state)
                    consumer_state.advance()
                    stage = consumer_state.index
                    pipe.consumer_wait(consumer_state, pipe.consumer_try_wait(consumer_state))
                if const_expr(self.direct_bgroup_pipeline):
                    self._blockscaled_load_direct_k_block_a_scale_fragments(
                        tiled_mma,
                        tiled_copy_a,
                        tCsA,
                        tCrA,
                        sSFA,
                        sSFB,
                        sfa_frag,
                        sfb_frag,
                        tidx,
                        k_block_next,
                        stage,
                        tile_extent_m,
                        tile_extent_n,
                        tile_extent_k,
                        scale_stage_thread_idx,
                        scale_stage_thread_count,
                        scale_barrier_id,
                    )
                    for b_group_idx in cutlass.range_constexpr(2):
                        self._blockscaled_load_direct_k_block_b_group_fragment(
                            tiled_copy_b,
                            tCsB,
                            tCrB,
                            k_block_next,
                            b_group_idx,
                            stage,
                        )
                    self._blockscaled_direct_gemm_k_block_bgroup_pipeline(
                        tiled_mma,
                        tiled_copy_b,
                        acc,
                        tCsB,
                        tCrB,
                        a_frag,
                        b_frag,
                        sfa_frag,
                        sfb_frag,
                        k_block_idx,
                        stage,
                    )
                _sm120.load_mxf4nvf4_direct_tma_k_block_fragments(
                    tiled_mma,
                    tiled_copy_a,
                    tiled_copy_b,
                    tCsA,
                    tCsB,
                    tCrA,
                    tCrB,
                    sSFA,
                    sSFB,
                    sfa_frag,
                    sfb_frag,
                    tidx,
                    k_block_next,
                    stage_idx=stage,
                    major_extent_sfa=tile_extent_m,
                    major_extent_sfb=tile_extent_n,
                    tile_k=tile_extent_k,
                    sf_vec_size=16,
                    scale_first=self.direct_scale_prefetch_first,
                )
                self._blockscaled_direct_gemm_k_block(
                    tiled_mma,
                    acc,
                    a_frag,
                    b_frag,
                    sfa_frag,
                    sfb_frag,
                    k_block_idx,
                )
        for k_block_idx in cutlass.range_constexpr(2):
            k_block_next = 0 if const_expr(k_block_idx == 1) else 1
            if const_expr(k_block_idx == 1):
                if const_expr(self.direct_kblock_barrier):
                    self._direct_kblock_barrier()
                pipe.consumer_release(consumer_state)
                consumer_state.advance()
            if const_expr(k_block_idx == 0):
                if const_expr(self.direct_bgroup_pipeline):
                    self._blockscaled_load_direct_k_block_a_scale_fragments(
                        tiled_mma,
                        tiled_copy_a,
                        tCsA,
                        tCrA,
                        sSFA,
                        sSFB,
                        sfa_frag,
                        sfb_frag,
                        tidx,
                        k_block_next,
                        stage,
                        tile_extent_m,
                        tile_extent_n,
                        tile_extent_k,
                    )
                    for b_group_idx in cutlass.range_constexpr(2):
                        self._blockscaled_load_direct_k_block_b_group_fragment(
                            tiled_copy_b,
                            tCsB,
                            tCrB,
                            k_block_next,
                            b_group_idx,
                            stage,
                        )
                else:
                    _sm120.load_mxf4nvf4_direct_tma_k_block_fragments(
                        tiled_mma,
                        tiled_copy_a,
                        tiled_copy_b,
                        tCsA,
                        tCsB,
                        tCrA,
                        tCrB,
                        sSFA,
                        sSFB,
                        sfa_frag,
                        sfb_frag,
                        tidx,
                        k_block_next,
                        stage_idx=stage,
                        major_extent_sfa=tile_extent_m,
                        major_extent_sfb=tile_extent_n,
                        tile_k=tile_extent_k,
                        sf_vec_size=16,
                        scale_first=self.direct_scale_prefetch_first,
                    )
            if const_expr(self.direct_bgroup_pipeline):
                self._blockscaled_direct_gemm_k_block_bgroup_pipeline(
                    tiled_mma,
                    tiled_copy_b,
                    acc,
                    tCsB,
                    tCrB,
                    a_frag,
                    b_frag,
                    sfa_frag,
                    sfb_frag,
                    k_block_idx,
                    stage,
                )
            else:
                self._blockscaled_direct_gemm_k_block(
                    tiled_mma,
                    acc,
                    a_frag,
                    b_frag,
                    sfa_frag,
                    sfb_frag,
                    k_block_idx,
                )

        if const_expr(self.direct_global_store or self.direct_global_store_probe):
            self._blockscaled_store_full_tile_direct_global(
                tiled_mma,
                acc,
                gD,
                tile_mnl,
                tidx,
                warp_group_idx,
            )
            return consumer_state

        if const_expr(self.direct_full_tma_pipeline and self.direct_pingpong_barriers):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
            self.pingpong_barrier_sync(warp_group_idx, stage="epi")
        if const_expr(self.direct_delay_tma_store):
            if warp_idx == 0:
                epi_store_pipeline.producer_acquire()
        for epi_tile_idx in cutlass.range_constexpr(self.direct_epi_tiles):
            epi_m = epi_tile_idx // self.direct_epi_n_tiles
            epi_n = epi_tile_idx - epi_m * self.direct_epi_n_tiles
            epi_stage_idx = epi_tile_idx % self.epi_stage
            prev_epi_tile_idx = epi_tile_idx - 1
            prev_epi_m = prev_epi_tile_idx // self.direct_epi_n_tiles
            prev_epi_n = prev_epi_tile_idx - prev_epi_m * self.direct_epi_n_tiles
            prev_epi_stage_idx = prev_epi_tile_idx % self.epi_stage
            acc_epi_m = epi_m
            if const_expr(self.direct_delay_tma_store and epi_tile_idx != 0):
                if warp_idx == 0:
                    self._copy_blockscaled_epilogue_tma_store(
                        tma_atom_d,
                        tDsD,
                        tDgD,
                        prev_epi_m,
                        prev_epi_n,
                        prev_epi_stage_idx,
                    )
                    epi_store_pipeline.producer_commit()
                    epi_store_pipeline.producer_acquire()
                self.epilogue_barrier.arrive_and_wait()
            elif const_expr(not self.direct_delay_tma_store) and warp_idx == 0:
                epi_store_pipeline.producer_acquire()
            if const_expr(not self.direct_epi_barrier_trim):
                self.epilogue_barrier.arrive_and_wait()
            if const_expr(
                (not self.pingpong)
                or self.direct_pingpong_split_tiles
                or self.direct_full_tma_pipeline
            ):
                self._blockscaled_stage_full_tile_to_epi_smem(
                    tiled_mma,
                    epilogue_params,
                    sD_epi,
                    acc,
                    tidx,
                    acc_epi_m,
                    epi_n,
                    epi_stage_idx,
                )
            elif const_expr(epi_tile_idx == 0):
                if warp_group_idx == 0:
                    self._blockscaled_stage_full_tile_to_epi_smem(
                        tiled_mma,
                        epilogue_params,
                        sD_epi,
                        acc,
                        tidx,
                        acc_epi_m,
                        epi_n,
                        epi_stage_idx,
                    )
            else:
                if warp_group_idx == 1:
                    self._blockscaled_stage_full_tile_to_epi_smem(
                        tiled_mma,
                        epilogue_params,
                        sD_epi,
                        acc,
                        tidx,
                        acc_epi_m,
                        epi_n,
                        epi_stage_idx,
                    )
            cute.arch.fence_view_async_shared()
            self.epilogue_barrier.arrive_and_wait()
            if const_expr(not self.direct_delay_tma_store):
                if warp_idx == 0:
                    self._copy_blockscaled_epilogue_tma_store(
                        tma_atom_d,
                        tDsD,
                        tDgD,
                        epi_m,
                        epi_n,
                        0,
                    )
                    epi_store_pipeline.producer_commit()
                self.epilogue_barrier.arrive_and_wait()
        if const_expr(self.direct_delay_tma_store):
            last_epi_tile_idx = self.direct_epi_tiles - 1
            last_epi_m = last_epi_tile_idx // self.direct_epi_n_tiles
            last_epi_n = last_epi_tile_idx - last_epi_m * self.direct_epi_n_tiles
            last_epi_stage_idx = last_epi_tile_idx % self.epi_stage
            if warp_idx == 0:
                self._copy_blockscaled_epilogue_tma_store(
                    tma_atom_d,
                    tDsD,
                    tDgD,
                    last_epi_m,
                    last_epi_n,
                    last_epi_stage_idx,
                )
                epi_store_pipeline.producer_commit()
            self.epilogue_barrier.arrive_and_wait()
        if warp_idx == 0:
            epi_store_pipeline.producer_tail()
        if const_expr(self.direct_full_tma_pipeline and self.direct_pingpong_barriers):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
        return consumer_state

    @cute.jit
    def _split_tma_consumer_wait(
        self,
        pipe_mk,
        pipe_nk,
        consumer_state_mk: cutlass.pipeline.PipelineState,
        consumer_state_nk: cutlass.pipeline.PipelineState,
    ) -> None:
        _sm120.mxf4nvf4_split_tma_consumer_wait(
            pipe_mk,
            pipe_nk,
            consumer_state_mk,
            consumer_state_nk,
            join_split_tma_barrier=self.direct_join_split_tma_barrier,
        )

    @cute.jit
    def _split_tma_consumer_wait_with_tokens(
        self,
        pipe_mk,
        pipe_nk,
        consumer_state_mk: cutlass.pipeline.PipelineState,
        consumer_state_nk: cutlass.pipeline.PipelineState,
        try_wait_token_mk: Boolean,
        try_wait_token_nk: Boolean,
    ) -> None:
        _sm120.mxf4nvf4_split_tma_consumer_wait(
            pipe_mk,
            pipe_nk,
            consumer_state_mk,
            consumer_state_nk,
            join_split_tma_barrier=self.direct_join_split_tma_barrier,
            try_wait_token_mk=try_wait_token_mk,
            try_wait_token_nk=try_wait_token_nk,
        )

    @cute.jit
    def _split_tma_consumer_release(
        self,
        pipe_mk,
        pipe_nk,
        consumer_state_mk: cutlass.pipeline.PipelineState,
        consumer_state_nk: cutlass.pipeline.PipelineState,
    ) -> None:
        _sm120.mxf4nvf4_split_tma_consumer_release(
            pipe_mk,
            pipe_nk,
            consumer_state_mk,
            consumer_state_nk,
            join_split_tma_barrier=self.direct_join_split_tma_barrier,
        )

    @cute.jit
    def _direct_kblock_barrier(self) -> None:
        _sm120.mxf4nvf4_mma_warpgroup_barrier_sync(
            number_of_threads=self.num_mma_warps * cute.arch.WARP_SIZE,
        )

    def pingpong_barrier_sync(self, warp_group_idx: Int32, stage: str):
        _sm120.mxf4nvf4_pingpong_barrier_sync(warp_group_idx, stage)

    def pingpong_barrier_arrive(self, warp_group_idx: Int32, stage: str):
        _sm120.mxf4nvf4_pingpong_barrier_arrive(warp_group_idx, stage)

    @cute.jit
    def _partition_blockscaled_epilogue_tma_store(
        self,
        tma_atom_d: cute.CopyAtom,
        tma_tensor_d: cute.Tensor,
        sD_epi: cute.Tensor,
        cta_m: Int32,
        cta_n: Int32,
        batch_idx: Int32,
    ):
        return _sm120.partition_mxf4nvf4_epilogue_tma_store(
            tma_atom_d,
            tma_tensor_d,
            sD_epi,
            (cta_m, cta_n, batch_idx),
            cta_tiler=self.cta_tile_shape_mnk,
            epi_tile=self.epi_tile,
        )

    @cute.jit
    def _copy_blockscaled_epilogue_tma_store(
        self,
        tma_atom_d: cute.CopyAtom,
        tDsD: cute.Tensor,
        tDgD: cute.Tensor,
        epi_m: cutlass.Constexpr[int],
        epi_n: cutlass.Constexpr[int],
        epi_stage_idx: cutlass.Constexpr[int],
    ) -> None:
        if const_expr(self.direct_epi_tma_rank3):
            # The rank-3 TMA partition keeps the epilogue stage in the source
            # tensor basis. The copy source coordinate is the TMA vector mode.
            cute.copy(tma_atom_d, tDsD[None, epi_stage_idx], tDgD[None, (epi_m, epi_n)])
        else:
            _sm120.issue_mxf4nvf4_epilogue_tma_store(
                tma_atom_d,
                tDsD,
                tDgD,
                epi_m=epi_m,
                epi_n=epi_n,
                stage_idx=epi_stage_idx,
            )

    @cute.jit
    def _blockscaled_compute_store_full_tile_k_loop_split_tma(
        self,
        tiled_mma: cute.TiledMma,
        pipe_mk,
        pipe_nk,
        consumer_state_mk: cutlass.pipeline.PipelineState,
        consumer_state_nk: cutlass.pipeline.PipelineState,
        sA_consumer: cute.Tensor,
        sB_consumer: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        epilogue_params,
        sD_epi: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        tDsD: cute.Tensor,
        tDgD: cute.Tensor,
        gD: Optional[cute.Tensor],
        tile_mnl,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        warp_idx: Int32,
        warp_group_idx: Int32,
        lane_idx: Int32,
        k_tile_count: Int32,
        tile_extent_m: cutlass.Constexpr[int],
        tile_extent_n: cutlass.Constexpr[int],
        tile_extent_k: cutlass.Constexpr[int],
        initial_try_wait_mk: Optional[Boolean] = None,
        initial_try_wait_nk: Optional[Boolean] = None,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        if const_expr(self.direct_fragment_contract == "thread_mma"):
            thr_mma_for_frag = tiled_mma.get_slice(tidx)
            tCsA_mma = thr_mma_for_frag.partition_A(sA_consumer)
            tCsB_mma = thr_mma_for_frag.partition_B(sB_consumer)
            a_frag = tiled_mma.make_fragment_A(tCsA_mma[(None, None, None, 0)])
            b_frag = tiled_mma.make_fragment_B(tCsB_mma[(None, None, None, 0)])
        else:
            a_frag = cute.make_rmem_tensor(
                tiled_mma.partition_shape_A((tile_extent_m, tile_extent_k)),
                cutlass.Float4E2M1FN,
            )
            b_frag = cute.make_rmem_tensor(
                tiled_mma.partition_shape_B((tile_extent_n, tile_extent_k)),
                cutlass.Float4E2M1FN,
            )
        if const_expr(self.direct_ab_tma_layout == "unpack"):
            a_copy_frag = cute.recast_tensor(a_frag, cutlass.Uint8)
            b_copy_frag = cute.recast_tensor(b_frag, cutlass.Uint8)
        else:
            a_copy_frag = a_frag
            b_copy_frag = b_frag
        acc = cute.make_rmem_tensor(
            tiled_mma.partition_shape_C((tile_extent_m, tile_extent_n)),
            cutlass.Float32,
        )
        acc.fill(0.0)
        if const_expr(self.direct_ab_tma_layout == "unpack"):
            copy_atom_a = cute.make_copy_atom(
                warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4, unpack_bits=4),
                cutlass.Uint8,
            )
            copy_atom_b = cute.make_copy_atom(
                warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4, unpack_bits=4),
                cutlass.Uint8,
            )
        else:
            copy_atom_a = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.Float4E2M1FN,
            )
            copy_atom_b = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.Float4E2M1FN,
            )
        tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma)
        tiled_copy_b = cute.make_tiled_copy_B(copy_atom_b, tiled_mma)
        thr_copy_a = tiled_copy_a.get_slice(tidx)
        thr_copy_b = tiled_copy_b.get_slice(tidx)
        tCsA = thr_copy_a.partition_S(cute.as_position_independent_swizzle_tensor(sA_consumer))
        tCsB = thr_copy_b.partition_S(cute.as_position_independent_swizzle_tensor(sB_consumer))
        if const_expr(self.direct_ab_tma_layout == "unpack"):
            tCsA = cute.make_tensor(tCsA.iterator.align(16), tCsA.layout)
            tCsB = cute.make_tensor(tCsB.iterator.align(16), tCsB.layout)
        tCrA = thr_copy_a.retile(a_copy_frag)
        tCrB = thr_copy_b.retile(b_copy_frag)

        sSFA_f8 = cute.recast_tensor(sSFA, cutlass.Float8E4M3FN)
        sSFB_f8 = cute.recast_tensor(sSFB, cutlass.Float8E4M3FN)
        sfa_frag, sfb_frag = _sm120.make_mxf4nvf4_direct_tma_scale_fragments(
            tiled_mma,
            sSFA,
            sSFB,
            tidx,
            tile_shape_mnk=(tile_extent_m, tile_extent_n, tile_extent_k),
            sf_vec_size=16,
        )
        scale_stage_thread_idx = None
        scale_stage_thread_count = cute.arch.WARP_SIZE
        scale_barrier_id = Int32(8)

        stage = consumer_state_mk.index
        if const_expr(self.direct_try_wait_before_pingpong_barrier):
            self._split_tma_consumer_wait_with_tokens(
                pipe_mk,
                pipe_nk,
                consumer_state_mk,
                consumer_state_nk,
                initial_try_wait_mk,
                initial_try_wait_nk,
            )
        else:
            self._split_tma_consumer_wait(pipe_mk, pipe_nk, consumer_state_mk, consumer_state_nk)

        if const_expr(self.direct_scale_prefetch_first):
            self._blockscaled_load_scale_fragments(
                tiled_mma,
                sSFA,
                sSFB,
                sfa_frag,
                sfb_frag,
                tidx,
                0,
                stage,
                tile_extent_m,
                tile_extent_n,
                tile_extent_k,
            )
        cute.copy(
            tiled_copy_a,
            tCsA[(None, None, 0, stage)],
            tCrA[(None, None, 0)],
        )
        cute.copy(
            tiled_copy_b,
            tCsB[(None, None, 0, stage)],
            tCrB[(None, None, 0)],
        )
        self._shift_unpack_ab_fragments(tCrA, tCrB, 0)
        if const_expr(not self.direct_scale_prefetch_first):
            self._blockscaled_load_scale_fragments(
                tiled_mma,
                sSFA,
                sSFB,
                sfa_frag,
                sfb_frag,
                tidx,
                0,
                stage,
                tile_extent_m,
                tile_extent_n,
                tile_extent_k,
            )

        for _k_tile in cutlass.range(k_tile_count - 1, unroll=1):
            for k_block_idx in cutlass.range_constexpr(2):
                k_block_next = 0 if const_expr(k_block_idx == 1) else 1
                if const_expr(k_block_idx == 1):
                    if const_expr(self.direct_kblock_barrier):
                        self._direct_kblock_barrier()
                    self._split_tma_consumer_release(
                        pipe_mk, pipe_nk, consumer_state_mk, consumer_state_nk
                    )
                    consumer_state_mk.advance()
                    consumer_state_nk.advance()
                    stage = consumer_state_mk.index
                    self._split_tma_consumer_wait(
                        pipe_mk, pipe_nk, consumer_state_mk, consumer_state_nk
                    )
                if const_expr(self.direct_scale_prefetch_first):
                    self._blockscaled_load_scale_fragments(
                        tiled_mma,
                        sSFA,
                        sSFB,
                        sfa_frag,
                        sfb_frag,
                        tidx,
                        k_block_next,
                        stage,
                        tile_extent_m,
                        tile_extent_n,
                        tile_extent_k,
                    )
                cute.copy(
                    tiled_copy_a,
                    tCsA[(None, None, k_block_next, stage)],
                    tCrA[(None, None, k_block_next)],
                )
                cute.copy(
                    tiled_copy_b,
                    tCsB[(None, None, k_block_next, stage)],
                    tCrB[(None, None, k_block_next)],
                )
                self._shift_unpack_ab_fragments(tCrA, tCrB, k_block_next)
                if const_expr(not self.direct_scale_prefetch_first):
                    self._blockscaled_load_scale_fragments(
                        tiled_mma,
                        sSFA,
                        sSFB,
                        sfa_frag,
                        sfb_frag,
                        tidx,
                        k_block_next,
                        stage,
                        tile_extent_m,
                        tile_extent_n,
                        tile_extent_k,
                    )
                self._blockscaled_direct_gemm_k_block(
                    tiled_mma,
                    acc,
                    a_frag,
                    b_frag,
                    sfa_frag,
                    sfb_frag,
                    k_block_idx,
                )
        for k_block_idx in cutlass.range_constexpr(2):
            k_block_next = 0 if const_expr(k_block_idx == 1) else 1
            if const_expr(k_block_idx == 1):
                if const_expr(self.direct_kblock_barrier):
                    self._direct_kblock_barrier()
                self._split_tma_consumer_release(
                    pipe_mk, pipe_nk, consumer_state_mk, consumer_state_nk
                )
                consumer_state_mk.advance()
                consumer_state_nk.advance()
            if const_expr(k_block_idx == 0):
                if const_expr(self.direct_scale_prefetch_first):
                    self._blockscaled_load_scale_fragments(
                        tiled_mma,
                        sSFA,
                        sSFB,
                        sfa_frag,
                        sfb_frag,
                        tidx,
                        k_block_next,
                        stage,
                        tile_extent_m,
                        tile_extent_n,
                        tile_extent_k,
                    )
                cute.copy(
                    tiled_copy_a,
                    tCsA[(None, None, k_block_next, stage)],
                    tCrA[(None, None, k_block_next)],
                )
                cute.copy(
                    tiled_copy_b,
                    tCsB[(None, None, k_block_next, stage)],
                    tCrB[(None, None, k_block_next)],
                )
                self._shift_unpack_ab_fragments(tCrA, tCrB, k_block_next)
                if const_expr(not self.direct_scale_prefetch_first):
                    self._blockscaled_load_scale_fragments(
                        tiled_mma,
                        sSFA,
                        sSFB,
                        sfa_frag,
                        sfb_frag,
                        tidx,
                        k_block_next,
                        stage,
                        tile_extent_m,
                        tile_extent_n,
                        tile_extent_k,
                    )
            self._blockscaled_direct_gemm_k_block(
                tiled_mma,
                acc,
                a_frag,
                b_frag,
                sfa_frag,
                sfb_frag,
                k_block_idx,
            )
        if const_expr(self.direct_global_store or self.direct_global_store_probe):
            self._blockscaled_store_full_tile_direct_global(
                tiled_mma,
                acc,
                gD,
                tile_mnl,
                tidx,
                warp_group_idx,
            )
            return consumer_state_mk, consumer_state_nk
        if const_expr(self.direct_pingpong_split_tiles and self.direct_pingpong_barriers):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
            self.pingpong_barrier_sync(warp_group_idx, stage="epi")
        if const_expr(self.direct_delay_tma_store):
            if warp_idx == 0:
                epi_store_pipeline.producer_acquire()
        for epi_tile_idx in cutlass.range_constexpr(self.direct_epi_tiles):
            epi_m = epi_tile_idx // self.direct_epi_n_tiles
            epi_n = epi_tile_idx - epi_m * self.direct_epi_n_tiles
            epi_stage_idx = epi_tile_idx % self.epi_stage
            prev_epi_tile_idx = epi_tile_idx - 1
            prev_epi_m = prev_epi_tile_idx // self.direct_epi_n_tiles
            prev_epi_n = prev_epi_tile_idx - prev_epi_m * self.direct_epi_n_tiles
            prev_epi_stage_idx = prev_epi_tile_idx % self.epi_stage
            acc_epi_m = epi_m
            pingpong_epi_tile = warp_group_idx
            if const_expr(self.direct_delay_tma_store and epi_tile_idx != 0):
                if warp_idx == 0:
                    self._copy_blockscaled_epilogue_tma_store(
                        tma_atom_d,
                        tDsD,
                        tDgD,
                        prev_epi_m,
                        prev_epi_n,
                        prev_epi_stage_idx,
                    )
                    epi_store_pipeline.producer_commit()
                    epi_store_pipeline.producer_acquire()
                self.epilogue_barrier.arrive_and_wait()
            elif const_expr(not self.direct_delay_tma_store) and warp_idx == 0:
                epi_store_pipeline.producer_acquire()
            if const_expr(not self.direct_epi_barrier_trim):
                self.epilogue_barrier.arrive_and_wait()
            if const_expr((not self.pingpong) or self.direct_pingpong_split_tiles):
                self._blockscaled_stage_full_tile_to_epi_smem(
                    tiled_mma,
                    epilogue_params,
                    sD_epi,
                    acc,
                    tidx,
                    acc_epi_m,
                    epi_n,
                    epi_stage_idx,
                )
            elif const_expr(epi_tile_idx == 0):
                if pingpong_epi_tile == 0:
                    self._blockscaled_stage_full_tile_to_epi_smem(
                        tiled_mma,
                        epilogue_params,
                        sD_epi,
                        acc,
                        tidx,
                        acc_epi_m,
                        epi_n,
                        epi_stage_idx,
                    )
            else:
                if pingpong_epi_tile == 1:
                    self._blockscaled_stage_full_tile_to_epi_smem(
                        tiled_mma,
                        epilogue_params,
                        sD_epi,
                        acc,
                        tidx,
                        acc_epi_m,
                        epi_n,
                        epi_stage_idx,
                    )
            cute.arch.fence_view_async_shared()
            self.epilogue_barrier.arrive_and_wait()
            if const_expr(not self.direct_delay_tma_store):
                if warp_idx == 0:
                    self._copy_blockscaled_epilogue_tma_store(
                        tma_atom_d,
                        tDsD,
                        tDgD,
                        epi_m,
                        epi_n,
                        0,
                    )
                    epi_store_pipeline.producer_commit()
                self.epilogue_barrier.arrive_and_wait()
        if const_expr(self.direct_delay_tma_store):
            last_epi_tile_idx = self.direct_epi_tiles - 1
            last_epi_m = last_epi_tile_idx // self.direct_epi_n_tiles
            last_epi_n = last_epi_tile_idx - last_epi_m * self.direct_epi_n_tiles
            last_epi_stage_idx = last_epi_tile_idx % self.epi_stage
            if warp_idx == 0:
                self._copy_blockscaled_epilogue_tma_store(
                    tma_atom_d,
                    tDsD,
                    tDgD,
                    last_epi_m,
                    last_epi_n,
                    last_epi_stage_idx,
                )
                epi_store_pipeline.producer_commit()
            self.epilogue_barrier.arrive_and_wait()
        if warp_idx == 0:
            epi_store_pipeline.producer_tail()
        if const_expr(self.direct_pingpong_split_tiles and self.direct_pingpong_barriers):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
        return consumer_state_mk, consumer_state_nk

    @cute.jit
    def _shift_unpack_ab_fragments(
        self,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        k_block_idx: cutlass.Constexpr[int],
    ) -> None:
        if const_expr(self.direct_ab_tma_layout == "unpack" and self.direct_unpack_shift):
            _sm120.shift_mxf4nvf4_post_ldsm_fp4_fragment(tCrA[(None, None, k_block_idx)])
            _sm120.shift_mxf4nvf4_post_ldsm_fp4_fragment(tCrB[(None, None, k_block_idx)])

    @cute.jit
    def _blockscaled_load_scale_fragments(
        self,
        tiled_mma: cute.TiledMma,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sfa_frag: cute.Tensor,
        sfb_frag: cute.Tensor,
        tidx: Int32,
        k_block_idx: cutlass.Constexpr[int],
        stage: Int32,
        tile_extent_m: cutlass.Constexpr[int],
        tile_extent_n: cutlass.Constexpr[int],
        tile_extent_k: cutlass.Constexpr[int],
    ) -> None:
        _sm120.copy_mxf4nvf4_direct_tma_scale_fragments(
            tiled_mma,
            sSFA,
            sSFB,
            sfa_frag,
            sfb_frag,
            tidx,
            k_block_idx,
            stage_idx=stage,
            major_extent_sfa=tile_extent_m,
            major_extent_sfb=tile_extent_n,
            tile_k=tile_extent_k,
            sf_vec_size=16,
            scale_smem_format=self.direct_scale_smem_format,
        )

    @cute.jit
    def _blockscaled_load_direct_k_block_a_scale_fragments(
        self,
        tiled_mma: cute.TiledMma,
        tiled_copy_a: cute.TiledCopy,
        tCsA: cute.Tensor,
        tCrA: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sfa_frag: cute.Tensor,
        sfb_frag: cute.Tensor,
        tidx: Int32,
        k_block_idx: cutlass.Constexpr[int],
        stage: Int32,
        tile_extent_m: cutlass.Constexpr[int],
        tile_extent_n: cutlass.Constexpr[int],
        tile_extent_k: cutlass.Constexpr[int],
    ) -> None:
        _sm120.load_mxf4nvf4_direct_tma_k_block_a_scale_fragments(
            tiled_mma,
            tiled_copy_a,
            tCsA,
            tCrA,
            sSFA,
            sSFB,
            sfa_frag,
            sfb_frag,
            tidx,
            k_block_idx,
            stage_idx=stage,
            major_extent_sfa=tile_extent_m,
            major_extent_sfb=tile_extent_n,
            tile_k=tile_extent_k,
            sf_vec_size=16,
            scale_first=self.direct_scale_prefetch_first,
        )

    @cute.jit
    def _blockscaled_load_direct_k_block_b_group_fragment(
        self,
        tiled_copy_b: cute.TiledCopy,
        tCsB: cute.Tensor,
        tCrB: cute.Tensor,
        k_block_idx: cutlass.Constexpr[int],
        b_group_idx: cutlass.Constexpr[int],
        stage: Int32,
    ) -> None:
        _sm120.load_mxf4nvf4_direct_tma_k_block_b_group_fragment(
            tiled_copy_b,
            tCsB,
            tCrB,
            k_block_idx,
            b_group_idx,
            stage_idx=stage,
        )

    @cute.jit
    def _blockscaled_direct_gemm_k_block_b_group(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        a_frag: cute.Tensor,
        b_frag: cute.Tensor,
        sfa_frag: cute.Tensor,
        sfb_frag: cute.Tensor,
        k_block_idx: cutlass.Constexpr[int],
        b_group_idx: cutlass.Constexpr[int],
    ) -> None:
        _sm120.gemm_mxf4nvf4_direct_tma_k_block_b_group(
            tiled_mma,
            acc,
            a_frag,
            b_frag,
            sfa_frag,
            sfb_frag,
            k_block_idx,
            b_group_idx,
        )

    @cute.jit
    def _blockscaled_direct_gemm_k_block_bgroup_pipeline(
        self,
        tiled_mma: cute.TiledMma,
        tiled_copy_b: cute.TiledCopy,
        acc: cute.Tensor,
        tCsB: cute.Tensor,
        tCrB: cute.Tensor,
        a_frag: cute.Tensor,
        b_frag: cute.Tensor,
        sfa_frag: cute.Tensor,
        sfb_frag: cute.Tensor,
        k_block_idx: cutlass.Constexpr[int],
        stage: Int32,
    ) -> None:
        self._blockscaled_direct_gemm_k_block_b_group(
            tiled_mma,
            acc,
            a_frag,
            b_frag,
            sfa_frag,
            sfb_frag,
            k_block_idx,
            0,
        )
        self._blockscaled_load_direct_k_block_b_group_fragment(
            tiled_copy_b,
            tCsB,
            tCrB,
            k_block_idx,
            2,
            stage,
        )
        self._blockscaled_direct_gemm_k_block_b_group(
            tiled_mma,
            acc,
            a_frag,
            b_frag,
            sfa_frag,
            sfb_frag,
            k_block_idx,
            1,
        )
        self._blockscaled_load_direct_k_block_b_group_fragment(
            tiled_copy_b,
            tCsB,
            tCrB,
            k_block_idx,
            3,
            stage,
        )
        for b_group_idx in cutlass.range_constexpr(2, 4):
            self._blockscaled_direct_gemm_k_block_b_group(
                tiled_mma,
                acc,
                a_frag,
                b_frag,
                sfa_frag,
                sfb_frag,
                k_block_idx,
                b_group_idx,
            )

    @cute.jit
    def _blockscaled_direct_gemm_k_block(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        a_frag: cute.Tensor,
        b_frag: cute.Tensor,
        sfa_frag: cute.Tensor,
        sfb_frag: cute.Tensor,
        k_block_idx: cutlass.Constexpr[int],
    ) -> None:
        _sm120.gemm_mxf4nvf4_direct_tma_k_block(
            tiled_mma,
            acc,
            a_frag,
            b_frag,
            sfa_frag,
            sfb_frag,
            k_block_idx,
            ab_smem_format=self.direct_ab_tma_layout,
            n_major=self.direct_mma_n_major,
            sync_warp_before=self.direct_pre_mma_warp_sync,
        )

    @cute.jit
    def _blockscaled_stage_full_tile_to_epi_smem(
        self,
        tiled_mma: cute.TiledMma,
        epilogue_params,
        sD_epi: cute.Tensor,
        acc: cute.Tensor,
        tidx: Int32,
        epi_m: cutlass.Constexpr[int] = 0,
        epi_n: cutlass.Constexpr[int] = 0,
        epi_stage_idx: cutlass.Constexpr[int] = 0,
    ) -> None:
        sD_tile = sD_epi[(None, None, epi_stage_idx)]
        if const_expr(self.direct_epi_m_tiles == 1 and self.direct_epi_n_tiles == 1):
            rD_acc = cute.make_rmem_tensor(acc.shape, cutlass.Float32)
            rD_acc.store(acc.load())
            self.epi_visit_subtile(
                epilogue_params,
                {
                    "alpha": None,
                    "beta": None,
                    "mRowVecBroadcast": None,
                    "mColVecBroadcast": None,
                },
                rD_acc,
                None,
            )
            thr_mma = tiled_mma.get_slice(tidx)
            tCsD = thr_mma.partition_C(sD_tile)
            rD = cute.make_rmem_tensor(acc.shape, cutlass.BFloat16)
            rD.store(rD_acc.load().to(cutlass.BFloat16))
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16)
            cute.copy(atom, rD, tCsD)
            return

        tiled_copy_r2s, tRS_rD, tRS_sD, tRS_rAcc = _sm120.make_mxf4nvf4_epilogue_stmatrix_views(
            tiled_mma,
            acc,
            sD_tile,
            tidx,
            epi_tile_shape=self.epi_tile_shape,
            num_matrices=self.direct_epi_stsm_matrices,
        )
        _sm120.load_mxf4nvf4_accumulator_epilogue_subtile(
            tRS_rAcc,
            tRS_rD,
            (epi_m, epi_n),
        )
        self.epi_visit_subtile(
            epilogue_params,
            {
                "alpha": None,
                "beta": None,
                "mRowVecBroadcast": None,
                "mColVecBroadcast": None,
            },
            tRS_rD,
            None,
        )
        _sm120.copy_mxf4nvf4_epilogue_registers_to_smem(
            tiled_copy_r2s,
            tRS_rD,
            tRS_sD,
        )

    @cute.jit
    def _blockscaled_compute_store_mn_group_k_loop(
        self,
        tiled_mma: cute.TiledMma,
        pipe,
        sA_consumer: cute.Tensor,
        sB_consumer: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        epilogue_params,
        sD_epi: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        tDsD: cute.Tensor,
        tDgD: cute.Tensor,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        m_group: cutlass.Constexpr[int],
        n_group: cutlass.Constexpr[int],
        lane_idx: Int32,
        k_tile_count: Int32,
        n_atoms: cutlass.Constexpr[int],
        tile_extent_m: cutlass.Constexpr[int],
        tile_extent_n: cutlass.Constexpr[int],
        tile_extent_k: cutlass.Constexpr[int],
    ) -> None:
        n_group_count = 2
        n_atoms_per_group = n_atoms // n_group_count
        accs = [
            [
                cute.make_rmem_tensor(tiled_mma.partition_shape_C((16, 8)), cutlass.Float32)
                for _ in range(n_atoms_per_group)
            ]
            for _ in range(2)
        ]
        for m_offset in cutlass.range_constexpr(2):
            for n_atom_local in cutlass.range_constexpr(n_atoms_per_group):
                accs[m_offset][n_atom_local].fill(0.0)

        consumer_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
        for _k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = consumer_state.index
            pipe.consumer_wait(consumer_state, pipe.consumer_try_wait(consumer_state))
            for m_offset in cutlass.range_constexpr(2):
                m_atom = m_group + 4 * m_offset
                for n_atom_local in cutlass.range_constexpr(n_atoms_per_group):
                    n_atom = n_atom_local * n_group_count + n_group
                    self._blockscaled_mma_n_atom(
                        tiled_mma,
                        sA_consumer,
                        sB_consumer,
                        sSFA,
                        sSFB,
                        accs[m_offset][n_atom_local],
                        m_atom,
                        n_atom,
                        lane_idx,
                        stage,
                        tile_extent_m,
                        tile_extent_n,
                        tile_extent_k,
                    )
            self._direct_kblock_barrier()
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            pipe.consumer_release(consumer_state)
            consumer_state.advance()

        if const_expr(m_group == 0 and n_group == 0):
            epi_store_pipeline.producer_acquire()
        self.epilogue_barrier.arrive_and_wait()
        for m_offset in cutlass.range_constexpr(2):
            m_atom = m_group + 4 * m_offset
            for n_atom_local in cutlass.range_constexpr(n_atoms_per_group):
                n_atom = n_atom_local * n_group_count + n_group
                self._blockscaled_stage_n_atom_to_epi_smem(
                    tiled_mma,
                    epilogue_params,
                    sD_epi,
                    accs[m_offset][n_atom_local],
                    m_atom,
                    n_atom,
                    lane_idx,
                )
        cute.arch.fence_view_async_shared()
        self.epilogue_barrier.arrive_and_wait()
        if const_expr(m_group == 0 and n_group == 0):
            cute.copy(tma_atom_d, tDsD[None, 0], tDgD[None, 0, 0])
            epi_store_pipeline.producer_commit()
        self.epilogue_barrier.arrive_and_wait()
        if const_expr(m_group == 0 and n_group == 0):
            epi_store_pipeline.producer_tail()

    @cute.jit
    def _blockscaled_compute_store_m_atom_k_loop(
        self,
        tiled_mma: cute.TiledMma,
        pipe,
        sA_consumer: cute.Tensor,
        sB_consumer: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        epilogue_params,
        sD_epi: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        tDsD: cute.Tensor,
        tDgD: cute.Tensor,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        m_atom: cutlass.Constexpr[int],
        lane_idx: Int32,
        k_tile_count: Int32,
        n_atoms: cutlass.Constexpr[int],
        tile_extent_m: cutlass.Constexpr[int],
        tile_extent_n: cutlass.Constexpr[int],
        tile_extent_k: cutlass.Constexpr[int],
        n_group: cutlass.Constexpr[int],
        n_group_count: cutlass.Constexpr[int],
        do_tma_store: cutlass.Constexpr[bool],
    ) -> None:
        n_atoms_per_group = n_atoms // n_group_count
        accs = [
            cute.make_rmem_tensor(tiled_mma.partition_shape_C((16, 8)), cutlass.Float32)
            for _ in range(n_atoms_per_group)
        ]
        for n_atom_local in cutlass.range_constexpr(n_atoms_per_group):
            accs[n_atom_local].fill(0.0)

        consumer_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
        for _k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = consumer_state.index
            pipe.consumer_wait(consumer_state, pipe.consumer_try_wait(consumer_state))
            for n_atom_local in cutlass.range_constexpr(n_atoms_per_group):
                n_atom = n_atom_local * n_group_count + n_group
                self._blockscaled_mma_n_atom(
                    tiled_mma,
                    sA_consumer,
                    sB_consumer,
                    sSFA,
                    sSFB,
                    accs[n_atom_local],
                    m_atom,
                    n_atom,
                    lane_idx,
                    stage,
                    tile_extent_m,
                    tile_extent_n,
                    tile_extent_k,
                )
            self._direct_kblock_barrier()
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            pipe.consumer_release(consumer_state)
            consumer_state.advance()

        if const_expr(do_tma_store):
            epi_store_pipeline.producer_acquire()
        self.epilogue_barrier.arrive_and_wait()
        for n_atom_local in cutlass.range_constexpr(n_atoms_per_group):
            n_atom = n_atom_local * n_group_count + n_group
            self._blockscaled_stage_n_atom_to_epi_smem(
                tiled_mma,
                epilogue_params,
                sD_epi,
                accs[n_atom_local],
                m_atom,
                n_atom,
                lane_idx,
            )
        cute.arch.fence_view_async_shared()
        self.epilogue_barrier.arrive_and_wait()
        if const_expr(do_tma_store):
            cute.copy(tma_atom_d, tDsD[None, 0], tDgD[None, 0, 0])
            epi_store_pipeline.producer_commit()
        self.epilogue_barrier.arrive_and_wait()
        if const_expr(do_tma_store):
            epi_store_pipeline.producer_tail()

    @cute.jit
    def _blockscaled_mma_n_atom(
        self,
        tiled_mma: cute.TiledMma,
        sA_consumer: cute.Tensor,
        sB_consumer: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        acc: cute.Tensor,
        m_atom: cutlass.Constexpr[int],
        n_atom: cutlass.Constexpr[int],
        lane_idx: Int32,
        consumer_stage: Int32,
        tile_extent_m: cutlass.Constexpr[int],
        tile_extent_n: cutlass.Constexpr[int],
        tile_extent_k: cutlass.Constexpr[int],
    ) -> None:
        sA_atom, sB_atom = _sm120.make_mxf4nvf4_ab_consumer_microtile_views(
            sA_consumer,
            sB_consumer,
            m_atom=m_atom,
            n_atom=n_atom,
        )
        a_frag, b_frag = _sm120.make_mxf4nvf4_ab_fragments_from_consumer_smem(
            tiled_mma,
            sA_atom,
            sB_atom,
            lane_idx=lane_idx,
        )
        sfa, sfb = _sm120.make_mxf4nvf4_scale_fragments()
        sSFA_atom = cute.make_tensor(
            sSFA.iterator + cutlass.Int32(m_atom * 16),
            sSFA.layout,
        )
        sSFB_atom = cute.make_tensor(
            sSFB.iterator + cutlass.Int32(n_atom * 8),
            sSFB.layout,
        )
        for k_block_idx in cutlass.range_constexpr(2):
            _sm120.load_mxf4nvf4_ab_fragments_from_consumer_smem(
                tiled_mma,
                sA_atom,
                sB_atom,
                a_frag,
                b_frag,
                lane_idx,
                k_block_idx,
                consumer_stage_idx=consumer_stage,
            )
            sfa_src, sfb_src = _sm120.make_mxf4nvf4_scale_fragment_views_from_direct_tma(
                sSFA_atom,
                sSFB_atom,
                k_block_idx,
                stage_idx=consumer_stage,
                major_extent_sfa=tile_extent_m,
                major_extent_sfb=tile_extent_n,
                tile_k=tile_extent_k,
                sf_vec_size=16,
            )
            _sm120.load_mxf4nvf4_sfa_fragment(sfa_src, sfa)
            _sm120.load_mxf4nvf4_sfb_fragment(sfb_src, sfb)
            cute.gemm(
                tiled_mma,
                acc,
                (a_frag[(None, 0, k_block_idx)], sfa),
                (b_frag[(None, 0, k_block_idx)], sfb),
                acc,
            )

    @cute.jit
    def _blockscaled_stage_n_atom_to_epi_smem(
        self,
        tiled_mma: cute.TiledMma,
        epilogue_params,
        sD_epi: cute.Tensor,
        acc: cute.Tensor,
        epi_m_atom: cutlass.Constexpr[int],
        epi_n_atom: cutlass.Constexpr[int],
        lane_idx: Int32,
    ) -> None:
        sD_tile = sD_epi[(None, None, 0)]
        sD_atom = cute.local_tile(sD_tile, (16, 8), (epi_m_atom, epi_n_atom))
        thr_mma = tiled_mma.get_slice(lane_idx)
        tCsD = thr_mma.partition_C(sD_atom)
        rD_acc = cute.make_rmem_tensor(acc.shape, cutlass.Float32)
        rD_acc.store(acc.load())
        self.epi_visit_subtile(
            epilogue_params,
            {
                "alpha": None,
                "beta": None,
                "mRowVecBroadcast": None,
                "mColVecBroadcast": None,
            },
            rD_acc,
            None,
        )
        rD = cute.make_rmem_tensor(acc.shape, cutlass.BFloat16)
        rD.store(rD_acc.load().to(cutlass.BFloat16))
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16)
        cute.copy(atom, rD, tCsD)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: Optional[cute.CopyAtom],
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: Optional[cute.Tensor],
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        epilogue_params,
        varlen_params: VarlenManager.Params,
        cluster_layout_mnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_c_smem_layout: cute.ComposedLayout,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        trace_ptr: Optional[cutlass.Int64] = None,
    ):
        from quack.trace import TraceContext

        tctx = TraceContext.create(trace_ptr)

        varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
        varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
        if const_expr(self.gather_A):
            assert varlen_m or varlen_k
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptors
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
        )
        epi_pipeline = None
        has_epi_load = const_expr(self.epi_c_stage > 0)
        if const_expr(has_epi_load):
            epi_pipeline = self.make_epi_pipeline(
                epi_pipeline_mbar_ptr=storage.epi_pipeline_array_ptr.data_ptr(),
                tx_count=self.epi_load_bytes_per_stage,
            )
        sched_pipeline = None
        sched_data = None
        if const_expr(self.is_persistent):
            sched_pipeline = self.make_sched_pipeline(
                cluster_layout_mnk,
                sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
                varlen_k=varlen_k,
            )
            sched_data = storage.sched_data.get_tensor((4, self.sched_stage))

        # Cluster sync
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)

        # SMEM tensors
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = None
        if const_expr(has_D):
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sC = None
        if const_expr(has_C):
            sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

        varlen_manager = VarlenManager.create(
            varlen_params,
            len_m_static=Int32(
                cute.size(mA_mkl, mode=[0])
                if varlen_k or varlen_params.mAIdx is None
                else varlen_params.mAIdx.shape[0]
            ),
            len_k_static=Int32(cute.size(mA_mkl, mode=[1])),
        )

        TileSchedulerCls = partial(
            TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline
        )

        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if (
                warp_idx >= self.ab_load_warp_id
                and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
                # Get mcast mask
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1
                )
                b_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                # Persistent tile scheduling loop
                is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                if const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                while work_tile.is_valid_tile:
                    tctx.b("tma_load")
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    # Local_tile partition global tensors
                    copy_A, prefetch_A = None, None
                    if const_expr(not self.gather_A):
                        mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                        # (bM, bK, RestK)
                        gA_mk = cute.local_tile(
                            mA_mk,
                            cute.select(self.cta_tile_shape_mnk, [0, 2]),
                            (tile_coord_mnkl[0], None),
                        )
                        #  TMA load A partition_S/D
                        copy_A, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_a,
                            cta_coord=block_in_cluster_coord_mnk[1],
                            cta_layout=cute.make_layout(
                                cute.slice_(cluster_layout_mnk, (0, None, 0)).shape
                            ),
                            src_tensor=gA_mk,
                            dst_tensor=sA,
                            mcast_mask=a_mcast_mask,
                        )
                    else:
                        copy_A, prefetch_A = self._make_gather_A_copy(
                            mA_mkl, sA, varlen_manager, tile_coord_mnkl, batch_idx
                        )
                    # (bN, bK, RestK)
                    gB_nk = cute.local_tile(
                        varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                        cute.select(self.cta_tile_shape_mnk, [1, 2]),
                        (tile_coord_mnkl[1], None),
                    )
                    # TMA load B partition_S/D
                    copy_B, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_b,
                        cta_coord=block_in_cluster_coord_mnk[0],
                        cta_layout=cute.make_layout(
                            cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape
                        ),
                        src_tensor=gB_nk,
                        dst_tensor=sB,
                        mcast_mask=b_mcast_mask,
                    )
                    len_k = varlen_manager.len_k(batch_idx)
                    k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    if const_expr(not self.gather_A):
                        ab_producer_state = self.load_tma(
                            ab_pipeline, ab_producer_state, [copy_A, copy_B], k_tile_cnt
                        )
                    else:
                        ab_producer_state = self.load_AB_gather_A(
                            ab_pipeline,
                            ab_producer_state,
                            copy_A,
                            prefetch_A,
                            copy_B,
                            k_tile_cnt,
                            varlen_m=varlen_m,
                        )
                    tctx.e("tma_load")
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                if const_expr(self.pingpong and not varlen_k):
                    # Need to write the tile_idx to smem for the next WG in the pingpong mode
                    if is_scheduler_warp:
                        tile_scheduler.write_work_tile_to_smem(work_tile)
                    work_tile = tile_scheduler.get_current_work()
                ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

        # =====================================================================
        # MMA warps
        # =====================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            is_tma_warp = Boolean(
                (not self.pingpong and warp_idx == 0)
                or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            tidx, _, _ = cute.arch.thread_idx()
            # For pingpong, adjust tidx to within-warp-group index
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group

            # ldmatrix copy atoms for SMEM → RMEM
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)
            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)

            # Make fragments
            thr_mma = tiled_mma.get_slice(tidx)
            acc, tCsA, tCsB, tCrA, tCrB = sm80_utils.partition_fragment_ABC(
                thr_mma, self.cta_tile_shape_mnk, sA, sB
            )

            if const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # WG0 needs a start signal at the very beginning
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            k_tile_cnt_static = cute.ceil_div(
                cute.size(mA_mkl, mode=[1]), self.cta_tile_shape_mnk[2]
            )
            c_tile_cnt = cute.size(cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile))

            ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            epi_store_pipeline = self.make_epi_store_pipeline()
            epi_read_state = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_c_stage
            )
            epi_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.epi_c_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()

            if const_expr(self.pingpong):
                if warp_idx >= 4:
                    # Advance 2nd Math WG pipeline states to the end of 1st Math WG
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    if const_expr(not varlen_k):
                        ab_read_state.advance_iters(k_tile_cnt_static)
                    else:
                        len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                        k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        ab_read_state.advance_iters(k_tile_cnt)
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                len_k = varlen_manager.len_k(batch_idx)
                k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                acc.fill(0.0)
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                tctx.b("mma")
                ab_read_state = self.mma(
                    ab_pipeline,
                    ab_read_state,
                    tiled_mma,
                    acc,
                    k_tile_cnt,
                    smem_tiled_copy_A,
                    smem_tiled_copy_B,
                    tCsA_copy_view,
                    tCsB_copy_view,
                    tCrA,
                    tCrB,
                )
                if const_expr(self.pingpong):
                    # Cue for next WG's MMA to start
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
                tctx.e("mma")

                # ============================================================
                # EPILOGUE — reuse SM90's epilogue flow
                # ============================================================
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")
                tctx.b("epilogue")

                copy_D = None
                if const_expr(has_D):
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d,
                        varlen_manager.offset_batch_epi(mD_mnl, tile_coord_mnkl[3]),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sD,
                        tile_coord_mnkl,
                    )
                copy_C = None
                if const_expr(has_C):
                    copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_c,
                        varlen_manager.offset_batch_epi(mC_mnl, tile_coord_mnkl[3]),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sC,
                        tile_coord_mnkl,
                    )
                    copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)
                if const_expr(has_epi_load):
                    tile_load_copy_fns = self.epi_tile_load_g2s_copy_fns(
                        epilogue_params,
                        epi_smem_tensors,
                        tile_coord_mnkl,
                        varlen_manager,
                        epi_pipeline,
                    )
                    copy_C = copy_utils.chain_tma_producer_copy_fns((copy_C, *tile_load_copy_fns))

                d_dtype_for_layout = self.d_dtype if self.d_dtype is not None else cutlass.BFloat16
                tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                    tiled_mma, self.d_layout, d_dtype_for_layout, sD, tidx
                )
                # (R2S, R2S_M, R2S_N, (epi_M, epi_N))
                tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
                load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
                if const_expr(has_C):
                    tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                        tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
                    )
                else:
                    tiled_copy_s2r, tSR_sC, tRS_rC, tSR_rC = None, None, None, None

                self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)

                epi_read_state, epi_producer_state = self.epilogue(
                    epilogue_params,
                    epi_smem_tensors,
                    epi_pipeline,
                    epi_store_pipeline,
                    epi_read_state,
                    epi_producer_state,
                    self.epi_tile,
                    load_acc_subtile,
                    tRS_rD,
                    tRS_rC,
                    None,  # tiled_copy_t2r, for Sm100 only
                    tiled_copy_r2s,
                    tRS_sD,
                    tiled_copy_s2r,
                    tSR_rC,
                    tSR_sC,
                    copy_D,
                    copy_C,
                    tile_coord_mnkl,
                    varlen_manager,
                    self.epilogue_barrier,
                    tile_scheduler,
                    tidx,
                    is_tma_warp,
                )

                if const_expr(self.pingpong):
                    # With pingpong, 2 WGs write two different output tiles to the same smem,
                    # so we have to make sure the smem content is done reading before signaling
                    # the next WG's epilogue.
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
                tctx.e("epilogue")

                if const_expr(not self.pingpong):
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                else:  # Skip a tile for pingpong
                    # Update starting load/store pipeline states for the next tile
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    # Update starting mainloop pipeline state for the next tile
                    if const_expr(not varlen_k):
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                        if work_tile.is_valid_tile:
                            len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                            k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                            ab_read_state.advance_iters(k_tile_cnt)
                            tile_scheduler.advance_to_next_work()
                            work_tile = tile_scheduler.get_current_work()

            # Wait for D store complete
            if const_expr(not self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()

        tctx.flush()

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        k_tile_cnt: Int32,
        smem_tiled_copy_A: cute.TiledCopy,
        smem_tiled_copy_B: cute.TiledCopy,
        tCsA_copy_view: cute.Tensor,
        tCsB_copy_view: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        """Warp-level MMA mainloop: ldmatrix SMEM→RMEM + warp MMA."""
        tCrA_copy_view = smem_tiled_copy_A.retile(tCrA)
        tCrB_copy_view = smem_tiled_copy_B.retile(tCrB)
        load_sA = partial(cute.copy, smem_tiled_copy_A)
        load_sB = partial(cute.copy, smem_tiled_copy_B)

        num_k_blocks = cute.size(tCrA, mode=[2])
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)

        # Load first k-block
        tCsA_p = tCsA_copy_view[None, None, None, ab_read_state.index]
        tCsB_p = tCsB_copy_view[None, None, None, ab_read_state.index]
        load_sA(tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
        load_sB(tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])

        for k_tile in cutlass.range(k_tile_cnt - 1, unroll=1):
            for k in cutlass.range_constexpr(num_k_blocks):
                k_next = 0 if k + 1 == num_k_blocks else k + 1
                if const_expr(k == num_k_blocks - 1):
                    # TMA writes this smem stage through the async proxy, while ldmatrix
                    # reads it through the generic proxy. Fence before release so the
                    # producer's next async-proxy write cannot race those reads; sync the
                    # warp because only one lane signals the empty mbarrier.
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    ab_pipeline.consumer_release(ab_read_state)
                    ab_read_state.advance()
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
                    tCsA_p = tCsA_copy_view[None, None, None, ab_read_state.index]
                    tCsB_p = tCsB_copy_view[None, None, None, ab_read_state.index]
                    ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
                load_sA(tCsA_p[None, None, k_next], tCrA_copy_view[None, None, k_next])
                load_sB(tCsB_p[None, None, k_next], tCrB_copy_view[None, None, k_next])
                cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)

        # Last k-tile (hoisted)
        if 0 < k_tile_cnt:
            for k in cutlass.range_constexpr(num_k_blocks):
                k_next = 0 if k + 1 == num_k_blocks else k + 1
                if const_expr(k == num_k_blocks - 1):
                    # TMA writes this smem stage through the async proxy, while ldmatrix
                    # reads it through the generic proxy. Fence before release so the
                    # producer's next async-proxy write cannot race those reads; sync the
                    # warp because only one lane signals the empty mbarrier.
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    ab_pipeline.consumer_release(ab_read_state)
                    ab_read_state.advance()
                if const_expr(k_next > 0):
                    load_sA(tCsA_p[None, None, k_next], tCrA_copy_view[None, None, k_next])
                    load_sB(tCsB_p[None, None, k_next], tCrB_copy_view[None, None, k_next])
                cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)

        return ab_read_state

    @staticmethod
    def _compute_tile_shape_or_override(
        cta_tile_shape_mnk: Tuple[int, int, int],
        atom_layout_mnk: Tuple[int, int, int],
        element_type: Optional[Type[cutlass.Numeric]] = None,
        epi_tile_override: Tuple[int, int] | None = None,
    ) -> Tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param cta_tile_shape_mnk: CTA tile shape (M,N,K)
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        n_perf = 64 if element_type is not None and element_type.width == 8 else 32
        tile_m = math.gcd(64, cute.size(cta_tile_shape_mnk, mode=[0]))
        tile_n = math.gcd(n_perf, cute.size(cta_tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)
