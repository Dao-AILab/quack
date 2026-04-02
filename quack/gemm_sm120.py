# Copyright (c) 2025-2026, Tri Dao.
# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py
# SM120-style GEMM using warp-level MMA (MmaF16BF16Op) + ldmatrix.
# Unlike SM90 WGMMA (which reads A/B from SMEM directly), warp-level MMA
# requires explicit SMEM→RMEM copies via ldmatrix before each MMA instruction.

# This is a work in progress and not very optimized.

import math
from typing import Tuple, Type, Callable, Optional
from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warp
from cutlass import Int32, Boolean, const_expr

from quack.varlen_utils import VarlenManager
from quack.pipeline import make_pipeline_state
import quack.copy_utils as copy_utils
from quack.gemm_sm90 import GemmSm90, NamedBarrierGemm


class GemmSm120(GemmSm90):
    """SM120-style GEMM using warp-level MMA instead of WGMMA.

    Key differences from SM90:
    - Uses MmaF16BF16Op (warp-level, 32 threads) instead of WGMMA (warp-group, 128 threads)
    - Requires explicit SMEM→RMEM copy via ldmatrix before MMA
    - Thread config: num_mma_warps regular warps + 1 DMA warp
    - No pingpong support
    - No fp8 support (warp-level MMA only supports fp16/bf16)
    """

    arch = 120

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mn: Tuple[int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        is_persistent: bool = True,
        gather_A: bool = False,
    ):
        # Don't call super().__init__ — we set up our own config
        self.acc_dtype = acc_dtype
        self.pingpong = False
        self.is_persistent = is_persistent
        self.use_clc_persistence = False
        self.fp8_slow_accum = False
        self.gather_A = gather_A
        if gather_A:
            assert cluster_shape_mnk[1] == 1

        self.cluster_shape_mnk = cluster_shape_mnk
        tile_M, tile_N = tile_shape_mn
        self.cta_tile_shape_mnk = (tile_M, tile_N, 1)

        # Warp-level MMA uses (2, 2, 1) atom layout like the example
        self.atom_layout_mnk = (2, 2, 1)
        self.mma_inst_mnk = (16, 8, 16)
        self.num_mma_warps = math.prod(self.atom_layout_mnk)
        self.threads_per_cta = (self.num_mma_warps + 1) * cute.arch.WARP_SIZE
        # For compatibility with SM90 code that uses warp groups
        self.num_threads_per_warp_group = 128
        self.mma_warp_groups = 1

        self.num_mcast_ctas_a = cluster_shape_mnk[1]
        if gather_A:
            assert self.num_mcast_ctas_a == 1
        self.num_mcast_ctas_b = cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.occupancy = 1
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}")

        # All MMA warps participate in epilogue
        self.num_epi_warps = self.num_mma_warps
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.Epilogue),
            num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )
        self.num_ab_load_warps = 1
        self.ab_load_warp_id = self.num_mma_warps

        self.num_regs_load = 40
        self.num_regs_mma = 232

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def _setup_tiled_mma(self):
        """Set up warp-level MMA (MmaF16BF16Op) and tile K dimension."""
        op = warp.MmaF16BF16Op(self.a_dtype, self.acc_dtype, self.mma_inst_mnk)
        tC = cute.make_layout(self.atom_layout_mnk)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_mnk[0],
            self.atom_layout_mnk[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)
        tile_k = self.mma_inst_mnk[2] * 4
        self.cta_tile_shape_mnk = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1],
            tile_k,
        )

    # __call__, _setup_attributes, make_ab_pipeline, make_epi_store_pipeline,
    # make_sched_pipeline, epilogue are all inherited from GemmSm90.

    def epi_retile_acc(self, acc, tRS_rD, tiled_copy_r2s):
        """Retile accumulator for epilogue. Warp-level MMA uses tiled_copy_r2s.retile."""
        thr_copy_r2s = tiled_copy_r2s.get_slice(cute.arch.thread_idx()[0])
        self._epi_size_tRS_rD = cute.size(tRS_rD)
        return thr_copy_r2s.retile(acc)

    @cute.jit
    def epi_load_acc_subtile(self, tRS_rAcc, tRS_rD, epi_idx):
        """Load acc subtile using retile-based flat indexing (warp-level MMA layout)."""
        size_rD = self._epi_size_tRS_rD
        for i in cutlass.range_constexpr(size_rD):
            tRS_rD[i] = tRS_rAcc[epi_idx * size_rD + i]

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
        load_sB = partial(cute.copy, smem_tiled_copy_A)

        num_k_blocks = cute.size(tCrA, mode=[2])
        acc.fill(0.0)
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
                    # Don't need to sync_warp: the previous instruction was mma.sync from cute.gemm
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
                    ab_pipeline.consumer_release(ab_read_state)
                    ab_read_state.advance()
                if const_expr(k_next > 0):
                    load_sA(tCsA_p[None, None, k_next], tCrA_copy_view[None, None, k_next])
                    load_sB(tCsB_p[None, None, k_next], tCrB_copy_view[None, None, k_next])
                cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        return ab_read_state

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
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
        _trace_ptr: Optional[cutlass.Int64] = None, # TODO: unused
    ):
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(cluster_layout_mnk, cluster_coord_mnk, mode=1)
        b_mcast_mask = cute.make_layout_image_mask(cluster_layout_mnk, cluster_coord_mnk, mode=0)
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
        )
        epi_pipeline = None
        if const_expr(has_C):
            epi_pipeline = self.make_epi_pipeline(
                c_smem_layout=cute.slice_(epi_c_smem_layout, (None, None, 0)),
                epi_pipeline_mbar_ptr=storage.epi_pipeline_array_ptr.data_ptr(),
            )
        sched_pipeline = None
        sched_data = None
        if const_expr(self.is_persistent):
            sched_pipeline = self.make_sched_pipeline(
                cluster_layout_mnk,
                sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
                varlen_k=False,
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
            len_m_static=Int32(mA_mkl.shape[0]),
            len_k_static=Int32(mA_mkl.shape[1]),
        )

        # MMA partition
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        acc_shape = tiled_mma.partition_shape_C(self.cta_tile_shape_mnk[:2])
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        TileSchedulerCls = partial(
            TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline
        )

        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        k_tile_cnt = cute.ceil_div(mA_mkl.shape[1], self.cta_tile_shape_mnk[2])

        # =====================================================================
        # DMA warp — reuses SM90's load_AB via tma_get_copy_fn
        # =====================================================================
        if warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                gA_mk = cute.local_tile(
                    mA_mk,
                    cute.select(self.cta_tile_shape_mnk, [0, 2]),
                    (tile_coord_mnkl[0], None),
                )
                gB_nk = cute.local_tile(
                    varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                    cute.select(self.cta_tile_shape_mnk, [1, 2]),
                    (tile_coord_mnkl[1], None),
                )
                copy_A, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_a,
                    cta_coord=cluster_coord_mnk[1],
                    cta_layout=cute.make_layout(
                        cute.slice_(cluster_layout_mnk, (0, None, 0)).shape
                    ),
                    src_tensor=gA_mk,
                    dst_tensor=sA,
                    mcast_mask=a_mcast_mask,
                )
                copy_B, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_b,
                    cta_coord=cluster_coord_mnk[0],
                    cta_layout=cute.make_layout(
                        cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape
                    ),
                    src_tensor=gB_nk,
                    dst_tensor=sB,
                    mcast_mask=b_mcast_mask,
                )
                ab_producer_state = self.load_AB(
                    ab_pipeline, ab_producer_state, copy_A, copy_B, k_tile_cnt
                )
                tile_scheduler.advance_to_next_work(is_scheduler_warp=True)
                work_tile = tile_scheduler.get_current_work()
            ab_pipeline.producer_tail(ab_producer_state)
            tile_scheduler.producer_tail()

        # =====================================================================
        # MMA warps
        # =====================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            is_tma_warp = Boolean(warp_idx == 0)

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

            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx

                ab_read_state = self.mma(
                    ab_pipeline,
                    ab_read_state,
                    tiled_mma,
                    accumulators,
                    k_tile_cnt,
                    smem_tiled_copy_A,
                    smem_tiled_copy_B,
                    tCsA_copy_view,
                    tCsB_copy_view,
                    tCrA,
                    tCrB,
                )

                # ============================================================
                # EPILOGUE — reuse SM90's epilogue flow
                # ============================================================
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

                d_dtype_for_layout = self.d_dtype if self.d_dtype is not None else cutlass.BFloat16
                tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                    tiled_mma, self.d_layout, d_dtype_for_layout, sD, tidx
                )
                tRS_rAcc = self.epi_retile_acc(accumulators, tRS_rD, tiled_copy_r2s)
                load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
                if const_expr(has_C):
                    tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                        tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
                    )
                else:
                    tiled_copy_s2r, tSR_sC, tRS_rC, tSR_rC = None, None, None, None

                self.epi_visit_acc(epilogue_params, accumulators, tiled_mma, tile_coord_mnkl, tidx)

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

                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            # Wait for D store complete
            if is_tma_warp:
                epi_store_pipeline.producer_tail()
