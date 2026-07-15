# Referenced-from: quack/gemm_base.py (GemmTmaBase)
"""GemmTpTmaBase: quack's GemmTmaBase + fused-communication epilogues (two_shot
reduce_scatter / all_reduce)."""

from typing import Callable, Dict, NamedTuple, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Int32, const_expr
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import llvm

import quack.copy_utils as copy_utils
from quack.cute_dsl_utils import ParamsBase, mlir_namedtuple
from quack.gemm_base import GemmTmaBase, NamedBarrierGemm  # noqa: F401 (re-exported)
from quack.rounding import RoundingMode, epilogue_sr_seed
from quack.varlen_utils import VarlenManager

# epi_reduce warp-group sync barrier (NamedBarrierGemm is an IntEnum, not subclassable)
EPI_REDUCE_BARRIER_ID = max(NamedBarrierGemm).value + 1


@mlir_namedtuple
class EpiReduceSchedulerParams(NamedTuple):
    tile_sched_params: utils.PersistentTileSchedulerParams
    num_persistent_clusters: Int32

    @staticmethod
    def create(problem_shape_ntile_mnl, cluster_shape_mnk, max_active_clusters):
        assert cluster_shape_mnk[2] == 1, (
            "EpiReduceSchedulerParams assumes cluster_shape_mnk[2] == 1"
        )
        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, cluster_shape_mnk
        )
        num_persistent_clusters = cutlass.min(
            cute.size(tile_sched_params.problem_layout_ncluster_mnl),
            max_active_clusters,
        )
        return EpiReduceSchedulerParams(tile_sched_params, num_persistent_clusters)


@cute.jit
def clc_block_to_static_scheduler_coord(cluster_shape_mn):
    """
    CLC launch grid uses grid=(cl_m * logical_cluster_id_over_MN, cl_n, batch).
    Convert this CTA's launch position to the static scheduler coordinate:
    (linear persistent cluster id, CTA m in cluster, CTA n in cluster).
    """
    bidx, bidy, bidz = cute.arch.block_idx()
    gdx, gdy, _ = cute.arch.grid_dim()
    cl_m, cl_n = cluster_shape_mn
    cluster_id = bidx // cl_m + (gdx // cl_m) * (
        bidy // cl_n + (gdy // cl_n) * bidz
    )
    return cluster_id, bidx % cl_m, bidy % cl_n


@cute.jit
def static_scheduler_coord_to_slot(
    cluster_id: Int32, cta_m: Int32, cta_n: Int32, cluster_shape_mn
) -> Int32:
    cl_m, cl_n = cluster_shape_mn
    return cluster_id * (cl_m * cl_n) + cta_n * cl_m + cta_m


@cute.jit
def make_epi_reduce_tile_scheduler(params: EpiReduceSchedulerParams):
    tile_sched_params = params.tile_sched_params
    cluster_shape_mn = tile_sched_params.cluster_shape_mn
    cl_m, cl_n = cluster_shape_mn
    cluster_id, cta_m, cta_n = clc_block_to_static_scheduler_coord(cluster_shape_mn)
    return utils.StaticPersistentTileScheduler.create(
        tile_sched_params,
        (cta_m, cta_n, cluster_id),
        (cl_m, cl_n, params.num_persistent_clusters),
    )


@cute.jit
def epi_reduce_exit_slot(params: EpiReduceSchedulerParams) -> Int32:
    # Keep the block_idx-derived coords inside one jit: returning the tuple to the
    # kernel and re-consuming it in a second jit mis-materializes the slot -> OOB write.
    cluster_shape_mn = params.tile_sched_params.cluster_shape_mn
    cluster_id, cta_m, cta_n = clc_block_to_static_scheduler_coord(cluster_shape_mn)
    return static_scheduler_coord_to_slot(cluster_id, cta_m, cta_n, cluster_shape_mn)


def _multimem_ld_reduce_128b(dtype):
    """128-bit multimem.ld_reduce(add) variant for dtype; all return 4x b32 (x, y, z, w)."""
    if dtype == cutlass.Float16:
        return utils.distributed.multimem_ld_reduce_8xf16
    if dtype == cutlass.Float32:
        return utils.distributed.multimem_ld_reduce_4xf32
    if dtype == cutlass.BFloat16:
        return utils.distributed.multimem_ld_reduce_8xbf16
    if dtype == cutlass.Float8E4M3FN:
        return utils.distributed.multimem_ld_reduce_16xe4m3
    if dtype == cutlass.Float8E5M2:
        return utils.distributed.multimem_ld_reduce_16xe5m2
    raise NotImplementedError(f"epilogue_reduce: unsupported D dtype {dtype}")


class GemmTpTmaBase(GemmTmaBase):
    """
    epilogue_skip_evt() performs D_store; EVT and C_load postponed to epilogue_reduce().
    Its parameter list matches GemmBase.epilogue so the epilog-warp call site selects
    either; the C/EVT args are accepted but unused.

    Pipeline changes:
    - epi_pipeline for C_load (G2S) used in epilogue_reduce()
    - epi_store_pipeline for D_store (S2G) used in epilogue_skip_evt()
    - [new] epi_reduce_store_pipeline for AuxOut store (S2G) used in epilogue_reduce()

    Note: D_store in epilogue_reduce() is a plain st.global to this rank's local slab
    (reduce_scatter) or a multimem_st to the multicast tensor (all_reduce)
    """

    EpilogueParams = ParamsBase

    # Staging tiles under use_epi_reduce (read by quack.epi_utils.setup_epi_tensor):
    #   D partials -> epi_tile (epilogue warps, unchanged)
    #   C / EpiOp aux -> epi_reduce_tile (epi_reduce warps)
    use_epi_reduce = None
    epi_reduce_tile = None

    @mlir_namedtuple
    class EpiReduceArguments(NamedTuple):
        """Comm-side tensors for use_epi_reduce. tile_flags/counters are sized to one
        problem shape (the tile->slot mapping); sync_barrier is per resident epi-reduce
        CTA slot, with num_sms allocation remaining a safe upper bound."""

        mD_mc: Optional[cute.Tensor] = None  # multicast view of symmetric D
        mD_peers: Optional[tuple] = None  # per-rank views of symmetric D
        tile_flags: Optional[cute.Tensor] = None  # producer->consumer, ceil(M/cta_M)*ceil(N/cta_N)*L
        tile_flags_mc: Optional[cute.Tensor] = None
        sync_barrier: Optional[cute.Tensor] = None  # exit barrier, one slot per resident CTA
        sync_barrier_mc: Optional[cute.Tensor] = None
        consumer_counters: Optional[cute.Tensor] = None  # consumer-private, slab_tiles_m*ceil(N/cta_N)*L

    @cute.jit
    def epilogue_skip_evt(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Dict[str, cute.Tensor],
        epi_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        epi_store_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        epi_read_state: Optional[cutlass.pipeline.PipelineState],
        epi_producer_state: Optional[cutlass.pipeline.PipelineState],
        epi_tile: cute.Tile,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor],
        tiled_copy_t2r: Optional[cute.TiledCopy],  # Only for Sm100
        tiled_copy_r2s: cute.TiledCopy,
        tRS_sD: cute.Tensor,
        tiled_copy_s2r: Optional[cute.ThrCopy],
        tSR_rC: Optional[cute.Tensor],
        tSR_sC: Optional[cute.Tensor],
        copy_D: Optional[Callable],
        copy_C: Optional[Callable],
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler,
        tidx: Int32,
        is_tma_warp: cutlass.Boolean,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        has_D = const_expr(copy_D is not None)
        use_tma_epi = const_expr(epi_store_pipeline is not None)
        use_stochastic_rounding = const_expr(
            self.rounding_mode == RoundingMode.RS
            and self.acc_dtype == cutlass.Float32
            and self.d_dtype == cutlass.BFloat16
        )

        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape, order=(0, 1) if const_expr(self.epi_m_major) else (1, 0)
        )
        epi_tile_num = cute.size(epi_tile_shape)
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)  # (epi_m, epi_n)
            # Copy from acc to D registers
            load_acc_subtile(tRS_rD, epi_coord)
            if const_expr(use_tma_epi):
                if is_tma_warp:
                    epi_store_pipeline.producer_acquire()
            else:
                epilogue_barrier.arrive_and_wait()
            if const_expr(use_tma_epi):
                epilogue_barrier.arrive_and_wait()
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            if const_expr(has_D):
                tRS_sD_cur = tRS_sD[None, None, None, epi_buffer]
                if const_expr(use_stochastic_rounding):
                    # No epi ops run here: coord-derived seed (params.sr_seed not wired).
                    epi_loop_tensors = {}
                    seed = epilogue_sr_seed(
                        epi_loop_tensors.get("sr_seed"),
                        tile_coord_mnkl,
                        num_prev_subtiles + epi_idx,
                    )
                    copy_utils.sr_cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD_cur, seed, tidx)
                else:
                    copy_utils.cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD_cur)

            if const_expr(use_tma_epi):
                cute.arch.fence_view_async_shared()
                epilogue_barrier.arrive_and_wait()
                if is_tma_warp:
                    if const_expr(has_D):
                        copy_D(src_idx=epi_buffer, dst_idx=epi_coord)
                    epi_store_pipeline.producer_commit()
            else:
                epilogue_barrier.arrive_and_wait()
                if const_expr(has_D):
                    copy_D(src_idx=epi_buffer, dst_idx=epi_coord)
                epilogue_barrier.arrive_and_wait()

        return epi_read_state, epi_producer_state

    @cute.jit
    def epilogue_reduce(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Dict[str, cute.Tensor],
        epi_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        epi_reduce_store_pipeline: cutlass.pipeline.PipelineTmaStore,
        epi_read_state: Optional[cutlass.pipeline.PipelineState],
        epi_producer_state: Optional[cutlass.pipeline.PipelineState],
        epi_tile: cute.Tile,
        frgD_mc: cute.Tensor,
        frgD_peer: cute.Tensor,
        frgD_crd: cute.Tensor,
        row_limit: Int32,
        col_limit: Int32,
        tSR_sC: Optional[cute.Tensor],
        copy_C: Optional[Callable],
        tiled_copy_fake: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler,
        tidx: Int32,
        is_tma_warp: cutlass.Boolean,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        has_C = const_expr(tSR_sC is not None)
        has_epi_load = const_expr(self.epi_c_stage > 0)
        use_tma_c = const_expr(epi_pipeline is not None)
        # copy_C selects C production: inline (SM90) vs dedicated epi-load warp (None).
        inline_epi_load = const_expr(copy_C is not None)
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape, order=(0, 1) if const_expr(self.epi_m_major) else (1, 0)
        )
        epi_tile_num = cute.size(epi_tile_shape)
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        # Aux outputs use the TileStore smem+TMA path, staged per epi_reduce_tile.
        aux_out_ctxs = self.epi_setup_aux_out(
            params,
            epi_smem_tensors,
            tiled_copy_fake,
            None,
            tile_coord_mnkl,
            varlen_manager,
            tidx,
        )

        epi_tensors = self.epi_begin(
            params,
            epi_smem_tensors,
            epi_tile,
            None,
            tiled_copy_fake,
            tile_coord_mnkl,
            varlen_manager,
            epilogue_barrier,
            tidx,
            None,
        )

        if const_expr(inline_epi_load):
            for epi_idx in cutlass.range(min(epi_tile_num, self.epi_c_stage), unroll=1):
                epi_coord_C = epi_tile_layout.get_hier_coord(epi_idx)
                if const_expr(use_tma_c):
                    if is_tma_warp:
                        epi_pipeline.producer_acquire(epi_producer_state)
                        copy_C(src_idx=epi_coord_C, producer_state=epi_producer_state)
                        epi_pipeline.producer_commit(epi_producer_state)
                    epi_producer_state.advance()
                else:
                    # TODO: turn this to cp.async instead of direct G2R copy
                    copy_C(src_idx=epi_coord_C, dst_idx=epi_idx % self.epi_c_stage)
            if const_expr(use_tma_c):
                epilogue_barrier.arrive_and_wait()

        # frgD_* shape = (atom, loop_m, loop_n); each epi_idx owns chunk = loop_m // epi_tile_num rows.
        _atom, loop_m, loop_n = frgD_mc.shape
        chunk = loop_m // epi_tile_num
        ld_reduce = _multimem_ld_reduce_128b(self.d_dtype)

        # C fragment per epi_reduce_tile; hier shape must match the reduce fragment's.
        tRS_rC, tSR_rC = None, None
        if const_expr(has_C):
            tRS_rC = cute.make_rmem_tensor((_atom, chunk, loop_n), self.c_dtype)
            tSR_rC = tiled_copy_fake.retile(tRS_rC)

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)  # (epi_m, epi_n)

            if const_expr(has_epi_load):
                if const_expr(use_tma_c):
                    epi_pipeline.consumer_wait(epi_read_state)
                    if const_expr(has_C):
                        cute.copy(
                            tiled_copy_fake, tSR_sC[None, None, None, epi_read_state.index], tSR_rC
                        )
                    self.epi_tile_load_s2r(params, epi_tensors, epi_read_state.index)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        epi_pipeline.consumer_release(epi_read_state)
                    epi_read_state.advance()
                else:
                    c_buffer = epi_idx % self.epi_c_stage
                    cute.copy(tiled_copy_fake, tSR_sC[None, None, None, c_buffer], tSR_rC)
                    # TODO: cp.async wait once we switch to cp.async
                    epilogue_barrier.arrive_and_wait()

            epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, epi_coord)

            if const_expr(inline_epi_load and epi_idx + self.epi_c_stage < epi_tile_num):
                epi_coord_C = epi_tile_layout.get_hier_coord(epi_idx + self.epi_c_stage)
                if const_expr(use_tma_c):
                    if is_tma_warp:
                        epi_pipeline.producer_acquire(epi_producer_state)
                        copy_C(src_idx=epi_coord_C, producer_state=epi_producer_state)
                        epi_pipeline.producer_commit(epi_producer_state)
                    epi_producer_state.advance()
                else:
                    epilogue_barrier.arrive_and_wait()
                    copy_C(
                        src_idx=epi_coord_C,
                        dst_idx=(epi_idx + self.epi_c_stage) % self.epi_c_stage,
                    )

            # (1) multimem reduce this epi-subtile across ranks into registers.
            # Rows/cols past row/col_limit (partial slab tile / N tail) are zeroed: keeps
            # visit reductions exact. N % (16B/elem) keeps vectors from straddling the edge.
            tmp_results = cute.make_rmem_tensor((4, chunk, loop_n), cutlass.Int32)
            for ii in cutlass.range_constexpr(chunk):
                i = epi_idx * chunk + ii
                for j in cutlass.range_constexpr(loop_n):
                    crd = frgD_crd[((0, 0), i, j)]
                    if crd[0] < row_limit and crd[1] < col_limit:
                        mc_ptr = frgD_mc[None, i, j].iterator
                        x, y, z, w = ld_reduce(mc_ptr)
                        tmp_results[0, ii, j] = x
                        tmp_results[1, ii, j] = y
                        tmp_results[2, ii, j] = z
                        tmp_results[3, ii, j] = w
                    else:
                        tmp_results[0, ii, j] = Int32(0)
                        tmp_results[1, ii, j] = Int32(0)
                        tmp_results[2, ii, j] = Int32(0)
                        tmp_results[3, ii, j] = Int32(0)
            tmp_rD = cute.recast_tensor(tmp_results, self.d_dtype)

            # (2) post-reduce EVT. Fragment must keep the partition's hier atom mode ((1, 8), ...):
            # a flat (8, ...) compiles, but VecReduce epi ops mis-pair modes (only slot 0 correct).
            tRS_rD = cute.make_rmem_tensor((_atom, chunk, loop_n), self.acc_dtype)
            tRS_rD.store(tmp_rD.load().to(self.acc_dtype))
            tRS_rAuxOuts = self.epi_visit_subtile(params, epi_loop_tensors, tRS_rD, tRS_rC)
            if const_expr(len(aux_out_ctxs) > 0):
                tRS_rAuxOuts_out = tuple(
                    self.epi_convert_aux_out(
                        i,
                        tRS_rAuxOuts[i],
                        epi_loop_tensors.get("sr_seed"),
                        tidx,
                        tile_coord_mnkl,
                        num_prev_subtiles,
                        epi_idx,
                    )
                    for i in range(len(aux_out_ctxs))
                )
                epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
                # Stage reuse: wait for the TMA store that last read this stage's smem.
                if is_tma_warp:
                    epi_reduce_store_pipeline.producer_acquire()
                epilogue_barrier.arrive_and_wait()
                for i in cutlass.range_constexpr(len(aux_out_ctxs)):
                    tiled_copy_aux_out_r2s, tRS_sAuxOut, _ = aux_out_ctxs[i]
                    cute.copy(
                        tiled_copy_aux_out_r2s,
                        tiled_copy_aux_out_r2s.retile(tRS_rAuxOuts_out[i]).contiguous(),
                        tRS_sAuxOut[None, None, None, epi_buffer],
                    )
                cute.arch.fence_view_async_shared()
                epilogue_barrier.arrive_and_wait()
                if is_tma_warp:
                    for i in cutlass.range_constexpr(len(aux_out_ctxs)):
                        _, _, copy_aux_out = aux_out_ctxs[i]
                        copy_aux_out(src_idx=epi_buffer, dst_idx=epi_coord)
                    epi_reduce_store_pipeline.producer_commit()
            self.epi_end_loop(
                params,
                epi_tensors,
                epi_coord,
                epi_tile,
                None,
                tiled_copy_fake,
                tile_coord_mnkl,
                varlen_manager,
                tidx,
            )

            # (3) reduce_scatter: store to this rank's D slab; all_reduce: multicast-broadcast.
            if const_expr(tRS_rD.element_type != self.d_dtype):
                tmp_out = cute.make_rmem_tensor(tRS_rD.layout.shape, self.d_dtype)
                tmp_out.store(tRS_rD.load().to(self.d_dtype))
            else:
                tmp_out = tRS_rD
            out_i32 = cute.recast_tensor(tmp_out, cutlass.Int32)
            for ii in cutlass.range_constexpr(chunk):
                i = epi_idx * chunk + ii
                for j in cutlass.range_constexpr(loop_n):
                    # Skip rows past the slab (a foreign-row store races the owner's reduce)
                    # and cols past N (n-major D: an OOB column wraps into the next row).
                    crd = frgD_crd[((0, 0), i, j)]
                    if crd[0] < row_limit and crd[1] < col_limit:
                        if const_expr(self.use_epi_reduce == "all_reduce"):
                            utils.distributed.multimem_st_4xb32(
                                frgD_mc[None, i, j].iterator,
                                out_i32[0, ii, j].ir_value(),
                                out_i32[1, ii, j].ir_value(),
                                out_i32[2, ii, j].ir_value(),
                                out_i32[3, ii, j].ir_value(),
                            )
                        else:
                            ptr_int = frgD_peer[None, i, j].iterator.toint().ir_value()
                            x, y, z, w = (
                                out_i32[0, ii, j].ir_value(),
                                out_i32[1, ii, j].ir_value(),
                                out_i32[2, ii, j].ir_value(),
                                out_i32[3, ii, j].ir_value(),
                            )
                            llvm.inline_asm(
                                T.i32(),
                                [ptr_int, x, y, z, w],
                                "st.global.sys.relaxed.v4.f32 [$1], {$2, $3, $4, $5};",
                                "=r,l,r,r,r,r",
                                has_side_effects=True,
                                asm_dialect=0,
                            )

        return epi_read_state, epi_producer_state

    def make_tma_epilogue_atoms_and_tensors(
        self,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args,
        varlen_m: bool,
    ):
        """
        For use_epi_reduce path, C_load in epilogue_reduce() is staged per epi_reduce_tile =
        (32, cta_N), rather than epi_tile = (cta_M, cta_N/num_epi_subtiles) used by the
        epilogue warps. D_store is unchanged (epi_tile).
        """
        tma_atom_d, tma_tensor_d = None, None
        if const_expr(mD is not None):
            tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
                copy_utils.create_ragged_tensor_for_tma(mD, ragged_dim=0, ptr_shift=True)
                if varlen_m
                else mD,
                self.epi_smem_layout_staged,
                self.epi_tile,
                op_type="store"
                if not (hasattr(epilogue_args, "add_to_output") and epilogue_args.add_to_output)
                else "add",
            )
        tma_atom_c, tma_tensor_c = None, None
        if const_expr(mC is not None):
            tma_atom_c, tma_tensor_c = self._make_tma_epi_atoms_and_tensors(
                mC,
                self.epi_c_smem_layout_staged,
                self.epi_reduce_tile
                if const_expr(self.use_epi_reduce is not None)
                else self.epi_tile,
                op_type="load",
            )
        return tma_atom_d, tma_tensor_d, tma_atom_c, tma_tensor_c

    def make_epi_pipeline(
        self,
        epi_pipeline_mbar_ptr: cute.Pointer,
        tx_count: int,
    ):
        """
        For use_epi_reduce path, consumers of the C_load pipeline are the epi_reduce warps,
        not the epilogue warps. Today both counts are 4 (override is a no-op) but they may
        differ in future.
        """
        epi_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Each warp will contribute 1 to the arrive count.
        consumer_arrive_cnt = (
            self.num_epi_warps
            if const_expr(self.use_epi_reduce is None)
            else self.num_epi_reduce_warps
        )
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=epi_pipeline_mbar_ptr,
            num_stages=self.epi_c_stage,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
            tx_count=tx_count,
            defer_sync=True,
        )

    def make_epi_reduce_store_pipeline(self):
        """
        AuxOut store (S2G) in epilogue_reduce(); the epilogue warps' D_store keeps its own
        epi_store_pipeline.
        """
        num_epi_reduce_threads = self.num_epi_reduce_warps * cute.arch.WARP_SIZE
        epi_reduce_store_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_epi_reduce_threads
        )
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage, producer_group=epi_reduce_store_producer_group
        )
