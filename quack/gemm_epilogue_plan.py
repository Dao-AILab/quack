# Copyright (c) 2025, Tri Dao.

from typing import Optional, Tuple, Callable

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Boolean, const_expr

import quack.copy_utils as copy_utils
from quack.rounding import RoundingMode
from quack.varlen_utils import VarlenManager


def default_epi_tile_layout(self, epi_tile_shape):
    return cute.make_ordered_layout(epi_tile_shape, order=(1, 0))


@cute.jit
def default_epi_commit(
    self,
    gmem_coord,
    epi_buffer,
    copy_D,
    copy_postact,
    postact_ctx,
    tile_coord_mnkl,
    is_tma_warp,
    epi_store_pipeline,
):
    del tile_coord_mnkl
    has_D = const_expr(copy_D is not None)
    if is_tma_warp:
        if const_expr(has_D):
            copy_D(src_idx=epi_buffer, dst_idx=gmem_coord)
        if const_expr(postact_ctx is not None):
            copy_postact(src_idx=epi_buffer, dst_idx=gmem_coord)
        epi_store_pipeline.producer_commit()


@cute.jit
def symmetric_epi_commit(
    self,
    gmem_coord,
    epi_buffer,
    copy_D,
    copy_postact,
    postact_ctx,
    tile_coord_mnkl,
    is_tma_warp,
    epi_store_pipeline,
):
    has_D = const_expr(copy_D is not None)
    if is_tma_warp:
        square_tile_m = tile_coord_mnkl[0] // self.cluster_shape_mnk[0]
        square_tile_n = tile_coord_mnkl[1] // self.cluster_shape_mnk[1]
        if const_expr(has_D):
            copy_D(src_idx=epi_buffer, dst_idx=gmem_coord)
        if const_expr(postact_ctx is not None) and square_tile_m != square_tile_n:
            copy_postact(src_idx=epi_buffer, dst_idx=gmem_coord)
        epi_store_pipeline.producer_commit()


@cute.jit
def run_epilogue_plan(
    self,
    params,
    epi_smem_tensors: Tuple[cute.Tensor, ...],
    epi_pipeline: cutlass.pipeline.PipelineAsync,
    epi_store_pipeline: cutlass.pipeline.PipelineAsync,
    epi_read_state: cutlass.pipeline.PipelineState,
    epi_producer_state: cutlass.pipeline.PipelineState,
    epi_tile: cute.Tile,
    load_acc_subtile: Callable,
    tRS_rD: cute.Tensor,
    tRS_rC: Optional[cute.Tensor],
    tiled_copy_t2r: Optional[cute.TiledCopy],
    tiled_copy_r2s: cute.TiledCopy,
    tRS_sD: cute.Tensor,
    tiled_copy_s2r: Optional[cute.TiledCopy],
    tSR_rC: Optional[cute.Tensor],
    tSR_sC: Optional[cute.Tensor],
    copy_D: Optional[Callable],
    copy_C: Optional[Callable],
    tile_coord_mnkl: cute.Coord,
    varlen_manager: VarlenManager,
    epilogue_barrier: cutlass.pipeline.NamedBarrier,
    tile_scheduler,
    tidx: Int32,
    is_tma_warp: Boolean,
) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
    has_C = const_expr(tRS_rC is not None)
    has_D = const_expr(copy_D is not None)

    postact_ctx = self.epi_setup_postact(
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    )

    epi_tile_shape = cute.zipped_divide(cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile).shape[1]
    epi_tile_layout = self.epi_plan_make_tile_layout(epi_tile_shape)
    epi_tile_num = cute.size(epi_tile_shape)
    num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

    epi_tensors = self.epi_begin(
        params,
        epi_smem_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        epilogue_barrier,
        tidx,
    )

    if const_expr(copy_C is not None):
        for epi_idx in cutlass.range(min(epi_tile_num, self.epi_c_stage), unroll=1):
            gmem_coord_C = epi_tile_layout.get_hier_coord(epi_idx)
            if is_tma_warp:
                epi_pipeline.producer_acquire(epi_producer_state)
                copy_C(src_idx=gmem_coord_C, producer_state=epi_producer_state)
                epi_pipeline.producer_commit(epi_producer_state)
            epi_producer_state.advance()

    for epi_idx in cutlass.range_constexpr(epi_tile_num):
        gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
        load_acc_subtile(tRS_rD, epi_idx)
        epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, gmem_coord)
        if const_expr(has_C):
            epi_pipeline.consumer_wait(epi_read_state)
            cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                epi_pipeline.consumer_release(epi_read_state)
            epi_read_state.advance()
        if const_expr(copy_C is not None and epi_idx + self.epi_c_stage < epi_tile_num):
            gmem_coord_C = epi_tile_layout.get_hier_coord(epi_idx + self.epi_c_stage)
            if is_tma_warp:
                epi_pipeline.producer_acquire(epi_producer_state)
                copy_C(src_idx=gmem_coord_C, producer_state=epi_producer_state)
                epi_pipeline.producer_commit(epi_producer_state)
            epi_producer_state.advance()
        tRS_rPostAct = self.epi_visit_subtile(params, epi_loop_tensors, tRS_rD, tRS_rC)
        if const_expr(postact_ctx is not None):
            tRS_rPostAct_out = self.epi_convert_postact(
                tRS_rPostAct,
                epi_loop_tensors["sr_seed"],
                tidx,
                tile_coord_mnkl,
                num_prev_subtiles,
                epi_idx,
            )
        if is_tma_warp:
            epi_store_pipeline.producer_acquire()
        epilogue_barrier.arrive_and_wait()
        epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
        if const_expr(has_D):
            if const_expr(
                self.rounding_mode == RoundingMode.RS
                and self.acc_dtype == cutlass.Float32
                and self.d_dtype == cutlass.BFloat16
            ):
                seed = epi_loop_tensors["sr_seed"] + (
                    tile_coord_mnkl[0] * 65537
                    + tile_coord_mnkl[1] * 257
                    + tile_coord_mnkl[3] * 17
                    + (num_prev_subtiles + epi_idx) * 7
                )
                copy_utils.sr_cvt_copy(
                    tiled_copy_r2s,
                    tRS_rD,
                    tRS_sD[None, None, None, epi_buffer],
                    seed,
                    tidx,
                )
            else:
                copy_utils.cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, epi_buffer])
        copy_postact = None
        if const_expr(postact_ctx is not None):
            tiled_copy_postact_r2s, tRS_sPostAct, copy_postact = postact_ctx
            cute.copy(
                tiled_copy_postact_r2s,
                tiled_copy_postact_r2s.retile(tRS_rPostAct_out),
                tRS_sPostAct[None, None, None, epi_buffer],
            )
        cute.arch.fence_view_async_shared()
        epilogue_barrier.arrive_and_wait()
        self.epi_plan_commit(
            gmem_coord,
            epi_buffer,
            copy_D,
            copy_postact,
            postact_ctx,
            tile_coord_mnkl,
            is_tma_warp,
            epi_store_pipeline,
        )

    self.epi_end(
        params,
        epi_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    )

    return epi_read_state, epi_producer_state
