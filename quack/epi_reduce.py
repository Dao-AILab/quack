"""Fused-communication (epi_reduce_mode) GEMM epilogue pieces; nothing here runs on
its own — gemm_sm100's kernel binds each piece into the shared GemmBase machinery.
Sections below: host contract / reducer tile scheduler / cross-launch exit barrier /
producer -> reducer tile signal / multimem reduce + store.

Dataflow: the producer runs epilogue(skip_epi_ops=True) via epilogue_split_k — D partials
through the normal TMA path into symmetric D, then the tile signal. The reducer runs
epilogue() between two bound functions: multimem ld_reduce in, EVT/C_load/aux
TileStores unchanged in the middle, register-direct store out (this rank's slab for
reduce_scatter, multimem_st broadcast for all_reduce). Pipelines: epi_pipeline stages
C for the reducer; epi_store_pipeline backs the producer's D store;
epi_reduce_store_pipeline backs the reducer's aux stores."""

from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, const_expr
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import llvm

from quack.cute_dsl_utils import mlir_namedtuple
from quack.dist_utils import multimem_ld_reduce_128b
from quack.fast_math import FastDivmod


# ---- host contract ----


@mlir_namedtuple
class EpiReduceArguments(NamedTuple):
    """Comm-side tensors for epi_reduce_mode. tile_flags/counters are sized to one
    problem shape (the tile->slot mapping); sync_barrier is per resident epi-reduce
    CTA slot, with num_sms allocation remaining a safe upper bound."""

    mD_mc: Optional[cute.Tensor] = None  # multicast view of symmetric D
    mD_peers: Optional[tuple] = None  # per-rank views of symmetric D
    # producer -> consumer, ceil(M/cta_M) * ceil(N/cta_N) * L entries
    tile_flags: Optional[cute.Tensor] = None
    tile_flags_mc: Optional[cute.Tensor] = None
    sync_barrier: Optional[cute.Tensor] = None  # exit barrier, one slot per resident CTA
    sync_barrier_mc: Optional[cute.Tensor] = None
    # consumer-private epoch bases, slab_tiles_m * ceil(N/cta_N) * L entries
    consumer_counters: Optional[cute.Tensor] = None


# ---- reducer tile scheduler ----
# The reducer warps and the epi-load warp (C staging) walk the same slab-order
# static persistent schedule.


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
    """CLC launch grid is grid=(cl_m * ncl_mn, cl_n, batch): block_idx is already the
    hierarchical coord, so peel the cluster-local part with FastDivmod and
    linearize the cluster coord through its layout.
    Returns (linear persistent cluster id, CTA m in cluster, CTA n in cluster)."""
    bidx, bidy, bidz = cute.arch.block_idx()
    gdx, gdy, gdz = cute.arch.grid_dim()
    cl_m, cl_n = cluster_shape_mn
    cl_m_fdd, cl_n_fdd = FastDivmod(cl_m), FastDivmod(cl_n)
    c_m, cta_m = divmod(bidx, cl_m_fdd)
    c_n, cta_n = divmod(bidy, cl_n_fdd)
    grid_m, _ = divmod(gdx, cl_m_fdd)
    grid_n, _ = divmod(gdy, cl_n_fdd)
    cluster_layout = cute.make_layout((grid_m, grid_n, gdz))
    return cluster_layout((c_m, c_n, bidz)), cta_m, cta_n


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


# ---- cross-launch exit sync barrier: one slot per resident CTA ----


@cute.jit
def epi_reduce_exit_slot(params: EpiReduceSchedulerParams) -> Int32:
    # Keep the block_idx-derived coords inside one jit: returning the tuple to the
    # kernel and re-consuming it in a second jit mis-materializes the slot -> OOB write.
    cluster_shape_mn = params.tile_sched_params.cluster_shape_mn
    cluster_id, cta_m, cta_n = clc_block_to_static_scheduler_coord(cluster_shape_mn)
    slot_layout = cute.make_layout((*cluster_shape_mn, params.num_persistent_clusters))
    return slot_layout((cta_m, cta_n, cluster_id))


# ---- producer -> reducer tile signal ----


@cute.jit
def signal_partial_committed(
    tile_flags_mc: cute.Tensor, tile_id: Int32, in_bounds: cutlass.Boolean
) -> None:
    """+1 on every rank's copy of this tile's flag: this rank's partial for the tile
    is in symmetric D. Flags are monotonic (never reset); consumers epoch-track via
    consumer_counters. Passed to epilogue_split_k as signal_finalized_tile — the local
    finalizer's "tile committed" is, globally, one partial of the cross-rank reduce.
    The caller binds addressing (tile_id, OOB guard); the wrapper owns ordering
    (drains the finalizer's D partial TMA stores before this release)."""
    if in_bounds:
        with cute.arch.elect_one():
            utils.distributed.multimem_red_add1(
                lock_ptr=tile_flags_mc.iterator + tile_id, scope="gpu", order="release"
            )


# ---- multimem reduce + store ----
# Tile-agnostic (no GEMM state; a standalone RS kernel could bind them), under a
# three-part contract: the fragment views are of a symmetric-heap tensor (mc view
# for ld_reduce/broadcast, peer view for the slab store); the partition's value
# atom is one contiguous 128b vector (n-major, N % (16B/elem) == 0); subtiles
# slice fragment rows evenly (chunk = loop_m / num_subtiles).


@cute.jit
def multimem_reduce_subtile(
    frgD_mc: cute.Tensor,
    frgD_crd: cute.Tensor,
    row_limit: Int32,
    col_limit: Int32,
    subtile_layout: cute.Layout,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
    no_release: cutlass.Constexpr[bool] = False,
) -> None:
    """Reduce this subtile's D partials across all ranks into tRS_rD via multimem
    ld_reduce; passed to epilogue() as load_acc_subtile by the reducer warps.
    Rows/cols past row/col_limit (partial slab tile / N tail) are zeroed: keeps
    visit reductions exact. N % (16B/elem) keeps vectors from straddling the edge."""
    _atom, chunk, loop_n = tRS_rD.shape
    epi_idx = subtile_layout(epi_coord)
    ld_reduce = multimem_ld_reduce_128b(frgD_mc.element_type)
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
    tmp_rD = cute.recast_tensor(tmp_results, frgD_mc.element_type)
    tRS_rD.store(tmp_rD.load().to(tRS_rD.element_type))


@cute.jit
def commit_reduced_subtile(
    frgD_mc: cute.Tensor,
    frgD_peer: cute.Tensor,
    frgD_crd: cute.Tensor,
    row_limit: Int32,
    col_limit: Int32,
    subtile_layout: cute.Layout,
    all_reduce: cutlass.Constexpr[bool],
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
) -> None:
    """Store the reduced, post-EVT subtile — reduce_scatter: st.global to this rank's
    D slab; all_reduce: multimem_st broadcast to every rank. Passed to epilogue() as
    commit_D by the reducer warps. Owns d_dtype conversion and edge predication: skip
    rows past the slab (a foreign-row store races the owner's reduce) and cols past N
    (n-major D: an OOB column wraps into the next row)."""
    _atom, chunk, loop_n = tRS_rD.shape
    epi_idx = subtile_layout(epi_coord)
    if const_expr(tRS_rD.element_type != frgD_mc.element_type):
        tmp_out = cute.make_rmem_tensor(tRS_rD.layout.shape, frgD_mc.element_type)
        tmp_out.store(tRS_rD.load().to(frgD_mc.element_type))
    else:
        tmp_out = tRS_rD
    out_i32 = cute.recast_tensor(tmp_out, cutlass.Int32)
    for ii in cutlass.range_constexpr(chunk):
        i = epi_idx * chunk + ii
        for j in cutlass.range_constexpr(loop_n):
            crd = frgD_crd[((0, 0), i, j)]
            if crd[0] < row_limit and crd[1] < col_limit:
                if const_expr(all_reduce):
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
