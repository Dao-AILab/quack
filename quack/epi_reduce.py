"""Fused-communication (epi_reduce_mode) GEMM epilogue pieces; nothing here runs on
its own — gemm_sm100's kernel binds each piece into the shared GemmBase machinery.
Sections below: host contract / reducer tile scheduler / cross-launch exit barrier /
multimem reduce + store. The split-rank protocol itself (flag contract, producer
partial commit, reducer spin + epoch counters) lives in gemm_base's
epilogue_split_rank / split_rank_partial_commit, the cross-rank siblings of the
split-K pair.

Dataflow: both warp groups run epilogue_split_rank. The producer's finalize action
is split_rank_partial_commit — register-direct d_dtype partials into the padded
symmetric workspace, then the tile signal. The reducer's is epilogue() between two
bound functions: multimem ld_reduce from the workspace in, EVT/C_load/aux
TileStores unchanged in the middle, register-direct store out (this rank's plain
slab-shaped D for reduce_scatter, multimem_st broadcast into symmetric D for
all_reduce). Pipelines: epi_pipeline stages C for the reducer; the reducer's aux
stores ride its own epi_store_pipeline instance."""

from typing import NamedTuple, Optional

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, const_expr

from quack.cute_dsl_utils import mlir_namedtuple
from quack.dist_utils import multimem_ld_reduce_128b
from quack.fast_math import FastDivmod


# ---- host contract ----


@mlir_namedtuple
class EpiReduceArguments(NamedTuple):
    """Comm-side tensors for epi_reduce_mode. tile_flags/counters are sized to one
    problem shape (the tile->slot mapping); sync_barrier is per resident epi-reduce
    CTA slot, with num_sms allocation remaining a safe upper bound."""

    mD_mc: Optional[cute.Tensor] = None  # multicast view of symmetric D — all_reduce only
    # d_dtype partials workspace, (M_pad, N_pad, L) at real (m, n) coords: this
    # rank's view (producer store target) and its multicast view (reducer ld_reduce).
    workspace: Optional[cute.Tensor] = None
    workspace_mc: Optional[cute.Tensor] = None
    # producer -> consumer, ceil(M/cta_M) * ceil(N/cta_N) * L entries
    tile_flags: Optional[cute.Tensor] = None
    tile_flags_mc: Optional[cute.Tensor] = None
    sync_barrier: Optional[cute.Tensor] = None  # exit barrier, one slot per resident CTA
    sync_barrier_mc: Optional[cute.Tensor] = None
    # consumer-private epoch bases, slab_tiles_m * ceil(N/cta_N) * L entries
    consumer_counters: Optional[cute.Tensor] = None


def epi_reduce_workspace_shape(m, n, cta_m, cta_n):
    """Padded workspace extents: N rounded to whole tiles, M rounded up plus one
    extra cta_m block. The pad keeps every workspace access in-allocation with no
    predication: reducer tiles anchor at rank*slab_m (not cta_m-aligned) and fully
    OOB phantom cluster CTAs store into the dead last block; nothing consumes pad."""
    m_pad = ((m + cta_m - 1) // cta_m + 1) * cta_m
    n_pad = (n + cta_n - 1) // cta_n * cta_n
    return m_pad, n_pad


def validate_epi_reduce_args(
    epi_reduce_args, D, mode, m, n, l, tile_M, tile_N, cluster_M, num_ranks
):
    """Guard the comm bundle against this call's geometry — the epi_reduce sibling of
    validate_ag_geometry, called per launch from every frontend (warm plan-cache hits
    skip trace-time asserts, so the host is the only per-call check). Everything here
    is a mismatch the kernel can only corrupt or hang on: multimem vector width,
    kernel-order comm views, and flag/counter capacities (an under-sized flag array
    is a silent OOB multimem write). m is the full GEMM M (from A): D carries only
    the slab under reduce_scatter."""
    era = epi_reduce_args
    if m % num_ranks:
        raise ValueError(f"epi_reduce_mode: m ({m}) must be divisible by world ({num_ranks})")
    if D is None:
        raise ValueError("epi_reduce_mode requires D (the output tensor)")
    vec = 16 // D.element_size()
    if n % vec:
        raise ValueError(f"epi_reduce_mode: n ({n}) must be divisible by {vec}")
    if D.stride(-1) != 1:
        raise ValueError("epi_reduce_mode: D must be n-major (multimem vectors)")
    d_rows = m // num_ranks if mode == "reduce_scatter" else m
    if D.shape[-2] != d_rows or D.shape[-1] != n:
        raise ValueError(
            f"epi_reduce_mode={mode}: D rows x cols ({d_rows}, {n}) expected, "
            f"got ({D.shape[-2]}, {D.shape[-1]})"
        )
    use_2cta = cluster_M % 2 == 0 and tile_M in (128, 256)
    cta_m = tile_M // (2 if use_2cta else 1)
    ws_mnl = (*epi_reduce_workspace_shape(m, n, cta_m, tile_N), l)
    for name, t in (("workspace", era.workspace), ("workspace_mc", era.workspace_mc)):
        if t is None or tuple(t.shape) != ws_mnl:
            raise ValueError(
                f"epi_reduce_args.{name}: kernel-order padded (m, n, l) {ws_mnl} expected, "
                f"got {None if t is None else tuple(t.shape)}"
            )
    if era.workspace.dtype != D.dtype or era.workspace.stride(1) != 1:
        raise ValueError("epi_reduce_args.workspace must be n-major with D's dtype")
    if mode == "reduce_scatter":
        if era.mD_mc is not None:
            raise ValueError("reduce_scatter commits to plain local D: mD_mc must be None")
    elif era.mD_mc is None or tuple(era.mD_mc.shape) != (m, n, l):
        raise ValueError(
            f"all_reduce broadcast needs mD_mc, kernel-order (m, n, l) {(m, n, l)}, got "
            f"{None if era.mD_mc is None else tuple(era.mD_mc.shape)}"
        )
    n_tiles = (n + tile_N - 1) // tile_N
    ntiles = ((m + cta_m - 1) // cta_m) * n_tiles * l
    if era.tile_flags.numel() < ntiles or era.tile_flags_mc.numel() < ntiles:
        raise ValueError(f"epi_reduce_args.tile_flags needs >= {ntiles} entries")
    num_sms = torch.cuda.get_device_properties(D.device).multi_processor_count
    if era.sync_barrier.numel() < num_sms or era.sync_barrier_mc.numel() < num_sms:
        raise ValueError(f"epi_reduce_args.sync_barrier needs >= {num_sms} entries")
    slab_tiles = ((m // num_ranks + cta_m - 1) // cta_m) * n_tiles * l
    if era.consumer_counters.numel() < slab_tiles:
        raise ValueError(f"epi_reduce_args.consumer_counters needs >= {slab_tiles} entries")


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


# ---- multimem reduce + store ----
# Tile-agnostic (no GEMM state; a standalone RS kernel could bind them), under a
# three-part contract: the reduce reads a symmetric padded workspace through its mc
# view (unpredicated — pad keeps every access in-allocation); the partition's value
# atom is one contiguous 128b vector (n-major, N % (16B/elem) == 0); subtile
# (mi, ni) owns the even fragment block rows [mi*chunk, (mi+1)*chunk) x cols
# [ni*sub_loop_n, (ni+1)*sub_loop_n).


@cute.jit
def multimem_reduce_subtile(
    frgWs_mc: cute.Tensor,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
    # load_acc_subtile signature compat (acc prepass); a multimem load has nothing to release.
    no_release: cutlass.Constexpr[bool] = False,
) -> None:
    """Reduce this subtile's workspace partials across all ranks into tRS_rD via
    multimem ld_reduce; passed to epilogue() as load_acc_subtile by the reducer
    warps. Unpredicated: the padded workspace keeps every access in-allocation;
    rows/cols past the slab or N edge carry garbage that the commit skips and epi
    ops predicate away (slab-framed limits)."""
    _atom, chunk, sub_loop_n = tRS_rD.shape
    ld_reduce = multimem_ld_reduce_128b(frgWs_mc.element_type)
    tmp_results = cute.make_rmem_tensor((4, chunk, sub_loop_n), cutlass.Int32)
    for ii in cutlass.range_constexpr(chunk):
        i = epi_coord[0] * chunk + ii
        for jj in cutlass.range_constexpr(sub_loop_n):
            j = epi_coord[1] * sub_loop_n + jj
            mc_ptr = frgWs_mc[None, i, j].iterator
            x, y, z, w = ld_reduce(mc_ptr)
            tmp_results[0, ii, jj] = x
            tmp_results[1, ii, jj] = y
            tmp_results[2, ii, jj] = z
            tmp_results[3, ii, jj] = w
    tmp_rD = cute.recast_tensor(tmp_results, frgWs_mc.element_type)
    tRS_rD.store(tmp_rD.load().to(tRS_rD.element_type))


def _subtile_to_dtype(tRS_rD, dtype):
    """d_dtype-converted register copy (tRS_rD itself when dtypes already match)."""
    if const_expr(tRS_rD.element_type == dtype):
        return tRS_rD
    tmp_out = cute.make_rmem_tensor(tRS_rD.layout.shape, dtype)
    tmp_out.store(tRS_rD.load().to(dtype))
    return tmp_out


@cute.jit
def commit_subtile_local(
    frgD: cute.Tensor,
    frgD_crd: cute.Tensor,
    row_limit: Int32,
    col_limit: Int32,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
) -> None:
    """reduce_scatter commit: vectorized stores of the reduced, post-EVT subtile
    into this rank's plain slab-shaped D at slab-local coords. Passed to epilogue()
    as commit_D. Owns d_dtype conversion and edge predication: D has no padding, so
    skip rows past the slab tail and cols past N (n-major D: an OOB column wraps
    into the next row)."""
    _atom, chunk, sub_loop_n = tRS_rD.shape
    tmp_out = _subtile_to_dtype(tRS_rD, frgD.element_type)
    for ii in cutlass.range_constexpr(chunk):
        i = epi_coord[0] * chunk + ii
        for jj in cutlass.range_constexpr(sub_loop_n):
            j = epi_coord[1] * sub_loop_n + jj
            crd = frgD_crd[((0, 0), i, j)]
            if crd[0] < row_limit and crd[1] < col_limit:
                cute.autovec_copy(tmp_out[None, ii, jj], frgD[None, i, j])


@cute.jit
def commit_subtile_broadcast(
    frgD_mc: cute.Tensor,
    frgD_crd: cute.Tensor,
    row_limit: Int32,
    col_limit: Int32,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
) -> None:
    """all_reduce commit: multimem_st broadcast of the reduced, post-EVT subtile to
    every rank's symmetric D. Passed to epilogue() as commit_D. Same d_dtype
    conversion and edge predication as the local commit (D is exactly (m, n, l))."""
    _atom, chunk, sub_loop_n = tRS_rD.shape
    tmp_out = _subtile_to_dtype(tRS_rD, frgD_mc.element_type)
    out_i32 = cute.recast_tensor(tmp_out, cutlass.Int32)
    for ii in cutlass.range_constexpr(chunk):
        i = epi_coord[0] * chunk + ii
        for jj in cutlass.range_constexpr(sub_loop_n):
            j = epi_coord[1] * sub_loop_n + jj
            crd = frgD_crd[((0, 0), i, j)]
            if crd[0] < row_limit and crd[1] < col_limit:
                utils.distributed.multimem_st_4xb32(
                    frgD_mc[None, i, j].iterator,
                    out_i32[0, ii, jj].ir_value(),
                    out_i32[1, ii, jj].ir_value(),
                    out_i32[2, ii, jj].ir_value(),
                    out_i32[3, ii, jj].ir_value(),
                )
