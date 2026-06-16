# Copyright (c) 2025, Tri Dao.

from typing import NamedTuple, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32, Float32, Boolean, const_expr
from cutlass._mlir.dialects import nvvm
from cutlass.cute.experimental import iket


import quack.utils as utils
from quack.fast_math import FastDivmod
from quack.pipeline import PipelineStateWAdvance
from quack.cute_dsl_utils import mlir_namedtuple


class RasterOrderOption(IntEnum):
    AlongM = 0
    AlongN = 1
    Heuristic = 2  # Pick AlongM if tiles_n > tiles_m, else AlongN


class RasterOrder(IntEnum):
    AlongM = 0
    AlongN = 1


class PersistenceMode(IntEnum):
    NONE = 0
    STATIC = 1
    DYNAMIC = 2
    # Cluster-launch-control work stealing, with the try_cancel response multicast
    # by hardware into every CTA's smem; each consumer warp decodes + swizzles
    # locally. The work idx comes from the canceled cluster's x coordinate rather
    # than a persistent linear counter in the z coordinate.
    CLC = 3


# Bytes per sched_smem stage slot: 4 Int32 — either the STAS-broadcast
# (pid_m, pid_n, batch_idx, is_valid) or the CLC try_cancel response. Also the
# expect_tx count both producers arm on the full barrier.
SCHED_SLOT_BYTES = 16

# Cap on fire-and-forget try_cancels a retiring cluster sprays to drain the pending
# pool tail (see TileScheduler.cancel_pending_tail). There is no loop: looping
# "until the pool is empty" requires observing responses, which is the synchronous
# drain this design replaces — a fixed blind count is the only non-observing option,
# and the launched-straggler cascade acts as the loop across generations.
#
# Sizing rule: CLC_DRAIN_CANCELS * num_resident_clusters >= typical padding, so the
# residents' first volley covers the pool in generation zero. Varlen padding is
# bounded by (L - 1) M-slots * ncluster_n (~2k for L=512, N-tiles=8); 32 * ~74
# residents = ~2.4k covers it (traced: only ~19 cancel/launch-race stragglers
# launch, no generational waves). Bounded from above by three costs: (1) on
# exact-grid kernels (dense/symmetric) the pool is already empty at retirement, so
# ALL sprays fail — a per-kernel tax that must stay cheap (measured free at 32);
# (2) the issuer stalls on async-proxy backpressure while enqueueing, holding its
# SM slot and delaying kernel end when there is nothing left to cancel; (3) past
# the pool size, extra cancels buy nothing.
#
# The budget is dynamic between these bounds (blog-style tiering, see
# cancel_pending_tail): a retiring cluster estimates the remaining tail from the
# phantom index it just decoded (tail <= grid_total - w) and sprays
# ceil(tail / max_active_clusters), so block-aligned seqlens (maximal padding,
# ~2x the random-length average) still drain in generation zero. The MIN keeps
# the estimate-free fallback; the MAX bounds the enqueue-backpressure stall a
# retiring cluster's SM slot endures (~256 * ~8ns = ~2us) — beyond it, extra
# generations (~20us empty-cluster waves) are cheaper than deeper stalls, and the
# batched-spray-plus-one-peek design is the real upgrade path.
CLC_DRAIN_CANCELS_MIN = 32
CLC_DRAIN_CANCELS_MAX = 256


@cute.jit
def cluster_idx_from_block_idx(
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape], *, loc=None, ip=None
) -> Tuple[Int32, Int32, Int32]:
    """blockIdx // cluster_shape with the cluster shape as a compile-time constant.
    cute.arch.cluster_idx() divides by the *runtime* cluster dims from special
    registers, which lowers to an I2F/FMUL/F2I float-reciprocal chain per component;
    the constexpr division here is a shift (or compile-time magic) instead."""
    bidx = cute.arch.block_idx()
    return tuple(
        Int32(b) if const_expr(s == 1) else Int32(Uint32(b) // s)
        for b, s in zip(bidx, cluster_shape_mnk)
    )


@cute.jit
def get_raster_order_from_option(
    raster_order_option: RasterOrderOption, problem_shape_ncluster_mn: cute.Shape, group_size: Int32
) -> RasterOrder:
    raster_order = (
        RasterOrder.AlongM
        if raster_order_option == RasterOrderOption.AlongM
        else RasterOrder.AlongN
    )
    if raster_order_option == RasterOrderOption.Heuristic:
        problem_blocks_m = cute.round_up(problem_shape_ncluster_mn[0], group_size)
        problem_blocks_n = cute.round_up(problem_shape_ncluster_mn[1], group_size)
        raster_order = (
            RasterOrder.AlongM if problem_blocks_n > problem_blocks_m else RasterOrder.AlongN
        )
    return raster_order


class WorkTileInfo:
    """Drop-in generalization of WorkTileInfo for split-K.

    tile_idx is (pid_m, pid_n, split_idx, batch_idx), with split_idx a dynamic value
    only when num_split_k > 1 (a static None otherwise, keeping the non-split case
    identical to the DSL class). The DSL's WorkTileInfo asserts exactly 4 dynamic
    loop-carried values in __new_from_mlir_values__; this one derives the counts from
    the template so the decoded split-K coordinate can cross persistent-loop
    boundaries.
    """

    def __init__(self, tile_idx: cute.Coord, is_valid_tile: Boolean):
        self._tile_idx = tile_idx
        self._is_valid_tile = Boolean(is_valid_tile)

    @property
    def tile_idx(self) -> cute.Coord:
        return self._tile_idx

    @property
    def is_valid_tile(self) -> Boolean:
        return self._is_valid_tile

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self._tile_idx)
        values.extend(cutlass.extract_mlir_values(self._is_valid_tile))
        return values

    def __new_from_mlir_values__(self, values):
        num_tile_idx_values = len(cutlass.extract_mlir_values(self._tile_idx))
        return WorkTileInfo(
            cutlass.new_from_mlir_values(self._tile_idx, values[:num_tile_idx_values]),
            cutlass.new_from_mlir_values(self._is_valid_tile, values[num_tile_idx_values:]),
        )


# Grouping arguments together that should be passed to __call__
@mlir_namedtuple
class TileSchedulerOptions(NamedTuple):
    max_active_clusters: Int32
    raster_order: cutlass.Constexpr[RasterOrderOption] = RasterOrderOption.Heuristic
    max_swizzle_size: Int32 = Int32(8)
    tile_count_semaphore: Optional[cute.Pointer] = None
    batch_idx_permute: Optional[cute.Tensor] = None


@dataclass
class TileSchedulerArguments:
    problem_shape_ntile_mnl: cute.Shape
    raster_order: cutlass.Constexpr[RasterOrderOption]
    group_size: Int32
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    tile_count_semaphore: Optional[cute.Pointer] = None
    batch_idx_permute: Optional[cute.Tensor] = None
    persistence_mode: cutlass.Constexpr[PersistenceMode] = PersistenceMode.NONE
    # Split-K: the L (z) dimension of the work-id space is multiplied by num_split_k, with the
    # split index as the fastest-varying component. problem_shape_ntile_mnl[2] stays the true L.
    num_split_k: cutlass.Constexpr[int] = 1


class TileScheduler:
    # Whether the launched grid can exceed the real work, i.e. whether padding work
    # indices exist. Exact-grid schedulers retire only on pool-empty (every granted
    # steal is a real tile), so the retirement cancel spray is dead code for them;
    # the varlen scheduler over-provisions (worst-case per-batch padding, see its
    # get_grid_shape) and overrides this.
    grid_may_exceed_work: bool = False

    @dataclass
    class Params:
        problem_shape_ncluster_mnl: cute.Shape
        raster_order: RasterOrder
        num_clusters_per_problem_fdd: FastDivmod
        num_groups_regular: Int32
        group_size_fdd: FastDivmod
        group_size_tail_fdd: FastDivmod
        num_clusters_in_group_fdd: FastDivmod
        tile_count_semaphore: Optional[cute.Pointer]
        batch_idx_permute: Optional[cute.Tensor]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]
        num_split_k: cutlass.Constexpr[int] = 1

        @staticmethod
        @cute.jit
        def create(args: TileSchedulerArguments, *, loc=None, ip=None) -> "TileScheduler.Params":
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = (
                cute.ceil_div(problem_shape_ntile_mn[0], args.cluster_shape_mnk[0]),
                cute.ceil_div(problem_shape_ntile_mn[1], args.cluster_shape_mnk[1]),
            )
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            num_clusters_per_problem = cute.size(problem_shape_ncluster_mn)
            raster_order = get_raster_order_from_option(
                args.raster_order, problem_shape_ncluster_mn, args.group_size
            )
            ncluster_fast = (
                problem_shape_ncluster_mn[0]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[1]
            )
            ncluster_slow = (
                problem_shape_ncluster_mn[1]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[0]
            )
            group_size = min(args.group_size, ncluster_fast)
            group_size_tail = ncluster_fast % group_size
            num_groups_regular = ncluster_fast // group_size
            num_clusters_in_group = group_size * ncluster_slow
            if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC):
                assert args.tile_count_semaphore is not None
            return TileScheduler.Params(
                problem_shape_ncluster_mnl,
                raster_order,
                FastDivmod(num_clusters_per_problem),
                num_groups_regular,
                FastDivmod(group_size),
                # Don't divide by 0
                FastDivmod(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod(num_clusters_in_group),
                args.tile_count_semaphore
                if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC)
                else None,
                args.batch_idx_permute,
                args.cluster_shape_mnk,
                args.persistence_mode,
                args.num_split_k,
            )

    def __init__(
        self,
        current_work_idx: Int32,
        num_tiles_executed: Int32,
        current_batch_idx: Int32,
        num_work_idx_before_cur_batch: Int32,
        cur_batch_end: Int32,
        cur_num_clusters_m: Int32,
        sched_smem: Optional[cute.Tensor],
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        pipeline_state: PipelineStateWAdvance,
        throttle_barrier: Optional[cutlass.pipeline.NamedBarrier],
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        self._current_work_idx = current_work_idx
        self.num_tiles_executed = num_tiles_executed
        self._current_batch_idx = current_batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch
        # Varlen fast-path cache: work-idx end (exclusive) and M-cluster count of the
        # batch resolved by the previous delinearize. A steal landing inside
        # [_num_work_idx_before_cur_batch, _cur_batch_end) skips the warp-cooperative
        # cu_seqlens window scan entirely. Unused (constant 0) for dense schedulers.
        self._cur_batch_end = cur_batch_end
        self._cur_num_clusters_m = cur_num_clusters_m
        self._sched_smem = sched_smem
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self._throttle_barrier = throttle_barrier
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return TileScheduler.Params.create(args, loc=loc, ip=ip)

    def _producer_state(self) -> PipelineStateWAdvance:
        """Producer-side view of this warp's consumer pipeline state: same stage
        index/count, phase flipped — the producer's phase is always the consumer's
        phase ^ 1, since each slot is filled exactly once per consume cycle."""
        return PipelineStateWAdvance(
            self._pipeline_state.stages,
            self._pipeline_state.count,
            self._pipeline_state.index,
            self._pipeline_state.phase ^ 1,
        )

    @staticmethod
    @cute.jit
    def _cluster_idx_to_work_idx_batch(
        params: Params, cluster_idx: Tuple[Int32, Int32, Int32], *, loc=None, ip=None
    ) -> Tuple[Int32, Optional[Int32]]:
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            current_work_idx = Int32(cluster_idx[0])
            batch_idx = Int32(cluster_idx[2])
            return current_work_idx, batch_idx
        else:
            current_work_idx = Int32(cluster_idx[2])
            batch_idx = None
            return current_work_idx, batch_idx

    @classmethod
    @cute.jit
    def create(
        cls,
        params: Params,
        sched_smem: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        throttle_barrier: Optional[cutlass.pipeline.NamedBarrier] = None,
        *,
        loc=None,
        ip=None,
    ) -> "TileScheduler":
        """Shared by all scheduler subclasses (cls dispatches Params and
        _cluster_idx_to_work_idx_batch overrides). is_scheduler_warp should only be
        true for one warp in the whole cluster."""
        cluster_idx = cluster_idx_from_block_idx(params.cluster_shape_mnk, loc=loc, ip=ip)
        current_work_idx, _ = cls._cluster_idx_to_work_idx_batch(
            params, cluster_idx, loc=loc, ip=ip
        )
        stages = 0
        if const_expr(params.persistence_mode != PersistenceMode.NONE):
            assert sched_smem is not None
            assert scheduler_pipeline is not None
            stages = const_expr(cute.size(sched_smem, mode=[1]))
        return cls(
            current_work_idx,
            Int32(0),  # num_tiles_executed
            Int32(0),  # current_batch_idx
            Int32(0),  # num_work_idx_before_cur_batch
            Int32(0),  # cur_batch_end (empty window: first delinearize takes the scan)
            Int32(0),  # cur_num_clusters_m
            sched_smem,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            throttle_barrier,
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            return (
                params.cluster_shape_mnk[0] * cute.size(params.problem_shape_ncluster_mnl[:2]),
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2]
                * params.problem_shape_ncluster_mnl[2]
                * params.num_split_k,
            )
        else:
            num_ctas_in_problem = (
                cute.size(params.problem_shape_ncluster_mnl, loc=loc, ip=ip)
                * cute.size(params.cluster_shape_mnk)
                * params.num_split_k
            )
            num_ctas_per_cluster = cute.size(params.cluster_shape_mnk, loc=loc, ip=ip)
            # Total ctas that can run in one wave
            num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
            num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
            return (
                params.cluster_shape_mnk[0],
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2] * num_persistent_clusters,
            )

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        # CTA Swizzle to promote L2 data reuse
        params = self.params
        group_id, id_in_group = divmod(cluster_id_in_problem, params.num_clusters_in_group_fdd)
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_fdd)
        else:  # tail part
            cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_tail_fdd)
        if group_id % 2 == 1:  # serpentine order
            ncluster_slow = (
                params.problem_shape_ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else params.problem_shape_ncluster_mnl[0]
            )
            cid_slow = ncluster_slow - 1 - cid_slow
        cid_fast = group_id * params.group_size_fdd.divisor + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n

    @cute.jit
    def _cluster_id_to_cta_id(
        self, cid_m: Int32, cid_n: Int32, *, block_zero_only: bool = False, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        if const_expr(
            block_zero_only or cute.size(self.params.cluster_shape_mnk, loc=loc, ip=ip) == 1
        ):
            bidx_in_cluster = (Int32(0), Int32(0))
        else:
            # Get the pid from cluster id
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.params.cluster_shape_mnk[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.params.cluster_shape_mnk[1] + bidx_in_cluster[1]
        return pid_m, pid_n

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> WorkTileInfo:
        params = self.params
        if const_expr(is_valid is None):
            if const_expr(params.persistence_mode == PersistenceMode.NONE):
                is_valid = self.num_tiles_executed == 0
            elif const_expr(params.persistence_mode == PersistenceMode.CLC):
                is_valid = work_idx < cute.size(params.problem_shape_ncluster_mnl[:2])
            else:
                is_valid = (
                    work_idx < cute.size(params.problem_shape_ncluster_mnl) * params.num_split_k
                )
        pid_m, pid_n, batch_idx = Int32(0), Int32(0), Int32(0)
        split_idx = Int32(0) if const_expr(params.num_split_k != 1) else None
        if is_valid:
            if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
                cluster_id_in_problem = work_idx
                bidz_ = (
                    bidz
                    if const_expr(bidz is not None)
                    else cluster_idx_from_block_idx(params.cluster_shape_mnk, loc=loc, ip=ip)[2]
                )
            else:
                if const_expr(params.num_split_k == 1):
                    bidz_, cluster_id_in_problem = divmod(
                        work_idx, params.num_clusters_per_problem_fdd
                    )
                else:
                    # Split index is the fastest-varying component of the linear work id, so all
                    # splits of one output tile are temporally adjacent (L2 + semaphore wait).
                    work_idx_tile = work_idx // params.num_split_k
                    split_idx = work_idx - work_idx_tile * params.num_split_k
                    l_idx, cluster_id_in_problem = divmod(
                        work_idx_tile, params.num_clusters_per_problem_fdd
                    )
                    # bidz_ carries the combined (l, split) index, like grid z in NONE/CLC modes.
                    bidz_ = l_idx * params.num_split_k + split_idx
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
            pid_m, pid_n = self._cluster_id_to_cta_id(
                cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
            )
            if const_expr(params.num_split_k == 1):
                batch_idx = (
                    bidz_
                    if const_expr(params.batch_idx_permute is None)
                    else params.batch_idx_permute[bidz_]
                )
            else:
                # bidz_ is the combined l * num_split_k + split index; the scheduler hands
                # back the DECODED coordinate (split in the k slot, the true batch in the
                # l slot). Permute applies to l only.
                l_idx = bidz_ // params.num_split_k
                split_idx = bidz_ - l_idx * params.num_split_k
                if const_expr(params.batch_idx_permute is not None):
                    l_idx = params.batch_idx_permute[l_idx]
                batch_idx = l_idx
        tile_coord_mnkl = (pid_m, pid_n, split_idx, batch_idx)
        return WorkTileInfo(tile_coord_mnkl, is_valid)

    @cute.jit
    def get_split_k_tile_range(
        self, k_tile_total: Int32, split_idx: Optional[Int32], *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        """K-tile subrange [start, start + cnt) owned by split_idx.

        The scheduler is the sole owner of the K-dim work decomposition: every warp
        (load, MMA, epilogue, pingpong bookkeeping) must derive its k-tile count from
        this one method, or the AB pipeline producer/consumer counts desync and hang.
        The first (k_tile_total % num_split_k) splits get one extra k-tile (balanced,
        same as the CUTLASS 3.x scheduler). Splits beyond k_tile_total get an empty
        range; they still run the epilogue (zero contribution) so the serial-mode
        semaphore turnstile advances.
        """
        num_split_k = self.params.num_split_k
        if const_expr(num_split_k == 1):
            return 0, k_tile_total
        k_tiles_per_split = k_tile_total // num_split_k
        remainder = k_tile_total - k_tiles_per_split * num_split_k
        k_tile_start = split_idx * k_tiles_per_split + cutlass.min(split_idx, remainder)
        k_tile_cnt = k_tiles_per_split + Int32(split_idx < remainder)
        return k_tile_start, k_tile_cnt

    @cute.jit
    def get_combined_batch_idx(
        self, batch_idx: Int32, split_idx: Optional[Int32], *, loc=None, ip=None
    ) -> Int32:
        """Inverse of the work-tile (l, split) decode: the combined l * num_split_k +
        split index. Staged split-K uses it as the partials-workspace batch coordinate."""
        if const_expr(self.params.num_split_k == 1):
            return batch_idx
        return batch_idx * self.params.num_split_k + split_idx

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        params = self.params
        if const_expr(params.persistence_mode == PersistenceMode.CLC):
            return self._get_current_work_clc(loc=loc, ip=ip)
        pid_m, pid_n, batch_idx, is_valid = Int32(0), Int32(0), Int32(0), Boolean(False)
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            pass
        else:
            iket.range_push("fetch_wait")
            self._scheduler_pipeline.consumer_wait(self._pipeline_state)
            iket.range_pop()
            iket.range_push("fetch_decode")
            pid_m, pid_n, batch_idx, is_valid_i32 = [
                self._sched_smem[i, self._pipeline_state.index] for i in range(4)
            ]
            # Need this fence since the STAS from the producer is using the async proxy.
            # Without this, we get race condition / deadlock.
            if const_expr(cute.size(params.cluster_shape_mnk) > 1):
                cute.arch.fence_view_async_shared()
            self._scheduler_pipeline.consumer_release(self._pipeline_state)
            self._pipeline_state.advance()
            is_valid = Boolean(is_valid_i32)
            iket.range_pop()
        split_idx = None
        if const_expr(params.num_split_k != 1):
            # The 4-int smem record carries the combined (l, split) index in the batch slot.
            zc = batch_idx
            batch_idx = zc // params.num_split_k
            split_idx = zc - batch_idx * params.num_split_k
        tile_coord_mnkl = (pid_m, pid_n, split_idx, batch_idx)
        return WorkTileInfo(tile_coord_mnkl, Boolean(is_valid))

    @cute.jit
    def _get_current_work_clc(self, *, loc=None, ip=None) -> WorkTileInfo:
        """Consumer side of the multicast CLC pipeline, called by every consumer warp
        in every CTA of the cluster. The hardware has multicast the 16-byte CLC response
        into this CTA's smem slot (completing the local full barrier), so each warp
        decodes the response and computes the swizzle itself instead of reading
        coordinates decoded once by the scheduler warp."""
        params = self.params
        iket.range_push("fetch_wait")
        self._scheduler_pipeline.consumer_wait(self._pipeline_state)
        iket.range_pop()
        iket.range_push("fetch_decode")
        clc_response_ptr = self._sched_smem[None, self._pipeline_state.index].iterator
        bidx, bidy, bidz, valid = cute.arch.clc_response(clc_response_ptr, loc=loc, ip=ip)
        # The CLC response is written by the async proxy; fence so our generic-proxy
        # read is ordered before the release below lets the producer's next CLC
        # query overwrite the slot.
        cute.arch.fence_view_async_shared()
        self._scheduler_pipeline.consumer_release(self._pipeline_state)
        self._pipeline_state.advance()
        # Deliberately decode/swizzle AFTER the release: only the b128 response load
        # needs the slot; freeing it here lets the scheduler warp recycle the stage
        # for the next query while this warp runs the (possibly expensive, e.g.
        # varlen scan) delinearization.
        cluster_idx = (
            Int32(Uint32(bidx) // params.cluster_shape_mnk[0]),
            Int32(Uint32(bidy) // params.cluster_shape_mnk[1]),
            Int32(Uint32(bidz) // params.cluster_shape_mnk[2]),
        )
        work_idx, batch_idx = self._cluster_idx_to_work_idx_batch(params, cluster_idx)
        # Remember the last decoded work index: at retirement it is the first phantom
        # this cluster saw, giving cancel_pending_tail its remaining-tail estimate.
        self._current_work_idx = work_idx
        ret = self._delinearize_work_idx(work_idx, batch_idx, Boolean(valid), loc=loc, ip=ip)
        iket.range_pop()
        return ret

    @cute.jit
    def _issue_clc_query_multicast(self, *, loc=None, ip=None) -> None:
        """Producer side of the multicast CLC pipeline; called only by the scheduler
        warp of CTA 0 in the cluster. Waits for all consumers (cluster-wide) to have
        released the slot, arms every CTA's full barrier with a 16-byte transaction,
        then issues one multicast CLC query. No STAS re-broadcast: the response lands
        in all CTAs' smem directly from the hardware."""
        params = self.params
        pipeline_state_producer = self._producer_state()
        self._scheduler_pipeline.producer_acquire(pipeline_state_producer)
        mbar_ptr = self._scheduler_pipeline.producer_get_barrier(pipeline_state_producer)
        lane_idx = cute.arch.lane_idx()
        if lane_idx < cute.size(params.cluster_shape_mnk):
            # Arm each CTA's full barrier: fused arrive (count 1, matching the
            # producer group) + expect_tx(16) for the multicast response.
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, SCHED_SLOT_BYTES, lane_idx)
        clc_response_ptr = self._sched_smem[None, self._pipeline_state.index].iterator
        with cute.arch.elect_one():
            cute.arch.issue_clc_query(mbar_ptr, clc_response_ptr, multicast=True, loc=loc, ip=ip)

    @cute.jit
    def throttle_producer_commit(
        self, is_producer_warp: bool | Boolean = True, *, loc=None, ip=None
    ) -> None:
        """Called once per work tile by the main load warp (CTA 0 of the cluster only),
        before it starts issuing the tile's loads. Signals the scheduler warp that one
        more multicast CLC query may be issued."""
        if const_expr(self._throttle_barrier is not None):
            if is_producer_warp:
                self._throttle_barrier.arrive()

    @cute.jit
    def cancel_pending_tail(self, *, loc=None, ip=None) -> None:
        """Fire-and-forget drain of the pending-cluster tail, called by the scheduler
        warp when its persistent loop exits (i.e. a steal decoded to an invalid tile).

        CORRECTNESS ASSUMPTION (grant monotonicity): once any fetch decodes into the
        invalid/padding region, no pending cluster maps to real work — so canceling
        arbitrary pending clusters without inspecting them is safe. PTX does not
        document try_cancel grant order; this holds for the observed FIFO-ish drain
        and is the same assumption made by the capped spray-and-pray drain in
        https://drisspg.github.io/nuggets/A-Tale-of-Two-Schedulers (which hits this
        problem at up to 64x padding in capacity-sized grouped GEMM). If it were
        violated, a real tile could be canceled unprocessed.

        Fires CLC_DRAIN_CANCELS non-multicast try_cancels at issue rate with no
        response waits (responses land in the dead stage-0 slot, tx pre-armed so the
        barrier stays balanced; nobody observes either again). The cancel requests
        outlive this cluster: their pool-removal effect happens at the work
        distributor whether or not the issuer is still resident; only the (unread)
        response write-back is orphaned by the exit.

        Pending clusters that launch anyway (cancel/launch races at retirement, or
        padding beyond the residents' first volley) see an invalid initial tile,
        skip their loop, and spray again on exit. Launches are gated by SM capacity
        and the sprayers die near-simultaneously (shared CWD backlog), so
        stragglers arrive in machine-width waves, each min(num_residents,
        remaining pool) clusters and costing ~one empty-cluster lifetime — a
        decaying cascade instead of the full launch stampede. See
        CLC_DRAIN_CANCELS for the cap sizing and why there is no drain loop."""
        if const_expr(
            self.params.persistence_mode == PersistenceMode.CLC and self.grid_may_exceed_work
        ):
            params = self.params
            # Remaining tail <= total work indices - the phantom index we just drew;
            # split it across the resident clusters, which all retire around now.
            grid_total = Int32(Uint32(cute.arch.grid_dim()[0]) // params.cluster_shape_mnk[0])
            tail = grid_total - self._current_work_idx
            budget = cutlass.min(
                Int32(CLC_DRAIN_CANCELS_MAX),
                cutlass.max(
                    Int32(CLC_DRAIN_CANCELS_MIN),
                    (tail + params.max_active_clusters - 1) // params.max_active_clusters,
                ),
            )
            state0 = PipelineStateWAdvance(
                self._pipeline_state.stages, Int32(0), Int32(0), Int32(0)
            )
            mbar_ptr = self._scheduler_pipeline.producer_get_barrier(state0)
            resp_ptr = self._sched_smem[None, 0].iterator
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, SCHED_SLOT_BYTES * budget)
                for _ in cutlass.range(budget):
                    cute.arch.issue_clc_query(mbar_ptr, resp_ptr, multicast=False, loc=loc, ip=ip)

    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self._delinearize_work_idx(self._current_work_idx, loc=loc, ip=ip)

    @cute.jit
    def _fetch_next_work_idx(self, *, loc=None, ip=None) -> Int32:
        """should only be called by the scheduler warp"""
        params = self.params
        num_persistent_clusters = Int32(
            Uint32(cute.arch.grid_dim()[2]) // params.cluster_shape_mnk[2]
        )
        if const_expr(params.persistence_mode == PersistenceMode.STATIC):
            return self._current_work_idx + num_persistent_clusters
        elif const_expr(params.persistence_mode == PersistenceMode.DYNAMIC):
            next_work_linear_idx = Int32(0)
            if cute.arch.lane_idx() == 0:
                # If varlen_m, problem_shape_ncluster_mnl[0] is None, so we use atomic_add
                # instead of atomic_inc, and at the end of the kernel must reset the semaphore to 0.
                if const_expr(params.problem_shape_ncluster_mnl[0] is not None):
                    next_work_linear_idx = num_persistent_clusters + Int32(
                        nvvm.atomicrmw(
                            op=nvvm.AtomicOpKind.INC,
                            ptr=params.tile_count_semaphore.llvm_ptr,
                            a=Int32(
                                cute.size(params.problem_shape_ncluster_mnl) * params.num_split_k
                                - 1
                            ).ir_value(),
                            loc=loc,
                            ip=ip,
                        )
                    )
                else:  # varlen_m
                    next_work_linear_idx = num_persistent_clusters + cute.arch.atomic_add(
                        params.tile_count_semaphore, Int32(1), loc=loc, ip=ip
                    )
            return cute.arch.shuffle_sync(next_work_linear_idx, 0)

    @cute.jit
    def write_work_tile_to_smem(self, work_tile_info: WorkTileInfo, *, loc=None, ip=None):
        params = self.params
        if const_expr(self._sched_smem is not None):
            pipeline_state_producer = self._producer_state()
            self._scheduler_pipeline.producer_acquire(pipeline_state_producer)
            batch_field = work_tile_info.tile_idx[3]
            if const_expr(params.num_split_k != 1):
                # The 4-int smem record carries the combined (l, split) index; consumers
                # decode it back in get_current_work.
                batch_field = batch_field * params.num_split_k + work_tile_info.tile_idx[2]
            sched_data = [
                work_tile_info.tile_idx[0],
                work_tile_info.tile_idx[1],
                batch_field,
                Int32(work_tile_info.is_valid_tile),
            ]
            lane_idx = cute.arch.lane_idx()
            if lane_idx < cute.size(params.cluster_shape_mnk):
                pipeline_idx = self._pipeline_state.index
                if const_expr(cute.size(params.cluster_shape_mnk) == 1):
                    for i in cutlass.range_constexpr(4):
                        self._sched_smem[i, pipeline_idx] = sched_data[i]
                    self._scheduler_pipeline.producer_commit(self._pipeline_state)
                else:
                    peer_cta_rank_in_cluster = lane_idx
                    # Here we assume that the block idx in cluster is linearized such that
                    # x is the fastest moving direction, followed by y, then z.
                    bidx_in_cluster = peer_cta_rank_in_cluster % params.cluster_shape_mnk[0]
                    bidy_in_cluster = (
                        peer_cta_rank_in_cluster // params.cluster_shape_mnk[0]
                    ) % params.cluster_shape_mnk[1]
                    mbar_ptr = self._scheduler_pipeline.producer_get_barrier(self._pipeline_state)
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar_ptr, SCHED_SLOT_BYTES, peer_cta_rank_in_cluster
                    )
                    utils.store_shared_remote_x4(
                        sched_data[0] + bidx_in_cluster,
                        sched_data[1] + bidy_in_cluster,
                        sched_data[2],
                        sched_data[3],
                        smem_ptr=self._sched_smem[None, pipeline_idx].iterator,
                        mbar_ptr=mbar_ptr,
                        peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                    )

    @cute.jit
    def advance_to_next_work(
        self,
        is_scheduler_warp: bool | Boolean = False,
        *,
        advance_count: int = 1,
        loc=None,
        ip=None,
    ):
        """Called by every consumer warp; only the producer work (fetch/query) is
        gated on is_scheduler_warp, which must be true for exactly one warp in the
        whole cluster (CTA 0's scheduler warp). If calling with
        is_scheduler_warp=True, advance_count must be 1."""
        params = self.params
        self.num_tiles_executed += Int32(advance_count)
        if const_expr(self._pipeline_state is not None and advance_count > 1):
            self._pipeline_state.advance_iters(advance_count - 1)
        if const_expr(params.persistence_mode in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC]):
            # We assume here that advance_count is 1 for scheduler_warp
            if is_scheduler_warp:
                self._current_work_idx = self._fetch_next_work_idx(loc=loc, ip=ip)
                work_tile_info = self._delinearize_work_idx(
                    self._current_work_idx, block_zero_only=True, loc=loc, ip=ip
                )
                self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)
        elif const_expr(params.persistence_mode == PersistenceMode.CLC):
            # We assume here that advance_count is 1 for scheduler_warp
            if is_scheduler_warp:
                if const_expr(self._throttle_barrier is not None):
                    # Throttle: pace queries to tiles actually started by the load warp.
                    # Without this, the multi-stage lookahead lets a cluster issue queries
                    # at CLC-round-trip cadence (~1us) instead of tile cadence,
                    # over-canceling pending clusters and starving other persistent
                    # workers of steals (cutlass's CLCThrottlePipeline serves this purpose
                    # with an mbarrier pipeline). A single named barrier suffices: the
                    # dependency chain (commit k+1 needs fetch k+1 needs query k+1 needs
                    # this sync k) guarantees producer/consumer arrivals strictly
                    # alternate, so at most one credit is ever outstanding. bar.sync also
                    # gives a hardware-scheduled wakeup instead of mbarrier
                    # PHASECHK+NANOSLEEP polling.
                    self._throttle_barrier.arrive_and_wait()
                self._issue_clc_query_multicast(loc=loc, ip=ip)

    def producer_tail(self):
        if const_expr(self._scheduler_pipeline is not None):
            self._scheduler_pipeline.producer_tail(self._producer_state())

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._current_work_idx,
            self.num_tiles_executed,
            self._current_batch_idx,
            self._num_work_idx_before_cur_batch,
            self._cur_batch_end,
            self._cur_num_clusters_m,
            self._sched_smem,
            self._scheduler_pipeline,
            self._pipeline_state,
            self._throttle_barrier,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_idx,
                self.num_tiles_executed,
                self._current_batch_idx,
                self._num_work_idx_before_cur_batch,
                self._cur_batch_end,
                self._cur_num_clusters_m,
                self._sched_smem,
                self._scheduler_pipeline,
                self._pipeline_state,
                self._throttle_barrier,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


@cute.jit
def triangular_idx_to_coord(idx: Int32) -> Tuple[Int32, Int32]:
    """
    Convert a triangular index to 2D coordinates.
    This is used to convert the linear index to 2D coordinates for triangular matrices.
    """
    row = Int32(cute.math.ceil(cute.math.sqrt(2 * idx + 2.25, approx=True) - 0.5)) - 1
    col = idx - (row * (row + 1)) // 2
    return row, col


class TriangularTileScheduler(TileScheduler):
    """We assume the tile size per cluster is square (e.g., 128 x 256 per CTA, with cluster 2 x 1)"""

    @dataclass
    class Params:
        problem_shape_ncluster_mnl: cute.Shape
        num_clusters_per_problem_fdd: FastDivmod
        group_size_inv_f32: Float32
        num_groups_regular: Int32
        group_size_fdd: FastDivmod
        group_size_tail_fdd: FastDivmod
        group_size_mul_group_size_fdd: FastDivmod
        group_size_tail_mul_group_size_fdd: FastDivmod
        tile_count_semaphore: Optional[cute.Pointer]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]
        num_split_k: cutlass.Constexpr[int] = 1

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "TriangularTileScheduler.Params":
            assert args.cluster_shape_mnk[2] == 1
            assert args.num_split_k == 1, "split_k is not supported by TriangularTileScheduler"
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = (
                cute.ceil_div(problem_shape_ntile_mn[0], args.cluster_shape_mnk[0]),
                cute.ceil_div(problem_shape_ntile_mn[1], args.cluster_shape_mnk[1]),
            )
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            cluster_m = problem_shape_ncluster_mn[0]
            # Assume that each cluster is responsible for a square tile
            num_clusters_per_problem = cluster_m * (cluster_m + 1) // 2
            group_size = min(args.group_size, cluster_m)
            group_size_tail = cluster_m % group_size
            num_groups_regular = cluster_m // group_size
            if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC):
                assert args.tile_count_semaphore is not None
            return TriangularTileScheduler.Params(
                problem_shape_ncluster_mnl,
                FastDivmod(num_clusters_per_problem),
                Float32(1.0 / group_size),
                num_groups_regular,
                FastDivmod(group_size),
                # Don't divide by 0
                FastDivmod(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod(group_size * group_size),
                FastDivmod((group_size_tail if group_size_tail > 0 else 1) * group_size),
                args.tile_count_semaphore
                if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC)
                else None,
                args.cluster_shape_mnk,
                args.persistence_mode,
            )

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return TriangularTileScheduler.Params.create(args, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        clusters = (params.num_clusters_per_problem_fdd.divisor, 1)
        num_ctas_mnl = (
            clusters[0] * params.cluster_shape_mnk[0],
            clusters[1] * params.cluster_shape_mnk[1],
            params.cluster_shape_mnk[2] * params.problem_shape_ncluster_mnl[2],
        )
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            return num_ctas_mnl
        else:
            num_ctas_in_problem = cute.size(num_ctas_mnl, loc=loc, ip=ip)
            num_ctas_per_cluster = cute.size(params.cluster_shape_mnk, loc=loc, ip=ip)
            # Total ctas that can run in one wave
            num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
            num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
            return (
                params.cluster_shape_mnk[0],
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2] * num_persistent_clusters,
            )

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        # CTA Swizzle to promote L2 data reuse
        params = self.params
        group_size = params.group_size_fdd.divisor
        group_id = (
            Int32(
                cute.math.ceil(
                    (cute.math.sqrt(2 * cluster_id_in_problem + 2.25, approx=True) - 0.5)
                    * params.group_size_inv_f32
                )
            )
            - 1
        )
        cid_m_start = group_id * group_size
        id_in_group = cluster_id_in_problem - (cid_m_start * (cid_m_start + 1)) // 2
        group_size_actual = (
            group_size
            if group_id < params.num_groups_regular
            else params.group_size_tail_fdd.divisor
        )
        group_col, group_remainder = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            group_col, group_remainder = divmod(id_in_group, params.group_size_mul_group_size_fdd)
        else:  # tail part
            group_col, group_remainder = divmod(
                id_in_group, params.group_size_tail_mul_group_size_fdd
            )
        cid_m_in_group, cid_n_in_group = Int32(0), Int32(0)
        if id_in_group >= group_size_actual * group_size * group_id:  # triangular tail
            cid_m_in_group, cid_n_in_group = triangular_idx_to_coord(group_remainder)
        else:
            if group_id < params.num_groups_regular:
                cid_n_in_group, cid_m_in_group = divmod(group_remainder, params.group_size_fdd)
            else:
                cid_n_in_group, cid_m_in_group = divmod(group_remainder, params.group_size_tail_fdd)
        cid_m = cid_m_start + cid_m_in_group
        cid_n = group_col * group_size + cid_n_in_group
        return cid_m, cid_n

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> WorkTileInfo:
        params = self.params
        if const_expr(is_valid is None):
            if const_expr(params.persistence_mode == PersistenceMode.NONE):
                is_valid = self.num_tiles_executed == 0
            else:
                is_valid = (
                    work_idx
                    < params.num_clusters_per_problem_fdd.divisor
                    * params.problem_shape_ncluster_mnl[2]
                )
        pid_m, pid_n, batch_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
                cluster_id_in_problem = work_idx
                bidz_ = (
                    bidz
                    if const_expr(bidz is not None)
                    else cluster_idx_from_block_idx(params.cluster_shape_mnk, loc=loc, ip=ip)[2]
                )
            else:
                bidz_, cluster_id_in_problem = divmod(work_idx, params.num_clusters_per_problem_fdd)
                cluster_id_in_problem = Int32(cluster_id_in_problem)  # divmod returns IntValue
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
            pid_m, pid_n = self._cluster_id_to_cta_id(
                cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
            )
            batch_idx = bidz_
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        return WorkTileInfo(tile_coord_mnkl, is_valid)


@dataclass
class VarlenMTileSchedulerArguments:
    problem_shape_ntile_mnl: cute.Shape
    total_m: Int32
    cu_seqlens_m: cute.Tensor
    max_active_clusters: Int32
    raster_order: cutlass.Constexpr[RasterOrderOption]
    group_size: Int32
    tile_shape_mn: cutlass.Constexpr[cute.Shape]
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    tile_count_semaphore: Optional[cute.Pointer] = None
    persistence_mode: cutlass.Constexpr[PersistenceMode] = PersistenceMode.NONE
    num_split_k: cutlass.Constexpr[int] = 1


class VarlenMTileScheduler(TileScheduler):
    grid_may_exceed_work: bool = True

    @dataclass
    class Params:
        problem_shape_ncluster_mnl: cute.Shape
        total_m: Int32
        cu_seqlens_m: cute.Tensor
        max_active_clusters: Int32
        raster_order: cutlass.Constexpr[RasterOrder]
        group_size: Int32
        group_size_fdd: Optional[FastDivmod]
        group_size_tail_fdd: Optional[FastDivmod]
        num_clusters_in_group_fdd: FastDivmod
        tile_shape_mn: cutlass.Constexpr[cute.Shape]
        tile_count_semaphore: Optional[cute.Pointer]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]
        num_split_k: cutlass.Constexpr[int] = 1

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "VarlenMTileScheduler.Params":
            assert args.num_split_k == 1, "split_k is not supported by VarlenMTileScheduler"
            # problem_shape_ntile_mnl[0] will be None for VarlenM
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = (
                None,
                cute.ceil_div(problem_shape_ntile_mn[1], args.cluster_shape_mnk[1]),
            )
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            raster_order = const_expr(
                RasterOrder.AlongM
                if args.raster_order == RasterOrderOption.AlongM
                else RasterOrder.AlongN  # For Heuristic we also use AlongN
            )
            ncluster_fast = problem_shape_ncluster_mn[
                0 if raster_order == RasterOrder.AlongM else 1
            ]
            ncluster_slow = problem_shape_ncluster_mn[
                1 if raster_order == RasterOrder.AlongM else 0
            ]
            if const_expr(ncluster_fast is not None):
                group_size = min(args.group_size, ncluster_fast)
                group_size_tail = ncluster_fast % group_size
            else:
                group_size, group_size_tail = args.group_size, None
            num_clusters_in_group = None
            if const_expr(ncluster_slow is not None):
                num_clusters_in_group = group_size * ncluster_slow
            if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC):
                assert args.tile_count_semaphore is not None
            return VarlenMTileScheduler.Params(
                problem_shape_ncluster_mnl,
                args.total_m,
                args.cu_seqlens_m,
                args.max_active_clusters,
                raster_order,
                group_size,
                FastDivmod(group_size) if ncluster_fast is not None else None,
                # Don't divide by 0
                FastDivmod(group_size_tail if group_size_tail > 0 else 1)
                if group_size_tail is not None
                else None,
                FastDivmod(num_clusters_in_group) if num_clusters_in_group is not None else None,
                args.tile_shape_mn,
                args.tile_count_semaphore
                if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC)
                else None,
                args.cluster_shape_mnk,
                args.persistence_mode,
            )

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return VarlenMTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def _cluster_idx_to_work_idx_batch(
        params: Params, cluster_idx: Tuple[Int32, Int32, Int32], *, loc=None, ip=None
    ) -> Tuple[Int32, Optional[Int32]]:
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            current_work_idx = Int32(cluster_idx[0])
        else:
            current_work_idx = Int32(cluster_idx[2])
        batch_idx = None
        return current_work_idx, batch_idx

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        block_size = params.tile_shape_mn[0] * params.cluster_shape_mnk[0]
        num_batch = params.problem_shape_ncluster_mnl[2]
        # Tight upper bound on sum(ceil(len_i / block)) given only (total_m, L):
        # achieved by adversarial lengths ≡ 1 (mod block), so no smaller grid is safe
        # without per-batch seqlens (a too-small grid = tiles with no work index =
        # wrong results under CLC). cancel_pending_tail makes the padding slots cheap.
        total_clusters_m_max = (params.total_m + num_batch * (block_size - 1)) // block_size
        total_clusters_max = total_clusters_m_max * params.problem_shape_ncluster_mnl[1]
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            return (
                params.cluster_shape_mnk[0] * total_clusters_max,
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2],
            )
        else:
            num_persistent_clusters = cutlass.min(max_active_clusters, total_clusters_max)
            return (
                params.cluster_shape_mnk[0],
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2] * num_persistent_clusters,
            )

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, num_clusters_m: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        params = self.params
        # CTA Swizzle to promote L2 data reuse
        if const_expr(params.num_clusters_in_group_fdd is not None):
            group_id, id_in_group = divmod(cluster_id_in_problem, params.num_clusters_in_group_fdd)
            num_clusters_in_group = params.num_clusters_in_group_fdd.divisor
        else:
            assert params.raster_order == RasterOrder.AlongN
            num_clusters_in_group = params.group_size * num_clusters_m
            group_id = cluster_id_in_problem // num_clusters_in_group
            id_in_group = cluster_id_in_problem - group_id * num_clusters_in_group
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if const_expr(params.group_size_fdd is not None and params.group_size_tail_fdd is not None):
            num_clusters = num_clusters_m * params.problem_shape_ncluster_mnl[1]
            if (group_id + 1) * num_clusters_in_group <= num_clusters:
                cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_fdd)
            else:  # tail part
                cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_tail_fdd)
        else:
            assert params.raster_order == RasterOrder.AlongM
            group_size_actual = cutlass.min(
                params.group_size, num_clusters_m - group_id * params.group_size
            )
            cid_slow = id_in_group // group_size_actual
            cid_fast_in_group = id_in_group - cid_slow * group_size_actual
        if group_id % 2 == 1:  # serpentine order
            ncluster_slow = (
                params.problem_shape_ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else num_clusters_m
            )
            cid_slow = ncluster_slow - 1 - cid_slow
        cid_fast = group_id * params.group_size + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n

    @cute.jit
    def _get_num_m_blocks(
        self, lane: Int32, bidb_start: Int32, block_size: cutlass.Constexpr[int]
    ) -> Int32:
        num_batch = self.params.problem_shape_ncluster_mnl[2]
        batch_idx = lane + bidb_start
        cur_cu_seqlen = Int32(0)
        if batch_idx <= num_batch:
            cur_cu_seqlen = self.params.cu_seqlens_m[batch_idx]
        next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
        seqlen = next_cu_seqlen - cur_cu_seqlen
        return (
            cute.ceil_div(seqlen, block_size)
            if batch_idx < num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,  # not used
        is_valid_: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> WorkTileInfo:
        assert bidz is None
        params = self.params
        lane_idx = cute.arch.lane_idx()
        num_batch = self.params.problem_shape_ncluster_mnl[2]
        block_size = params.tile_shape_mn[0] * params.cluster_shape_mnk[0]
        batch_idx = self._current_batch_idx
        next_tile_idx = work_idx

        problems_end_tile = self._num_work_idx_before_cur_batch
        # Pre-init: assigned under a dynamic `if` below, but read outside it (DSL
        # scoping requires the outer definition).
        num_work_idx_before_cur_batch = self._num_work_idx_before_cur_batch
        cur_batch_end = self._cur_batch_end
        num_clusters_m = self._cur_num_clusters_m
        num_clusters_cumulative, clusters_in_problems = Int32(0), Int32(0)
        is_valid = True if const_expr(is_valid_ is None) else is_valid_
        # Fast path: the work index lands in the batch resolved by the previous call
        # (cached window), so skip the warp-cooperative cu_seqlens window scan below.
        # The scan is a long serial dependence chain — gmem cu_seqlens load, warp
        # prefix-sum, ballot, shuffles — and without the cache it re-ran on EVERY
        # fetch (the loop condition seeds from the batch START, which is always
        # <= next_tile_idx). This exists for CLC: under the static scheduler,
        # work_idx is a register recurrence (idx += stride) known a full tile in
        # advance, so the SASS scheduler overlaps the chain with the loop body's
        # stalls (~140 cy/tile exposed, measured); a CLC steal doesn't exist until
        # the response mbarrier is consumed + fence.proxy.async, so nothing can be
        # hoisted and every warp ate the chain at its fetch site (~800-1100
        # cy/steal), a per-tile bubble in the tcgen05 issue stream that cost ~3-9%
        # e2e on mainloop-bound varlen shapes. With the cache, a steal within the
        # current batch decodes as cheaply as the dense scheduler; the scan only
        # runs when the batch actually changes.
        need_scan = (next_tile_idx < num_work_idx_before_cur_batch) | (
            next_tile_idx >= cur_batch_end
        )
        if is_valid:
            if need_scan:
                while problems_end_tile <= next_tile_idx:
                    num_clusters_m = self._get_num_m_blocks(
                        lane_idx, bidb_start=batch_idx, block_size=block_size
                    )
                    num_clusters = num_clusters_m * params.problem_shape_ncluster_mnl[1]
                    num_clusters_cumulative = utils.warp_prefix_sum(num_clusters, lane_idx)
                    # Total number of blocks for the next 31 problems, same for all lanes
                    clusters_in_problems = cute.arch.shuffle_sync(
                        num_clusters_cumulative, cute.arch.WARP_SIZE - 1
                    )
                    problems_end_tile += clusters_in_problems
                    if problems_end_tile <= next_tile_idx:
                        batch_idx += cute.arch.WARP_SIZE - 1
                    if batch_idx >= num_batch:
                        batch_idx = Int32(num_batch)
                        problems_end_tile = next_tile_idx + 1
                if batch_idx < num_batch:
                    problems_start_tile = problems_end_tile - clusters_in_problems
                    # The next problem to process is the first one that does not have
                    # ending tile position that is greater than or equal to tile index.
                    batch_idx_in_problems = cute.arch.popc(
                        cute.arch.vote_ballot_sync(
                            problems_start_tile + num_clusters_cumulative <= next_tile_idx
                        )
                    )
                    batch_idx += batch_idx_in_problems
                    num_clusters_prev_lane = (
                        0
                        if batch_idx_in_problems == 0
                        else cute.arch.shuffle_sync(
                            num_clusters_cumulative, batch_idx_in_problems - 1
                        )
                    )
                    num_clusters_m = cute.arch.shuffle_sync(num_clusters_m, batch_idx_in_problems)
                    num_work_idx_before_cur_batch = problems_start_tile + num_clusters_prev_lane
                    cur_batch_end = (
                        num_work_idx_before_cur_batch
                        + num_clusters_m * params.problem_shape_ncluster_mnl[1]
                    )
        else:
            batch_idx = Int32(num_batch)

        is_valid = batch_idx < num_batch
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            is_valid &= self.num_tiles_executed == 0
        cid_m, cid_n = Int32(0), Int32(0)
        if is_valid:
            cluster_id_in_problem = next_tile_idx - num_work_idx_before_cur_batch
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, num_clusters_m, loc=loc, ip=ip)
        pid_m, pid_n = self._cluster_id_to_cta_id(
            cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
        )
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        self._current_batch_idx = batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch
        self._cur_batch_end = cur_batch_end
        self._cur_num_clusters_m = num_clusters_m
        return WorkTileInfo(tile_coord_mnkl, is_valid)
