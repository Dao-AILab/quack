# Copyright (c) 2026, Tri Dao.

import enum
import math
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Boolean, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import LayoutEnum

import quack.copy_utils as copy_utils
import quack.utils as utils
from quack.cute_dsl_utils import ParamsBase
from quack.gemm_config import SplitKMode
from quack.epi_ops import EpiSmemBytes
from quack.pipeline import PipelineTmaAsync, PipelineTmaCpAsync
from quack.rounding import RoundingMode, epilogue_sr_seed
from quack.tile_scheduler import (
    PersistenceMode,
    TileScheduler,
    TileSchedulerArguments,
    VarlenMTileScheduler,
    VarlenMTileSchedulerArguments,
)
from quack.varlen_utils import VarlenManager


class NamedBarrierGemm(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    # For mainloop load warps to signal that the epilogue load warp can start.
    # This is to avoid loading C too early, interfering with loading A and B.
    EpilogueLoad = enum.auto()
    MmaWG0 = enum.auto()
    MmaWG1 = enum.auto()
    EpiWG0 = enum.auto()
    EpiWG1 = enum.auto()
    TmemPtr = enum.auto()
    # CLC-multicast throttle: CTA0 load warp arrives once per tile started,
    # CTA0 scheduler warp syncs once per CLC query (2 warps, 64 threads).
    ClcThrottle = enum.auto()


class GemmBase:
    """Common non-mainloop pieces shared by GEMM architectures."""

    arch = 0
    # Split-K along the contraction dim. Constexpr (lives on self): split_k == 1 compiles to
    # exactly the non-split kernel. The epilogue (the full epi mixin: alpha, beta*C, bias,
    # activations, aux outputs) runs exactly ONCE per output tile, on the entity that owns
    # the completed f32 sum — CUTLASS-3.x stream-K fixup semantics. Non-finalizing splits
    # run no epilogue at all: they dump raw f32 accumulator fragments into a per-tile
    # workspace region (split_k_partial_commit) and bump the tile's completion flag.
    # SERIAL: commits are turnstile-ordered by split index (bitwise deterministic); the
    #   last split waits flag == S-1, folds the workspace into its accumulator, and runs
    #   the full epilogue.
    # PARALLEL: commits are release-counted in arrival order, no waiting (lowest latency,
    #   NOT deterministic); the last split finalizes identically.
    # SEPARATE: every split runs the (op-less) epilogue storing raw f32 partials to its own
    #   workspace slice; a separate reduction kernel (quack/split_k_reduce.py) sums them
    #   and applies the full epilogue math.
    split_k = 1
    split_k_mode = SplitKMode.SERIAL

    @dataclass
    class EpilogueArguments:
        pass

    EpilogueParams = ParamsBase

    def epi_smem_warp_shape_mnk(self):
        return (self.num_epi_warps, 1, 1)

    def _init_split_k(self, split_k: int, split_k_mode: int):
        """Validate and store the constexpr split-K configuration. Call after self.gather_A."""
        assert split_k >= 1, "split_k must be >= 1"
        assert split_k_mode in tuple(SplitKMode), f"invalid split_k_mode: {split_k_mode}"
        self.split_k = split_k
        self.split_k_mode = SplitKMode(split_k_mode)
        if split_k > 1:
            assert not self.gather_A, "split_k does not support gather_A"

    @cute.jit
    def epilogue(
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
        split_k_ws: Optional[cute.Pointer] = None,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        has_C = const_expr(tRS_rC is not None)
        has_epi_load = const_expr(self.epi_c_stage > 0)
        has_D = const_expr(copy_D is not None)
        use_tma_epi = const_expr(epi_store_pipeline is not None)
        use_tma_c = const_expr(epi_pipeline is not None)
        inline_epi_load = const_expr(copy_C is not None)
        use_stochastic_rounding = const_expr(
            self.rounding_mode == RoundingMode.RS
            and self.acc_dtype == cutlass.Float32
            and self.d_dtype == cutlass.BFloat16
        )

        # Setup aux outputs. Returns a tuple of ``(tiled_copy_r2s,
        # tRS_sAuxOut, copy_aux_out)`` triples — empty for the default
        # epilogue, one entry for the standard ``GemmAct``/``GemmGated``
        # single-output mixins, multiple entries for multi-output mixins
        # (e.g. ``T*tanh`` + ``1-tanh^2`` from one GEMM).
        aux_out_ctxs = self.epi_setup_aux_out(
            params,
            epi_smem_tensors,
            tiled_copy_r2s,
            tiled_copy_t2r,
            tile_coord_mnkl,
            varlen_manager,
            tidx,
        )

        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape, order=(0, 1) if const_expr(self.epi_m_major) else (1, 0)
        )
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
            tRS_rD.layout,
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

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)  # (epi_m, epi_n)
            # Copy from acc to D registers
            load_acc_subtile(tRS_rD, epi_coord)
            if const_expr(split_k_ws is not None):
                # Finalizing split: fold the other splits' raw f32 partials into the
                # accumulator BEFORE any epilogue math. The tile's workspace region is a
                # flat (epi_subtile, thread, fragment) stripe written by
                # split_k_partial_commit with this exact partitioning, so a compact
                # same-shape view lines up element-for-element.
                tRS_gWs = cute.make_tensor(
                    split_k_ws
                    + (epi_idx * self.num_epi_warps * cute.arch.WARP_SIZE + tidx)
                    * cute.size(tRS_rD),
                    cute.make_layout(tRS_rD.shape),
                )
                tRS_rWs = cute.make_rmem_tensor(tRS_rD.shape, self.acc_dtype)
                cute.autovec_copy(tRS_gWs, tRS_rWs)
                tRS_rD.store(tRS_rD.load() + tRS_rWs.load())
            if const_expr(has_epi_load):
                if const_expr(use_tma_c):
                    epi_pipeline.consumer_wait(epi_read_state)
                    if const_expr(has_C):
                        cute.copy(
                            tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC
                        )
                    self.epi_tile_load_s2r(params, epi_tensors, epi_read_state.index)
                    cute.arch.fence_view_async_shared()
                    epi_pipeline.consumer_release(epi_read_state)
                    epi_read_state.advance()
                else:
                    c_buffer = epi_idx % self.epi_c_stage
                    cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, c_buffer], tSR_rC)
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
            # Returns a tuple of register tensors — one per aux output.
            # Length matches ``aux_out_ctxs``. ``()`` for the default
            # epilogue (no aux output).
            tRS_rAuxOuts = self.epi_visit_subtile(params, epi_loop_tensors, tRS_rD, tRS_rC)
            self.epi_end_loop(
                params,
                epi_tensors,
                epi_coord,
                epi_tile,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tile_coord_mnkl,
                varlen_manager,
                tidx,
            )
            # Convert each output to its storage dtype.
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
                    seed = epilogue_sr_seed(
                        epi_loop_tensors.get("sr_seed"),
                        tile_coord_mnkl,
                        num_prev_subtiles + epi_idx,
                    )
                    copy_utils.sr_cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD_cur, seed, tidx)
                else:
                    copy_utils.cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD_cur)
            # Copy each aux output from registers to shared memory. All share
            # the same ``epi_buffer`` index so the s2g TMA stores below happen
            # in lockstep after the fence.
            for i in cutlass.range_constexpr(len(aux_out_ctxs)):
                tiled_copy_aux_out_r2s, tRS_sAuxOut, _ = aux_out_ctxs[i]
                cute.copy(
                    tiled_copy_aux_out_r2s,
                    # Need contiguous for Sm80 and Sm120 where acc layout is ((2, 2), MMA_M, MMA_N)
                    tiled_copy_aux_out_r2s.retile(tRS_rAuxOuts_out[i]).contiguous(),
                    tRS_sAuxOut[None, None, None, epi_buffer],
                )
            if const_expr(use_tma_epi):
                cute.arch.fence_view_async_shared()
                epilogue_barrier.arrive_and_wait()
                if is_tma_warp:
                    if const_expr(has_D):
                        copy_D(src_idx=epi_buffer, dst_idx=epi_coord)
                    for i in cutlass.range_constexpr(len(aux_out_ctxs)):
                        _, _, copy_aux_out = aux_out_ctxs[i]
                        copy_aux_out(src_idx=epi_buffer, dst_idx=epi_coord)
                    epi_store_pipeline.producer_commit()
            else:
                epilogue_barrier.arrive_and_wait()
                if const_expr(has_D):
                    copy_D(src_idx=epi_buffer, dst_idx=epi_coord)
                for i in cutlass.range_constexpr(len(aux_out_ctxs)):
                    _, _, copy_aux_out = aux_out_ctxs[i]
                    copy_aux_out(src_idx=epi_buffer, dst_idx=epi_coord)
                epilogue_barrier.arrive_and_wait()

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

    @cute.jit
    def split_k_partial_commit(
        self,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        epi_tile: cute.Tile,
        ws_ptr: cute.Pointer,
        lock_ptr: cute.Pointer,
        split_idx: Int32,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tidx: Int32,
        is_tma_warp: cutlass.Boolean,
    ) -> None:
        """Non-finalizing split: commit the raw f32 accumulator partial, run no epilogue.

        The tile's workspace region is a flat (epi_subtile, thread, fragment) stripe of
        accumulator fragments — no (m, n) semantics, no predication; the finalizer reads
        it back with the identical partitioning (see the split_k_ws block in epilogue()).
        SERIAL: a turnstile (flag == number of committed splits) orders the f32 adds in
        split order — bitwise deterministic; split 0's plain store initializes the
        region, later splits red.add into it. PARALLEL: the workspace is host-zeroed and
        every split red.adds immediately in arrival order (no waiting, NOT
        deterministic), then release-increments the flag.
        """
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape, order=(0, 1) if const_expr(self.epi_m_major) else (1, 0)
        )
        num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
        frag_elems = cute.size(tRS_rD)
        if const_expr(self.split_k_mode == SplitKMode.SERIAL):
            # Wait until all preceding splits have committed; the barrier broadcasts
            # lane 0's acquire to every writing thread of the group.
            if is_tma_warp:
                utils.semaphore_wait_eq(lock_ptr, split_idx)
            epilogue_barrier.arrive_and_wait()
        for epi_idx in cutlass.range_constexpr(cute.size(epi_tile_shape)):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            load_acc_subtile(tRS_rD, epi_coord)
            frag_base = ws_ptr + (epi_idx * num_epi_threads + tidx) * frag_elems
            if const_expr(self.split_k_mode == SplitKMode.SERIAL):
                if split_idx == 0:
                    # First split initializes the (uninitialized) region in-order.
                    tRS_gWs = cute.make_tensor(frag_base, cute.make_layout(tRS_rD.shape))
                    cute.autovec_copy(tRS_rD, tRS_gWs)
                else:
                    self._red_add_frag(frag_base, tRS_rD)
            else:
                # PARALLEL: no initializing store (the host zero-fills the workspace),
                # every split reduces — in arrival order, hence not deterministic.
                self._red_add_frag(frag_base, tRS_rD)
        # The group barrier orders every thread's writes before lane 0's gpu-scope
        # release of the flag (CTA-scope happens-before chains into the release).
        epilogue_barrier.arrive_and_wait()
        if is_tma_warp:
            if const_expr(self.split_k_mode == SplitKMode.SERIAL):
                utils.semaphore_release(lock_ptr, split_idx + 1, drain_tma_store=False)
            else:
                utils.semaphore_arrive_inc(lock_ptr)

    @cute.jit
    def _red_add_frag(self, frag_base: cute.Pointer, tRS_rD: cute.Tensor) -> None:
        """red.add the fragment into gmem at frag_base — one-way L2 reductions (no
        read-back; the turnstile-latency-critical path in serial mode), vectorized
        v4.f32 when the fragment allows."""
        frag_elems = cute.size(tRS_rD)
        if const_expr(frag_elems % 4 == 0 and self.acc_dtype == cutlass.Float32):
            for v in cutlass.range_constexpr(frag_elems // 4):
                chunk = cute.make_tensor(tRS_rD.iterator + 4 * v, cute.make_layout(4))
                cute.arch.atomic_add(frag_base + 4 * v, chunk.load())
        else:
            for v in cutlass.range_constexpr(frag_elems):
                cute.arch.atomic_add(frag_base + v, tRS_rD[v])

    @cute.jit
    def epilogue_split_k(
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
        tiled_copy_t2r: Optional[cute.TiledCopy],
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
        """self.epilogue wrapped in the split-K finalization protocol.

        split_k == 1 and SEPARATE pass straight through (SEPARATE splits store raw f32
        partials to disjoint workspace slices via the normal TMA path; the host strips
        the epi-op arguments so the visit is a no-op, and the separate reduction kernel
        applies the full epilogue). SERIAL/PARALLEL: non-finalizing splits commit raw
        partials and skip the epilogue entirely — including the C loads — while the
        last split waits for the tile's completion flag, folds the workspace into its
        accumulator, and runs the full epi mixin exactly once.
        """
        if const_expr(self.split_k == 1 or self.split_k_mode == SplitKMode.SEPARATE):
            epi_read_state, epi_producer_state = self.epilogue(
                params,
                epi_smem_tensors,
                epi_pipeline,
                epi_store_pipeline,
                epi_read_state,
                epi_producer_state,
                epi_tile,
                load_acc_subtile,
                tRS_rD,
                tRS_rC,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tRS_sD,
                tiled_copy_s2r,
                tSR_rC,
                tSR_sC,
                copy_D,
                copy_C,
                tile_coord_mnkl,
                varlen_manager,
                epilogue_barrier,
                tile_scheduler,
                tidx,
                is_tma_warp,
            )
        else:
            # The flag and workspace are CuTe tensors over the (cluster-rounded) tile
            # domain — their layouts own the address computation.
            assert self.acc_dtype == cutlass.Float32, "split_k workspace is f32"
            batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
            lock_ptr = utils.elem_pointer(
                params.split_k_semaphore,
                (tile_coord_mnkl[0], tile_coord_mnkl[1], batch_idx),
            )
            ws_ptr = utils.elem_pointer(
                params.split_k_workspace,
                (0, tile_coord_mnkl[0], tile_coord_mnkl[1], batch_idx),
            )
            # The stripe must tile the host-allocated region exactly.
            epi_tile_num = cute.size(
                cute.zipped_divide(cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile).shape[1]
            )
            assert (
                cute.size(tRS_rD) * self.num_epi_warps * cute.arch.WARP_SIZE * epi_tile_num
                == self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[1]
            ), "split-K workspace stripe does not tile cta_tile_m * cta_tile_n"
            if split_idx < self.split_k - 1:
                self.split_k_partial_commit(
                    load_acc_subtile,
                    tRS_rD,
                    epi_tile,
                    ws_ptr,
                    lock_ptr,
                    split_idx,
                    epilogue_barrier,
                    tidx,
                    is_tma_warp,
                )
            else:
                # Finalizer (fixed: the last split, so e.g. the SM100 epi-load warp can
                # gate C loads statically). Wait for all S-1 sibling commits; the flag
                # counts committed splits in both modes.
                if is_tma_warp:
                    utils.semaphore_wait_eq(lock_ptr, self.split_k - 1)
                epilogue_barrier.arrive_and_wait()
                epi_read_state, epi_producer_state = self.epilogue(
                    params,
                    epi_smem_tensors,
                    epi_pipeline,
                    epi_store_pipeline,
                    epi_read_state,
                    epi_producer_state,
                    epi_tile,
                    load_acc_subtile,
                    tRS_rD,
                    tRS_rC,
                    tiled_copy_t2r,
                    tiled_copy_r2s,
                    tRS_sD,
                    tiled_copy_s2r,
                    tSR_rC,
                    tSR_sC,
                    copy_D,
                    copy_C,
                    tile_coord_mnkl,
                    varlen_manager,
                    epilogue_barrier,
                    tile_scheduler,
                    tidx,
                    is_tma_warp,
                    split_k_ws=ws_ptr,
                )
                # Self-clean the flag (fresh-zeros allocation also works, but this keeps
                # the tensor reusable if the host ever caches it).
                if is_tma_warp:
                    utils.semaphore_release(lock_ptr, Int32(0), drain_tma_store=False)
                # Drain this tile's TMA stores. The epi smem buffer rotation is seeded
                # by num_tiles_executed, which also counts the SKIPPED non-finalizing
                # tiles, so the next executed epilogue's first buffer index jumps by
                # (skipped * epi_tile_num) mod epi_stage — producer_acquire's
                # "<= epi_stage - 1 outstanding groups" guard is only safe for
                # consecutive (+1) rotation. Draining here makes any start index safe.
                if const_expr(epi_store_pipeline is not None):
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
        return epi_read_state, epi_producer_state

    def get_scheduler_class(self, varlen_m: bool = False):
        """Return the scheduler class to use. Override in subclasses for custom schedulers."""
        return TileScheduler if not varlen_m else VarlenMTileScheduler

    def resolve_epi_m_major(self, epilogue_args: EpilogueArguments):
        return True

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        scheduler_args,
        varlen_args,
        epilogue_args,
    ):
        """Create scheduler arguments. Override in subclasses for custom schedulers."""
        if const_expr(not self.is_persistent):
            persistence_mode = PersistenceMode.NONE
        else:
            if const_expr(self.arch >= 100 and self.use_clc_persistence):
                persistence_mode = PersistenceMode.CLC
            elif const_expr(scheduler_args.tile_count_semaphore is not None):
                persistence_mode = PersistenceMode.DYNAMIC
            else:
                persistence_mode = PersistenceMode.STATIC
        if const_expr(varlen_args.mCuSeqlensM is None):
            num_problems = (
                mD.shape[2]
                if mD is not None
                else (
                    mB.shape[2]
                    if varlen_args.mCuSeqlensK is None
                    else varlen_args.mCuSeqlensK.shape[0] - 1
                )
            )
            if const_expr(self.split_k > 1 and self.split_k_mode == SplitKMode.SEPARATE):
                # mD is the f32 partials workspace whose batch extent is L * split_k; the
                # scheduler needs the true L (it scales the work-id space by num_split_k
                # itself). B always carries the true L here (varlen is rejected).
                num_problems = mB.shape[2]
            problem_shape_ntile_mnl = (
                cute.ceil_div(cute.size(mA, mode=[0]), self.cta_tile_shape_mnk[0]),
                cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1]),
                num_problems,
            )
            tile_sched_args = TileSchedulerArguments(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                raster_order=scheduler_args.raster_order,
                group_size=scheduler_args.max_swizzle_size,
                cluster_shape_mnk=self.cluster_shape_mnk,
                tile_count_semaphore=scheduler_args.tile_count_semaphore,
                batch_idx_permute=scheduler_args.batch_idx_permute,
                persistence_mode=persistence_mode,
                num_split_k=self.split_k,
            )
        else:
            assert self.split_k == 1, "split_k does not support varlen_m"
            assert (mD is not None) or (epilogue_args.mAuxOut is not None) or (not self.gather_A)
            problem_shape_ntile_mnl = (
                None,
                cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1]),
                varlen_args.mCuSeqlensM.shape[0] - 1,
            )
            tile_sched_args = VarlenMTileSchedulerArguments(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                total_m=(
                    mD.shape[0]
                    if mD is not None
                    else (
                        varlen_args.mAIdx.shape[0]
                        if varlen_args.mAIdx is not None
                        else cute.size(mA, mode=[0])
                    )
                ),
                cu_seqlens_m=varlen_args.mCuSeqlensM,
                max_active_clusters=scheduler_args.max_active_clusters,
                raster_order=scheduler_args.raster_order,
                group_size=scheduler_args.max_swizzle_size,
                tile_shape_mn=self.cta_tile_shape_mnk[:2],
                cluster_shape_mnk=self.cluster_shape_mnk,
                tile_count_semaphore=scheduler_args.tile_count_semaphore,
                persistence_mode=persistence_mode,
            )
        return tile_sched_args

    @cute.jit
    def epi_load_acc_subtile(
        self,
        tRS_rAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        epi_coord,  # (int, int)
    ):
        cute.autovec_copy(tRS_rAcc[None, None, None, epi_coord], tRS_rD)

    @cute.jit
    def epi_begin(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Dict[str, cute.Tensor],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tidx: Int32,
        tRS_rD_layout=None,
    ) -> Tuple[cute.Tensor, ...]:
        return ()

    def epi_begin_loop(
        self, params: EpilogueParams, epi_tensors: Tuple[cute.Tensor, ...], epi_coord: cute.Coord
    ) -> Tuple[cute.Tensor, ...]:
        return ()

    def epi_visit_subtile(
        self,
        params: EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Tuple[cute.Tensor, ...]:
        return ()

    def epi_visit_acc(
        self,
        params: EpilogueParams,
        acc: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tile_coord_mnkl: cute.Coord,
        tidx: Int32,
    ) -> None:
        pass

    @cute.jit
    def epi_end_loop(
        self,
        params: EpilogueParams,
        epi_tensors: Tuple[cute.Tensor, ...],
        epi_coord: cute.Coord,
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager,
        tidx,
    ) -> None:
        pass

    @cute.jit
    def epi_end(
        self,
        params: EpilogueParams,
        epi_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager,
        tidx,
    ) -> None:
        pass

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        return self.EpilogueParams()

    def epi_get_tma_atoms(
        self, params: EpilogueParams, *, loc=None, ip=None
    ) -> list[cute.CopyAtom]:
        """Subclasses can override this."""
        return []

    def epi_tile_load_g2s_copy_fns(
        self,
        params,
        epi_smem_tensors,
        tile_coord_mnkl,
        varlen_manager,
        epi_pipeline,
    ):
        return ()

    @cute.jit
    def epi_tile_load_s2r(self, params, epi_tensors, stage_idx):
        pass

    @staticmethod
    def epi_smem_bytes(
        args: Optional[EpilogueArguments],
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
        warp_shape_mnk: Tuple[int, int, int] | None = None,
    ) -> EpiSmemBytes:
        return EpiSmemBytes()

    def epi_get_smem_struct(self, params: EpilogueParams):
        return cute.struct.MemRange[Int32, 0]  # Dummy struct

    def epi_get_smem_tensors(self, params: EpilogueParams, storage) -> Dict[str, cute.Tensor]:
        return {}

    def epi_setup_aux_out(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Return a tuple of ``(tiled_copy_r2s, tRS_sAuxOut, copy_aux_out)``
        triples — one per aux output. The default epilogue has no aux output,
        so the tuple is empty.
        """
        return ()

    @cute.jit
    def epi_convert_aux_out(
        self,
        output_idx: cutlass.Constexpr[int],
        tRS_rAuxOut,
        sr_seed,
        tidx,
        tile_coord_mnkl,
        num_prev_subtiles,
        epi_idx,
    ):
        """Convert one aux output register tensor from acc_dtype to its storage
        dtype. ``output_idx`` selects which aux output this call is for
        (single-output mixins can ignore it).
        """
        return tRS_rAuxOut


class GemmTmaBase(GemmBase):
    """Common TMA descriptor and pipeline helpers for SM90+ GEMM paths."""

    @cute.jit
    def load_tma(
        self,
        pipeline: cutlass.pipeline.PipelineAsync,
        producer_state: cutlass.pipeline.PipelineState,
        copy_fns: Sequence[Optional[Callable]],
        k_tile_cnt: Int32,
        k_tile_start: Int32 | int = 0,
    ) -> cutlass.pipeline.PipelineState:
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt.
        peek_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_empty_status = pipeline.producer_try_acquire(producer_state)
        # TMA load
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            # Wait for A/B buffers to be empty before loading into them.
            # Also sets the transaction barrier for the A/B buffers.
            pipeline.producer_acquire(producer_state, peek_empty_status)
            tma_bar_ptr = pipeline.producer_get_barrier(producer_state)
            smem_idx = producer_state.index
            for copy_fn in copy_fns:
                if const_expr(copy_fn is not None):
                    copy_fn(k_tile_start + k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            # Mainloop pipeline's producer commit is a NOP for TMA pipelines.
            pipeline.producer_commit(producer_state)
            producer_state.advance()
            peek_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_empty_status = pipeline.producer_try_acquire(producer_state)
        return producer_state

    def _make_gmem_tiled_copy_A(self, dtype, major_mode, num_threads, copy_bits=128):
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=copy_bits,
        )
        copy_elems = copy_bits // dtype.width
        loads_per_cache_line = 128 * 8 // copy_bits  # 128 bytes per cache line
        shape_dim_1 = cute.size(self.cta_tile_shape_mnk[2]) // copy_elems
        if shape_dim_1 > loads_per_cache_line:
            shape_dim_1 = math.gcd(shape_dim_1, loads_per_cache_line)
        # thread layout for copy
        thread_layout = cute.make_layout(
            (num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.cta_tile_shape_mnk[0]) // copy_elems
            if shape_dim_0 > loads_per_cache_line:
                shape_dim_0 = math.gcd(shape_dim_0, loads_per_cache_line)
            thread_layout = cute.make_layout(
                (shape_dim_0, num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        # Value layout for copy
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_async_copy, thread_layout, value_layout)

    def make_tma_load_atoms_and_tensors(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        varlen_k: bool,
    ):
        tma_atom_a, tma_tensor_a = None, None
        if const_expr(not self.gather_A):
            tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
                copy_utils.create_ragged_tensor_for_tma(mA, ragged_dim=1)
                if varlen_k and not self.gather_A
                else mA,
                a_smem_layout,
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
                self.cluster_shape_mnk[1],
            )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            copy_utils.create_ragged_tensor_for_tma(mB, ragged_dim=1) if varlen_k else mB,
            b_smem_layout,
            (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )
        return tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b

    def make_tma_epilogue_atoms_and_tensors(
        self,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args,
        varlen_m: bool,
    ):
        add_to_output = const_expr(
            hasattr(epilogue_args, "add_to_output") and epilogue_args.add_to_output
        )
        # Split-K needs no special D atom: only the finalizing entity stores D (a plain
        # store, or the reduce-add atom with add_to_output, exactly like the non-split
        # kernel); partials travel through the f32 workspace, not through D.
        tma_atom_d, tma_tensor_d = None, None
        if const_expr(mD is not None):
            tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
                copy_utils.create_ragged_tensor_for_tma(mD, ragged_dim=0, ptr_shift=True)
                if varlen_m
                else mD,
                self.epi_smem_layout_staged,
                self.epi_tile,
                op_type="store" if not add_to_output else "add",
            )
        tma_atom_c, tma_tensor_c = None, None
        if const_expr(mC is not None):
            tma_atom_c, tma_tensor_c = self._make_tma_epi_atoms_and_tensors(
                mC, self.epi_c_smem_layout_staged, self.epi_tile, op_type="load"
            )
        return (
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
        )

    def epilog_gmem_copy_and_partition(
        self,
        atom: cute.CopyAtom | cute.TiledCopy,
        mD_mn: cute.Tensor,
        tile_shape_mn: cute.Tile,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        gD = cute.local_tile(mD_mn, tile_shape_mn, tile_coord_mnkl[:2])  # (bM, bN)
        tDgD_for_tma_partition = cute.zipped_divide(gD, epi_tile)
        is_s2g = isinstance(
            atom.op, (cpasync.CopyBulkTensorTileS2GOp, cpasync.CopyReduceBulkTensorTileS2GOp)
        )
        src_tensor, dst_tensor = (
            (sD, tDgD_for_tma_partition) if is_s2g else (tDgD_for_tma_partition, sD)
        )
        return copy_utils.tma_get_copy_fn(
            atom,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=src_tensor,
            dst_tensor=dst_tensor,
        )

    def make_ab_pipeline(
        self,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
    ):
        # Threads/warps participating in this pipeline
        producer_cnt = 1 if const_expr(not self.gather_A) else 1 + self.num_ab_load_warps * 32
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, producer_cnt)
        # Each warp will contribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * tiled_mma.size // cute.arch.WARP_SIZE
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        pipeline_cls = pipeline.PipelineTmaAsync if not self.gather_A else PipelineTmaCpAsync
        return pipeline_cls.create(
            num_stages=self.ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_epi_pipeline(
        self,
        tx_count: int,
    ):
        epi_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Each warp will contribute 1 to the arrive count
        consumer_arrive_cnt = self.num_epi_warps
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return PipelineTmaAsync.create(
            num_stages=self.epi_c_stage,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
            tx_count=tx_count,
            defer_sync=True,
            elect_one_release=True,
            syncwarp_before_release=True,
        )

    def make_epi_store_pipeline(self):
        num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
        epi_store_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_epi_threads)
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage, producer_group=epi_store_producer_group
        )

    @staticmethod
    def _make_tma_epi_atoms_and_tensors(
        tensor_d: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: Tuple[int, int],
        op_type: Literal["store", "load", "add"],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for storing D or loading C."""
        assert op_type in ["load", "store", "add"]
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        d_cta_v_layout = cute.composition(cute.make_identity_layout(tensor_d.shape), epi_tile)
        op = {
            "load": cpasync.CopyBulkTensorTileG2SOp(),
            "store": cpasync.CopyBulkTensorTileS2GOp(),
            "add": cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD),
        }[op_type]
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            op, tensor_d, epi_smem_layout, d_cta_v_layout
        )
        return tma_atom_d, tma_tensor_d

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout: cute.ComposedLayout,
        smem_tile: Tuple[int, int],
        mcast_dim: int,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors."""
        # block_copy takes compiler-driven multicast metadata at the copy site,
        # so the TMA atom itself must stay the non-multicast variant here.
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(op, tensor, smem_layout, smem_tile)
        return tma_atom, tma_tensor
