# Copyright (c) 2025, Wentao Guo, Tri Dao.
from typing import NamedTuple, Tuple, Optional, Callable
from functools import lru_cache, partial
from dataclasses import dataclass

from torch import Tensor

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_og
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32, Float32, Boolean, const_expr

from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    ParamsBase,
    mlir_namedtuple,
    get_device_capacity,
    get_max_active_clusters,
    torch2cute_dtype_map,
)
from quack.varlen_utils import VarlenManager
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_tvm_ffi_utils import (
    get_major,
    perm3d_single,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    div_for_dtype,
    make_fake_gemm_tensors,
    cached_compile,
    compile_gemm_kernel,
)
from quack.layout_utils import permute_gated_Cregs_b16
import quack.sm90_utils as sm90_utils
import quack.copy_utils as copy_utils
from quack.activation import act_fn_map, gate_fn_map


class GemmActMixin(GemmDefaultEpiMixin):
    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None

    @dataclass
    class EpilogueParams(ParamsBase):
        tma_atom_postact: cute.CopyAtom
        mPostAct_mnl: cute.Tensor
        epi_postact_smem_layout_staged: cute.ComposedLayout
        epi_tile_postact: cute.Tile
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)

        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        epi_tile_postact = self.epi_tile
        utils_cls = sm100_utils if self.arch == 100 else sm90_utils
        epi_postact_smem_layout_staged = utils_cls.make_smem_layout_epi(
            self.postact_dtype, self.postact_layout, epi_tile_postact, self.epi_stage
        )
        tma_atom_postact, tma_tensor_postact = self._make_tma_epi_atoms_and_tensors(
            copy_utils.create_ragged_tensor_for_tma(args.mPostAct, ragged_dim=0, ptr_shift=True)
            if cute.rank(args.mPostAct) == 2
            else args.mPostAct,
            epi_postact_smem_layout_staged,
            epi_tile_postact,
            op_type="store",
        )
        # Assume all strides are divisible by 32 bits except the last stride
        new_stride = lambda t: tuple(
            cute.assume(s, divby=32 // t.element_type.width) if not cute.is_static(s) else s
            for s in t.stride
        )
        mRowVecBroadcast, mColVecBroadcast = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            if t is not None
            else None
            for t in (args.mRowVecBroadcast, args.mColVecBroadcast)
        ]
        return self.EpilogueParams(
            tma_atom_postact,
            tma_tensor_postact,
            epi_postact_smem_layout_staged,
            epi_tile_postact,
            args.act_fn,
            alpha=args.alpha,
            beta=args.beta,
            mRowVecBroadcast=mRowVecBroadcast,
            mColVecBroadcast=mColVecBroadcast,
        )

    def epi_get_tma_atoms(
        self, params: EpilogueParams, *, loc=None, ip=None
    ) -> list[cute.CopyAtom]:
        return [params.tma_atom_postact]

    @staticmethod
    def epi_smem_bytes_per_stage(
        args: EpilogueArguments, cta_tile_shape_mnk: Tuple[int, int, int], epi_tile: cute.Tile
    ) -> int:
        postact_dtype = args.mPostAct.element_type
        postact_bytes_per_stage = cute.size(cute.shape(epi_tile)) * (postact_dtype.width // 8)
        rowvec_colvec_bytes = GemmDefaultEpiMixin.epi_smem_bytes_per_stage(
            args, cta_tile_shape_mnk, epi_tile
        )
        return postact_bytes_per_stage + rowvec_colvec_bytes

    def epi_get_smem_struct(self, params: EpilogueParams):
        row_vec_smem_size = 0 if params.mRowVecBroadcast is None else self.cta_tile_shape_mnk[1]
        col_vec_smem_size = 0 if params.mColVecBroadcast is None else self.cta_tile_shape_mnk[0]
        row_vec_dtype = (
            params.mRowVecBroadcast.element_type if params.mRowVecBroadcast is not None else Float32
        )
        col_vec_dtype = (
            params.mColVecBroadcast.element_type if params.mColVecBroadcast is not None else Float32
        )

        @cute.struct
        class EpiSharedStorage:
            sRowVec: cute.struct.Align[cute.struct.MemRange[row_vec_dtype, row_vec_smem_size], 16]
            sColVec: cute.struct.Align[cute.struct.MemRange[col_vec_dtype, col_vec_smem_size], 16]
            sPostAct: cute.struct.Align[
                cute.struct.MemRange[
                    self.postact_dtype, cute.cosize(params.epi_postact_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        return EpiSharedStorage

    def epi_get_smem_tensors(self, params: EpilogueParams, storage) -> Tuple[cute.Tensor, ...]:
        sRowVec, sColVec = super().epi_get_smem_tensors(params, storage)
        sPostAct = storage.epi.sPostAct.get_tensor(
            params.epi_postact_smem_layout_staged.outer,
            swizzle=params.epi_postact_smem_layout_staged.inner,
        )
        return (sRowVec, sColVec, sPostAct)

    @cute.jit
    def epilogue(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        epi_pipeline: cutlass.pipeline.PipelineAsync,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        epi_read_state: cutlass.pipeline.PipelineState,
        epi_producer_state: cutlass.pipeline.PipelineState,
        epi_tile: cute.Tile,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor],
        tiled_copy_t2r: Optional[cute.TiledCopy],  # Only for Sm100
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

        tma_atom_postact = params.tma_atom_postact
        mPostAct_mnl = params.mPostAct_mnl
        sRowVec, sColVec, sPostAct = epi_smem_tensors
        get_smem_store_op = (
            partial(sm100_utils.get_smem_store_op, tiled_tmem_load=tiled_copy_t2r)
            if self.arch == 100
            else sm90_utils_og.sm90_get_smem_store_op
        )
        copy_atom_postact_r2s = get_smem_store_op(
            self.postact_layout, self.postact_dtype, self.acc_dtype
        )
        # tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        # tiled_copy_postact_r2s = cute.make_tiled_copy_S(copy_atom_postact_r2s, tiled_copy_C_atom)
        tiled_copy_postact_r2s = cute.make_tiled_copy_S(copy_atom_postact_r2s, tiled_copy_r2s)
        tRS_sPostAct = tiled_copy_postact_r2s.get_slice(tidx).partition_D(sPostAct)
        batch_idx = tile_coord_mnkl[3]
        copy_postact, _, _ = self.epilog_gmem_copy_and_partition(
            tma_atom_postact,
            varlen_manager.offset_batch_epi(mPostAct_mnl, batch_idx),
            self.cta_tile_shape_postact_mn,
            params.epi_tile_postact,
            sPostAct,
            tile_coord_mnkl,
        )

        # We iterate over epi tiles in the N dimension first before the M dimension
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))
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
            # The global memory coordinate for the current epi tile
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from acc to D registers
            load_acc_subtile(tRS_rD, epi_idx)
            epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, gmem_coord)
            if const_expr(has_C):
                epi_pipeline.consumer_wait(epi_read_state)
                cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC)
                # Fence to make sure shared memory read is visible to TMA load
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
            if is_tma_warp:
                epi_store_pipeline.producer_acquire()
            epilogue_barrier.arrive_and_wait()
            # Copy from D registers to shared memory
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            if const_expr(has_D):
                copy_utils.cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, epi_buffer])
            cute.copy(
                tiled_copy_postact_r2s,
                tiled_copy_postact_r2s.retile(tRS_rPostAct),
                tRS_sPostAct[None, None, None, epi_buffer],
            )
            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_view_async_shared()
            epilogue_barrier.arrive_and_wait()
            # Copy from shared memory to global memory
            if is_tma_warp:
                if const_expr(has_D):
                    copy_D(src_idx=epi_buffer, dst_idx=gmem_coord)
                copy_postact(src_idx=epi_buffer, dst_idx=gmem_coord)
                epi_store_pipeline.producer_commit()

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
    def epi_visit_subtile(
        self,
        params: EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        # Apply activation function if provided
        # If we don't have .shape here, the compiler generates local stores and loads
        if const_expr(params.act_fn is not None):
            tRS_rPostAct = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                    tRS_rPostAct[i] = params.act_fn(tRS_rD[i])
            else:
                for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                    tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1] = params.act_fn(
                        (tRS_rD[2 * i], tRS_rD[2 * i + 1])
                    )
        else:
            tRS_rPostAct = tRS_rD
        # Type conversion
        tRS_rPostAct_out = cute.make_fragment_like(tRS_rPostAct, self.postact_dtype)
        tRS_rPostAct_out.store(tRS_rPostAct.load().to(self.postact_dtype))
        return tRS_rPostAct_out


class GemmActSm90(GemmActMixin, GemmSm90):
    pass


class GemmActSm100(GemmActMixin, GemmSm100):
    pass


class GemmGatedMixin(GemmActMixin):
    def epi_to_underlying_arguments(
        self, args: GemmActMixin.EpilogueArguments, *, loc=None, ip=None
    ) -> GemmActMixin.EpilogueParams:
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        assert self.postact_dtype.width == 16, "GemmGated only supports 16bit postact for now"
        assert self.d_layout is None or self.d_layout.is_n_major_c()
        assert self.postact_layout.is_n_major_c()
        if self.arch == 90:
            assert self.cta_tile_shape_mnk[1] % 32 == 0, (
                "GemmGatedSm90 requires tileN to be divisible by 32"
            )

        self.cta_tile_shape_postact_mn = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1] // 2,
        )
        if isinstance(self.epi_tile[1], cute.Layout):
            epi_tile_postact_1 = cute.recast_layout(2, 1, self.epi_tile[1])
        else:
            epi_tile_postact_1 = self.epi_tile[1] // 2
        epi_tile_postact = (self.epi_tile[0], epi_tile_postact_1)
        utils_cls = sm100_utils if self.arch == 100 else sm90_utils
        epi_postact_smem_layout_staged = utils_cls.make_smem_layout_epi(
            self.postact_dtype, self.postact_layout, epi_tile_postact, self.epi_stage
        )
        tma_atom_postact, tma_tensor_postact = self._make_tma_epi_atoms_and_tensors(
            copy_utils.create_ragged_tensor_for_tma(args.mPostAct, ragged_dim=0, ptr_shift=True)
            if cute.rank(args.mPostAct) == 2
            else args.mPostAct,
            epi_postact_smem_layout_staged,
            epi_tile_postact,
            op_type="store",
        )
        # Assume all strides are divisible by 32 bits except the last stride
        new_stride = lambda t: tuple(
            cute.assume(s, divby=32 // t.element_type.width) if not cute.is_static(s) else s
            for s in t.stride
        )
        mRowVecBroadcast, mColVecBroadcast = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            if t is not None
            else None
            for t in (args.mRowVecBroadcast, args.mColVecBroadcast)
        ]
        return self.EpilogueParams(
            tma_atom_postact,
            tma_tensor_postact,
            epi_postact_smem_layout_staged,
            epi_tile_postact,
            args.act_fn,
            alpha=args.alpha,
            beta=args.beta,
            mRowVecBroadcast=mRowVecBroadcast,
            mColVecBroadcast=mColVecBroadcast,
        )

    @staticmethod
    def epi_smem_bytes_per_stage(
        args: GemmActMixin.EpilogueArguments,
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
    ) -> int:
        postact_dtype = args.mPostAct.element_type
        postact_bytes_per_stage = (cute.size(cute.shape(epi_tile)) // 2) * (
            postact_dtype.width // 8
        )
        rowvec_colvec_bytes = GemmDefaultEpiMixin.epi_smem_bytes_per_stage(
            args, cta_tile_shape_mnk, epi_tile
        )
        return postact_bytes_per_stage + rowvec_colvec_bytes

    @cute.jit
    def epi_visit_subtile(
        self,
        params: GemmActMixin.EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        tRS_rPostAct_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        # If we don't have .shape here, the compiler generates local stores and loads
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rPostAct_layout.shape, self.acc_dtype)
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rPostAct[i] = params.act_fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
        else:
            for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1] = params.act_fn(
                    (tRS_rD[4 * i], tRS_rD[4 * i + 2]), (tRS_rD[4 * i + 1], tRS_rD[4 * i + 3])
                )
        # Type conversion
        tRS_rPostAct_out = cute.make_rmem_tensor_like(tRS_rPostAct, self.postact_dtype)
        tRS_rPostAct_out.store(tRS_rPostAct.load().to(self.postact_dtype))
        if const_expr(self.arch == 90):
            # Only need this if we're using STSM
            permute_gated_Cregs_b16(tRS_rPostAct_out)
        return tRS_rPostAct_out


class GemmGatedSm90(GemmGatedMixin, GemmSm90):
    pass


class GemmGatedSm100(GemmGatedMixin, GemmSm100):
    pass


@lru_cache(maxsize=None)
def _compile_gemm_act(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    postact_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    postact_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    has_semaphore,
    activation,
    rowvec_dtype,
    colvec_dtype,
    colvec_ndim,
    varlen_m,
    gather_A,
    use_clc_persistence,
    device_capacity,
    gemm_cls_name,
):
    GemmCls = (
        {"act": GemmActSm100, "gated": GemmGatedSm100}[gemm_cls_name]
        if device_capacity[0] > 9
        else {"act": GemmActSm90, "gated": GemmGatedSm90}[gemm_cls_name]
    )
    pa_leading = 1 if postact_major == "n" else 0
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        varlen_m=varlen_m,
        gather_A=gather_A,
    )
    div_pa = div_for_dtype(postact_dtype)
    pa_n = cute.sym_int() if gemm_cls_name == "gated" else n
    pa_leading_dim = 1 if gemm_cls_name == "gated" else pa_leading
    pa_shape = (m, pa_n) if varlen_m else (m, pa_n, l)
    mPostAct = fake_tensor(postact_dtype, pa_shape, leading_dim=pa_leading_dim, divisibility=div_pa)

    mRowVec = fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4)
    if colvec_ndim == 2:
        mColVec = fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
    elif colvec_ndim == 1:
        mColVec = fake_tensor(colvec_dtype, (m,), leading_dim=0, divisibility=4)
    else:
        mColVec = None

    act_fn = act_fn_map[activation] if gemm_cls_name == "act" else gate_fn_map[activation]
    epi_args = GemmCls.EpilogueArguments(
        mPostAct,
        act_fn,
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
    )
    scheduler_args = make_fake_scheduler_args(has_semaphore, False, l)
    varlen_args = make_fake_varlen_args(varlen_m, False, gather_A, m if varlen_m else None)
    key = (
        "gemm_act",
        gemm_cls_name,
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        postact_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        postact_major,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        has_semaphore,
        activation,
        rowvec_dtype,
        colvec_dtype,
        colvec_ndim,
        varlen_m,
        gather_A,
        use_clc_persistence,
        device_capacity,
    )
    return cached_compile(
        key,
        lambda: compile_gemm_kernel(
            GemmCls,
            a_dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            pingpong,
            persistent,
            gather_A,
            use_clc_persistence,
            device_capacity,
            mA,
            mB,
            mD,
            mC,
            epi_args,
            scheduler_args,
            varlen_args,
        ),
    )


def gemm_act(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    D: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    PostAct: Tensor,  # (l, m, n) or (total_m, n//2) if gated
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = False,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,  # (l, n)
    colvec_bias: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
    use_clc_persistence: bool = False,
) -> None:
    if activation in gate_fn_map:
        gemm_cls_name = "gated"
    else:
        assert activation in act_fn_map, f"Unsupported activation {activation}"
        gemm_cls_name = "act"

    varlen_m = cu_seqlens_m is not None
    gather_A = A_idx is not None
    if varlen_m:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen"
        assert cluster_N == 1, "gather_A requires cluster_N=1"

    A_p = perm3d_single(A, varlen_m)
    B_p = perm3d_single(B)
    D_p = perm3d_single(D, varlen_m)
    C_p = perm3d_single(C, varlen_m)
    PostAct_p = perm3d_single(PostAct, varlen_m)

    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n") if D_p is not None else None
    c_major = get_major(C_p, "m", "n") if C_p is not None else None
    postact_major = get_major(PostAct_p, "m", "n")

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype] if D is not None else None
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    colvec_ndim = colvec_bias.ndim if colvec_bias is not None else 0

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10, 11], "Only SM90, SM100, and SM110 are supported"

    compiled_fn = _compile_gemm_act(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        postact_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        postact_major,
        (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        tile_count_semaphore is not None,
        activation,
        torch2cute_dtype_map[rowvec_bias.dtype] if rowvec_bias is not None else None,
        torch2cute_dtype_map[colvec_bias.dtype] if colvec_bias is not None else None,
        colvec_ndim,
        varlen_m,
        gather_A,
        use_clc_persistence,
        device_capacity,
        gemm_cls_name,
    )

    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    epi_args = GemmActMixin.EpilogueArguments(
        PostAct_p,
        None,  # act_fn is Constexpr, pass None at call time
        mRowVecBroadcast=rowvec_bias,
        mColVecBroadcast=colvec_bias,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
    )
    varlen_args = make_varlen_args(cu_seqlens_m, None, A_idx)

    if device_capacity[0] > 9:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None, None)
    else:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args)


gemm_gated = gemm_act
