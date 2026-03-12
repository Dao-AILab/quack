# Copyright (c) 2025-2026, Tri Dao.
from typing import NamedTuple, Optional, Tuple, Callable, Type
from functools import lru_cache, partial
from dataclasses import dataclass
import operator

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr
import cutlass.utils.blackwell_helpers as sm100_utils

import quack.sm90_utils as sm90_utils
from quack.sm90_utils import partition_for_epilogue
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_act import GemmActMixin
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    ParamsBase,
    mlir_namedtuple,
    torch2cute_dtype_map,
    get_device_capacity,
    get_max_active_clusters,
)
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
from quack.varlen_utils import VarlenManager
from quack import copy_utils
from quack.rounding import RoundingMode
import quack.layout_utils as layout_utils
from quack.activation import dact_fn_map, dgate_fn_map


class GemmDActMixin(GemmActMixin):
    # Different from GemmActSm90, here act_bwd_fn must take in 2 arguments (x, dout)
    # and return 2 arguments (dx, out)
    EpilogueArguments = GemmActMixin.EpilogueArguments
    EpilogueParams = GemmActMixin.EpilogueParams

    @cute.jit
    def epi_visit_subtile(
        self,
        params: EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        assert tRS_rC is not None
        # We don't add C to the accumulator
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None)
        tRS_rC_acc = cute.make_fragment_like(tRS_rC, self.acc_dtype)
        tRS_rC_acc.store(tRS_rC.load().to(self.acc_dtype))
        # If we don't have .shape here, the compiler generates local stores and loads
        if const_expr(params.act_fn is not None):
            tRS_rPostAct = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                    tRS_rD[i], tRS_rPostAct[i] = params.act_fn(tRS_rC_acc[i], tRS_rD[i])
            else:
                for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                    (
                        (tRS_rD[2 * i], tRS_rD[2 * i + 1]),
                        (tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1]),
                    ) = params.act_fn(
                        (tRS_rC_acc[2 * i], tRS_rC_acc[2 * i + 1]),
                        (tRS_rD[2 * i], tRS_rD[2 * i + 1]),
                    )
        else:
            tRS_rPostAct = tRS_rC_acc
        return tRS_rPostAct


class GemmDActSm90(GemmDActMixin, GemmSm90):
    pass


class GemmDActSm100(GemmDActMixin, GemmSm100):
    pass


class GemmDGatedMixin(GemmActMixin):
    # Different from GemmActMixin, here act_bwd_fn must take in 3 arguments (x, y, dout)
    # and return 3 arguments (dx, dy, out)
    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32] = None

    @dataclass
    class EpilogueParams(ParamsBase):
        tma_atom_postact: cute.CopyAtom
        mPostAct_mnl: cute.Tensor
        epi_postact_smem_layout_staged: cute.ComposedLayout
        epi_tile_postact: cute.Tile
        act_bwd_fn: cutlass.Constexpr[Callable]
        implicit_dtype: Type[cutlass.Numeric]
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        sr_seed: Optional[Int32] = None

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        self.rounding_mode = args.rounding_mode
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        # C and D are implicitly 2 16-bit elements packed into 32 bits, simply for the purpose
        # for reusing the existing load/store code.
        assert self.implicit_dtype.width == 16, "GemmDGated only supports 16bit for now"
        assert self.d_dtype.width == 32, "D storage type must be 32 bit"
        assert self.c_dtype.width == 32, "C storage type must be 32 bit"

        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        epi_tile_postact = self.epi_tile
        utils_cls = sm100_utils if self.arch >= 100 else sm90_utils
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
        mRowVecBroadcast, mColVecBroadcast, mColVecReduce = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            if t is not None
            else None
            for t in (args.mRowVecBroadcast, args.mColVecBroadcast, args.mColVecReduce)
        ]
        return self.EpilogueParams(
            tma_atom_postact,
            tma_tensor_postact,
            epi_postact_smem_layout_staged,
            epi_tile_postact,
            args.act_bwd_fn,
            self.implicit_dtype,
            alpha=args.alpha,
            beta=args.beta,
            mRowVecBroadcast=mRowVecBroadcast,
            mColVecBroadcast=mColVecBroadcast,
            mColVecReduce=mColVecReduce,
            sr_seed=args.sr_seed,
        )

    @cute.jit
    def epi_begin(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tidx: Int32,
    ) -> Tuple[cute.Tensor, ...]:
        epi_tensors = GemmDefaultEpiMixin.epi_begin(
            self,
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
        partition_for_epilogue_fn = partial(
            partition_for_epilogue,
            epi_tile=epi_tile,
            tiled_copy=tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s,
            tidx=tidx,
            reference_src=tiled_copy_t2r is None,
        )
        tDrColVecReduce = None
        if const_expr(params.mColVecReduce is not None):
            colvec_mma_layout = cute.make_layout(self.cta_tile_shape_mnk[:2], stride=(1, 0))
            tDrColVec_layout = partition_for_epilogue_fn(
                cute.make_rmem_tensor(colvec_mma_layout, Float32)
            ).layout
            tDrColVecReduce = cute.make_rmem_tensor(tDrColVec_layout, Float32)
            cute.filter_zeros(tDrColVecReduce).fill(0.0)
        return (*epi_tensors, tDrColVecReduce)

    def epi_begin_loop(self, params: EpilogueParams, epi_tensors, epi_coord: cute.Coord):
        epi_tensors, tDrColVecReduce = epi_tensors[:-1], epi_tensors[-1]
        epi_loop_tensors = super().epi_begin_loop(params, epi_tensors, epi_coord)
        tDrColVecReduce_cur = None
        if const_expr(tDrColVecReduce is not None):
            tDrColVecReduce_cur = cute.group_modes(tDrColVecReduce, 3, cute.rank(tDrColVecReduce))[
                None, None, None, epi_coord
            ]
        return (*epi_loop_tensors, tDrColVecReduce_cur)

    @cute.jit
    def epi_visit_subtile(
        self,
        params: EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        alpha, beta, tDrRowVec, tDrColVec, tDrColVecReduce = epi_loop_tensors
        assert alpha is None and beta is None and tDrRowVec is None  # We don't use these for now
        assert tRS_rC is not None
        implicit_dtype = params.implicit_dtype
        assert implicit_dtype.width == 16, "GemmDGatedMixin only supports 16bit for now"
        tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)
        tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_f16x2.layout, Float32)
        tRS_rXY_f32x2.store(tRS_rXY_f16x2.load().to(Float32))
        tRS_rdXY_f32x2 = cute.make_rmem_tensor_like(tRS_rXY_f32x2, Float32)
        tRS_rOut = cute.make_rmem_tensor_like(tRS_rD, Float32)
        tRS_rD_scaled = cute.make_rmem_tensor_like(tRS_rD)
        if const_expr(tDrColVec is not None):  # Scale D by colvec
            if const_expr(self.arch < 100):
                tRS_rD_scaled.store(tRS_rD.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVec.layout)
                tRS_rD_scaled_mn = layout_utils.convert_layout_zero_stride(
                    tRS_rD_scaled, tDrColVec.layout
                )
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(
                        cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True
                    ):
                        (
                            tRS_rD_scaled_mn[m, 2 * n],
                            tRS_rD_scaled_mn[m, 2 * n + 1],
                        ) = cute.arch.mul_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        else:
            tRS_rD_scaled.store(tRS_rD.load())
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rD)):
                (
                    tRS_rdXY_f32x2[2 * i],
                    tRS_rdXY_f32x2[2 * i + 1],
                    tRS_rOut[i],
                ) = params.act_bwd_fn(
                    tRS_rXY_f32x2[2 * i], tRS_rXY_f32x2[2 * i + 1], tRS_rD_scaled[i]
                )
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                (
                    (tRS_rdXY_f32x2[4 * i], tRS_rdXY_f32x2[4 * i + 2]),
                    (tRS_rdXY_f32x2[4 * i + 1], tRS_rdXY_f32x2[4 * i + 3]),
                    (tRS_rOut[2 * i], tRS_rOut[2 * i + 1]),
                ) = params.act_bwd_fn(
                    (tRS_rXY_f32x2[4 * i], tRS_rXY_f32x2[4 * i + 2]),
                    (tRS_rXY_f32x2[4 * i + 1], tRS_rXY_f32x2[4 * i + 3]),
                    (tRS_rD_scaled[2 * i], tRS_rD_scaled[2 * i + 1]),
                )
        if const_expr(tDrColVecReduce is not None):
            # Need to multiply before D is scaled by colvec_scale
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tDrColVecReduce), unroll_full=True):
                    tDrColVecReduce[i] += tRS_rOut[i] * tRS_rD[i]
            else:
                tDrColVecReduce_mn = layout_utils.convert_layout_zero_stride(
                    tDrColVecReduce, tDrColVecReduce.layout
                )
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVecReduce.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(
                    tRS_rOut, tDrColVecReduce.layout
                )
                for m in cutlass.range(cute.size(tDrColVecReduce_mn, mode=[0]), unroll_full=True):
                    row_sum = cute.arch.mul_packed_f32x2(
                        (tRS_rD_mn[m, 0], tRS_rD_mn[m, 1]), (tRS_rOut_mn[m, 0], tRS_rOut_mn[m, 1])
                    )
                    for n in cutlass.range(
                        1, cute.size(tDrColVecReduce_mn, mode=[1]) // 2, unroll_full=True
                    ):
                        row_sum = cute.arch.fma_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                            row_sum,
                        )
                    tDrColVecReduce_mn[m, 0] += row_sum[0] + row_sum[1]

        if const_expr(tDrColVec is not None):  # Scale Out by colvec
            if const_expr(self.arch < 100):
                tRS_rOut.store(tRS_rOut.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(
                        cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True
                    ):
                        tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1] = (
                            cute.arch.mul_packed_f32x2(
                                (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                                (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                            )
                        )
        # Type conversion
        tRS_rdXY_f16x2 = cute.make_rmem_tensor(tRS_rdXY_f32x2.layout, implicit_dtype)
        tRS_rdXY_f16x2.store(tRS_rdXY_f32x2.load().to(implicit_dtype))
        tRS_rD.store(cute.recast_tensor(tRS_rdXY_f16x2, Float32).load())
        return tRS_rOut

    @cute.jit
    def epi_end(
        self,
        params: EpilogueParams,
        epi_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        tidx: Int32,
    ) -> None:
        partition_for_epilogue_fn = partial(
            partition_for_epilogue,
            epi_tile=epi_tile,
            tiled_copy=tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s,
            tidx=tidx,
            reference_src=tiled_copy_t2r is None,
        )
        tDrColVecReduce = epi_tensors[-1]
        tile_M, tile_N = self.cta_tile_shape_mnk[:2]
        if const_expr(params.mColVecReduce is not None):
            tDrCVR_flt = cute.filter_zeros(tDrColVecReduce)
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tDrCVR_flt), unroll_full=True):
                    tDrCVR_flt[i] = cute.arch.warp_reduction(
                        tDrCVR_flt[i], operator.add, threads_in_group=4
                    )
            else:
                # Don't need warp_reduce since we load from tmem with one thread per row
                assert self.d_layout.is_n_major_c(), (
                    "GemmDGated only supports n-major output for now"
                )
            batch_idx = tile_coord_mnkl[3]
            limit_n = (
                params.mColVecReduce.shape[2]
                if not varlen_manager.varlen_m
                else params.mColVecReduce.shape[1]
            )
            if tile_coord_mnkl[1] < limit_n:
                if const_expr(not varlen_manager.varlen_m):
                    mColVec = params.mColVecReduce[batch_idx, None, tile_coord_mnkl[1]]
                else:
                    mColVec = cute.domain_offset(
                        (varlen_manager.params.cu_seqlens_m[batch_idx],),
                        params.mColVecReduce[None, tile_coord_mnkl[1]],
                    )
                gColVec = cute.local_tile(mColVec, (tile_M,), (tile_coord_mnkl[0],))
                limit_m = min(varlen_manager.len_m(batch_idx) - tile_coord_mnkl[0] * tile_M, tile_M)
                tDcCV = partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
                tDrColVecReduce_m = layout_utils.convert_layout_zero_stride(
                    tDrColVecReduce, tDrColVecReduce.layout
                )[None, 0]
                tDcCV_m = layout_utils.convert_layout_zero_stride(tDcCV, tDrColVecReduce.layout)[
                    None, 0
                ]
                if tDcCV_m[0][1] == 0:
                    for m in cutlass.range(cute.size(tDcCV_m, mode=[0])):
                        row_idx = tDcCV_m[m][0]
                        if row_idx < limit_m:
                            gColVec[row_idx] = tDrColVecReduce_m[m]


class GemmDGatedSm90(GemmDGatedMixin, GemmSm90):
    pass


class GemmDGatedSm100(GemmDGatedMixin, GemmSm100):
    pass


@lru_cache(maxsize=None)
def _compile_gemm_dact(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    postact_dtype,
    implicit_dtype,
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
    colvec_scale_dtype,
    colvec_scale_ndim,
    colvec_reduce_dtype,
    colvec_reduce_ndim,
    varlen_m,
    gather_A,
    device_capacity,
    gemm_cls_name,
):
    is_dgated = gemm_cls_name == "dgated"
    if is_dgated:
        GemmCls = GemmDGatedSm100 if device_capacity[0] > 9 else GemmDGatedSm90
    else:
        GemmCls = GemmDActSm100 if device_capacity[0] > 9 else GemmDActSm90
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
    pa_leading = 1 if postact_major == "n" else 0
    pa_shape = (m, n) if varlen_m else (m, n, l)
    mPostAct = fake_tensor(postact_dtype, pa_shape, leading_dim=pa_leading, divisibility=div_pa)

    if is_dgated:
        act_fn = dgate_fn_map[activation]

        mColVec = None
        if colvec_scale_ndim == 2:
            mColVec = fake_tensor(colvec_scale_dtype, (l, m), leading_dim=1, divisibility=4)
        elif colvec_scale_ndim == 1:
            mColVec = fake_tensor(colvec_scale_dtype, (m,), leading_dim=0, divisibility=4)
        mColVecReduce = None
        n_tiles = cute.sym_int()
        if colvec_reduce_ndim == 3:
            mColVecReduce = fake_tensor(
                colvec_reduce_dtype,
                (l, m, n_tiles),
                leading_dim=2,
                divisibility=1,
            )
        elif colvec_reduce_ndim == 2:
            mColVecReduce = fake_tensor(
                colvec_reduce_dtype,
                (m, n_tiles),
                leading_dim=1,
                divisibility=1,
            )
        epi_args = GemmCls.EpilogueArguments(
            mPostAct,
            act_fn,
            mColVecBroadcast=mColVec,
            mColVecReduce=mColVecReduce,
        )

        def _set_implicit_dtype(gemm_obj):
            gemm_obj.implicit_dtype = implicit_dtype

        post_init = _set_implicit_dtype
    else:
        act_fn = dact_fn_map[activation]
        epi_args = GemmCls.EpilogueArguments(mPostAct, act_fn)
        post_init = None

    scheduler_args = make_fake_scheduler_args(has_semaphore, False, l)
    varlen_args = make_fake_varlen_args(varlen_m, False, gather_A, m if varlen_m else None)
    key = (
        "gemm_dact",
        gemm_cls_name,
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        postact_dtype,
        implicit_dtype,
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
        colvec_scale_dtype,
        colvec_scale_ndim,
        colvec_reduce_dtype,
        colvec_reduce_ndim,
        varlen_m,
        gather_A,
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
            device_capacity,
            mA,
            mB,
            mD,
            mC,
            epi_args,
            scheduler_args,
            varlen_args,
            post_init=post_init,
        ),
    )


def gemm_dact(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    Out: Tensor,  # (l, m, n) or (total_m, n) if varlen_m; or (l, m, 2*n)/(total_m, 2*n) if dgated
    PreAct: Tensor,  # same shape as Out
    PostAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = True,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    colvec_scale: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m (dgated only)
    # (l, m, ceildiv(n, tile_n)), or (total_m, ceildiv(n, tile_n)) if varlen_m (dgated only)
    colvec_reduce: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
) -> None:
    is_dgated = activation in dgate_fn_map
    if not is_dgated:
        assert activation in dact_fn_map, f"Unsupported activation {activation}"
        assert colvec_scale is None, "colvec_scale is only supported for gated activations"
        assert colvec_reduce is None, "colvec_reduce is only supported for gated activations"
    gemm_cls_name = "dgated" if is_dgated else "dact"

    varlen_m = cu_seqlens_m is not None
    gather_A = A_idx is not None
    if varlen_m:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        assert Out.stride(-1) == 1, "varlen_m requires Out to be n-major"
        assert PreAct.stride(-1) == 1, "varlen_m requires PreAct to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen"
        assert cluster_N == 1, "gather_A requires cluster_N=1"

    # For dgated, capture implicit_dtype before viewing Out/PreAct as f32
    implicit_dtype = None
    if is_dgated:
        AB_swapped = Out.stride(-1) != 1
        implicit_dtype = torch2cute_dtype_map[Out.dtype]
        assert Out.element_size() == 2, "Out dtype must be fp16 or bf16"
        assert PreAct.element_size() == 2, "Preact dtype must be fp16 or bf16"
        if varlen_m or not AB_swapped:
            Out = Out.view(torch.float32)
            PreAct = PreAct.view(torch.float32)
        else:
            Out = Out.mT.view(torch.float32).mT
            PreAct = PreAct.mT.view(torch.float32).mT

    A_p = perm3d_single(A, varlen_m)
    B_p = perm3d_single(B)
    Out_p = perm3d_single(Out, varlen_m)
    PreAct_p = perm3d_single(PreAct, varlen_m)
    PostAct_p = perm3d_single(PostAct, varlen_m)

    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(Out_p, "m", "n")
    c_major = get_major(PreAct_p, "m", "n")
    postact_major = get_major(PostAct_p, "m", "n")

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[Out.dtype]
    c_dtype = torch2cute_dtype_map[PreAct.dtype]
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10, 11], "Only SM90, SM100, and SM110 are supported"

    compiled_fn = _compile_gemm_dact(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        postact_dtype,
        implicit_dtype,
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
        torch2cute_dtype_map[colvec_scale.dtype] if colvec_scale is not None else None,
        colvec_scale.ndim if colvec_scale is not None else 0,
        torch2cute_dtype_map[colvec_reduce.dtype] if colvec_reduce is not None else None,
        colvec_reduce.ndim if colvec_reduce is not None else 0,
        varlen_m,
        gather_A,
        device_capacity,
        gemm_cls_name,
    )

    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    if is_dgated:
        epi_args = GemmDGatedMixin.EpilogueArguments(
            PostAct_p,
            None,  # act_bwd_fn is Constexpr
            mColVecBroadcast=colvec_scale,
            mColVecReduce=colvec_reduce,
            rounding_mode=None,
            sr_seed=None,
        )
    else:
        epi_args = GemmDActMixin.EpilogueArguments(
            PostAct_p, None, rounding_mode=None, sr_seed=None,
        )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
    )
    varlen_args = make_varlen_args(cu_seqlens_m, None, A_idx)

    if device_capacity[0] > 9:
        compiled_fn(A_p, B_p, Out_p, PreAct_p, epi_args, scheduler_args, varlen_args, None, None)
    else:
        compiled_fn(A_p, B_p, Out_p, PreAct_p, epi_args, scheduler_args, varlen_args)


gemm_dgated = gemm_dact
