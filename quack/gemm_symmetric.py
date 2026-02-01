from typing import Tuple, Optional, Callable
from functools import partial
from torch import Tensor
from quack.gemm_act import GemmActMixin, act_fn_map, gemm_act
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.tile_scheduler import TriangularTileScheduler
from quack.gemm_wrapper_utils import GemmWrapperBase
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.varlen_utils import VarlenManager
import quack.copy_utils as copy_utils
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import make_ptr
from cutlass import Int32, Float32, Boolean, const_expr
import cutlass.utils.hopper_helpers as sm90_utils_og
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cutlass_dsl import if_generate


class GemmSymmetricMixin(GemmActMixin, GemmSm90):
    def get_scheduler_class(self, varlen_m: bool = False):
        return TriangularTileScheduler

    @cute.jit
    def epilogue(
        self,
        params: GemmActMixin.EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        tma_desc_epi_ptrs: list[Optional[cute.Pointer]],
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
        (tma_desc_postact_ptr,) = tma_desc_epi_ptrs
        batch_idx = tile_coord_mnkl[3]
        copy_postact, _, _ = self.epilog_gmem_copy_and_partition(
            tma_atom_postact,
            varlen_manager.offset_batch_epi(mPostAct_mnl, batch_idx),
            self.cta_tile_shape_postact_mn,
            params.epi_tile_postact,
            sPostAct,
            tile_coord_mnkl,
            tma_desc_ptr=tma_desc_postact_ptr,
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

        def tma_store_fn(src_idx, dst_idx, tile_coord_mnkl):
            pid_m = tile_coord_mnkl[0]
            pid_n = tile_coord_mnkl[1]
            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_view_async_shared()
            epilogue_barrier.arrive_and_wait()
            # Copy from shared memory to global memory
            if is_tma_warp:
                square_tile_m = pid_m // self.cluster_shape_mnk[0]
                square_tile_n = pid_n // self.cluster_shape_mnk[1]
                if const_expr(has_D):
                    copy_D(src_idx=src_idx, dst_idx=dst_idx)
                if square_tile_m != square_tile_n:  # don't write twice to the same tile
                    copy_postact(src_idx=src_idx, dst_idx=dst_idx)
            # Can't use if statement here, epi_store_pipeline object isn't captured somehow
            if_generate(is_tma_warp, lambda: epi_store_pipeline.producer_commit())
            if_generate(is_tma_warp, lambda: epi_store_pipeline.producer_acquire())
            epilogue_barrier.arrive_and_wait()

        delay_tma_store = True

        src_idx_prev, dst_idx_prev = None, None
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
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            if const_expr(delay_tma_store):
                if const_expr(epi_idx > 0):
                    tma_store_fn(
                        src_idx=src_idx_prev, dst_idx=dst_idx_prev, tile_coord_mnkl=tile_coord_mnkl
                    )
                src_idx_prev, dst_idx_prev = epi_buffer, gmem_coord
            # Copy from D registers to shared memory
            if const_expr(has_D):
                copy_utils.cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, epi_buffer])
            cute.copy(
                tiled_copy_postact_r2s,
                tiled_copy_postact_r2s.retile(tRS_rPostAct),
                tRS_sPostAct[None, None, None, epi_buffer],
            )
            if const_expr(not delay_tma_store):
                tma_store_fn(
                    src_idx=epi_buffer, dst_idx=gmem_coord, tile_coord_mnkl=tile_coord_mnkl
                )

        if const_expr(delay_tma_store):
            tma_store_fn(
                src_idx=src_idx_prev, dst_idx=dst_idx_prev, tile_coord_mnkl=tile_coord_mnkl
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


class GemmSymmetricSm90(GemmSymmetricMixin, GemmSm90):
    pass


class GemmSymmetricSm100(GemmSymmetricMixin, GemmSm100):
    pass


def gemm_symmetric(
    A: Tensor,  # (l, m, k)
    B: Tensor,  # (l, m, k)
    D: Optional[Tensor],  # (l, m, m)
    C: Optional[Tensor],  # (l, m, m)
    tile_count_semaphore: Optional[Tensor],  # (1,)
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = False,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
) -> None:
    # Tranpose D so the "activation" is a write to the mirrored tile
    PostAct = D.mT

    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A, B, D, C, additional_tensors={"PostAct": PostAct}
    )
    assert M == N, "M and N must be the same; symmetric gemm only supports square matrices"
    GemmWrapperBase.permute_tensors(tensor_infos)
    GemmWrapperBase.extract_dtypes(tensor_infos)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
        "PostAct": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10], "Only SM90 and SM100 are supported"
    GemmCls = GemmSymmetricSm90 if device_capacity[0] == 9 else GemmSymmetricSm100

    acc_dtype = Float32
    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    if not GemmCls.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        acc_dtype,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Skipping due to unsupported combination of types and majors")

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    GemmWrapperBase.create_cute_tensors({k: v for k, v in tensor_infos.items()}, major_configs)

    def scalar_arg(scalar: float | Tensor):
        if isinstance(scalar, float):
            return Float32(scalar) if scalar != 1.0 else None
        else:
            assert isinstance(scalar, Tensor)
            return make_ptr(Float32, scalar.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)

    activation = None  # Equivalent to identity
    act_fn = act_fn_map[activation]
    epi_args = GemmCls.EpilogueArguments(
        tensor_infos["PostAct"].cute_tensor, act_fn, scalar_arg(alpha), scalar_arg(beta)
    )
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters, tile_count_semaphore, max_swizzle_size=max_swizzle_size
    )
    varlen_args = None

    current_stream = cutlass_torch.current_stream()
    compile_key = GemmWrapperBase.get_compile_key(
        tensor_infos,
        activation,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        tile_count_semaphore is not None,
        device_capacity,
        max_swizzle_size,
        2 if isinstance(alpha, Tensor) else (1 if alpha == 1.0 else 0),
        2 if isinstance(beta, Tensor) else (1 if beta == 1.0 else 0),
        key_tensor_names=("A", "B", "D", "PostAct", "C"),
    )
    cache = gemm_act.compile_cache
    if compile_key not in cache:
        if device_capacity[0] == 9:
            GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
        gemm_obj = GemmCls(
            acc_dtype,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            gather_A=False,
        )
        cache[compile_key] = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
        )
    cache[compile_key](
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
    )


gemm_act.compile_cache = {}
