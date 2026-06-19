# Copyright (c) 2025-2026, QuACK team.
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
from cutlass.utils import LayoutEnum

from quack.varlen_utils import VarlenArguments, VarlenManager
from quack.pipeline import make_pipeline_state
from quack import copy_utils
from quack.gemm_sm90 import GemmSm90, NamedBarrierGemm
from quack import sm80_utils


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


@cute.jit
def _sm120_blockscaled_scale_fragment(
    dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    sf_vec_size: cutlass.Constexpr[int],
    is_sfa: cutlass.Constexpr[bool],
):
    if const_expr(sf_vec_size == 16):
        if const_expr(is_sfa):
            return warp.make_mxf4nvf4_sfa_fragment(dtype)
        return warp.make_mxf4nvf4_sfb_fragment(dtype)
    return cute.make_rmem_tensor(cute.make_layout(((32, 2),), stride=((0, 1),)), dtype)


@cute.jit
def _load_sm120_blockscaled_selector0_scale_fragments(
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    stage: cutlass.Int32,
    m_atom_base: cutlass.Int32,
    n_atom_base: cutlass.Int32,
    k_scale_base: cutlass.Int32,
    sf_vec_size: cutlass.Constexpr[int],
    sf_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
):
    """Load selector-zero SM120 FP4 blockscaled scale packets from SMEM.

    The tuple-lowered SM120 blockscaled MMA uses byte-id-a=byte-id-b=0. For
    selector zero, SFA provider lanes map tid 0 to row group and tid 1 to row
    group+8; SFB provider lanes map group 0..7 to logical N columns 0..7.
    """
    lane = cute.arch.lane_idx()
    group = lane >> 2
    tid = lane & 3
    sfa_row = m_atom_base + group + 8 * (tid & 1)
    sfb_col = n_atom_base + group
    sfa = _sm120_blockscaled_scale_fragment(sf_dtype, sf_vec_size, True)
    sfb = _sm120_blockscaled_scale_fragment(sf_dtype, sf_vec_size, False)
    compact_sfa = cute.filter_zeros(sfa)
    compact_sfb = cute.filter_zeros(sfb)
    for kb in cutlass.range_constexpr(64 // sf_vec_size):
        compact_sfa[kb] = sSFA[sfa_row, k_scale_base + kb, stage]
        compact_sfb[kb] = sSFB[sfb_col, k_scale_base + kb, stage]
    return sfa, sfb


@cute.jit
def _make_sm120_fp4_ldmatrix_smem_view(
    smem: cute.Tensor,
    mn: cutlass.Constexpr[int],
):
    return cute.make_tensor(
        smem.iterator,
        cute.make_layout((mn, (8, 4)), stride=(64, (1, 16))),
    )


@cute.jit
def _expand_compact_fp4_to_sm120_ldmatrix_smem(
    compact: cute.Tensor,
    padded: cute.Tensor,
    mn: cutlass.Constexpr[int],
):
    """Expand packed FP4 bytes into the padded SM120 ldmatrix SMEM layout."""
    tidx, _, _ = cute.arch.thread_idx()

    for i in cutlass.range((mn * 64 + 31) // 32, unroll_full=True):
        flat = tidx + i * 32
        if flat < mn * 64:
            padded[flat // 64, flat % 64] = cutlass.Int8(0)

    for i in cutlass.range((mn * 32 + 31) // 32, unroll_full=True):
        flat = tidx + i * 32
        if flat < mn * 32:
            row = flat // 32
            packed_k = flat - row * 32
            group = packed_k // 8
            in_group = packed_k - group * 8
            padded[row, group * 16 + in_group] = compact[row, packed_k]


class GemmSm120(GemmSm90):
    """SM120-style GEMM using warp-level MMA instead of WGMMA.

    Key differences from SM90:
    - Uses MmaF16BF16Op (warp-level, 32 threads) instead of WGMMA (warp-group, 128 threads)
    - Requires explicit SMEM→RMEM copy via ldmatrix before MMA
    - Thread config: num_mma_warps regular warps + 1 DMA warp
    - Pingpong: 2 warp groups of (2,2,1), each processing alternating tiles
    - No fp8 support (warp-level MMA only supports fp16/bf16)
    """

    arch = 120

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Tuple[int, int] | Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = True,
        gather_A: bool = False,
        concat_layout: tuple | None = None,
        use_pdl: bool = True,
        sf_vec_size: int | None = None,
        sf_dtype: Type[cutlass.Numeric] | None = None,
    ):
        # Don't call super().__init__ — we set up our own config
        self.acc_dtype = acc_dtype
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        self.use_clc_persistence = False
        self.use_pdl = use_pdl
        self.fp8_slow_accum = False
        self.gather_A = gather_A
        self.concat_layout = concat_layout or ()
        self.blockscaled = sf_vec_size is not None
        self.sf_vec_size = sf_vec_size
        self.sf_dtype = sf_dtype
        if self.blockscaled:
            if a_dtype is not cutlass.Float4E2M1FN:
                raise ValueError("SM120 blockscaled path currently supports Float4E2M1FN A/B only")
            if acc_dtype is not cutlass.Float32:
                raise ValueError("SM120 blockscaled path requires Float32 accumulation")
            if sf_vec_size not in (16, 32):
                raise ValueError("SM120 blockscaled path supports sf_vec_size 16 or 32")
            expected_sf_dtype = cutlass.Float8E4M3FN if sf_vec_size == 16 else cutlass.Float8E8M0FNU
            if sf_dtype is not expected_sf_dtype:
                raise ValueError(
                    f"SM120 blockscaled sf_vec_size={sf_vec_size} requires {expected_sf_dtype}"
                )
            if pingpong:
                raise NotImplementedError("SM120 blockscaled pingpong is not implemented")
            if gather_A:
                raise NotImplementedError("SM120 blockscaled gather_A is not implemented")
            if cluster_shape_mnk != (1, 1, 1):
                raise ValueError("SM120 blockscaled path requires cluster_shape_mnk=(1,1,1)")
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"
        if gather_A:
            assert cluster_shape_mnk[1] == 1

        self.cluster_shape_mnk = cluster_shape_mnk
        assert len(tile_shape_mnk) in [2, 3], "CTA tile shape must be (M, N) or (M, N, K)"
        # K dimension: if user provides 3 values, use their K; otherwise default in _setup_tiled_mma.
        self.cta_tile_shape_mnk = (
            tuple(tile_shape_mnk) if len(tile_shape_mnk) == 3 else (*tile_shape_mnk, 0)
        )
        tile_M, tile_N = self.cta_tile_shape_mnk[:2]

        # Pingpong: 2 warp groups each with (2,2,1) atom layout
        # Non-pingpong: 1 group of 8 warps with (4,2,1) atom layout
        self.mma_inst_mnk = (16, 8, 64) if self.blockscaled else (16, 8, 16)
        self.atom_layout_mnk = (4, 2, 1) if not self.pingpong else (2, 2, 1)
        # num_mma_warps = total warps doing MMA (both warp groups in pingpong)
        self.num_mma_warps = math.prod(self.atom_layout_mnk) * (1 if not self.pingpong else 2)
        # For compatibility with SM90 code that uses warp groups
        self.num_threads_per_warp_group = 128
        assert self.num_mma_warps % 4 == 0
        self.mma_warp_groups = self.num_mma_warps // 4
        if self.pingpong:
            assert self.mma_warp_groups == 2
        # threads_per_cta must be a multiple of 128 (warp group size) so that
        # the DMA warp's setmaxnreg.dec.sync has a complete warp group to sync with.
        self.threads_per_cta = (self.mma_warp_groups + 1) * self.num_threads_per_warp_group

        self.num_mcast_ctas_a = cluster_shape_mnk[1]
        if gather_A:
            assert self.num_mcast_ctas_a == 1
        self.num_mcast_ctas_b = cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.occupancy = 1
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}")

        # In pingpong, only 1 warp group (4 warps) participates in epilogue at a time
        self.num_epi_warps = (self.mma_warp_groups if not self.pingpong else 1) * 4
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.Epilogue),
            num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )
        self.num_ab_load_warps = 1 if not self.gather_A else 4
        self.ab_load_warp_id = self.num_mma_warps

        if not self.gather_A:
            self.num_regs_load = 40
            self.num_regs_mma = 232
        else:
            self.num_regs_load = 56
            self.num_regs_mma = 224

        self.ab_stage = None
        self.epi_stage = None
        self.epi_m_major = True
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def epi_smem_warp_shape_mnk(self):
        return self.atom_layout_mnk

    def _setup_tiled_mma(self):
        """Set up warp-level MMA (MmaF16BF16Op) and tile K dimension."""
        if const_expr(self.blockscaled):
            if self.sf_vec_size == 16:
                op = warp.MmaMXF4NVF4Op(self.a_dtype, self.acc_dtype, self.sf_dtype)
            else:
                op = warp.MmaMXF4Op(self.a_dtype, self.acc_dtype, self.sf_dtype)
            self.mma_inst_mnk = (16, 8, 64)
        else:
            op = warp.MmaF16BF16Op(self.a_dtype, self.acc_dtype, self.mma_inst_mnk)
        tC = cute.make_layout(self.atom_layout_mnk)
        atom_m, atom_n, atom_k = self.atom_layout_mnk
        # We want each warp to have 16 consecutive elements in the N direction, for STSM
        # and for gated epilogue.
        permutation_n = cute.make_ordered_layout((self.mma_inst_mnk[1], atom_n, 2), order=(0, 2, 1))
        permutation_mnk = (
            atom_m * self.mma_inst_mnk[0],
            permutation_n,
            atom_k * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)
        tile_k = (
            self.cta_tile_shape_mnk[2]
            if self.cta_tile_shape_mnk[2] > 0
            else self.mma_inst_mnk[2] * 4
        )
        assert tile_k > 0, "CTA tile K must be positive"
        assert tile_k % self.mma_inst_mnk[2] == 0, (
            f"CTA tile K ({tile_k}) must be divisible by MMA instruction K ({self.mma_inst_mnk[2]})"
        )
        self.cta_tile_shape_mnk = (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], tile_k)

    # Dense __call__, _setup_attributes, make_ab_pipeline, make_epi_store_pipeline,
    # make_sched_pipeline, epilogue are inherited from GemmSm90.

    @staticmethod
    def padded_blockscale_cols(k: int, sf_vec_size: int) -> int:
        if k % 64 != 0:
            raise ValueError("SM120 blockscaled GEMM requires K divisible by 64")
        return _round_up(k // sf_vec_size, 16)

    @staticmethod
    def can_implement_blockscaled(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        d_dtype: Type[cutlass.Numeric],
        mma_tiler_mnk: Tuple[int, int] | Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        d_major: str,
    ) -> bool:
        del d_major
        if ab_dtype is not cutlass.Float4E2M1FN:
            return False
        if sf_vec_size == 16:
            if sf_dtype is not cutlass.Float8E4M3FN:
                return False
        elif sf_vec_size == 32:
            if sf_dtype is not cutlass.Float8E8M0FNU:
                return False
        else:
            return False
        if d_dtype is not cutlass.BFloat16:
            return False
        if a_major != "k" or b_major != "k":
            return False
        if cluster_shape_mn != (1, 1):
            return False
        if len(mma_tiler_mnk) == 3 and mma_tiler_mnk[2] != 64:
            return False
        if mma_tiler_mnk[0] != 128 or mma_tiler_mnk[1] != 128:
            return False
        return m % 128 == 0 and n % 128 == 0 and k % 64 == 0 and l == 1

    @staticmethod
    def _shape_tuple(tensor: cute.Tensor) -> tuple[int, ...]:
        return tuple(int(dim) for dim in tensor.shape)

    @staticmethod
    def _is_empty_varlen(varlen_args: VarlenArguments) -> bool:
        return (
            varlen_args.mCuSeqlensM is None
            and varlen_args.mCuSeqlensK is None
            and varlen_args.mAIdx is None
        )

    def blockscaled_call(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args: tuple,
        scheduler_args,
        varlen_args: Optional[VarlenArguments],
        stream,
        mSFA: Optional[cute.Tensor] = None,
        mSFB: Optional[cute.Tensor] = None,
        trace_ptr: Optional[cutlass.Int64] = None,
    ):
        varlen_args = self._validate_blockscaled_call(
            mA, mB, mD, mC, mSFA, mSFB, epilogue_args, scheduler_args, varlen_args, trace_ptr
        )
        return self._blockscaled_call_jit(
            mA,
            mB,
            mD,
            varlen_args,
            stream,
            mSFA,
            mSFB,
            trace_ptr,
        )

    @cute.jit
    def _blockscaled_call_jit(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        varlen_args: Optional[VarlenArguments],
        stream,
        mSFA: cute.Tensor,
        mSFB: cute.Tensor,
        trace_ptr: Optional[cutlass.Int64] = None,
    ):
        if const_expr(varlen_args is None):
            varlen_args = VarlenArguments()
        return self._call_blockscaled(
            mA,
            mB,
            mD,
            varlen_args,
            stream,
            mSFA,
            mSFB,
            trace_ptr,
        )

    def _validate_blockscaled_call(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mC: Optional[cute.Tensor],
        mSFA: Optional[cute.Tensor],
        mSFB: Optional[cute.Tensor],
        epilogue_args: tuple,
        scheduler_args,
        varlen_args: Optional[VarlenArguments],
        trace_ptr: Optional[cutlass.Int64],
    ) -> VarlenArguments:
        if mSFA is None or mSFB is None:
            raise ValueError("SM120 blockscaled GEMM requires SFA and SFB scale tensors")
        if mD is None:
            raise ValueError("SM120 blockscaled GEMM requires an output tensor")
        if mC is not None:
            raise NotImplementedError("SM120 blockscaled C/beta path is not implemented")
        if epilogue_args not in (None, ()):
            beta = getattr(epilogue_args, "beta", None)
            add_to_output = getattr(epilogue_args, "add_to_output", False)
            if beta not in (None, 0) or add_to_output:
                raise NotImplementedError("SM120 blockscaled C/beta path is not implemented")
        del scheduler_args
        if trace_ptr is not None:
            raise NotImplementedError("SM120 blockscaled trace path is not implemented")
        if self.cta_tile_shape_mnk != (128, 128, 64):
            raise NotImplementedError("SM120 blockscaled path currently supports 128x128x64 tiles")
        if self.cluster_shape_mnk != (1, 1, 1):
            raise NotImplementedError("SM120 blockscaled path requires cluster_shape_mnk=(1,1,1)")
        if (
            mA.element_type is not cutlass.Float4E2M1FN
            or mB.element_type is not cutlass.Float4E2M1FN
        ):
            raise TypeError("SM120 blockscaled path requires Float4E2M1FN A/B")
        if mSFA.element_type is not self.sf_dtype or mSFB.element_type is not self.sf_dtype:
            raise TypeError(f"SM120 blockscaled path requires {self.sf_dtype} SFA/SFB")
        if mD.element_type is not cutlass.BFloat16:
            raise NotImplementedError("SM120 blockscaled path currently supports only BF16 D")

        if varlen_args is None:
            varlen_args = VarlenArguments()
        if not self._is_empty_varlen(varlen_args):
            raise NotImplementedError("SM120 blockscaled varlen GEMM is not implemented")

        a_shape = self._shape_tuple(mA)
        b_shape = self._shape_tuple(mB)
        d_shape = self._shape_tuple(mD)
        if len(a_shape) != 3 or len(b_shape) != 3 or len(d_shape) != 3:
            raise ValueError("SM120 blockscaled tensors must use logical rank-3 shapes")
        m, k, l = a_shape
        n, kb, lb = b_shape
        if k != kb or l != lb:
            raise ValueError("SM120 blockscaled A/B K and L extents must match")
        if k % 64 != 0:
            if k * 2 % 64 == 0:
                raise ValueError(
                    "SM120 blockscaled class call expects logical Float4E2M1FN K extent; "
                    "use compile_blockscaled_gemm_tvm_ffi for packed torch.float4_e2m1fn_x2 "
                    "storage"
                )
            raise ValueError("SM120 blockscaled path requires logical K to be divisible by 64")
        if d_shape != (m, n, l):
            raise ValueError(f"SM120 blockscaled D shape must be {(m, n, l)}, got {d_shape}")
        if m % 128 != 0 or n % 128 != 0:
            raise NotImplementedError("SM120 blockscaled path requires M/N multiples of 128")
        if l != 1:
            raise NotImplementedError("SM120 blockscaled path currently supports L=1")
        scale_cols = self.padded_blockscale_cols(k, self.sf_vec_size)
        if self._shape_tuple(mSFA) != (m, scale_cols, l):
            raise ValueError(
                f"SFA shape must be {(m, scale_cols, l)}, got {self._shape_tuple(mSFA)}"
            )
        if self._shape_tuple(mSFB) != (n, scale_cols, l):
            raise ValueError(
                f"SFB shape must be {(n, scale_cols, l)}, got {self._shape_tuple(mSFB)}"
            )
        return varlen_args

    @cute.jit
    def _call_blockscaled(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        varlen_args: Optional[VarlenArguments],
        stream,
        mSFA: cute.Tensor,
        mSFB: cute.Tensor,
        trace_ptr: Optional[cutlass.Int64] = None,
    ):
        del stream, trace_ptr

        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type
        self.c_dtype = None
        self.sf_dtype = mSFA.element_type
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD)
        self.c_layout = None
        self._setup_attributes(())
        self.ab_stage = 1
        tile_m, tile_n, tile_k = self.cta_tile_shape_mnk

        self.a_smem_layout_staged = cute.make_layout(
            (tile_m, tile_k, self.ab_stage), stride=(tile_k, 1, tile_m * tile_k)
        )
        self.b_smem_layout_staged = cute.make_layout(
            (tile_n, tile_k, self.ab_stage), stride=(tile_k, 1, tile_n * tile_k)
        )
        scale_tile_k = 16
        a_compact_smem_layout_staged = cute.make_layout(
            (tile_m, tile_k // 2, self.ab_stage),
            stride=(tile_k // 2, 1, tile_m * (tile_k // 2)),
        )
        b_compact_smem_layout_staged = cute.make_layout(
            (tile_n, tile_k // 2, self.ab_stage),
            stride=(tile_k // 2, 1, tile_n * (tile_k // 2)),
        )
        self.sfa_smem_layout_staged = cute.make_layout(
            (tile_m, scale_tile_k, self.ab_stage),
            stride=(scale_tile_k, 1, tile_m * scale_tile_k),
        )
        self.sfb_smem_layout_staged = cute.make_layout(
            (tile_n, scale_tile_k, self.ab_stage),
            stride=(scale_tile_k, 1, tile_n * scale_tile_k),
        )
        acc_smem_layout = cute.make_layout((tile_m, tile_n), stride=(tile_n, 1))

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        a_compact_smem_layout = cute.slice_(a_compact_smem_layout_staged, (None, None, 0))
        b_compact_smem_layout = cute.slice_(b_compact_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, 0))

        m_extent = cute.size(mA, mode=[0])
        k_extent = cute.size(mA, mode=[1])
        packed_k_extent = k_extent // 2
        n_extent = cute.size(mB, mode=[0])
        l_extent = cute.size(mA, mode=[2])
        mA_u8 = cute.make_tensor(
            cute.recast_ptr(mA.iterator, dtype=cutlass.Uint8),
            cute.make_layout(
                (m_extent, packed_k_extent, l_extent),
                stride=(packed_k_extent, 1, m_extent * packed_k_extent),
            ),
        )
        mB_u8 = cute.make_tensor(
            cute.recast_ptr(mB.iterator, dtype=cutlass.Uint8),
            cute.make_layout(
                (n_extent, packed_k_extent, l_extent),
                stride=(packed_k_extent, 1, n_extent * packed_k_extent),
            ),
        )
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            op, mA_u8, a_compact_smem_layout, (tile_m, tile_k // 2)
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            op, mB_u8, b_compact_smem_layout, (tile_n, tile_k // 2)
        )
        tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
            mSFA, sfa_smem_layout, (tile_m, scale_tile_k), 1
        )
        tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
            mSFB, sfb_smem_layout, (tile_n, scale_tile_k), 1
        )
        self.num_tma_load_bytes = (
            cute.size_in_bytes(cutlass.Uint8, a_compact_smem_layout)
            + cute.size_in_bytes(cutlass.Uint8, b_compact_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        )

        a_compact_smem_size = cute.cosize(a_compact_smem_layout_staged)
        b_compact_smem_size = cute.cosize(b_compact_smem_layout_staged)
        sfa_smem_size = cute.cosize(self.sfa_smem_layout_staged)
        sfb_smem_size = cute.cosize(self.sfb_smem_layout_staged)
        acc_smem_size = cute.cosize(acc_smem_layout)

        @cute.struct
        class BlockscaledSharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sACompact: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, a_compact_smem_size],
                self.buffer_align_bytes,
            ]
            sBCompact: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, b_compact_smem_size],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfa_smem_size],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb_smem_size],
                self.buffer_align_bytes,
            ]
            # Correctness scratch: for K > 64, K64 partials stay in FP32 here
            # until the final K tile writes BF16 D exactly once.
            sAcc: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, acc_smem_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = BlockscaledSharedStorage

        if const_expr(self.sf_vec_size == 16):
            mma_op = warp.MmaMXF4NVF4Op(cutlass.Float4E2M1FN, cutlass.Float32, self.sf_dtype)
        else:
            mma_op = warp.MmaMXF4Op(cutlass.Float4E2M1FN, cutlass.Float32, self.sf_dtype)
        one_warp_mma = cute.make_tiled_mma(mma_op)
        varlen_params = VarlenManager.to_underlying_arguments(varlen_args)
        self.blockscaled_kernel(
            one_warp_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            mD,
            varlen_params,
            cute.make_layout((1, 1, 1)),
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            a_compact_smem_layout_staged,
            b_compact_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            acc_smem_layout,
        ).launch(
            grid=[cute.ceil_div(m_extent, tile_m), cute.ceil_div(n_extent, tile_n), l_extent],
            block=[64, 1, 1],
            cluster=(1, 1, 1),
        )

    @cute.kernel
    def blockscaled_kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl16: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl16: cute.Tensor,
        mD_mnl: cute.Tensor,
        varlen_params: VarlenManager.Params,
        cluster_layout_mnk: cute.Layout,
        a_smem_layout: cute.Layout,
        b_smem_layout: cute.Layout,
        a_compact_smem_layout: cute.Layout,
        b_compact_smem_layout: cute.Layout,
        sfa_smem_layout: cute.Layout,
        sfb_smem_layout: cute.Layout,
        acc_smem_layout: cute.Layout,
    ):
        del varlen_params

        tidx, _, _ = cute.arch.thread_idx()
        cta_m, cta_n, cta_l = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 1:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
        )

        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
        sA = storage.sA.get_tensor(a_smem_layout)
        sB = storage.sB.get_tensor(b_smem_layout)
        sACompact = storage.sACompact.get_tensor(a_compact_smem_layout)
        sBCompact = storage.sBCompact.get_tensor(b_compact_smem_layout)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout)
        sAcc = storage.sAcc.get_tensor(acc_smem_layout)
        pipeline_init_wait(cluster_shape_mn=(1, 1))

        k_tile_count = cute.size(mA_mkl, mode=[1]) // (self.cta_tile_shape_mnk[2] // 2)
        scales_per_k_tile = 64 // self.sf_vec_size

        if warp_idx == 1:
            producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            gA_mk = cute.local_tile(
                mA_mkl[None, None, cta_l],
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2] // 2),
                (cta_m, None),
            )
            gB_nk = cute.local_tile(
                mB_nkl[None, None, cta_l],
                (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2] // 2),
                (cta_n, None),
            )
            gSFA_mk16 = cute.local_tile(
                mSFA_mkl16[None, None, cta_l],
                (self.cta_tile_shape_mnk[0], 16),
                (cta_m, None),
            )
            gSFB_nk16 = cute.local_tile(
                mSFB_nkl16[None, None, cta_l],
                (self.cta_tile_shape_mnk[1], 16),
                (cta_n, None),
            )
            if const_expr(k_tile_count == 1):
                producer_state = self.load_blockscaled_tma_tile(
                    ab_pipeline,
                    producer_state,
                    tma_atom_a,
                    gA_mk,
                    tma_atom_b,
                    gB_nk,
                    tma_atom_sfa,
                    gSFA_mk16,
                    tma_atom_sfb,
                    gSFB_nk16,
                    sACompact,
                    sBCompact,
                    sSFA,
                    sSFB,
                )
            else:
                for k_tile in cutlass.range(k_tile_count, unroll=1):
                    scale_base = k_tile * scales_per_k_tile
                    scale_page = scale_base // 16
                    producer_state = self.load_blockscaled_tma_tile_indexed(
                        ab_pipeline,
                        producer_state,
                        k_tile,
                        scale_page,
                        tma_atom_a,
                        gA_mk,
                        tma_atom_b,
                        gB_nk,
                        tma_atom_sfa,
                        gSFA_mk16,
                        tma_atom_sfb,
                        gSFB_nk16,
                        sACompact,
                        sBCompact,
                        sSFA,
                        sSFB,
                    )
            ab_pipeline.producer_tail(producer_state)

        if warp_idx == 0:
            gD_mn = cute.local_tile(
                mD_mnl[None, None, cta_l],
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]),
                (cta_m, cta_n),
            )
            read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            self.mma_blockscaled_kloop_store_bf16(
                ab_pipeline,
                read_state,
                tiled_mma,
                sA,
                sB,
                sACompact,
                sBCompact,
                sSFA,
                sSFB,
                sAcc,
                gD_mn,
                k_tile_count,
                tidx,
            )

    @cute.jit
    def load_blockscaled_tma_tile(
        self,
        ab_pipeline: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        tma_atom_a: cute.CopyAtom,
        gA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        gB_nk: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        gSFA_mk16: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        gSFB_nk16: cute.Tensor,
        sACompact: cute.Tensor,
        sBCompact: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
    ) -> pipeline.PipelineState:
        copy_A, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_a,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gA_mk,
            dst_tensor=sACompact,
            mcast_mask=0,
        )
        copy_B, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_b,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gB_nk,
            dst_tensor=sBCompact,
            mcast_mask=0,
        )
        copy_SFA, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_sfa,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gSFA_mk16,
            dst_tensor=sSFA,
            mcast_mask=0,
        )
        copy_SFB, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_sfb,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gSFB_nk16,
            dst_tensor=sSFB,
            mcast_mask=0,
        )
        return self.load_tma(
            ab_pipeline,
            producer_state,
            [copy_A, copy_B, copy_SFA, copy_SFB],
            Int32(1),
        )

    @cute.jit
    def load_blockscaled_tma_tile_indexed(
        self,
        ab_pipeline: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        k_tile: cutlass.Int32,
        scale_page: cutlass.Int32,
        tma_atom_a: cute.CopyAtom,
        gA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        gB_nk: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        gSFA_mk16: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        gSFB_nk16: cute.Tensor,
        sACompact: cute.Tensor,
        sBCompact: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
    ) -> pipeline.PipelineState:
        copy_A, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_a,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gA_mk,
            dst_tensor=sACompact,
            mcast_mask=0,
        )
        copy_B, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_b,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gB_nk,
            dst_tensor=sBCompact,
            mcast_mask=0,
        )
        copy_SFA, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_sfa,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gSFA_mk16,
            dst_tensor=sSFA,
            mcast_mask=0,
        )
        copy_SFB, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_sfb,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=gSFB_nk16,
            dst_tensor=sSFB,
            mcast_mask=0,
        )
        peek_empty_status = ab_pipeline.producer_try_acquire(producer_state)
        ab_pipeline.producer_acquire(producer_state, peek_empty_status)
        tma_bar_ptr = ab_pipeline.producer_get_barrier(producer_state)
        smem_idx = producer_state.index
        copy_A(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
        copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
        copy_SFA(scale_page, smem_idx, tma_bar_ptr=tma_bar_ptr)
        copy_SFB(scale_page, smem_idx, tma_bar_ptr=tma_bar_ptr)
        ab_pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def mma_blockscaled_kloop_store_bf16(
        self,
        ab_pipeline: pipeline.PipelineAsync,
        read_state: pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sACompact: cute.Tensor,
        sBCompact: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sAcc: cute.Tensor,
        gD_mn: cute.Tensor,
        k_tile_count: cutlass.Int32,
        tidx: cutlass.Int32,
    ) -> None:
        thr_mma = tiled_mma.get_slice(tidx)
        a_shape = tiled_mma.partition_shape_A((16, 64))
        b_shape = tiled_mma.partition_shape_B((8, 64))
        acc_shape = tiled_mma.partition_shape_C((16, 8))
        accum_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
        store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gD_mn.element_type)
        scales_per_k_tile = 64 // self.sf_vec_size
        if const_expr(k_tile_count == 1):
            read_state = self.mma_blockscaled_one_k_tile_accumulate_smem(
                ab_pipeline,
                read_state,
                tiled_mma,
                sA,
                sB,
                sACompact,
                sBCompact,
                sSFA,
                sSFB,
                sAcc,
                gD_mn,
                tidx,
                thr_mma,
                a_shape,
                b_shape,
                acc_shape,
                accum_atom,
                store_atom,
                scales_per_k_tile,
                cutlass.Int32(0),
                False,
                True,
            )
        else:
            read_state = self.mma_blockscaled_one_k_tile_accumulate_smem(
                ab_pipeline,
                read_state,
                tiled_mma,
                sA,
                sB,
                sACompact,
                sBCompact,
                sSFA,
                sSFB,
                sAcc,
                gD_mn,
                tidx,
                thr_mma,
                a_shape,
                b_shape,
                acc_shape,
                accum_atom,
                store_atom,
                scales_per_k_tile,
                cutlass.Int32(0),
                False,
                False,
            )
            for k_iter in cutlass.range(k_tile_count - 2, unroll=1):
                k_tile = k_iter + 1
                read_state = self.mma_blockscaled_one_k_tile_accumulate_smem(
                    ab_pipeline,
                    read_state,
                    tiled_mma,
                    sA,
                    sB,
                    sACompact,
                    sBCompact,
                    sSFA,
                    sSFB,
                    sAcc,
                    gD_mn,
                    tidx,
                    thr_mma,
                    a_shape,
                    b_shape,
                    acc_shape,
                    accum_atom,
                    store_atom,
                    scales_per_k_tile,
                    k_tile,
                    True,
                    False,
                )
            read_state = self.mma_blockscaled_one_k_tile_accumulate_smem(
                ab_pipeline,
                read_state,
                tiled_mma,
                sA,
                sB,
                sACompact,
                sBCompact,
                sSFA,
                sSFB,
                sAcc,
                gD_mn,
                tidx,
                thr_mma,
                a_shape,
                b_shape,
                acc_shape,
                accum_atom,
                store_atom,
                scales_per_k_tile,
                k_tile_count - 1,
                True,
                True,
            )

    @cute.jit
    def mma_blockscaled_one_k_tile_accumulate_smem(
        self,
        ab_pipeline: pipeline.PipelineAsync,
        read_state: pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sACompact: cute.Tensor,
        sBCompact: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sAcc: cute.Tensor,
        gD_mn: cute.Tensor,
        tidx: cutlass.Int32,
        thr_mma: cute.TiledMmaThrVal,
        a_shape: cute.Shape,
        b_shape: cute.Shape,
        acc_shape: cute.Shape,
        accum_atom: cute.CopyAtom,
        store_atom: cute.CopyAtom,
        scales_per_k_tile: cutlass.Constexpr[int],
        k_tile: cutlass.Int32,
        add_existing: cutlass.Constexpr[bool],
        store_final: cutlass.Constexpr[bool],
    ) -> pipeline.PipelineState:
        peek_ab_full_status = ab_pipeline.consumer_try_wait(read_state)
        ab_pipeline.consumer_wait(read_state, peek_ab_full_status)
        _expand_compact_fp4_to_sm120_ldmatrix_smem(
            sACompact[None, None, read_state.index],
            sA[None, None, read_state.index],
            128,
        )
        _expand_compact_fp4_to_sm120_ldmatrix_smem(
            sBCompact[None, None, read_state.index],
            sB[None, None, read_state.index],
            128,
        )
        cute.arch.sync_warp()
        scale_base = k_tile * scales_per_k_tile
        scale_page = scale_base // 16
        scale_page_offset = scale_base - scale_page * 16
        for m_atom in cutlass.range_constexpr(8):
            for n_atom in cutlass.range_constexpr(16):
                acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
                acc.fill(0.0)
                self.mma_blockscaled_tile_k64_accumulate(
                    tiled_mma,
                    acc,
                    sA[None, None, read_state.index],
                    sB[None, None, read_state.index],
                    sSFA,
                    sSFB,
                    read_state.index,
                    scale_page_offset,
                    cutlass.Int32(m_atom * 16),
                    cutlass.Int32(n_atom * 8),
                    tidx,
                    a_shape,
                    b_shape,
                )
                self.store_blockscaled_accum_smem_atom(
                    thr_mma,
                    accum_atom,
                    acc,
                    sAcc,
                    store_atom,
                    gD_mn,
                    m_atom,
                    n_atom,
                    add_existing,
                    store_final,
                )
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        ab_pipeline.consumer_release(read_state)
        read_state.advance()
        return read_state

    @cute.jit
    def mma_blockscaled(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        sA_stage: cute.Tensor,
        sB_stage: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        stage: cutlass.Int32,
        m_atom_base: cutlass.Int32,
        n_atom_base: cutlass.Int32,
        k_scale_base: cutlass.Int32,
        tidx: cutlass.Int32,
        a_shape: cute.Shape,
        b_shape: cute.Shape,
    ) -> None:
        sA_atom = cute.domain_offset((m_atom_base, None), sA_stage)
        sB_atom = cute.domain_offset((n_atom_base, None), sB_stage)
        copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x16x8bOp(num_matrices=1, unpack_bits=4),
            cutlass.Int8,
        )
        tiled_copy_a = cute.make_tiled_copy_A(copy_atom, tiled_mma)
        tiled_copy_b = cute.make_tiled_copy_B(copy_atom, tiled_mma)
        a0, a1, a2, a3 = warp.sm120_mxf4nvf4_ldmatrix_A_regs(
            tiled_copy_a,
            tidx,
            _make_sm120_fp4_ldmatrix_smem_view(sA_atom, 16),
        )
        b0, b1 = warp.sm120_mxf4nvf4_ldmatrix_B_regs(
            tiled_copy_b,
            tidx,
            _make_sm120_fp4_ldmatrix_smem_view(sB_atom, 8),
        )
        a = cute.make_rmem_tensor(a_shape, cutlass.Float4E2M1FN)
        b = cute.make_rmem_tensor(b_shape, cutlass.Float4E2M1FN)
        a_i32 = cute.recast_tensor(a, cutlass.Int32)
        b_i32 = cute.recast_tensor(b, cutlass.Int32)
        # The asymmetric SM120 blockscaled tests catch this placement.
        a_i32[0] = a0
        a_i32[1] = a2
        a_i32[2] = a1
        a_i32[3] = a3
        b_i32[0] = b0
        b_i32[1] = b1
        sfa, sfb = _load_sm120_blockscaled_selector0_scale_fragments(
            sSFA,
            sSFB,
            stage,
            m_atom_base,
            n_atom_base,
            k_scale_base,
            self.sf_vec_size,
            self.sf_dtype,
        )
        cute.gemm(tiled_mma, acc, (a, sfa), (b, sfb), acc)

    @cute.jit
    def mma_blockscaled_tile_k64_accumulate(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        stage: cutlass.Int32,
        scale_page_offset: cutlass.Int32,
        m_atom_base: cutlass.Int32,
        n_atom_base: cutlass.Int32,
        tidx: cutlass.Int32,
        a_shape: cute.Shape,
        b_shape: cute.Shape,
    ) -> None:
        del tiled_mma
        if const_expr(self.sf_vec_size == 16):
            mma_op = warp.MmaMXF4NVF4Op(cutlass.Float4E2M1FN, cutlass.Float32, self.sf_dtype)
        else:
            mma_op = warp.MmaMXF4Op(cutlass.Float4E2M1FN, cutlass.Float32, self.sf_dtype)
        local_tiled_mma = cute.make_tiled_mma(mma_op)
        self.mma_blockscaled(
            local_tiled_mma,
            acc,
            sA,
            sB,
            sSFA,
            sSFB,
            stage,
            m_atom_base,
            n_atom_base,
            scale_page_offset,
            tidx,
            a_shape,
            b_shape,
        )

    @cute.jit
    def store_blockscaled_accum_smem_atom(
        self,
        thr_mma: cute.TiledMmaThrVal,
        accum_atom: cute.CopyAtom,
        acc: cute.Tensor,
        sAcc: cute.Tensor,
        store_atom: cute.CopyAtom,
        mD_mn: cute.Tensor,
        m_atom: cutlass.Constexpr[int],
        n_atom: cutlass.Constexpr[int],
        add_existing: cutlass.Constexpr[bool],
        store_final: cutlass.Constexpr[bool],
    ) -> None:
        sAcc_atom = cute.local_tile(sAcc, (16, 8), (m_atom, n_atom))
        tCsAcc = thr_mma.partition_C(sAcc_atom)
        if const_expr(add_existing):
            tCrPrev = cute.make_rmem_tensor(acc.layout, cutlass.Float32)
            cute.copy(accum_atom, tCsAcc, tCrPrev)
            acc.store(acc.load() + tCrPrev.load())
        if const_expr(store_final):
            gD_atom = cute.local_tile(mD_mn, (16, 8), (m_atom, n_atom))
            tCgD = thr_mma.partition_C(gD_atom)
            tCrD = cute.make_rmem_tensor(acc.layout, mD_mn.element_type)
            tCrD.store(acc.load().to(mD_mn.element_type))
            cute.copy(store_atom, tCrD, tCgD)
        else:
            cute.copy(accum_atom, acc, tCsAcc)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: Optional[cute.CopyAtom],
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
        trace_ptr: Optional[cutlass.Int64] = None,
    ):
        from quack.trace import TraceContext

        tctx = TraceContext.create(trace_ptr)

        varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
        varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
        if const_expr(self.gather_A):
            assert varlen_m or varlen_k
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptors
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
        )
        epi_pipeline = None
        has_epi_load = const_expr(self.epi_c_stage > 0)
        if const_expr(has_epi_load):
            epi_pipeline = self.make_epi_pipeline(
                epi_pipeline_mbar_ptr=storage.epi_pipeline_array_ptr.data_ptr(),
                tx_count=self.epi_load_bytes_per_stage,
            )
        sched_pipeline = None
        sched_data = None
        if const_expr(self.is_persistent):
            sched_pipeline = self.make_sched_pipeline(
                cluster_layout_mnk,
                sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
                varlen_k=varlen_k,
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
            len_m_static=Int32(
                cute.size(mA_mkl, mode=[0])
                if varlen_k or varlen_params.mAIdx is None
                else varlen_params.mAIdx.shape[0]
            ),
            len_k_static=Int32(cute.size(mA_mkl, mode=[1])),
        )

        TileSchedulerCls = partial(
            TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline
        )

        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if (
                warp_idx >= self.ab_load_warp_id
                and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
                # Get mcast mask
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1
                )
                b_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                # Persistent tile scheduling loop
                is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                if const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                while work_tile.is_valid_tile:
                    tctx.b("tma_load")
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    # Local_tile partition global tensors
                    copy_A, prefetch_A = None, None
                    if const_expr(not self.gather_A):
                        mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                        # (bM, bK, RestK)
                        gA_mk = cute.local_tile(
                            mA_mk,
                            cute.select(self.cta_tile_shape_mnk, [0, 2]),
                            (tile_coord_mnkl[0], None),
                        )
                        #  TMA load A partition_S/D
                        copy_A, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_a,
                            cta_coord=block_in_cluster_coord_mnk[1],
                            cta_layout=cute.make_layout(
                                cute.slice_(cluster_layout_mnk, (0, None, 0)).shape
                            ),
                            src_tensor=gA_mk,
                            dst_tensor=sA,
                            mcast_mask=a_mcast_mask,
                        )
                    else:
                        copy_A, prefetch_A = self._make_gather_A_copy(
                            mA_mkl, sA, varlen_manager, tile_coord_mnkl, batch_idx
                        )
                    # (bN, bK, RestK)
                    gB_nk = cute.local_tile(
                        varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                        cute.select(self.cta_tile_shape_mnk, [1, 2]),
                        (tile_coord_mnkl[1], None),
                    )
                    # TMA load B partition_S/D
                    copy_B, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_b,
                        cta_coord=block_in_cluster_coord_mnk[0],
                        cta_layout=cute.make_layout(
                            cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape
                        ),
                        src_tensor=gB_nk,
                        dst_tensor=sB,
                        mcast_mask=b_mcast_mask,
                    )
                    len_k = varlen_manager.len_k(batch_idx)
                    k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    if const_expr(not self.gather_A):
                        ab_producer_state = self.load_tma(
                            ab_pipeline, ab_producer_state, [copy_A, copy_B], k_tile_cnt
                        )
                    else:
                        ab_producer_state = self.load_AB_gather_A(
                            ab_pipeline,
                            ab_producer_state,
                            copy_A,
                            prefetch_A,
                            copy_B,
                            k_tile_cnt,
                            varlen_m=varlen_m,
                        )
                    tctx.e("tma_load")
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                if const_expr(self.pingpong and not varlen_k):
                    # Need to write the tile_idx to smem for the next WG in the pingpong mode
                    if is_scheduler_warp:
                        tile_scheduler.write_work_tile_to_smem(work_tile)
                    work_tile = tile_scheduler.get_current_work()
                ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

        # =====================================================================
        # MMA warps
        # =====================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            is_tma_warp = Boolean(
                (not self.pingpong and warp_idx == 0)
                or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            tidx, _, _ = cute.arch.thread_idx()
            # For pingpong, adjust tidx to within-warp-group index
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group

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

            # Make fragments
            thr_mma = tiled_mma.get_slice(tidx)
            acc, tCsA, tCsB, tCrA, tCrB = sm80_utils.partition_fragment_ABC(
                thr_mma, self.cta_tile_shape_mnk, sA, sB
            )

            if const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # WG0 needs a start signal at the very beginning
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            k_tile_cnt_static = cute.ceil_div(
                cute.size(mA_mkl, mode=[1]), self.cta_tile_shape_mnk[2]
            )
            c_tile_cnt = cute.size(cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile))

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

            if const_expr(self.pingpong):
                if warp_idx >= 4:
                    # Advance 2nd Math WG pipeline states to the end of 1st Math WG
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    if const_expr(not varlen_k):
                        ab_read_state.advance_iters(k_tile_cnt_static)
                    else:
                        len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                        k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        ab_read_state.advance_iters(k_tile_cnt)
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                len_k = varlen_manager.len_k(batch_idx)
                k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                acc.fill(0.0)
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                tctx.b("mma")
                ab_read_state = self.mma(
                    ab_pipeline,
                    ab_read_state,
                    tiled_mma,
                    acc,
                    k_tile_cnt,
                    smem_tiled_copy_A,
                    smem_tiled_copy_B,
                    tCsA_copy_view,
                    tCsB_copy_view,
                    tCrA,
                    tCrB,
                )
                if const_expr(self.pingpong):
                    # Cue for next WG's MMA to start
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
                tctx.e("mma")

                # ============================================================
                # EPILOGUE — reuse SM90's epilogue flow
                # ============================================================
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")
                tctx.b("epilogue")

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
                if const_expr(has_epi_load):
                    tile_load_copy_fns = self.epi_tile_load_g2s_copy_fns(
                        epilogue_params,
                        epi_smem_tensors,
                        tile_coord_mnkl,
                        varlen_manager,
                        epi_pipeline,
                    )
                    copy_C = copy_utils.chain_tma_producer_copy_fns((copy_C, *tile_load_copy_fns))

                d_dtype_for_layout = self.d_dtype if self.d_dtype is not None else cutlass.BFloat16
                tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                    tiled_mma, self.d_layout, d_dtype_for_layout, sD, tidx
                )
                # (R2S, R2S_M, R2S_N, (epi_M, epi_N))
                tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
                load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
                if const_expr(has_C):
                    tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                        tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
                    )
                else:
                    tiled_copy_s2r, tSR_sC, tRS_rC, tSR_rC = None, None, None, None

                self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)

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

                if const_expr(self.pingpong):
                    # With pingpong, 2 WGs write two different output tiles to the same smem,
                    # so we have to make sure the smem content is done reading before signaling
                    # the next WG's epilogue.
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
                tctx.e("epilogue")

                if const_expr(not self.pingpong):
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                else:  # Skip a tile for pingpong
                    # Update starting load/store pipeline states for the next tile
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    # Update starting mainloop pipeline state for the next tile
                    if const_expr(not varlen_k):
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                        if work_tile.is_valid_tile:
                            len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                            k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                            ab_read_state.advance_iters(k_tile_cnt)
                            tile_scheduler.advance_to_next_work()
                            work_tile = tile_scheduler.get_current_work()

            # Wait for D store complete
            if const_expr(not self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()

        tctx.flush()

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
        load_sB = partial(cute.copy, smem_tiled_copy_B)

        num_k_blocks = cute.size(tCrA, mode=[2])
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
                    # TMA writes this smem stage through the async proxy, while ldmatrix
                    # reads it through the generic proxy. Fence before release so the
                    # producer's next async-proxy write cannot race those reads; sync the
                    # warp because only one lane signals the empty mbarrier.
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
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
                    # TMA writes this smem stage through the async proxy, while ldmatrix
                    # reads it through the generic proxy. Fence before release so the
                    # producer's next async-proxy write cannot race those reads; sync the
                    # warp because only one lane signals the empty mbarrier.
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    ab_pipeline.consumer_release(ab_read_state)
                    ab_read_state.advance()
                if const_expr(k_next > 0):
                    load_sA(tCsA_p[None, None, k_next], tCrA_copy_view[None, None, k_next])
                    load_sB(tCsB_p[None, None, k_next], tCrB_copy_view[None, None, k_next])
                cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)

        return ab_read_state

    @staticmethod
    def _compute_tile_shape_or_override(
        cta_tile_shape_mnk: Tuple[int, int, int],
        atom_layout_mnk: Tuple[int, int, int],
        element_type: Optional[Type[cutlass.Numeric]] = None,
        epi_tile_override: Tuple[int, int] | None = None,
    ) -> Tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param cta_tile_shape_mnk: CTA tile shape (M,N,K)
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        n_perf = 64 if element_type is not None and element_type.width == 8 else 32
        tile_m = math.gcd(64, cute.size(cta_tile_shape_mnk, mode=[0]))
        tile_n = math.gcd(n_perf, cute.size(cta_tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)
