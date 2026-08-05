# Copyright (c) 2025-2026, Tri Dao.

from typing import Callable, Literal, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.nvgpu import warp, warpgroup, tcgen05
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cutlass_dsl import Numeric
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.utils import LayoutEnum


@cute.jit
def gemm_sm100(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    frag_A: cute.Tensor,
    frag_B: cute.Tensor,
    stage,
    *,
    stage_B=None,
    zero_init=False,
    pre_kblock_fn: Optional[Callable] = None,
) -> None:
    """Issue one tcgen05 GEMM over all static K-blocks for a staged A/B view."""
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    stage_B = stage if cutlass.const_expr(stage_B is None) else stage_B
    for k_blk in cutlass.range_constexpr(cute.size(frag_A, mode=[2])):
        if cutlass.const_expr(pre_kblock_fn is not None):
            pre_kblock_fn(mma_atom, k_blk)
        mma_atom.set(tcgen05.Field.ACCUMULATE, not zero_init or k_blk != 0)
        cute.gemm(
            mma_atom,
            acc,
            frag_A[None, None, k_blk, stage],
            frag_B[None, None, k_blk, stage_B],
            acc,
        )


def operand_leading_atom(rows: int, cta_group: int) -> Tuple[int, int]:
    """Return `(atom_rows, rest_rows)` for a per-CTA MMA operand tile."""
    assert cta_group in (1, 2), f"MMA operand layouts support cta_group 1 or 2, got {cta_group}"
    full_rows = rows * cta_group
    inst_rows = full_rows if full_rows <= 256 else full_rows // 2
    assert full_rows % inst_rows == 0, (
        f"full leading dim {full_rows} must be divisible by instruction leading dim {inst_rows}"
    )
    assert inst_rows % cta_group == 0, (
        f"instruction leading dim {inst_rows} must be divisible by cta_group {cta_group}"
    )
    return inst_rows // cta_group, full_rows // inst_rows


def resolve_mma_inst_k(dtype: Type[Numeric], mma_inst_k: Optional[int] = None) -> int:
    if mma_inst_k is not None:
        assert mma_inst_k > 0, f"mma_inst_k must be positive, got {mma_inst_k}"
        return mma_inst_k
    return 256 // dtype.width


def make_tiled_mma_for_arch(
    spec,
    source: Literal["SS", "RS", "TS"] = "SS",
    atom_layout_mnk: Tuple[int, int, int] = (1, 1, 1),
    acc_dtype: Type[cutlass.Numeric] = Float32,
    permutation_mnk: Optional[Tuple[int, int, int]] = None,
    arch=None,
) -> cute.TiledMma:
    """Arch-dispatched TiledMma builder for a MatmulSpec.

    Source modes are arch-specific: SM90 accepts SS/RS, SM100 accepts SS/TS.
    cta_group comes off the spec (a storage-distribution property of the
    operands); spec.M/N/K are full-tile dims.
    """
    if arch is None:
        arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    cta_group = spec.cta_group
    if arch.major == 9:  # Hopper — WGMMA
        assert cta_group == 1, f"SM90 tiled_mma requires cta_group=1, got {cta_group}"
        assert source in ("SS", "RS"), f"SM90 tiled_mma source must be SS or RS, got {source}"
        assert permutation_mnk is None, "SM90 tiled_mma does not accept permutation_mnk"
        # WGMMA RS source means the physical A operand is a register fragment,
        # whose layout convention is K-major. This can differ from the logical
        # TensorSpec storage (e.g. a P.T view may have MN-major SMEM backing but
        # be fed directly from registers). Treat RS A as K-major regardless of
        # the spec's SMEM layout.
        a_major = "K" if source == "RS" else spec._operand_major(spec.A, is_A=True)
        b_major = spec._operand_major(spec.B, is_A=False)
        mode = {"K": cute.nvgpu.OperandMajorMode.K, "MN": cute.nvgpu.OperandMajorMode.MN}
        a_source = warpgroup.OperandSource.RMEM if source == "RS" else warpgroup.OperandSource.SMEM
        return sm90_utils.make_trivial_tiled_mma(
            spec.A.dtype,
            spec.B.dtype,
            mode[a_major],
            mode[b_major],
            acc_dtype,
            atom_layout_mnk=atom_layout_mnk,
            # `atom_layout_mnk[1]` partitions the logical/physical N tile across
            # warp-groups. The WGMMA atom's per-warpgroup N extent is therefore
            # the full matmul N divided by that atom-layout N factor.
            tiler_mn=(64, spec.N // atom_layout_mnk[1]),
            a_source=a_source,
        )
    elif arch.major in [8, 12]:  # SM8x and SM12x — warp-level MMA
        assert cta_group == 1, f"warp-level tiled_mma requires cta_group=1, got {cta_group}"
        assert source in ("SS", "RS"), f"warp-level tiled_mma source must be SS or RS, got {source}"
        mma_inst_mnk = (16, 8, 16)
        if spec.A.dtype.width == 16:  # fp16 / bf16
            op = warp.MmaF16BF16Op(spec.A.dtype, acc_dtype, mma_inst_mnk)
        else:
            raise NotImplementedError(
                "warp-level MMA backend doesn't yet support "
                f"a_dtype={spec.A.dtype} (width={spec.A.dtype.width})"
            )
        tC = cute.make_layout(atom_layout_mnk)
        if permutation_mnk is None:
            atom_m, atom_n, atom_k = atom_layout_mnk
            # The N dim is multiplied by 2 to leverage ldmatrix.x4 (matches the reference
            # blackwell_geforce/dense_gemm.py). A nested-layout permutation_n adds extra
            # modes to partition_A/B output, which breaks the standard mainloop slicing.
            permutation_mnk = (
                atom_m * mma_inst_mnk[0],
                atom_n * mma_inst_mnk[1] * 2,
                atom_k * mma_inst_mnk[2],
            )
        return cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)
    elif arch.major in [10, 11]:  # Blackwell tcgen05
        assert source in ("SS", "TS"), f"SM100 tiled_mma source must be SS or TS, got {source}"
        assert permutation_mnk is None, "SM100 tiled_mma does not accept permutation_mnk"
        cta_group_enum = tcgen05.CtaGroup.TWO if cta_group == 2 else tcgen05.CtaGroup.ONE
        m_full, n_full = spec.M, spec.N
        n_inst = n_full if n_full <= 256 else n_full // 2
        if source == "TS":
            # TMEM A is a freshly materialized physical operand, not a logical
            # view of existing SMEM/TMA storage. Ignore TensorSpec.transposed so
            # `S.T` can be stored as a row-major `(D, N)` TS-A tile.
            a_major = (
                OperandMajorMode.K if spec.A.layout == LayoutEnum.ROW_MAJOR else OperandMajorMode.MN
            )
        else:
            a_major = (
                OperandMajorMode.K
                if spec._storage_major(spec.A, is_A=True) == "K"
                else OperandMajorMode.MN
            )
        b_major = (
            OperandMajorMode.K
            if spec._storage_major(spec.B, is_A=False) == "K"
            else OperandMajorMode.MN
        )
        a_source = tcgen05.OperandSource.TMEM if source == "TS" else tcgen05.OperandSource.SMEM
        return sm100_utils.make_trivial_tiled_mma(
            spec.A.dtype,
            spec.B.dtype,
            a_major,
            b_major,
            acc_dtype,
            cta_group_enum,
            (m_full, n_inst),
            a_source,
        )
    raise NotImplementedError(
        f"make_tiled_mma_for_arch has no backend for {arch.name} (major={arch.major})."
    )
