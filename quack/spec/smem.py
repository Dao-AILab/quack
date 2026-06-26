# Copyright (c) 2025-2026, Tri Dao.

from typing import Type, Tuple, Union, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup, tcgen05, OperandMajorMode
from cutlass.cutlass_dsl import Numeric, dsl_user_op
from cutlass import const_expr
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.utils import LayoutEnum

from quack.spec import mma as spec_mma


def make_smem_layout_kmajor(
    dtype: Type[Numeric],
    shape: Tuple[int, int],
    stages: int,
    cta_group: int = 1,
    mma_inst_k: Optional[int] = None,
):
    """SM100 operand SMEM layout where K is the fast storage axis.

    `shape = (mn, k)`, per CTA. This matches operands whose storage-contiguous
    axis is K, independent of whether the operand is used as A or B.
    """
    rows, cols = shape
    k_inst = spec_mma.resolve_mma_inst_k(dtype, mma_inst_k)
    assert cols % k_inst == 0, f"K-major cols must be divisible by {k_inst}, got {cols}"
    atom_rows, rest_rows = spec_mma.operand_leading_atom(rows, cta_group)
    atom = sm100_utils.make_smem_layout_atom(
        sm100_utils.get_smem_layout_atom_ab(OperandMajorMode.K, dtype, shape),
        dtype,
    )
    return cute.make_composed_layout(
        atom.inner,
        0,
        cute.make_layout(
            ((atom_rows, k_inst), rest_rows, cols // k_inst, stages),
            stride=(
                (cols, 1),
                atom_rows * cols if rest_rows > 1 else 0,
                k_inst,
                rows * cols,
            ),
        ),
    )


def make_smem_layout_mnmajor(
    dtype: Type[Numeric],
    shape: Tuple[int, int],
    stages: int,
    cta_group: int = 1,
    mma_inst_k: Optional[int] = None,
):
    """SM100 operand SMEM layout where MN is the fast storage axis.

    `shape = (mn, k)`, per CTA. The logical MN axis is storage-contiguous, while
    K is the slow axis in the logical operand view. The constructed nested
    layout still factors the K extent with `mma_inst_k`, matching CUTLASS'
    role-aware A/B helpers.
    """
    rows, cols = shape
    k_inst = spec_mma.resolve_mma_inst_k(dtype, mma_inst_k)
    assert cols % k_inst == 0, f"MN-major cols must be divisible by {k_inst}, got {cols}"
    atom_rows, rest_rows = spec_mma.operand_leading_atom(rows, cta_group)
    atom = sm100_utils.make_smem_layout_atom(
        sm100_utils.get_smem_layout_atom_ab(OperandMajorMode.MN, dtype, shape),
        dtype,
    )
    mn_contiguous = atom.outer.shape[0]
    assert atom_rows % mn_contiguous == 0, (
        f"MN-major atom rows {atom_rows} must be divisible by contiguous row atom {mn_contiguous}"
    )
    if atom_rows == mn_contiguous:
        return cute.make_composed_layout(
            atom.inner,
            0,
            cute.make_layout(
                ((atom_rows, k_inst), rest_rows, cols // k_inst, stages),
                stride=(
                    (1, atom_rows),
                    atom_rows * cols if rest_rows > 1 else 0,
                    k_inst * atom_rows,
                    rows * cols,
                ),
            ),
        )
    return cute.make_composed_layout(
        atom.inner,
        0,
        cute.make_layout(
            (
                ((mn_contiguous, atom_rows // mn_contiguous), k_inst),
                rest_rows,
                cols // k_inst,
                stages,
            ),
            stride=(
                ((1, mn_contiguous * cols), mn_contiguous),
                atom_rows * cols if rest_rows > 1 else 0,
                k_inst * mn_contiguous,
                rows * cols,
            ),
        ),
    )


def _sm100_smem_tile_shape(smem_layout):
    layout = (
        cute.select(smem_layout, mode=[0, 1, 2]) if cute.rank(smem_layout) == 4 else smem_layout
    )
    return layout.outer.shape if hasattr(layout, "outer") else layout.shape


@dsl_user_op
def make_smem_layout(
    dtype: Type[Numeric],
    layout: LayoutEnum,
    tile: cute.Tile,
    num_stages: Optional[int] = None,
    major_mode_size: Optional[int] = None,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    shape = cute.product_each(cute.shape(tile, loc=loc, ip=ip), loc=loc, ip=ip)
    if const_expr(major_mode_size is None):
        major_mode_size = shape[1] if layout.is_n_major_c() else shape[0]
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    if arch.major not in [10, 11]:
        smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(layout, dtype, major_mode_size),
            dtype,
        )
    else:  # Blackwell
        major_mode = OperandMajorMode.MN if layout.is_m_major_c() else OperandMajorMode.K
        smem_layout_atom = tcgen05.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(major_mode, dtype, tile),
            dtype,
        )
    order = (1, 0, 2) if const_expr(layout.is_m_major_c()) else (0, 1, 2)
    smem_layout_staged = cute.tile_to_shape(
        smem_layout_atom,
        cute.append(shape, num_stages) if const_expr(num_stages is not None) else shape,
        order=order if const_expr(num_stages is not None) else order[:2],
    )
    # TensorSpec exposes a role-free storage/allocation view. Coalesce removes
    # swizzle-atom factoring from the outer modes while preserving addressing,
    # so callers see a canonical (M, N[, stage]) layout; MMA-specific nested
    # operand views are constructed separately in MatmulSpec.
    return cute.coalesce(
        smem_layout_staged,
        target_profile=(1, 1, 1) if const_expr(num_stages is not None) else (1, 1),
        loc=loc,
        ip=ip,
    )
