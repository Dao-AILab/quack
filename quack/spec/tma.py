# Copyright (c) 2025-2026, Tri Dao.

"""TMA helpers that sit below TensorSpec.

TensorSpec-owned single-CTA TMA uses the same flat CTA-value map as CuTe's
generic tile helper. SM100 2-CTA dense loads need an explicit tcgen05 map
because a peer CTA owns instruction panels, not a contiguous half tile.
"""

from typing import Any, Optional, Tuple, Type, Union, cast

import cutlass
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
import cutlass.cute as cute
import cutlass.cute.atom as cute_atom
import cutlass.cute.core as cute_core
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.cpasync.copy import (
    CopyBulkTensorTileG2SNonExecTrait,
    CopyBulkTensorTileG2SMulticastNonExecTrait,
    CopyBulkTensorTileS2GNonExecTrait,
    CopyReduceBulkTensorTileS2GNonExecTrait,
)
from cutlass.cute.nvgpu.cpasync.helpers import TmaInfo
from cutlass.cute.typing import NumericMeta
from cutlass.utils import LayoutEnum

from quack.spec import mma as spec_mma
from quack.spec import smem as spec_smem


@dsl_user_op
def _make_tiled_tma_atom_from_cta_v_map(
    op: Union[
        cpasync.CopyBulkTensorTileG2SOp,
        cpasync.CopyBulkTensorTileG2SMulticastOp,
        cpasync.CopyBulkTensorTileS2GOp,
        cpasync.CopyReduceBulkTensorTileS2GOp,
    ],
    gmem_tensor: cute.Tensor,
    smem_layout,
    cta_v_map,
    num_multicast: int = 1,
    *,
    internal_type: Optional[Type[cutlass.Numeric]] = None,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> TmaInfo:
    """Build a tiled TMA load atom from an explicit CTA-value map."""
    stored_smem_layout = smem_layout
    smem_rank = cute.rank(smem_layout)
    map_rank = cute.rank(cta_v_map)
    if smem_rank == map_rank + 1:
        smem_layout = cute.select(smem_layout, mode=list(range(map_rank)))

    smem_for_ir = smem_layout
    if isinstance(smem_for_ir, cute_core._ComposedLayout):
        smem_for_ir = smem_for_ir.value

    tma_format = None
    if internal_type is not None:
        itype: Any = internal_type
        if not isinstance(internal_type, NumericMeta):
            raise TypeError(f"internal_type must be a Numeric, but got {internal_type}")

        use_unpack = (
            itype.width == 8
            and isinstance(gmem_tensor.element_type, NumericMeta)
            and gmem_tensor.element_type.width < 8
        )
        internal_mlir_type = gmem_tensor.element_type.mlir_type if use_unpack else itype.mlir_type
        tma_format = _cute_nvgpu_ir.TmaDataFormat(
            _cute_nvgpu_ir.get_default_tma_format(internal_mlir_type, use_unpack)
        )

    if isinstance(op, cpasync.CopyBulkTensorTileG2SOp):
        if num_multicast != 1:
            raise ValueError(
                f"non-multicast G2S copies require num_multicast=1, got {num_multicast}"
            )
        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
            cast(Any, gmem_tensor).value,
            smem_for_ir,
            cta_v_map,
            op._to_ir(),
            num_multicast=num_multicast,
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        return TmaInfo(
            cute_atom.CopyAtom(op, CopyBulkTensorTileG2SNonExecTrait(res[0])),
            res[1],
            stored_smem_layout,
        )

    if isinstance(op, cpasync.CopyBulkTensorTileG2SMulticastOp):
        if num_multicast < 1:
            raise ValueError(
                f"multicast G2S copies require num_multicast >= 1, got {num_multicast}"
            )
        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
            cast(Any, gmem_tensor).value,
            smem_for_ir,
            cta_v_map,
            op._to_ir(),
            num_multicast=num_multicast,
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        return TmaInfo(
            cute_atom.CopyAtom(op, CopyBulkTensorTileG2SMulticastNonExecTrait(res[0])),
            res[1],
            stored_smem_layout,
        )

    if isinstance(op, cpasync.CopyBulkTensorTileS2GOp):
        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_store(
            cast(Any, gmem_tensor).value,
            smem_for_ir,
            cta_v_map,
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        return TmaInfo(
            cute_atom.CopyAtom(op, CopyBulkTensorTileS2GNonExecTrait(res[0])),
            res[1],
            stored_smem_layout,
        )

    if isinstance(op, cpasync.CopyReduceBulkTensorTileS2GOp):
        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_reduce(
            cast(Any, gmem_tensor).value,
            smem_for_ir,
            cta_v_map,
            op._to_ir(),
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        return TmaInfo(
            cute_atom.CopyAtom(op, CopyReduceBulkTensorTileS2GNonExecTrait(res[0])),
            res[1],
            stored_smem_layout,
        )

    raise ValueError(f"expects a bulk tensor (TMA) Copy Op, but got {op}")


def _sm100_dense_tma_cta_v_map(
    shape: Tuple[int, int],
    smem_layout,
):
    rows, cols = shape
    tile_shape = spec_smem._sm100_smem_tile_shape(smem_layout)
    atom_shape, rest_l, rest_k = tile_shape
    atom_l_shape, atom_k = atom_shape
    atom_l = cute.size(atom_l_shape)
    assert rows == atom_l * rest_l, (
        f"SMEM layout leading shape {atom_l}*{rest_l} does not match operand rows {rows}"
    )
    assert cols == atom_k * rest_k, (
        f"SMEM layout K shape {atom_k}*{rest_k} does not match operand cols {cols}"
    )
    leading_panel_stride = rows * cute.E(0) if rest_l > 1 else 0

    return cute.make_layout(
        ((atom_l, atom_k), rest_l, rest_k),
        stride=(
            (cute.E(0), cute.E(1)),
            leading_panel_stride,
            atom_k * cute.E(1),
        ),
    )


def _sm100_dense_tma_cta_v_map_from_shape(
    dtype: Type[cutlass.Numeric],
    shape: Tuple[int, int],
    cta_group: int = 1,
    mma_inst_k: Optional[int] = None,
):
    """CTA-value map for a dense SM100 operand tile.

    `shape` is the local per-CTA tile. For cta_group=2 the TMA map must address
    the full two-CTA tile in tcgen05 instruction panels, not a contiguous local
    half. Example: local `(256, 64)` for full leading size 512 maps two 128-row
    panels separated by a 256-row stride.
    """
    rows, cols = shape
    k_inst = spec_mma.resolve_mma_inst_k(dtype, mma_inst_k)
    assert cols % k_inst == 0, f"TMA cols must be divisible by {k_inst}, got {cols}"
    atom_rows, rest_rows = spec_mma.operand_leading_atom(rows, cta_group)
    leading_panel_stride = rows * cute.E(0) if rest_rows > 1 else 0
    return cute.make_layout(
        ((atom_rows, k_inst), rest_rows, cols // k_inst),
        stride=(
            (cute.E(0), cute.E(1)),
            leading_panel_stride,
            k_inst * cute.E(1),
        ),
    )


def _sm100_dense_tma_flat_cta_v_map(
    shape: Tuple[int, int],
    cta_group: int = 1,
):
    """CTA-value map for a flat role-free SM100 TMA storage view.

    `shape` is the local per-CTA tile. For a 2-CTA full leading tile of 512,
    each CTA owns two 128-wide instruction panels: CTA0 maps rows
    0..127 and 256..383, while CTA1 maps the complementary panels. The gap
    belongs in the CTA-value map; the SMEM descriptor can stay the normal flat
    `(local_leading, k)` TensorSpec storage layout.
    """
    rows, cols = shape
    atom_rows, rest_rows = spec_mma.operand_leading_atom(rows, cta_group)
    leading_panel_stride = rows * cute.E(0) if rest_rows > 1 else 0
    return cute.coalesce(
        cute.make_layout(
            ((atom_rows, rest_rows), cols),
            stride=((cute.E(0), leading_panel_stride), cute.E(1)),
        ),
        target_profile=(1, 1),
    )


def _sm100_dense_tma_layout_and_cta_v_map(
    dtype: Type[cutlass.Numeric],
    shape: Tuple[int, int],
    stage: int,
    layout: LayoutEnum,
    cta_group: int = 1,
    mma_inst_k: Optional[int] = None,
):
    smem_layout = (
        spec_smem.make_smem_layout_kmajor(dtype, shape, stage, cta_group, mma_inst_k)
        if layout == LayoutEnum.ROW_MAJOR
        else spec_smem.make_smem_layout_mnmajor(dtype, shape, stage, cta_group, mma_inst_k)
    )
    if cute.rank(smem_layout) == 4:
        smem_layout = cute.slice_(smem_layout, (None, None, None, 0))
    cta_v_map = _sm100_dense_tma_cta_v_map_from_shape(dtype, shape, cta_group, mma_inst_k)
    return smem_layout, cta_v_map


def _make_tma_atom_operand_from_op(
    op: Union[cpasync.CopyBulkTensorTileG2SOp, cpasync.CopyBulkTensorTileG2SMulticastOp],
    dtype: Type[cutlass.Numeric],
    shape: Tuple[int, int],
    stage: int,
    layout: LayoutEnum,
    gmem_tensor: cute.Tensor,
    *,
    num_multicast: int = 1,
    cta_group: int = 1,
    mma_inst_k: Optional[int] = None,
    internal_type: Optional[Type[cutlass.Numeric]] = None,
):
    """Role-free SM100 TMA load atom for a swizzled dense operand tile.

    Returns a `TmaInfo` whose atom and tensor are equivalent to CUTLASS'
    role-aware `make_tiled_tma_atom_A/B` for the same operand distribution.
    """
    smem_layout, cta_v_map = _sm100_dense_tma_layout_and_cta_v_map(
        dtype, shape, stage, layout, cta_group, mma_inst_k
    )
    return _make_tiled_tma_atom_from_cta_v_map(
        op,
        gmem_tensor,
        smem_layout,
        cta_v_map,
        num_multicast,
        internal_type=internal_type,
    )


def _make_tma_atom_operand(
    dtype: Type[cutlass.Numeric],
    shape: Tuple[int, int],
    stage: int,
    layout: LayoutEnum,
    gmem_tensor: cute.Tensor,
    *,
    num_multicast: int = 1,
    cta_group: int = 1,
    mma_inst_k: Optional[int] = None,
    internal_type: Optional[Type[cutlass.Numeric]] = None,
):
    cg = tcgen05.CtaGroup.TWO if cta_group == 2 else tcgen05.CtaGroup.ONE
    op = (
        cpasync.CopyBulkTensorTileG2SMulticastOp(cg)
        if num_multicast > 1
        else cpasync.CopyBulkTensorTileG2SOp(cg)
    )
    return _make_tma_atom_operand_from_op(
        op,
        dtype,
        shape,
        stage,
        layout,
        gmem_tensor,
        num_multicast=num_multicast,
        cta_group=cta_group,
        mma_inst_k=mma_inst_k,
        internal_type=internal_type,
    )
