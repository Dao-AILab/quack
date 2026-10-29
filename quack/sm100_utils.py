# Copyright (c) 2025, Tri Dao.

from enum import Enum
from typing import Type, Union

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils_og
from cutlass.cutlass_dsl import Numeric, dsl_user_op
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05


class MmaMXF4NVF4Op(tcgen05.MmaMXF4NVF4Op):
    """tcgen05 kind::mxf4nvf4 MMA op with the scale config as a parameter.

    kind::mxf4nvf4 is ONE MMA atom covering both both-fp4 scale configs -
    scale_vec::2X (vec 32, e8m0; the mxfp4 instantiation, which PTX also
    spells ``kind::mxf4``) and scale_vec::4X (vec 16; nvfp4) - mirroring
    CUTLASS C++'s ``SM100_MMA_MXF4_SS<..., VS>``. The DSL class pins
    sf_vec_size = 16; this subclass restores the parameter. Both vec sizes
    build the identical MLIR atom type (it has no kind attribute - the
    backend derives the PTX spelling from the vec size), so this changes
    which Python op models the atom, not the lowered instruction.
    """

    def __init__(self, sf_dtype, sf_vec_size, instruction_shape, cta_group, a_src):
        tcgen05.BlockScaledMmaOp.__init__(
            self,
            cutlass.Float4E2M1FN,
            cutlass.Float4E2M1FN,
            cutlass.Float32,
            sf_dtype,
            sf_vec_size,
            instruction_shape,
            cta_group,
            a_src,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        self._verify()


class MmaKind(Enum):
    """tcgen05 MMA kinds (PTX ``kind::<value>``)."""

    F16 = "f16"
    TF32 = "tf32"
    F8F6F4 = "f8f6f4"
    MXF8F6F4 = "mxf8f6f4"
    MXF4NVF4 = "mxf4nvf4"

    @property
    def k_element_bits(self) -> int:
        """Bits per K element as the instruction reads it; mxf8f6f4 reads fp4/fp6
        operands from byte containers."""
        return {
            MmaKind.F16: 16,
            MmaKind.TF32: 32,
            MmaKind.F8F6F4: 8,
            MmaKind.MXF8F6F4: 8,
            MmaKind.MXF4NVF4: 4,
        }[self]

    @staticmethod
    def for_operands(a_dtype, b_dtype, blockscaled: bool) -> "MmaKind":
        """kind::mxf4nvf4 is one atom for both-packed-fp4 pairs (vec 32 e8m0 = mxfp4,
        vec 16 = nvfp4); other blockscaled pairs (mixed, fp6) are kind::mxf8f6f4. Mirrors
        quack.blockscaled.operand.mma_kind_for_pair; test_mma_kind_mirrors_kernel_kind
        pins the correspondence."""
        if blockscaled:
            both_fp4 = a_dtype is cutlass.Float4E2M1FN and b_dtype is cutlass.Float4E2M1FN
            return MmaKind.MXF4NVF4 if both_fp4 else MmaKind.MXF8F6F4
        return {8: MmaKind.F8F6F4, 16: MmaKind.F16, 32: MmaKind.TF32}[a_dtype.width]


def make_tiled_mma(
    kind: MmaKind,
    inst_mnk,
    a_dtype,
    b_dtype,
    a_major,
    b_major,
    cta_group,
    acc_dtype=cutlass.Float32,
    sf_dtype=None,
    sf_vec_size=None,
) -> cute.TiledMma:
    """tcgen05 tiled MMA for one instruction shape (M, N, K). K, in operand bits, selects
    the form: 256 on SM10x, 512 for sm_107's fp8/fp4 forms."""
    k_bits = inst_mnk[2] * kind.k_element_bits
    if k_bits == 512:
        import cutlass.utils.rubin_helpers as rubin_helpers  # cutlass-dsl 4.8+

        if kind is MmaKind.F8F6F4:
            return rubin_helpers.make_trivial_tiled_mma(
                a_dtype, b_dtype, a_major, b_major, acc_dtype, cta_group, inst_mnk
            )
        return rubin_helpers.make_blockscaled_trivial_tiled_mma(
            a_dtype, b_dtype, a_major, b_major, sf_dtype, sf_vec_size, cta_group, inst_mnk
        )
    assert k_bits == 256, f"no tcgen05 kind::{kind.value} form with {k_bits}-bit K"
    if kind is MmaKind.MXF4NVF4:
        # One atom for both scale configs; the DSL helper splits it by vec size.
        op = MmaMXF4NVF4Op(sf_dtype, sf_vec_size, inst_mnk, cta_group, tcgen05.OperandSource.SMEM)
        return cute.make_tiled_mma(cute.make_mma_atom(op))
    if kind is MmaKind.MXF8F6F4:
        return sm100_utils_og.make_blockscaled_trivial_tiled_mma(
            a_dtype, b_dtype, a_major, b_major, sf_dtype, sf_vec_size, cta_group, inst_mnk[:2]
        )
    return sm100_utils_og.make_trivial_tiled_mma(
        a_dtype, b_dtype, a_major, b_major, acc_dtype, cta_group, inst_mnk[:2]
    )


@dsl_user_op
def make_smem_layout_cpasync_a(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    a_dtype: Type[Numeric],
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """
    :param tiled_mma: The tiled MMA used to partition tensor A
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The MMA tile shape
    :type mma_tiler_mnk: cute.cute.Tile
    :param a_dtype: The element type for tensor A
    :type a_dtype: Type[Numeric]
    :param num_stages: The number of pipeline stages for tensor A
    :type num_stages: int

    :return: SMEM layout for tensor A
    :rtype: Union[cute.Layout, cute.ComposedLayout]
    """

    is_k_major = tiled_mma.op.a_major_mode == OperandMajorMode.K
    a_smem_shape = tiled_mma.partition_shape_A(
        cute.dice(mma_tiler_mnk, (1, None, 1), loc=loc, ip=ip)
    )
    a_smem_shape_mn_k = (
        cute.size(a_smem_shape[0][0], loc=loc, ip=ip) * a_smem_shape[1],
        cute.size(a_smem_shape[0][1], loc=loc, ip=ip) * a_smem_shape[2],
    )
    a_smem_layout_atom = sm100_utils_og.make_smem_layout_atom(
        sm100_utils_og.get_smem_layout_atom_ab(
            tiled_mma.op.a_major_mode,
            a_dtype,
            a_smem_shape_mn_k,
            loc=loc,
            ip=ip,
        ),
        a_dtype,
        loc=loc,
        ip=ip,
    )
    a_smem_layout_staged = cute.tile_to_shape(
        a_smem_layout_atom,
        cute.append(a_smem_shape_mn_k, num_stages, loc=loc, ip=ip),
        order=((1, 0, 2) if not is_k_major else (0, 1, 2)),
        loc=loc,
        ip=ip,
    )
    return a_smem_layout_staged


@dsl_user_op
def make_smem_layout_atom_tma_gather_a(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    a_dtype: Type[Numeric],
    gather_size: int = 4,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """SMEM load layout atom for A with TMA gather4."""
    is_k_major = tiled_mma.op.a_major_mode == OperandMajorMode.K
    a_smem_shape = tiled_mma.partition_shape_A(
        cute.dice(mma_tiler_mnk, (1, None, 1), loc=loc, ip=ip)
    )
    a_smem_shape_mn_k = (
        cute.size(a_smem_shape[0][0], loc=loc, ip=ip) * a_smem_shape[1],
        cute.size(a_smem_shape[0][1], loc=loc, ip=ip) * a_smem_shape[2],
    )
    # e,g., S<3, 4, 3> o  0 o (8, 64):(64, 1) for k_major
    # e,g., S<3, 4, 3> o  0 o (64, 8):(1, 64) for m_major
    a_smem_layout_atom = sm100_utils_og.make_smem_layout_atom(
        sm100_utils_og.get_smem_layout_atom_ab(
            tiled_mma.op.a_major_mode, a_dtype, a_smem_shape_mn_k, loc=loc, ip=ip
        ),
        a_dtype,
        loc=loc,
        ip=ip,
    )
    swizzle = a_smem_layout_atom.inner
    smem_layout = a_smem_layout_atom.outer
    if is_k_major:
        # Replace M-dim with 4 for gather4, keep original strides
        a_smem_layout_atom = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (gather_size, smem_layout.shape[1]), stride=smem_layout.stride, loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )
    else:
        # Replace K-dim with 4 for gather4, keep original strides
        a_smem_layout_atom = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (smem_layout.shape[0], gather_size), stride=smem_layout.stride, loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )
    return a_smem_layout_atom


@dsl_user_op
def make_smem_layout_tma_gather_a(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    a_dtype: Type[Numeric],
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """SMEM load layout for A with TMA gather4."""
    is_k_major = tiled_mma.op.a_major_mode == OperandMajorMode.K
    a_smem_shape = tiled_mma.partition_shape_A(
        cute.dice(mma_tiler_mnk, (1, None, 1), loc=loc, ip=ip)
    )
    a_smem_shape_mn_k = (
        cute.size(a_smem_shape[0][0], loc=loc, ip=ip) * a_smem_shape[1],
        cute.size(a_smem_shape[0][1], loc=loc, ip=ip) * a_smem_shape[2],
    )
    a_smem_layout_atom = make_smem_layout_atom_tma_gather_a(
        tiled_mma, mma_tiler_mnk, a_dtype, loc=loc, ip=ip
    )
    a_smem_layout_staged = cute.tile_to_shape(
        a_smem_layout_atom,
        cute.append(a_smem_shape_mn_k, num_stages, loc=loc, ip=ip),
        order=(1, 0, 2) if not is_k_major else (0, 1, 2),
        loc=loc,
        ip=ip,
    )
    return a_smem_layout_staged
