# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import re
from typing import Optional, Type, Tuple, Callable

import cutlass
import cutlass.cute as cute

from cutlass import Int32, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.pipeline
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir


@dsl_user_op
def cvt_copy(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    retile: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    assert isinstance(src.iterator, cute.Pointer) and src.memspace == cute.AddressSpace.rmem
    if const_expr(src.element_type != dst.element_type):
        src_cvt = cute.make_fragment_like(src, dst.element_type)
        src_cvt.store(src.load().to(dst.element_type))
        src = src_cvt
    if const_expr(retile):
        src = tiled_copy.retile(src)
    cute.copy(tiled_copy, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


@dsl_user_op
def load_s2r(src: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    dst = cute.make_fragment_like(src, src.element_type, loc=loc, ip=ip)
    cute.autovec_copy(src, dst, loc=loc, ip=ip)
    return dst


@dsl_user_op
def load_s2r_retile(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst_shape: cute.Tensor | cute.Shape,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    # Will also accept dst_shape being a tensor, in which case we write into that tensor
    if const_expr(not isinstance(dst_shape, cute.Tensor)):
        dst = cute.make_rmem_tensor(dst_shape, src.element_type, loc=loc, ip=ip)
    else:
        dst = dst_shape
    cute.copy(tiled_copy, src, tiled_copy.retile(dst), loc=loc, ip=ip)
    return dst


@dsl_user_op
def get_copy_atom(
    dtype: Type[cutlass.Numeric], num_copy_elems: int, is_async: bool = False, *, loc=None, ip=None
) -> cute.CopyAtom:
    num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


@dsl_user_op
def copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    is_async: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    num_copy_elems = src.shape[0][0]
    copy_atom = get_copy_atom(src.element_type, num_copy_elems, is_async)
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


def tiled_copy_1d(
    dtype: Type[cutlass.Numeric], num_threads: int, num_copy_elems: int = 1, is_async: bool = False
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(num_copy_elems)
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    assert num_threads % threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


# def tiled_copy_2d(
#     dtype: Type[cutlass.Numeric], major_mode_size: int, num_threads: int, is_async: bool = False
# ) -> cute.TiledCopy:
#     num_copy_bits = math.gcd(major_mode_size, 128 // dtype.width) * dtype.width
#     copy_elems = num_copy_bits // dtype.width
#     copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
#     copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
#     gmem_threads_per_row = major_mode_size // copy_elems
#     assert num_threads % gmem_threads_per_row == 0
#     thr_layout = cute.make_ordered_layout(
#         (num_threads // gmem_threads_per_row, gmem_threads_per_row),
#         order=(1, 0),
#     )
#     val_layout = cute.make_layout((1, copy_elems))
#     return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def parse_swizzle_from_pointer(ptr: cute.Pointer) -> Tuple[int, int, int]:
    """Extract swizzle parameters from a pointer's swizzle_type.

    The swizzle_type string has the form '!cute.swizzle<"S<b,m,s>">' where
    b, m, s are the swizzle parameters (bits, base, shift).

    Returns:
        A cute.Swizzle object constructed from the extracted parameters

    Raises:
        ValueError: If the swizzle_type string cannot be parsed
    """
    # Ideally there should be a better API to get swizzle parameters, but we'll just parse
    # the string here.
    swizzle_str = str(ptr.type.swizzle_type)
    # Extract the inner part "S<b,m,s>"
    match = re.search(r"S<(\d+),(\d+),(\d+)>", swizzle_str)
    if match:
        b, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return b, m, s
    else:
        raise ValueError(f"Could not parse swizzle_type: {swizzle_str}")


def swizzle_int(ptr_int: Int32, b: int, m: int, s: int) -> Int32:
    bit_msk = (1 << b) - 1
    yyy_msk = bit_msk << (m + s)
    return ptr_int ^ ((ptr_int & yyy_msk) >> s)


def swizzle_ptr(ptr: cute.Pointer):
    b, m, s = parse_swizzle_from_pointer(ptr)
    ptr_int = swizzle_int(ptr.toint(), b, m, s)
    return cute.make_ptr(ptr.dtype, ptr_int, ptr.memspace, assumed_align=ptr.alignment)


def as_position_independent_swizzle_tensor(tensor: cute.Tensor) -> cute.Tensor:
    outer = tensor.layout
    width = tensor.element_type.width
    inner = cute.make_swizzle(*parse_swizzle_from_pointer(tensor.iterator))
    # Need to recast the swizzle from byte (e.g. <3, 4, 3> to element units (e.g. <3, 3, 3> for
    # for 16 bits and <3, 2, 3> for 32 bits)
    new_layout = cute.recast_layout(
        width, 8, cute.make_composed_layout(inner, 0, cute.recast_layout(8, width, outer))
    )
    # recast_ptr to remove the pointer swizzle
    return cute.make_tensor(cute.recast_ptr(tensor.iterator, dtype=tensor.element_type), new_layout)


def partition_D_position_independent(
    thr_copy: cute.core.ThrCopy, tensor: cute.Tensor
) -> cute.Tensor:
    return cute.make_tensor(
        swizzle_ptr(thr_copy.partition_D(tensor).iterator),
        thr_copy.partition_D(as_position_independent_swizzle_tensor(tensor)).layout,
    )


def partition_S_position_independent(
    thr_copy: cute.core.ThrCopy, tensor: cute.Tensor
) -> cute.Tensor:
    return cute.make_tensor(
        swizzle_ptr(thr_copy.partition_S(tensor).iterator),
        thr_copy.partition_S(as_position_independent_swizzle_tensor(tensor)).layout,
    )


@dsl_user_op
def sm90_get_smem_load_op(
    layout_c: cutlass.utils.LayoutEnum,
    elem_ty_c: Type[cutlass.Numeric],
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    """
    Selects the largest vectorized smem load atom available subject to constraint of gmem layout.

    Parameters:
    -----------
    layout_c : LayoutEnum
        The layout enum of the output tensor D.

    elem_ty_c : Type[Numeric]
        The element type for output tensor D.

    Returns:
    --------
    Either SmemLoadMatrix or SimtSyncCopy, based on the input parameters.
    """

    if not isinstance(elem_ty_c, cutlass.cutlass_dsl.NumericMeta):
        raise TypeError(f"elem_ty_c must be a Numeric, but got {elem_ty_c}")
    is_m_major = layout_c.is_m_major_c()
    if elem_ty_c.width == 16:
        return cute.make_copy_atom(warp.LdMatrix8x8x16bOp(is_m_major, 4), elem_ty_c, loc=loc, ip=ip)
    else:
        return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), elem_ty_c, loc=loc, ip=ip)


def get_smem_store_atom(
    arch: cutlass.Constexpr[int], element_type: Type[cute.Numeric], transpose: bool = False
) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=(2 if not transpose else 1) * element_type.width,
        )
    else:
        return cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
            element_type,
        )


def get_smem_load_atom(
    arch: cutlass.Constexpr[int], element_type: Type[cute.Numeric], transpose: bool = False
) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=(2 if not transpose else 1) * element_type.width,
        )
    else:
        return cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
            element_type,
        )


def get_smem_store_C(
    tiled_mma: cute.TiledMma,
    sC: cute.Tensor,
    tidx: Int32,
    arch: int,
    transpose: bool = False,
    position_independent=False,
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor]:
    dtype = sC.element_type
    copy_atom = get_smem_store_atom(arch, dtype, transpose)
    tiled_copy = cute.make_tiled_copy_C(copy_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(tidx)
    if const_expr(not position_independent):
        tRS_sC = thr_copy.partition_D(sC)
    else:
        tRS_sC = partition_D_position_independent(thr_copy, sC)

    def copy_fn(src: cute.Tensor, dst_idx: Optional[Int32] = None, **new_kwargs):
        dst_tensor = tRS_sC if const_expr(dst_idx is None) else tRS_sC[None, None, None, dst_idx]
        cvt_copy(tiled_copy, src, dst_tensor, retile=True, **new_kwargs)

    return copy_fn, thr_copy, tRS_sC


def get_smem_load_C(
    tiled_mma: cute.TiledMma,
    sC: cute.Tensor,
    tidx: Int32,
    arch: int,
    transpose: bool = False,
    position_independent=False,
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor]:
    dtype = sC.element_type
    copy_atom = get_smem_load_atom(arch, dtype, transpose)
    tiled_copy = cute.make_tiled_copy_C(copy_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(tidx)
    if const_expr(not position_independent):
        tSR_sC = thr_copy.partition_S(sC)
    else:
        tSR_sC = partition_S_position_independent(thr_copy, sC)
    copy_atom_RS = get_smem_store_atom(arch, dtype, transpose)
    thr_copy_RS = cute.make_tiled_copy_C(copy_atom_RS, tiled_mma).get_slice(tidx)
    tRS_shape = thr_copy_RS.partition_S(cute.make_identity_tensor(sC.shape[:2])).shape

    def copy_fn(src_idx: Optional[Int32] = None, **new_kwargs):
        src_tensor = tSR_sC if const_expr(src_idx is None) else tSR_sC[None, None, None, src_idx]
        return load_s2r_retile(tiled_copy, src_tensor, dst_shape=tRS_shape, **new_kwargs)

    return copy_fn, thr_copy, tSR_sC


def epilog_smem_copy_atom(
    tiled_mma: cute.TiledMma, epi_tile: cute.Shape, transpose: bool = False
) -> cute.TiledCopy:
    copy_atom_C = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose, num_matrices=4 if epi_tile[1] % 16 == 0 else 2),
        cutlass.Float16,  # this is just to get the right source layout
    )
    tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
    return tiled_copy_C_atom


def get_smem_store_epi(
    tiled_mma: cute.TiledMma,
    epi_tile: cute.Shape,
    sC: Optional[cute.Tensor],
    tidx: Int32,
    arch: int,
    transpose: bool = False,
    position_independent=False,
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor, cute.Tensor]:
    dtype = sC.element_type if const_expr(sC is not None) else cutlass.Float16
    tiled_copy_C_atom = epilog_smem_copy_atom(tiled_mma, epi_tile)
    copy_atom = get_smem_store_atom(arch, dtype, transpose)
    tiled_copy = cute.make_tiled_copy_S(copy_atom, tiled_copy_C_atom)
    thr_copy = tiled_copy.get_slice(tidx)
    tRS_sC = None
    if const_expr(sC is not None):
        if const_expr(not position_independent):
            tRS_sC = thr_copy.partition_D(sC)
        else:
            tRS_sC = partition_D_position_independent(thr_copy, sC)
    sC_shape = sC.shape[:2] if sC is not None else epi_tile
    # (R2S, R2S_M, R2S_N, PIPE_C)
    tRS_rC_shape = thr_copy.partition_S(cute.make_identity_tensor(sC_shape)).shape
    tRS_rC = cute.make_rmem_tensor(tRS_rC_shape, tiled_mma.op.acc_dtype)

    def copy_fn(src: cute.Tensor, dst_idx: Int32, **new_kwargs):
        cvt_copy(tiled_copy, src, tRS_sC[None, None, None, dst_idx], **new_kwargs)

    return copy_fn if const_expr(sC is not None) else None, thr_copy, tRS_sC, tRS_rC


def get_smem_store_A(
    tiled_mma: cute.TiledMma, sA: cute.Tensor, tidx: Int32, arch: int, position_independent=False
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor]:
    dtype = sA.element_type
    transpose = tiled_mma.op.a_major_mode == warpgroup.OperandMajorMode.MN
    copy_atom = get_smem_store_atom(arch, dtype, transpose)
    tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(tidx)
    if const_expr(not position_independent):
        tRS_sA = thr_copy.partition_D(sA)
    else:
        tRS_sA = partition_D_position_independent(thr_copy, sA)

    def copy_fn(src: cute.Tensor, dst_idx: Int32, **new_kwargs):
        cvt_copy(tiled_copy, src, tRS_sA[None, None, None, dst_idx], retile=True, **new_kwargs)

    return copy_fn, thr_copy, tRS_sA


def get_smem_load_A(
    tiled_mma: cute.TiledMma,
    sA: cute.Tensor,
    tidx: Int32,
    arch: int,
    with_dst_tensor: bool = False,
    position_independent=False,
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor]:
    dtype = sA.element_type
    transpose = tiled_mma.op.a_major_mode == warpgroup.OperandMajorMode.MN
    copy_atom = get_smem_load_atom(arch, dtype, transpose)
    tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(tidx)
    if const_expr(not position_independent):
        tSR_sA = thr_copy.partition_S(sA)
    else:
        tSR_sA = partition_S_position_independent(thr_copy, sA)
    tRS_shape = tiled_mma.partition_shape_A(sA.shape[:2])

    def copy_fn(src_idx: Int32, **new_kwargs):
        return load_s2r_retile(
            tiled_copy, tSR_sA[None, None, None, src_idx], dst_shape=tRS_shape, **new_kwargs
        )

    def copy_fn_w_dst_tensor(src_idx: Int32, dst: cute.Tensor, **new_kwargs):
        return load_s2r_retile(tiled_copy, tSR_sA[None, None, None, src_idx], dst, **new_kwargs)

    return copy_fn if not with_dst_tensor else copy_fn_w_dst_tensor, thr_copy, tSR_sA


@dsl_user_op
def cpasync_reduce_bulk_add_f32(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    store_bytes: int | Int32,
    *,
    loc=None,
    ip=None,
):
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    # cache_hint = cutlass.Int64(0x14F0000000000000)  # EVICT_LAST
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value()],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r",
        # [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value(), cache_hint.ir_value()],
        # "cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32 [$0], [$1], $2, $3;",
        # "l,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def get_tma_desc_addr(tma_atom: cute.CopyAtom, *, loc=None, ip=None) -> cute.Pointer:
    """
    Gets the address of the TMA descriptor embedded in a TMA Copy Atom.

    Extracts the constant memory address of the TMA descriptor for use with
    custom PTX instructions.

    :param tma_atom: TMA Copy Atom from make_tiled_tma_atom
    :return: Pointer to TMA descriptor in constant memory

    Example:
        >>> desc_ptr = get_tma_descriptor_address(tma_atom)
    """
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    tma_desc_ptr_type = ir.Type.parse(
        "!cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<128>>"
    )
    return _cute_nvgpu_ir.get_tma_desc_addr(tma_desc_ptr_type, exec_atom, loc=loc, ip=ip)


def cpasync_bulk_get_copy_fn(
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    single_stage: bool = False,
    **kwargs,
) -> Callable:
    group_rank_src = const_expr(cute.rank(src_tensor) - (1 if not single_stage else 0))
    group_rank_dst = const_expr(cute.rank(dst_tensor) - (1 if not single_stage else 0))
    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
    src = cute.group_modes(src_tensor, 0, group_rank_src)
    dst = cute.group_modes(dst_tensor, 0, group_rank_dst)

    def copy_bulk(src_idx, dst_idx, tma_bar_ptr: cute.Pointer, **new_kwargs):
        atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), src.element_type)
        with cute.arch.elect_one():
            cute.copy(
                atom,
                src[None, src_idx],
                dst[None, dst_idx],
                mbar_ptr=tma_bar_ptr,
                **new_kwargs,
                **kwargs,
            )

    def copy_bulk_single_stage(tma_bar_ptr: cute.Pointer, **new_kwargs):
        atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), src.element_type)
        with cute.arch.elect_one():
            cute.copy(atom, src, dst, mbar_ptr=tma_bar_ptr, **new_kwargs, **kwargs)

    return copy_bulk if const_expr(not single_stage) else copy_bulk_single_stage


def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    single_stage: bool = False,
    **kwargs,
) -> Callable:
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    group_rank_smem = const_expr(cute.rank(smem_tensor) - (1 if not single_stage else 0))
    group_rank_gmem = const_expr(cute.rank(gmem_tensor) - (1 if not single_stage else 0))
    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
    )
    if const_expr(filter_zeros):
        s = cute.filter_zeros(s)
        g = cute.filter_zeros(g)
    src, dst = (s, g) if src_is_smem else (g, s)

    def copy_tma(src_idx, dst_idx, **new_kwargs):
        cute.copy(atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs)

    def copy_tma_single_stage(**new_kwargs):
        cute.copy(atom, src, dst, **new_kwargs, **kwargs)

    return (copy_tma if const_expr(not single_stage) else copy_tma_single_stage), s, g


def tma_producer_copy_fn(copy: Callable, pipeline: cutlass.pipeline.PipelineAsync):
    def copy_fn(src_idx, producer_state: cutlass.pipeline.PipelineState, **new_kwargs):
        copy(
            src_idx=src_idx,
            dst_idx=producer_state.index,
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state),
            **new_kwargs,
        )

    return copy_fn


@cute.jit
def gather_m_get_copy_fn(
    thr_copy_A: cute.ThrCopy,
    mA: cute.Tensor,  # (whatever, K)
    sA: cute.Tensor,  # (tile_M, tile_N, STAGE)
    gsAIdx: cute.Tensor,  # (tile_M), either gmem or smem
    limit_m: Int32,
    limit_k: Int32,
) -> Callable:
    tile_shape_mk = (cute.size(sA, mode=[0]), cute.size(sA, mode=[1]))
    tAsA = thr_copy_A.partition_D(sA)
    # k-major
    assert tAsA.shape[2] == 1
    tAsA = cute.group_modes(cute.slice_(tAsA, (None, None, 0, None)), 0, 2)

    is_even_m_smem = tile_shape_mk[0] % thr_copy_A.tiler_mn[0].shape == 0
    if const_expr(not is_even_m_smem):
        limit_m = min(limit_m, tile_shape_mk[0])
    elems_per_load = cute.size(tAsA.shape[0][0])
    cA = cute.make_identity_tensor(tile_shape_mk)
    tAcA = thr_copy_A.partition_S(cA)
    t0AcA = thr_copy_A.get_slice(0).partition_S(cA)
    # Instead of comparing tAcA to limit_m, we instead compare t0AcA to limit_m - tAcA[0][0]
    # since we know that tAcA[m][0] = t0AcA[m][0] + tAcA[0][0].
    # This is so that when we do the comparison, t0AcA is known at compile time.
    limit_m = limit_m - tAcA[0][0]
    limit_k = limit_k - tAcA[0][1]
    # Read and cache indices for A
    rows_per_thread = const_expr(cute.size(tAcA.shape, mode=[1]))
    cols_per_thread = const_expr(cute.size(tAcA.shape, mode=[2]))
    tApA_m = cute.make_rmem_tensor(rows_per_thread, Boolean)
    for m in cutlass.range(rows_per_thread, unroll_full=True):
        tApA_m[m] = t0AcA[0, m, 0][0] < limit_m
    m_idx = cute.make_rmem_tensor(rows_per_thread, Int32)
    for m in cutlass.range(rows_per_thread, unroll_full=True):
        row_idx = tAcA[0, m, 0][0]
        if tApA_m[m]:
            m_idx[m] = gsAIdx[row_idx]
        else:
            m_idx[m] = 0  # It's ok to load row 0 in the case of OOB

    mA_k = cute.logical_divide(mA, (None, tile_shape_mk[1]))

    def copy_fn(src_idx, dst_idx, pred: bool = False):
        tApA_k = None
        if const_expr(pred):
            tApA_k = cute.make_rmem_tensor(cols_per_thread, Boolean)
            limit_k_cur = limit_k - src_idx * tile_shape_mk[1]
            for k in cutlass.range(cols_per_thread, unroll_full=True):
                tApA_k[k] = t0AcA[0, 0, k][1] < limit_k_cur
        mA_cur = mA_k[None, (None, src_idx)]
        for m in cutlass.range_constexpr(tAcA.shape[1]):
            # cute.tiled_divide(mA_cur[m_idx[m], None], (elems_per_load,)) would give shape
            # ((elems_per_load), thread_per_row)
            # But we actually want shape ((elems_per_load, 1), thread_per_row) to match tAsA
            # So we append 1s to the last dimension and then do tiled_divide, then slice.
            mA_row = cute.tiled_divide(
                cute.append_ones(mA_cur[m_idx[m], None], up_to_rank=2), (elems_per_load, 1)
            )[None, None, 0]
            if const_expr(is_even_m_smem) or tApA_m[m]:
                # There's only 1 load per row
                assert cute.size(tAcA.shape, mode=[2]) == 1
                ki = tAcA[0, 0, 0][1] // elems_per_load
                cute.copy(thr_copy_A, mA_row[None, ki], tAsA[(None, m), dst_idx], pred=tApA_k)

    return copy_fn


@cute.jit
def gather_k_get_copy_fn(
    thr_copy_A: cute.ThrCopy,
    mA: cute.Tensor,  # (tile_M, whatever)
    sA: cute.Tensor,  # (tile_M, tile_N, STAGE)
    gsAIdx: cute.Tensor,  # (tile_K, RestK), either gmem or smem
    limit_m: Int32,
    limit_k: Int32,
) -> Callable:
    gAIdx, sAIdx = None, None
    if const_expr(gsAIdx.memspace == cute.AddressSpace.gmem):
        gAIdx = gsAIdx
    else:
        assert gsAIdx.memspace == cute.AddressSpace.smem
        sAIdx = gsAIdx
    tile_shape_mk = (cute.size(sA, mode=[0]), cute.size(sA, mode=[1]))
    # (atom_v, CPY_M, 1, STAGE)
    tAsA = thr_copy_A.partition_D(sA)
    # m-major
    tAsA = cute.group_modes(tAsA, 0, 3)

    is_even_m_smem = tile_shape_mk[0] % thr_copy_A.tiler_mn[0].shape == 0
    if const_expr(not is_even_m_smem):
        limit_m = min(limit_m, tile_shape_mk[0])
    elems_per_load = cute.size(tAsA.shape[0][0])
    cA = cute.make_identity_tensor(tile_shape_mk)
    tAcA = thr_copy_A.partition_S(cA)
    t0AcA = thr_copy_A.get_slice(0).partition_S(cA)
    # Instead of comparing tAcA to limit_m, we instead compare t0AcA to limit_m - tAcA[0][0]
    # since we know that tAcA[m][0] = t0AcA[m][0] + tAcA[0][0].
    # This is so that when we do the comparison, t0AcA is known at compile time.
    limit_m = limit_m - tAcA[0][0]
    limit_k = limit_k - tAcA[0][1]
    # Read and cache indices for A
    rows_per_thread = const_expr(cute.size(tAcA.shape, mode=[1]))
    cols_per_thread = const_expr(cute.size(tAcA.shape, mode=[2]))
    tApA_m = cute.make_rmem_tensor(rows_per_thread, Boolean)
    for m in cutlass.range(rows_per_thread, unroll_full=True):
        tApA_m[m] = t0AcA[0, m, 0][0] < limit_m
    threads_per_col = const_expr(thr_copy_A.tiler_mn[0].shape // elems_per_load)
    # This is very convoluted but idk a better way
    # for tile_M=128, flat_divide gives (8, 16, K),
    # then logical_divide gives ((8, 1), (8, 2), K).
    tidx = thr_copy_A.thr_idx
    tAmA = cute.logical_divide(
        cute.flat_divide(mA, (elems_per_load,)), (elems_per_load, threads_per_col)
    )[None, (tidx % threads_per_col, None), None]  # ((8, 1), 2, K)

    def prefetch_from_gmem_fn(src_idx, pred: bool = False) -> Tuple[cute.Tensor, cute.Tensor]:
        # Prefetch mAIdx early, even before smem is free
        tApA_k = None
        if const_expr(pred):
            tApA_k = cute.make_rmem_tensor(cols_per_thread, Boolean)
            limit_k_cur = limit_k - src_idx * tile_shape_mk[1]
            for k in cutlass.range(cols_per_thread, unroll_full=True):
                tApA_k[k] = t0AcA[0, 0, k][1] < limit_k_cur
        gAIdx_cur = gAIdx[None, src_idx]
        k_idx = cute.make_rmem_tensor(cols_per_thread, Int32)
        for k in cutlass.range(cols_per_thread):
            col_idx = tAcA[0, 0, k][1]
            if const_expr(not pred):
                k_idx[k] = gAIdx_cur[col_idx]
            else:
                if tApA_k[k]:
                    k_idx[k] = gAIdx_cur[col_idx]
                else:
                    k_idx[k] = -1
        return k_idx, tApA_k

    def prefetch_from_smem_fn(
        a_prefetch_pipeline, src_idx, dst_idx, a_prefetch_consumer_state, pred: bool = False
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        tApA_k = None
        if const_expr(pred):
            tApA_k = cute.make_rmem_tensor(cols_per_thread, Boolean)
            limit_k_cur = limit_k - src_idx * tile_shape_mk[1]
            for k in cutlass.range(cols_per_thread, unroll_full=True):
                tApA_k[k] = t0AcA[0, 0, k][1] < limit_k_cur
        a_prefetch_pipeline.consumer_wait(a_prefetch_consumer_state)
        sAIdx_cur = sAIdx[None, dst_idx]
        k_idx = cute.make_rmem_tensor(cols_per_thread, Int32)
        for k in cutlass.range(cols_per_thread):
            col_idx = tAcA[0, 0, k][1]
            k_idx[k] = sAIdx_cur[col_idx]
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            a_prefetch_pipeline.consumer_release(a_prefetch_consumer_state)
        return k_idx, tApA_k

    def copy_fn(
        src_idx, dst_idx, k_idx_tApA_k: Tuple[cute.Tensor, cute.Tensor], pred: bool = False
    ):
        k_idx, tApA_k = k_idx_tApA_k
        tApA_k_pred = None
        if const_expr(pred):
            tApA_k_pred = cute.prepend_ones(tApA_k, up_to_rank=2)  # (1, cols_per_thread)
        for k in cutlass.range_constexpr(tAcA.shape[2]):
            # copy_A(tAmA[None, None, k_idx[k]], tAsA[(None, None, k), smem_idx], pred=cute.prepend_ones(tApA_m, up_to_rank=2))
            for m in cutlass.range_constexpr(tAcA.shape[1]):
                if tApA_m[m]:
                    cute.copy(
                        thr_copy_A,
                        tAmA[None, m, k_idx[k]],
                        tAsA[(None, m, k), dst_idx],
                        pred=None if const_expr(tApA_k_pred is None) else tApA_k_pred[None, k],
                    )

    return copy_fn, prefetch_from_gmem_fn if const_expr(
        gAIdx is not None
    ) else prefetch_from_smem_fn
