# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import math
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute

from cutlass import Float32, Int32, const_expr
from cutlass.base_dsl.typing import Vector
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def load_scalar_or_pointer(x, dtype=Float32):
    if const_expr(isinstance(x, cute.Pointer)):
        return dtype(cute.make_tensor(x, cute.make_layout(1))[0])
    else:
        return x


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    dsmem_ptr = cute.arch.map_dsmem_ptr(smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip)
    return Int32(dsmem_ptr.toint(loc=loc, ip=ip))


@dsl_user_op
def store_shared_remote(
    val: float | Float32 | Int32 | cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip)
    remote_mbar_ptr_i32 = set_block_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip)
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, Int32, cutlass.Int64)), "val must be Float32, Int32, or Int64"
    suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
    cute.arch.inline_ptx(
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} "
        "[{$r0}], {$r1}, [{$r2}];",
        read_only_args=[remote_smem_ptr_i32, val, remote_mbar_ptr_i32],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_shared_remote_x4(
    val0: Float32 | Int32,
    val1: Float32 | Int32,
    val2: Float32 | Int32,
    val3: Float32 | Int32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip)
    remote_mbar_ptr_i32 = set_block_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip)
    assert isinstance(val0, (Float32, Int32)), "val must be Float32, or Int32"
    dtype = Float32 if isinstance(val0, Float32) else Int32
    suffix = {Float32: "f32", Int32: "s32"}[dtype]
    cute.arch.inline_ptx(
        "{\n\t"
        f".reg .v4 .{suffix} abcd;\n\t"
        f"mov.{suffix} abcd.x, {{$r2}};\n\t"
        f"mov.{suffix} abcd.y, {{$r3}};\n\t"
        f"mov.{suffix} abcd.z, {{$r4}};\n\t"
        f"mov.{suffix} abcd.w, {{$r5}};\n\t"
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.{suffix} "
        "[{$r0}], abcd, [{$r1}];\n\t"
        "}\n",
        read_only_args=[
            remote_smem_ptr_i32,
            remote_mbar_ptr_i32,
            dtype(val0),
            dtype(val1),
            dtype(val2),
            dtype(val3),
        ],
        loc=loc,
        ip=ip,
    )


@cute.jit
def fill_oob(tXsX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric) -> None:
    """Fill out-of-bounds values in shared memory tensor.

    Args:
        tXsX: Shared memory tensor to fill
        tXpX: Predicate tensor indicating valid elements
        fill_value: Value to fill OOB locations with
    """
    tXrX_fill = cute.make_rmem_tensor_like(tXsX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])


# ---------------------------------------------------------------------------
# General-purpose DSL store / vector helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def make_vector(elem_type, *values, loc=None, ip=None):
    """Build an MLIR vector <N x elem_type> from N scalar DSL values.

    Example: make_vector(cutlass.Uint32, v0, v1) -> <2 x i32> MLIR vector
    """
    scalars = tuple(elem_type(v) for v in values)
    return Vector.from_elements(scalars, elem_type, loc=loc, ip=ip).ir_value()


@dsl_user_op
def f32x2_to_i64(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int64:
    vec = Vector.from_elements((Float32(a), Float32(b)), Float32, loc=loc, ip=ip)
    return vec.bitcast(cutlass.Int64, loc=loc, ip=ip).to_elements(loc=loc, ip=ip)[0]


@dsl_user_op
def i64_to_f32x2(c: cutlass.Int64, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    vec = Vector.from_elements((cutlass.Int64(c),), cutlass.Int64, loc=loc, ip=ip)
    res0, res1 = vec.bitcast(Float32, loc=loc, ip=ip).to_elements(loc=loc, ip=ip)
    return res0, res1


@cute.jit
def warp_prefix_sum(val: Int32, lane: Optional[Int32] = None) -> Int32:
    if const_expr(lane is None):
        lane = cute.arch.lane_idx()
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
    return val


@dsl_user_op
def domain_offset_aligned(
    coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor:
    assert isinstance(tensor.iterator, cute.Pointer)
    # We assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        elem_pointer(tensor, coord).toint(),
        tensor.memspace,
        assumed_align=tensor.iterator.alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)
