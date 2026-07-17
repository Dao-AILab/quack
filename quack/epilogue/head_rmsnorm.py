# Copyright (c) 2026, Tri Dao.
"""Per-head RMSNorm resources for composed GEMM epilogues."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

import quack.copy_utils as copy_utils
import quack.layout_utils as layout_utils
from quack.epi_ops import GroupedColStatsBase


class HeadRMSNormStats(GroupedColStatsBase):
    """Per-head RMSNorm statistics and scale resource: prepass sink (sums of
    the squared values the prepass fn returns, per (row, head), via the
    deterministic GroupedColStatsBase fold — no float atomics) + main-phase
    value port (fn_prepare: dense per-element rsqrt(mean + eps) * w
    multiplier). The group is the head: group_cols = weight length."""

    def __init__(self, name, eps=1e-6):
        super().__init__(name)
        self.eps = eps

    def config_key(self):
        return (self.eps,)

    def host_arg_key(self, value):
        from quack.cute_dsl_utils import torch2cute_dtype_map

        return (torch2cute_dtype_map[value.dtype], value.shape[0])

    def host_fake_arg(self, key, fctx):
        from quack.compile_utils import make_fake_tensor

        dtype, head_dim = key
        return make_fake_tensor(dtype, (head_dim,), leading_dim=0, divisibility=128 // dtype.width)

    def param_fields(self):
        return [(self.name, object, None)]

    def _group_cols(self, arg_tensor):
        return arg_tensor.shape[0]

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        assert gemm.arch in (90, 100, 120), (
            "head RMSNorm stats need the acc prepass (SM90/SM100/SM120)"
        )
        head_dim = const_expr(param.shape[0])
        tile_M, tile_N = ctx.tile_M, ctx.tile_N
        assert tile_N % head_dim == 0, "head RMSNorm needs whole heads per tile"
        stats = self.stats_begin(gemm, smem_tensor, ctx, head_dim)
        # Weight resource: broadcast w over rows, repeated per head along N.
        gW = cute.make_tensor(
            param.iterator,
            cute.make_layout(
                (tile_M, (head_dim, tile_N // head_dim)), stride=(0, (param.stride[0], 0))
            ),
        )
        tDgW = ctx.partition_for_epilogue_fn(gW)
        if const_expr(ctx.tiled_copy_t2r is not None):
            tDgW = ctx.tiled_copy_r2s.retile(tDgW)
        tDgW = cute.group_modes(tDgW, 3, cute.rank(tDgW))
        wbuf_raw = cute.make_rmem_tensor_like(tDgW[None, None, None, 0])
        copy_vec = const_expr(
            min(
                cute.max_common_vector(
                    cute.coalesce(cute.filter_zeros(tDgW[None, None, None, 0])),
                    cute.coalesce(cute.filter_zeros(wbuf_raw)),
                ),
                128 // param.element_type.width,
            )
        )
        if const_expr(param.element_type != Float32):
            wbuf_cvt = cute.make_rmem_tensor(wbuf_raw.shape, Float32)
        else:
            wbuf_cvt = wbuf_raw
        return [stats, tDgW, wbuf_raw, wbuf_cvt, copy_vec]

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        stats, tDgW, wbuf_raw, wbuf_cvt, copy_vec = state
        tDgW_cur = tDgW[None, None, None, epi_coord]
        tiler = cute.make_layout(copy_vec)
        cute.copy(
            copy_utils.get_copy_atom(wbuf_raw.element_type, copy_vec),
            cute.zipped_divide(cute.coalesce(cute.filter_zeros(tDgW_cur)), tiler),
            cute.zipped_divide(cute.coalesce(cute.filter_zeros(wbuf_raw)), tiler),
        )
        if const_expr(wbuf_cvt is not wbuf_raw):
            wbuf_cvt.store(wbuf_raw.load().to(Float32))
        # Stats state first: GroupedColStatsBase.fn_sink_flush reads state[0].
        return [self.stats_slice(stats, epi_coord), wbuf_cvt]

    # fn_sink_flush inherited: deterministic per-(row, head) sum of the
    # squared values the prepass fn returns.

    @cute.jit
    def fn_prepare(self, gemm, state, paired):
        """Main-phase value: dense per-element rsqrt(mean + eps) * w multiplier.
        stat_total sums the per-warp_n partials in fixed order (deterministic)."""
        stats, wfrag = state[0], state[1]
        c_cur, ref_layout, head_dim = stats[1], stats[2], stats[3]
        inv_d = const_expr(1.0 / head_dim)
        out = cute.make_rmem_tensor(wfrag.layout.shape, Float32)
        out_mn = layout_utils.convert_layout_zero_stride(out, ref_layout)
        w_mn = layout_utils.convert_layout_zero_stride(wfrag, ref_layout)
        c_mn = layout_utils.convert_layout_zero_stride(c_cur, ref_layout)
        for r in cutlass.range_constexpr(cute.size(out_mn, mode=[0])):
            coord = c_mn[r, 0]
            total = self.stat_total(stats, coord[0], coord[1] // head_dim)
            rstd = cute.math.rsqrt(total * inv_d + Float32(self.eps), fastmath=True)
            for c in cutlass.range(cute.size(out_mn, mode=[1]), unroll_full=True):
                out_mn[r, c] = rstd * w_mn[r, c]
        return out
