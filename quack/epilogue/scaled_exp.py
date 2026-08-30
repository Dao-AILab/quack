# Copyright (c) 2026, Tri Dao.
"""Scaled-exp epilogue: two-phase stable exp store with LSE partials.

Stores E = exp(acc) with a per-(row, n-tile) POWER-OF-TWO offset and emits
the partials to reconstruct it — the generic stable-softmax-numerator /
logsumexp building block (the linear-CE forward gemm1 is the motivating
consumer; the CE itself lives in the host glue).

Phase 1 (acc prepass): per-(row, n-tile) MAX of the raw accumulator via a
max-combine GroupedColStats fold (-inf identity: a TRUE max, so k < 0 on
all-negative tiles -- the tightest offset, no E underflow however negative
the logits get).

Phase 2: k = rne(max * log2e) per row feeds back as a value port; the main
fn stores E = exp2(acc*log2e - k) bf16 (a POWER-OF-TWO offset: downstream
pow2 strip scales are exact bf16 multiplies; RNE not ceil, so E <= sqrt(2)
rather than <= 1 -- integer-ness is what pow2 exactness needs, and bf16 has
range to spare), and emits per-(row, n-tile) partials sum_exp = sum(E) and
max_log2_out = k ITSELF, written straight from the finalized prepass
statistics by a GroupedColStatsOut companion (one store per (row, n-tile),
nothing in the per-element path). Consumers read k directly -- there is
nothing to re-derive and no rounding convention to match on the host side.
"""

import cutlass.cute as cute
from cutlass import Float32, const_expr

from quack.epilogue.ops import ColVecLoad, ColVecReduce, ColVecSelect, GroupedColStatsBase
from quack.epilogue.math import pexp2
from quack.epilogue.frontend import gemm_epilogue

LOG2E = 1.4426950408889634


class MaxLog2(GroupedColStatsBase):
    """Prepass sink: per-(row, whole-tile_N) MAX via the base max combine
    (-inf identity: a true max, negative for all-negative tiles), delivered
    in log2 units. Value port: k = max * log2e per
    row, rounded to an integer (RNE) when ``round_to_int`` -- integer k makes
    downstream 2^(k - k_r) scales exact powers of two (host re-derivation
    must round the same way: roundeven == libdevice rint == torch.round, all
    RNE). round_to_int=False skips the rounding for the tightest offset
    (E <= 1 instead of <= sqrt(2)) when pow2 exactness doesn't matter. Host
    arg: the group width as a plain int (== tile_N), e.g. max_log2=192; a
    1-D tensor works too (its length is the width, contents unused)."""

    combine = "max"

    def __init__(self, name, round_to_int=True):
        super().__init__(name)
        self.round_to_int = round_to_int

    def config_key(self):
        return (self.round_to_int,)

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        group_cols = const_expr(self._group_cols(param))
        # One k per (row, n-tile): the host-side re-derivation assumes it.
        assert ctx.tile_N == group_cols, "MaxLog2 groups the whole tile_N"
        return (self.stats_begin(gemm, smem_tensor, ctx, group_cols),)

    @cute.jit
    def stat_value(self, total, group_cols):
        k = total * Float32(LOG2E)
        if const_expr(self.round_to_int):
            k = cute.math.roundeven(k)
        return k


def _max_prepass(acc):
    return {"max_log2": acc}


_max_log2_op = MaxLog2("max_log2")


@gemm_epilogue(
    ops={"max_log2": _max_log2_op},
    prepass=_max_prepass,
    prepass_outs=("max_log2",),
    reduces={"sum_exp": ColVecReduce("sum_exp")},
    # max_log2_out: the finalized k, written straight from the prepass stats
    # smem (one store per (row, n-tile), zero per-element cost). Its host
    # validation enforces tile_N | N, which the pow2-offset protocol needs
    # anyway (sum_exp -- a plain add reduce -- would count the OOB lanes'
    # exp2(-k) on a ragged tile).
    extra_ops=(_max_log2_op.out("max_log2_out"),),
)
def scaled_exp_epi(acc, max_log2):
    # sum_exp folds e (POST-exp); max_log2 is the finalized k from the
    # phase-1 prepass (PRE-exp, log2 units), written out via max_log2_out.
    e = pexp2(acc * LOG2E - max_log2)
    return {"D": e, "sum_exp": e}


_max_log2_op_t = MaxLog2("max_log2")
_target_idx_op = ColVecLoad("target")


@gemm_epilogue(
    ops={"max_log2": _max_log2_op_t},
    prepass=_max_prepass,
    prepass_outs=("max_log2",),
    reduces={"sum_exp": ColVecReduce("sum_exp")},
    outs={"target_logit": ColVecSelect("target_logit", idx_op=_target_idx_op)},
    extra_ops=(_max_log2_op_t.out("max_log2_out"), _target_idx_op),
)
def scaled_exp_target_epi(acc, max_log2):
    """scaled_exp_epi + the target column's RAW logit gathered to an (m,)
    f32 colvec (``target`` int colvec in epi_args; see ColVecSelect). For
    the linear-CE glue this deletes the exact-Zy recompute (a D-length
    x/W-gather dot per row) — and the emitted Zy is the SAME accumulator
    value E was computed from, so the target-fix term is self-consistent
    rather than a reconstruction."""
    e = pexp2(acc * LOG2E - max_log2)
    return {"D": e, "sum_exp": e, "target_logit": acc}
