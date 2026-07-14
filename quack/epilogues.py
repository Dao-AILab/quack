# Copyright (c) 2026, Han Guo, Tri Dao.
"""Ready-to-use fused-GEMM epilogues, written with the @gemm_epilogue fn
contract (quack.gemm_epilogue) on top of the EpiOp library (quack.epi_ops).

Each entry is a plain function over the accumulator plus declared resources;
kernels are minted/cached per (fn source, op config, tensor metadata). Pass
tensors via ``mod.gemm(A, B, D, C, epi_args={...}, tile_M=..., ...)``.

Sections:
  * packed-polymorphic math helpers (pexp, pabs) — raw cute.math fns are not
    F2/Pair-aware; wrap transcendentals like these do.
  * reusable domain resources live in ``quack.epilogue``: rotary table loads
    in ``rotary`` and per-head RMSNorm statistics in ``head_rmsnorm``.
  * elementwise mods: linear/bias, residual, activation factories, RMS-fused
    (sq-sum reduce), amax (quantization stats), per-tile LSE partials, online
    LSE (stable), transformer-block forward (rms_partial -> rstd_swiglu) and
    the rmsnorm-backward link.
  * paired mods (gated / RoPE) and packed-C/D mods (dgated), via unpack/pack.

Numerics of every mod here are pinned by tests/test_gemm_epilogue.py against
torch references.
"""

from __future__ import annotations

import functools

import cutlass.cute as cute

from cutlass import Int32

from quack.activation import (
    act_fn_map,
    dact_fn_map,
    dgate_fn_map,
    dgelu_tanh_approx,
    dswiglu,
    gate_fn_map,
    gelu_tanh_approx,
    relu,
    relu_sq,
    swiglu,
)
from quack.epi_ops import (
    ColVecLoad,
    ColVecReduce,
    OnlineLSEReduce,
    RowVecLoad,
    RowVecReduce,
    Scalar,
)
from quack.epilogue.head_rmsnorm import HeadRMSNormStats
from quack.epilogue.rotary import rotary_cos_sin_load
from quack.gemm_epilogue import F2, gemm_epilogue, pack, unpack


@gemm_epilogue(outputs=("postact",))
def norm_gelu(acc, rstd, weight):
    x = acc * rstd * weight
    return {"D": x, "postact": gelu_tanh_approx(x)}


@gemm_epilogue()
def scaled_residual(acc, c, alpha):
    return {"D": acc * alpha + c}


@gemm_epilogue()
def linear_epi(acc, c, alpha, beta, bias_n, bias_m):
    """The default (linear) epilogue as a mod: alpha*acc + beta*C + rowvec + colvec."""
    return {"D": acc * alpha + c * beta + bias_n + bias_m}


def make_act_mod(fn):
    """Activation-mod factory — exercises closure-salted cache identity."""

    @gemm_epilogue(outputs=("postact",))
    def act_mod(acc):
        return {"D": acc, "postact": fn(acc)}

    return act_mod


relu_mod = make_act_mod(relu)


relu_sq_mod = make_act_mod(relu_sq)


@gemm_epilogue(outputs=("postact",))
def dgelu_mod(acc, c):
    """GemmDAct as a mod: c is the preact, acc is dout; D = dx, postact = act(c)."""
    dx, out = dgelu_tanh_approx(c, acc)
    return {"D": dx, "postact": out}


@gemm_epilogue(outputs=("premult",), reduces={"sqsum": ColVecReduce("sqsum", scaled=True)})
def rms_fused(acc, weight):
    """GemmSqReduce as a mod: sqsum accumulated on pre-scale acc, D = acc * weight.
    The scaled reduce returns the factors so the fold is one fma(acc, acc, sum)."""
    return {"D": acc * weight, "sqsum": (acc, acc), "premult": acc}


@gemm_epilogue(outputs=("postact",), mode="packed_cd_b16x2")
def dswiglu_mod(acc, c):
    """GemmDGated as a mod: acc = dout (per pair), c = packed (x, y) preact.
    Packing is declared by mode and validated against 16-bit C/D at twice GEMM-N."""
    x, y = unpack(c)
    dx, dy, out = dswiglu(x, y, acc)
    return {"D": pack(dx, dy), "postact": out}


@gemm_epilogue(
    outputs=("postact",),
    reduces={"dsum": ColVecReduce("dsum", scaled=True)},
    mode="packed_cd_b16x2",
)
def dswiglu_norm_mod(acc, c, rstd):
    """Full dgated: colvec-scaled dout, reduce on unscaled dout (folded as
    fma(out, dout, acc) via the scaled reduce), scaled postact."""
    x, y = unpack(c)
    dx, dy, out = dswiglu(x, y, acc * rstd)
    return {"D": pack(dx, dy), "postact": out * rstd, "dsum": (out, acc)}


@gemm_epilogue(
    outputs=("postact",),
    ops={"rstd": ColVecLoad("rstd")},
    reduces={"dsum": ColVecReduce("dsum")},
    mode="packed_cd_b16x2",
)
def dswiglu_rstd_preact_mod(acc, c, rstd):
    """Backward of s = swiglu(rstd * GU) — scale BEFORE activation, the
    llama/rstd_swiglu_preact_epi convention (dswiglu_norm_mod above inverts
    s = rstd * swiglu(GU), scale-after). acc = ds (dout), c = packed UNSCALED
    preact GU, rstd colvec. Emits D = dGU (grad wrt the unscaled preact —
    rstd folded here so both downstream bwd GEMMs are plain), the exact
    recomputed postact s (fc2-wgrad operand), and the rstd-gradient stat
    dsum = sum_pairs(dgs*g + dus*u) per tile (host: corr = -(rstd^3/d)*sum).
    dsum is a plain reduce: the stat is a two-product sum, so the scaled
    (val, scale) fma fold does not apply."""
    g, u = unpack(c)
    gs, us = g * rstd, u * rstd
    dgs, dus, s = dswiglu(gs, us, acc)
    return {"D": pack(dgs * rstd, dus * rstd), "postact": s, "dsum": dgs * g + dus * u}


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def swiglu_mod(acc):
    """GemmGated as a mod: the accumulator pairs over adjacent N because the
    postact buffer is half of GEMM-N."""
    gate, up = unpack(acc)
    return {"postact": swiglu(gate, up)}


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def norm_swiglu_mod(acc, rstd, bias):
    """Gated with rowvec bias (arrives paired) + colvec (scalar), D writeback.
    Pair arithmetic is lane-wise, so the affine part runs before unpacking."""
    v = acc * rstd + bias
    g, u = unpack(v)
    return {"postact": swiglu(g, u), "D": pack(g, u)}


@gemm_epilogue()
def identity_epi(acc):
    """Plain GEMM through the mod path. D's dtype is free, so this is the
    fp32-output wgrad building block: dW(n,k) = dout^T @ x runs as
    ``identity_epi.gemm(dout.mT, x.mT, dW_f32)`` — A arrives M'-major and B
    N'-major (both views, no copies) and the epilogue writes the f32
    accumulator directly (bf16 matmuls can't emit fp32 through torch)."""
    return {"D": acc}


@gemm_epilogue()
def residual_epi(acc, res):
    """Coda Residual: full-tile aux input added to the accumulator."""
    return {"D": acc + res}


@gemm_epilogue(mode="acc_pair")
def rope_epi(acc, table):
    """Coda RoPE: rotate adjacent-N pairs by an interleaved cos/sin table
    ((..., 2j) = cos, (..., 2j+1) = sin), congruent with the D tile."""
    x1, x2 = unpack(acc)
    cos, sin = unpack(table)
    return {"D": pack(x1 * cos - x2 * sin, x1 * sin + x2 * cos)}


def pexp(v):
    """Tuple-polymorphic exp, activation.py-style: raw cute.math fns are not
    F2-aware, so mods needing transcendentals wrap them like this (candidate
    for a shared quack epi-math module)."""
    if isinstance(v, tuple):
        return F2(*cute.arch.exp_packed_f32x2(v))
    return cute.math.exp(v, fastmath=True)


@gemm_epilogue(reduces={"sexp": ColVecReduce("sexp")})
def lse_partial_epi(acc, scale):
    """Coda LSE, per-tile flavor: sexp[m, tile] = sum_n exp(acc * scale);
    the host finalizes log(sum(partials)). NOTE: no online max — needs a
    max-combine reduce for large-logit stability (Coda's LSEReduce is online)."""
    return {"D": acc, "sexp": pexp(acc * scale)}


@gemm_epilogue(ops={"rstd": ColVecLoad("rstd")}, outs={"lse": OnlineLSEReduce("lse")})
def rstd_lse_epi(acc, rstd):
    """LM-head epilogue of a Coda-style transformer stack: apply the final
    boundary's deferred rstd (colvec), write the logits, and accumulate the
    stable online (max, sum) LSE partials (l, m, n_tiles). Host CE forward:
    lse = logsumexp(partials, -1); loss = lse - logits.gather(-1, target)."""
    v = acc * rstd
    return {"D": v, "lse": v}


def pabs(v):
    if isinstance(v, tuple):
        return F2(pabs(v[0]), pabs(v[1]))
    return cute.arch.fmax(v, -v)


@gemm_epilogue(reduces={"amax": ColVecReduce("amax", combine="max")})
def amax_epi(acc):
    """Per-tile column amax — the quantized-output (SFD) building block.
    |x| >= 0, so the zero OOB accumulator lanes of a ragged last tile can't
    corrupt the max (see VecReduce.combine note)."""
    return {"D": acc, "amax": pabs(acc)}


def _sq_prepass(acc):
    """Prepass fn: the statistic input, explicit (this replaces epirope's
    _prenorm_vec_ops replay registry — any pre-norm transform would be
    duplicated here in plain sight)."""
    return {"qk": acc * acc}


@gemm_epilogue(
    ops={"qk": HeadRMSNormStats("qk", eps=1e-6)},
    prepass=_sq_prepass,
    prepass_outs=("qk",),
)
def qknorm_epi(acc, qk):
    return {"D": acc * qk}


@gemm_epilogue(
    ops={"cs": rotary_cos_sin_load("cs"), "qk": HeadRMSNormStats("qk", eps=1e-6)},
    prepass=_sq_prepass,
    prepass_outs=("qk",),
    mode="acc_pair",
)
def qk_rope_epi(acc, cs, qk):
    """The full epirope composition: per-head RMSNorm (prepass stats) then
    rotary, in five lines of fn math. TMA table (see rotary_cos_sin_load):
    at the winning clustered-pingpong configs the LDG table's register cost
    is what tips this composition into spills."""
    x1, x2 = unpack(acc * qk)
    c, s = unpack(cs)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(
    ops={"cs": rotary_cos_sin_load("cs", tma=False), "qk": HeadRMSNormStats("qk", eps=1e-6)},
    prepass=_sq_prepass,
    prepass_outs=("qk",),
    mode="acc_pair",
)
def qk_rope_ldg_epi(acc, cs, qk):
    """qk_rope_epi on the gmem->rmem table op (see rope_table_ldg_epi)."""
    x1, x2 = unpack(acc * qk)
    c, s = unpack(cs)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(
    outputs=("resid_out",),
    ops={"weight": RowVecLoad("weight")},
    reduces={"sqsum": ColVecReduce("sqsum", scaled=True)},
)
def rms_partial_epi(acc, c, weight):
    """GEMM1 of a block: y = acc + residual(C); write the residual stream (aux),
    the weight-applied output (D — rstd deferred: row scaling commutes through
    the NEXT gemm), and the per-tile sq-sum partials for rstd finalization."""
    y = acc + c
    return {"D": y * weight, "resid_out": y, "sqsum": (y, y)}


@gemm_epilogue(outputs=("postact",), ops={"rstd": ColVecLoad("rstd")}, mode="acc_pair")
def rstd_swiglu_epi(acc, rstd):
    """GEMM2 of a block: apply the deferred rstd (colvec), then swiglu pairs."""
    g, u = unpack(acc * rstd)
    return {"postact": swiglu(g, u)}


@gemm_epilogue(outputs=("postact",), ops={"rstd": ColVecLoad("rstd")}, mode="acc_pair")
def rstd_swiglu_preact_epi(acc, rstd):
    """rstd_swiglu_epi that ALSO stores the UNSCALED preact (D = the raw
    accumulator pairs) — the training-mode gate_up epilogue: the saved preact
    is the exact operand dswiglu_rstd_preact_mod's backward needs (rstd is
    saved separately as an f32 colvec, so scaling stays exact in bwd)."""
    g, u = unpack(acc)
    gs, us = unpack(acc * rstd)
    return {"D": pack(g, u), "postact": swiglu(gs, us)}


@gemm_epilogue(reduces={"dots": ColVecReduce("dots", scaled=True)})
def rms_bwd_partial_epi(acc, y, rstd, w):
    """RMSNorm backward around a dgrad GEMM: acc = dz @ W2^T (= d(norm out)).
    t = acc*w, xhat = saved_prenorm(TileLoad) * rstd; write D = rstd*t and the
    per-tile partials of the correction dot mean(t * xhat)."""
    t = acc * w
    xhat = y * rstd
    return {"D": t * rstd, "dots": (t, xhat)}


# --- Deferred-rstd norm backward (transformer-block bwd, Coda-style) ----------
# Mid-stack boundary: the fwd applied gamma at the producing GEMM (a = h*gamma,
# rstd deferred into the NEXT gemm's epilogue), so the dgrad of a boundary GEMM
# receives acc = da carrying no rstd. dh = da*gamma + residual grad + the
# rstd-gradient correction colvec (host: corr = -(rstd^3/d) * dsum_finalized,
# with dsum from the fc2-dgrad epilogue; qkv side: -(rstd^2/d) * sum(dQKV*QKV)).
# dgamma = sum_m da*h lands as RowVecReduce partials (l, m_tiles, n), finalized
# by a host .sum over tiles.


@gemm_epilogue(
    ops={"w": RowVecLoad("w"), "corr": ColVecLoad("corr")},
    reduces={"dw": RowVecReduce("dw", scaled=True)},
)
def rms_bwd_apply_epi(acc, c, y, w, corr):
    """Norm-bwd apply around a dgrad GEMM (mid-stack boundary): acc = da,
    c = incoming residual-stream grad, y = saved residual h (TileLoad),
    w = gamma rowvec, corr colvec. D = dh_total; dw partials = dgamma."""
    return {"D": acc * w + c + y * corr, "dw": (acc, y)}


@gemm_epilogue(
    ops={"w": RowVecLoad("w"), "corr": ColVecLoad("corr")},
    reduces={"dw": RowVecReduce("dw", scaled=True)},
)
def rms_bwd_apply_last_epi(acc, y, w, corr):
    """rms_bwd_apply_epi without the residual-grad C: the FINAL boundary
    (lm_head dgrad) — the last residual h_L has no consumer besides the norm."""
    return {"D": acc * w + y * corr, "dw": (acc, y)}


@gemm_epilogue(
    ops={"w": RowVecLoad("w"), "rstd": ColVecLoad("rstd")},
    reduces={
        "dots": ColVecReduce("dots", scaled=True),
        "dw": RowVecReduce("dw", scaled=True),
    },
)
def rms_bwd_entry_epi(acc, c, y, rstd, w):
    """Conventional (non-deferred) rmsnorm bwd around a dgrad GEMM, for the
    ENTRY boundary where the fwd used a standalone full rmsnorm: acc = da with
    a = (y*rstd)*w. Emits D = rstd*t + c (t = acc*w) plus BOTH stats — the
    correction dot partials (finalized and applied by a terminal elementwise
    pass: dh = D - xhat*rstd*mean(dots)) and dgamma partials over xhat."""
    t = acc * w
    xhat = y * rstd
    return {"D": t * rstd + c, "dots": (t, xhat), "dw": (acc, xhat)}


# --- Variant mod factories ----------------------------------------------------
# The public gemm_act / gemm_dact / gemm_norm_act / gemm_sq_reduce wrappers
# ride these. The operand names (mAuxOut, mRowVecBroadcast, mColVecBroadcast,
# mColVecReduce, sr_seed) are the wire names shared with run_*_plan epi_values
# and concat_layout keys.
#
# The frontend derives a fn's operands from its SIGNATURE, and an absent
# operand must not exist in the signature at all (that is what compiles the
# term out), so each present-operand combination is its own generated fn:
# the factory assembles the source, exec-compiles it, and the fail-closed
# semantic fingerprint keys on the code object (per-combination bytecode +
# the referenced activation fn's own source through getclosurevars).
#
# Why generated source and not a closure over the flags: a closure would be
# ONE function whose parameter list carries every possible operand — but the
# signature IS the frontend's interface. A dead `bias_n` parameter would be
# inferred as a real RowVecLoad operand (smem, cp.async, a loop input the
# vectorizer must chew), and passing a neutral value instead (bias=0.0)
# reaches the kernel as a live runtime term, not a compiled-out one. Faking
# it via __signature__ doesn't help either: operand kinds, the trace-time
# call, and the co_varnames-based fingerprint all read the real code object.
# Generating the def is the only path where "operand absent" means "term
# does not exist in the kernel".


def _gen_epi_fn(fname, tag, params, body, ns):
    lines = [f"def {fname}({', '.join(['acc', *params])}):"]
    lines.append(f'    """generated epilogue [{tag}]"""')
    lines += [f"    {ln}" for ln in body]
    src = "\n".join(lines)
    code = compile(src, f"<quack-epilogue:{tag}>", "exec")
    ns = {**ns, "unpack": unpack, "pack": pack, "__name__": "quack.epilogues_generated"}
    exec(code, ns)
    return ns[fname]


def _vec_pins(params):
    pins = {}
    if "mRowVecBroadcast" in params:
        pins["mRowVecBroadcast"] = RowVecLoad("mRowVecBroadcast")
    if "mColVecBroadcast" in params:
        pins["mColVecBroadcast"] = ColVecLoad("mColVecBroadcast")
    return pins


_SR_OPS = (Scalar("sr_seed", dtype=Int32),)


@functools.lru_cache(maxsize=None)
def linear_act_mod(activation, *, gated, has_c, has_rowvec, has_colvec, sr=False):
    """gemm_act/gemm_gated as a mod: D = acc (+ C + rowvec + colvec), aux =
    act(D). Math order matches apply_linear_epilogue: C, rowvec, colvec."""
    fn_map = gate_fn_map if gated else act_fn_map
    act = fn_map[activation]
    params, body = [], []
    if has_c:
        params.append("c")
    if has_rowvec:
        params.append("mRowVecBroadcast")
    if has_colvec:
        params.append("mColVecBroadcast")
    expr = "acc"
    if has_c:
        body.append("x = acc + c")
        expr = "x"
    if has_rowvec:
        body.append(f"x = {expr} + mRowVecBroadcast")
        expr = "x"
    if has_colvec:
        body.append(f"x = {expr} + mColVecBroadcast")
        expr = "x"
    if gated:
        body.append(f"g, u = unpack({expr})")
        body.append('return {"D": pack(g, u), "mAuxOut": act(g, u)}')
    elif act is not None:
        body.append(f'return {{"D": {expr}, "mAuxOut": act({expr})}}')
    else:
        body.append(f'return {{"D": {expr}, "mAuxOut": {expr}}}')
    tag = f"act:{activation}:g{int(gated)}c{int(has_c)}r{int(has_rowvec)}v{int(has_colvec)}"
    fn = _gen_epi_fn("linear_act_epi", tag, params, body, {"act": act})
    return gemm_epilogue(
        outputs=("mAuxOut",),
        ops=_vec_pins(params),
        mode="acc_pair" if gated else None,
        extra_ops=_SR_OPS if sr else (),
    )(fn)


@functools.lru_cache(maxsize=None)
def norm_act_mod(activation, *, gated, has_c, has_rowvec, has_colvec, sr=False):
    """gemm_norm_act as a mod: x = (acc + C) * colvec * rowvec; D = x,
    aux = act(x). Scale order: colvec then rowvec."""
    fn_map = gate_fn_map if gated else act_fn_map
    act = fn_map[activation]
    params, body = [], []
    if has_c:
        params.append("c")
    if has_rowvec:
        params.append("mRowVecBroadcast")
    if has_colvec:
        params.append("mColVecBroadcast")
    expr = "acc"
    if has_c:
        body.append("x = acc + c")
        expr = "x"
    if has_colvec:
        body.append(f"x = {expr} * mColVecBroadcast")
        expr = "x"
    if has_rowvec:
        body.append(f"x = {expr} * mRowVecBroadcast")
        expr = "x"
    if gated:
        body.append(f"g, u = unpack({expr})")
        body.append('return {"D": pack(g, u), "mAuxOut": act(g, u)}')
    elif act is not None:
        body.append(f'return {{"D": {expr}, "mAuxOut": act({expr})}}')
    else:
        body.append(f'return {{"D": {expr}, "mAuxOut": {expr}}}')
    tag = f"norm_act:{activation}:g{int(gated)}c{int(has_c)}r{int(has_rowvec)}v{int(has_colvec)}"
    fn = _gen_epi_fn("norm_act_epi", tag, params, body, {"act": act})
    return gemm_epilogue(
        outputs=("mAuxOut",),
        ops=_vec_pins(params),
        mode="acc_pair" if gated else None,
        extra_ops=_SR_OPS if sr else (),
    )(fn)


@functools.lru_cache(maxsize=None)
def dact_mod(activation):
    """gemm_dact as a mod: c is the preact, acc is dout; D = dx, aux = act(c)."""
    dact = dact_fn_map[activation]
    if dact is None:
        body = ['return {"D": acc, "mAuxOut": c}']
    else:
        body = ["dx, out = dact(c, acc)", 'return {"D": dx, "mAuxOut": out}']
    fn = _gen_epi_fn("dact_epi", f"dact:{activation}", ["c"], body, {"dact": dact})
    return gemm_epilogue(outputs=("mAuxOut",))(fn)


@functools.lru_cache(maxsize=None)
def dgated_mod(activation, *, has_scale, has_reduce):
    """gemm_dgated as a mod: acc = dout (per pair), c = packed (x, y) preact.
    Reduce accumulates postact * unscaled dout; postact is scaled after."""
    dgate = dgate_fn_map[activation]
    params, body = ["c"], []
    if has_scale:
        params.append("mColVecBroadcast")
    body.append("x, y = unpack(c)")
    dout = "acc * mColVecBroadcast" if has_scale else "acc"
    body.append(f"dx, dy, out = dgate(x, y, {dout})")
    postact = "out * mColVecBroadcast" if has_scale else "out"
    if has_reduce:
        # Scaled reduce: return the factors so the fold is one
        # fma(out, dout, acc) per pair.
        body.append(
            f'return {{"D": pack(dx, dy), "mAuxOut": {postact}, "mColVecReduce": (out, acc)}}'
        )
    else:
        body.append(f'return {{"D": pack(dx, dy), "mAuxOut": {postact}}}')
    tag = f"dgated:{activation}:s{int(has_scale)}r{int(has_reduce)}"
    fn = _gen_epi_fn("dgated_epi", tag, params, body, {"dgate": dgate})
    return gemm_epilogue(
        outputs=("mAuxOut",),
        ops=_vec_pins(params),
        reduces={"mColVecReduce": ColVecReduce("mColVecReduce", scaled=True)}
        if has_reduce
        else None,
        mode="packed_cd_b16x2",
    )(fn)


@functools.lru_cache(maxsize=None)
def sq_reduce_mod(*, has_c, has_rowvec, has_aux):
    """gemm_sq_reduce as a mod: x = acc (+ C); reduce[m] += sum_n x^2 (before
    the rowvec scale); optional aux = x; D = x * rowvec."""
    params, body = [], []
    if has_c:
        params.append("c")
    if has_rowvec:
        params.append("mRowVecBroadcast")
    expr = "acc"
    if has_c:
        body.append("x = acc + c")
        expr = "x"
    d = f"{expr} * mRowVecBroadcast" if has_rowvec else expr
    # Scaled reduce: (x, x) folds as one fma(x, x, acc) per pair instead of
    # FMUL+FADD.
    ret = f'"D": {d}, "mColVecReduce": ({expr}, {expr})'
    if has_aux:
        ret += f', "mAuxOut": {expr}'
    body.append(f"return {{{ret}}}")
    tag = f"sq_reduce:c{int(has_c)}r{int(has_rowvec)}a{int(has_aux)}"
    fn = _gen_epi_fn("sq_reduce_epi", tag, params, body, {})
    # No host finalize: __call__ returns the RAW per-tile partials, and
    # gemm_rms fuses sum + rsqrt in rms_final_reduce (bitwise-equal to the
    # pre-object pipeline; a host torch.sum would reorder the fold).
    reduce = ColVecReduce("mColVecReduce", scaled=True)
    reduce.host_finalize = None
    return gemm_epilogue(
        outputs=("mAuxOut",) if has_aux else (),
        ops=_vec_pins(params),
        reduces={"mColVecReduce": reduce},
    )(fn)


# --- SigLIP / deferred-LayerNorm transformer block ----------------------------
# LayerNorm defers through a GEMM like RMSNorm with ONE extra colvec: for
#   z = LN(h) @ W^T + b,  LN(h) = (h - mu)*sig*gamma + beta, sig = rsqrt(var+eps)
#   z = sig ⊙ [(h*gamma) @ W^T] - (sig*mu) ⊙ (W@gamma) + (W@beta + b)
# The producing GEMM writes a = h*gamma plus (sum, sqsum) partials (host
# finalizes mu, sig); the consuming GEMM applies two colvecs (s = sig,
# t = sig*mu) and two HOST-PRECOMPUTED per-layer rowvecs (wg = W@gamma,
# wb = W@beta + linear_bias). Backward: the boundary's stat reduces (r1 =
# sum_j dz*z, column-sums dwb/dwg for dbeta and the rank-1 dW corrections)
# live in the epilogue that PRODUCES dz — one GEMM earlier, so the correction
# colvecs arrive finalized at the next dgrad's apply epilogue (no self-stat).
# Full algebra + host finalizers: tests/test_siglip_pipeline.py; procedure:
# skills/codaify.


@gemm_epilogue(
    ops={
        "s": ColVecLoad("s"),
        "t": ColVecLoad("t"),
        "wg": RowVecLoad("wg"),
        "wb": RowVecLoad("wb"),
    }
)
def ln_affine_epi(acc, s, t, wg, wb):
    """Consuming GEMM of a deferred-LayerNorm boundary (SigLIP qkv proj):
    D = s*acc - t*wg + wb."""
    return {"D": (acc * s - t * wg) + wb}


@gemm_epilogue(
    outputs=("postact",),
    ops={
        "s": ColVecLoad("s"),
        "t": ColVecLoad("t"),
        "wg": RowVecLoad("wg"),
        "wb": RowVecLoad("wb"),
    },
)
def ln_affine_gelu_epi(acc, s, t, wg, wb):
    """SigLIP fc1: deferred-LN affine then tanh-GELU; D = the bf16 preact
    (saved for dgelu in bwd), postact feeds fc2."""
    z = (acc * s - t * wg) + wb
    return {"D": z, "postact": gelu_tanh_approx(z)}


@gemm_epilogue(
    outputs=("resid_out",),
    ops={"bias": RowVecLoad("bias"), "weight": RowVecLoad("weight")},
    reduces={
        "hsum": ColVecReduce("hsum"),
        "sqsum": ColVecReduce("sqsum", scaled=True),
    },
)
def ln_partial_epi(acc, c, bias, weight):
    """Producing GEMM of a deferred-LayerNorm boundary (SigLIP out_proj/fc2):
    y = acc + linear bias + residual(C); write the residual stream, the
    gamma-applied next-GEMM input, and BOTH LayerNorm stats' per-tile partials
    (host: mu = sum/d, sig = rsqrt(sqsum/d - mu^2 + eps), t = sig*mu)."""
    y = acc + bias + c
    return {"D": y * weight, "resid_out": y, "hsum": y, "sqsum": (y, y)}


@gemm_epilogue(
    ops={"s": ColVecLoad("s"), "t": ColVecLoad("t")},
    reduces={
        "r1": ColVecReduce("r1", scaled=True),
        "dwb": RowVecReduce("dwb"),
        "dwg": RowVecReduce("dwg", scaled=True),
    },
)
def dgelu_ln_stats_epi(acc, c, s, t):
    """SigLIP fc2-dgrad: acc = dpostact, c = saved bf16 preact z. dz through
    tanh-GELU'; D = s*dz (the boundary's sig folded so fc1-dgrad AND
    fc1-wgrad stay plain). Boundary-bwd stats: r1 = per-row sum dz*z (feeds
    dsig with the wb-dot host correction), dwb = column-sum dz (dbeta +
    linear-bias grad + rank-1 dW term), dwg = column-sum t*dz (dgamma's
    W-path + rank-1 dW term)."""
    dz, _ = dgelu_tanh_approx(c, acc)
    return {"D": dz * s, "r1": (dz, c), "dwb": dz, "dwg": (dz, t)}


@gemm_epilogue(
    ops={
        "w": RowVecLoad("w"),
        "corr_mul": ColVecLoad("corr_mul"),
        "corr_add": ColVecLoad("corr_add"),
    },
    reduces={
        "dw": RowVecReduce("dw", scaled=True),
        "dbias": RowVecReduce("dbias"),
    },
)
def ln_bwd_apply_epi(acc, c, y, w, corr_mul, corr_add):
    """LayerNorm-bwd apply around a dgrad GEMM (SigLIP fc1-dgrad/qkv-dgrad):
    acc = da (grad of a = h*gamma), c = residual grad, y = saved residual h.
    D = dh_total = acc*w + c + y*corr_mul + corr_add, with the two correction
    colvecs host-finalized from the PREVIOUS bwd GEMM's boundary stats:
      dsig = (r1 - sum_j dz*wb)/sig ; dmu = -sig * sum_j dz*wg
      corr_mul = -dsig*sig^3/d ; corr_add = (dmu + dsig*sig^3*mu)/d.
    dw partials = dgamma's a-path (sum_m da*h); dbias = column-sums of the
    OUTPUT dh (the producing linear's bias grad lives downstream)."""
    dh = acc * w + c + y * corr_mul + corr_add
    return {"D": dh, "dw": (acc, y), "dbias": dh}


@gemm_epilogue(reduces={"dwb": RowVecReduce("dwb")})
def dgelu_dbias_epi(acc, c):
    """Standard-organization fc2-dgrad (SigLIP): tanh-GELU' from the saved
    preact c, UNSCALED dz out (the boundary's LN-bwd runs as a separate
    narrow fused kernel), plus the fc1-bias grad column-sums. Contrast
    dgelu_ln_stats_epi (deferred org: sig-folded D + 3 boundary-stat sinks)."""
    dz, _ = dgelu_tanh_approx(c, acc)
    return {"D": dz, "dwb": dz}
