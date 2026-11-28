# Copyright (c) 2026, Tri Dao.
"""``@a_transform``: author an A-operand transform as a plain Python function,
mirroring ``@gemm_epilogue`` (the fn is the COMPOSITION site; the kernel-side
plumbing — canonical s2r load or blob TMA, the interleaved produce/WGMMA
schedule, fences and commit groups — is written once and never by the fn).

Two families, one decorator:

* VALUE transforms (default): unpacked 16-bit A, canonically ldmatrix-loaded;
  the fn is called per lane per ``vec_size`` fragment elements as a TensorSSA
  vector in the MMA dtype and returns the transformed vector::

      @a_transform(vec_size=2)
      def halve_a(x):
          return x * 0.5

  The vector is FRAGMENT-SLOT-ORDERED, not k-contiguous (a lane's 8 elements
  per k16 block are 2 adjacent k x 2 rows x 2 k-halves); chunks are
  pair-aligned so packed 16-bit math vectorizes. ``vec_size`` in {2, 4, 8} and
  is capped at one k16 block: the schedule (produce(b+1) overlapping WGMMA(b))
  belongs to the framework, never the fn. Compile-time constants may be closed
  over — they are part of the semantic key. ``consts=callable`` is called once
  per kernel (hoisted — LUTs, packed constants); its result is the fn's LAST
  parameter.

  Runtime operands are declared per fn parameter with ``args={param: kind}``,
  a kind taxonomy over the transform's (M, K) index space — the mainloop
  mirror of EpiOps' operand kinds over (M, N). Each kind owns its indexing,
  delivery and register staging; the fn just receives a same-length TensorSSA
  vector of values aligned with x. Kinds (kinds.ARG_KINDS):

  * the strip family — one MMA-dtype value per (m-group, k-group) at 2-D
    granularity, delivered via the aux A-side TMA slot (per-stage smem under
    the AB mbarrier): ``"colvec_ktile"`` (per (row, k-tile)),
    ``"colvec_k64"`` / ``"colvec_k32"`` / ``"colvec_k16"`` (per (row, 64 /
    32 / 16 elements) — dense blockscaled-SF granularities), ``"kvec_m64"``
    (per (m64 row block, k-element) — the LCE dw strip; resolve
    coarser-than-64 M granularity host-side, e.g. a vocab-tile row per m64
    block). E.g. the linear-CE dx pow2 rescale::

        @a_transform(vec_size=8, args={"u": "colvec_ktile"})
        def dx_scale(x, u):
            return x * u

    The host passes A as the bundle
    ``host.transform_a_operand(mod, A, {"u": strip}, tile_m, tile_k)`` with
    ``strip`` a contiguous (ceil(K / tile_K) * g, M) tensor in A's dtype
    (one row per k-group; ragged K padded to whole k-tiles).

  There is deliberately NO k-invariant ``colvec`` kind: a per-row scale
  commutes through the GEMM — ``(u ⊙ A) @ B = u ⊙ (A @ B)`` — so it belongs
  in the EPILOGUE as an fp32 colvec (exact, measured free), and per-row
  additive terms commute via the rank-1 ``z · colsum(B)`` correction; only a
  fn NONLINEAR in x would need per-row mainloop values.

* DROPOUT (``dropout_a(p)``): not a fn — a dedicated mask-only transform
  (philox keep-mask ANDed onto the fragment at ~3 SASS per 2 elements; see
  :class:`~quack.operand_transform.transform.TransformADropout`). The
  (2,) int64 [seed, offset] tensor rides the same bundle
  (``host.transform_a_operand(dropout_a(p), A, {"seed": t}, tile_m)``, or
  ``TransformAOperand(A, t)`` at the direct-compile layer). The mask of
  (m, k) is a pure function of (m, k, seed, offset) — any kernel
  regenerates it — and is split-k invariant. Mask-only: fold 1/(1-p) into
  the epilogue.

* PACKED decodes (``packed=PackedInput(...)``): the fn IS the decode — the
  :meth:`~quack.operand_transform.formats.DecodeFormat.decode_k16` body,
  ``fn(xw, sfw, b, consts) -> 4 packed regs`` — and the ``PackedInput``
  carries the geometry (w8 / tile_k) and the host bundle (prepare /
  quantize_reference / dequant_reference) that must stay consistent with it.
  The result IS a ``DecodeFormat`` (:class:`FnFormat`), so it slots into
  everything the class-based formats can (TransformAW4, the gemm_w4 wrapper,
  the roundtrip test fixture). ``sfw`` is always None for now (scale-factor strips need the
  aux A-side operand, not ported yet — sf_words must be 0).

Every transform handle is a mod (:class:`~quack.operand_transform.mod.
TransformModBase`): a value fn (:class:`ATransformMod`), a decode format,
dropout. A mod is a factory ``gemm -> TransformA`` — pass it straight to
``GemmSm90(transform_a=mod)``. Its ``semantic_digest`` fail-closed
fingerprints the fn (source + every capture, via the gemm_epilogue keyer), so
mods compose with the jit-cache machinery.

Torch surfaces: ``EpiMod.__call__``/``plan`` take ``transform_a=mod`` plus
``transform_operands={name: tensor}`` for runtime-operand mods (raw values —
the TransformAOperand bundle is built there from the resolved config; only
``mod.gemm`` takes pre-built bundles). Under torch.compile the same call
records the single ``quack::gemm_epi`` custom op: mods cross by
``semantic_digest`` (registered at construction; bind the mod to a module
global for graphs that must survive a fresh process) and operand tensors
ride the op's input list. See quack.gemm_runtime.torch_op.

Staying async-compilable: the rules are the ones in quack.epilogue.frontend
("Staying async-compilable"). The one that bites transforms: a
``DecodeFormat`` subclass has a ``@cute.jit`` ``decode_k16``, so it must be a
module-level class. (For a packed FN that class is :class:`FnFormat`, which
calls the fn as plain Python.)
"""

import inspect
from dataclasses import dataclass
from typing import Callable, Optional


import cutlass.cute as cute

from quack.operand_transform.formats import DecodeFormat
from quack.gemm_runtime.identity import (
    function_semantic_key as _function_semantic_key,
    semantic_value_key as _semantic_value_key,
)
from quack.operand_transform.kinds import ARG_KINDS
from quack.operand_transform.mod import TransformModBase
from quack.operand_transform.transform import TransformAValue

__all__ = ["a_transform", "ATransformMod", "FnFormat", "PackedInput", "dropout_a"]


@dataclass(frozen=True)
class PackedInput:
    """Storage geometry + host bundle for a packed-storage fn transform.
    Field meanings match :class:`~quack.operand_transform.formats.DecodeFormat`
    (w8: 32 B/thread raw words; tile_k: the k-tile the repack is built
    around). The host callables must stay consistent with the fn — the
    roundtrip test fixture is what pins that."""

    name: str
    w8: bool = False
    tile_k: int = 64
    make_consts: Optional[Callable] = None
    prepare: Optional[Callable] = None
    quantize_reference: Optional[Callable] = None
    dequant_reference: Optional[Callable] = None


class FnFormat(DecodeFormat):
    """A packed decode authored as a fn (``@a_transform(packed=...)``): the
    fn is the k16 decode, the PackedInput the geometry and the host side.
    Module-level, so an async-compile worker gets the class, and the source
    of its ``@cute.jit`` decode, by import; the instance (the fn, the spec)
    travels by value."""

    def __init__(self, fn, spec: PackedInput):
        self.fn = fn
        self.name, self.w8, self.tile_k = spec.name, spec.w8, spec.tile_k
        # make_consts shapes device code, so it is public state and part of
        # the identity; the other host callables don't reach the kernel
        # (repack consistency is pinned by the roundtrip fixture).
        self.consts_fn = spec.make_consts
        self._spec = spec
        super().__init__()

    def make_consts(self):
        return self.consts_fn() if self.consts_fn is not None else None

    @cute.jit
    def decode_k16(self, xw, sfw, b, consts):
        return self.fn(xw, sfw, b, consts)

    def quantize_reference(self, w):
        return self._spec.quantize_reference(w)

    def dequant_reference(self, q, sf):
        return self._spec.dequant_reference(q, sf)

    def prepare(self, q, sf):
        return self._spec.prepare(q, sf)


class ATransformMod(TransformModBase):
    """A fn-authored VALUE transform; callable as the ``transform_a=``
    factory. See module docstring for the fn contract."""

    def __init__(self, fn, vec_size, consts=None, regs=None, args=None):
        self.fn = fn
        self.name = getattr(fn, "__name__", "a_transform")
        vec_size = 2 if vec_size is None else vec_size
        assert vec_size in (2, 4, 8), "vec_size must be 2, 4 or 8 (one k16 block max)"
        self.vec_size = vec_size
        self.consts = consts
        self.regs = regs
        self.args = _normalize_args(fn, args)
        self._identify(
            (
                "a_transform",
                _semantic_value_key(self.fn, set()),
                self.vec_size,
                _function_semantic_key(self.consts) if self.consts is not None else None,
                self.regs,
                self.args,
            )
        )

    def __call__(self, gemm):
        return TransformAValue(gemm, self)


def _normalize_args(fn, args) -> tuple:
    """Validate an ``args`` declaration ({fn param name: kind name}) against
    the fn signature and the kind registry; normalize to ((name, kind), ...)
    in fn-parameter order (the staging order in the kernel)."""
    if not args:
        return ()
    fn_kinds = {k for k, kind in ARG_KINDS.items() if kind.fn_facing}
    unknown = set(args.values()) - fn_kinds
    assert not unknown, f"unknown operand kind(s) {unknown}; have {fn_kinds}"
    params = list(inspect.signature(fn).parameters)
    missing = set(args) - set(params[1:])
    assert not missing, f"args {missing} are not parameters of {fn.__name__} (after x)"
    return tuple((name, args[name]) for name in params[1:] if name in args)


class DropoutAMod(TransformModBase):
    """``transform_a=`` handle for dropout on A (see TransformADropout): the
    keep-mask of element (m, k) is a pure function of (m, k, seed, offset),
    reproducible by any kernel and invariant under split-k. MASK-ONLY — fold
    1/(1-p) into the epilogue. The (2,) int64 [seed, offset] CUDA tensor is
    the runtime operand, riding the TransformAOperand bundle's sf slot
    (``args`` declares it so the generic host plumbing — mod.gemm unpack,
    trace fakes, plan keys — treats it like any strip operand)."""

    args = (("seed", "seed_i64x2"),)

    def __init__(self, p: float, rounds: int = 7):
        assert 0.0 <= p < 1.0, "drop probability must be in [0, 1)"
        # keep iff byte >= threshold: P(drop) = threshold / 256, exactly
        self.p = p
        self.threshold = min(int(round(p * 256)), 255)
        self.rounds = rounds
        self.name = f"dropout_a_t{self.threshold}"
        # trailing int: scheme version
        self._identify(("dropout_a", self.threshold, rounds, 1))

    def __call__(self, gemm):
        from quack.operand_transform.transform import TransformADropout

        return TransformADropout(gemm, self)


def dropout_a(p: float, rounds: int = 7) -> DropoutAMod:
    """A ``transform_a=`` handle for dropout on A with drop probability
    ``round(p * 256) / 256`` (mask-only; scale via the epilogue)."""
    return DropoutAMod(p, rounds)


def a_transform(
    vec_size: Optional[int] = None,
    packed=None,
    consts: Optional[Callable] = None,
    regs: Optional[tuple] = None,
    args: Optional[dict] = None,
):
    """Decorator: turn a plain fn into an A-operand transform mod. See the
    module docstring for the two fn contracts.

    ``consts=callable`` (value fns): called once per kernel (hoisted — LUTs,
    packed constants); its result is the fn's LAST parameter.
    ``args={param: kind}`` (value fns): runtime operands — each named fn
    parameter (between x and consts) receives its per-element values as a
    TensorSSA vector, staged by its kind (kinds.ARG_KINDS;
    the (M, K) mirror of EpiOps' operand kinds). A crosses as the
    ``host.transform_a_operand(mod, A, {param: tensor}, tile_m)`` bundle.
    ``regs=(load, mma)`` overrides the register budget split (multiples of 8,
    see setmaxnreg constraints in TransformAW4)."""

    def wrap(fn):
        if packed is not None:
            assert vec_size in (None, 8), "packed fn transforms decode whole k16 blocks"
            assert consts is None, "packed fns take consts via PackedInput.make_consts"
            assert not args and regs is None, "runtime operands and regs are value-fn only"
            return FnFormat(fn, packed)
        return ATransformMod(fn, vec_size=vec_size, consts=consts, regs=regs, args=args)

    return wrap
