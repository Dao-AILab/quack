# Copyright (c) 2026, Han Guo, Tri Dao.
"""FlexAttention-style epilogue authoring: a plain Python function over the
accumulator, lowered onto the EpiOp machinery. The design in one page:

Layer map
---------
* quack.epi_ops       — EpiOps: per-tensor RESOURCE lifecycle (smem, TMA,
                        fragments, flushes: begin/begin_loop/end_loop/end),
                        host schema hooks (host_arg_key/host_fake_arg/
                        host_call_arg), and VALUE PORTS (below).
* quack.gemm_host     — generic host layer: one jit-cached compile fn (the
                        kernel CLASS is a key argument — classes pickle by
                        module+qualname, so disk keys are stable and async
                        workers import the right module by unpickling), plan
                        cache/build/run driven entirely by the op schema.
* this module         — the fn contract + kernel-class minting. A mod is ONE
                        Python function called per element at trace time; the
                        minted class is a standard ComposableEpiMixin subclass,
                        so hand-written mixins remain first-class and anything
                        the fn form can't express (e.g. symmetric's scheduler)
                        stays a mixin.
* quack.epilogues     — the library of ready mods and reusable ops.

The fn contract
---------------
``fn(acc, **operands) -> {"D": ..., <outputs/sinks>...}``, called once per
accumulator element (or pair). Values are Float32 scalars on pre-SM100 and
:class:`F2` packed pairs on SM100 (same scalar-or-f32x2 contract as
quack.activation, whose functions compose directly; F2 arithmetic lowers to
packed f32x2 intrinsics). ``"D"`` is optional — omit it to leave the raw
accumulator. Loop shapes mirror the hand-written mixins exactly; no tracer
guesses about vectorization.

* OPERANDS are the fn's parameter names after ``acc``. Kinds are inferred
  from tensor metadata at plan time: (l, n) row broadcast / (l, m) col
  broadcast ((total_m,) rank-1 under varlen) / (l, m, n) full-tile load /
  python scalar or 1-element tensor -> Scalar. ``c`` is reserved for the GEMM
  C operand. ``ops={name: EpiOp(...)}`` pins a name to an explicit op when
  inference is ambiguous (m == n) or the op is custom.
* PAIRING (gated / dgated) is also inferred — aux buffer at half GEMM-N pairs
  the accumulator over adjacent N columns; 16-bit C at twice GEMM-N packs C
  and D two lanes per f32 element — and expressed in the body via
  ``unpack``/``pack`` (see :class:`Pair`, whose arithmetic is lane-wise).
  ``paired=("acc",)`` declares pairing when tensors give no signal (RoPE:
  full-width D, no aux).
* SINKS: ``outputs=(names,)`` declares aux tile stores (each TileStore owns
  its own dtype/rounding, so multiple mixed-dtype outputs compose); ``reduces={name:
  ColVecReduce/RowVecReduce(name, combine="add"|"max")}`` declares reduce
  outputs (fn returns the per-element value; buffers are per-CTA-tile
  partials). A reduce of a PRODUCT should use ``scaled=True`` and return the
  two factors — ``{"sqsum": (x, x)}`` — so the fold is one fused
  ``fma(val, scale, acc)``: the product is never rounded on its own (bitwise
  parity with folding the product directly, one FFMA instead of FMUL+FADD).
  ``outs={name: sink_op}`` is the general form for any sink op
  (e.g. OnlineLSEReduce's coupled (max, sum) accumulator).
* PREPASS: ``prepass=fn2, prepass_outs=(names,)`` runs fn2 over the RAW
  accumulator before any store (driver flag epi_needs_acc_prepass; needs a
  re-readable accumulator — SM90 registers, SM100 tmem with no_release;
  pingpong is fine, its per-warpgroup epilogues are strictly exclusive so the
  stats smem is only temporally shared), feeding prepass sink ops; the same
  op then serves the main fn as a value operand carrying the finalized
  statistic (QK-norm: sq-sums -> dense rstd*w multiplier). Any transform the
  statistics must see is explicit duplicate math in fn2 — by design.
* VARLEN: pass ``cu_seqlens_m`` (and ``A_idx`` for gather); operands/outputs
  are (total_m, ...) shaped, colvecs rank-1. No prepass or TileLoad under
  varlen yet.

Value ports (how new ops join the fn dataflow)
----------------------------------------------
An EpiOp declares ``fn_port``:
* "value": the fn receives op.name as a value; ``fn_prepare`` turns the op's
  begin_loop state into a dense per-loop-index fragment.
* "apply": the fn receives a CALLABLE — ``y = rope(acc)`` — so the op's math
  slots into the fn dataflow at a user-chosen, review-visible point.
* "sink": the fn returns op.name; the frontend collects a dense fragment and
  hands it to ``fn_sink_flush`` once per subtile (fragment-level, so sinks own
  aliasing/coupled-accumulator numerics and per-subtile rescales).
One method makes a custom op compose with everything else here.

Speed-of-light rules (bugs otherwise; all were hit once)
--------------------------------------------------------
* Inside cutlass.range bodies, python-static branches must be
  ``const_expr(...)`` (dict comprehensions dodge the AST rewriting).
* vectorize=True demands plain loop-index addressing (no 2*i — use
  flat_divide pair views) and nonzero strides (densify broadcast fragments
  with an unrolled scalar copy first; see the _dense helpers).
* Sinks fold at fragment level: per-element accumulation into the zero-stride
  aliased accumulator slice double-counts on the SM100 packed path.
* Ragged last N-tile: OOB accumulator lanes are ZERO — the identity for add,
  not for max (reduce |x|-like quantities) and not for LSE (OnlineLSEReduce
  predicates OOB per element by default; check_oob=False compiles the
  predicate out and the host then requires divisible N).

Caching and identity
--------------------
An EpiMod's ``semantic_key`` deep-fingerprints the fn (source plus every
global/closure it references, recursively — factory patterns and helper
edits change it; formatting does not) together with the prepass fn, outputs,
mode, and each op's ``cache_key()`` (type + name + ``config_key()``). The
fingerprint is FAIL-CLOSED: primitives, containers, enums, modules/classes,
functions (incl. wrapped/partial/builtin), dataclasses, and objects
implementing ``__quack_semantic_key__(self) -> object`` are supported;
anything else raises at decoration time — a capture we cannot fingerprint
must never reach the compile cache, because a too-coarse key silently reuses
the wrong kernel. EpiOps implement the protocol as their ``cache_key()``.
Kernel classes are minted per (semantic digest, operand kinds, SM, modes)
but never cross the jit-cache boundary directly: compiles carry a picklable
:class:`~quack.gemm_host.GemmClassRef` recipe that re-mints the class at the
point of use — by importing the module-global EpiMod, or, for EpiMods with
no importable anchor (``__main__``, notebooks), via a digest-validated
cloudpickle payload installed into async workers as a side channel that
never touches the cache key. Same digest -> same disk ``.o``, across
processes and workers.

Why this design (and not an epilogue IR)
----------------------------------------
The goals are speed of light AND low marginal cost per new epilogue, and the
two pressures meet at the op boundary:

* The fn is the COMPOSITION site. Ordering (``rope(acc) * alpha`` vs
  ``rope(acc * alpha)``) is explicit user code, reviewable in place — not
  graph topology (EVT trees) or sequencing lists. CuTe-DSL's tracing already
  inlines the fn; an epilogue-level IR would only re-derive what MLIR below
  us optimizes anyway (shared subexpressions are shared SSA values for
  free), while hiding the packed-intrinsic and register-layout control the
  hand-written mixins prove is worth having.
* EpiOps are the EXTENSION site. Whoever adds an op — a new reduction, a
  quantized store, a table load with its own prefetch pipeline — writes the
  resource lifecycle once, and ONE port method (value / apply / sink) makes
  it usable from every fn, composed with every other op, with host plumbing,
  caching, and launch inherited from the schema. The proof cases:
  RotaryCosSinLoad ran verbatim with a 15-line adapter; OnlineLSEReduce
  (a coupled accumulator no combine= flag could express) is one class that
  every mod can name in ``outs=``. Ops written before this frontend existed
  keep working in hand-written mixins unchanged — the fn form is a shortcut
  onto the same machinery, never a second framework.
* The escape hatches are graded, not cliffs: pin one operand (``ops=``),
  add one port method, or drop to a full mixin — each step keeps everything
  else composing.

Everything here is pinned by tests/test_gemm_epilogue.py: bitwise-or-1-ulp
and <=1% perf vs the hand-written mixins for every expressible epilogue.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import inspect
import os
import sys
import sysconfig
import types
from typing import NamedTuple, Optional

from cutlass import Float32

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

from quack.cute_dsl_utils import get_device_capacity, mlir_namedtuple, torch2cute_dtype_map
from quack.epi_composable import ComposableEpiMixin
from quack.epi_ops import (
    ColVecLoad,
    EpiOp,
    RowVecLoad,
    Scalar,
    TileLoad,
    TileStore,
    VecLoad,
    VecReduce,
)
from quack.gemm_host import (
    GemmClassRef,
    GemmEpiPlan,
    build_gemm_epi_plan,
    register_local_epi_mod,
    run_gemm_epi_plan,
)
from quack.gemm_config import blockscaled_default_config, default_config
from quack.gemm_tvm_ffi_utils import tensor_key
from quack.gemm_sm80 import GemmSm80
import quack.layout_utils as layout_utils
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_sm120 import GemmSm120
from quack.rounding import RoundingMode

_SM_BASE = {8: GemmSm80, 9: GemmSm90, 10: GemmSm100, 11: GemmSm100, 12: GemmSm120}

_EPI_MODES = {"element", "acc_pair", "packed_cd_b16x2"}


def _semantic_value_key(value, seen):
    """Fail-closed semantic fingerprint of a value an epilogue fn depends on.

    Supported: primitives, containers, enums, modules/classes (by qualname —
    their source is covered by the package fingerprint), functions/methods/
    builtins/partials, dataclasses, and anything implementing
    ``__quack_semantic_key__(self) -> object`` (recursed through this same
    keyer). Everything else raises: a value we cannot fingerprint must never
    reach the compile cache, because a too-coarse key silently reuses the
    wrong kernel.
    """
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    qsk = getattr(type(value), "__quack_semantic_key__", None)
    if qsk is not None:
        marker = ("id", id(value))
        if marker in seen:
            return ("qsk_ref", type(value).__module__, type(value).__qualname__)
        seen.add(marker)
        return (
            "qsk",
            type(value).__module__,
            type(value).__qualname__,
            _semantic_value_key(qsk(value), seen),
        )
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__module__, type(value).__qualname__, value.value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_semantic_value_key(v, seen) for v in value))
    if isinstance(value, list):
        return ("list", tuple(_semantic_value_key(v, seen) for v in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(sorted((repr(k), _semantic_value_key(v, seen)) for k, v in value.items())),
        )
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(repr(_semantic_value_key(v, seen)) for v in value)))
    if isinstance(value, types.ModuleType):
        return ("module", value.__name__)
    if inspect.isfunction(value):
        return _function_semantic_key(value, seen)
    if inspect.ismethod(value):
        return (
            "method",
            _function_semantic_key(value.__func__, seen),
            _semantic_value_key(value.__self__, seen),
        )
    if isinstance(value, (types.BuiltinFunctionType, types.MethodWrapperType)):
        return ("builtin", getattr(value, "__module__", None), value.__qualname__)
    wrapped = getattr(value, "__wrapped__", None)
    if callable(value) and wrapped is not None:
        # Decorator wrappers (functools.wraps chains: lru_cache, dsl_user_op,
        # cute.jit): the semantics live in the wrapped function.
        return ("wrapped", _semantic_value_key(wrapped, seen))
    if isinstance(value, functools.partial):
        return (
            "partial",
            _semantic_value_key(value.func, seen),
            _semantic_value_key(value.args, seen),
            _semantic_value_key(value.keywords, seen),
        )
    if inspect.isclass(value):
        return ("class", value.__module__, value.__qualname__)
    if dataclasses.is_dataclass(value):
        marker = ("id", id(value))
        if marker in seen:
            return ("dataclass_ref", type(value).__module__, type(value).__qualname__)
        seen.add(marker)
        return (
            "dataclass",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (f.name, _semantic_value_key(getattr(value, f.name), seen))
                for f in dataclasses.fields(value)
            ),
        )
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return ("torch.dtype", str(value))
    raise TypeError(
        f"epilogue fn depends on {value!r} (type {type(value).__module__}."
        f"{type(value).__qualname__}), which has no fail-closed semantic key. "
        "Supported: primitives, containers, enums, functions, dataclasses, "
        "modules/classes. For anything else, implement "
        "__quack_semantic_key__(self) -> object returning a supported value "
        "that changes whenever the traced math would."
    )


@functools.lru_cache(maxsize=1)
def _stdlib_root() -> str:
    return os.path.abspath(sysconfig.get_paths()["stdlib"]) + os.sep


def _is_extern_function(fn) -> bool:
    """True for functions defined in installed (stdlib / site-packages /
    dist-packages) code outside the quack package. Like classes and modules,
    they fingerprint by qualname only: their source is pinned by the installed
    distribution (the disk cache additionally stamps the cutlass version and
    hashes every quack source file), and recursing into them would pull
    runtime-MUTABLE library globals into the digest — e.g. any fn touching
    cutlass's dsl_user_op machinery reaches cutlass._mlir_helpers.op, which
    lazily materializes _DSL_PACKAGE_ROOT(S) on the first traced op, so the
    digest would depend on whether this process compiled anything yet (async
    workers resolve module-global EpiMods by re-import and reject the ref as
    "changed" on any mismatch)."""
    module = getattr(fn, "__module__", None) or ""
    if module == "quack" or module.startswith("quack."):
        return False
    code = getattr(fn, "__code__", None)
    if code is None:
        return False
    filename = code.co_filename
    if f"{os.sep}site-packages{os.sep}" in filename or f"{os.sep}dist-packages{os.sep}" in filename:
        return True
    return filename.startswith(_stdlib_root())


def _function_semantic_key(fn, seen=None):
    """Fingerprint source plus the globals/closures that can change its math."""
    seen = set() if seen is None else seen
    ident = (fn.__module__, fn.__qualname__)
    if ident in seen:
        return ("function_ref", *ident)
    seen.add(ident)
    if _is_extern_function(fn):
        return ("extern_function", *ident)
    try:
        source = inspect.getsource(fn).encode()
    except (OSError, TypeError):
        code = getattr(fn, "__code__", None)
        if code is None:
            raise TypeError(f"cannot fingerprint epilogue callable {fn!r}") from None
        source = code.co_code + repr(code.co_consts).encode()
    try:
        closure_vars = inspect.getclosurevars(fn)
        referenced = {
            **closure_vars.globals,
            **closure_vars.nonlocals,
        }
    except TypeError:
        referenced = {}
    deps = tuple(
        (name, _semantic_value_key(value, seen))
        for name, value in sorted(referenced.items())
        if not name.startswith("__")
    )
    return (
        "function",
        *ident,
        hashlib.sha256(source).hexdigest(),
        _semantic_value_key(fn.__defaults__, seen),
        _semantic_value_key(fn.__kwdefaults__, seen),
        deps,
    )


class Pair(NamedTuple):
    """A two-lanes-per-logical-element epilogue value.

    Pairing is declared with ``mode=`` — the fn body calls ``unpack``/``pack``
    where it uses the lanes:

    * aux output buffer at half of GEMM-N — the accumulator pairs over
      adjacent N columns (gated): ``gate, up = unpack(acc)``; aux values are
      per-pair, and returning ``"D": pack(g, u)`` writes both lanes back.
    * 16-bit C at twice GEMM-N — C and D pack two lanes per 32-bit element
      (dgated): ``x, y = unpack(c)``, return ``"D": pack(dx, dy)``; pass C/D
      as their natural 16-bit tensors.

    As a value it is a plain tuple of the two lanes with lane-wise ``+ - *``
    (scalars broadcast), so ``acc * rstd + bias`` works before unpacking."""

    a: object
    b: object

    @staticmethod
    def _lift(v):
        return v if isinstance(v, tuple) else (v, v)

    def __add__(self, other):
        o = Pair._lift(other)
        return Pair(self.a + o[0], self.b + o[1])

    __radd__ = __add__

    def __mul__(self, other):
        o = Pair._lift(other)
        return Pair(self.a * o[0], self.b * o[1])

    __rmul__ = __mul__

    def __sub__(self, other):
        o = Pair._lift(other)
        return Pair(self.a - o[0], self.b - o[1])

    def __rsub__(self, other):
        o = Pair._lift(other)
        return Pair(o[0] - self.a, o[1] - self.b)

    def __neg__(self):
        return Pair(-self.a, -self.b)


def unpack(value):
    """Split a paired epilogue value into its two lanes: ``x, y = unpack(c)``.
    Fails loudly at trace time if the tensors didn't imply pairing."""
    assert isinstance(value, Pair), (
        "unpack() got a non-paired value. Declare mode='acc_pair' to pair adjacent "
        "accumulator lanes or mode='packed_cd_b16x2' to unpack 16-bit C/D lanes."
    )
    return value.a, value.b


pack = Pair  # returning {"D": pack(dx, dy)} packs both lanes back


class F2(NamedTuple):
    """A packed f32x2 lane pair. IS a tuple, so ``quack.activation`` functions
    take it on their packed path; arithmetic lowers to packed intrinsics.
    Scalar operands broadcast: ``x * alpha`` and ``alpha * x`` both work."""

    lo: object
    hi: object

    @staticmethod
    def _pair(v):
        return v if isinstance(v, tuple) else (v, v)

    def __add__(self, other):
        if isinstance(other, F16Lanes):
            return other.__radd__(self)
        return F2(*cute.arch.add_packed_f32x2(self, F2._pair(other)))

    __radd__ = __add__

    def __mul__(self, other):
        return F2(*cute.arch.mul_packed_f32x2(self, F2._pair(other)))

    __rmul__ = __mul__

    def __sub__(self, other):
        return F2(*cute.arch.sub_packed_f32x2(self, F2._pair(other)))

    def __rsub__(self, other):
        return F2(*cute.arch.sub_packed_f32x2(F2._pair(other), self))

    def __neg__(self):
        return F2(-self.lo, -self.hi)

    def fma(self, mul, add):
        """self * mul + add as one packed FMA."""
        return F2(*cute.arch.fma_packed_f32x2(self, F2._pair(mul), F2._pair(add)))


class F16Lanes(F2):
    """An F2 whose lanes were promoted from a 16-bit float C fragment (fp16 OR
    bf16 — "f16" as in floating-point compute; the PTX forms below take both
    .atypes), remembering the raw 16-bit lanes. Semantically it IS the promoted F2 (activation fns,
    muls, packed intrinsics — every existing use behaves identically), but the
    operations with a mixed-precision ISA form pick the scalar lowering where
    the promote folds into the op, exactly:

    * ``x + c`` / ``c + x`` -> PTX ``add.rn.f32.{f16,bf16}`` -> SASS FHADD
    * ``c - x``             -> PTX ``sub.rn.f32.{f16,bf16}`` -> FHADD w/ neg
      (``x - c`` has no mixed form; it materializes like everything else)

    When only these consume the value, the eager promotes emitted here are
    dead code and NVVM removes them. Not yet exploited: ``fma.rn.f32.abtype``
    (BOTH multiplicands 16-bit -> FHFMA, always bitwise-safe because a 16-bit
    x 16-bit product is exact in f32) — needs a lazy-product value type and a
    consumer; no current epilogue fn multiplies two raw 16-bit operands."""

    def __new__(cls, a16, b16):
        self = super().__new__(cls, a16.to(Float32), b16.to(Float32))
        self._a16 = a16
        self._b16 = b16
        return self

    def __add__(self, other):
        if isinstance(other, F16Lanes):
            # both sides 16-bit: promote one side, mixed-add the other
            other = other._f2()
        if isinstance(other, F2):
            return F2(other.lo + self._a16.to(Float32), other.hi + self._b16.to(Float32))
        return self._f2() + other

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, F16Lanes):
            other = other._f2()
        if isinstance(other, F2):
            return F2(self._a16.to(Float32) - other.lo, self._b16.to(Float32) - other.hi)
        return self._f2() - other

    def _f2(self):
        return F2(self.lo, self.hi)


class _EpiModMixinBase(ComposableEpiMixin):
    """Generic hooks for minted epilogue-mod kernels. The minted class supplies
    ``_epi_ops``, ``_epi_mod_fn``, ``_epi_mod_operands`` ((name, kind) pairs),
    ``_epi_mod_outputs``, and ``EpilogueArguments``."""

    _epi_mod_fn = None
    _epi_mod_operands = ()
    _epi_mod_outputs = ()
    _epi_mod_sinks = ()  # names of sink-port ops (fn returns them; op consumes)
    _epi_mod_group_n = 1  # 2 = gated: fn consumes adjacent-N pairs, aux is half-width
    _epi_mod_packed_cd = False  # dgated: C/D pack 2 x implicit_dtype lanes per f32
    _epi_mod_prepass_fn = None  # fn run over the raw accumulator before any store
    _epi_mod_prepass_operands = ()  # ((name, kind), ...) subset the prepass fn reads
    _epi_mod_prepass_outs = ()  # sink-op names the prepass fn returns
    _epi_mod_rounding = RoundingMode.RN  # kernel-global rounding (D store + default for TileStores)
    _epi_mod_vectorize = None  # False = keep the SM100 loop vectorizer off (escape hatch)
    _extra_param_fields = ()  # the fn is a class attr, not a param

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = self._epi_mod_rounding
        self.epi_needs_acc_prepass = self._epi_mod_prepass_fn is not None
        if self._epi_mod_packed_cd:
            assert self.implicit_dtype.width == 16, "packed_cd lanes must be 16-bit"
            assert self.d_dtype.width == 32, "packed_cd D storage must be 32-bit (f32 view)"
            assert self.c_dtype.width == 32, "packed_cd C storage must be 32-bit (f32 view)"
        # Aux-output constraints (gated 16-bit n-major, SM90 tile_N % 32) are
        # asserted by each TileStore op in to_params; the store path itself is
        # the generic ComposableEpiMixin/TileStore one.
        d = self._epi_ops_to_params_dict(args)
        for key in getattr(self, "concat_layout", None) or ():
            if key in d:
                d[key] = layout_utils.concat_to_interleave(d[key], 1)
        return self.EpilogueParams(**d)

    def _make_sink_tmps(self, ops_by_name, shape):
        """One collection fragment per sink op; scaled reduces get a
        (val, scale) fragment pair so the fold can be a single fused FMA."""
        return tuple(
            (
                (
                    cute.make_rmem_tensor(shape, self.acc_dtype),
                    cute.make_rmem_tensor(shape, self.acc_dtype),
                )
                if getattr(ops_by_name[s], "scaled", False)
                else cute.make_rmem_tensor(shape, self.acc_dtype)
            )
            for s in self._epi_mod_sinks
        )

    @cute.jit
    def _flush_sinks(self, ops_by_name, epi_loop_tensors, sink_tmps):
        for sname, stmp in zip(self._epi_mod_sinks, sink_tmps):
            if const_expr(isinstance(stmp, tuple)):
                ops_by_name[sname].fn_sink_flush(
                    self, epi_loop_tensors[sname], stmp[0], scale=stmp[1]
                )
            else:
                ops_by_name[sname].fn_sink_flush(self, epi_loop_tensors[sname], stmp)

    @cute.jit
    def epi_prepass_subtile(self, params, epi_tensors, tRS_rD, epi_coord, epi_idx):
        """Driver prepass hook (epi_needs_acc_prepass): run the prepass fn over
        this subtile's raw accumulator, collect its returns, flush to the
        prepass sink ops. Scalar unrolled loop — the prepass is a statistics
        sweep, not the store path."""
        pfn = self._epi_mod_prepass_fn
        ops_by_name = {op.name: op for op in self._epi_ops}
        frags = {}
        for name, kind in self._epi_mod_prepass_operands:
            state = ops_by_name[name].begin_loop(self, epi_tensors[name], epi_coord)
            if const_expr(kind == "tile"):
                state = state.to(self.acc_dtype)
            frags[name] = state
        sink_states = {
            name: ops_by_name[name].begin_loop(self, epi_tensors[name], epi_coord)
            for name in self._epi_mod_prepass_outs
        }
        tmps = {
            name: cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
            for name in self._epi_mod_prepass_outs
        }
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            kw = {
                name: (frags[name] if kind == "scalar" else frags[name][i])
                for name, kind in self._epi_mod_prepass_operands
            }
            res = pfn(tRS_rD[i], **kw)
            for name in self._epi_mod_prepass_outs:
                tmps[name][i] = res[name]
        for name in self._epi_mod_prepass_outs:
            ops_by_name[name].fn_sink_flush(self, sink_states[name], tmps[name])

    @cute.jit
    def epi_prepass_end(self, params, epi_tensors):
        # Flush register-accumulated statistics to smem (ops that batch the
        # prepass sweep in registers expose fn_prepass_end), then order every
        # thread's prepass sink writes before the store pass reads the
        # finalized statistics.
        ops_by_name = {op.name: op for op in self._epi_ops}
        for name in self._epi_mod_prepass_outs:
            op = ops_by_name[name]
            if const_expr(hasattr(op, "fn_prepass_end")):
                op.fn_prepass_end(self, epi_tensors[name])
        self.epilogue_barrier.arrive_and_wait()

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        fn = self._epi_mod_fn
        ops_by_name = {op.name: op for op in self._epi_ops}
        paired = self._epi_mod_group_n == 2
        # SM100 element mode with 16-bit full-tile inputs (the C operand,
        # TileLoad residual streams): keep them unwidened and hand the fn
        # F16Lanes pairs — additive uses lower to mixed-precision scalar adds
        # (FHADD.BF16/.F16: the promote folds into the add, exactly, saving
        # the cvt/PRMT per lane); every other use sees the promoted F2 and the
        # unused promotes are DCE'd. Only the packed-lane loop needs this;
        # scalar loops get the same fusion from NVVM automatically.
        mixed_lanes_ok = const_expr(
            self.arch == 100
            and not paired
            and not self._epi_mod_packed_cd
            and self.acc_dtype == Float32
            and cute.size(tRS_rD) % 2 == 0  # only the packed-lane loop consumes F16Lanes
        )
        mixed_names = set()
        frags = {}
        for name, kind in self._epi_mod_operands:
            if const_expr(kind == "apply"):
                # Apply-port op: per-subtile port state; the fn gets a callable.
                frags[name] = ops_by_name[name].fn_prepare(self, epi_loop_tensors[name], paired)
            elif const_expr(kind == "c"):
                assert tRS_rC is not None, f"epilogue operand '{name}' requires the C operand"
                if const_expr(mixed_lanes_ok and tRS_rC.element_type.width == 16):
                    frags[name] = tRS_rC
                    mixed_names.add(name)
                elif const_expr(not self._epi_mod_packed_cd):
                    frags[name] = tRS_rC.to(self.acc_dtype)
                # packed_cd: C is recast/unpacked in the packed branch below.
            elif const_expr(kind == "tile"):
                if const_expr(mixed_lanes_ok and epi_loop_tensors[name].element_type.width == 16):
                    frags[name] = epi_loop_tensors[name]
                    mixed_names.add(name)
                else:
                    frags[name] = epi_loop_tensors[name].to(self.acc_dtype)
            elif const_expr(kind == "value"):
                # Custom value-source op: fn_prepare turns its begin_loop state
                # into the dense per-element fragment the loops index (default
                # fn_prepare is identity for ops whose begin_loop IS the frag).
                frags[name] = ops_by_name[name].fn_prepare(self, epi_loop_tensors[name], paired)
            else:  # "row" / "col" fragments are already acc dtype; "scalar" is a value
                frags[name] = epi_loop_tensors[name]
        if const_expr(self._epi_mod_packed_cd):
            # dgated shape: the accumulator is already per-pair (one dout per
            # gate/up pair); C and D pack two implicit-dtype (16-bit) lanes
            # into each 32-bit element. Structure mirrors the hand-written
            # GemmDGatedMixin: recast C -> widen to f32 -> pair views; scalar
            # calls with vectorize on SM100; pack (dx, dy) back into tRS_rD.
            implicit = self.implicit_dtype
            xy16 = cute.recast_tensor(tRS_rC, implicit)
            xy = xy16.to(Float32)
            xy_pair = cute.flat_divide(xy, cute.make_layout(2))
            xv, yv = xy_pair[0, ...], xy_pair[1, ...]
            dxy = cute.make_rmem_tensor(xy16.layout, Float32)
            dxy_pair = cute.flat_divide(dxy, cute.make_layout(2))
            dxv, dyv = dxy_pair[0, ...], dxy_pair[1, ...]
            n_el = cute.size(tRS_rD)

            def _dense1(view):
                # Zero-stride broadcast frags are invalid vectorized loads.
                out = cute.make_rmem_tensor(n_el, self.acc_dtype)
                for j in cutlass.range(n_el, unroll_full=True):
                    out[j] = view[j]
                return out

            views = {}
            for name, kind in self._epi_mod_operands:
                if const_expr(kind in ("row", "col")):
                    views[name] = _dense1(frags[name])
                elif const_expr(kind != "c"):
                    views[name] = frags[name]  # scalar / dense tile frag / apply pstate
            outs = tuple(
                cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
                for _ in self._epi_mod_outputs
            )
            sink_tmps = self._make_sink_tmps(ops_by_name, tRS_rD.layout.shape)
            val_names = self._epi_mod_outputs + self._epi_mod_sinks
            val_frags = outs + sink_tmps
            vectorize = const_expr(self.arch == 100 and self._epi_mod_vectorize is not False)
            for i in cutlass.range(n_el, vectorize=vectorize):
                kw = {
                    name: (
                        (lambda v, _n=name, _i=i: ops_by_name[_n].fn_apply(self, views[_n], _i, v))
                        if kind == "apply"
                        else Pair(xv[i], yv[i])
                        if kind == "c"
                        else (views[name] if kind == "scalar" else views[name][i])
                    )
                    for name, kind in self._epi_mod_operands
                }
                res = fn(tRS_rD[i], **kw)
                d = res["D"]  # required: it carries the (dx, dy) pair to pack
                dxv[i], dyv[i] = d[0], d[1]
                for vname, vfrag in zip(val_names, val_frags):
                    if const_expr(isinstance(vfrag, tuple)):
                        # Scaled sink: the fn returns the (val, scale) factors.
                        v, s = res[vname]
                        vfrag[0][i], vfrag[1][i] = v, s
                    else:
                        vfrag[i] = res[vname]
            dxy16 = dxy.to(implicit)
            tRS_rD.store(cute.recast_tensor(dxy16, Float32).load())
            self._flush_sinks(ops_by_name, epi_loop_tensors, sink_tmps)
            return outs

        if const_expr(paired):
            # Gated pairs: adjacent-N accumulator lanes feed one fn call; aux
            # fragments are half-width. Same structure as the hand-written
            # GemmGatedMixin: flat_divide pair views built OUTSIDE the loop so
            # every in-loop access is a plain loop index (the SM100 vectorizer
            # rejects affine indices like 2*i), scalar calls + vectorize=True.
            aux_shape = cute.recast_layout(2, 1, tRS_rD.layout).shape
            outs = tuple(
                cute.make_rmem_tensor(aux_shape, self.acc_dtype) for _ in self._epi_mod_outputs
            )
            # Sink values span both lanes (full N): collect through pair views.
            # (Scaled sinks are rejected in acc_pair mode at EpiMod init: a
            # tuple return already means the two lanes here.)
            sink_tmps = tuple(
                cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
                for _ in self._epi_mod_sinks
            )
            sink_views = tuple(
                (p[0, ...], p[1, ...])
                for p in (cute.flat_divide(t, cute.make_layout(2)) for t in sink_tmps)
            )
            acc_pair = cute.flat_divide(tRS_rD, cute.make_layout(2))
            acc0, acc1 = acc_pair[0, ...], acc_pair[1, ...]
            n_groups = cute.size(acc0)

            def _dense(view):
                # Broadcast-vector fragments have zero-stride modes, which the
                # vectorizer rejects as loop loads; materialize a stride-1 copy
                # with an unrolled scalar loop (legal on zero-stride views).
                out = cute.make_rmem_tensor(n_groups, self.acc_dtype)
                for j in cutlass.range(n_groups, unroll_full=True):
                    out[j] = view[j]
                return out

            views = {}
            for name, kind in self._epi_mod_operands:
                if const_expr(kind in ("scalar", "apply")):
                    views[name] = frags[name]
                else:
                    p = cute.flat_divide(frags[name], cute.make_layout(2))
                    if const_expr(kind == "col"):
                        # colvec broadcasts along N: both lanes are identical.
                        views[name] = _dense(p[0, ...])
                    elif const_expr(kind == "row"):
                        views[name] = (_dense(p[0, ...]), _dense(p[1, ...]))
                    else:  # tile / c views are dense by construction
                        views[name] = (p[0, ...], p[1, ...])
            vectorize = const_expr(self.arch == 100 and self._epi_mod_vectorize is not False)
            for i in cutlass.range(cute.size(acc0), unroll_full=True, vectorize=vectorize):
                kw = {
                    name: (
                        (lambda v, _n=name, _i=i: ops_by_name[_n].fn_apply(self, views[_n], _i, v))
                        if kind == "apply"
                        else views[name]
                        if kind == "scalar"
                        else (
                            views[name][i]
                            if kind == "col"
                            else Pair(views[name][0][i], views[name][1][i])
                        )
                    )
                    for name, kind in self._epi_mod_operands
                }
                res = fn(Pair(acc0[i], acc1[i]), **kw)
                for oname, ofrag in zip(self._epi_mod_outputs, outs):
                    ofrag[i] = res[oname]
                for (s0, s1), sname in zip(sink_views, self._epi_mod_sinks):
                    v = res[sname]
                    s0[i], s1[i] = v[0], v[1]
                if const_expr("D" in res):
                    d = res["D"]
                    acc0[i], acc1[i] = d[0], d[1]
            for sname, stmp in zip(self._epi_mod_sinks, sink_tmps):
                ops_by_name[sname].fn_sink_flush(self, epi_loop_tensors[sname], stmp)
            return outs

        outs = tuple(
            cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
            for _ in self._epi_mod_outputs
        )
        # Sink values are collected into a plain fragment per sink op (a
        # (val, scale) fragment pair for scaled reduces), then handed to the
        # op's fn_sink_flush (fragment-level: the op owns the fold into its —
        # possibly aliased, possibly coupled — accumulators).
        sink_tmps = self._make_sink_tmps(ops_by_name, tRS_rD.layout.shape)
        # Names written by the fn, in collection order after "D".
        val_names = self._epi_mod_outputs + self._epi_mod_sinks
        val_frags = outs + sink_tmps
        if const_expr(self.arch == 100 and cute.size(tRS_rD) % 2 == 0):
            # Packed f32x2 lanes: same loop shape as the hand-written SM100 mixins.
            for i in cutlass.range(cute.size(tRS_rD) // 2, unroll_full=True):
                kw = {
                    name: (
                        (lambda v, _n=name, _i=i: ops_by_name[_n].fn_apply(self, frags[_n], _i, v))
                        if kind == "apply"
                        else frags[name]
                        if kind == "scalar"
                        else F16Lanes(frags[name][2 * i], frags[name][2 * i + 1])
                        if name in mixed_names
                        else F2(frags[name][2 * i], frags[name][2 * i + 1])
                    )
                    for name, kind in self._epi_mod_operands
                }
                res = fn(F2(tRS_rD[2 * i], tRS_rD[2 * i + 1]), **kw)
                if const_expr("D" in res):
                    d = res["D"]
                    tRS_rD[2 * i], tRS_rD[2 * i + 1] = d[0], d[1]
                for vname, vfrag in zip(val_names, val_frags):
                    if const_expr(isinstance(vfrag, tuple)):
                        v, s = res[vname]
                        vfrag[0][2 * i], vfrag[0][2 * i + 1] = v[0], v[1]
                        vfrag[1][2 * i], vfrag[1][2 * i + 1] = s[0], s[1]
                    else:
                        v = res[vname]
                        vfrag[2 * i], vfrag[2 * i + 1] = v[0], v[1]
        else:
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                kw = {
                    name: (
                        (lambda v, _n=name, _i=i: ops_by_name[_n].fn_apply(self, frags[_n], _i, v))
                        if kind == "apply"
                        else (frags[name] if kind == "scalar" else frags[name][i])
                    )
                    for name, kind in self._epi_mod_operands
                }
                res = fn(tRS_rD[i], **kw)
                if const_expr("D" in res):
                    tRS_rD[i] = res["D"]
                for vname, vfrag in zip(val_names, val_frags):
                    if const_expr(isinstance(vfrag, tuple)):
                        v, s = res[vname]
                        vfrag[0][i], vfrag[1][i] = v, s
                    else:
                        vfrag[i] = res[vname]
        self._flush_sinks(ops_by_name, epi_loop_tensors, sink_tmps)
        return outs


_KIND_TO_OP = {
    "row": RowVecLoad,
    "col": ColVecLoad,
    "tile": TileLoad,
    "scalar": Scalar,
}


def _infer_kind(name, value, m, n, varlen_m=False):
    if not hasattr(value, "stride"):  # python number
        return "scalar"
    if value.ndim == 0 or value.numel() == 1:
        return "scalar"
    if value.ndim in (2, 3) and tuple(value.shape[-2:]) == (m, n):
        return "tile"
    inner = value.shape[-1]
    if value.ndim <= 2 and inner in (m, n):
        if value.ndim == 1:
            # Rank-1 vectors are the varlen colvec form: (total_m,), offset per
            # segment via cu_seqlens on the device side.
            if not (varlen_m and inner == m):
                raise ValueError(
                    f"operand '{name}': rank-1 vectors are varlen colvecs (total_m,); "
                    f"dense calls pass (l, dim)"
                )
            return "col"
        if m == n:
            raise ValueError(
                f"operand '{name}': m == n makes row/col inference ambiguous; "
                f"pin it via @gemm_epilogue(ops={{'{name}': RowVecLoad(...) or ColVecLoad(...)}})"
            )
        return "row" if inner == n else "col"
    raise ValueError(
        f"cannot infer epilogue operand kind for '{name}' with shape {tuple(value.shape)}"
    )


def _require_shape(name, tensor, expected):
    if tensor is None:
        return
    actual = tuple(tensor.shape)
    expected = tuple(expected)
    if actual != expected:
        raise ValueError(f"{name} must have shape {expected}, got {actual}")


def _tile_shape(batch, m, n, varlen_m):
    return (m, n) if varlen_m or batch is None else (batch, m, n)


def _validate_packed_tensor(name, tensor):
    import torch

    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"{name} must be float16 or bfloat16 in packed_cd_b16x2 mode")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be N-major in packed_cd_b16x2 mode")
    if tensor.storage_offset() % 2 or any(s % 2 for s in tensor.stride()[:-1]):
        raise ValueError(f"{name} storage offset and outer strides must permit a float32 view")


def _mod_gemm_key(A, B, D, C, epi_args, epi_key_overrides, *tail) -> tuple:
    """Plan-cache key for EpiMod.gemm, built from raw call inputs only (no
    validation, no kind inference): full tensor metadata for the GEMM operands
    and every epi_args entry (tensor metadata subsumes the inferred kinds and
    the shape checks; scalars key by presence — their compile mode arrives via
    epi_key_overrides), plus the config tail. A hit is exactly a replay of a
    previously validated call with different data pointers."""
    epi_meta = tuple(
        (name, tensor_key(v) if hasattr(v, "stride") else v is not None)
        for name, v in sorted(epi_args.items())
    )
    overrides = tuple(sorted((epi_key_overrides or {}).items()))
    return (
        tensor_key(A),
        tensor_key(B),
        tensor_key(D),
        tensor_key(C),
        epi_meta,
        overrides,
        *tail,
    )


class EpiMod:
    """A user epilogue function plus the machinery to mint and launch kernels."""

    def __init__(
        self,
        fn,
        outputs=(),
        ops=None,
        reduces=None,
        mode=None,
        paired=(),
        outs=None,
        prepass=None,
        prepass_outs=(),
        extra_ops=(),
        vectorize=None,
    ):
        self.fn = fn
        # ``outputs`` entries are names or TileStore instances (per-op config:
        # rounding, predicate); a bare name gets a default TileStore.
        self.output_ops = {}
        out_names = []
        for out in outputs:
            if isinstance(out, TileStore):
                out_names.append(out.name)
                self.output_ops[out.name] = out
            else:
                out_names.append(out)
        self.outputs = tuple(out_names)
        # ``extra_ops``: ops the driver consumes that the fn never sees (e.g.
        # Scalar("sr_seed") feeding stochastic-rounding stores). Their values
        # travel through epi_args under the op name.
        self.extra_ops = tuple(extra_ops)
        self.ops = dict(ops or {})  # explicit EpiOp pins: {operand_name: EpiOp instance}
        # Sink-port ops by output name; ``reduces`` is kept as sugar for the
        # common VecReduce case, ``outs`` is the general form (any fn_port ==
        # "sink" op: OnlineLSEReduce, future quant stores, ...).
        self.sinks = {**dict(reduces or {}), **dict(outs or {})}
        # ``paired=('acc',)`` remains a compatibility spelling for mode="acc_pair".
        paired = tuple(paired)
        if paired:
            if set(paired) != {"acc"}:
                raise ValueError("paired= only supports ('acc',)")
            if mode not in (None, "acc_pair"):
                raise ValueError("paired=('acc',) conflicts with the requested mode")
            mode = "acc_pair"
        self.mode = "element" if mode is None else mode
        if self.mode not in _EPI_MODES:
            raise ValueError(f"unsupported epilogue mode {self.mode!r}; choose one of {_EPI_MODES}")
        self.paired = ("acc",) if self.mode == "acc_pair" else ()
        # None = vectorize the fn loop where supported (SM100). False = keep
        # the vectorizer off for this epilogue: escape hatch for the DSL
        # vectorizer's crash bugs (fused sincos, arith values reused across
        # pack lanes); free when the epilogue is mainloop-hidden, ~20pp of
        # kernel time when epilogue-exposed (see gemm_epilogue docstring).
        if vectorize not in (None, False):
            raise ValueError("vectorize= accepts None (auto) or False (escape hatch)")
        self.vectorize = vectorize
        self.prepass = prepass
        self.prepass_outs = tuple(prepass_outs)
        if (prepass is None) != (not self.prepass_outs):
            raise ValueError("prepass= and prepass_outs= come together")
        if prepass is not None:
            psig = list(inspect.signature(prepass).parameters)
            if not psig or psig[0] != "acc":
                raise ValueError("prepass fn must take 'acc' first")
            self.prepass_operand_names = tuple(psig[1:])
        else:
            self.prepass_operand_names = ()
        for name, op in self.ops.items():
            if not isinstance(op, EpiOp) or op.name != name:
                raise ValueError(f"op for {name!r} must be an EpiOp named {name!r}")
        for name, op in self.sinks.items():
            if not isinstance(op, EpiOp) or op.fn_port != "sink" or op.name != name:
                raise ValueError(
                    f"sink op for {name!r} must have fn_port == 'sink' and be named {name!r}"
                )
            if getattr(op, "scaled", False) and self.mode == "acc_pair":
                raise ValueError(
                    f"sink {name!r}: scaled reduces are not supported in acc_pair mode yet "
                    "(a tuple return already carries the two lanes there)"
                )
        sig = inspect.signature(fn)
        params = list(sig.parameters)
        if not params or params[0] != "acc":
            raise ValueError("epilogue fn must take 'acc' first")
        self.operand_names = tuple(params[1:])
        reserved = {"acc", "D"}
        all_names = set(self.operand_names) | set(self.outputs) | set(self.sinks)
        if all_names & reserved - {"D"}:
            raise ValueError(f"operand/output names may not use reserved names {reserved}")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("duplicate output names")
        for op in self.extra_ops:
            if not isinstance(op, EpiOp) or op.name in all_names:
                raise ValueError(f"extra op {op!r} must be an EpiOp with a fresh name")
        self.semantic_key = (
            _function_semantic_key(fn),
            _function_semantic_key(prepass) if prepass is not None else None,
            self.outputs,
            self.mode,
            self.prepass_outs,
            tuple(op.cache_key() for _, op in sorted(self.ops.items())),
            tuple(op.cache_key() for _, op in sorted(self.sinks.items())),
            tuple(op.cache_key() for _, op in sorted(self.output_ops.items())),
            tuple(op.cache_key() for op in self.extra_ops),
            # Appended only when set so default-config digests (and their
            # disk-cached kernels) are unchanged.
            *(() if self.vectorize is None else (("vectorize", self.vectorize),)),
        )
        self.semantic_digest = hashlib.sha256(repr(self.semantic_key).encode()).hexdigest()
        self._ident = f"{fn.__name__}_{self.semantic_digest[:16]}"
        self._minted = {}
        self._plan_cache = {}
        self._call_cache = {}  # eager __call__ fast path: metadata -> launch recipe

    def __getstate__(self):
        # Shipped by value (cloudpickle) to async-compile workers when there
        # is no importable anchor: the minted classes and launch plans are
        # process-local (compiled fns, registered dynamic classes) — the
        # worker re-mints from the recipe.
        state = self.__dict__.copy()
        state["_minted"] = {}
        state["_plan_cache"] = {}
        return state

    def _module_locator(self):
        """(module, global_name) if this EpiMod is reachable by import in a
        fresh process (async workers rebuild it that way), else None: defined
        in __main__ (scripts, notebooks) or never bound to a module global."""
        module_name = self.fn.__module__
        if module_name == "__main__":
            return None
        module = sys.modules.get(module_name)
        if module is None:
            return None
        preferred = self.fn.__name__
        if getattr(module, preferred, None) is self:
            return module_name, preferred
        names = sorted(name for name, value in vars(module).items() if value is self)
        if not names:
            return None
        return module_name, names[0]

    def _class_ref(self, mint_key):
        locator = self._module_locator()
        if locator is None:
            # No importable anchor: the disk key stays semantically correct
            # (the digest is in the ref), resolution goes through the
            # process-local registry, and pool submission ships this EpiMod
            # by value (GemmClassRef.__quack_pool_payload__).
            register_local_epi_mod(self.semantic_digest, self)
            return GemmClassRef(
                "epi_mod_local",
                "",
                "",
                mint_key=mint_key,
                semantic_digest=self.semantic_digest,
            )
        return GemmClassRef(
            "epi_mod",
            *locator,
            mint_key=mint_key,
            semantic_digest=self.semantic_digest,
        )

    def _mint(self, kind_sig, sm, paired_acc, packed_c, prepass_sig=(), rounding=RoundingMode.RN):
        key = (kind_sig, sm, paired_acc, packed_c, prepass_sig, rounding)
        cls = self._minted.get(key)
        if cls is not None:
            return cls
        epi_ops = []
        for name, kind in kind_sig:
            if name in self.ops:
                op = self.ops[name]
                if not isinstance(op, EpiOp) or op.name != name:
                    raise ValueError(f"op for {name!r} must be an EpiOp named {name!r}")
                if type(op) in (ColVecLoad, RowVecLoad) and _pinned_visit_kind(op) != kind:
                    # Swap-at-trace flipped this pin into kernel coords; the
                    # visit-kind entry is authoritative so async-worker
                    # re-mints from the mint_key alone reconstruct the same
                    # class.
                    op = _KIND_TO_OP[kind](name)
                epi_ops.append(op)
            elif kind != "c":
                epi_ops.append(_KIND_TO_OP[kind](name))
        for out_name in self.outputs:
            op = self.output_ops.get(out_name)
            if op is None:
                op = TileStore(out_name, gated=paired_acc)
            elif paired_acc and not op.gated:
                raise ValueError(f"output op {out_name!r} must be gated in acc_pair mode")
            epi_ops.append(op)
        epi_ops.extend(self.sinks.values())
        epi_ops.extend(self.extra_ops)
        # The DSL's TVM-FFI arg-spec converter reads per-field type hints off
        # the NamedTuple, so mint through typing.NamedTuple with the same
        # annotations the hand-written EpilogueArguments use.
        arg_specs = [
            (
                op.name,
                Optional[(op.dtype or Float32) | cute.Tensor]
                if isinstance(op, Scalar)
                else Optional[cute.Tensor],
            )
            for op in epi_ops
        ]
        Args = NamedTuple("EpilogueArguments", arg_specs)
        Args.__new__.__defaults__ = (None,) * len(arg_specs)
        Args = mlir_namedtuple(Args)
        cls_name = (
            f"GemmMod_{self._ident}_{'g' if paired_acc else ''}{'p' if packed_c else ''}_"
            f"{'_'.join(k for _, k in kind_sig) or 'none'}_sm{sm}"
            # rounding is a call-time knob on an otherwise identical mint:
            # it must distinguish the class name or remints collide.
            f"{'' if rounding == RoundingMode.RN else f'_r{int(rounding)}'}"
        )
        class_semantic_key = (self.semantic_digest, key)
        existing = getattr(sys.modules[__name__], cls_name, None)
        if existing is not None:
            if getattr(existing, "_epi_mod_class_semantic_key", None) != class_semantic_key:
                raise RuntimeError(f"dynamic epilogue class-name collision for {cls_name}")
            self._minted[key] = existing
            return existing
        # The store path (incl. the gated halved-tile machinery) is TileStore
        # config, so every minted class shares one base pair.
        cls = type(
            cls_name,
            (_EpiModMixinBase, _SM_BASE[sm]),
            {
                "_epi_ops": tuple(epi_ops),
                "_epi_mod_fn": staticmethod(self.fn),
                "_epi_mod_operands": kind_sig,
                "_epi_mod_outputs": self.outputs,
                "_epi_mod_sinks": tuple(self.sinks),
                "_epi_mod_group_n": 2 if paired_acc else 1,
                "_epi_mod_packed_cd": packed_c,
                "_epi_mod_prepass_fn": staticmethod(self.prepass) if self.prepass else None,
                "_epi_mod_prepass_operands": prepass_sig,
                "_epi_mod_prepass_outs": self.prepass_outs,
                "_epi_mod_rounding": rounding,
                "_epi_mod_vectorize": self.vectorize,
                "_extra_param_fields": (),
                "_epi_mod_class_semantic_key": class_semantic_key,
                "EpilogueArguments": Args,
                "__module__": __name__,
                "__qualname__": cls_name,
            },
        )
        # Registration is useful for inspection and in-process reuse. The JIT
        # cache receives a GemmClassRef, never this process-local class object.
        setattr(sys.modules[__name__], cls_name, cls)
        self._minted[key] = cls
        return cls

    def gemm_tuned(self, A, B, D, C=None, *, epi_args: dict, b_kn: bool = False):
        """Autotuned gemm(): config-space sweep via quack.epi_autotune (lazy
        import — this module sits below gemm_config in the import graph).
        Returns TunedModGemm(plan, config, sinks); see tuned_mod_gemm."""
        from quack.epi_autotune import tuned_mod_gemm

        return tuned_mod_gemm(self, A, B, D, C, epi_args=epi_args, b_kn=b_kn)

    def gemm(
        self,
        A,
        B,
        D,
        C=None,
        *,
        epi_args: dict,
        tile_M: int = 128,
        tile_N: int = 256,
        cluster_M: int = 2,
        cluster_N: int = 1,
        tile_K: Optional[int] = None,
        pingpong: bool = False,
        persistent: bool = True,
        is_dynamic_persistent: bool = False,
        max_swizzle_size: int = 8,
        tile_count_semaphore=None,
        cu_seqlens_m=None,
        A_idx=None,
        rounding_mode: int = RoundingMode.RN,
        epi_key_overrides=None,  # {op_name: key} when the caller owns the key rule (scalar modes)
        b_kn: bool = False,  # B passed (k, n) / (l, k, n), relabeled at trace time (dense SM90+)
        use_tma_gather: bool = False,
        concat_layout=None,
        SFA=None,  # blockscaled scale factors, see quack.gemm
        SFB=None,
        # BlockScaledFormat names for A / B (independent; required when SFA/SFB
        # are passed) — the descriptors drive validation and the MMA dtypes.
        bs_format_a=None,
        bs_format_b=None,
        swap_ab=False,  # swap-at-trace: requires b_kn (B given (k, n)); dense element mode
        ag_args=None,  # AllGather+GEMM flags contract (see quack/distributed/)
        use_epi_reduce=None,  # "reduce_scatter" | "all_reduce" (see GemmTpTmaBase)
        epi_reduce_args=None,  # EpiReduceArguments over torch tensors
        _launch=True,  # False: resolve/compile only (EpiMod.plan) — no kernel launch
    ) -> GemmEpiPlan:
        varlen_m = cu_seqlens_m is not None
        gather_A = A_idx is not None
        blockscaled = SFA is not None
        concat_key = tuple(sorted(concat_layout)) if concat_layout else ()
        if tile_count_semaphore is not None and not is_dynamic_persistent:
            raise ValueError("tile_count_semaphore requires is_dynamic_persistent=True")
        if swap_ab:
            # Swap-at-trace contract: dense, element-mode, sink-less, B (k, n).
            if not b_kn:
                raise ValueError("swap_ab requires b_kn=True (B passed (k, n))")
            if varlen_m or gather_A or blockscaled or concat_key:
                raise ValueError("swap_ab: dense non-blockscaled only")
            if self.mode != "element" or self.sinks:
                raise ValueError("swap_ab supports element-mode sink-less epilogues only")
            if ag_args is not None:
                # With swapped slots kernel-A is the caller's B: the AG gate
                # would gate the wrong operand (and the wrong M geometry).
                raise ValueError("swap_ab does not support ag_args (AG shards kernel-A along M)")
        num_ranks = None
        if use_epi_reduce is not None:
            import torch.distributed as dist

            if varlen_m or gather_A or blockscaled or swap_ab or ag_args is not None:
                raise ValueError("use_epi_reduce: dense non-blockscaled unswapped only")
            if self.mode != "element":
                raise ValueError("use_epi_reduce supports element-mode epilogues only")
            if epi_reduce_args is None:
                raise ValueError("use_epi_reduce requires epi_reduce_args")
            num_ranks = dist.get_world_size()
        # Warm fast path: probe the plan cache on raw-input metadata before any
        # validation or kind inference — a hit is exactly a replay of a
        # previously validated call (the key subsumes everything validation
        # reads), so only the per-call views and the launch remain. The key
        # matches the cold path's record below.
        key = _mod_gemm_key(
            A,
            B,
            D,
            C,
            epi_args,
            epi_key_overrides,
            tile_M,
            tile_N,
            tile_K,
            cluster_M,
            cluster_N,
            pingpong,
            persistent,
            is_dynamic_persistent,
            max_swizzle_size,
            A.device,
            tensor_key(cu_seqlens_m),
            gather_A,
            rounding_mode,
            b_kn,
            use_tma_gather,
            concat_key,
            tensor_key(SFA),
            tensor_key(SFB),
            bs_format_a,
            bs_format_b,
            swap_ab,
            ag_args is not None,
            use_epi_reduce,
            num_ranks,
        )
        plan = self._plan_cache.get(key)
        if plan is not None:
            if _launch:
                run_gemm_epi_plan(
                    plan,
                    B if swap_ab else A,
                    A if swap_ab else B,
                    D,
                    C,
                    epi_args,
                    ag_args=ag_args,
                    epi_reduce_args=epi_reduce_args,
                    tile_count_semaphore=tile_count_semaphore,
                    cu_seqlens_m=cu_seqlens_m,
                    A_idx=A_idx,
                    SFA=SFA,
                    SFB=SFB,
                )
            return plan
        if blockscaled:
            if varlen_m or gather_A:
                raise ValueError("blockscaled GEMM does not support varlen/gather yet")
            if concat_key:
                raise ValueError("blockscaled GEMM does not support concat_layout")
            if tile_K is not None:
                raise ValueError("blockscaled GEMM derives tile_K from the MMA instruction")
        if varlen_m:
            if not persistent:
                raise ValueError("varlen_m requires persistent=True")
            num_seqs = cu_seqlens_m.shape[0] - 1
            if B.ndim != 3 or B.shape[0] != num_seqs:
                raise ValueError(
                    f"varlen_m B is per-sequence indexed: expected (num_seqs={num_seqs}, "
                    f"...) got {tuple(B.shape)}; broadcast a shared B zero-copy via "
                    "B.unsqueeze(0).expand(num_seqs, -1, -1)"
                )
            if A.ndim != 2 or A.stride(-1) != 1:
                raise ValueError("varlen_m: A is (total_m, k), k-major")
            if D is not None and (D.ndim != 2 or D.stride(-1) != 1):
                raise ValueError("varlen_m: D is (total_m, n), n-major")
            if self.prepass is not None:
                raise ValueError("acc prepass + varlen: not supported yet")
        if gather_A:
            if not varlen_m:
                raise ValueError("gather_A requires varlen")
            if cluster_N != 1:
                raise ValueError("gather_A requires cluster_N=1")
        n_gemm = B.shape[-1] if b_kn else B.shape[-2]
        # Kernel coords under swap-at-trace: kernel m = caller n, kernel n =
        # caller m. Shape checks on D/C/outputs stay caller-oriented (the
        # tensors cross natively; the trace transposes); operand-kind
        # inference and vec shape checks use kernel coords.
        paired_acc = self.mode == "acc_pair"
        packed_c = self.mode == "packed_cd_b16x2"
        if paired_acc and (n_gemm % 2 or tile_N % 2):
            raise ValueError("acc_pair mode requires even GEMM N and tile_N")
        post_init_attrs = ()
        packed_form = None
        if packed_c:
            import torch

            # Callers pass C (preact pairs) and D (dx/dy out) in their natural
            # 16-bit dtype; the tensors cross the boundary RAW and the trace
            # recasts them to the f32 packed view (GemmBase._recast_packed_cd
            # via cd_packed — no per-call torch views). The original dtype
            # travels to the trace via post_init (implicit_dtype) exactly like
            # the hand-written dgated host path.
            if "c" not in self.operand_names:
                raise ValueError("packed_cd_b16x2 mode requires a 'c' fn parameter")
            if C is None or D is None:
                raise ValueError("packed_cd_b16x2 mode requires both C and D")
            if D.dtype != C.dtype or D.shape != C.shape:
                raise ValueError("packed C requires a matching D of the same dtype and shape")
            if C.dtype not in (torch.float16, torch.bfloat16):
                raise TypeError("C must be float16 or bfloat16 in packed_cd_b16x2 mode")
            post_init_attrs = (("implicit_dtype", torch2cute_dtype_map[C.dtype]),)
        n = n_gemm
        if varlen_m:
            # total_m for operand inference (colvec length); A rows differ
            # under gather_A, so prefer an output's leading extent.
            ref_t = D if D is not None else epi_args.get((self.outputs or (None,))[0])
            if ref_t is None:
                raise ValueError("varlen_m needs D or an aux output")
            m = ref_t.shape[0]
        else:
            m = A.shape[-2]
        # Inference/vec-check dims in kernel coords; base_shape (D/C/outputs)
        # stays caller-oriented (swap-at-trace transposes those at trace).
        if use_epi_reduce is not None and m % num_ranks:
            raise ValueError(f"use_epi_reduce: m ({m}) must be divisible by world ({num_ranks})")
        # use_epi_reduce: C and every epi output/sink are slab-local (m / world); D stays full-M.
        m_epi = m if use_epi_reduce is None else m // num_ranks
        m_i, n_i = (n_gemm, m) if swap_ab else (m_epi, n_gemm)
        batch = B.shape[0] if B.ndim == 3 else None
        base_shape = _tile_shape(batch, m, n_gemm, varlen_m)
        epi_base_shape = _tile_shape(batch, m_epi, n_gemm, varlen_m)
        if use_epi_reduce is not None:
            # Guard what the kernel can only corrupt on: multimem vector width,
            # kernel-order comm views, and flag/counter capacities (an under-sized
            # flag array is a silent OOB multimem write). Warm plan-cache hits
            # skip trace-time asserts, so the host is the only per-call check.
            import torch

            era = epi_reduce_args
            if D is None:
                raise ValueError("use_epi_reduce requires D (the symmetric work buffer)")
            vec = 16 // D.element_size()
            if n_gemm % vec:
                raise ValueError(f"use_epi_reduce: n ({n_gemm}) must be divisible by {vec}")
            if D.stride(-1) != 1:
                raise ValueError("use_epi_reduce: D must be n-major (multimem vectors)")
            if len(era.mD_peers) != num_ranks:
                raise ValueError(
                    f"epi_reduce_args.mD_peers has {len(era.mD_peers)} views, world {num_ranks}"
                )
            mnl = (m, n_gemm, batch if batch is not None else 1)
            for name, t in (("mD_mc", era.mD_mc), ("mD_peers[0]", era.mD_peers[0])):
                if tuple(t.shape) != mnl:
                    raise ValueError(
                        f"epi_reduce_args.{name}: kernel-order (m, n, l) {mnl} expected, "
                        f"got {tuple(t.shape)}"
                    )
            use_2cta = cluster_M % 2 == 0 and tile_M in (128, 256)
            cta_m = tile_M // (2 if use_2cta else 1)
            n_tiles = (n_gemm + tile_N - 1) // tile_N
            ntiles = ((m + cta_m - 1) // cta_m) * n_tiles * mnl[2]
            if era.tile_flags.numel() < ntiles or era.tile_flags_mc.numel() < ntiles:
                raise ValueError(f"epi_reduce_args.tile_flags needs >= {ntiles} entries")
            num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
            if era.sync_barrier.numel() < num_sms or era.sync_barrier_mc.numel() < num_sms:
                raise ValueError(f"epi_reduce_args.sync_barrier needs >= {num_sms} entries")
            slab_tiles = ((m_epi + cta_m - 1) // cta_m) * n_tiles * mnl[2]
            if era.consumer_counters.numel() < slab_tiles:
                raise ValueError(f"epi_reduce_args.consumer_counters needs >= {slab_tiles} entries")
        if packed_c:
            if C.stride(-1) == 1 or varlen_m:
                packed_shape = _tile_shape(batch, m, 2 * n_gemm, varlen_m)
                _require_shape("C", C, packed_shape)
                _require_shape("D", D, packed_shape)
                _validate_packed_tensor("C", C)
                _validate_packed_tensor("D", D)
                packed_form = "n"
            else:
                # AB-swapped callers pass m-major C/D: the 16-bit pairs pack
                # along the contiguous M dim (the trace halves M instead of N;
                # same layout the hand-written dgated host's .mT views built).
                packed_shape = _tile_shape(batch, 2 * m, n_gemm, varlen_m)
                _require_shape("C", C, packed_shape)
                _require_shape("D", D, packed_shape)
                if C.stride(-2) != 1 or D.stride(-2) != 1:
                    raise ValueError("packed m-major C/D must be contiguous along M")
                if C.storage_offset() % 2 or D.storage_offset() % 2:
                    raise ValueError("packed m-major C/D storage offsets must permit a f32 view")
                if any(s % 2 for s in (*C.stride()[-1:], *D.stride()[-1:])) or (
                    batch is not None and any(s % 2 for s in (C.stride(0), D.stride(0)))
                ):
                    raise ValueError("packed m-major C/D outer strides must permit a f32 view")
                packed_form = "m"
        else:
            _require_shape("C", C, epi_base_shape)
            _require_shape("D", D, base_shape)
        for out_name in self.outputs:
            if out_name not in epi_args:
                raise ValueError(f"missing epilogue output buffer '{out_name}'")
            out_n = n_gemm // 2 if paired_acc else n_gemm
            _require_shape(out_name, epi_args[out_name], _tile_shape(batch, m_epi, out_n, varlen_m))
            if paired_acc:
                aux = epi_args[out_name]
                if aux.element_size() != 2:
                    raise TypeError("acc_pair auxiliary output must have a 16-bit dtype")
                if aux.stride(-1) != 1 or (D is not None and D.stride(-1) != 1):
                    raise ValueError("acc_pair auxiliary output and D must be N-major")
        # Swap-at-trace relabels pinned vec pins into KERNEL coordinates: a
        # caller colvec is the swapped kernel's rowvec (and vice versa), so the
        # pin's class flips for this call. Other orientation-sensitive vec pins
        # (varlen subclasses, reduces) have no swapped form and fail loudly.
        pins = {}
        for name, op in self.ops.items():
            if swap_ab and type(op) in (ColVecLoad, RowVecLoad):
                pins[name] = (RowVecLoad if type(op) is ColVecLoad else ColVecLoad)(name)
            elif swap_ab and isinstance(op, (VecLoad, VecReduce)):
                raise ValueError(
                    f"swap_ab: pinned vec op {name!r} of type {type(op).__name__} "
                    "has no swapped orientation"
                )
            else:
                pins[name] = op
        epi_values = {}
        kind_sig = []
        for name in self.operand_names:
            if name == "c":
                if C is None:
                    raise ValueError("epilogue fn takes 'c' but no C operand was passed")
                kind_sig.append(("c", "c"))
                continue
            if name not in epi_args:
                raise ValueError(f"missing epilogue operand '{name}'")
            kind = (
                "pinned" if name in pins else _infer_kind(name, epi_args[name], m_i, n_i, varlen_m)
            )
            if varlen_m and kind == "tile":
                raise ValueError(f"operand '{name}': TileLoad does not support varlen_m yet")
            visit_kind = _pinned_visit_kind(pins[name]) if kind == "pinned" else kind
            batch_l = B.shape[0] if B.ndim == 3 else 1
            # Pinned ops own their host schema (host_arg_key validates the
            # value); the built-in shape rules only apply to inferred kinds.
            if kind == "pinned":
                pass
            elif visit_kind == "row":
                _require_shape(name, epi_args[name], (batch_l, n_i))
            elif visit_kind == "col":
                expected = (m_i,) if varlen_m else (batch_l, m_i)
                _require_shape(name, epi_args[name], expected)
            elif visit_kind == "tile":
                _require_shape(name, epi_args[name], epi_base_shape)
            kind_sig.append((name, kind if kind != "pinned" else pins[name].__class__.__name__))
            epi_values[name] = epi_args[name]
        for out_name in self.outputs:
            epi_values[out_name] = epi_args[out_name]
        for sink_name in self.sinks:
            if sink_name not in epi_args:
                raise ValueError(f"missing sink output buffer '{sink_name}'")
            op = self.sinks[sink_name]
            if hasattr(op, "dim"):
                if op.dim == 0:
                    inner = (m_epi, (n_gemm + tile_N - 1) // tile_N)
                else:
                    inner = ((m_epi + tile_M - 1) // tile_M, n_gemm)
                expected = inner if varlen_m or batch is None else (batch, *inner)
                _require_shape(sink_name, epi_args[sink_name], expected)
            if getattr(op, "check_oob", True) is False and n_gemm % tile_N:
                raise ValueError(
                    f"sink '{sink_name}': check_oob=False requires N divisible by tile_N "
                    f"(N={n_gemm}, tile_N={tile_N})"
                )
            epi_values[sink_name] = epi_args[sink_name]
        for op in self.extra_ops:
            if op.name in epi_args:
                epi_values[op.name] = epi_args[op.name]
        kind_sig = tuple(kind_sig)

        device_capacity = get_device_capacity(A.device)
        if is_dynamic_persistent and device_capacity[0] == 9 and tile_count_semaphore is None:
            raise ValueError("SM90 dynamic persistent scheduling requires tile_count_semaphore")
        if paired_acc and self.outputs and device_capacity[0] == 9 and tile_N % 32:
            raise ValueError("SM90 acc_pair auxiliary output requires tile_N divisible by 32")
        sf_dtype = sf_vec_size = None
        a_mma_dtype = b_mma_dtype = None
        if blockscaled:
            from quack.gemm_tvm_ffi_utils import (
                resolve_blockscaled_formats,
                validate_blockscaled_sf,
            )

            fmt_a, fmt_b = resolve_blockscaled_formats(bs_format_a, bs_format_b)
            a_mma_dtype = fmt_a.to_cutlass_dtype()
            b_mma_dtype = fmt_b.to_cutlass_dtype()
            sf_dtype, sf_vec_size = validate_blockscaled_sf(
                A, B, SFA, SFB, device_capacity, b_kn=b_kn, fmt_a=fmt_a, fmt_b=fmt_b
            )
        # Re-map pinned ops' kind for the device loop: explicit pins still
        # need a fragment kind; VecLoads present as their dim.
        visit_sig = tuple(
            (name, _pinned_visit_kind(pins[name]) if name in pins else kind)
            for name, kind in kind_sig
        )
        prepass_sig = ()
        if self.prepass is not None:
            if packed_c:
                raise ValueError("prepass + packed C: not supported")
            unknown = set(self.prepass_operand_names) - {n for n, _ in visit_sig}
            if unknown:
                raise ValueError(f"prepass fn reads undeclared operands {unknown}")
            for out_name in self.prepass_outs:
                if out_name not in {n for n, _ in visit_sig} | set(self.sinks):
                    raise ValueError(f"prepass out '{out_name}' must be a declared op")
            prepass_sig = tuple((n, k) for n, k in visit_sig if n in self.prepass_operand_names)
        mint_key = (
            visit_sig,
            device_capacity[0],
            paired_acc,
            packed_c,
            prepass_sig,
            rounding_mode,
        )
        GemmCls = self._mint(*mint_key)
        A_s, B_s = (B, A) if swap_ab else (A, B)
        plan = build_gemm_epi_plan(
            GemmCls,
            device_capacity,
            A_s,
            B_s,
            D,
            C,
            epi_values=epi_values,
            epi_key_overrides=epi_key_overrides,
            tile_M=tile_M,
            tile_N=tile_N,
            cluster_M=cluster_M,
            cluster_N=cluster_N,
            tile_K=tile_K,
            pingpong=pingpong,
            persistent=persistent,
            is_dynamic_persistent=is_dynamic_persistent,
            max_swizzle_size=max_swizzle_size,
            varlen_m=varlen_m,
            gather_A=gather_A,
            b_kn=b_kn and not swap_ab,  # slot-A relabels via a_transposed instead
            swap_ab=swap_ab,
            use_tma_gather=use_tma_gather,
            concat_layout=concat_key,
            sf_dtype=sf_dtype,
            sf_vec_size=sf_vec_size,
            sf_batched=SFA.ndim == 6 if blockscaled else True,
            a_mma_dtype=a_mma_dtype,
            b_mma_dtype=b_mma_dtype,
            post_init_attrs=post_init_attrs,
            gemm_cls_ref=self._class_ref(mint_key),
            packed_cd=packed_form,
            has_ag=ag_args is not None,
            use_epi_reduce=use_epi_reduce,
        )
        self._plan_cache[key] = plan
        if _launch:
            run_gemm_epi_plan(
                plan,
                B if swap_ab else A,
                A if swap_ab else B,
                D,
                C,
                epi_values,
                ag_args=ag_args,
                epi_reduce_args=epi_reduce_args,
                tile_count_semaphore=tile_count_semaphore,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx,
                SFA=SFA,
                SFB=SFB,
            )
        return plan

    # ── Torch-facing interface (Tier 4 surface) ──────────────────────────
    # One interface parameterized by the epilogue object (see HANDOFF Tier 4):
    # eager ``__call__`` (autotunes via quack.epi_autotune, allocates missing
    # outputs), ``plan()`` -> EpiPlan (resolve once; ``run()`` makes zero host
    # decisions). B is (k, n) logical at this surface — the torch convention —
    # with the physical layout free.

    def _lead_shape(self, A, cu_seqlens_m, A_idx):
        if cu_seqlens_m is not None:
            return ((A_idx.shape[0] if A_idx is not None else A.shape[0]),)
        return tuple(A.shape[:-1])

    def _alloc_outputs(self, out, A, B, C, store_d, out_dtype, cu_seqlens_m, A_idx):
        """Fill in the outputs the caller left out; out= buffers win."""
        import torch

        out = dict(out) if out else {}
        n = B.shape[-1]
        lead = self._lead_shape(A, cu_seqlens_m, A_idx)
        dt = out_dtype if out_dtype is not None else A.dtype
        if store_d and out.get("D") is None:
            if self.mode == "packed_cd_b16x2":
                if C is None:
                    raise ValueError("packed_cd_b16x2 requires C; D matches its shape/dtype")
                out["D"] = torch.empty_like(C)
            else:
                out["D"] = torch.empty((*lead, n), dtype=dt, device=A.device)
        n_store = n // 2 if self.mode == "acc_pair" else n
        for name in self.outputs:
            if out.get(name) is None:
                out[name] = torch.empty((*lead, n_store), dtype=dt, device=A.device)
        return out

    def _alloc_sinks(self, epi_args, lead, n, config, device):
        """Reduce partials are config-shaped scratch, not outputs: allocated
        per call (stream-safe, graph-pool friendly). A caller-provided buffer
        (same name in operands) is used as-is and returned raw."""
        import torch

        bufs = {}
        for name, op in self.sinks.items():
            if epi_args.get(name) is not None:
                continue
            if op.dim == 0:
                shape = (*lead, -(-n // config.tile_n))
            else:
                shape = (*lead[:-1], -(-lead[-1] // config.tile_m), n)
            bufs[name] = epi_args[name] = torch.empty(shape, dtype=torch.float32, device=device)
        return bufs

    def _iface_execute(self, config, dynamic_scheduler, ctx, _launch=True):
        import torch

        A, C, D = ctx["A"], ctx.get("C"), ctx.get("D")
        dyn = dynamic_scheduler or config.is_dynamic_persistent
        epi_args = dict(ctx["operands"])
        for name in self.outputs:
            epi_args[name] = ctx["out"][name]
        sink_bufs = self._alloc_sinks(epi_args, ctx["lead"], ctx["n"], config, A.device)
        semaphore = (
            torch.zeros(1, dtype=torch.int32, device=A.device)
            if dyn and get_device_capacity(A.device)[0] == 9
            else None
        )
        blockscaled = ctx.get("SFA") is not None
        plan = self.gemm(
            A,
            ctx["B_d"],
            D,
            C,
            epi_args=epi_args,
            tile_M=config.tile_m,
            tile_N=config.tile_n,
            tile_K=None if blockscaled else config.tile_k,
            cluster_M=config.cluster_m,
            cluster_N=config.cluster_n,
            pingpong=config.pingpong,
            persistent=True,
            is_dynamic_persistent=dyn,
            max_swizzle_size=config.max_swizzle_size,
            tile_count_semaphore=semaphore,
            cu_seqlens_m=ctx.get("cu_seqlens_m"),
            A_idx=ctx.get("A_idx"),
            SFA=ctx.get("SFA"),
            SFB=ctx.get("SFB"),
            bs_format_a=ctx.get("bs_format_a"),
            bs_format_b=ctx.get("bs_format_b"),
            rounding_mode=ctx["rounding_mode"],
            epi_key_overrides=ctx["epi_key_overrides"],
            b_kn=ctx["b_kn"],
            swap_ab=config.swap_ab,
            use_tma_gather=config.use_tma_gather,
            concat_layout=ctx.get("concat_layout"),
            _launch=_launch,
        )
        return config, dyn, plan, sink_bufs

    def __call__(
        self,
        A,
        B,  # (k, n) or (l, k, n) — torch convention; physical layout free
        C=None,
        *,
        out=None,  # {name: buffer} incl. "D"; missing entries are allocated
        out_dtype=None,
        store_d=True,
        config=None,  # explicit GemmConfig pins; None + tuned=True autotunes
        tuned=True,
        dynamic_scheduler=False,
        cu_seqlens_m=None,
        A_idx=None,
        SFA=None,
        SFB=None,
        bs_format_a=None,  # BlockScaledFormat names (see EpiMod.gemm)
        bs_format_b=None,
        rounding_mode=RoundingMode.RN,
        epi_key_overrides=None,
        concat_layout=None,  # tensors whose non-contiguous dim is concat [gate; up]
        **operands,  # epilogue operand tensors/scalars by fn-parameter name
    ):
        """Eager torch-facing call: resolve config (autotune via
        quack.epi_autotune on first sight of a metadata class), allocate
        missing outputs, launch. Returns {"D": ..., <declared outputs>,
        <finalized reduces>}; reduce sinks come back finalized via their op's
        ``host_finalize`` (partials are internal scratch) unless the caller
        passed the partial buffer as an operand.

        The tuned path covers the non-SR surface (see quack.epi_autotune,
        incl. varlen/gather/blockscaled/concat and dynamic_scheduler=True);
        other calls resolve with the explicit ``config=`` or the per-arch
        default.

        Under torch.compile the call records the single ``quack::gemm_epi``
        custom op (see quack.epi_torch_op); with reduce sinks the config is
        pinned there (partials must be graph-allocated at exact shapes)."""
        import torch

        if torch.compiler.is_compiling():
            if dynamic_scheduler or epi_key_overrides is not None:
                raise NotImplementedError(
                    "dynamic_scheduler/epi_key_overrides under torch.compile: not supported yet"
                )
            from quack.epi_torch_op import compile_call

            return compile_call(
                self,
                A,
                B,
                C,
                out=out,
                out_dtype=out_dtype,
                store_d=store_d,
                config=config,
                tuned=tuned,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx,
                SFA=SFA,
                SFB=SFB,
                bs_format_a=bs_format_a,
                bs_format_b=bs_format_b,
                rounding_mode=rounding_mode,
                operands=operands,
            )

        # ── Eager warm fast path: one metadata key -> captured launch recipe.
        # Everything metadata-derived (config resolution incl. the autotuned
        # winner, output recipes, sink shapes, slot order) was recorded by a
        # previous identical-metadata call; only allocation + launch remain.
        # packed_cd rides it too (the f32 recast is trace-level, cd_packed);
        # concat is excluded (its per-call B views live in mod.gemm).
        ck = None
        if concat_layout is None:
            ck = (
                tensor_key(A),
                tensor_key(B),
                tensor_key(C),
                tuple(
                    sorted(
                        (kk, tensor_key(v) if hasattr(v, "stride") else v is not None)
                        for kk, v in operands.items()
                    )
                ),
                None
                if out is None
                else tuple(sorted((kk, tensor_key(v)) for kk, v in out.items())),
                out_dtype,
                store_d,
                # Pinned configs key by identity: they are module-level
                # constants in practice; a recreated equal config just
                # re-records (correct, one extra cold pass).
                None if config is None else id(config),
                tuned,
                dynamic_scheduler,
                tensor_key(cu_seqlens_m),
                tensor_key(A_idx),
                tensor_key(SFA),
                tensor_key(SFB),
                bs_format_a,
                bs_format_b,
                rounding_mode,
                None if epi_key_overrides is None else tuple(sorted(epi_key_overrides.items())),
            )
            entry = self._call_cache.get(ck)
            if entry is not None:
                import torch as _t

                plan, recipes, sink_shapes, b_kn_c, swapped, sem_dyn = entry
                outs = dict(out) if out else {}
                for name, shape, dt in recipes:
                    if outs.get(name) is None:
                        outs[name] = _t.empty(shape, dtype=dt, device=A.device)
                epi_values = dict(operands)
                for name in self.outputs:
                    epi_values[name] = outs[name]
                sink_bufs = {}
                for name, shape in sink_shapes:
                    if epi_values.get(name) is None:
                        buf = _t.empty(shape, dtype=_t.float32, device=A.device)
                        sink_bufs[name] = epi_values[name] = buf
                B_w = B if b_kn_c else B.mT
                sem = _t.zeros(1, dtype=_t.int32, device=A.device) if sem_dyn else None
                run_gemm_epi_plan(
                    plan,
                    B_w if swapped else A,
                    A if swapped else B_w,
                    outs.get("D") if store_d else None,
                    C,
                    epi_values,
                    tile_count_semaphore=sem,
                    cu_seqlens_m=cu_seqlens_m,
                    A_idx=A_idx,
                    SFA=SFA,
                    SFB=SFB,
                )
                result = dict(outs) if store_d else {kk: v for kk, v in outs.items() if kk != "D"}
                for name, buf in sink_bufs.items():
                    finalize = getattr(self.sinks[name], "host_finalize", None)
                    result[name] = finalize(buf) if finalize is not None else buf
                return result

        varlen_m = cu_seqlens_m is not None
        # concat reads B (k, n) through per-call views, so it vetoes the b_kn
        # trace-time relabel (the interleave lives in mod.gemm).
        b_kn = get_device_capacity(A.device)[0] >= 9 and not concat_layout
        B_d = B if b_kn else B.mT
        provided_out = frozenset(k for k, v in (out or {}).items() if v is not None)
        out = self._alloc_outputs(out, A, B, C, store_d, out_dtype, cu_seqlens_m, A_idx)
        D = out.get("D") if store_d else None
        lead = self._lead_shape(A, cu_seqlens_m, A_idx)
        n = B.shape[-1]
        use_tuner = (
            tuned
            and config is None
            and rounding_mode == RoundingMode.RN
            and epi_key_overrides is None
        )
        if use_tuner:
            from quack.epi_autotune import sink_arg_shapes, tuned_mod_gemm

            epi_args = dict(operands)
            for name in self.outputs:
                epi_args[name] = out[name]
            owned_sinks = {}
            if self.sinks:
                # sink_arg_shapes walks the full config space — cache the
                # worst-case shapes per metadata (it's on the warm path).
                l = lead[0] if len(lead) == 2 else None
                shape_key = (lead[-1], n, l, str(A.device))
                cache = self.__dict__.setdefault("_sink_shape_cache", {})
                shapes = cache.get(shape_key)
                if shapes is None:
                    shapes = sink_arg_shapes(self, lead[-1], n, l=l, device=A.device)
                    cache[shape_key] = shapes
                for name, shape in shapes.items():
                    if epi_args.get(name) is None:
                        epi_args[name] = torch.empty(shape, dtype=torch.float32, device=A.device)
                        owned_sinks[name] = True
            res = tuned_mod_gemm(
                self,
                A,
                B_d,
                D,
                C,
                epi_args=epi_args,
                b_kn=b_kn,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx,
                dynamic_scheduler=dynamic_scheduler,
                SFA=SFA,
                SFB=SFB,
                bs_format_a=bs_format_a,
                bs_format_b=bs_format_b,
                concat_layout=concat_layout,
            )
            sink_bufs = {name: res.sinks[name] for name in owned_sinks}
            cfg_used, plan_used = res.config, res.plan
        else:
            ctx = dict(
                A=A,
                B_d=B_d,
                C=C,
                D=D,
                out=out,
                operands=dict(operands),
                n=n,
                lead=lead,
                b_kn=b_kn,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx,
                SFA=SFA,
                SFB=SFB,
                bs_format_a=bs_format_a,
                bs_format_b=bs_format_b,
                rounding_mode=rounding_mode,
                epi_key_overrides=epi_key_overrides,
                concat_layout=concat_layout,
            )
            if config is not None:
                cfg = config
            elif SFA is not None:
                cfg = blockscaled_default_config(A.shape[-2], n)
            else:
                cfg = default_config(A.device)
            _, _, plan_used, sink_bufs = self._iface_execute(cfg, dynamic_scheduler, ctx)
            cfg_used = cfg
        if ck is not None:
            # Record the launch recipe for identical-metadata replays. Sink
            # shapes come from the winning config's buffers (tuned sweeps
            # allocate worst-case; the recipe stores the exact live shape).
            recipes = tuple(
                (name, tuple(t.shape), t.dtype)
                for name, t in out.items()
                if name not in provided_out
            )
            # tuned sink_bufs are already the winning config's exact slices
            sink_shapes = tuple((name, tuple(b.shape)) for name, b in sink_bufs.items())
            self._call_cache[ck] = (
                plan_used,
                recipes,
                sink_shapes,
                b_kn,
                cfg_used.swap_ab,
                (dynamic_scheduler or cfg_used.is_dynamic_persistent)
                and get_device_capacity(A.device)[0] == 9,
            )
        result = dict(out) if store_d else {k: v for k, v in out.items() if k != "D"}
        for name, buf in sink_bufs.items():
            finalize = getattr(self.sinks[name], "host_finalize", None)
            result[name] = finalize(buf) if finalize is not None else buf
        return result

    def plan(
        self,
        A,
        B,  # (k, n) logical, as in __call__
        C=None,
        *,
        out,  # REQUIRED: buffers for D + every declared output (their
        #        metadata selects the kernel); no "D" entry = no D store
        config=None,  # explicit GemmConfig; None = per-arch default. No
        #               autotuning here (autotune via an eager call, or pass
        #               the tuned config) — plan() never launches.
        dynamic_scheduler=False,
        cu_seqlens_m=None,
        A_idx=None,
        SFA=None,
        SFB=None,
        bs_format_a=None,  # BlockScaledFormat names (see EpiMod.gemm)
        bs_format_b=None,
        rounding_mode=RoundingMode.RN,
        epi_key_overrides=None,
        **operands,
    ):
        """Resolve (and compile on a cold cache) WITHOUT launching; returns an
        :class:`EpiPlan` whose ``run()`` makes zero host decisions. Default
        reduce scratch is allocated here and attached to the plan (pass fresh
        ``scratch=`` to run() for concurrent streams)."""
        varlen_m = cu_seqlens_m is not None
        b_kn = get_device_capacity(A.device)[0] >= 9
        B_d = B if b_kn else B.mT
        cfg = config if config is not None else default_config(A.device)
        dyn = dynamic_scheduler or cfg.is_dynamic_persistent
        out = dict(out)
        D = out.get("D")
        for name in self.outputs:
            if out.get(name) is None:
                raise ValueError(f"plan() requires a buffer for output {name!r}")
        ctx = dict(
            A=A,
            B_d=B_d,
            C=C,
            D=D,
            out=out,
            operands=dict(operands),
            n=B.shape[-1],
            lead=self._lead_shape(A, cu_seqlens_m, A_idx),
            b_kn=b_kn,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=A_idx,
            SFA=SFA,
            SFB=SFB,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            rounding_mode=rounding_mode,
            epi_key_overrides=epi_key_overrides,
        )
        _, _, gemm_plan, sink_bufs = self._iface_execute(cfg, dyn, ctx, _launch=False)
        return EpiPlan(
            gemm_plan=gemm_plan,
            config=cfg,
            out_names=tuple(self.outputs),
            b_kn=b_kn,
            swapped=cfg.swap_ab,
            scratch={
                **sink_bufs,
                **{k: v for k, v in ctx["operands"].items() if k in self.sinks},
            },
        )


def _pinned_visit_kind(op):
    if op.fn_port == "apply":
        return "apply"
    if op.fn_port == "value":
        # Custom value-source op: its begin_loop fragment must be elementwise
        # congruent with tRS_rD and DENSE (the vectorizer rejects zero-stride
        # loop loads); it is indexed like a tile fragment in every mode.
        return "value"
    if isinstance(op, RowVecLoad):
        return "row"
    if isinstance(op, ColVecLoad):
        return "col"
    if isinstance(op, TileLoad):
        return "tile"
    if isinstance(op, Scalar):
        return "scalar"
    raise ValueError(f"cannot use {type(op).__name__} as a fn-frontend operand (write a mixin)")


def gemm_epilogue(
    outputs=(),
    ops=None,
    reduces=None,
    mode=None,
    paired=(),
    outs=None,
    prepass=None,
    prepass_outs=(),
    extra_ops=(),
    vectorize=None,
):
    """Decorator: turn an elementwise fn into a fused GEMM epilogue. See module
    docstring for the contract. ``ops`` pins operand names to explicit EpiOp
    instances when shape inference is ambiguous. ``reduces`` declares reduce
    outputs ({name: ColVecReduce(name) or RowVecReduce(name)}): the fn returns
    the per-element value to accumulate under that name, the buffer arrives in
    ``epi_args`` shaped (l, m, n_tiles) for col / (l, m_tiles, n) for row.

    ``mode='acc_pair'`` is expressed in the fn body with ``unpack``/``pack``
    (see :class:`Pair`): gated is ``gate, up = unpack(acc)`` with a
    half-of-GEMM-N aux buffer
    (per-pair aux is 16-bit n-major; interleave gate/up along N in B exactly
    as with the hand-written kernels; row/tile/c operands arrive paired, col
    operands as one scalar since they broadcast along N). Use
    ``mode='packed_cd_b16x2'`` for dgated:
    ``x, y = unpack(c)`` + ``"D": pack(dx, dy)`` with C/D passed as their
    natural 16-bit n-major tensors at twice GEMM-N.

    ``vectorize=False`` keeps the SM100 fn-loop vectorizer off for this
    epilogue (no effect on other archs — only SM100 vectorizes this loop).
    Escape hatch for the upstream DSL vectorizer's crash bugs (fused
    two-result ``cute.math.sincos``; an arith-computed value reused in both
    ``pack()`` lanes — nondeterministic segfault/double-free/compile hang).
    It disables f32x2 packing of the fn math: free while the epilogue hides
    under the mainloop (rope_posfreq B300 K=4096: novec +0.01% vs vectorized
    +0.46% over bias-only), expensive once exposed (K=512: +72% vs +51% — the
    unpacked stream goes issue-bound). Escape hatch, not a tuning knob.
    Changing it changes the kernel cache key."""

    def wrap(fn):
        return EpiMod(
            fn,
            outputs=outputs,
            ops=ops,
            reduces=reduces,
            mode=mode,
            paired=paired,
            outs=outs,
            prepass=prepass,
            prepass_outs=prepass_outs,
            extra_ops=extra_ops,
            vectorize=vectorize,
        )

    return wrap


class EpiPlan:
    """Prepared launch for one metadata class (the decode / CUDA-graph entry
    point). ``run()`` performs zero host decisions by construction: no key, no
    compile-cache probe, no allocation — only the B relabel the compiled
    signature demands (off SM90+/varlen) and the launch. Tensors must match
    the metadata plan() saw; that promise is the caller's (an interface layer
    holding this plan keys for it)."""

    __slots__ = ("gemm_plan", "config", "out_names", "b_kn", "scratch", "swapped")

    def __init__(self, *, gemm_plan, config, out_names, b_kn, scratch, swapped=False):
        self.gemm_plan = gemm_plan
        self.config = config
        self.out_names = out_names
        self.b_kn = b_kn
        self.swapped = swapped
        # Default reduce partial buffers, allocated at plan(). Shared across
        # run() calls on one stream; pass scratch= for concurrent streams.
        self.scratch = scratch

    def run(
        self,
        A,
        B,  # (k, n) logical, as given to plan()
        C=None,
        *,
        out,
        scratch=None,
        tile_count_semaphore=None,
        cu_seqlens_m=None,
        A_idx=None,
        SFA=None,
        SFB=None,
        **operands,
    ):
        D = out.get("D")
        if not self.b_kn:
            B = B.mT
        epi_values = dict(self.scratch if scratch is None else scratch)
        epi_values.update(operands)
        for name in self.out_names:
            v = out.get(name)
            if v is not None:
                epi_values[name] = v
        run_gemm_epi_plan(
            self.gemm_plan,
            B if self.swapped else A,
            A if self.swapped else B,
            D,
            C,
            epi_values,
            tile_count_semaphore=tile_count_semaphore,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=A_idx,
            SFA=SFA,
            SFB=SFB,
        )


class StaticEpi:
    """Rung-3 escape hatch: the same plan/run interface for a hand-written
    GEMM class (custom EpiOps, dataflow the fn contract can't express). Power
    API: no operand inference and no allocation — B arrives dispatch-shaped
    (n, k) unless b_kn, epi_args are explicit, outputs are epi_args entries.
    Requires every op with a host argument to implement the host schema trio
    (host_arg_key / host_fake_arg / host_call_arg)."""

    def __init__(self, GemmCls):
        self.GemmCls = GemmCls

    def plan(
        self,
        A,
        B,
        D,
        C=None,
        *,
        epi_args,
        config=None,
        dynamic_scheduler=False,
        b_kn=False,
        varlen_m=False,
        gather_A=False,
        epi_key_overrides=None,
    ):
        cfg = config if config is not None else default_config(A.device)
        dyn = dynamic_scheduler or cfg.is_dynamic_persistent
        gemm_plan = build_gemm_epi_plan(
            self.GemmCls,
            get_device_capacity(A.device),
            A,
            B,
            D,
            C,
            epi_values=epi_args,
            epi_key_overrides=epi_key_overrides,
            tile_M=cfg.tile_m,
            tile_N=cfg.tile_n,
            tile_K=cfg.tile_k,
            cluster_M=cfg.cluster_m,
            cluster_N=cfg.cluster_n,
            pingpong=cfg.pingpong,
            persistent=True,
            is_dynamic_persistent=dyn,
            max_swizzle_size=cfg.max_swizzle_size,
            varlen_m=varlen_m,
            gather_A=gather_A,
            b_kn=b_kn,
        )
        return EpiPlan(
            gemm_plan=gemm_plan,
            config=cfg,
            out_names=(),
            b_kn=True,  # B is already dispatch-shaped: run() must not relabel
            scratch={},
        )


def epilogue_from_class(GemmCls) -> StaticEpi:
    """Wrap a hand-written epilogue GEMM class in the plan/run interface."""
    return StaticEpi(GemmCls)
