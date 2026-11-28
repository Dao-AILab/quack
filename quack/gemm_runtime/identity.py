# Copyright (c) 2026, Han Guo, Tri Dao.
"""Identity layer for epilogue and transform mods — a LEAF module (imports
nothing from quack at module level; anyone may import it):

* fail-closed semantic fingerprinting (:func:`function_semantic_key` /
  :func:`semantic_value_key`) — the compile-cache identity of fn-authored
  mods;
* :class:`GemmMod` — what an epilogue and an A-operand transform have in
  common outside their own process: an identity and a way to travel;
* ``MODS_BY_DIGEST`` + :func:`module_locator` — digest -> mod and its
  import anchor, for the torch custom op: a compiled graph can only carry
  strings.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import functools
import hashlib
import inspect
import linecache
import os
import sys
import sysconfig
import types
from typing import Optional


def semantic_value_key(value, seen):
    """Fail-closed semantic fingerprint of a value an epilogue/transform fn
    depends on.

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
            semantic_value_key(qsk(value), seen),
        )
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__module__, type(value).__qualname__, value.value)
    if isinstance(value, tuple):
        return ("tuple", tuple(semantic_value_key(v, seen) for v in value))
    if isinstance(value, list):
        return ("list", tuple(semantic_value_key(v, seen) for v in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(sorted((repr(k), semantic_value_key(v, seen)) for k, v in value.items())),
        )
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(repr(semantic_value_key(v, seen)) for v in value)))
    if isinstance(value, types.ModuleType):
        return ("module", value.__name__)
    if inspect.isfunction(value):
        return function_semantic_key(value, seen)
    if inspect.ismethod(value):
        return (
            "method",
            function_semantic_key(value.__func__, seen),
            semantic_value_key(value.__self__, seen),
        )
    if isinstance(value, (types.BuiltinFunctionType, types.MethodWrapperType)):
        return ("builtin", getattr(value, "__module__", None), value.__qualname__)
    wrapped = getattr(value, "__wrapped__", None)
    if callable(value) and wrapped is not None:
        # Decorator wrappers (functools.wraps chains: lru_cache, dsl_user_op,
        # cute.jit): the semantics live in the wrapped function.
        return ("wrapped", semantic_value_key(wrapped, seen))
    if isinstance(value, functools.partial):
        return (
            "partial",
            semantic_value_key(value.func, seen),
            semantic_value_key(value.args, seen),
            semantic_value_key(value.keywords, seen),
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
                (f.name, semantic_value_key(getattr(value, f.name), seen))
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


# Packages whose functions fingerprint by name: the toolchain the traced code
# is written against, pinned by the environment (the disk cache stamps the
# cutlass and tvm_ffi versions).
_EXTERN_PACKAGES = frozenset({"cutlass", "tvm_ffi", "torch", "numpy", "cuda"})


def _is_extern_function(fn) -> bool:
    """True for functions of the standard library and of the toolchain
    (:data:`_EXTERN_PACKAGES`). Like classes and modules, they fingerprint by
    qualname only: their source is pinned by the environment, and recursing
    into them would pull runtime-MUTABLE library globals into the digest —
    e.g. any fn touching cutlass's dsl_user_op machinery reaches
    cutlass._mlir_helpers.op, which lazily materializes _DSL_PACKAGE_ROOT(S)
    on the first traced op, so the digest would depend on whether this process
    compiled anything yet.

    Everything else is fingerprinted by content, installed or not: a
    downstream package in site-packages is user code, and two closures of one
    of its factories are two different epilogues."""
    # Where the function was WRITTEN: a user function under @cute.jit or
    # @dsl_user_op is a wrapper whose own code lives in cutlass, and judging
    # by that would fingerprint every such function by name alone.
    inner = inspect.unwrap(fn)
    module = getattr(inner, "__module__", None) or ""
    if module.partition(".")[0] in _EXTERN_PACKAGES:
        return True
    filename = getattr(getattr(inner, "__code__", None), "co_filename", "")
    installed = f"{os.sep}site-packages{os.sep}" in filename or (
        f"{os.sep}dist-packages{os.sep}" in filename
    )
    return filename.startswith(_stdlib_root()) and not installed


@functools.lru_cache(maxsize=256)
def _definition_spans(source: str) -> dict:
    """(name, line) -> (first line, last line) of every ``def`` in a file's
    source, entered under both its ``def`` line and its first decorator line."""
    spans = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            first = min([d.lineno for d in node.decorator_list] + [node.lineno])
            spans[node.name, node.lineno] = spans[node.name, first] = (first, node.end_lineno)
    return spans


def _definition_source(fn) -> Optional[str]:
    """``fn``'s definition as written in its file, decorators included; None
    when the file or the ``def`` cannot be found.

    ``inspect.getsource`` is not a stable answer for ``@cute.jit`` functions:
    once the DSL has traced one it swaps in a rewritten function whose first
    line is the ``def``, and the decorator lines drop out of the result. A mod
    built after such a trace then got a different digest from one built
    before it, for the same source: one kernel under two cache keys, and a
    mismatch whenever another process rebuilt the mod. So locate the ``def``
    in the file by either line and always return the decorated form."""
    inner = inspect.unwrap(fn)
    code = getattr(inner, "__code__", None)
    if code is None:
        return None
    linecache.checkcache(code.co_filename)  # a file edited and reloaded in this process
    lines = linecache.getlines(code.co_filename, getattr(inner, "__globals__", None))
    try:
        span = _definition_spans("".join(lines)).get((code.co_name, code.co_firstlineno))
    except SyntaxError:
        return None
    return None if span is None else "".join(lines[span[0] - 1 : span[1]])


def function_semantic_key(fn, seen=None):
    """Fingerprint source plus the globals/closures that can change its math.

    A plain function is followed all the way: its source, its defaults, and
    every global and closure cell it names, recursively.

    A DECORATED function (``@cute.jit``, ``@dsl_user_op``, anything built
    with ``functools.wraps``) is fingerprinted by its own source only. What
    it reads from its module — a constant, another helper it calls — is NOT
    followed: the DSL rewrites a traced function in place, so the names it
    references are not the same before and after a trace, and a key built
    from them would depend on what the process has compiled so far. Such
    code is covered by the disk cache's source fingerprint instead, which
    hashes every file of the quack package and of each directory in
    ``quack.cache.EXTRA_SOURCE_DIRS``. So inside quack nothing is missed; a
    downstream project either lists its source directory there, or keeps its
    ``@cute.jit`` helpers self-contained and passes what varies through the
    epilogue fn (a plain function, followed in full)."""
    seen = set() if seen is None else seen
    ident = (fn.__module__, fn.__qualname__)
    # Guard against cycles by OBJECT: two closures of one factory share a
    # qualname and differ in what they captured. A decorated function is one
    # function however many wrapper layers it has, so mark the whole chain:
    # the layers hold each other in their closures.
    if ("id", id(fn)) in seen:
        return ("function_ref", *ident)
    layer = fn
    for _ in range(32):
        seen.add(("id", id(layer)))
        layer = getattr(layer, "__wrapped__", None)
        if layer is None:
            break
    if _is_extern_function(fn):
        return ("extern_function", *ident)
    try:
        source = (_definition_source(fn) or inspect.getsource(fn)).encode()
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
        (name, semantic_value_key(value, seen))
        for name, value in sorted(referenced.items())
        if not name.startswith("__")
    )
    return (
        "function",
        *ident,
        hashlib.sha256(source).hexdigest(),
        semantic_value_key(fn.__defaults__, seen),
        semantic_value_key(fn.__kwdefaults__, seen),
        deps,
    )


def module_locator(obj, fn) -> Optional[tuple]:
    """(module, global_name) if ``obj`` is reachable by import in a fresh
    process — a torch.compile graph in another process re-resolves
    digest-keyed objects that way. ``fn`` anchors the search (the authored
    function whose ``__module__``/``__name__`` name the natural home);
    returns None for ``__main__`` / notebook objects and objects never bound
    to a module global.

    A binding found here may not exist in a fresh process: a module global
    bound at CALL time (e.g. a test helper's ``global`` trick) satisfies this
    scan but is never replayed by a fresh import. That can't be detected on
    the fn object (a ``global``-declared def even strips the ``<locals>``
    qualname marker), so the torch op checks the digest of what it imports."""
    module_name = getattr(fn, "__module__", None)
    if module_name is None or module_name == "__main__":
        return None
    module = sys.modules.get(module_name)
    if module is None:
        return None
    preferred = getattr(fn, "__name__", None)
    if preferred and getattr(module, preferred, None) is obj:
        return module_name, preferred
    names = sorted(name for name, value in vars(module).items() if value is obj)
    if not names:
        return None
    return module_name, names[0]


# digest -> mod, for the one boundary that can only carry a string: the
# quack::gemm_epi torch custom op (quack.gemm_runtime.torch_op) names its
# epilogue and transform by digest. Filled at CONSTRUCTION (GemmMod._identify),
# never inside compile_call: Dynamo buffers global dict mutations as deferred
# side effects, so a trace-time write would be invisible to the fake-tensor
# pass that resolves the digest. Same-digest reconstruction overwrites in place.
MODS_BY_DIGEST: dict[str, object] = {}


class GemmMod:
    """Something that specializes a GEMM kernel at trace time: an epilogue
    (EpiMod) or an A-operand transform. It is traceable code plus static
    config. Outside its own process two things about it matter, and they are
    separate:

    * WHAT it is: ``semantic_digest``, a hash of its content and of nothing
      else (not which object it is, how it was reached, or what this process
      has traced so far). Same digest, same kernel: the jit cache keys on it,
      and two mods with one digest are equal.
    * GETTING it somewhere else: it pickles. A mod is passed to the compile
      function as itself, and an async-compile worker receives it by value
      (cloudpickle), wherever it was defined. Nothing is rebuilt from a
      description. The copy carries module-level functions and classes BY
      NAME and the worker imports them, so it trusts that they are the same
      there: module state that some other module changes at import time is
      not replayed, and the worker does not recompute the digest.

    A subclass calls :meth:`_identify` once, at the end of ``__init__``, with
    a key covering everything the traced kernel depends on (build it with
    :func:`function_semantic_key` / :func:`semantic_value_key`), and lists
    its per-process caches in ``_PROCESS_LOCAL_CACHES``."""

    semantic_key: object
    semantic_digest: str
    # Attributes holding state that only means something in this process
    # (minted classes, launch plans, compiled kernels): dicts, emptied when
    # the mod is pickled. Compiled kernels do not pickle, so a cache left out
    # of this tuple makes the mod unshippable to async-compile workers.
    _PROCESS_LOCAL_CACHES: tuple = ()

    def _identify(self, semantic_key) -> None:
        self.semantic_key = semantic_key
        self.semantic_digest = hashlib.sha256(repr(semantic_key).encode()).hexdigest()
        MODS_BY_DIGEST[self.semantic_digest] = self

    def __quack_semantic_key__(self):
        """A mod captured by another mod's fn counts as its digest."""
        return self.semantic_digest

    def __quack_cache_key__(self):
        """What stands for this mod in a jit-cache disk key (quack.cache.jit)."""
        return ("quack.GemmMod", self.semantic_digest)

    def __eq__(self, other):
        return isinstance(other, GemmMod) and self.semantic_digest == other.semantic_digest

    def __hash__(self):
        return hash(self.semantic_digest)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update((name, {}) for name in self._PROCESS_LOCAL_CACHES)
        return state

    def _module_locator(self):
        """(module, global_name) if this mod is reachable by import in a fresh
        process (a torch.compile graph in another process re-resolves it that
        way), else None: no authored fn, defined in ``__main__`` (scripts,
        notebooks), or never bound to a module global."""
        fn = getattr(self, "fn", None)
        return None if fn is None else module_locator(self, fn)
