# Epilogue / operand-transform stack: layer map and design

Rewritten 2026-11-28 to describe the stack as it is. It started as the plan for
the 2026-07-29 reorg (placement, duplication, import direction); the identity
and transport model changed in 2026-11. Both are summarized under History.

## What specializes a GEMM kernel

A GEMM kernel class is fixed code plus two pluggable pieces, both authored as
plain Python functions:

* an **epilogue** (`EpiMod`, from `@gemm_epilogue`): a fn over the accumulator,
  lowered onto `EpiOp`s;
* optionally an **A-operand transform** (from `@a_transform`, a decode format,
  or `dropout_a`): what happens to A between memory and the MMA.

Two rules decide every placement question:

* **A feature is one object.** Device behavior, host schema and cache identity
  live on the class that implements the feature (`EpiOp`, an operand kind, a
  decode format). The framework composes objects; it never switches on them.
* **The framework never enumerates flavours; the handle answers.** Where the
  generic layer would write `if owned_fmt ... elif mod.args ...`, it calls a
  method on the transform mod. Where three call surfaces would restate a
  shape rule, they call a method on the op.

What must stay true:

* A new epilogue is ONE file (`epilogue/scaled_exp.py` touched nothing else); a
  new packed format is one class or one `@a_transform(packed=...)` fn.
* One torch custom op, `quack::gemm_epi`, for every epilogue.
* The semantic fingerprint is fail-closed: a capture it cannot key raises at
  decoration time.
* The warm-path caches (`_call_cache`, `_plan_cache`): host launch overhead is
  a measured product constraint (~3.5 us floor). Nothing may add work there.

## Layout

    quack/epilogue/
        ops.py           EpiOp vocabulary: resource lifecycle, value ports (fn_port),
                         host schema (host_arg_key / host_fake_arg / host_call_arg),
                         cache identity (cache_key)
        mixin.py         ComposableEpiMixin: composes _epi_ops into the kernel hooks
        visit.py         the traced per-element visit loop (device half of the frontend)
        frontend.py      @gemm_epilogue, EpiMod: kernel-class minting and the three
                         call surfaces (gemm / __call__ / plan)
        math.py          value vocabulary (Pair, F2, pack/unpack, pexp)
        library.py       ready mods and the generated-fn factories
        rotary.py, scaled_exp.py, head_rmsnorm.py, quantize_out.py   domain content

    quack/operand_transform/
        transform.py     KERNEL side: TransformA and its subclasses
                         (TransformAValue, TransformAW4, TransformADropout)
        kinds.py         KERNEL side: one class per runtime-operand kind
        mod.py           TransformModBase: the transform_a= handle protocol
        formats/         DecodeFormat (a transform mod) + the W4_FORMATS registry; qtip.py
        frontend.py      @a_transform, ATransformMod (value fns), FnFormat (packed
                         fns), dropout_a
        host.py          as_transform_mod, operand bundles and fakes, W4 config pickers
        rng.py

    quack/gemm_runtime/
        identity.py      LEAF: semantic fingerprints, GemmMod, MODS_BY_DIGEST,
                         module_locator
        host.py          _compile_gemm_epi (jit-cached), build_gemm_epi_plan,
                         run_gemm_epi_plan
        torch_op.py      quack::gemm_epi / quack::gemm_epi_f, compile_call
        autotune.py      tuned_mod_gemm

`gemm_tvm_ffi_utils.py` stays at top level: it serves the whole GEMM family,
including `quack/gemm.py`. `gemm_w4.py` is thin sugar over `EpiMod.gemm`.

## Identity and transport

An epilogue and a transform are the same kind of thing outside their own
process: a **mod**, traceable code plus static config. `GemmMod`
(`gemm_runtime/identity.py`) is their common base. Two questions about a mod
are separate.

**What is it?** Its `semantic_digest`: a hash of the function as written, what
it captures, and its static config. It depends on content only, not on which
object it is, how it was reached, or what the process has traced so far. Same
digest, same kernel. Mods are equal, and hash, by digest. A subclass calls
`_identify(key)` once at the end of `__init__`.

**How does another process get it?** It pickles. A mod is traceable code, and
code can only be moved by importing it or sending it; nothing is rebuilt from
a description. Per-process caches are listed in `_PROCESS_LOCAL_CACHES` and
emptied on pickling.

There are three boundaries, and each needs one thing:

| boundary | needs | mechanism |
|---|---|---|
| jit cache key, in memory and on disk | identity | the digest (`__quack_cache_key__`, read by `quack/cache/jit.py`) |
| async-compile worker (another process, now) | the object | cloudpickle: the compile function's arguments go by value |
| torch.compile graph (strings only, maybe a later process) | a name that resolves | `MODS_BY_DIGEST`, plus `module_locator` for a fresh process |

So a mod is passed to the compile function as itself:

    _compile_gemm_epi(gemm_cls, ..., transform_a=mod)
        gemm_cls: a GEMM class written in a module, or (epi_mod, mint_key)

A minted kernel class exists only in the process that minted it, so it never
crosses a boundary: `(epi_mod, mint_key)` does, and the class is minted again
where the compile runs.

Consequences worth knowing:

* A by-value copy carries module-level functions and classes BY NAME; the
  worker imports them, source and all. Code defined inside a function travels
  as bytecode with no file, and the DSL cannot trace it. Hence the first rule
  of "Staying async-compilable" in `quack/epilogue/frontend.py`: anything
  under `@cute.jit` lives at module level. The epilogue fn itself is exempt
  (it is called as plain Python).
* The worker trusts the copy. It does not recompute the digest, and module
  state that some other module changes at import time is not replayed there.
* A decorated helper (`@cute.jit`, `@dsl_user_op`) counts by its own source
  only. What it reads from its module is covered by the disk cache's source
  fingerprint of the quack package (downstream: `quack.cache.EXTRA_SOURCE_DIRS`),
  not by the digest. See `function_semantic_key`.
* The in-memory jit cache compares plain arguments with `==`; the disk key
  pickles them. An IntEnum and its int are one entry in memory and two on
  disk, so plan builders pass one spelling (`rounding_mode` is normalized).
* `--async-compile-strict` fails a test run in which the active pool did not
  compile a kernel it should have; the summary names the kernel, test and
  cause.

## Protocols

### Transform mods (`operand_transform/mod.py`)

Every `transform_a=` handle is a `TransformModBase` (a `GemmMod`). The generic
layers normalize with `as_transform_mod` (a W4 registry name is looked up; a
mod is returned as it is) and then only call methods:

* `__call__(gemm)` -> the kernel-side `TransformA`
* `owned_fmt` -> the DecodeFormat when the transform owns A's storage layout,
  else None; `needs_operands` -> True when calls carry runtime operand tensors
* `resolve_operands(A, B, transform_sf, tile_m, tile_k) -> ResolvedOperands`:
  what crosses in the kernel's A and B slots, the padded N, the resolved tile_K
* `bundle` / `fake_bundle`, `fake_operands`, `compile_dims`: the runtime and
  trace-time operand views and the static geometry
* `default_config(A, B)`, `config_ok(cfg)`, `padded_n(B)`, `plan_key()`

The layout-owning vs value branch is written once, in these methods.

The mods:

* `ATransformMod`: a value fn (`@a_transform(vec_size=..., args=..., consts=...)`).
* `DecodeFormat` and its subclasses: packed-weight formats. A format IS the
  mod; `W4_FORMATS` maps names to them. `@a_transform(packed=PackedInput(...))`
  returns one (`FnFormat`). A subclass with its own `__init__` sets its
  attributes, then calls `super().__init__()`, which fingerprints the class,
  the geometry, the public instance attributes and the device hooks.
* `DropoutAMod` (`dropout_a(p)`).

A packed format owns A's layout, and that changes the call contract: the A and
B slots flip, D is caller-oriented, the geometry is static, tile_K is fixed.
The `owned_fmt` checks in the host layer express that real difference.

### Operand kinds (`operand_transform/kinds.py`)

One object per runtime-operand kind in `ARG_KINDS`, owning the device staging
(`device_arg`) and the host side (`host_view`, `host_fake`), with the strip
geometry computed once (`strip_geometry`). `fn_facing` says whether the kind
may appear in `@a_transform(args=...)`; `seed_i64x2` (dropout) may not.

### Sink shapes

`EpiOp.sink_alloc_shape(lead, n, tile_m, tile_n, num_seqs=None)` is the only
statement of the partial-buffer rule; the frontend, the torch op and autotune
all call it.

## Import-graph invariants

* `gemm_runtime/identity.py` imports nothing from `quack`. Anyone may import
  it; the `quack.gemm_runtime` package root is empty so that stays cheap.
* `operand_transform/mod.py` imports only the identity leaf at module level.
  `formats/` and `frontend.py` both sit above it.
* `operand_transform/transform.py` and `kinds.py` are imported by the kernel
  classes. `transform.py` must not import `formats/` at module level
  (`GemmSm90` would drag in the whole format registry and `qtip.py`); the
  format is resolved by the handle and passed in.
* The `quack.epilogue` and `quack.operand_transform` package roots are
  PEP-562 lazy; kernel classes import submodules directly.
* Nothing under `epilogue/`, `operand_transform/` or `gemm_runtime/` imports
  variant sugar (`gemm_w4.py`) or `gemm_interface.py`.

## Open items

* Migrate `gemm_interface.py`'s hand-written custom ops onto `quack::gemm_epi`;
  unify the spellings of eager-bypass.
* Merge `gemm_runtime.autotune._prune_for_mod` with
  `gemm_interface.prune_invalid_gemm_configs` into one predicate library in
  `gemm_config` (the blockscaled part, `blockscaled_config_ok`, is shared).
* `quack/gemm.py`'s parallel plain-GEMM stack; `rmsnorm.py`.
* Under torch.compile, `transform_a="int4"` (a bare name) is still rejected in
  favour of the format object. A name now resolves with a plain lookup, so
  the restriction has no reason left; lifting it is untested under Dynamo.
* One key function for the in-memory and the disk jit cache would close the
  equal-but-pickles-differently class for good. It puts a hash on the warm
  path, so it needs a measurement first.

## History

**2026-07-29, reorg.** The stack had grown in two generations (epi_ops /
epi_composable, then the fn frontend) plus the transform port, with every
concept in the file that was convenient at the time: a 2568-line
`gemm_epilogue.py` that was four modules, rules restated up to seven times
(sink shapes, the transform-flavour branch, padded N), host and device kind
registries that had drifted, and ~25 lazy imports that existed only to break
cycles. The reorg gave each concern one module and a one-way import order.

| old | new |
|---|---|
| `quack/epi_ops.py`, `quack/epi_utils.py` | `quack/epilogue/ops.py` |
| `quack/epi_composable.py` | `quack/epilogue/mixin.py` |
| `quack/gemm_epilogue.py` | `epilogue/frontend.py` + `visit.py` + `math.py`; fingerprinter -> `gemm_runtime/identity.py` |
| `quack/epilogues.py` | `quack/epilogue/library.py` |
| `quack/gemm_host.py` | `quack/gemm_runtime/host.py` |
| `quack/epi_torch_op.py` | `quack/gemm_runtime/torch_op.py` |
| `quack/epi_autotune.py` | `quack/gemm_runtime/autotune.py` |
| `quack/operand_transform/transform_a.py` | `quack/operand_transform/transform.py` |
| `quack/blockscaled/decode_formats.py`, `qtip.py` | `quack/operand_transform/formats/` |
| `gemm_w4.py` config pickers | `quack/operand_transform/host.py` |

Follow-ups that landed in the same period: `fn_port` on every op (the
frontend no longer isinstance-dispatches), `gemm_w4a16` onto `EpiMod.gemm`,
transform-aware autotune (`TransformModBase.config_ok`).

**2026-11-28, identity and transport.** Until then a mod crossed the jit-cache
boundary as a picklable reference: `GemmClassRef` / `TransformARef`, resolved
by import when the mod was bound to a module global, by registry name for W4
formats, and otherwise through a local registry filled by a cloudpickle
payload that travelled beside the arguments. cloudpickle was optional, so
most epilogue kernels silently compiled in-process when it was missing, and
the cache key depended on how a mod was found. With cloudpickle a declared
dependency, all of that was replaced by the model above: mods are arguments,
keyed by digest, shipped by value. Removed: both ref classes, the two local
registries, the payload channel and its installers, `PackedFormatMod`,
`w4_transform`, `as_decode_format`. Decode formats became mods. The digest
became the whole identity of a mod, which exposed, and fixed, several places
where it was too coarse or depended on process state (see the commit).
`AI/` scratch scripts pin older paths and names; fix on next use.
