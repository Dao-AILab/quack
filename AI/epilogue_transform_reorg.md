# Epilogue / operand-transform reorg

2026-07-29. Design for reorganizing the GEMM epilogue and operand-transform (A)
stacks so each concern lives in exactly one module, `quack/epilogue/` and
`quack/operand_transform/` expose "just the right API", and the shared host
plumbing (plan/compile, torch op, autotune, identity) becomes a package of its
own. Companion to the epi_ops fn-frontend design (see `quack/gemm_epilogue.py`
module docstring) — the *concepts* are unchanged; this is about placement,
duplication, and import direction.

## Diagnosis (what the accidents of history are)

The stack grew in two generations (epi_ops/epi_composable 2026-03, the fn
frontend 2026-07) plus a transform port (2026-07, from the transformA branch).
Every concept landed in the file that was convenient at the time:

1. **`gemm_epilogue.py` (2568 L) is four modules**: value vocabulary
   (`Pair`/`F2`), the semantic fingerprinter, the device-side traced visit loop
   (`_EpiModMixinBase`), and the ~1400-line `EpiMod` host orchestrator with
   three overlapping call surfaces (`gemm` / `__call__` / `plan`).
2. **Identity machinery duplicated wholesale** between epilogues and
   transforms: `GemmClassRef` + `_LOCAL_EPI_MODS` + `install_epi_mod_payload`
   (gemm_host.py:64-169) vs `TransformARef` + `_LOCAL_TRANSFORM_MODS` +
   `install_transform_mod_payload` (operand_transform/host.py:43-124) — same
   cloudpickle side-channel, ~80 duplicated lines, opposite eviction policies.
   `_module_locator` is line-for-line identical in `EpiMod` and
   `ATransformMod`, and missing from `PackedFormatMod`/`DropoutAMod` (silently
   degrading their cross-process torch.compile portability). The
   `TORCH_OP_{EPI,TRANSFORM}_MODS` registries are declared in `gemm_host`,
   written in `gemm_epilogue`/`operand_transform.frontend`, read in
   `epi_torch_op` (via a `noqa: E402` mid-file import).
3. **Rules restated instead of single-sourced.** The sink partial-buffer shape
   rule (`dim==0 -> (*lead, cdiv(n, tile_n))`) exists ~7x across
   `gemm_epilogue` (validation + `_alloc_sinks`), `epi_torch_op` (2x),
   `epi_autotune` (3x). The "which transform flavour" branch
   (`owned_fmt` / `args` / plain) is written out 5x (gemm_host x2,
   EpiMod.gemm/__call__/plan). "blob rows x 64 = padded N" appears 7x. The
   `tile_k or 64` fallback appears 3x. Strip geometry has an admitted host
   mirror (`host._strip_dims` vs `transform_a._strip_geometry`), and the kind
   registries have drifted: `seed_i64x2` exists only host-side
   (`_ARG_KIND_HOST`), so the frontend rejects an `args=` declaration the host
   layer would happily build.
4. **Layering violations both directions**: `epi_ops.py` (device vocabulary)
   does 13 lazy imports of host code, including `gemm_tvm_ffi_utils` from
   inside `TileStore.host_fake_arg`; `gemm_host` reaches back into `epi_ops`;
   `gemm_epilogue._default_config` imports *up* into `gemm_w4` for the W4
   config pickers; importing `GemmSm90` drags in the whole W4 format registry
   plus `qtip.py` because `transform_a.py` imports `decode_formats` at module
   level. ~25 lazy imports exist purely to break cycles.
5. **Naming/content confusion**: `epilogues.py` vs `epilogue/` split on no
   rule (`epilogues.rope_epi` duplicates `rotary.rope_table_epi`; `pexp` vs
   `pexp2`); `blockscaled/operand.py` (SM100 blockscaled operand container)
   shares zero code with `operand_transform/` yet sits under a confusable
   name; `decode_formats.py` is consumed only by `operand_transform` +
   `gemm_w4` but lives in `blockscaled/`.
6. **Two dispatch mechanisms for one question**: `fn_port` on 3 op families vs
   `isinstance` in `_pinned_visit_kind` for the other 4. (Out of scope here —
   see Later campaigns.)

What is already right and must be preserved:

* A new epilogue is ONE file (`epilogue/scaled_exp.py` touched nothing else);
  a new packed format via `PackedInput` is one file. This property comes from
  the `EpiOp` colocation of device lifecycle + value port + host schema +
  cache identity on one class. The colocation is not the bug; the placement of
  the shared helpers it needs is.
* The single `quack::gemm_epi` custom op with digest resolution.
* The fail-closed semantic fingerprint and `GemmClassRef` minting design.
* The warm-path caches (`_call_cache`, `_plan_cache`): host launch overhead is
  a measured product constraint (~3.5us floor); no reorg may add work there.

## First principles

Seven concerns, one module each, one-way import order:

    content  ->  frontend (authoring/minting)  ->  host plan  ->  tvm-ffi

with three cross-cutting LEAVES importable by anyone (and importing no kernel
code): **device vocabulary** (imported by the kernel classes), **identity**
(fingerprints, refs, registries, payloads), and the **torch-op / autotune
adapters** that sit beside the host plan.

Two rules decide every placement question:

* **A feature is one object.** Device behavior, host schema, and cache
  identity live on the class that implements the feature (EpiOp already does
  this; transform kinds and decode formats are brought up to the same
  standard). The framework composes objects; it never switches on them.
* **The framework never enumerates flavours; the handle answers.** Any place
  the generic layer writes `if owned_fmt is not None ... elif mod.args ...`
  becomes a method call on the transform handle. Any place three call
  surfaces restate a shape rule becomes a method call on the op.

## Target layout

    quack/epilogue/                    # absorbs epilogues.py (kills the name-twin)
        __init__.py                    # PEP-562 lazy public API
        ops.py                         # <- epi_ops.py (+ absorbs epi_utils.py)
        mixin.py                       # <- epi_composable.py
        visit.py                       # <- _EpiModMixinBase (device half of gemm_epilogue.py)
        frontend.py                    # <- EpiMod, @gemm_epilogue, EpiPlan, StaticEpi, minting
        math.py                        # <- Pair/F2/unpack/pack/F16Lanes + pexp/pexp2 (one impl)
        library.py                     # <- epilogues.py (mods + _gen_epi_fn factories)
        rotary.py, scaled_exp.py, head_rmsnorm.py   # domain content, unchanged pattern

    quack/operand_transform/
        __init__.py                    # PEP-562 lazy, as today
        transform.py                   # <- transform_a.py (device side)
        kinds.py                       # NEW: one class per runtime-operand kind, owning
                                       #   device staging + host (view, fake) + geometry;
                                       #   merges A_TRANSFORM_ARG_KINDS + _ARG_KIND_HOST +
                                       #   _strip_geometry/_strip_dims; includes seed_i64x2
        frontend.py                    # @a_transform, w4_transform, dropout_a (as today)
        host.py                        # bundle builders (transform_a_operand, w4_operand_views)
                                       #   + W4/W4A8 default-config pickers (from gemm_w4)
        formats/                       # <- blockscaled/decode_formats.py + qtip.py
            __init__.py                # DecodeFormat base + W4_FORMATS registry + builtins
            qtip.py
        rng.py

    quack/gemm_runtime/                # the generic epilogue-GEMM runtime
        __init__.py
        identity.py                    # LEAF, zero kernel imports: semantic fingerprinter
                                       #   (public names), module_locator(), LocalModRegistry,
                                       #   payload install, TORCH_OP_* digest registries
        host.py                        # <- gemm_host.py, flavour-blind after the handle protocol
        torch_op.py                    # <- epi_torch_op.py (quack::gemm_epi, compile_call)
        autotune.py                    # <- epi_autotune.py

`gemm_tvm_ffi_utils.py` stays at top level: it serves the whole GEMM family
including `quack/gemm.py` (out of scope). `blockscaled/` keeps what it is
actually about (`operand.py`, `quantize.py`, `nvfp4_utils.py`, `utils.py`).
`gemm_w4.py` stays as thin sugar with updated imports.

### Old -> new mapping

| old | new |
|---|---|
| `quack/epi_ops.py` | `quack/epilogue/ops.py` |
| `quack/epi_utils.py` | folded into `epilogue/ops.py`; dead `assume_broadcast_strides` deleted |
| `quack/epi_composable.py` | `quack/epilogue/mixin.py` |
| `quack/gemm_epilogue.py` | split: `epilogue/frontend.py` + `epilogue/visit.py` + `epilogue/math.py`; fingerprinter -> `gemm_runtime/identity.py` |
| `quack/epilogues.py` | `quack/epilogue/library.py` (rope dedup: keep the `rotary.py` family) |
| `quack/gemm_host.py` | `quack/gemm_runtime/host.py` |
| `quack/epi_torch_op.py` | `quack/gemm_runtime/torch_op.py` |
| `quack/epi_autotune.py` | `quack/gemm_runtime/autotune.py` |
| `quack/operand_transform/transform_a.py` | `quack/operand_transform/transform.py` |
| `quack/operand_transform/host.py` (ref/payload part) | `gemm_runtime/identity.py` |
| `quack/operand_transform/host.py` (kind geometry part) | `operand_transform/kinds.py` |
| `quack/blockscaled/decode_formats.py` | `quack/operand_transform/formats/__init__.py` |
| `quack/blockscaled/qtip.py` | `quack/operand_transform/formats/qtip.py` |
| `quack/gemm_w4.py` `_pick_w4_cfg`/`_pick_w4a8_cfg` | `operand_transform/host.py` (kills the upward import from the epilogue frontend) |

No back-compat shims: all in-tree importers (`quack/`, `tests/`,
`benchmarks/`) are updated in the same change. `AI/` scratch scripts are not
updated (they pin old paths; fix on next use).

## New protocols

### Transform handle protocol (kills the 5x flavour dispatch)

Every `transform_a=` handle (`ATransformMod`, `PackedFormatMod`,
`DropoutAMod`) implements, in addition to `__call__(gemm)` and
`__quack_semantic_key__`:

* `resolve_operands(A, tile_m, tile_k) -> (A_for_metadata, bundle, extra_plan_key)` —
  owns the `owned_fmt` blob unpack / `TransformAOperand` assertion / plain
  pass-through, and the padded-N geometry recovery (`blob rows x 64`) in ONE
  place.
* `fake_operands(mA_or_none, a_dtype, tile_m, tile_k, dims)` — the fake/trace
  twin, used by `_compile_gemm_epi` (replaces the two hand-written branches in
  gemm_host).
* `compile_dims(A) -> transform_dims or None` — static geometry for the
  picklable compile args.
* `default_config(M, N, K, ...) -> GemmConfig or None` — replaces
  `gemm_epilogue._default_config`'s import of `gemm_w4` pickers.
* `mma_dtype_override() -> dtype or None` — layout-owning formats decode to a
  compute dtype decoupled from the blob storage dtype.
* `_module_locator()` — provided by a shared base in `identity.py` so
  `PackedFormatMod`/`DropoutAMod` stop silently lacking it.

The generic layer calls methods; `isinstance(A, TransformAOperand)` and
`getattr(mod, "args", ())` disappear from `gemm_runtime/host.py` and the three
EpiMod call surfaces. `tile_k or 64` lives inside `resolve_operands` only.

### Operand-kind protocol (`operand_transform/kinds.py`)

One class per kind, one registry, no host/device twin dicts:

    class OperandKind:
        name: str
        fn_facing: bool          # may appear in @a_transform(args=...)
        def geometry(tile_m, tile_k) -> (gran_m, g_m, gran_k, g_k, k_inner)
        def device_stage(...)    # today's _StripArg construction
        def host_view(mod, A, value, tile_m, tile_k)   # today's _ARG_KIND_HOST view half
        def host_fake(mod, mA_fake, a_dtype, tile_m, tile_k)  # fake half

`seed_i64x2` becomes a first-class kind with `fn_facing=False` (dropout-only),
fixing the registry drift. `transform.py` imports `kinds.py` for staging;
`host.py` imports it for views/fakes; geometry is computed once.

### Sink shapes

`EpiOp.sink_alloc_shape(key, m, n, tile_m, tile_n)` (already exists) becomes
the ONLY statement of the partial-buffer rule. All seven sites
(`frontend._require_shape`, `frontend._alloc_sinks`, `torch_op` x2,
`autotune` x3) call it.

### Identity leaf (`gemm_runtime/identity.py`)

* `function_semantic_key` / `semantic_value_key` (public; today private in
  `gemm_epilogue`, imported by `operand_transform.frontend` underscore-style).
* `module_locator(obj, fn)` — one implementation.
* `LocalModRegistry(consume: bool)` — one class behind `_LOCAL_EPI_MODS`
  (consume=True) and `_LOCAL_TRANSFORM_MODS` (consume=False, policy now
  explicit in one line instead of two divergent modules).
* Payload install functions and the `PoolPayload` glue for both ref types.
* `TORCH_OP_EPI_MODS` / `TORCH_OP_TRANSFORM_MODS` — written at construction
  time (the Dynamo constraint stands), read by `torch_op.py` with a normal
  top-of-file import.
* `GemmClassRef` and `TransformARef` stay distinct NamedTuples (one resolves
  to a minted class, the other to a mod) but share the locator/registry/
  payload machinery.

## Import-graph invariants (enforced by review, stated here)

* `epilogue/ops.py`, `epilogue/mixin.py`, `operand_transform/transform.py`,
  `operand_transform/kinds.py` are importable by kernel classes and import NO
  host-plan/torch/tvm modules at module level. The EpiOp host-schema trio may
  lazily import `torch` and fake-tensor helpers (unavoidable: schema methods
  run host-side only), but never `gemm_tvm_ffi_utils` — the two helpers it
  used (`div_for_dtype`, `fake_batched`) move to `compile_utils`.
* `gemm_runtime/identity.py` imports nothing from `quack` except stdlib-level
  utils. Anyone may import it.
* Package `__init__.py`s are PEP-562 lazy; kernel classes import submodules
  directly (`from quack.epilogue.ops import ...`), never the package root.
* `operand_transform/transform.py` must not import `formats/` at module level
  (today's `GemmSm90 -> decode_formats -> qtip` drag): the format is resolved
  by the frontend/handle and passed in.
* Nothing under `epilogue/` or `operand_transform/` imports variant sugar
  (`gemm_w4.py`) or `gemm_interface.py`.

## Cache epoch

Semantic digests hash `module.qualname` of fns and every capture;
`GemmClassRef`/`TransformARef` resolve by module path. The moves therefore
invalidate disk-cached kernels and autotune JSON once. Decision: accept the
cold epoch, bump the jit-cache schema version, and DELETE the key-compat
tombstones being carried for old disk caches (`add_to_output` trailing-default
mint key, conditional `vectorize` in the semantic key). Disk-cache
compatibility stops being a design constraint on key schemas.

## Migration steps (each independently green)

1. **Leaf extraction, no behavior change**: `gemm_runtime/identity.py` +
   `epilogue/math.py`. Convert the lazy imports this un-cycles into top-level
   imports. Gate: import-graph smoke (`python -c "import quack.gemm_sm90"`
   must not pull torch-op/tvm modules), epilogue + transform test subset.
2. **Single-source the rules**: sink shapes -> `sink_alloc_shape` everywhere;
   `kinds.py`; handle protocol; delete `gemm_host.gemm_epi_plan_key` (dead)
   and the `gemm_w4` upward import. Gate: full `test_gemm_transform.py`,
   `test_gemm_w4.py` roundtrips, epilogue subset incl. sinks
   (`test_gemm_epilogue -k "reduce or lse"`), torch.compile tests.
3. **Directory moves + merges**: the table above, `epilogues.py` ->
   `library.py` with the rope dedup, importer updates everywhere. Purely
   mechanical; gate: whole epilogue/transform/w4 test files with
   `--async-compile` (also exercises `GemmClassRef` resolution in workers).
4. **Split `gemm_epilogue.py`** into `frontend.py`/`visit.py`; collapse the
   `gemm`/`__call__`/`plan` cold paths onto one resolve -> plan -> run
   pipeline. Warm paths keep their existing shape (dict-copy + positional
   `_make`); cold-path consolidation only. Gate: `test_epilogue_iface.py`
   (all), launch-overhead sanity (no new per-call allocations on the warm
   path by inspection), full epilogue tests.

## Later campaigns (explicitly out of scope)

* Migrate `gemm_interface.py`'s ~13 hand-written custom ops onto
  `quack::gemm_epi`; unify the three spellings of eager-bypass.
* ~~`fn_port` on all ops~~ DONE (2026-07-29 round 2): the four load ops
  declare `fn_port = "row"/"col"/"tile"/"scalar"`; `_pinned_visit_kind` is
  port-only with an `_OPERAND_PORTS` whitelist. Mint-key compat tombstones
  also deleted (`_mint` takes all 8 params, `vectorize` unconditionally in
  the semantic key — one more deliberate digest epoch).
* ~~`gemm_w4a16` onto `EpiMod.gemm`~~ DONE (round 2): `_w4a16_alpha` mod +
  `EpiMod.gemm` gained `post_init_attrs` (plan-key aware, merged with the
  packed_cd attr) and `split_k_buffers` passthrough; the hand-rolled
  `_plan_cache`/`build_gemm_epi_plan` path and its `w4_operand_views` call
  are gone (bundle building now rides `resolve_operands` like w4a8).
  Scalar floats are launch-time immediates (host_call_arg), so tensor_scale
  values share one plan safely.
* ~~Transform-aware autotune~~ DONE (round 3): `TransformModBase.config_ok`
  is the cheap prune hook (swap_ab always out; W4: no pingpong, cluster_m=1,
  tile_m%64, tile_k in {None, fmt.tile_k}; padded-N divisibility checked in
  `_prune_for_mod` and hardened into `resolve_operands` — the w4 wrappers
  stopped restating it). `tuned_mod_gemm` takes
  transform_a/transform_sf/transform_operands: the handle digest keys the
  tuner (and `_MOD_TUNERS`), strips ride as top-level `ta__<name>` kwargs
  (keyed + L2-cloned by the Autotuner), and runtime-operand bundles are
  rebuilt per config inside the ValueError->RuntimeError rewrap so
  geometry-mismatched configs bench as inf pre-compile. `EpiMod.__call__`'s
  tuner gate no longer excludes transforms. Tests:
  test_epi_autotune.py::test_tuned_{value,w4}_transform.
* ~~Prune-predicate merge~~ DONE (round 3): the blockscaled constraint set is
  `gemm_config.blockscaled_config_ok`, called by both
  `gemm_interface.prune_invalid_gemm_configs` and
  `gemm_runtime.autotune._prune_for_mod`.
* Merge `epi_autotune._prune_for_mod` with
  `gemm_interface.prune_invalid_gemm_configs` into one predicate library in
  `gemm_config`.
* `quack/gemm.py`'s parallel plain-GEMM stack; `rmsnorm.py`.

## Implementation status (2026-07-29): SHIPPED

All four migration steps landed on the working tree (no commits), each gated
green on H100 (test_gemm_epilogue 222, test_gemm_transform 42, test_gemm_w4
72, test_epilogue_iface/test_epi_ops/test_epi_autotune/test_gemm_tile_load
29+, test_linear_cross_entropy — same counts as the pre-reorg baseline).
Deviations from the plan above:

* `TransformModBase` (operand_transform/frontend.py) implements the whole
  flavour contract in ONE place — the layout-owning vs value branch lives in
  the base class methods, not in per-class mixins; `PackedFormatMod` /
  `ATransformMod` only override ``owned_fmt`` (+ by-name ``compile_ref``).
  ``as_transform_mod`` (host.py) memoizes name/instance handles, so warm
  paths never re-fingerprint (also fixes the per-call `w4_transform` registry
  leak).
* `resolve_operands` returns a `ResolvedOperands` slot tuple, so
  EpiMod.gemm's three launch sites collapsed onto one (A_slot, B_slot) pair.
* The `gemm`/`__call__`/`plan` cold paths already shared `_iface_execute`
  after the handle protocol landed; the planned further ctx-literal dedup was
  skipped as not worth the churn.
* `sink_alloc_shape(lead, n, tile_m, tile_n)` also replaced the autotune
  slicing/pruning restatements (generic `_sink_slice`), fixing a latent
  KeyError for ColVecSelect sinks under the tuned path.
* gemm_w4a16's `if split_k is None` branch is NOT dead (explicitly-tiled
  callers reach it) — kept.
* Pre-existing, untouched: `quack/sort` E402, benchmark F821s,
  tests/test_gemm_rowvec_reduce.py importing a module absent from this tree,
  async-compile pool workers occasionally failing CUDA init on
  device-pinned runs (tests fall back to in-process compiles and pass).

## Known small fixes riding along

* `tests/test_gemm_w4.py` roundtrip fixture filter inverted to opt-out (today
  a new strip-free bf16 format silently gets no correctness coverage).
* Stale doc pointers: `decode_formats.py:14` -> nonexistent
  `test_gemm_nvfp4.py`; `rng.py`/`transform_a.py` -> branch-only
  `AI/transform_a_plan.md`.
* `gemm_w4a16` dead `if split_k is None` branch.
* `rng.py`'s private `_asm_i32` import from `blockscaled/nvfp4_utils.py`
  replaced with a proper export.
