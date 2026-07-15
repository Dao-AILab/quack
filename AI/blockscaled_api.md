# Block-scaled tensor API

Design for the blockscaled GEMM operand and output API: a typed operand container
is the ONLY operand form for blockscaled (MXFP8 / MXFP4 / NVFP4 / MXFP6) GEMMs
(it replaced the removed `(data, scale_factor)` tuple convention), and blockscaled
*output* dtypes complete the (a, b, d) format triple.

## 1. Motivation

The removed tuple API (`gemm((A, SFA), (B, SFB))`) had three problems:

1. **Ambiguous detection.** Future compressed operand types (2:4 structured-sparse
   with metadata, LUT-indexed) will also be `(data, aux)` pairs; `isinstance(X, tuple)`
   cannot distinguish them.
2. **No MXFP6.** PyTorch has no fp6 dtype, so a bare tensor cannot express e2m3/e3m2
   elements; a "scale dtype implies vec size" convention (e8m0->32, e4m3->16) cannot
   express new (scale dtype, vec size) combinations either.
3. **The tuple API loses NVFP4's per-tensor scale** - `blockscaled_quantize`
   requires the caller to fold `pts_A * pts_B` into GEMM `alpha` manually; forgetting
   it degrades accuracy silently.

## 2. Design decisions (with rationale)

Each decision is stated with the failure mode it prevents. Cross-checks against
torchao (`MXTensor`/`NVFP4Tensor`), PyTorch subclass mechanics, and PTX/CUTLASS fp6
constraints anchor the rationale where noted.

| # | Decision |
|---|---|
| D1 | **The container is a non-differentiable plain object** - a frozen dataclass, not a `torch.Tensor` subclass, so it structurally cannot enter autograd. Training on quantized operands is out of scope by design: a forward-quantized operand is scaled along the wrong axis for both backward GEMMs (dgrad contracts N, wgrad contracts M; scales block along K), and the square-weight case (K==N) fails *silently*. Any training story is hp master weights + per-GEMM requantization, layered above this container. |
| D2 | **`shape`/`dtype` report the logical hp view** (unpacked shape, `orig_dtype`, default bf16) as honest computed properties. Storage truth lives in `qdata`/`scale`. The container is unpacked at the top of each public entry point (where `_unpack_operand` sits); **nothing below that line ever sees it** - custom ops, autotuner, `jit_cache`, TVM-FFI all see flat plain tensors. |
| D3 | **Transpose = qdata stride-swap view; scale untouched.** The blocked `(rm, rk, 32, 4, 4)` scale atom is not view-transposable; scale is defined as "always blocked along K in either orientation". Preserves the load-bearing `weight.T` / `qw.mT` idiom. qdata<->scale shape coupling is validated at gemm dispatch time (`validate_blockscaled_sf`), NOT at construction - varlen padded scale buffers must stay constructible. |
| D4 | **Format = frozen dataclass descriptor** (`BlockScaledFormat`), not an enum: storage dtype, elem bits, elems-per-container, scale dtype, sf_vec_size, k-alignment are *data per format*. New formats are data, not code. Registry + singletons; legacy string names (`"mxfp8"`) coerce. |
| D5 | **Format crosses the custom-op boundary explicitly** as a scalar `bs_format: str` op-schema arg. Formats are never derived from tensor dtypes (no `SF_DTYPE_TO_VEC_SIZE` table, no fp4 `A.dtype` logical-K special case, no `gemm_blockscaled_ref` sf-vec ternary); `_sf_decode` is format-driven - this kills the "uint8 scale view decoded as e8m0 => NVFP4 silently runs as vec-32 MXFP8" trap. There is NO dtype->format inference site anywhere (the tuple path's `_legacy_format_from_dtypes`, the last one, was deleted with tuple support - D10). |
| D6 | **MXFP6 qdata = byte-per-element `uint8`** (identity K shape map): matches the PTX `kind::mxf8f6f4` 8-bit SMEM container, CUDA `__nv_fp6_storage_t`, cuBLASLt `CUDA_R_6F_*`; torchao does the same. Documented consequence: byte-per-element MXFP6 has fp8's bandwidth - its value is accuracy headroom over MXFP4, not speed. A packed-6-bit gmem format (CUTLASS 16U6 TMA) would change only descriptor data. |
| D7 | **Loud failure, explicit surface.** No aten interception: torch ops reject the non-tensor container with a plain `TypeError`; the supported surface is the explicit methods (`mT`/`T`/`transpose` restricted to last-two-dims, `to(device)` - dtype conversion raises pointing at `dequantize()`, `clone`, `dequantize`). **No dequantize fallback** in a kernels library. Targeted `TypeError` guards at non-blockscaled entry points (`gemm_dact`, `gemm_symmetric`, `gemm_rms`, `gemm_norm_act`) for message quality. |
| D8 | **`per_tensor_scale` is a field** (NVFP4 only). `gemm` folds `pts_A * pts_B` into alpha at the unwrap site via the existing *tensor-alpha* path (no `.item()` sync; nvfp4-with-pts compiles the tensor-alpha kernel variant). |
| D9 | **torch.compile support via pytree registration** plus a normal inlineable construction helper for in-graph quantize. It must not use `allow_in_graph`: that makes Dynamo treat the helper as an opaque op, whose non-Tensor `BlockScaledOperand` return is illegal under `fullgraph=True`. `_sf_encode` (e8m0->uint8 view across the op boundary, an Inductor `decompose_auto_functionalized` workaround) sits at the unwrap site. |
| D10 | **Containers are the only blockscaled operand form**: `(data, scale_factor)` tuples/lists are rejected at the unwrap site with a `TypeError` pointing at `BlockScaledOperand.from_parts`. Every blockscaled operand is therefore construction-validated before it reaches a GEMM. `blockscaled_quantize` / `blockscaled_quantize_dim0` keep their raw `(q, sf)` returns - they are quantizers, not operand constructors (changing the return silently corrupts `qa, sfa = ...` call sites); wrap via `from_parts`, or use `BlockScaledOperand.quantize` directly. The TVM-FFI direct path (`compile_blockscaled_gemm_tvm_ffi`) is plain-tensors-only forever. |
| D11 | **A and B carry independent formats end-to-end** (`bs_format_a` / `bs_format_b` through every layer, including the op schemas). Pair *legality* is hardware fact, owned by `mma_kind_for_pair` in `operand.py` (PTX: `mxf4nvf4` and `mxf4` need matching element types; `mxf8f6f4` mixes fp8/fp6/fp4 freely but stores every element as an **8-bit container** - so packed-fp4x2 operands cannot join mixed pairs; the byte-container `mxfp4_byte` registry entry is what a mixed pair consumes). Pair *implementation* is enforced **per-architecture**: each `gemm_smXXX` kernel class owns the asserts for the (A, B, SF, D) dtype combinations it supports (`GemmSm100.is_valid_dtypes_and_scale_factor_vec_size`, asserted in its blockscaled setup path; different SM versions support different mx dtypes). SM100's matrix admits **mixed fp8 pairs (e4m3 x e5m2, both orders)**: the DSL's `MmaMXF8F6F4Op` takes independent a/b dtypes, and instruction-K is kind-derived (mxf4/mxf4nvf4 => 64 elements, mxf8f6f4 => 32), not element-width-derived. Pairs involving fp4/fp6 sit outside SM100's matrix; section 6 describes the registry and plumbing groundwork for them and the intended smem-expansion mechanism. `validate_blockscaled_sf` checks per-operand dtypes and cross-checks logical K (`A.shape[-1]*epc_a == B.shape[-1]*epc_b` - packings may differ). |
| D12 | **Blockscaled output D is a first-class API**: `out_dtype: torch.dtype \| BlockScaledFormat \| str` on `gemm`/`gemm_add`/`gemm_act(+postact_dtype)`; the op-schema plumbing is `SFD` (mutated) + `bs_format_d`, completing the (a, b, d) format triple. See section 7. |

### Rejected alternatives

- **torchao `MXTensor` dependency**: its 2-D lazily swizzled scales do not fit quack's
  permanently-blocked scale atom or varlen padded buffers.
- **Enum-based format**: recreates the switch-on-format duplication one level up; the
  descriptor dataclass makes new formats data, not code (D4).
- **`torch.Tensor` subclass container**: rejected because there is no
  `nn.Parameter`/`state_dict`/autograd requirement to serve, and without one,
  wrapper-subclass construction, a `__torch_dispatch__` table, the traceable-subclass
  protocol, and advertised-metadata "lies" buy nothing over explicit methods on a
  frozen dataclass. The container's field set and constructors are exactly what a
  tensor-subclass wrapper would hold, so module citizenship can layer on top later
  without breaking this API.

### Out of scope

- **`nn.Module` / `quack/linear.py` integration and autograd.** The container is
  non-differentiable by construction (D1). A kernel constraint shapes any future
  module layer: there is no mixed hp x quantized GEMM - both operands must carry
  scale factors.
- **Training.** A forward-quantized operand is scaled along the wrong axis for both
  backward GEMMs (D1); a training story would keep hp master weights and requantize
  per GEMM along each contraction axis. Relatedly for blockscaled outputs (section 7):
  a fused-quantize output is gemm + STE quantize, and saved quantized outputs are
  N-axis-blocked - the wrong axis for the consumer's wgrad - so training would save hp
  activations or accept dim1-requant error (the standard MX tradeoff).
- **NVFP4 outputs with a computed global amax** (needs a delayed or two-pass amax;
  section 7 requires an explicitly provided per-tensor scale instead).
- **An e5m2 `to_mx` encoder** (`from_parts` is the e5m2 construction path).

## 3. `BlockScaledFormat` and `BlockScaledOperand` (`quack/blockscaled/operand.py`)

```python
@dataclass(frozen=True)
class BlockScaledFormat:
    name: str                    # "mxfp8_e4m3" | "mxfp8_e5m2" | "mxfp4" | "mxfp4_byte" | "nvfp4" | "mxfp6_e2m3" | "mxfp6_e3m2"
    qdata_dtype: torch.dtype     # e4m3 | e5m2 | float4_e2m1fn_x2 | uint8 (byte-container fp4/fp6)
    elem_bits: int               # 8 | 4 | 6
    elems_per_container: int     # 1 | 2 (fp4x2)
    cutlass_dtype_name: str      # CuTe-DSL MMA element type (may differ from storage: fp6)
    scale_dtype: torch.dtype     # float8_e8m0fnu | float8_e4m3fn
    sf_vec_size: int             # 32 | 16 (also the min logical-K divisibility)
    has_per_tensor_scale: bool   # nvfp4 only
```

- Hashable/picklable -> usable as dynamo guard ctx and kernel-cache key material; only
  `name` crosses the op schema.
- `logical_k(packed_k)` / `packed_k(logical_k)`; `to_cutlass_dtype()` (lazy cutlass
  import: `Float8E4M3FN`, `Float8E5M2`, `Float4E2M1FN`, `Float6E2M3FN`, `Float6E3M2FN`);
  `from_name`, `from_cutlass_dtypes`.
- Kernel support is NOT a registry property: fp6 descriptors are registered and
  constructible (checkpoints, round-trips), and whether a GEMM can execute a given
  (A, B, D) dtype combination is asserted per-architecture inside the `gemm_smXXX`
  classes (SM100's matrix admits fp8 pairs, including mixed e4m3 x e5m2, and
  homogeneous fp4; fp6 and mixed sub-byte pairs sit outside it - see section 6).
- E5M2: descriptor + `from_parts` work (the kernel admits e5m2); `quantize` raises
  because `to_mx` has no e5m2 encoder - `from_parts` is the e5m2 construction path.

```python
@dataclass(frozen=True, eq=False)        # plain container, not a Tensor
class BlockScaledOperand:
    qdata: Tensor                        # storage dtype, packed shape, K-/MN-contiguous
    scale: Tensor                        # (rm, rk, 32, 4, 4) or (L, ...), atom strides (16, 4, 1)
    format: BlockScaledFormat
    per_tensor_scale: Optional[Tensor]   # scalar fp32, NVFP4 only
    orig_dtype: torch.dtype              # hp provenance; reported by .dtype
    quant_dim: int                       # quantized-axis flag (see below)
```

- **Logical metadata as computed properties**: `shape` (unpacked; packed dim = the
  unit-stride dim of qdata - fp4 formats are K-major by kernel requirement, so
  unit-stride == packed == K), `dtype = orig_dtype` (hp provenance; storage truth is
  `qdata.dtype`), `device`, `ndim`.
- **Quantized axis** `quant_dim in {-1, -2}`: which logical dim the scale vector runs
  along - the direct analogue of CUTLASS's `UMMA::Major` SF-atom parameter. Storage
  alone cannot express it - a square fp8 canonical operand and its `.mT` view are
  byte-identical. fp4 packing pins it to the packed dim (conflicting requests raise);
  byte formats default to -1 (quantize's convention); transpose views flip it;
  `quantize(dim=-2)` and `from_parts(quant_dim=-2)` build the other direction
  directly. Carried through pytree/pickle; used by `dequantize()`. **The interface
  enforces per operand slot that the quantized axis is the contraction axis** (A: -1,
  B: -2), which makes the square-shape wrong-axis case a loud error instead of silent
  garbage.
- **Cross-check vs CUTLASS (sm100_blockscaled_layout.hpp):** CUTLASS models the same
  two quantization directions as `SfKMajorAtom` / `SfMNMajorAtom` in
  `Sm1xxBlockScaledBasicChunk` - and the MN-major atom is literally the mode-swap
  (transpose) of the K-major one, same 512 B physical block. Our `quant_dim` flag +
  `.mT` views are isomorphic to that template parameter: an "MN-quantized (M, K)
  tensor" is byte-identical to a K-quantized (K, M) tensor viewed transposed. For MMA
  *operands* only the K direction is consumable (`Sm1xxBlockScaledConfig`, used by
  every SM100 blockscaled mainloop, pins the K-major chunk - tcgen05 applies SF along
  the reduction dim by instruction definition). The MN direction exists for the
  *output* side and non-GEMM consumers - see section 7.
- **Constructor validation (mode-independent only, D3, in `__post_init__`):** qdata
  dtype <-> format; qdata has a unit-stride dim; scale ndim in {5, 6}, inner atom
  `(32, 4, 4)` with strides `(16, 4, 1)`; scale dtype = `fmt.scale_dtype` or a uint8
  view (re-viewed to the format dtype - canonicalization); pts only when
  `fmt.has_per_tensor_scale`. **No qdata<->scale shape coupling** (varlen buffers).
- Constructors: `from_parts(qdata, scale, format, *, per_tensor_scale=None,
  orig_dtype=bf16)` and `quantize(x_hp, format, *, per_tensor_scale=None)`, both via an
  ordinary helper that Dynamo inlines through under `torch.compile`.
  Explicit `dequantize()`; never implicit. It requires the canonical dense scale
  shape: padded varlen buffers need `cu_seqlens` offsets the container does not carry,
  so dequantization rejects them rather than silently applying padding as real scales.
- torch.compile / pytree: `register_pytree_node` with children `(qdata, scale, pts)`
  and ctx `(format.name, orig_dtype, quant_dim)` - same mechanics as the legacy tuple.
- Serialization: native dataclass pickling;
  `add_safe_globals([BlockScaledOperand, BlockScaledFormat])` at import for
  `weights_only=True` loading.
- `repr` is metadata-only (no storage access).


### Extension seams (forward pointers)

The descriptor bundles two orthogonal things: the *element encoding* (qdata
dtype, bits, packing, DSL type) and the *scale recipe* (scale dtype,
`sf_vec_size` = the K-extent of a `(1, sf_vec_size)` block grid). The other
axes — 2D scale grids, physical scale layouts, DSL-typeless formats, per-kind
sided-ness — are designed in section 9 and stress-tested in section 10; the
hardenings they justify in this PR (`cutlass_dtype_name` Optional; explicit
element-class + recipe gates in `mma_kind_for_pair`) are already in place.

## 4. GEMM interface integration (`quack/gemm_interface.py`)

- `_unpack_operand(X)` -> `_Operand(data, sf, fmt, pts, quant_dim)` NamedTuple; accepts
  `BlockScaledOperand | Tensor`. A tuple/list raises `TypeError` pointing at
  `BlockScaledOperand.from_parts` (tuple support and its
  `_legacy_format_from_dtypes` dtype-inference site are gone - D10).
- Blockscaled-capable entry points (`gemm`, `gemm_add`, `gemm_add_inplace`, `gemm_act`,
  `gemm_gated`): pair-legality check via `mma_kind_for_pair` (D11); scale atoms were
  already validated at construction (every blockscaled operand is a container), so the
  unwrap site only applies `_sf_encode`; pts folding into alpha (D8).
- Both operands must be blockscaled or neither: there is no mixed hp x quantized GEMM
  kernel, so one-sided scale factors are rejected.
- Custom ops `quack::gemm_out`, `gemm_add_out`, `gemm_add_inplace`, `gemm_act_out`,
  `gemm_gated_out` carry `bs_format_a` / `bs_format_b` (`Optional[str]`; string args
  need no fake-side changes; fakes stay no-ops - D11: A and B formats are
  independent). Op bodies resolve both descriptors; `_sf_decode(SF, fmt)` is
  format-driven per operand.
- The `gemm_add` add-to-output path passes flat, already-unpacked parts instead of
  rebuilding operand containers.
- `gemm_blockscaled_ref`: accepts containers; `sf_vec = fmt.sf_vec_size`; applies
  `pts_A * pts_B`.
- Rejection guards (`TypeError`) at `gemm_dact`, `gemm_symmetric`, `gemm_rms`,
  `gemm_norm_act`.

## 5. Kernel-path plumbing (`gemm.py`, `gemm_act.py`, `gemm_tvm_ffi_utils.py`)

- `quack.gemm.gemm` and `quack.gemm_act.gemm_act` take `bs_format_a`/`bs_format_b`,
  **both required** when SFA is passed - the kernel-level API carries formats
  explicitly, same as the custom ops.
- `validate_blockscaled_sf(A, B, SFA, SFB, ..., fmt_a=, fmt_b=)`: per-operand vec
  size / scale dtype / qdata dtype from the descriptors; logical K =
  `shape[-1] * elems_per_container` per operand with an A<->B cross-check (no fp4
  dtype special case; fp6-correct by construction); validates layout and
  consistency ONLY - dtype-combination support is enforced per-architecture by the
  kernel classes. **`gemm_act.py` is a full second blockscaled dispatch path** (its
  own `validate_blockscaled_sf` call and `_compile_gemm_act` cache keys) and gets
  identical treatment.
- `_compile_gemm` / `_compile_gemm_act` cache keys: `(sf_dtype, sf_vec_size,
  a_mma_dtype, b_mma_dtype)`, all derived from `fmt` at the caller. Both compile
  paths are `@jit_cache`-keyed on their full argument tuples INCLUDING the MMA
  dtypes, so same-storage formats that differ only in element type (fp6
  e2m3/e3m2, byte-fp4 vs fp8) key distinctly - no collision when the per-arch
  gate admits them. The AB cute dtype comes from `torch2cute_dtype_map[A.dtype]`
  (correct for all admitted formats). For byte-container sub-byte operands,
  `GemmSm100` separates the MMA dtype from the copy dtype: `a/b_mma_dtype`
  (threaded through both compile paths) drives the MMA builder and dtype-validity
  checks, while SMEM layouts/TMA/sizes run on the storage (container) types. Note
  that `div_for_dtype` with width 6 is non-integral, so a width-6 storage format
  needs byte-extent handling at the `_compile_gemm`/fake-tensor level.
- `_blockscaled_format_of` is a shim over `BlockScaledFormat.from_cutlass_dtypes`.
- `compile_blockscaled_gemm_tvm_ffi`: asserts plain tensors; docstring states the policy.

## 6. Sub-byte operands: mixed fp4/fp6/fp8

SM100's tcgen05 `kind::mxf8f6f4` stores every operand element in an 8-bit SMEM
container regardless of gmem width, and its instruction-K is kind-derived
(mxf4/mxf4nvf4 => 64 elements, mxf8f6f4 => 32). Two design pieces support sub-byte
elements under this kind:

- **Registry**: byte-container formats `mxfp4_byte` / `mxfp6_e2m3` / `mxfp6_e3m2`
  (uint8 storage per D6) with quantizers (the `_to_mx_floatx_unpacked` family) and
  format-keyed dequantization.
- **Kernel plumbing**: the mma-dtype/copy-dtype split - `a/b_mma_dtype` threaded
  through `_compile_gemm`/`_compile_gemm_act` into `GemmSm100` (MMA builder +
  validity on MMA types; SMEM layouts/TMA/sizes on storage types).

Which pairs execute is each architecture's dtype matrix (D11); fp6 and fp4/fp6-mixed
pairs sit outside SM100's matrix.

**TMA unpack is the intended mechanism for sub-byte SMEM expansion**, so gmem can
stay packed:

- `TmaDataFormat` IntEnum in the `cute_nvgpu` dialect has `U4_UNPACK_U8 = 12` and
  `U6_UNPACK_U8 = 13`; the TMA atom op (`cute_nvgpu.atom.make_non_exec_tiled_tma_load`)
  takes an optional `tma_format` attribute.
- The sanctioned Python hook is `internal_type=cutlass.Uint8` on
  `make_tiled_tma_atom_A/B` (`cute/nvgpu/helpers.py:153-175`): with a gmem tensor
  typed sub-byte (packed fp4/fp6) and an 8-bit internal type, `use_unpack=True`
  selects the expanding tensormap - packed gmem, byte containers in SMEM.
- Precedent: `examples/python/CuTeDSL/blackwell/mixed_input_gemm/*` feeds packed
  narrow operands exactly this way (then upconverts via `nvgpu.cvt_fpext`, whose
  docstring confirms the DSL's f6 register convention is i8 with 2-bit MSB padding);
  `sm103_dense_blockscaled_gemm_persistent.py` shows the byte-domain SMEM layout
  pattern (`make_smem_layout_atom(K_SW128, Uint8)`, `MemRange[Uint8]`,
  `adapt_layout_for_tma_*`).

Sketch, per unpack operand (a sub-byte MMA dtype under kind mxf8f6f4): (1) packed
gmem (the existing `mxfp4` storage; packed-6-bit gmem for fp6, 16 codes / 12 B,
16 B-aligned rows); (2) a TMA atom with `internal_type=Uint8`; (3) SMEM
layout/storage/load-byte counts in the byte domain (Uint8); (4) an MMA fragment over
a hand-built layout in the MMA element's addressing domain - for fp4: outer strides
x2 with the swizzle base shifted one bit (`cute.recast_layout` at
`cute/core.py:4229` is the related helper; the shape must stay at logical element
count, so hand-scale rather than recast); for fp6 (6-bit units cannot express byte
spacing) the byte-domain layout with an f6-typed pointer - the i8-padding register
convention suggests the MLIR addresses f6 SMEM by bytes. Plain byte storage without
the hand-built layout leaves the fragment/descriptor width math inconsistent, and a
naive recast keeps unscaled layouts. With TMA unpack carrying the GEMM path, the
byte-container gmem formats serve checkpoints/interop only (quantizers stay).

## 7. Blockscaled outputs: SF-generation epilogue

All other kernel `sf_*` plumbing is input-side; blockscaled D adds the output side.

- API (D12): `out_dtype` accepts a `BlockScaledFormat` (or name) on
  `gemm`/`gemm_add`/`gemm_act`; the plain blockscaled output default stays bf16. A
  format request makes `gemm` allocate qdata + an SFD buffer in the blocked layout
  and return a `BlockScaledOperand`; `out=` accepts a container (unwrapped at the
  operand seam); the ops carry `SFD: Optional[Tensor]` (in `mutates_args`) +
  `bs_format_d` - completing the (a, b, d) format triple; `_sf_encode` applies to
  SFD.
- Semantics: quantize along the **last dim (N)** - the next GEMM's contraction dim; the
  N-contiguous output is automatically a K-major operand for the consumer (what fp4
  requires). MX formats only (single-pass per-tile amax); NVFP4 output requires an
  explicitly provided per-tensor scale (a global amax is not computable in one pass).
- **SFD direction is a parameter, not a constant** (cross-checked vs CUTLASS):
  `Sm1xxBlockScaledOutputConfig<SFVecSize, major>` supports vec-along-N (Major::K,
  for D consumed as the next GEMM's A) AND vec-along-M (Major::MN, for D consumed as
  the next B / dgrad-style consumers); the SM100 epilogue SFD store instantiates both,
  selected by the requested SFD layout tag (see also examples/91_fp4_gemv's
  `kIsKMajorSFD`). The API therefore carries a direction knob alongside
  `bs_format_d` (default: vec along N), producing a `BlockScaledOperand` with the
  corresponding `quant_dim`.
- Kernel: CUTLASS `LinCombBlockScaleFactor` precedent - per-32-block amax over the
  accumulator subtile, e8m0 scale with the same floor rule as `to_mx` (bit-exact
  testability against `to_mx(gemm_ref(...))`), divide+convert, write qdata + 512 B SF
  atoms. `tile_n % 64` already enforced. Requires `split_k == 1` (quantization is
  only valid after the full K reduction). Works on hp-input GEMMs too (bf16 x bf16 ->
  mxfp8 is the network's first quantization); `gemm_act`/`gemm_gated` composition
  (quantized postact) is the MLP forward fusion payoff.

## 8. Testing

Everything is validated numerically against dequantization references, never by
shape or smoke checks alone.

- Container (`tests/test_blockscaled_operand.py`): construction rejects (dtype/format
  mismatch, bad atom strides, pts on non-nvfp4), uint8-scale canonicalization,
  varlen-padded scale construction; `clone`/`.to(device)` (dtype-`to` raises pointing
  at dequantize); `.mT`/`.T`/`transpose` round-trips with the SAME scale object; loud
  TypeError from torch ops; repr; deepcopy; save/`load(weights_only=True)`; pytree
  round-trip (orientation flag included); rejection guards at the non-blockscaled
  entry points.
- Interface (`tests/test_gemm_blockscaled_interface.py`): the dtype matrix runs on
  containers (the only operand form); a negative test pins the tuple/list
  `TypeError` (removal regression); regression tests encoding the failure modes the
  design exists to prevent: **pts-folding** (non-unit pts, default alpha => matches
  the dequant ref), **square-weight orientation** (K==N `.mT`), **uint8-scale-view
  decode** (nvfp4 stays vec-16), **mixed-format rejection**; one **e5m2 from_parts**
  run.
- Compile: container as fullgraph input (attribute reads + the e8m0/Inductor
  regression); in-graph `quantize` + gemm fullgraph; an in-graph `from_parts`
  compile test with raw (q, sf) graph inputs.
- Numerics references: `gemm_blockscaled_ref` dequant-matmul, plus bit-exact
  `torch._scaled_mm` comparison via `scale_blocked_for_cublas`.

## 9. Generalization: scale recipes, layouts, and software kinds

Design for extending the operand API beyond MX/NVFP4 microscaling. Consumers
in flight: the SM90 blockwise-promotion GEMM (sm90kscale: fp32 scales,
`kscale_block in {128, 256}`, one-sided bf16), the SM100 linear-SF cp.async
loader (sfcpasync: per-operand `sfa_linear`/`sfb_linear`), the SM90
weight-only family (sm90w4: W4A8 / W4A16 / W8A16), and DeepSeek-style fp8
recipes (acts 1x128, weights 128x128). The generalization lands WITH its
first consumer kernel; only the seams live in this PR.

### 9.1 Model

A block-scaled operand represents

```
X[i, j] ~= decode(qdata)[i, j] * S[i // bm, j // bk] * (pts if present)
```

- **Level-0 scale grid**: block shape `(bm, bk)` over the logical `(MN, K)`
  index space -> grid `(ceil(MN/bm), ceil(K/bk))`, any scale dtype.
- **Level-1**: the optional per-tensor scalar. Scaling is a hierarchy of
  levels; we support exactly two, level 1 fixed at per-tensor fp32 (a
  deliberate cap, stated like D1).
- **Element encoding is orthogonal** and includes full-precision types: bf16
  elements with per-K-block fp32 scales (kscale) are a valid operand. The
  container is about *block scaling*; quantization is the special case of a
  narrow element type.

The recipe axis is the ecosystem's own taxonomy, factored into fields instead
of an enum: cuBLASLt's `cublasLtMatmulMatrixScale_t` (`SCALAR_32F`,
`VEC16_UE4M3`, `VEC32_UE8M0`, `OUTER_VEC_32F`, `VEC128_32F`,
`BLK128x128_32F`, `PER_BATCH_SCALAR_32F` - set independently per operand via
`A_SCALE_MODE`/`B_SCALE_MODE`, i.e. D11 is their design too) and torch's
`_ScalingType` (`TensorWise, RowWise, BlockWise1x16/1x32/1x128/128x128`) are
flat enumerations of the same `(bm, bk, scale dtype)` points.

| recipe | bm | bk | scale dtype | elements |
|---|---|---|---|---|
| mxfp8 / mxfp4 / mxfp6 | 1 | 32 | e8m0 | fp8/4/6 |
| nvfp4 | 1 | 16 | e4m3 (+pts) | fp4 |
| DeepSeek 1D (acts) | 1 | 128 | fp32 | e4m3 |
| DeepSeek 2D (weights) | 128 | 128 | fp32 | e4m3 |
| kscale bf16 | 1 | 128/256 | fp32 | bf16 |
| int4-g128 (AWQ/GPTQ, W4A16) | 1 | 128 | bf16 | uint4b8 |
| rowwise (per-token / per-channel) | 1 | K (full) | fp32 | fp8/int8 |

### 9.2 Three axes, three homes

| concern | lives on | crosses op boundary as |
|---|---|---|
| element encoding + scale recipe | `BlockScaledFormat` | `bs_format_{a,b}` name |
| physical scale layout | `BlockScaledOperand` | `bs_layout_{a,b}` name |
| orientation | `BlockScaledOperand.quant_dim` | (existing) |

The format is the **interchange contract** (numerics); the layout is a
**storage detail** (like strides) - the same format legitimately exists in
multiple layouts (mxfp8 blocked for tcgen05, mxfp8 linear from a fused quant
producer), so layout is per-OPERAND (the kernels take `sfa_linear` /
`sfb_linear` independently) and never part of the format identity. Layout is
never inferred: blocked (5/6-D) vs linear (2/3-D) is rank-sniffable today,
but that is the D5 trap again and breaks on the first same-rank layout.

### 9.3 Descriptor changes (all defaulted - zero churn for existing formats)

- `sf_block_mn: int = 1` - the MN-extent of the block grid (128 for 2D).
- `sf_vec_size: Optional[int]` - `None` = the full contraction extent
  (rowwise; cuBLASLt `OUTER_VEC`). AMENDED from the first draft: forced by
  w4a8 per-token activation scales and w8a16 per-channel weight scales - a
  format descriptor is shape-independent and cannot spell a literal K.
- `canonical_scale_layout: str = "mx_blocked"` - what `quantize()` emits and
  what an unspecified operand layout means. DECIDED: MX formats stay blocked;
  "linear" is an accepted per-operand option; fp32-scale recipes are
  canonically linear (no blocked form exists for them).
- New registry rows land with their consumer kernels (never speculatively);
  see the commented examples next to `BLOCKSCALED_FORMAT_REGISTRY` and
  `test_future_recipe_examples`.
- `quantize()` policy: fp8 recipes get a generic `(bm, bk)` amax quantizer
  (e8m0 keeps the floor rule). bf16 and int-group formats RAISE - floating-
  point relative precision is scale-invariant and group scales are upstream-
  structural (GPTQ/AWQ groups, folded norms), so `from_parts` is the
  construction path (the e5m2 precedent).

### 9.4 Scale layouts (per-operand)

`BlockScaledOperand.scale_layout: Optional[str]` (`None` -> the format's
canonical layout; carried by `_view`, pytree ctx, pickle).

| tag | shape | constraints |
|---|---|---|
| `"mx_blocked"` | `(L?, rm, rk, 32, 4, 4)`, atom strides `(16,4,1)` | requires `bm == 1` and e8m0/e4m3 scales (it IS the tcgen05 atom) |
| `"linear"` | `(L?, ceil(MN/bm), ceil(K/bk))`, K-block dim contiguous | ecosystem name (FlashInfer/TRT-LLM `layout_linear`; torchao "plain"; scaled_mm NO_SWIZZLE) |

Construction validation becomes a dispatch keyed on the tag; future tags are
additive. `repack_scales(layout)` converts (linear->blocked is the existing
`pack_scale_2d_to_blocked_contig`; the quantizers already produce the linear
grid internally, so a linear-canonical `quantize` is cheaper, not costlier).
`quant_dim` and the `.mT` scale-carry trick are untouched - the grid axes are
interpreted through `quant_dim`, so D3 generalizes to 2D grids for free.

### 9.5 Kinds and sided-ness: `gemm_kind_for_pair`

Supersedes `mma_kind_for_pair` (which remains, delegated to, for the hardware
subset - it already gates element class AND recipe so software recipes fail
at kind selection with the reason). Takes `Optional` formats on both sides:
**sided-ness is per-kind, not an interface invariant**.

```
(bs, bs)  hw recipe on both (bm==1, bk in {16,32}, e8m0/e4m3, DSL elem)
    -> mxf4nvf4 | mxf4 | mxf8f6f4          # existing rules, unchanged
(bs, bs)  fp8 elements, fp32 scales         -> blockwise_promote  # 1d1d / 1d2d
(bs, None) bf16 elements, fp32 K-block scales -> blockwise_promote  # kscale
(None, bs) plain acts x group-scaled weights -> upconvert (W4A16/W8A16 family)
(None, None)                                 -> the ordinary GEMM path
```

Pairing rule (AMENDED): sides that block K must agree on `bk` (the promotion
interval must align); **k-invariant scales compose freely** - rowwise and
per-tensor scales factor out of the K loop entirely
(`out[m,n] = sfa[m]*sfb[n]*sum_k ...`) and apply at epilogue or
promotion-end, so w4a8 legally pairs rowwise fp8 acts with group-128 int4
weights. One-sided pairs use a **plain Tensor** for the unscaled side
(DECIDED - no identity-scale wrapper: the container's contract is "carries
block scales"). Whether scales apply in the register decode (W4A16 nvfp4:
2 significand bits x e4m3's 4 fit bf16 exactly), at k-tile promotion (w4a8,
DeepGEMM), or in the epilogue (rowwise) is per-arch implementation, invisible
to the API. Integer elements (e0m3/nvint4 = int4 + a recipe) get an
int-accumulate promote variant via tcgen05 `kind::i8` - exact int32 inner
products, scales at promotion.

Per-arch gates own the support matrices, as in D11: SM100 tcgen05 kinds +
linear-loader constraints (`K/vec % 4 == 0`; `sfa_linear` needs cluster
N == 1); SM90 promote/upconvert matrices (`bk in {128, 256}`, `split_k == 1`,
no `gather_A` - the kernels' existing asserts, fronted by descriptors).

### 9.6 Host plumbing

`gemm()` signature unchanged - everything rides on the operands. Op schemas
gain `bs_layout_{a,b}` strings (same pattern as `bs_format_{a,b}`); autotune
and compile keys gain the layout tags; kernel flags are derived from operands
(`sfa_linear = (opA.scale_layout == "linear")`, `kscale_sfb_1d =
(fmt_b.sf_block_mn == 1)`, ...). The D8 precedent extends: scales needed to
*interpret an operand's bytes* travel WITH the operand - a forgotten `(N,)`
per-channel multiply after a W8A16 GEMM is the pts-folding silent-accuracy
trap again, and the container is what killed that class of bug.

### 9.7 Prepared operands are NOT formats (a fence)

The weight-only kernels repack weights offline into kernel-private blobs
(WGMMA-fragment nibble shuffles, SF strips like `(N/64, K/64, 32, 4, 2)`).
These must never enter the format/layout vocabulary: they are not interchange
forms (no second consumer can read them) and they version with the kernel.
They are a third tier - **prepare-and-cache above the container**, keyed on
`(qdata storage identity, kernel key)`; weights are static so the one-time
cost amortizes. The canonical container is what serializes; prepared blobs
are rebuilt on load, which is what makes kernel-layout changes non-breaking.
Precedent: torch.ao prepacked params, TRT-LLM
`preprocess_weights_for_mixed_gemm`.

### 9.8 Decisions and sequencing

`sf_vec_size` keeps its name (`sf_block_mn` is the only new recipe field
besides the full-extent sentinel). MX canonical layout stays blocked.
One-sided = plain Tensor. bk=256 rows, ue8m0-scale DeepSeek variants, and all
registry rows land on first demand, with their consumer kernels
(sm90kscale / sfcpasync / sm90w4 integration - whichever merges first
carries the descriptor generalization).

## 10. Stress tests (worked examples)

This table is the abstraction's regression suite: every new format proposal
gets checked against it and either lands as a data row (good) or forces a
minimal, named amendment (recorded). Seven adversarial cases produced two
one-line amendments. Executable versions:
`test_future_recipe_examples` / `test_format_without_dsl_element_type`.

| case | axis exercised | change forced |
|---|---|---|
| e3m4 (no DSL type; bundled MLIR has `f8E3M4` but the DSL's reverse dispatch and dtype tables do not) | DSL-typeless elements | `cutlass_dtype_name` Optional + element-class gate (LANDED in this PR) |
| e0m3 / nvint4 (= int4 element + nvfp4's recipe; MLIR `i4`, DSL `Int4`, torch.int4 all exist) | integer elements | none - int promote via `kind::i8` is a kind, not a descriptor change |
| DeepSeek 1x128 / 128x128 fp32 scales | recipe: scale dtype + 2D grid | `sf_block_mn` (follow-up field) |
| kscale bf16 + per-row fp32 K-block scales | full-precision elements; one-sided | vocabulary ("quantized" -> "block-scaled"); sided-ness moved into kind rules |
| sfcpasync linear SF `(MN, K/vec)` | physical layout | `scale_layout` per-operand tag (follow-up field) |
| W4A16 nvfp4 weights x bf16 acts | same format, new pairing | none - the SM100 tcgen05 operand and the SM90 upconvert weight are ONE format |
| W4A8 / W8A16 (rowwise acts / per-channel weights) | full-extent bk; mixed-recipe pairs | `sf_vec_size` sentinel; pairing rule relaxed to "k-invariant composes freely" |

Known unmodeled, deliberately (add only with a customer): per-group zero
points (a third block-grid tensor; uint4b8's FIXED offset is element decode
and costs nothing) and per-L batch scales (cuBLASLt `PER_BATCH_SCALAR_32F` -
a third grid extent).
