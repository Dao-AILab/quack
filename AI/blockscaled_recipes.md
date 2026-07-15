# Block-scale recipes and scale layouts

Follow-up design to AI/blockscaled_api.md (PR #169). Generalizes the
operand API from "MX/NVFP4 microscaling" to arbitrary block-scale recipes and
physical scale layouts, driven by three in-flight consumers:

- **sfcpasync** (SM100): SFA/SFB in the *linear* layout `(M, K/vec, L)` /
  `(N, K/vec, L)` loaded via cp.async, per-operand (`sfa_linear`/`sfb_linear`),
  instead of the pre-blocked 128x4 atom.
- **sm90kscale** (SM90): DeepGEMM-style blockwise-promotion GEMM. fp32 scales,
  `kscale_block in {128, 256}`; SFA per-row `(1, bk)`; SFB per-row (1d1d) or
  per-`(128, bk)` block (1d2d); **bf16 elements with SFA only** (one-sided).
- **DeepSeek fp8 recipes** generally: acts `1x128`, weights `128x128`, fp32
  scales (V3.1 variant: ue8m0 scales).

## 1. Model (first principles)

A block-scaled operand represents

```
X[i, j] ~= decode(qdata)[i, j] * S[i // bm, j // bk] * (pts if present)
```

- **Level-0 scale grid**: block shape `(bm, bk)` over the logical `(MN, K)`
  index space -> grid `(ceil(MN/bm), ceil(K/bk))`, any scale dtype.
- **Level-1**: optional per-tensor scalar (existing `per_tensor_scale`).
  Scaling is a hierarchy of levels; we support exactly two, level 1 fixed at
  per-tensor fp32 — a deliberate cap, stated like D1.
- **Element encoding is orthogonal** and includes full-precision types: bf16
  with fp32 block scales is a valid operand (kscale). The container is about
  *block scaling*, not quantization; quantization is the special case of a
  narrow element type. Docstrings change vocabulary accordingly.

Every recipe in use is a point in this space:

| recipe            | bm  | bk  | scale dtype | elements | notes |
|-------------------|-----|-----|-------------|----------|-------|
| mxfp8 / mxfp4 / mxfp6 | 1 | 32  | e8m0        | fp8/4/6  | tcgen05 SF |
| nvfp4             | 1   | 16  | e4m3 (+pts) | fp4      | tcgen05 SF |
| DeepSeek 1D (acts)| 1   | 128 | fp32        | e4m3     | promote |
| DeepSeek 2D (wts) | 128 | 128 | fp32        | e4m3     | promote |
| kscale bf16       | 1   | 128/256 | fp32    | bf16     | promote, SFA-only |
| rowwise / tensorwise | 1/MN | K | fp32     | fp8      | degenerate; expressible, not implemented |

## 2. Three axes, three homes

| concern | contents | lives on | crosses op boundary as |
|---|---|---|---|
| element encoding | qdata dtype, bits, packing, DSL type | `BlockScaledFormat` | `bs_format_{a,b}` name |
| scale recipe | `(bm, bk)`, scale dtype, pts level | `BlockScaledFormat` | (same name) |
| physical scale layout | how the grid sits in memory | `BlockScaledOperand` | `bs_layout_{a,b}` name |
| orientation | which stored axis is K-blocked | `BlockScaledOperand.quant_dim` | `quant_dim` (existing) |

The format is the **interchange contract** (numerics); the layout is a
**storage detail** (like strides) — the same format legitimately exists in
multiple layouts (mxfp8 blocked for tcgen05, mxfp8 linear from a fused quant
producer), so layout must not be part of the format identity, and it is
per-*operand* because the kernels take it per-operand (`sfa_linear` vs
`sfb_linear` are independent flags).

Layout is never inferred. Blocked (5/6-D) vs linear (2/3-D) is rank-sniffable
today, but that is the same trap D5 killed for dtypes, and it breaks on the
first same-rank layout (a K-major linear variant).

## 3. `BlockScaledFormat` changes (all defaulted — zero churn for existing formats)

```python
@dataclass(frozen=True)
class BlockScaledFormat:
    name: str
    qdata_dtype: torch.dtype
    cutlass_dtype_name: Optional[str]   # WAS str. None = no DSL element type;
                                        # from_cutlass_dtypes skips None entries
                                        # (fixes the e3m4-style registry crash)
    elem_bits: int
    elems_per_container: int
    scale_dtype: torch.dtype            # now includes torch.float32
    sf_vec_size: int                    # unchanged: K elements per scale block (bk)
    sf_block_mn: int = 1                # NEW: 1 = row-granular; 128 = 2D blocks (bm)
    has_per_tensor_scale: bool = False
    canonical_scale_layout: str = "mx_blocked"  # NEW: what quantize() emits and
                                        # what an unspecified operand layout means
```

New registry rows (added with their consumer kernels, not before):

```python
FP8_E4M3_1x128   = ("fp8_e4m3_1x128",   e4m3, "Float8E4M3FN", 8, 1, fp32, 128, 1,   linear)
FP8_E4M3_128x128 = ("fp8_e4m3_128x128", e4m3, "Float8E4M3FN", 8, 1, fp32, 128, 128, linear)
BF16_1x128       = ("bf16_1x128",       bf16, "BFloat16",    16, 1, fp32, 128, 1,   linear)
# + bk=256 siblings for kscale_block=256; + ue8m0-scale variants if DSv3.1-style lands
```

`quantize()` policy: fp8 recipes get a generic `_to_blockwise(x, bm, bk,
scale_dtype)` amax quantizer (reshape to `(M/bm, bm, K/bk, bk)`, amax over the
block axes; e8m0 keeps the floor rule). **BF16 formats raise** — floating-point
relative precision is scale-invariant, so an amax encoder is meaningless; the
scales are upstream-structural (GPTQ/AWQ groups, folded norms, emulation
limbs) and `from_parts` is the construction path (same stance as e5m2).

## 4. Scale layouts

`BlockScaledOperand` gains `scale_layout: Optional[str] = None` (None ->
`format.canonical_scale_layout`; carried by `_view`, pytree ctx, pickle;
`from_parts(..., scale_layout=)`).

| tag | shape | validation | constraints |
|---|---|---|---|
| `"mx_blocked"` | `(L?, rm, rk, 32, 4, 4)`, atom strides `(16,4,1)` | today's `check_blocked_scale_atom` | requires `bm == 1` and scale dtype in {e8m0, e4m3} (it IS the tcgen05 atom) |
| `"linear"` | `(L?, ceil(MN/bm), ceil(K/bk))`, K-block dim contiguous | rank + `stride(-1) == 1` | ecosystem name (FlashInfer/TRT-LLM `layout_linear`; torchao "plain"; scaled_mm NO_SWIZZLE) |

Construction validation becomes a dispatch keyed on the tag
(`SCALE_LAYOUT_VALIDATORS`); future tags (e.g. a K-major DeepGEMM variant) are
additive rows. `op.repack_scales(layout)` converts (linear->blocked is the
existing `pack_scale_2d_to_blocked_contig`; the quantizers already produce the
linear grid internally and pack as a last step, so `quantize` to a linear
canonical layout is *cheaper*, not costlier).

`quant_dim` is untouched: the grid axes are interpreted through it, so `.mT`
(swap qdata strides, flip quant_dim, carry scale) works unchanged — including
for 128x128 grids. D3 generalizes for free.

## 5. Pair legality: `gemm_kind_for_pair`

Supersedes `mma_kind_for_pair` (which remains, delegated to, for the hardware
subset). Signature takes `Optional` B-format — **sided-ness is per-kind, not an
interface invariant**; kscale-bf16 is the concrete one-sided customer, so the
"both operands or neither" assert moves out of `_prep_blockscaled` into here.

```
(bs, bs), hw recipe on both (bm==1, bk in {16,32}, e8m0/e4m3 scales, DSL elem)
    -> mxf4nvf4 | mxf4 | mxf8f6f4        # existing rules, unchanged
(bs, bs), fp8 elements, fp32 scales, bk_a == bk_b   # promotion interval must align
    -> blockwise_promote                  # 1d1d (bm_b==1) and 1d2d (bm_b==128)
(bs, None), bf16 elements, fp32 scales
    -> blockwise_promote                  # kscale SFA-only
else -> ValueError
```

Per-arch gates own implementation, as in D11:
- **SM100** (tcgen05 kinds): existing matrix; layout support = blocked always,
  linear per operand where the cp.async loader exists (constraints from
  sfcpasync: `K/vec % 4 == 0`, `sfa_linear` requires cluster N == 1).
- **SM90** (new gate, `blockwise_promote` only): `bk in {128, 256}`; elements
  bf16 (one-sided) or fp8 (both-sided, SFB 1d or 2d); linear layout only;
  `split_k == 1`; no `gather_A`. These are the kscale kernel's existing
  asserts, fronted by descriptors.

## 6. Host plumbing

- `gemm()` signature unchanged — everything rides on the operands.
- Op schemas gain `bs_layout_a` / `bs_layout_b: Optional[str]` (same pattern as
  `bs_format_{a,b}`; strings, no fake changes). Autotune and compile keys gain
  the layout tags (they select kernel variants; `sfa_linear`/`sfb_linear` and
  `kscale_*` remain kernel `__init__` args, so `jit_cache` keys them once
  threaded, like `a_mma_dtype` was).
- The interface derives kernel flags from operands:
  `sfa_linear = (opA.scale_layout == "linear")`, kscale config from B's format
  (`kscale_sfb_1d = (fmt_b.sf_block_mn == 1)`), etc.
- **One-sided operands: the unscaled side is a plain `Tensor` (DECIDED).**
  No identity-scale wrapper format: the container's contract is "carries block
  scales", and a wrapper would advertise scale machinery that does not exist,
  force the hp path through blockscaled dispatch, and add a format whose only
  meaning is "not a format". `_unpack_operand` already yields
  `_Operand(data)` for plain tensors, so dispatch is
  `gemm_kind_for_pair(fmt_a | None, fmt_b | None)` with `(None, None)` = the
  ordinary GEMM path. The rule is symmetric: `(None, fmt_b)` is reserved for
  the mirror-image future case (weight-only group scales x hp activations,
  W4A16-style), which the SM90 gate simply does not admit yet.
- Eventual public spelling of kscale:
  `gemm(BlockScaledOperand.from_parts(a_bf16, sfa, "bf16_1x128"), W.mT)` —
  self-describing, orientation-checked, no new `gemm` arguments.

## 7. Dequant / references

- `dequantize()` gains one `repeat_interleave` along MN (`bm > 1`), and a
  per-layout grid unpack (linear is the identity).
- `gemm_blockscaled_ref` generalizes the same way + a one-sided variant.
- Bit-exact external references: `torch._scaled_mm` supports 1x128 / 128x128
  blockwise fp8 on Hopper — same role `scale_blocked_for_cublas` plays for MX.

## 8. Sequencing

1. **PR #169 carries the seams, not the generalization** (DONE): the
   "Extension seams" section in blockscaled_api.md; `cutlass_dtype_name` made
   Optional with the registry lookup skipping None entries; `mma_kind_for_pair`
   gates on the MMA element class explicitly instead of falling through to
   mxf8f6f4. `sf_vec_size` keeps its name (DECIDED).
2. This generalization lands **with its first consumer kernel** (whichever of
   sm90kscale / sfcpasync is ready first) — a type-system generalization with
   no kernel consuming it is exactly the speculative plumbing 169 trimmed.
3. All changes are defaulted/additive: existing formats, tests, checkpoints,
   and the op schemas stay valid; `sf_vec_size` alias keeps external readers
   working.

## 9. Testing

- Construction matrix (format x layout), incl. rejections: `mx_blocked` with
  `bm > 1`; linear stride violations; bf16 `quantize` raises.
- `.mT` round-trips for 2D grids (dequant equality, same scale object).
- `gemm_kind_for_pair` truth table incl. one-sided and bk-mismatch rejection;
  extend the kind-mirror test to the promote kind and the SM90 gate.
- Numerics: dequant-ref and `torch._scaled_mm` blockwise comparisons for the
  fp8 recipes; kscale bf16 vs fp32-einsum reference with block-expanded SFA.

## Open questions

1. Rename `sf_vec_size` -> `sf_vec_size` inside #169 pre-merge (cheap now,
   alias forever after) or alias at follow-up time?
2. Once sfcpasync lands, does the MX formats' *canonical* layout flip to
   linear (producers skip the pack; blocked becomes the repack target for the
   TMA path) or stay blocked for compatibility?
3. One-sided spelling: B as a plain tensor (proposed) vs requiring both sides
   wrapped with a "no-scale" format — plain tensor keeps `gemm`'s hp path
   untouched and matches "the container is for block-scaled operands only".
4. Register bk=256 rows now (kscale supports them) or on first model demand?
