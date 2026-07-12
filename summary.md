# PR: feature/blockscaled-mixed-input

**Title:**

```
[GEMM] Native mixed FP4/FP6/FP8 block-scaled A/B on SM100 via TMA unpack
```

**Description:**

Stacked on #<operand-api PR> (`blockscaled-operand-api`) - review only the top two commits.

## What

Any A/B format pair of {mxfp8_e4m3, mxfp8_e5m2, mxfp4, mxfp6_e2m3, mxfp6_e3m2} now runs
natively on SM100 under tcgen05 `kind::mxf8f6f4`, with sub-byte operands loaded from
packed gmem through the TMA unpack tensormaps (`CU_TENSOR_MAP_DATA_TYPE_16U4/16U6_ALIGN16B`).
Verified on GB200 at ~2.2e-3 max rel error vs the dequantized-operand reference for every
pair - indistinguishable from the fp8 x fp8 baseline - including fp6 x fp4, where both
operands take the unpack path. Design/recipe notes: `AI/blockscaled_api.md` section 7.

## How

The unpack path is reached by passing `internal_type=cutlass.Uint8` to
`make_tiled_tma_atom_A/B` while the gmem tensor keeps its true sub-byte element type
(recasting gmem to Uint8 selects a plain U8 tensormap and silently kills the unpack).
Per unpack operand, `GemmSm100` follows the CUTLASS C++ `SmemAllocType = uint8_t`
convention:

- All SMEM-side accounting is byte-domain `Uint8` - layout atom selection, `MemRange`
  storage, stage budgeting. TMA expands each 16 packed elements to a 16-byte smem
  footprint; the Uint8-typed smem tensor feeds `make_fragment_A/B` directly (the smem
  descriptor is built from width x stride products, so the byte layout yields the
  correct LBO/SBO - an fp4-element-domain layout produces the packed `mxf4` convention
  instead, which is wrong here).
- mbarrier `tx_count` counts packed bytes (elements x 4 or 6 bits; the intra-16B gaps
  are not written), matching `sizeof_bits<ElementA> x cosize(SmemLayoutA)` in CUTLASS.
- Constraints enforced with clean errors: unpack operands are K-major with
  logical K % 128 (tensormap `globalDim[0]` granule; arg-spec divisibility for fp4,
  host-side assert for fp6) and 32B base alignment.

**fp6 FFI boundary:** torch has no fp6 dtype, so packed-fp6 qdata crosses tvm-ffi as raw
`uint8` of shape `(..., 3K/4)` - a little-endian 6-bit stream (CUTLASS `SubbyteReference`
layout, verified bit-exact against the DSL's own fp6 fill kernel) - with an independent
byte-extent K sym (divisibility 96 B = 128 elements) and an in-kernel reinterpret to the
fp6 element type. `a/b_mma_dtype` marks the operand at the boundary.

**Formats:** `mxfp6_e2m3/e3m2` switch from byte-container to packed 6-bit storage
(`pack_uint6`/`unpack_uint6`); the `mxfp4_byte` workaround format is deleted (packed
`mxfp4` joins mixed pairs natively); `elems_per_container` is replaced by
elem_bits-derived `storage_k`/`logical_k`/`is_packed` on `BlockScaledFormat` (an int
cannot express fp6's 4-elements-per-3-bytes).

**One fp4 atom (second commit):** both-fp4 pairs always construct `MmaMXF4NVF4Op` -
`kind::mxf4nvf4` is one MMA atom parameterized by the scale config (vec 32 e8m0 = mxfp4,
whose instantiation PTX also spells `kind::mxf4`; vec 16 = nvfp4), mirroring CUTLASS
C++'s `SM100_MMA_MXF4_SS<..., VS>`. quack subclasses the DSL op to un-pin its hardcoded
vec 16; both DSL fp4 op classes build the identical MLIR atom type (no kind attribute -
the backend derives the PTX spelling from the vec size), so the lowered instruction is
unchanged. `mma_kind_for_pair` no longer treats `mxf4` as a separate kind.

## Testing

- `test_blockscaled_gemm_mixed`: 10-pair mixed matrix, each checked against both the
  blockscaled reference and the float32 dequantized product (rel < 5e-3).
- `test_mixed_fp4_unpack_k_granule_rejected`: K=320 (passes the old K%32 rule, violates
  the unpack K%128 granule) must be rejected cleanly.
- Packed-fp6 bit-layout and quantize/dequantize round-trip tests; gate unit tests for
  the new dtype combinations; mxfp4/nvfp4 numerics re-verified through the unified atom.
- Full suite (`pytest tests/ -n 8 --async-compile=32`): no failures outside a
  pre-existing `test_linear_varlen_m` flake that reproduces at `main` (grossly wrong
  outputs for non-blockscaled bf16 varlen_m under multi-process GPU load; passes in
  isolation - tracked separately) and environment-dependent async-compile infra tests.
