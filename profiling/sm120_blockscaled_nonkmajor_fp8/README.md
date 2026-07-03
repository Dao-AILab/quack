# SM120 block-scaled GEMM: M-major A (FP8 non-K-major)

Follow-up enabling **M-major A operands for FP8** on SM120 (the activations-in-
row-major case), at parity performance with K-major.

## Root cause of the old non-K-major block
`make_ldmatrix_atom` always used `LdMatrix8x8x16bOp` (a 16-bit StMatrix-style
atom). With `transpose=True` on 8-bit data its source needs a 128-bit-aligned
16-bit lane pattern, which an M-major FP8 SMEM tile (8-bit elements) cannot
satisfy → ICE `src ptr alignment (64 bits) does not meet requirement (128 bits)`.

This was **not** a hardware limit — the correct primitive for transposing 8-bit
data is `ldmatrix.m16n16.trans.b8` (`LdMatrix16x16x8bOp`, `num_matrices=2`).

## Fix
`make_ldmatrix_atom`: when `transpose and operand_dtype.width == 8` (and not the
mixed FP4 path), use `LdMatrix16x16x8bOp(transpose=True, num_matrices=2)`. With
that atom, M-major A FP8 compiles and is bit-correct.

## Scope and the B / FP4 boundary (structural)
The warp MMA is **m16n8k32**. The A (M) fragment is 16-wide → meets the 16
elements `ldmatrix.m16n16.trans.b8` needs, so **M-major A works**. The B (N)
fragment is only **8-wide** → below 16, so **N-major B cannot** feed the 8-bit
transpose path and stays K-major-only. FP4 (4-bit) has no ldmatrix transpose
primitive at all, so **FP4 A must stay K-major**.

Net SM120 major support after this PR:
- A: K-major (any dtype) **or M-major (FP8 only)**
- B: K-major only

This covers the common case (A = activations, possibly row-major; B = weights,
storable K-major). N-major B would require loading 2 N-tiles together or a
different fragment layout — a deeper mainloop change for little practical gain.

## Tests (RTX PRO 6000, sm_120)
`test_gemm_sm100_blockscaled.py`: **102 passed, 3 skipped, 0 failed** (was 77/4).
`test_blockscaled_mxfp8_major_modes` now runs A M-major on SM120. Remaining
skips: 2 N-major-B + 1 invalid FP4/sf-dtype combo.

## Perf (MXFP8 → bf16) — `sm120_amajor_fp8_bench.json`
| A major | M=N=K | TFLOP/s |
|---------|-------|---------|
| K-major | 8192 | 543.9 |
| M-major | 8192 | 544.6 |

M-major A is at **parity** with K-major — the m16n16.trans.b8 atom adds no cost.
