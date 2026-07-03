# SM120 block-scaled GEMM: mixed FP4×FP8

Follow-up to the SM120 block-scaled PRs. Enables **mixed-precision FP4×FP8** on
SM120: one operand is FP4 (e2m1), the other FP8 (e4m3 / e5m2), with e8m0 scales
at sf_vec_size=32, f16/bf16/f32 output.

## How
The vendored geforce kernel already supported `a_dtype != b_dtype`: `_setup_attributes`
sets `mixed_mode`, allocates the FP4 operand as Int8 in SMEM, uses the
`MmaMXF8F6F4Op` MMA op (atom shape (16,8,32)) and the mixed-mode ldmatrix path
(`LdMatrix8x16x8bOp` with `unpack_bits=4` + the `<<2` FP4 shift). The only thing
missing was a way to pass two operand dtypes through the public interface.

This PR adds an optional `b_dtype` argument to `compile_blockscaled_gemm_tvm_ffi`
(defaults to `ab_dtype`, so all existing same-dtype calls are unchanged). On
SM120 the dispatch validates the pair is {FP4, FP8} with e8m0/vec32 scales and
threads `b_dtype` into the kernel's A/B compile tensors. On SM100/SM110 a mixed
pair raises `NotImplementedError` (kept SM120-only).

## Scope
A,B ∈ {Float4E2M1FN} × {Float8E4M3FN, Float8E5M2} (either order), e8m0 scales,
sf_vec_size=32, K-major, f16/bf16/f32 output. Same-width mixed (e.g. e4m3+e5m2)
and FP6 pairs remain unsupported (not in `MXF8F6F4_SUPPORTED_PAIRS`).

## Tests (RTX PRO 6000, sm_120)
`test_blockscaled_mixed_fp4_fp8`: 4 pairs × 3 shapes × {bf16,f32} = 24 cases,
all pass (rel_err ~3e-3). Same-dtype paths unchanged.

## Perf (→ bf16) — `sm120_mixed_fp4_fp8_bench.json`
| pair | M=N=K | TFLOP/s |
|------|-------|---------|
| A=FP4 B=FP8 | 8192 | 554 |
| A=FP8 B=FP4 | 8192 | 540 |

Mixed sits between pure FP8 (~507) and pure FP4 (~1400): the FP8 operand
dominates the load/MMA cost, so throughput tracks the FP8 side as expected.
