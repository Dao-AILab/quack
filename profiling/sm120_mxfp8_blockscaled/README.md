# SM120 (Blackwell GeForce / RTX PRO 6000) block-scaled GEMM

## What
Brings block-scaled GEMM to SM120 in QuACK, reaching parity with the SM100
path for **dense, K-major MXFP8, MXFP4, and NVFP4** (the torchao /
`torch._scaled_mm` low-precision cases).

Before: `compile_blockscaled_gemm_tvm_ffi` raised
`RuntimeError("Blockscaled SM100 GEMM requires SM100/SM110")` on SM120, even
though SM120 hardware has the block-scaled warp MMA (`mma.sync.kind::mxf8f6f4`
for FP8, `MmaMXF4Op`/`MmaMXF4NVF4Op` for FP4).

## How
SM100 block-scaling uses tcgen05 + TMEM, which SM120 does **not** have, so the
code cannot be shared. Instead we vendor NVIDIA's CUTLASS DSL geforce reference
(`examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/`)
into QuACK:
- `quack/gemm_sm120_blockscaled.py` — `Sm120BlockScaledGemmKernel` (warp-level
  block-scaled MMA + ldmatrix, TMA load of A/B/SF, f16/bf16 epilogue).
- `quack/sm120_blockscaled_dispatch.py` — MMA-op dispatch + ldmatrix atoms
  (MXFP8 / MXFP4 / NVFP4 / mixed).
- `quack/blockscaled_gemm_utils.py` — `compile_blockscaled_gemm_tvm_ffi`
  dispatches to `_compile_blockscaled_gemm_sm120` when device major == 12.

The SF packed layout (`(l, rm, rk, 512)` BlockScaledBasicChunk, atom (32,4)×4)
is identical between QuACK's packer, the SM100 path, and the SM120 reference,
so scale factors are consumed without repacking (strided slices made contiguous).

## Capability boundary on SM120 (everything else stays SM100/SM110-only)
Supported (same-dtype A/B, K-major, f16/bf16 out, CTA tile 128×128×128):
- MXFP8: e4m3/e5m2 + e8m0 scales, sf_vec=32
- MXFP4: e2m1 + e8m0 scales, sf_vec=32
- NVFP4: e2m1 + e4m3 scales, sf_vec=16

Not supported on SM120 (raise `NotImplementedError`, tests skip):
- Float32 output — unsupported even by the NVIDIA geforce reference kernel
- non-K-major A/B — fp8/fp4 transpose ldmatrix is 64-bit-unaligned
- mixed FP4×FP8 (single ab_dtype in this interface) and varlen_m/k
- batched (l>1) FP4 — QuACK's FP4 operand builder lays L innermost; follow-up

## Tests
`tests/test_gemm_sm100_blockscaled.py`: 46 passed, 23 skipped, 0 failed on
RTX PRO 6000. Includes new `test_blockscaled_fp4_f16out` (MXFP4 + NVFP4 ×
f16/bf16). Numerical correctness vs `blockscaled_gemm_reference`
(rel_err ~2-3e-3).

## Perf (RTX PRO 6000, → bf16 output)
Peak FP8 ~536 TFLOPS, peak FP4 ~1447 TFLOPS @ 8192³ — well above the sm120 bf16
tensor-core roofline (~390 TFLOPS). See `sm120_mxfp8_blockscaled_bench.json` and
`sm120_fp4_blockscaled_bench.json`.

| fmt | M=N=K | TFLOPS |
|-----|-------|--------|
| mxfp8 | 4096 | 497 |
| mxfp8 | 8192 | 536 |
| mxfp4 | 8192 | 1442 |
| nvfp4 | 8192 | 1447 |
