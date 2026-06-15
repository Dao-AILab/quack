# SM120 (Blackwell GeForce / RTX PRO 6000) MXFP8 block-scaled GEMM

## What
Brings block-scaled GEMM to SM120 in QuACK, reaching parity with the SM100
path for the **dense, K-major MXFP8** case (the torchao / `torch._scaled_mm`
common case).

Before: `compile_blockscaled_gemm_tvm_ffi` raised
`RuntimeError("Blockscaled SM100 GEMM requires SM100/SM110")` on SM120, even
though SM120 hardware has the `mma.sync.kind::mxf8f6f4` block-scaled warp MMA.

## How
SM100 block-scaling uses tcgen05 + TMEM, which SM120 does **not** have, so the
code cannot be shared. Instead we vendor NVIDIA's CUTLASS DSL geforce reference
(`examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/`)
into QuACK:
- `quack/gemm_sm120_blockscaled.py` — `Sm120BlockScaledGemmKernel` (warp-level
  block-scaled MMA + ldmatrix, 16-bit STMatrix epilogue, TMA load of A/B/SF).
- `quack/sm120_blockscaled_dispatch.py` — MMA-op dispatch + ldmatrix atoms.
- `quack/blockscaled_gemm_utils.py` — `compile_blockscaled_gemm_tvm_ffi` now
  dispatches to `_compile_blockscaled_gemm_sm120` when device major == 12.

The SF packed layout (`(l, rm, rk, 512)` BlockScaledBasicChunk, atom (32,4)×4)
is identical between QuACK's packer, the SM100 path, and the SM120 reference,
so scale factors are consumed without repacking (except strided slices, which
are made contiguous in the run closure).

## Capability boundary on SM120 (everything else stays SM100/SM110-only)
Supported: dense, K-major MXFP8 (e4m3/e5m2 operands + e8m0 scales, sf_vec=32),
f16/bf16 output, CTA tile 128×128×128.
Not yet wired (raise `NotImplementedError`, tests skip on SM120):
- FP4 / NVFP4 / MXFP4 / mixed FP4×FP8 (dispatch + kernel support exist in the
  vendored files; needs FP4 SF + 16-bit epilogue plumbing through the interface)
- Float32 output (kernel epilogue uses 16-bit STMatrix)
- non-K-major A/B (fp8 transpose ldmatrix is 64-bit-unaligned on SM120)
- varlen_m / varlen_k

## Tests
`tests/test_gemm_sm100_blockscaled.py`: 34 passed, 23 skipped, 0 failed on
RTX PRO 6000. Numerical correctness vs `blockscaled_gemm_reference`
(max_err ~1e-3 bf16 out).

## Perf (RTX PRO 6000, MXFP8 e4m3 → bf16)
See `sm120_mxfp8_blockscaled_bench.json`. Peak ~536 TFLOPS @ 8192³ — above the
sm120 bf16 tensor-core roofline (~390 TFLOPS), confirming the FP8 ~2× win.

| M | N | K | TFLOPS |
|---|---|---|--------|
| 2048 | 2048 | 2048 | 361 |
| 4096 | 4096 | 4096 | 497 |
| 8192 | 8192 | 8192 | 536 |
