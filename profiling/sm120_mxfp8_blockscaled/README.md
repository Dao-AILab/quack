# SM120 (Blackwell GeForce / RTX PRO 6000) block-scaled GEMM

## What
Brings block-scaled GEMM to SM120 in QuACK, reaching parity with the SM100
path for **dense, K-major MXFP8 / MXFP4 / NVFP4** with f16/bf16/f32 output —
the torchao / `torch._scaled_mm` low-precision cases.

Before: `compile_blockscaled_gemm_tvm_ffi` raised
`RuntimeError("Blockscaled SM100 GEMM requires SM100/SM110")` on SM120, even
though SM120 hardware has the block-scaled warp MMA (`mma.sync.kind::mxf8f6f4`
for FP8, `MmaMXF4Op` / `MmaMXF4NVF4Op` for FP4).

## How
SM100 block-scaling uses tcgen05 + TMEM, which SM120 does **not** have, so the
code cannot be shared. We vendor NVIDIA's CUTLASS DSL geforce reference
(`examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/`)
into QuACK:
- `quack/gemm_sm120_blockscaled.py` — `Sm120BlockScaledGemmKernel` (warp-level
  block-scaled MMA + ldmatrix, TMA load of A/B/SF, f16/bf16/f32 epilogue).
- `quack/sm120_blockscaled_dispatch.py` — MMA-op + ldmatrix-atom dispatch.
- `quack/blockscaled_gemm_utils.py` — `compile_blockscaled_gemm_tvm_ffi`
  dispatches to `_compile_blockscaled_gemm_sm120` when device major == 12.

Works on **stock `nvidia-cutlass-dsl[cu13]` 4.5.2** — no unmerged CUTLASS
dependency.

### Float32-output fix
The vendored epilogue built the C copy atom from `self.c_dtype`. That atom only
defines the register→smem **tiling geometry** (via `make_tiled_copy_C_atom`);
the actual store is done by `copy_atom_r2s` (which `sm120_get_smem_store_op`
already selects correctly — StMatrix for 16-bit, universal copy for f32). With
`c_dtype=f32` the geometry atom had 1 value vs the 4-value StMatrix geometry the
epilogue loop assumes → mis-tiled store (≈74% garbage). Fixed by building the C
atom from `Float16` unconditionally (matching QuACK's `gemm_sm90`). f32 output
is now **exact** (rel_err 0.0). This bug is also present in the NVIDIA reference.

### Compact-scale TMA: not an issue on this SF layout
QuACK packs scale factors into the `(l, rm, rk, 512)` BlockScaledBasicChunk
layout (atom (32,4)×4), whose innermost tile is always 512 contiguous bytes.
TMA therefore never sees a sub-16-byte row stride, so the compact-scale TMA trap
that affects padded row-major `(M, cols, 1)` scale tensors does **not** apply
here. Verified across `sf_k ∈ {4,8,12,16,20,32,64,128}` (incl. non-mult-of-16)
for all three formats — see `compact_scale_probe.txt` (all PASS).

## Capability boundary on SM120
Supported (same-dtype A/B, K-major, f16/bf16/f32 out, CTA tile 128×128×128):
- **MXFP8** — e4m3/e5m2 + e8m0 scales, sf_vec=32
- **MXFP4** — e2m1 + e8m0 scales, sf_vec=32
- **NVFP4** — e2m1 + e4m3 scales, sf_vec=16

Still SM100/SM110-only (raise `NotImplementedError`, tests skip on SM120):
- non-K-major A/B (fp8/fp4 transpose ldmatrix is 64-bit-unaligned on SM120)
- mixed FP4×FP8 (single ab_dtype in this interface) and varlen_m/k
- batched (l>1) FP4 (QuACK's FP4 operand builder lays L innermost; follow-up)

## Tests
`tests/test_gemm_sm100_blockscaled.py` on RTX PRO 6000:
**59 passed, 16 skipped, 0 failed** (16 = 12 varlen + 3 non-K-major + 1 invalid
FP4/sf-dtype combo). `test_blockscaled_fp4` covers MXFP4 + NVFP4 × f16/bf16/f32.
SM100/SM110 behavior unchanged.

## Perf (RTX PRO 6000) — `sm120_blockscaled_full_bench.json`
bf16 output:

| fmt | M=N=K | TFLOP/s |
|-----|-------|---------|
| mxfp8 | 4096 | 496 |
| mxfp8 | 8192 | 507 |
| mxfp4 | 4096 | 1388 |
| mxfp4 | 8192 | 1431 |
| nvfp4 | 8192 | 1364 |

f32 output (exact, rel_err 0.0; higher store bandwidth → lower TFLOP/s):

| fmt | M=N=K | TFLOP/s |
|-----|-------|---------|
| mxfp8 | 8192 | 527 |
| mxfp4 | 8192 | 799 |
| nvfp4 | 8192 | 769 |

FP8 ~500-530, FP4 ~1400 TFLOP/s — well above the SM120 bf16 tensor-core
roofline (~390 TFLOP/s), as expected for 8-bit / 4-bit.

## Relation to PRs #127 / #128 (alecco)
Those add an SM120 FP4 path by extending `GemmSm120` and **require the unmerged
`NVIDIA/cutlass#3185`**; #128's optimized packed-LDSM path reports ~103 TFLOP/s.
This PR is independent: vendored standalone kernel, **no CUTLASS#3185 dep**, adds
**MXFP8** (not just FP4) and **f32 output**, immune to the compact-scale TMA trap
they document, and benches ~1400 TFLOP/s for FP4. See the PR description for the
full comparison.
