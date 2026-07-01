# SM120 block-scaled GEMM: varlen (grouped) + batched FP4

Follow-up to the SM120 block-scaled GEMM PR (MXFP8 + FP4/NVFP4 dense). Closes
two of the remaining capability gaps on SM120.

## 1. varlen_m / varlen_k (grouped GEMM)
The vendored geforce kernel is a dense persistent kernel with no grouped /
cu_seqlens scheduler. Rather than rewrite the scheduler, varlen is implemented
as **per-expert dense dispatch** in the runner: each expert is launched as its
own dense GEMM. This is correct because:
- the dense kernel already handles arbitrary, non-128-aligned M and K (verified),
- one dense kernel is compiled once and reused across experts via dynamic
  layouts (verified correct including experts larger than the seed shape),
- scale factors use the same dQaccum-padded BlockScaledBasicChunk layout as the
  SM100 path; each expert i occupies a contiguous run of rm tiles (varlen_m) or
  rk tiles (varlen_k) at tile offset `cu[i] // 128 + i`, sliced directly from
  the packed `(1, rm, rk, 512)` buffer.

Scope: MXFP8 (matches the SM100 varlen operand builders), f16/bf16/f32 output.

Tradeoff: per-expert launch has fixed overhead, so small experts are
launch-bound; large/many-expert cases reach useful throughput (see perf). A
fused SM120 grouped scheduler is future work.

## 2. batched (l>1) FP4
QuACK's FP4 operand builder packed `float4_e2m1fn_x2` with **L innermost** for
l>1 (stride (k/2, 1) on (mn, k/2) but L fastest), which is not K-major and
tripped the SM120 K-major guard. Fixed `_create_fp4_operand_tensor` to emit a
K-major layout (L outermost, K stride 1) via a byte-level reorder — each packed
byte holds 2 FP4 along K, so the reorder preserves the 2-per-byte K packing.
l=1 is returned unchanged (byte-identical). Batched MXFP4 / NVFP4 (l>1) now run.

## Tests (RTX PRO 6000, sm_120)
`tests/test_gemm_sm100_blockscaled.py`: **77 passed, 4 skipped, 0 failed**
(was 59/16 in the dense PR). Remaining skips: 3 non-K-major + 1 invalid
FP4/sf-dtype combo — genuine boundaries. SM100/SM110 behaviour unchanged
(the FP4 builder change is l=1-identical and only reorders l>1 memory).

## Perf — `profiling/sm120_blockscaled_varlen_batched/sm120_blockscaled_varlen_bench.json`
varlen, MXFP8 → bf16, per-expert dense dispatch:

| mode | experts | seqlens | TFLOP/s |
|------|---------|---------|---------|
| varlen_m | 8 | 8×1024, N=K=4096 | 337 |
| varlen_m | 4 | [256,768,512,1024], N=K=2048 | 115 |
| varlen_k | 4 | [256,768,512,1024], M=N=2048 | 91 |

Throughput grows with expert size / count (amortizing launch overhead); a fused
grouped kernel would lift the small-expert cases.
