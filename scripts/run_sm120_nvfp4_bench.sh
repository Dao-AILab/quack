#!/usr/bin/env bash
set -euo pipefail

export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"

python benchmarks/benchmark_gemm.py \
  --mnkl "${MNKL:-4096,4096,4096,1}" \
  --tile_shape_mnk "${TILE:-128,128,128}" \
  --cluster_shape_mnk 1,1,1 \
  --ab_dtype Float4E2M1FN \
  --sf_dtype Float8E4M3FN \
  --sf_vec_size 16 \
  --d_dtype BFloat16 \
  --sm120_nvfp4_path "${SM120_NVFP4_PATH:-validated}" \
  --warmup_iterations "${WARMUP:-5}" \
  --iterations "${ITERS:-10}" \
  --skip_ref_check
