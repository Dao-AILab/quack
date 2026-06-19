# SM120 Blockscaled Performance Notes

This note explains the first SM120 blockscaled performance path in
`GemmSm120`.  It is intentionally narrower than the experimental branch: the PR
keeps the proven per-atom TMA mechanism and adds only the packed shared-memory
consumer path plus a `64x64x128` tile.

## Supported Scope

The performance path is opt-in:

```bash
QUACK_SM120_BLOCKSCALED_PACKED_LDSM=1
```

Current intended benchmark shape:

```bash
QUACK_SM120_BLOCKSCALED_PACKED_LDSM=1 CUTE_DSL_ARCH=sm_120a \
python benchmarks/benchmark_gemm.py \
  --mnkl 4096,4096,4096,1 \
  --tile_shape_mnk 64,64,128 \
  --cluster_shape_mnk 1,1,1 \
  --ab_dtype Float4E2M1FN \
  --sf_dtype Float8E4M3FN \
  --sf_vec_size 16 \
  --d_dtype BFloat16 \
  --warmup_iterations 5 \
  --iterations 30 \
  --skip_ref_check
```

The benchmark is a launch and timing harness.  Numerical coverage lives in
`tests/test_gemm_blockscaled.py`, including asymmetric FP4 values, poisoned
scale padding, K-page crossing, and PTX checks.

Targeted correctness gate for this path:

```bash
QUACK_SM120_BLOCKSCALED_PACKED_LDSM=1 CUTE_DSL_ARCH=sm_120a \
pytest -q tests/test_gemm_blockscaled.py -k "sm120 and packed_ldsm" -n 16 -s -rs
```

Supported formats for this SM120 path are:

- NVFP4: `Float4E2M1FN` A/B, `Float8E4M3FN` scales, `sf_vec_size=16`
- MXFP4: `Float4E2M1FN` A/B, `Float8E8M0FNU` scales, `sf_vec_size=32`
- BF16 output, `C is None`, `beta=0`
- cluster shape `(1, 1, 1)`

The packed performance path supports `64x64x64` and `64x64x128` CTA tiles.  For
`tile_K=128`, logical K must be divisible by 128.

## Why Packed LDSM

The correctness-first SM120 path expanded compact FP4 bytes into the padded
`.b4x16_p64` ldmatrix shared-memory format.  That path was useful for proving
the tuple MMA, scale mapping, and padded scale TMA, but profiling showed a large
shared-memory load bottleneck around the generated `b4x16_p64` ldmatrix
instruction.

The packed path instead stages FP4 into a swizzled packed shared-memory layout
and consumes it with:

```text
ldmatrix.sync.aligned.m8n8.x4.shared.b16
mma.sync.aligned.m16n8k64.kind::mxf4nvf4
```

This direction is based on the local CUTLASS GeForce reference in
`examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu`,
while this PR keeps QuACK's narrower per-atom TMA producer path.

Tests assert that the packed path does not regress back to `b4x16_p64`,
`m8n16`, multicast TMA, or `shared::cluster`.

## Why 64x64x128 First

The packed `64x64x64` path removed most of the original shared-load wavefront
excess, but the kernel was still producer/barrier heavy.  Moving to
`64x64x128` keeps the same accumulator ownership and correctness surface while
doubling the K work per producer/barrier cycle.  Local Nsight Compute runs on a
noisy workstation showed the expected direction:

- shared-load excessive wavefronts stayed near the packed-path level
- tensor pipe active increased materially
- barrier and MIO stall samples per issued instruction dropped
- runtime improved over `64x64x64`

Treat these numbers as direction, not a stable benchmark claim.  The following
benchmark runs were taken on an RTX 5060 workstation with reference checking
skipped because the pytest suite owns numerical validation:

```text
base sm120-blockscaled, correctness-first path:
  4096x4096x4096 NVFP4 -> BF16, 128x128x64: 60.571 ms,   2.3 TFLOP/s

this PR, packed-LDSM path:
  4096x4096x4096 NVFP4 -> BF16, 64x64x64:   2.988 ms,  46.0 TFLOP/s
  4096x4096x4096 NVFP4 -> BF16, 64x64x128:  1.329 ms, 103.4 TFLOP/s
```

## Why Not Full-Tile TMA In This PR

The natural next architecture is full-tile or grouped A/B TMA into the final
packed/swizzled shared-memory layout.  Local experiments were not clean enough
for this PR:

- raw subbyte swizzled full-tile TMA failed CuTe DSL legalization
- byte-addressable recast layouts compiled but produced many tiny static TMA
  sites and could hang at runtime
- nested grouped raw FP4 layouts hit compile/codegen timeouts

Those findings point to a separate minimal layout-lowering repro/upstream issue.
This PR keeps production on the proven per-atom TMA path and uses `tile_K=128`
as the low-risk amortization step.
