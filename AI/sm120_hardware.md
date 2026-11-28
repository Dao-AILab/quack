# SM120 (RTX 5090 / GeForce Blackwell) Hardware Features

## Hardware Specs
- 170 SMs, 32 GB GDDR7, ~1.8 TB/s memory bandwidth
- 99 KB shared memory per block (vs 228 KB on SM90 H100)
- SM capability: 12.0

## Available Features

### TMA (Tensor Memory Accelerator)
- TMA loads and stores work (cp.async.bulk)
- Used for all reduction kernels and GEMM AB loads / epilogue stores

### Cluster
- Supported with cluster sizes up to 8 (max_active_clusters: 170/85/41/19 for sizes 1/2/4/8)
- Used for cross-CTA reduction in rmsnorm, softmax, cross_entropy
- For fp32, tighter cluster thresholds are needed due to 99 KB SMEM limit

### Register count adjustment
- `setmaxnreg.inc.sync` / `setmaxnreg.dec.sync` work
- Used in GEMM to give MMA warps more registers and load warps fewer

### Warp-level MMA
- `MmaF16BF16Op` (16x8x16, warp-level, 32 threads) — fp16/bf16 only
- Requires explicit SMEM-to-RMEM copies via `ldmatrix` before each MMA

## NOT Available

### TMA Multicast
- **Not available** on SM120 (confirmed via PTX docs)
- On SM90/SM100, TMA multicast allows a single TMA load to broadcast data
  to multiple CTAs in a cluster; SM120 lacks this hardware feature
- This is why GEMM uses cluster_shape=(1,1,1) on SM120 — multicast is
  required for efficient multi-CTA GEMM data sharing. `GemmSm120.__init__`
  rejects any other cluster shape: ptxas would compile a cluster's multicast
  loads to driver syscalls (`__cuda_syscall_cp_async_bulk_tensor_*_tile_multicast`,
  software TMA in libcuda) that share no bandwidth and assert
  (`emulate_tma: INVALID_DESCRIPTOR_FORMAT`) on 4-bit (FP4) tensormaps.
- Reduction kernels use cluster for cross-CTA mbarrier synchronization
  (not TMA multicast), so cluster works fine without multicast

### TMEM (Tensor Memory)
- SM100-only feature; not available on SM120
- TMEM is a special memory region for SM100's UMMA (unified MMA) instructions
- SM120 uses warp-level MMA registers instead
- Epilogue code that assumes TMEM register layout (e.g., gated activations,
  gemm+rms) does not work on SM120 without adaptation

### WGMMA (Warp Group MMA)
- SM90/SM100 feature; SM120 uses warp-level MMA instead
- WGMMA reads A/B directly from SMEM; warp MMA requires ldmatrix to RMEM first
- SM120 GEMM uses `GemmSm120` class with explicit SMEM→RMEM copy pipeline

### Stochastic Rounding
- Hardware stochastic rounding is SM100-only
- Tests skip on non-SM100

### CLC (Cluster Launch Control)
- SM100 persistent kernel scheduling via CLC is not available on SM120
- SM120 uses the standard dynamic persistent tile scheduler

## SMEM Constraints

The 99 KB SMEM limit affects kernel configurations:

| Kernel | fp32 max N | bf16 max N |
|---|---|---|
| Reduction (1 tensor) | 131072 | unlimited |
| Reduction (2 tensors) | 65536 | 131072 |
| RMSNorm bwd (2 tensors x2 stages) | 32768 | 65536 |

fp32 uses tighter clustering thresholds on SM12x to keep per-block SMEM under 99 KB.
