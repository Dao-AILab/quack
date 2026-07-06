// gpu_pipe_microbench.cu
//
// One-file microbenchmark for SM-local data-movement / collective pipes.
// Currently supports, in one templated kernel:
//   * conflict-free shared-memory read or write: uint4 per lane
//   * warp shuffle: shfl.sync.bfly.b32
//   * warp integer reduction / collective reduction:
//       - redux.sync.add.s32 -> REDUX.SUM.S32
//       - redux.sync.{max,min}.{s32,u32} -> CREDUX.{MAX,MIN} on SM100+
//       - optionally redux.sync.{max,min}{.abs}.f32 -> CREDUX.F32 on SM100a/f+
//
// Counting convention:
//   * SMEM bytes are actual shared-memory bytes.
//   * SHFL is primarily counted as warp-instructions/clock/SM.  A 32-bit full-warp
//     SHFL also delivers 32 * 4 B = 128 useful bytes.
//   * REDUX/CREDUX are primarily counted as warp-instructions/clock/SM.  The table
//     also prints the input-byte rate (32 * 4 B per full-warp collective).  CREDUX
//     is a SASS instruction selected by ptxas for max/min reductions on SM100+;
//     the PTX spelling is still redux.sync.*.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=native microbenchmarks/gpu_pipe_microbench.cu -o gpu_pipe_microbench
//   # or use an explicit target, e.g. -arch=sm_90 / -arch=sm_100
//
// Run default sweep:
//   ./gpu_pipe_microbench
//   ./gpu_pipe_microbench --iters 50000 --threads 256
//
// Run a single custom mixed kernel.  --shfl and --redux are per interleave step;
// there are 16 steps/iteration, so --shfl 4 means 64 SHFL instructions per warp
// per iteration.  If --smem is read/write, each step also performs one SMEM op.
//   ./gpu_pipe_microbench --single --smem read  --shfl 4
//   ./gpu_pipe_microbench --single --smem write --redux 2
//   ./gpu_pipe_microbench --single --smem none  --cred 4        # integer CREDUX on SM100+
//   ./gpu_pipe_microbench --single --smem read  --redux 4 --redux-op max_s32
//   ./gpu_pipe_microbench --single --smem none  --cred-f32 4    # f32 CREDUX; see below
//   ./gpu_pipe_microbench --single --smem read  --shfl 2 --redux 1
//   ./gpu_pipe_microbench --single --smem none  --shfl 8
//
// Note: CUDA ptxas requires an architecture-specific/family-specific SM100+
//       target for redux.sync.*.f32.  For B300, build with e.g.
//       nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a \
//            -DGPU_PIPE_ENABLE_REDUX_F32=1 microbenchmarks/gpu_pipe_microbench.cu \
//            -o gpu_pipe_microbench
//
// Extension point: add a new per-warp/per-lane op by adding state variables,
// an op_step() helper, a per-step template count, and its metric fields in
// Result/fill_timing().  The kernel body already has a stable interleave shape.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_CUDA(expr)                                                          \
    do {                                                                          \
        cudaError_t err__ = (expr);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                         cudaGetErrorString(err__));                              \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

namespace {

constexpr int kWarpSize = 32;
constexpr int kInterleaveSteps = 16;
constexpr int kVecBytes = 16;      // uint4 per lane for SMEM
constexpr int kScalarBytes = 4;    // 32-bit SHFL / REDUX lane value
constexpr int kAlignBytes = 128;

#ifndef GPU_PIPE_ENABLE_REDUX_F32
#define GPU_PIPE_ENABLE_REDUX_F32 0
#endif

enum class SmemMode { None, Read, Write };

enum class ReduxOp {
    SumS32,
    MaxS32,
    MinS32,
    MaxU32,
    MinU32,
    MaxF32,
    MinF32,
    MaxAbsF32,
    MinAbsF32,
    Count
};

template <ReduxOp Op>
constexpr bool is_f32_redux_v = Op == ReduxOp::MaxF32 || Op == ReduxOp::MinF32
                             || Op == ReduxOp::MaxAbsF32 || Op == ReduxOp::MinAbsF32;

enum class RowKind { Single, SweepBaseline, SweepCombined };

struct Spec {
    SmemMode smem = SmemMode::None;
    int shfl_per_step = 0;
    int redux_per_step = 0;
    RowKind kind = RowKind::Single;
    ReduxOp redux_op = ReduxOp::SumS32;
};

struct Result {
    Spec spec{};
    int threads = 0;
    int active_blocks_per_sm = 0;
    unsigned long long min_cycles = 0;
    unsigned long long median_cycles = 0;
    unsigned long long max_cycles = 0;

    double smem_bytes_per_block = 0.0;
    double shfl_warp_inst_per_block = 0.0;
    double shfl_useful_bytes_per_block = 0.0;
    double redux_warp_inst_per_block = 0.0;
    double redux_input_bytes_per_block = 0.0;

    double smem_bytes_per_cycle = 0.0;
    double shfl_warp_inst_per_cycle = 0.0;
    double shfl_useful_bytes_per_cycle = 0.0;
    double redux_warp_inst_per_cycle = 0.0;
    double redux_input_bytes_per_cycle = 0.0;
    double total_useful_bytes_per_cycle = 0.0;
};

const char* smem_mode_name(SmemMode mode) {
    switch (mode) {
        case SmemMode::None: return "none";
        case SmemMode::Read: return "read";
        case SmemMode::Write: return "write";
    }
    return "unknown";
}

const char* redux_op_name(ReduxOp op) {
    switch (op) {
        case ReduxOp::SumS32: return "sum_s32";
        case ReduxOp::MaxS32: return "max_s32";
        case ReduxOp::MinS32: return "min_s32";
        case ReduxOp::MaxU32: return "max_u32";
        case ReduxOp::MinU32: return "min_u32";
        case ReduxOp::MaxF32: return "max_f32";
        case ReduxOp::MinF32: return "min_f32";
        case ReduxOp::MaxAbsF32: return "maxabs_f32";
        case ReduxOp::MinAbsF32: return "minabs_f32";
        case ReduxOp::Count: break;
    }
    return "unknown";
}

__device__ __forceinline__ unsigned lane_id() { return threadIdx.x & 31u; }

__device__ __forceinline__ uint32_t shared_addr_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t aligned_dynamic_smem_base() {
    extern __shared__ unsigned char smem_raw[];
    uint32_t base = shared_addr_u32(smem_raw);
    return (base + (kAlignBytes - 1)) & ~(kAlignBytes - 1);
}

__device__ __forceinline__ void st_shared_v4_u32(uint32_t addr, uint4 v) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ void ld_shared_v4_u32(uint32_t addr) {
    // Volatile keeps the load even though this pure throughput test does not
    // consume the values.  Destination registers stay inside inline PTX.
    asm volatile("{ .reg .u32 x, y, z, w;\n\t"
                 "ld.volatile.shared.v4.u32 {x, y, z, w}, [%0];\n\t"
                 "}"
                 :
                 : "r"(addr)
                 : "memory");
}

__device__ __forceinline__ uint32_t shfl_bfly_u32(uint32_t v, int mask) {
    uint32_t out;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=r"(out)
                 : "r"(v), "r"(mask));
    return out;
}

template <ReduxOp Op>
__device__ __forceinline__ uint32_t redux_collective_u32(uint32_t x) {
    if constexpr (Op == ReduxOp::SumS32) {
        int32_t y;
        int32_t sx = static_cast<int32_t>(x);
        asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;" : "=r"(y) : "r"(sx));
        return static_cast<uint32_t>(y);
    } else if constexpr (Op == ReduxOp::MaxS32) {
        int32_t y;
        int32_t sx = static_cast<int32_t>(x);
        asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(y) : "r"(sx));
        return static_cast<uint32_t>(y);
    } else if constexpr (Op == ReduxOp::MinS32) {
        int32_t y;
        int32_t sx = static_cast<int32_t>(x);
        asm volatile("redux.sync.min.s32 %0, %1, 0xffffffff;" : "=r"(y) : "r"(sx));
        return static_cast<uint32_t>(y);
    } else if constexpr (Op == ReduxOp::MaxU32) {
        uint32_t y;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(y) : "r"(x));
        return y;
    } else {
        uint32_t y;
        asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(y) : "r"(x));
        return y;
    }
}

#if GPU_PIPE_ENABLE_REDUX_F32
template <ReduxOp Op>
__device__ __forceinline__ float redux_collective_f32(float x) {
    float y;
    if constexpr (Op == ReduxOp::MaxF32) {
        asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(y) : "f"(x));
    } else if constexpr (Op == ReduxOp::MinF32) {
        asm volatile("redux.sync.min.f32 %0, %1, 0xffffffff;" : "=f"(y) : "f"(x));
    } else if constexpr (Op == ReduxOp::MaxAbsF32) {
        asm volatile("redux.sync.max.abs.f32 %0, %1, 0xffffffff;" : "=f"(y) : "f"(x));
    } else {
        asm volatile("redux.sync.min.abs.f32 %0, %1, 0xffffffff;" : "=f"(y) : "f"(x));
    }
    return y;
}
#endif

template <ReduxOp Op>
__device__ __forceinline__ uint32_t redux_collective(uint32_t x) {
#if GPU_PIPE_ENABLE_REDUX_F32
    if constexpr (is_f32_redux_v<Op>) {
        return __float_as_uint(redux_collective_f32<Op>(__uint_as_float(x)));
    } else
#endif
    {
        return redux_collective_u32<Op>(x);
    }
}

template <int Chain>
__device__ __forceinline__ void shfl_chain_step(uint32_t& x0, uint32_t& x1, uint32_t& x2,
                                                uint32_t& x3, uint32_t& x4, uint32_t& x5,
                                                uint32_t& x6, uint32_t& x7, int imm) {
    if constexpr (Chain == 0) {
        x0 = shfl_bfly_u32(x0, imm);
    } else if constexpr (Chain == 1) {
        x1 = shfl_bfly_u32(x1, imm);
    } else if constexpr (Chain == 2) {
        x2 = shfl_bfly_u32(x2, imm);
    } else if constexpr (Chain == 3) {
        x3 = shfl_bfly_u32(x3, imm);
    } else if constexpr (Chain == 4) {
        x4 = shfl_bfly_u32(x4, imm);
    } else if constexpr (Chain == 5) {
        x5 = shfl_bfly_u32(x5, imm);
    } else if constexpr (Chain == 6) {
        x6 = shfl_bfly_u32(x6, imm);
    } else {
        x7 = shfl_bfly_u32(x7, imm);
    }
}

__device__ __forceinline__ void shfl_step(uint32_t& x0, uint32_t& x1, uint32_t& x2,
                                          uint32_t& x3, uint32_t& x4, uint32_t& x5,
                                          uint32_t& x6, uint32_t& x7, int op) {
    const int imm = (op * 7 + 1) & 31;
    switch (op & 7) {
        case 0: shfl_chain_step<0>(x0, x1, x2, x3, x4, x5, x6, x7, imm); break;
        case 1: shfl_chain_step<1>(x0, x1, x2, x3, x4, x5, x6, x7, imm); break;
        case 2: shfl_chain_step<2>(x0, x1, x2, x3, x4, x5, x6, x7, imm); break;
        case 3: shfl_chain_step<3>(x0, x1, x2, x3, x4, x5, x6, x7, imm); break;
        case 4: shfl_chain_step<4>(x0, x1, x2, x3, x4, x5, x6, x7, imm); break;
        case 5: shfl_chain_step<5>(x0, x1, x2, x3, x4, x5, x6, x7, imm); break;
        case 6: shfl_chain_step<6>(x0, x1, x2, x3, x4, x5, x6, x7, imm); break;
        default: shfl_chain_step<7>(x0, x1, x2, x3, x4, x5, x6, x7, imm); break;
    }
}

template <ReduxOp Op, int Chain>
__device__ __forceinline__ void redux_chain_step(uint32_t& x0, uint32_t& x1, uint32_t& x2,
                                                 uint32_t& x3, uint32_t& x4, uint32_t& x5,
                                                 uint32_t& x6, uint32_t& x7) {
    if constexpr (Chain == 0) {
        x0 = redux_collective<Op>(x0);
    } else if constexpr (Chain == 1) {
        x1 = redux_collective<Op>(x1);
    } else if constexpr (Chain == 2) {
        x2 = redux_collective<Op>(x2);
    } else if constexpr (Chain == 3) {
        x3 = redux_collective<Op>(x3);
    } else if constexpr (Chain == 4) {
        x4 = redux_collective<Op>(x4);
    } else if constexpr (Chain == 5) {
        x5 = redux_collective<Op>(x5);
    } else if constexpr (Chain == 6) {
        x6 = redux_collective<Op>(x6);
    } else {
        x7 = redux_collective<Op>(x7);
    }
}

template <ReduxOp Op>
__device__ __forceinline__ void redux_step(uint32_t& x0, uint32_t& x1, uint32_t& x2,
                                           uint32_t& x3, uint32_t& x4, uint32_t& x5,
                                           uint32_t& x6, uint32_t& x7, int op) {
    switch (op & 7) {
        case 0: redux_chain_step<Op, 0>(x0, x1, x2, x3, x4, x5, x6, x7); break;
        case 1: redux_chain_step<Op, 1>(x0, x1, x2, x3, x4, x5, x6, x7); break;
        case 2: redux_chain_step<Op, 2>(x0, x1, x2, x3, x4, x5, x6, x7); break;
        case 3: redux_chain_step<Op, 3>(x0, x1, x2, x3, x4, x5, x6, x7); break;
        case 4: redux_chain_step<Op, 4>(x0, x1, x2, x3, x4, x5, x6, x7); break;
        case 5: redux_chain_step<Op, 5>(x0, x1, x2, x3, x4, x5, x6, x7); break;
        case 6: redux_chain_step<Op, 6>(x0, x1, x2, x3, x4, x5, x6, x7); break;
        default: redux_chain_step<Op, 7>(x0, x1, x2, x3, x4, x5, x6, x7); break;
    }
}

template <SmemMode Mode>
__device__ __forceinline__ void initialize_smem_conflict_free(uint32_t base) {
    if constexpr (Mode == SmemMode::Read) {
        const int tid = threadIdx.x;
        const uint4 value = make_uint4(0x12340000u + unsigned(tid),
                                       0x23450000u + unsigned(tid),
                                       0x34560000u + unsigned(tid),
                                       0x45670000u + unsigned(tid));
#pragma unroll
        for (int step = 0; step < kInterleaveSteps; ++step) {
            uint32_t addr = base + uint32_t((step * blockDim.x + tid) * kVecBytes);
            st_shared_v4_u32(addr, value);
        }
    }
}

template <SmemMode Mode, int ShflPerStep, int ReduxPerStep, ReduxOp Op>
__global__ void pipe_bench_kernel(int iters, unsigned long long* cycles, uint32_t* sink) {
    const int tid = threadIdx.x;
    const uint32_t base = aligned_dynamic_smem_base();
    const uint4 store_value = make_uint4(0x89abcdefu + unsigned(tid),
                                         0x76543210u + unsigned(tid),
                                         0xfedcba98u + unsigned(tid),
                                         0x01234567u + unsigned(tid));

    uint32_t s0 = 0x9e3779b9u ^ uint32_t(tid + 0x00u);
    uint32_t s1 = 0x85ebca6bu ^ uint32_t(tid + 0x20u);
    uint32_t s2 = 0xc2b2ae35u ^ uint32_t(tid + 0x40u);
    uint32_t s3 = 0x27d4eb2fu ^ uint32_t(tid + 0x60u);
    uint32_t s4 = 0x165667b1u ^ uint32_t(tid + 0x80u);
    uint32_t s5 = 0xd3a2646cu ^ uint32_t(tid + 0xa0u);
    uint32_t s6 = 0xfd7046c5u ^ uint32_t(tid + 0xc0u);
    uint32_t s7 = 0xb55a4f09u ^ uint32_t(tid + 0xe0u);

    uint32_t r0 = __float_as_uint(float(tid) + 0.125f);
    uint32_t r1 = __float_as_uint(float(tid) + 17.25f);
    uint32_t r2 = __float_as_uint(float(tid) + 33.375f);
    uint32_t r3 = __float_as_uint(float(tid) + 49.5f);
    uint32_t r4 = __float_as_uint(float(tid) + 65.625f);
    uint32_t r5 = __float_as_uint(float(tid) + 81.75f);
    uint32_t r6 = __float_as_uint(float(tid) + 97.875f);
    uint32_t r7 = __float_as_uint(float(tid) + 113.0f);

    initialize_smem_conflict_free<Mode>(base);
    __syncthreads();

    asm volatile("" ::: "memory");
    const unsigned long long start = clock64();

    for (int iter = 0; iter < iters; ++iter) {
#pragma unroll
        for (int step = 0; step < kInterleaveSteps; ++step) {
            if constexpr (Mode != SmemMode::None) {
                uint32_t addr = base + uint32_t((step * blockDim.x + tid) * kVecBytes);
                if constexpr (Mode == SmemMode::Read) {
                    ld_shared_v4_u32(addr);
                } else {
                    st_shared_v4_u32(addr, store_value);
                }
            }
#pragma unroll
            for (int j = 0; j < ShflPerStep; ++j) {
                shfl_step(s0, s1, s2, s3, s4, s5, s6, s7, step * ShflPerStep + j);
            }
#pragma unroll
            for (int j = 0; j < ReduxPerStep; ++j) {
                redux_step<Op>(r0, r1, r2, r3, r4, r5, r6, r7, step * ReduxPerStep + j);
            }
        }
    }

    asm volatile("" ::: "memory");
    __syncthreads();
    const unsigned long long stop = clock64();

    const uint32_t mixed = s0 ^ s1 ^ s2 ^ s3 ^ s4 ^ s5 ^ s6 ^ s7
                         ^ (r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7)
                         ^ lane_id() ^ uint32_t(stop);
    if (tid == 0) {
        cycles[blockIdx.x] = stop - start;
        sink[blockIdx.x] = mixed;
    }
}

template <typename Kernel>
int configure_kernel(Kernel kernel, int threads, int dynamic_smem_bytes,
                     int default_smem_per_block) {
    CHECK_CUDA(cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared));
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                    cudaSharedmemCarveoutMaxShared));
    if (dynamic_smem_bytes > default_smem_per_block) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        dynamic_smem_bytes));
    }

    int active_blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, threads,
                                                             dynamic_smem_bytes));
    return active_blocks;
}

template <SmemMode Mode, int ShflPerStep, int ReduxPerStep, ReduxOp Op>
bool launch_once(int threads, int iters, int blocks, int dynamic_smem_bytes,
                 unsigned long long* d_cycles, uint32_t* d_sink) {
    pipe_bench_kernel<Mode, ShflPerStep, ReduxPerStep, Op>
        <<<blocks, threads, dynamic_smem_bytes>>>(iters, d_cycles, d_sink);
    cudaError_t err = cudaGetLastError();
    if (err == cudaErrorLaunchOutOfResources || err == cudaErrorInvalidConfiguration) {
        return false;
    }
    CHECK_CUDA(err);
    CHECK_CUDA(cudaDeviceSynchronize());
    return true;
}

void fill_timing(Result* result, int blocks, int threads, int iters,
                 unsigned long long* d_cycles) {
    std::vector<unsigned long long> h_cycles(blocks);
    CHECK_CUDA(cudaMemcpy(h_cycles.data(), d_cycles, h_cycles.size() * sizeof(h_cycles[0]),
                          cudaMemcpyDeviceToHost));
    std::sort(h_cycles.begin(), h_cycles.end());

    result->min_cycles = h_cycles.front();
    result->median_cycles = h_cycles[h_cycles.size() / 2];
    result->max_cycles = h_cycles.back();

    const int warps_per_block = threads / kWarpSize;
    const int smem_ops_per_iter = result->spec.smem == SmemMode::None ? 0 : kInterleaveSteps;
    const int shfls_per_iter = kInterleaveSteps * result->spec.shfl_per_step;
    const int reduxes_per_iter = kInterleaveSteps * result->spec.redux_per_step;

    result->smem_bytes_per_block = double(iters) * double(threads) * double(smem_ops_per_iter)
                                 * kVecBytes;
    result->shfl_warp_inst_per_block = double(iters) * double(warps_per_block)
                                      * double(shfls_per_iter);
    result->shfl_useful_bytes_per_block = result->shfl_warp_inst_per_block * kWarpSize
                                        * kScalarBytes;
    result->redux_warp_inst_per_block = double(iters) * double(warps_per_block)
                                       * double(reduxes_per_iter);
    result->redux_input_bytes_per_block = result->redux_warp_inst_per_block * kWarpSize
                                        * kScalarBytes;

    result->smem_bytes_per_cycle = result->smem_bytes_per_block / double(result->max_cycles);
    result->shfl_warp_inst_per_cycle = result->shfl_warp_inst_per_block
                                     / double(result->max_cycles);
    result->shfl_useful_bytes_per_cycle = result->shfl_useful_bytes_per_block
                                        / double(result->max_cycles);
    result->redux_warp_inst_per_cycle = result->redux_warp_inst_per_block
                                      / double(result->max_cycles);
    result->redux_input_bytes_per_cycle = result->redux_input_bytes_per_block
                                        / double(result->max_cycles);
    result->total_useful_bytes_per_cycle = (result->smem_bytes_per_block
                                           + result->shfl_useful_bytes_per_block
                                           + result->redux_input_bytes_per_block)
                                         / double(result->max_cycles);
}

template <SmemMode Mode, int ShflPerStep, int ReduxPerStep, ReduxOp Op>
bool run_typed(const Spec& spec, int threads, int iters, int blocks, int smem_dynamic_bytes,
               int default_smem_per_block, unsigned long long* d_cycles, uint32_t* d_sink,
               Result* result) {
    const int dynamic_smem_bytes = Mode == SmemMode::None ? 0 : smem_dynamic_bytes;
    const int active_blocks = configure_kernel(
        pipe_bench_kernel<Mode, ShflPerStep, ReduxPerStep, Op>, threads, dynamic_smem_bytes,
        default_smem_per_block);
    if (active_blocks < 1) {
        return false;
    }

    // Warm-up, then measured launch.  The measured value comes from clock64()
    // inside the kernel, not event time.
    if (!launch_once<Mode, ShflPerStep, ReduxPerStep, Op>(threads, iters, blocks,
                                                          dynamic_smem_bytes, d_cycles, d_sink)) {
        return false;
    }
    if (!launch_once<Mode, ShflPerStep, ReduxPerStep, Op>(threads, iters, blocks,
                                                          dynamic_smem_bytes, d_cycles, d_sink)) {
        return false;
    }

    result->spec = spec;
    result->threads = threads;
    result->active_blocks_per_sm = active_blocks;
    fill_timing(result, blocks, threads, iters, d_cycles);
    return true;
}

template <SmemMode Mode, int ShflPerStep, ReduxOp Op>
bool dispatch_redux_count(const Spec& spec, int threads, int iters, int blocks,
                          int smem_dynamic_bytes, int default_smem_per_block,
                          unsigned long long* d_cycles, uint32_t* d_sink, Result* result) {
    switch (spec.redux_per_step) {
        case 0:
            return run_typed<Mode, ShflPerStep, 0, Op>(spec, threads, iters, blocks,
                                                       smem_dynamic_bytes, default_smem_per_block,
                                                       d_cycles, d_sink, result);
        case 1:
            return run_typed<Mode, ShflPerStep, 1, Op>(spec, threads, iters, blocks,
                                                       smem_dynamic_bytes, default_smem_per_block,
                                                       d_cycles, d_sink, result);
        case 2:
            return run_typed<Mode, ShflPerStep, 2, Op>(spec, threads, iters, blocks,
                                                       smem_dynamic_bytes, default_smem_per_block,
                                                       d_cycles, d_sink, result);
        case 4:
            return run_typed<Mode, ShflPerStep, 4, Op>(spec, threads, iters, blocks,
                                                       smem_dynamic_bytes, default_smem_per_block,
                                                       d_cycles, d_sink, result);
        case 8:
            return run_typed<Mode, ShflPerStep, 8, Op>(spec, threads, iters, blocks,
                                                       smem_dynamic_bytes, default_smem_per_block,
                                                       d_cycles, d_sink, result);
        default: return false;
    }
}

template <SmemMode Mode, int ShflPerStep>
bool dispatch_redux(const Spec& spec, int threads, int iters, int blocks, int smem_dynamic_bytes,
                    int default_smem_per_block, unsigned long long* d_cycles, uint32_t* d_sink,
                    Result* result) {
    switch (spec.redux_op) {
        case ReduxOp::SumS32:
            return dispatch_redux_count<Mode, ShflPerStep, ReduxOp::SumS32>(
                spec, threads, iters, blocks, smem_dynamic_bytes, default_smem_per_block,
                d_cycles, d_sink, result);
        case ReduxOp::MaxS32:
            return dispatch_redux_count<Mode, ShflPerStep, ReduxOp::MaxS32>(
                spec, threads, iters, blocks, smem_dynamic_bytes, default_smem_per_block,
                d_cycles, d_sink, result);
        case ReduxOp::MinS32:
            return dispatch_redux_count<Mode, ShflPerStep, ReduxOp::MinS32>(
                spec, threads, iters, blocks, smem_dynamic_bytes, default_smem_per_block,
                d_cycles, d_sink, result);
        case ReduxOp::MaxU32:
            return dispatch_redux_count<Mode, ShflPerStep, ReduxOp::MaxU32>(
                spec, threads, iters, blocks, smem_dynamic_bytes, default_smem_per_block,
                d_cycles, d_sink, result);
        case ReduxOp::MinU32:
            return dispatch_redux_count<Mode, ShflPerStep, ReduxOp::MinU32>(
                spec, threads, iters, blocks, smem_dynamic_bytes, default_smem_per_block,
                d_cycles, d_sink, result);
        case ReduxOp::MaxF32:
#if GPU_PIPE_ENABLE_REDUX_F32
            return dispatch_redux_count<Mode, ShflPerStep, ReduxOp::MaxF32>(
                spec, threads, iters, blocks, smem_dynamic_bytes, default_smem_per_block,
                d_cycles, d_sink, result);
#else
            std::fprintf(stderr,
                         "max_f32 requires -DGPU_PIPE_ENABLE_REDUX_F32=1 and an SM100a/f+ "
                         "target, e.g. -gencode arch=compute_103a,code=sm_103a.\n");
            return false;
#endif
        case ReduxOp::MinF32:
#if GPU_PIPE_ENABLE_REDUX_F32
            return dispatch_redux_count<Mode, ShflPerStep, ReduxOp::MinF32>(
                spec, threads, iters, blocks, smem_dynamic_bytes, default_smem_per_block,
                d_cycles, d_sink, result);
#else
            std::fprintf(stderr,
                         "min_f32 requires -DGPU_PIPE_ENABLE_REDUX_F32=1 and an SM100a/f+ "
                         "target, e.g. -gencode arch=compute_103a,code=sm_103a.\n");
            return false;
#endif
        case ReduxOp::MaxAbsF32:
#if GPU_PIPE_ENABLE_REDUX_F32
            return dispatch_redux_count<Mode, ShflPerStep, ReduxOp::MaxAbsF32>(
                spec, threads, iters, blocks, smem_dynamic_bytes, default_smem_per_block,
                d_cycles, d_sink, result);
#else
            std::fprintf(stderr,
                         "maxabs_f32 requires -DGPU_PIPE_ENABLE_REDUX_F32=1 and an SM100a/f+ "
                         "target, e.g. -gencode arch=compute_103a,code=sm_103a.\n");
            return false;
#endif
        case ReduxOp::MinAbsF32:
#if GPU_PIPE_ENABLE_REDUX_F32
            return dispatch_redux_count<Mode, ShflPerStep, ReduxOp::MinAbsF32>(
                spec, threads, iters, blocks, smem_dynamic_bytes, default_smem_per_block,
                d_cycles, d_sink, result);
#else
            std::fprintf(stderr,
                         "minabs_f32 requires -DGPU_PIPE_ENABLE_REDUX_F32=1 and an SM100a/f+ "
                         "target, e.g. -gencode arch=compute_103a,code=sm_103a.\n");
            return false;
#endif
        case ReduxOp::Count: break;
    }
    return false;
}

template <SmemMode Mode>
bool dispatch_shfl(const Spec& spec, int threads, int iters, int blocks, int smem_dynamic_bytes,
                   int default_smem_per_block, unsigned long long* d_cycles, uint32_t* d_sink,
                   Result* result) {
    switch (spec.shfl_per_step) {
        case 0:
            return dispatch_redux<Mode, 0>(spec, threads, iters, blocks, smem_dynamic_bytes,
                                           default_smem_per_block, d_cycles, d_sink, result);
        case 1:
            return dispatch_redux<Mode, 1>(spec, threads, iters, blocks, smem_dynamic_bytes,
                                           default_smem_per_block, d_cycles, d_sink, result);
        case 2:
            return dispatch_redux<Mode, 2>(spec, threads, iters, blocks, smem_dynamic_bytes,
                                           default_smem_per_block, d_cycles, d_sink, result);
        case 4:
            return dispatch_redux<Mode, 4>(spec, threads, iters, blocks, smem_dynamic_bytes,
                                           default_smem_per_block, d_cycles, d_sink, result);
        case 8:
            return dispatch_redux<Mode, 8>(spec, threads, iters, blocks, smem_dynamic_bytes,
                                           default_smem_per_block, d_cycles, d_sink, result);
        default: return false;
    }
}

bool run_spec(const Spec& spec, int threads, int iters, int blocks, int smem_dynamic_bytes,
              int default_smem_per_block, unsigned long long* d_cycles, uint32_t* d_sink,
              Result* result) {
    switch (spec.smem) {
        case SmemMode::None:
            return dispatch_shfl<SmemMode::None>(spec, threads, iters, blocks,
                                                 smem_dynamic_bytes, default_smem_per_block,
                                                 d_cycles, d_sink, result);
        case SmemMode::Read:
            return dispatch_shfl<SmemMode::Read>(spec, threads, iters, blocks,
                                                 smem_dynamic_bytes, default_smem_per_block,
                                                 d_cycles, d_sink, result);
        case SmemMode::Write:
            return dispatch_shfl<SmemMode::Write>(spec, threads, iters, blocks,
                                                  smem_dynamic_bytes, default_smem_per_block,
                                                  d_cycles, d_sink, result);
    }
    return false;
}

bool valid_count(int n) { return n == 0 || n == 1 || n == 2 || n == 4 || n == 8; }

void print_result_header() {
    std::printf("%-6s %5s %5s %-10s %9s %14s %14s %14s %14s %14s %14s\n", "smem",
                "shfl", "redux", "redop", "maxCTA/SM", "max cycles", "smem B/clk",
                "shfl inst/clk", "shfl B/clk", "redux inst/clk", "redux inB/clk");
}

void print_result_row(const Result& r) {
    std::printf("%-6s %5d %5d %-10s %9d %14llu %14.2f %14.2f %14.2f %14.2f %14.2f\n",
                smem_mode_name(r.spec.smem), r.spec.shfl_per_step, r.spec.redux_per_step,
                redux_op_name(r.spec.redux_op), r.active_blocks_per_sm, r.max_cycles,
                r.smem_bytes_per_cycle, r.shfl_warp_inst_per_cycle,
                r.shfl_useful_bytes_per_cycle, r.redux_warp_inst_per_cycle,
                r.redux_input_bytes_per_cycle);
}

void maybe_push(bool ok, const Result& result, std::vector<Result>* results) {
    if (ok) {
        results->push_back(result);
    }
}

void print_usage(const char* prog) {
    std::printf("Usage: %s [options]\n", prog);
    std::printf("\n");
    std::printf("Default with no --single runs a sweep.  For a custom case use --single.\n");
    std::printf("\nOptions:\n");
    std::printf("  --single          Run one case from --smem/--shfl/--redux\n");
    std::printf("  --cred N          Alias for --redux N --redux-op max_s32 (CREDUX on SM100+)\n");
    std::printf("  --cred-f32 N      Alias for --redux N --redux-op max_f32 (requires SM100a/f+ build)\n");
    std::printf("  --sweep           Run the default baseline + competition sweep\n");
    std::printf("  --smem MODE       none|read|write for the single case (default: none)\n");
    std::printf("  --shfl N          SHFL instructions per interleave step; N in {0,1,2,4,8}\n");
    std::printf("  --redux N         REDUX/CREDUX instructions per interleave step; N in {0,1,2,4,8}\n");
    std::printf("  --redux-op OP     sum_s32|max_s32|min_s32|max_u32|min_u32|max_f32|min_f32|"
                "maxabs_f32|minabs_f32\n");
    std::printf("  --iters, -n N     Loop iterations per CTA (default: 50000)\n");
    std::printf("  --threads, -t T   Threads per CTA; multiple of 32 (default: 256)\n");
    std::printf("  --help, -h        Show this message\n");
    std::printf("\nEach iteration has %d interleave steps.  If SMEM is enabled, each step does one "
                "uint4 SMEM op per lane.\n",
                kInterleaveSteps);
}

bool parse_smem_mode(const char* s, SmemMode* mode) {
    if (!std::strcmp(s, "none")) {
        *mode = SmemMode::None;
        return true;
    }
    if (!std::strcmp(s, "read") || !std::strcmp(s, "load")) {
        *mode = SmemMode::Read;
        return true;
    }
    if (!std::strcmp(s, "write") || !std::strcmp(s, "store")) {
        *mode = SmemMode::Write;
        return true;
    }
    return false;
}

bool parse_redux_op(const char* s, ReduxOp* op) {
    if (!std::strcmp(s, "sum") || !std::strcmp(s, "sum_s32") || !std::strcmp(s, "redux")) {
        *op = ReduxOp::SumS32;
        return true;
    }
    if (!std::strcmp(s, "max") || !std::strcmp(s, "max_s32") || !std::strcmp(s, "cred")) {
        *op = ReduxOp::MaxS32;
        return true;
    }
    if (!std::strcmp(s, "min") || !std::strcmp(s, "min_s32")) {
        *op = ReduxOp::MinS32;
        return true;
    }
    if (!std::strcmp(s, "maxu") || !std::strcmp(s, "max_u32")) {
        *op = ReduxOp::MaxU32;
        return true;
    }
    if (!std::strcmp(s, "minu") || !std::strcmp(s, "min_u32")) {
        *op = ReduxOp::MinU32;
        return true;
    }
    if (!std::strcmp(s, "maxf") || !std::strcmp(s, "max_f32") || !std::strcmp(s, "f32")
        || !std::strcmp(s, "credux_f32")) {
        *op = ReduxOp::MaxF32;
        return true;
    }
    if (!std::strcmp(s, "minf") || !std::strcmp(s, "min_f32")) {
        *op = ReduxOp::MinF32;
        return true;
    }
    if (!std::strcmp(s, "maxabs") || !std::strcmp(s, "maxabs_f32")
        || !std::strcmp(s, "max_abs_f32")) {
        *op = ReduxOp::MaxAbsF32;
        return true;
    }
    if (!std::strcmp(s, "minabs") || !std::strcmp(s, "minabs_f32")
        || !std::strcmp(s, "min_abs_f32")) {
        *op = ReduxOp::MinAbsF32;
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 50000;
    int threads = 256;
    bool run_single_case = false;
    bool force_sweep = false;

    Spec single_spec;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") || !std::strcmp(argv[i], "-n")) {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            iters = std::stoi(argv[i]);
        } else if (!std::strcmp(argv[i], "--threads") || !std::strcmp(argv[i], "-t")) {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            threads = std::stoi(argv[i]);
        } else if (!std::strcmp(argv[i], "--single")) {
            run_single_case = true;
        } else if (!std::strcmp(argv[i], "--sweep")) {
            force_sweep = true;
        } else if (!std::strcmp(argv[i], "--smem")) {
            if (++i >= argc || !parse_smem_mode(argv[i], &single_spec.smem)) {
                std::fprintf(stderr, "--smem expects none|read|write\n");
                return 1;
            }
            run_single_case = true;
        } else if (!std::strcmp(argv[i], "--shfl") || !std::strcmp(argv[i], "--shuffle")) {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            single_spec.shfl_per_step = std::stoi(argv[i]);
            run_single_case = true;
        } else if (!std::strcmp(argv[i], "--redux")) {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            single_spec.redux_per_step = std::stoi(argv[i]);
            run_single_case = true;
        } else if (!std::strcmp(argv[i], "--cred") || !std::strcmp(argv[i], "--credux")) {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            single_spec.redux_per_step = std::stoi(argv[i]);
            single_spec.redux_op = ReduxOp::MaxS32;
            run_single_case = true;
        } else if (!std::strcmp(argv[i], "--cred-f32")
                   || !std::strcmp(argv[i], "--credux-f32")) {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            single_spec.redux_per_step = std::stoi(argv[i]);
            single_spec.redux_op = ReduxOp::MaxF32;
            run_single_case = true;
        } else if (!std::strcmp(argv[i], "--redux-op")) {
            if (++i >= argc || !parse_redux_op(argv[i], &single_spec.redux_op)) {
                std::fprintf(stderr,
                             "--redux-op expects sum_s32|max_s32|min_s32|max_u32|min_u32|"
                             "max_f32|min_f32|maxabs_f32|minabs_f32\n");
                return 1;
            }
            run_single_case = true;
        } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (force_sweep) {
        run_single_case = false;
    }
    if (iters <= 0) {
        std::fprintf(stderr, "iters must be positive\n");
        return 1;
    }
    if (threads < kWarpSize || threads > 1024 || threads % kWarpSize != 0) {
        std::fprintf(stderr, "threads must be a multiple of 32 in [32, 1024]\n");
        return 1;
    }
    if (!valid_count(single_spec.shfl_per_step) || !valid_count(single_spec.redux_per_step)) {
        std::fprintf(stderr, "--shfl and --redux must be in {0,1,2,4,8}\n");
        return 1;
    }

    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    int optin_smem_per_block = 0;
    int smem_per_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&optin_smem_per_block,
                                      cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CHECK_CUDA(cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                                      dev));

    int smem_dynamic_bytes = optin_smem_per_block > 0 ? optin_smem_per_block
                                                       : int(prop.sharedMemPerBlock);
    smem_dynamic_bytes = std::min(smem_dynamic_bytes, smem_per_sm);

    const int smem_footprint = threads * kInterleaveSteps * kVecBytes + kAlignBytes;
    if (smem_footprint > smem_dynamic_bytes) {
        std::fprintf(stderr,
                     "SMEM footprint %.1f KiB exceeds dynamic shared memory %.1f KiB; reduce "
                     "--threads.\n",
                     smem_footprint / 1024.0, smem_dynamic_bytes / 1024.0);
        return 1;
    }

    const int blocks = prop.multiProcessorCount;
    unsigned long long* d_cycles = nullptr;
    uint32_t* d_sink = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cycles, size_t(blocks) * sizeof(*d_cycles)));
    CHECK_CUDA(cudaMalloc(&d_sink, size_t(blocks) * sizeof(*d_sink)));

    std::printf("Device: %s\n", prop.name);
    std::printf("SMs: %d; launching one CTA per SM\n", prop.multiProcessorCount);
    std::printf("Threads/CTA: %d; interleave steps/iter: %d; iterations: %d\n", threads,
                kInterleaveSteps, iters);
    std::printf("SMEM dynamic allocation for SMEM cases: %.1f KiB/CTA\n",
                smem_dynamic_bytes / 1024.0);
    std::printf("SMEM access: uint4/lane, contiguous/128-B aligned, bank-conflict-free\n");
    std::printf("Per-step flags: --shfl N, --redux N, --cred N; N in {0,1,2,4,8}\n");
    std::printf("CREDUX note: integer --redux-op max/min lowers to CREDUX on SM100+, REDUX on SM90.\n");
#if GPU_PIPE_ENABLE_REDUX_F32
    std::printf("CREDUX f32: enabled in this build; use --cred-f32 N or --redux-op max_f32.\n\n");
#else
    std::printf("CREDUX f32: disabled in this build; rebuild with -DGPU_PIPE_ENABLE_REDUX_F32=1 "
                "and an SM100a/f+ target.\n\n");
#endif

    if (run_single_case) {
        Result result;
        single_spec.kind = RowKind::Single;
        if (!run_spec(single_spec, threads, iters, blocks, smem_dynamic_bytes,
                      int(prop.sharedMemPerBlock), d_cycles, d_sink, &result)) {
            std::fprintf(stderr, "Requested case did not launch.\n");
            CHECK_CUDA(cudaFree(d_cycles));
            CHECK_CUDA(cudaFree(d_sink));
            return 1;
        }
        print_result_header();
        print_result_row(result);
        std::printf("\nTotals per warp per iteration: shfl=%d warp-inst, %s=%d warp-inst%s\n",
                    kInterleaveSteps * single_spec.shfl_per_step,
                    redux_op_name(single_spec.redux_op),
                    kInterleaveSteps * single_spec.redux_per_step,
                    single_spec.smem == SmemMode::None ? "" : ", smem=16 uint4/lane ops");
        CHECK_CUDA(cudaFree(d_cycles));
        CHECK_CUDA(cudaFree(d_sink));
        return 0;
    }

    std::vector<Result> baselines;
    std::vector<Result> combined;
    Result result;

    auto run_and_push = [&](Spec spec, std::vector<Result>* out) {
        maybe_push(run_spec(spec, threads, iters, blocks, smem_dynamic_bytes,
                            int(prop.sharedMemPerBlock), d_cycles, d_sink, &result),
                   result, out);
    };

    // Baselines.
    run_and_push(Spec{SmemMode::Read, 0, 0, RowKind::SweepBaseline}, &baselines);
    run_and_push(Spec{SmemMode::Write, 0, 0, RowKind::SweepBaseline}, &baselines);
    for (int n : {1, 2, 4, 8}) {
        run_and_push(Spec{SmemMode::None, n, 0, RowKind::SweepBaseline}, &baselines);
    }
    for (int n : {1, 2, 4, 8}) {
        run_and_push(Spec{SmemMode::None, 0, n, RowKind::SweepBaseline}, &baselines);
    }
    // PTX redux.sync.max.s32 lowers to CREDUX.MAX.S32 on SM100+.
    for (int n : {1, 2, 4, 8}) {
        run_and_push(Spec{SmemMode::None, 0, n, RowKind::SweepBaseline, ReduxOp::MaxS32},
                     &baselines);
    }
#if GPU_PIPE_ENABLE_REDUX_F32
    for (int n : {1, 2, 4, 8}) {
        run_and_push(Spec{SmemMode::None, 0, n, RowKind::SweepBaseline, ReduxOp::MaxF32},
                     &baselines);
    }
#endif

    // Pairwise competition sweeps.
    for (int n : {1, 2, 4, 8}) {
        run_and_push(Spec{SmemMode::Read, n, 0, RowKind::SweepCombined}, &combined);
        run_and_push(Spec{SmemMode::Write, n, 0, RowKind::SweepCombined}, &combined);
    }
    for (int n : {1, 2, 4, 8}) {
        run_and_push(Spec{SmemMode::Read, 0, n, RowKind::SweepCombined}, &combined);
        run_and_push(Spec{SmemMode::Write, 0, n, RowKind::SweepCombined}, &combined);
    }
    for (int n : {1, 2, 4, 8}) {
        run_and_push(Spec{SmemMode::Read, 0, n, RowKind::SweepCombined, ReduxOp::MaxS32},
                     &combined);
        run_and_push(Spec{SmemMode::Write, 0, n, RowKind::SweepCombined, ReduxOp::MaxS32},
                     &combined);
    }
#if GPU_PIPE_ENABLE_REDUX_F32
    for (int n : {1, 2, 4, 8}) {
        run_and_push(Spec{SmemMode::Read, 0, n, RowKind::SweepCombined, ReduxOp::MaxF32},
                     &combined);
        run_and_push(Spec{SmemMode::Write, 0, n, RowKind::SweepCombined, ReduxOp::MaxF32},
                     &combined);
    }
#endif
    // A couple of all-three examples.
    run_and_push(Spec{SmemMode::Read, 2, 1, RowKind::SweepCombined}, &combined);
    run_and_push(Spec{SmemMode::Write, 2, 1, RowKind::SweepCombined}, &combined);

    double read_base = 0.0;
    double write_base = 0.0;
    double shfl_inst_base = 0.0;
    double redux_inst_base = 0.0;
    double redux_inst_base_by_op[static_cast<int>(ReduxOp::Count)] = {};
    for (const Result& r : baselines) {
        if (r.spec.smem == SmemMode::Read) {
            read_base = std::max(read_base, r.smem_bytes_per_cycle);
        } else if (r.spec.smem == SmemMode::Write) {
            write_base = std::max(write_base, r.smem_bytes_per_cycle);
        }
        shfl_inst_base = std::max(shfl_inst_base, r.shfl_warp_inst_per_cycle);
        redux_inst_base = std::max(redux_inst_base, r.redux_warp_inst_per_cycle);
        const int op_idx = static_cast<int>(r.spec.redux_op);
        redux_inst_base_by_op[op_idx] = std::max(redux_inst_base_by_op[op_idx],
                                                 r.redux_warp_inst_per_cycle);
    }

    std::printf("Baselines:\n");
    print_result_header();
    for (const Result& r : baselines) {
        print_result_row(r);
    }

    std::printf("\nBest standalone rates:\n");
    std::printf("  SMEM read : %.2f B/clock/SM\n", read_base);
    std::printf("  SMEM write: %.2f B/clock/SM\n", write_base);
    std::printf("  SHFL      : %.3f warp-inst/clock/SM", shfl_inst_base);
    if (shfl_inst_base > 0.0) {
        std::printf("  (%.1f B/warp-inst vs read baseline = %.2f B/lane)",
                    read_base / shfl_inst_base, (read_base / shfl_inst_base) / kWarpSize);
    }
    std::printf("\n");
    std::printf("  REDUX best: %.3f warp-inst/clock/SM", redux_inst_base);
    if (redux_inst_base > 0.0) {
        std::printf("  (%.1f B/warp-inst vs read baseline = %.2f B/lane)",
                    read_base / redux_inst_base, (read_base / redux_inst_base) / kWarpSize);
    }
    std::printf("\n");
    if (redux_inst_base_by_op[static_cast<int>(ReduxOp::MaxS32)] > 0.0) {
        const double cred_base = redux_inst_base_by_op[static_cast<int>(ReduxOp::MaxS32)];
        std::printf("  CREDUX/MAX: %.3f warp-inst/clock/SM", cred_base);
        std::printf("  (%.1f B/warp-inst vs read baseline = %.2f B/lane)\n",
                    read_base / cred_base, (read_base / cred_base) / kWarpSize);
    }
    if (redux_inst_base_by_op[static_cast<int>(ReduxOp::MaxF32)] > 0.0) {
        const double cred_f32_base = redux_inst_base_by_op[static_cast<int>(ReduxOp::MaxF32)];
        std::printf("  CREDUX/F32: %.3f warp-inst/clock/SM", cred_f32_base);
        std::printf("  (%.1f B/warp-inst vs read baseline = %.2f B/lane)\n",
                    read_base / cred_f32_base, (read_base / cred_f32_base) / kWarpSize);
    }

    std::printf("\nCombined cases:\n");
    std::printf("%-6s %5s %5s %-10s %9s %14s %14s %14s %14s %10s %10s\n", "smem",
                "shfl", "redux", "redop", "maxCTA/SM", "max cycles", "smem B/clk",
                "shfl inst/clk", "redux inst/clk", "obs/ind", "obs/sum");
    for (const Result& r : combined) {
        double independent_cycles = 0.0;
        double additive_cycles = 0.0;
        if (r.spec.smem != SmemMode::None) {
            const double smem_base = r.spec.smem == SmemMode::Read ? read_base : write_base;
            const double smem_cycles = r.smem_bytes_per_block / smem_base;
            independent_cycles = std::max(independent_cycles, smem_cycles);
            additive_cycles += smem_cycles;
        }
        if (r.shfl_warp_inst_per_block > 0.0) {
            const double shfl_cycles = r.shfl_warp_inst_per_block / shfl_inst_base;
            independent_cycles = std::max(independent_cycles, shfl_cycles);
            additive_cycles += shfl_cycles;
        }
        if (r.redux_warp_inst_per_block > 0.0) {
            const double op_base = redux_inst_base_by_op[static_cast<int>(r.spec.redux_op)];
            const double redux_cycles = r.redux_warp_inst_per_block / op_base;
            independent_cycles = std::max(independent_cycles, redux_cycles);
            additive_cycles += redux_cycles;
        }
        const double obs_over_ind = double(r.max_cycles) / independent_cycles;
        const double obs_over_sum = double(r.max_cycles) / additive_cycles;

        std::printf("%-6s %5d %5d %-10s %9d %14llu %14.2f %14.2f %14.2f %10.3f %10.3f\n",
                    smem_mode_name(r.spec.smem), r.spec.shfl_per_step, r.spec.redux_per_step,
                    redux_op_name(r.spec.redux_op), r.active_blocks_per_sm, r.max_cycles,
                    r.smem_bytes_per_cycle, r.shfl_warp_inst_per_cycle,
                    r.redux_warp_inst_per_cycle, obs_over_ind, obs_over_sum);
    }

    std::printf("\nInterpretation: obs/ind ~= 1 means the observed time matches perfect overlap of "
                "standalone resources. obs/sum ~= 1 means the time is close to adding "
                "standalone times, i.e. strong competition/serialization.\n");

    CHECK_CUDA(cudaFree(d_cycles));
    CHECK_CUDA(cudaFree(d_sink));
    return 0;
}
