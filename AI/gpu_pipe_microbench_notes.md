# SM-local Pipe Throughput: SMEM, SHFL, REDUX/CREDUX

Microbenchmarks:

- CUDA: `microbenchmarks/gpu_pipe_microbench.cu`
- CuTe DSL: `microbenchmarks/gpu_pipe_microbench.py`

Both benchmarks launch one CTA per SM, time the hot loop with `clock64()`, and
use the same interleaved instruction shape. The shared-memory access pattern is
conflict-free: each lane executes one `uint4` load or store, so each warp touches
a contiguous 512 B region split into four aligned 128 B shared-memory
transactions.

## Counting

Shared memory is counted as real transferred bytes:

```text
1 ld/st.shared.v4.u32 per lane = 16 B/lane = 512 B/warp
```

SHFL is counted as useful delivered data:

```text
1 full-warp 32-bit SHFL = 32 lanes * 4 B = 128 useful B
```

REDUX/CREDUX is reported primarily as warp-instructions/clk/SM. The printed
`redux inB/clk` column is only an input-byte normalization:

```text
1 full-warp 32-bit collective input = 32 lanes * 4 B = 128 input B
```

That input-byte count is useful for scale, but collective reductions do not
behave like simple SMEM bandwidth consumers.

## Results Summary

Representative settings:

```text
iterations = 50000
threads    = 256
ops/iter   = 16
```

Approximate standalone peaks:

| GPU / target | Case | Throughput |
| --- | --- | --- |
| H100 / SM90 | SMEM read | 128 B/clk/SM |
| H100 / SM90 | SMEM write | 128 B/clk/SM |
| H100 / SM90 | SHFL.BFLY.B32 | 1.00 warp-inst/clk/SM, 128 useful B/clk/SM |
| H100 / SM90 | REDUX.SUM.S32 | 0.50 warp-inst/clk/SM |
| B300 / SM103a | CREDUX.MAX.F32 | 1.70-1.71 warp-inst/clk/SM, 217-218 input B/clk/SM |

## SHFL vs SMEM

SHFL competes strongly with SMEM in these tests. A balanced mixed kernel with
one SMEM `uint4` operation and four 32-bit SHFLs per interleave step gives:

```text
read  + shfl: total ~120 B/clk/SM, obs/sum ~= 1.06
write + shfl: total ~122 B/clk/SM, obs/sum ~= 1.05
```

`obs/sum ~= 1` means the mixed runtime is close to the sum of the standalone
SMEM time and standalone SHFL time. The useful model is:

```text
1 full-warp 32-bit SHFL ~= 128 B of SM-local data-movement pressure
                         ~= 4 B/lane
```

A SMEM exchange of one 32-bit value costs a store plus a load:

```text
SMEM round trip = 4 B/lane store + 4 B/lane load = 8 B/lane
```

So, from a throughput-pressure perspective, one conflict-free SMEM round trip is
roughly comparable to two 32-bit SHFL instructions. SHFL can still win on
latency and simplicity.

## REDUX/CREDUX vs SMEM

REDUX/CREDUX does not behave like SHFL.

On H100 / SM90, standalone `REDUX.SUM.S32` is about:

```text
1 REDUX.SUM.S32 warp instruction ~= 2 clocks
```

Mixed SM90 cases show good overlap with SMEM reads and partial interference with
SMEM writes:

```text
read  + redux r4: obs/ind ~= 1.0
write + redux r4: obs/ind ~= 1.2-1.25
```

On B300 / SM103a, CREDUX.F32 overlaps with both conflict-free shared loads and
stores. Paired CUDA single-case measurements:

```text
standalone smem read        : ~128 B/clk/SM
standalone smem write       : ~112 B/clk/SM
standalone CREDUX.MAX.F32 r4: ~1.71 warp-inst/clk/SM
```

Mixed CREDUX.F32 cases:

```text
case                    obs/ind   obs/sum   interpretation
read  + CREDUX.MAX.F32   ~1.03     ~0.65    near-perfect overlap
write + CREDUX.MAX.F32   ~1.00     ~0.66    near-perfect overlap
```

`obs/ind ~= 1` means the observed runtime is close to perfect overlap of the
standalone resources. `obs/sum ~= 1` would mean additive or serialized time.

Practical model:

- SHFL: model as SMEM-like SM-local data-movement pressure, about `4 B/lane` per
  32-bit SHFL.
- SM90 `REDUX.SUM.S32`: model as a separate, slower collective pipe, about
  `0.5 warp-inst/clk/SM`, mostly overlapping with SMEM reads and partially
  interfering with SMEM writes.
- SM100-family `CREDUX.MAX.{S32,F32}`: model as a separate collective pipe. On
  B300, f32 max reaches about `1.7 warp-inst/clk/SM` and overlaps with SMEM
  loads/stores in the tested r4 cases.

## CREDUX Forms

PTX uses the `redux.sync.*` spelling for both REDUX and CREDUX-like operations.
The SASS instruction depends on target and operation:

- `REDUX`: integer AND/OR/XOR/SUM/MIN/MAX forms on SM90.
- `CREDUX`: SM100+ collective max/min-style forms, including f32 forms on
  architecture-specific or family-specific SM100+ targets.

The microbenchmarks expose this through `--redux-op` and aliases:

```bash
./gpu_pipe_microbench --single --redux 4 --redux-op sum_s32
./gpu_pipe_microbench --single --cred 4
./gpu_pipe_microbench --single --cred-f32 4

python microbenchmarks/gpu_pipe_microbench.py --no-suite --redux --redux-op sum_s32
python microbenchmarks/gpu_pipe_microbench.py --no-suite --cred 4
python microbenchmarks/gpu_pipe_microbench.py --no-suite --cred-f32 4
```

On H100 / SM90, `--cred` still lowers to `REDUX.MAX.S32`, not CREDUX. On SM100+
integer max/min lowers to CREDUX. F32 CREDUX requires an SM100-family/specific
target such as `sm_103a`.

Generic B300 `-arch=native` targets `sm_103`, and ptxas rejects f32 CREDUX:

```text
Instruction 'redux.f32' not supported on .target 'sm_103'
```

Use an architecture-specific or family-specific target:

```bash
nvcc -O3 -std=c++17 \
  -gencode arch=compute_103a,code=sm_103a \
  -DGPU_PIPE_ENABLE_REDUX_F32=1 \
  microbenchmarks/gpu_pipe_microbench.cu -o /tmp/gpu_pipe_microbench_f32
```

On the local B300 system, CUDA 13.2 lowers the f32 PTX forms to:

```text
CREDUX.MAX.F32
CREDUX.MAXABS.F32
CREDUX.MIN.F32
CREDUX.MINABS.F32
```

## Reproduction Commands

CUDA baseline:

```bash
nvcc -O3 -std=c++17 -arch=native \
  microbenchmarks/gpu_pipe_microbench.cu -o /tmp/gpu_pipe_microbench
/tmp/gpu_pipe_microbench --iters 50000 --threads 256
```

CUDA f32 CREDUX on B300 / SM103a:

```bash
nvcc -O3 -std=c++17 \
  -gencode arch=compute_103a,code=sm_103a \
  -DGPU_PIPE_ENABLE_REDUX_F32=1 \
  microbenchmarks/gpu_pipe_microbench.cu -o /tmp/gpu_pipe_microbench_f32
/tmp/gpu_pipe_microbench_f32 --single --smem none --cred-f32 4 \
  --iters 50000 --threads 256
```

CuTe DSL:

```bash
python microbenchmarks/gpu_pipe_microbench.py \
  --iterations 50000 --threads 256 --rep 1 --warmup 0

python microbenchmarks/gpu_pipe_microbench.py \
  --no-suite --cred-f32 4 --iterations 50000 --threads 256 --rep 3 --warmup 1
```

CuTe DSL dump for loop-body checks:

```bash
rm -rf /tmp/cute_cred_dump
mkdir -p /tmp/cute_cred_dump
CUTE_DSL_KEEP_PTX=1 \
CUTE_DSL_KEEP_CUBIN=1 \
CUTE_DSL_DUMP_DIR=/tmp/cute_cred_dump \
CUTE_DSL_PTXAS_PATH="$(command -v ptxas)" \
python microbenchmarks/gpu_pipe_microbench.py \
  --no-suite --cred-f32 4 --iterations 50000 --threads 256 \
  --rep 1 --warmup 0 --reserve-smem-kib 1

rg -c 'redux\.sync\.max\.f32' /tmp/cute_cred_dump/*.ptx
cuobjdump --dump-sass /tmp/cute_cred_dump/*.cubin | rg -c 'CREDUX\.MAX\.F32'
```
