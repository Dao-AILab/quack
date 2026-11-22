# syntax=docker/dockerfile:1.7
# All CUDA runtime libs come from torch's pip wheels (under
# /usr/local/lib/python3.12/dist-packages/nvidia/), and ptxas comes from triton.
# A plain ubuntu base is ~3-7 GB smaller than the nvidia/cuda:*-runtime/-devel bases.
#
# CUDA 13 only. No forward-compat libcuda is bundled, so the runner's driver
# must itself support CUDA 13.

# Build-time helpers, bind-mounted into the RUN steps below; never part of the image.
FROM scratch AS helpers
COPY --chmod=755 tools/ci/docker/mkstub.sh /mkstub
COPY --chmod=755 <<'BASH' /strip-elf
#!/bin/bash
# strip-elf PATH...: drop debug symbols from every shared object under each
# directory and from each named binary. Paths that do not exist are skipped.
set -euo pipefail
for path in "$@"; do
    if [ -d "$path" ]; then
        find -H "$path" -type f \( -name '*.so' -o -name '*.so.*' \) -print0
    elif [ -f "$path" ]; then
        printf '%s\0' "$path"
    fi
done | xargs -0 -r -P"$(nproc)" -n 16 sh -c '
    for f in "$@"; do
        strip --strip-unneeded "$f" 2>/dev/null || true
    done
' _
BASH

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        python3 python3.12 python3.12-dev python3-pip \
        git ca-certificates gcc && \
    rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.15.0
# The wheel index pyproject.toml's `cu13` extra and the README install line use.
ARG TORCH_CUDA=cu132

# Install pinned stable torch from the matching cu wheel index. Torch's wheel
# brings cublas/cudnn/cudart/etc. and triton brings ptxas — together these are
# everything cute-dsl needs to JIT. cusolver isn't linked from libtorch_cuda
# (we don't use torch.linalg here), so we drop it in the same RUN to shrink
# the layer (~hundreds of MB). Also dropped: Triton's static profiler archives
# and its AMD code generator (only the library; Triton imports every backend's
# Python modules at start-up and loads this library lazily), and nvJitLink.
#
# Libraries libtorch_cuda hard-links but the suite never calls are replaced by
# stubs (see mkstub.sh): cuDNN (whose engine libraries, ~850 MB, are deleted),
# cuSPARSE, cuSPARSELt, cuFFT, NCCL, nvshmem. Calling into one of them kills
# the process with SIGILL. Kept real: cuBLAS (tests use torch matmul as their
# reference) and nvrtc (torch JIT-compiles ops such as torch.exp2 with it).
# Consequence: the multi-GPU tests (test_distributed_gemm.py,
# test_blockscaled_distributed_gemm.py) cannot run in this image. CI never runs
# them: each xdist worker sees one GPU, so their two-GPU guard skips.
#
# Bytecode (.pyc) is intentionally kept so worker startup stays fast.
RUN --mount=type=bind,from=helpers,source=/strip-elf,target=/usr/local/bin/strip-elf \
    --mount=type=bind,from=helpers,source=/mkstub,target=/usr/local/bin/mkstub \
    pip install --break-system-packages --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA} && \
    cd /usr/local/lib/python3.12/dist-packages && \
    rm -rf nvidia/cu13/lib/libcusolver*.so.* \
           nvidia/cu13/lib/libnvJitLink.so.* \
           nvidia/cudnn/lib/libcudnn_*.so.* \
           triton/backends/nvidia/lib/cupti*/*_static.a \
           triton/backends/amd/lib \
           nvidia/cu13/lib/libcufftw.so.* \
           nvidia/nvshmem/lib/*.bc nvidia/nvshmem/lib/*.a && \
    strip-elf "$PWD" \
              "$PWD/torch/bin/protoc" "$PWD"/torch/bin/protoc-* \
              "$PWD/triton/FileCheck" && \
    hardlink -t -s 4MiB triton/backends/nvidia/lib && \
    mkstub nvidia/cu13/lib/libcusparse.so.12 nvidia/cusparselt/lib/libcusparseLt.so.0 \
           nvidia/cu13/lib/libcufft.so.12 nvidia/nccl/lib/libnccl.so.2 \
           nvidia/nvshmem/lib/libnvshmem_host.so.3 nvidia/cudnn/lib/libcudnn.so.9 && \
    python3 -c "import torch; print('torch', torch.__version__, 'imports with stubs')"

# Install quack-kernels' transitive dependencies. The package itself is
# overwritten at run time by an editable install from the mounted repo.
# Install only the test runner here; lint runs in its own workflow job.
#
# Note on cu13 install order: nvidia-cutlass-dsl-libs-cu13 is an OVERLAY on top
# of -libs-base — both wheels write the same nvidia_cutlass_dsl/lib/*.so paths,
# and upstream's design is for cu13 to win by virtue of being installed last.
# We use pip (not uv) here because pip's topological installer always installs
# dependencies before dependents, so -libs-cu13 (which depends on -libs-base)
# is guaranteed to install last and its files deterministically overlay -libs-base.
# Switching to uv pip here re-introduces non-deterministic install order.
#
# cutlass-dsl always installs its CUDA 12 flavour next to cu13; this image is
# CUDA 13 only, so remove it (both the IR module and its cu12/ runtime dir, or
# the loader reports an incomplete install).
RUN --mount=type=bind,from=helpers,source=/strip-elf,target=/usr/local/bin/strip-elf \
    --mount=type=bind,source=pyproject.toml,target=/tmp/src/pyproject.toml \
    --mount=type=bind,source=quack,target=/tmp/src/quack \
    cp -r /tmp/src /tmp/quack && \
    pip install --break-system-packages --no-cache-dir '/tmp/quack[cu13]' pytest pytest-xdist && \
    rm -rf /tmp/quack && \
    cd /usr/local/lib/python3.12/dist-packages && \
    rm -rf nvidia_cutlass_dsl/cu12 \
           nvidia_cutlass_dsl/dsl_packages/cutlass/_mlir/_mlir_libs/_cutlass_ir.cu12.* \
           nvidia_cutlass_dsl/dsl_packages/cutlass/_mlir/_mlir_libs/_cutlass_ir_version_cu12.py && \
    strip-elf "$PWD/nvidia_cutlass_dsl" "$PWD"/numpy* "$PWD/tvm_ffi"

# Enroot/Pyxis (and Docker's nvidia runtime) only inject the GPU driver into
# images that ask for it; Apptainer's --nv ignores these.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["/bin/bash"]
