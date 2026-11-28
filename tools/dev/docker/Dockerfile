# syntax=docker/dockerfile:1.7
# General amd64 CUDA/PyTorch kernel-development image.
#
# Design notes:
# - Use Ubuntu as the final stage rather than nvidia/cuda:*-(base|devel). The
#   CUDA images are convenient, but they bake in large defaults such as CUDA
#   compat payloads and broad devel/runtime packages. Starting from Ubuntu lets
#   us install only the CUDA apt packages needed for kernel work.
# - Use apt for the latest CUDA compiler/profiler/debugger stack. Python runtime
#   libraries come from the official PyTorch cu134 dependency wheel set, so
#   normal `pip install ...` inside the container keeps a complete CUDA
#   dependency graph. cuBLAS exists once: the wheel is pinned to the toolkit's
#   version and the toolkit links against it, instead of installing a second
#   609 MB copy from apt.
# - No CUDA forward-compat libcuda is bundled: the host driver must itself
#   support CUDA 13 (the 13.1 shim for 575-series drivers is gone).
# - Keep this image Apptainer-friendly: no baked-in non-root user, writable
#   workspaces are expected to be bind-mounted at runtime.
#
# Build (needs RUN heredocs, cache and bind mounts: BuildKit, i.e. docker buildx,
# or Podman/Buildah. The legacy docker builder cannot):
#   tools/dev/docker/build.sh [--image] [--push] [--sif] [--sqsh]
#   (tags tridao/quack-kernels-dev:YY.MM.DD; the script itself calls docker)
# or by hand:
#   docker build --platform linux/amd64 -f tools/dev/docker/Dockerfile -t kernel-dev:cu135 .
#   podman build --format docker --platform linux/amd64 -f tools/dev/docker/Dockerfile -t kernel-dev:cu135 .
# Keep --format docker: in the default OCI format, Podman before 6.0 ignores SHELL,
# so the RUN heredocs run under /bin/sh and fail on `set -o pipefail` (Podman 5.8.9
# fails without it; 6.2 builds either way but still drops SHELL from the image).
#
# Every version pin is a build arg (`--build-arg NAME=value`); the ARG lines
# below are the single source of truth for the defaults.

ARG BASE_IMAGE=ubuntu:24.04

# Build-time helpers, bind-mounted into the RUN steps below; never part of the image.
FROM scratch AS helpers
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
done | xargs -0 -r -P"$(nproc)" -n 16 bash -c '
    for elf in "$@"; do
        if file "$elf" | grep -q "ELF.*not stripped"; then
            strip --strip-unneeded "$elf" || true
        fi
    done
' _
BASH

FROM ${BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COMPILE=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    CUDA_HOME=/usr/local/cuda \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64

# ---- System layer: apt toolchain, CUDA compiler/profiler/debugger ----------
#
# ARGs are declared right before the RUN that consumes them, not at the top of
# the stage: every ARG in scope is part of the cache key of all later RUNs, so
# a Python pin declared up here would rebuild this multi-GB layer on each bump.
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG CUDA_APT_SUFFIX=13-5

# Plain Ubuntu has no CUDA apt repo configured, so install the CUDA keyring
# explicitly. The CUDA package list is intentionally selective: it keeps the
# compiler/profiler/debugger path for kernel development without pulling the
# entire nvidia/cuda:devel image. Cleanup stays in this same layer so removed
# Nsight GUI/docs/cross-arch files, static archives, pyc files, and stripped
# debug symbols do not survive in lower Docker layers. sqlite3 is intentionally kept:
# it is small and useful for inspecting profiler traces, PyTorch/Inductor caches,
# and other tool-generated SQLite databases.
RUN --mount=type=bind,from=helpers,source=/strip-elf,target=/usr/local/bin/strip-elf <<'BASH'
set -euo pipefail
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg
curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        ccache \
        cmake \
        cuda-command-line-tools-${CUDA_APT_SUFFIX} \
        cuda-minimal-build-${CUDA_APT_SUFFIX} \
        cuda-nsight-compute-${CUDA_APT_SUFFIX} \
        cuda-nsight-systems-${CUDA_APT_SUFFIX} \
        fd-find \
        file \
        gdb \
        git \
        git-lfs \
        htop \
        jq \
        less \
        ltrace \
        mosh \
        ninja-build \
        numactl \
        openssh-client \
        openssh-server \
        patchelf \
        pciutils \
        pkg-config \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        python-is-python3 \
        ripgrep \
        rsync \
        sollya \
        sqlite3 \
        strace \
        tmux \
        tree \
        unison \
        unzip \
        vim \
        wget \
        zip \
        zsh
ln -sf /usr/bin/fdfind /usr/local/bin/fd
mkdir -p /run/sshd
rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub

# Keep the CLI profilers, not their GUI, documentation, or cross-arch targets.
rm -rf \
        /opt/nvidia/nsight-compute/*/docs \
        /opt/nvidia/nsight-compute/*/host \
        /opt/nvidia/nsight-compute/*/target/linux-desktop-t210-a64 \
        /opt/nvidia/nsight-compute/*/target/linux-desktop-glibc_2_11_3-x86 \
        /opt/nvidia/nsight-compute/*/extras/python \
        /opt/nvidia/nsight-compute/*/extras/samples \
        /opt/nvidia/nsight-systems/*/documentation \
        /opt/nvidia/nsight-systems/*/host-linux-x64 \
        /opt/nvidia/nsight-systems/*/target-linux-x64/samples \
        /usr/local/cuda/compute-sanitizer/docs \
        /usr/local/cuda/extras/CUPTI/doc \
        /usr/local/cuda/extras/CUPTI/samples
# nvprune (47 MB, prunes fat objects and static libraries) is left out for size.
rm -f \
        /usr/local/cuda/targets/x86_64-linux/lib/libcupti_static.a \
        /usr/local/cuda/targets/x86_64-linux/lib/libnvperf_host_static.a \
        /usr/local/cuda/targets/x86_64-linux/lib/libnvptxcompiler_static.a \
        /usr/local/cuda/bin/nvprune
# cuda-gdb ships one TUI build per Python version; keep the one for this image's Python.
find /usr/local/cuda/bin/ -name 'cuda-gdb-python3.*-tui' \
    ! -name "cuda-gdb-python${PYTHON_VERSION}-tui" -delete
find /usr /opt/nvidia -type d -name __pycache__ -prune -exec rm -rf {} +
strip-elf /usr/local/cuda /opt/nvidia \
    /usr/local/cuda/bin/cuda-gdb-minimal \
    /usr/local/cuda/bin/cuda-gdb-python${PYTHON_VERSION}-tui \
    /opt/nvidia/nsight-systems/*/target-linux-x64/plugins/efa_metrics/nic_sampler
# Preserve every path; coalesce identical large files only within this package.
hardlink -t -s 4MiB /opt/nvidia/nsight-systems
echo '/usr/local/cuda/nvvm/lib64' > /etc/ld.so.conf.d/010-cuda-nvvm.conf
ldconfig
rm -f /tmp/cuda-keyring.deb /usr/local/bin/nsys-ui /usr/local/cuda/bin/ncu-ui
rm -rf /var/lib/apt/lists/*
BASH

ENV PATH=/usr/local/cuda/bin:${PATH}

# ---- Python layer: torch, CuTe-DSL, dev tooling ----------------------------
#
# cu134 is the newest official CUDA wheel for PyTorch 2.15.0.
ARG TORCH_CUDA=cu134
ARG TORCH_VERSION=2.15.0
ARG CUDNN_VERSION=9.28.1.4
# The cuBLAS that CUDA_APT_SUFFIX's toolkit ships (CUDA 13.5: 13.10.0.20); torch
# pins an older one. Bump together with CUDA_APT_SUFFIX.
ARG CUBLAS_VERSION=13.10.0.20
ARG CUDNN_FRONTEND_VERSION=1.33.0
ARG CUDA_PYTHON_VERSION=13.5.0
ARG CUTLASS_DSL_VERSION=4.8.1
ARG TILELANG_VERSION=0.1.15
ARG MATMUL_HEURISTICS_VERSION=0.1.0.27
ARG NSIGHT_PYTHON_VERSION=1.1.1
ARG COMPILEIQ_VERSION=1.1.0
ARG UV_VERSION=0.13.9
# Match .pre-commit-config.yaml and CI to keep local formatting consistent.
ARG RUFF_VERSION=0.17.6
# From PyPI: a 2 MB static binary, where apt's clang-format pulls in 180 MB of LLVM.
ARG CLANG_FORMAT_VERSION=18.1.8

# Install Python packages in one layer so replaced/stripped files do not remain
# in lower Docker layers. Torch is installed normally from the official PyTorch
# cu134 index, including Triton and CUDA wheel dependencies. Override cuDNN to
# CUDNN_VERSION for newer frontend testing and cuBLAS to the toolkit's
# CUBLAS_VERSION, then patch torch's installed METADATA so pip's resolver and
# `pip check` treat those versions as the expected ones.
#
# Install basic nsight-python, not nsight-python[cu13], so cuda-core[cu13] does
# not pull an additional CUDA toolkit stack.
#
# PYTHONDONTWRITEBYTECODE is set for this build step only: the image ships no
# .pyc beyond the hot path baked at the end, yet Python caches bytecode normally
# at run time (notably for the bind-mounted repo).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=helpers,source=/strip-elf,target=/usr/local/bin/strip-elf <<'BASH'
set -euo pipefail
export UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1
python${PYTHON_VERSION} -m pip install 'setuptools<82' wheel packaging "uv==${UV_VERSION}"
uv pip install --system --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    "torch==${TORCH_VERSION}"
uv pip install --system "nvidia-cudnn-cu13==${CUDNN_VERSION}" "nvidia-cublas==${CUBLAS_VERSION}"
python${PYTHON_VERSION} - <<'PY'
from importlib.metadata import distribution, version
import re

torch = distribution("torch")
meta, = (f.locate() for f in torch.files if f.match("*.dist-info/METADATA"))
text = meta.read_text()
for name in ("nvidia-cudnn-cu13", "nvidia-cublas"):
    text, count = re.subn(
        rf'^(Requires-Dist: {re.escape(name)})[<>=!~][^;\s]*(?=; platform_system == "Linux"$)',
        rf"\g<1>=={version(name)}", text, flags=re.MULTILINE,
    )
    assert count == 1, f"expected one torch pin for {name}, found {count}"
meta.write_text(text)
PY

uv pip install --system \
    "clang-format==${CLANG_FORMAT_VERSION}" \
    "compileiq==${COMPILEIQ_VERSION}" \
    "cuda-python==${CUDA_PYTHON_VERSION}" \
    "nsight-python==${NSIGHT_PYTHON_VERSION}" \
    "nvidia-cudnn-frontend==${CUDNN_FRONTEND_VERSION}" \
    "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" \
    "nvidia-matmul-heuristics==${MATMUL_HEURISTICS_VERSION}" \
    "ruff==${RUFF_VERSION}" \
    "tilelang==${TILELANG_VERSION}" \
    cloudpickle einops ipdb ipython loguru matplotlib nvitop numpy pandas pre-commit py-spy pyelftools \
    pytest pytest-benchmark pytest-timeout pytest-xdist ty tyro
uv pip check --system

packages=/usr/local/lib/python${PYTHON_VERSION}/dist-packages
# Wheel payload this image has no use for: static profiler archives, Triton's AMD
# code generator, the multi-GPU cuSOLVER that nothing links against, and
# cutlass-dsl's CUDA 12 flavour (this image is CUDA 13 only; both the IR module
# and cu12/ runtime dir must go together or the loader reports an incomplete install).
rm -rf "$packages"/triton/backends/nvidia/lib/cupti*/{libcupti_static,libnvperf_host_static}.a \
    "$packages"/triton/backends/amd/lib \
    "$packages"/nvidia/cu13/lib/libcusolverMg.so.* \
    "$packages"/nvidia_cutlass_dsl/cu12 \
    "$packages"/nvidia_cutlass_dsl/dsl_packages/cutlass/_mlir/_mlir_libs/_cutlass_ir.cu12.* \
    "$packages"/nvidia_cutlass_dsl/dsl_packages/cutlass/_mlir/_mlir_libs/_cutlass_ir_version_cu12.py

# One cuBLAS, not two. Expose the wheel's libraries and headers to the toolkit so
# `nvcc ... -lcublas` builds against the same cuBLAS torch runs, then prove it links.
cublas=$packages/nvidia/cu13
for lib in "$cublas"/lib/lib{cublas,cublasLt,nvblas}.so.*; do
    soname=${lib##*/}
    ln -s "$lib" "/usr/local/cuda/lib64/$soname"
    ln -s "$soname" "/usr/local/cuda/lib64/${soname%.so.*}.so"
done
ln -s "$cublas"/include/{cublas,cublas_api,cublas_v2,cublasLt,cublasXt,nvblas}.h /usr/local/cuda/include/
# `ln -s` happily links to nothing; fail here if the wheel's layout ever changes.
if find -L /usr/local/cuda/lib64/ /usr/local/cuda/include/ -maxdepth 1 -name '*blas*' -type l | grep .; then
    echo "dangling cuBLAS links: the nvidia-cublas wheel layout changed" >&2
    exit 1
fi
ldconfig
echo 'int main() { cublasHandle_t h; return cublasCreate(&h) != CUBLAS_STATUS_SUCCESS; }' |
    g++ -x c++ -include cublas_v2.h - -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas \
        -o /tmp/cublas-link-test
rm /tmp/cublas-link-test

find "$packages" -type d -name __pycache__ -prune -exec rm -rf {} +
strip-elf "$packages" /usr/local/bin/{py-spy,z3} \
    "$packages"/torch/bin/protoc "$packages"/torch/bin/protoc-* "$packages"/triton/FileCheck
# Keep independent packages independent when users upgrade one inside the image.
for dir in "$packages/z3/lib" "$packages/triton/backends/nvidia/lib"; do
    hardlink -t -s 4MiB "$dir"
done

# Query the loaded library, not just its wheel metadata. No GPU required.
python${PYTHON_VERSION} - <<'PY'
import os
import torch

major, minor, patch, *_ = map(int, os.environ["CUDNN_VERSION"].split("."))
expected = major * 10000 + minor * 100 + patch
loaded = torch.backends.cudnn.version()
assert loaded == expected, f"torch loaded cuDNN {loaded}, expected {expected}"
PY

# Bake bytecode for the imports nearly every session starts with (about 41 MB).
# A read-only SIF can never cache it and every fresh container starts without
# it; baked, those imports take about 2.2 s instead of 5.5 s.
PYTHONDONTWRITEBYTECODE= python${PYTHON_VERSION} -c 'import torch, numpy, cutlass.cute, pytest'
BASH

# Independent CLI: a version bump needn't reinstall CUDA or torch. dpkg does not
# resolve dependencies; gh's only one is git, installed by the system layer.
ARG GH_VERSION=2.105.0
RUN curl -fsSL -o /tmp/gh.deb \
        "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.deb" && \
    dpkg -i /tmp/gh.deb && rm /tmp/gh.deb

RUN mkdir -p /workspace && chmod 0777 /workspace

WORKDIR /workspace

CMD ["/bin/bash"]
