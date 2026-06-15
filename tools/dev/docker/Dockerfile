# syntax=docker/dockerfile:1.7
# General amd64 CUDA/PyTorch kernel-development image.
#
# Design notes:
# - Use Ubuntu as the final stage rather than nvidia/cuda:*-(base|devel). The
#   CUDA images are convenient, but they bake in large defaults such as CUDA
#   compat payloads and broad devel/runtime packages. Starting from Ubuntu lets
#   us install only the CUDA apt packages needed for kernel work.
# - Use apt for the latest CUDA compiler/profiler/debugger stack. Python runtime
#   libraries still come from the official PyTorch cu132 dependency wheel set
#   until PyTorch publishes cu133 wheels, so normal `pip install ...` inside the
#   container does not try to repair a hand-pruned no-deps torch install.
# - Keep this image Apptainer-friendly: no baked-in non-root user, writable
#   workspaces are expected to be bind-mounted at runtime.
#
# Build:
#   docker build -f tools/dev/docker/Dockerfile -t kernel-dev:cu133 .
#
# Useful build args:
#   --build-arg CUTLASS_DSL_VERSION=4.5.2
#   --build-arg TILELANG_VERSION=0.1.10
#   --build-arg CUDA_APT_SUFFIX=13-3
#   --build-arg CUDNN_VERSION=9.22.0.52
#   --build-arg CUDNN_FRONTEND_VERSION=1.24.0
#   --build-arg CUDA_PYTHON_VERSION=13.3.0
#   --build-arg MATMUL_HEURISTICS_VERSION=0.1.0.27
#   --build-arg NSIGHT_PYTHON_VERSION=0.9.6
#   --build-arg COMPILEIQ_VERSION=1.0.0
#   --build-arg UV_VERSION=0.11.16
#   --build-arg GH_VERSION=2.91.0

ARG BASE_IMAGE=ubuntu:24.04
ARG CUDA_COMPAT_IMAGE=nvidia/cuda:13.1.0-base-ubuntu24.04

FROM ${CUDA_COMPAT_IMAGE} AS cuda131-compat
FROM ${BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG CUDA_APT_SUFFIX=13-3
ARG TORCH_CUDA=cu132
ARG TORCH_VERSION=2.12.0
ARG CUDNN_VERSION=9.22.0.52
ARG CUDNN_FRONTEND_VERSION=1.24.0
ARG CUDA_PYTHON_VERSION=13.3.0
ARG MATMUL_HEURISTICS_VERSION=0.1.0.27
ARG NSIGHT_PYTHON_VERSION=0.9.6
ARG COMPILEIQ_VERSION=1.0.0
ARG UV_VERSION=0.11.16
ARG CUTLASS_DSL_VERSION=4.5.2
ARG TILELANG_VERSION=0.1.10
ARG GH_VERSION=2.91.0

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COMPILE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_DISABLE_REQUIRE=1 \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64

# Newer CUDA forward-compat libcuda payloads can be too new for 575-series
# kernel drivers. Copy only the CUDA 13.1 compat shim so libcuda.so.590 wins
# under Docker/Apptainer on 575-series hosts, while avoiding newer compat
# payloads from NVIDIA's CUDA base images.
COPY --from=cuda131-compat /NGC-DL-CONTAINER-LICENSE /
COPY --from=cuda131-compat /usr/local/cuda/compat /opt/cuda131-compat
ENV LD_LIBRARY_PATH=/opt/cuda131-compat:${LD_LIBRARY_PATH}
RUN echo "/opt/cuda131-compat" > /etc/ld.so.conf.d/000-cuda-compat.conf && ldconfig

# Plain Ubuntu has no CUDA apt repo configured, so install the CUDA keyring
# explicitly. The CUDA package list is intentionally selective: it keeps the
# compiler/profiler/debugger path for kernel development without pulling the
# entire nvidia/cuda:devel image. Cleanup stays in this same layer so removed
# Nsight GUI/docs/cross-arch files, static archives, pyc files, and stripped
# debug symbols do not survive in lower Docker layers. sqlite3 is intentionally kept:
# it is small and useful for inspecting profiler traces, PyTorch/Inductor caches,
# and other tool-generated SQLite databases.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
    curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    apt-get update && \
    curl -fsSL -o /tmp/gh.deb \
        "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.deb" && \
    apt-get install -y --no-install-recommends \
        /tmp/gh.deb \
        bash-completion \
        build-essential \
        ccache \
        clang-format \
        cmake \
        cuda-command-line-tools-${CUDA_APT_SUFFIX} \
        cuda-minimal-build-${CUDA_APT_SUFFIX} \
        libcublas-dev-${CUDA_APT_SUFFIX} \
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
        zsh && \
    ln -sf /usr/bin/fdfind /usr/local/bin/fd && \
    mkdir -p /run/sshd && \
    rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub && \
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
        /usr/local/cuda/extras/CUPTI/samples && \
    rm -f \
        /usr/local/cuda/targets/x86_64-linux/lib/libcublas*_static.a \
        /usr/local/cuda/targets/x86_64-linux/lib/libcupti_static.a \
        /usr/local/cuda/targets/x86_64-linux/lib/libnvperf_host_static.a \
        /usr/local/cuda/targets/x86_64-linux/lib/libnvptxcompiler_static.a \
        /usr/local/cuda/bin/cuda-gdb-python3.{8,9,10,11}-tui && \
    find /usr /opt/nvidia -type d -name __pycache__ -prune -exec rm -rf {} + && \
    find -H /usr/local/cuda /opt/nvidia -type f \( -name '*.so' -o -name '*.so.*' \) -print0 | \
        while IFS= read -r -d '' lib; do \
            if file "$lib" | grep -q 'ELF.*not stripped'; then strip --strip-unneeded "$lib" || true; fi; \
        done && \
    for bin in \
        /usr/local/cuda/bin/cuda-gdb-minimal \
        /usr/local/cuda/bin/cuda-gdb-python3.12-tui \
        /opt/nvidia/nsight-systems/*/target-linux-x64/plugins/efa_metrics/nic_sampler; do \
            if [ -f "$bin" ]; then strip --strip-unneeded "$bin" || true; fi; \
        done && \
    echo "/usr/local/cuda/nvvm/lib64" > /etc/ld.so.conf.d/010-cuda-nvvm.conf && \
    ldconfig && \
    rm -f /tmp/gh.deb /tmp/cuda-keyring.deb /usr/local/bin/nsys-ui /usr/local/cuda/bin/ncu-ui && \
    rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}

# Install Python packages in one layer so replaced/stripped files do not remain
# in lower Docker layers. Torch is installed normally from the official PyTorch
# cu132 index, including Triton and CUDA wheel dependencies. PyTorch has no
# cu133 wheels yet, and a no-deps torch install makes later interactive
# `pip install ...` try to repair the missing CUDA dependency graph. We still
# override cuDNN to CUDNN_VERSION for
# newer frontend testing, then patch torch's installed METADATA so pip's resolver
# and `pip check` treat that cuDNN version as the expected one.
#
# Install basic nsight-python, not nsight-python[cu13], so cuda-core[cu13] does
# not pull an additional CUDA toolkit stack. Use pip for CUTLASS DSL: uv can
# install the cu13 overlay in the wrong order
# (https://github.com/NVIDIA/cutlass/issues/3259).
RUN python -m pip install --upgrade pip "setuptools<82" wheel packaging "uv==${UV_VERSION}" && \
    export UV_CACHE_DIR=/tmp/uv-cache && \
    uv pip install \
        einops \
        ipdb \
        ipython \
        loguru \
        matplotlib \
        "nsight-python==${NSIGHT_PYTHON_VERSION}" \
        "cuda-python==${CUDA_PYTHON_VERSION}" \
        "compileiq==${COMPILEIQ_VERSION}" \
        "nvidia-cudnn-frontend==${CUDNN_FRONTEND_VERSION}" \
        "nvidia-matmul-heuristics==${MATMUL_HEURISTICS_VERSION}" \
        nvitop \
        numpy \
        pandas \
        pre-commit \
        py-spy \
        pyelftools \
        pytest \
        pytest-benchmark \
        pytest-timeout \
        pytest-xdist \
        ruff \
        ty && \
    uv pip install \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
        "torch==${TORCH_VERSION}" && \
    uv pip install "nvidia-cudnn-cu13==${CUDNN_VERSION}" && \
    CUDNN_VERSION="${CUDNN_VERSION}" python -c 'from importlib.metadata import distribution; from pathlib import Path; import os, re; version=os.environ["CUDNN_VERSION"]; meta=Path(distribution("torch")._path)/"METADATA"; text=meta.read_text(); pattern=re.compile("^(Requires-Dist: nvidia-cudnn-cu13==)([^;\\s]+)(; platform_system == "+chr(34)+"Linux"+chr(34)+")$", re.M); new,count=pattern.subn(lambda m: f"{m.group(1)}{version}{m.group(3)}", text); assert count == 1, f"patched {count} torch cuDNN dependency lines"; meta.write_text(new)' && \
    python -m pip install "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" && \
    uv pip install "tilelang==${TILELANG_VERSION}" && \
    python -m pip check && \
    rm -rf /tmp/uv-cache && \
    rm -f \
        /opt/venv/lib/python${PYTHON_VERSION}/site-packages/triton/backends/nvidia/lib/cupti/libcupti_static.a \
        /opt/venv/lib/python${PYTHON_VERSION}/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_host_static.a && \
    find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + && \
    find /opt/venv -type f \( -name '*.so' -o -name '*.so.*' \) -print0 | \
        while IFS= read -r -d '' lib; do \
            if file "$lib" | grep -q 'ELF.*not stripped'; then strip --strip-unneeded "$lib" || true; fi; \
        done && \
    for bin in \
        /opt/venv/bin/py-spy \
        /opt/venv/bin/z3 \
        /opt/venv/lib/python${PYTHON_VERSION}/site-packages/torch/bin/protoc \
        /opt/venv/lib/python${PYTHON_VERSION}/site-packages/torch/bin/protoc-3.13.0.0 \
        /opt/venv/lib/python${PYTHON_VERSION}/site-packages/triton/FileCheck; do \
            if [ -f "$bin" ]; then strip --strip-unneeded "$bin" || true; fi; \
        done

RUN mkdir -p /workspace && chmod 0777 /workspace

WORKDIR /workspace

CMD ["/bin/bash"]
