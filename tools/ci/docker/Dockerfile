# All CUDA runtime libs come from torch's pip wheels (under
# /usr/local/lib/python3.12/dist-packages/nvidia/), and ptxas comes from triton.
# A plain ubuntu base is ~3-7 GB smaller than the nvidia/cuda:*-runtime/-devel bases.
FROM ubuntu:24.04 AS base

# Defaults match build.sh's cu12.9 invocation so a bare `docker build` (no
# --build-arg) produces a working image on our cu12.9 runners. Pass
# --build-arg QUACK_EXTRAS=cu13,dev (plus --target cu13) for the cu13.2 variant.
ARG TORCH_CUDA=cu126
ARG QUACK_EXTRAS=dev

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y \
        python3 python3.12 python3.12-dev python3.12-venv python3-pip \
        git ca-certificates gcc && \
    rm -rf /var/lib/apt/lists/*

# Install latest stable torch from the matching cu wheel index. Torch's wheel
# brings cublas/cudnn/cudart/etc. and triton brings ptxas — together these are
# everything cute-dsl needs to JIT. cusolver isn't linked from libtorch_cuda
# (we don't use torch.linalg here), so we drop it in the same RUN to shrink
# the layer (~470 MB on cu12.6, ~235 MB on cu13.x).
#
# Note: build.sh pins TORCH_CUDA=cu126 for the cu12.9 image (PyTorch 2.12
# skipped cu128/cu129 wheels; cu126 is the newest 12.x index that ships 2.12
# and is runnable against any CUDA 12.x driver) and TORCH_CUDA=cu132 for the
# cu13.2 image (matches the cu13 cute-dsl backend and the cu13.x compat
# shim added by the `cu13` target). The cu13 target's compat libcuda.so
# is what makes cu132 torch runnable on older kernel drivers like 575.x.
RUN pip install --break-system-packages --no-cache-dir \
        "torch==2.12.1" \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA} && \
    rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cusolver \
           /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcusolver*.so.*

# Install quack-kernels' transitive dependencies. The package itself is
# overwritten at run time by an editable install from the mounted repo.
#
# Note on cu13 install order: nvidia-cutlass-dsl-libs-cu13 is an OVERLAY on top
# of -libs-base — both wheels write the same nvidia_cutlass_dsl/lib/*.so paths,
# and upstream's design is for cu13 to win by virtue of being installed last.
# We use pip (not uv) here because pip's topological installer always installs
# dependencies before dependents, so -libs-cu13 (which depends on -libs-base)
# is guaranteed to install last and its files deterministically overlay -libs-base.
# Switching to uv pip here re-introduces non-deterministic install order.
COPY pyproject.toml /tmp/quack/pyproject.toml
COPY quack/ /tmp/quack/quack/
RUN pip install --break-system-packages --no-cache-dir "/tmp/quack[${QUACK_EXTRAS}]"

CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# cu13 variant: layer CUDA 13.x forward-compatibility libs on top of base.
#
# `cute-dsl[cu13]`'s runtime checks whether the loaded user-mode libcuda.so
# advertises CUDA 13.x; the kernel-mode driver only needs to be a matching
# (or newer) tier. CUDA forward compatibility is exactly that arrangement:
# a user-mode libcuda.so.590.* shim talking down to an older kernel driver
# (e.g. 575). Without these libs cu13 cute-dsl errors out at first JIT with
# `cudaErrorInsufficientDriver` even though kernels would run fine.
#
# We pull `/usr/local/cuda/compat/` from nvidia/cuda:13.1.0-base-ubuntu24.04
# (~100 MB, mostly libcuda.so.590) and prepend it to ld.so.conf so it wins
# over the host's libcuda.so. Only the cu13 build target gets this, keeping
# the cu12.9 image slim.
# ---------------------------------------------------------------------------
FROM base AS cu13

COPY --from=nvidia/cuda:13.1.0-base-ubuntu24.04 /usr/local/cuda/compat /usr/local/cuda/compat
RUN echo "/usr/local/cuda/compat" > /etc/ld.so.conf.d/000-cuda-compat.conf && ldconfig

# Under `docker run --gpus`, the host driver libs are bind-mounted under
# /usr/lib/x86_64-linux-gnu/ and ld.so.conf's `/usr/local/cuda/compat` entry
# above is what wins. Under `apptainer exec --nv`, however, the host's libcuda
# is dropped into /.singularity.d/libs and that dir is prepended to
# LD_LIBRARY_PATH, which would beat ld.so.conf. Setting LD_LIBRARY_PATH here
# in the image puts the compat dir FIRST; apptainer then appends
# /.singularity.d/libs to it, so /usr/local/cuda/compat/libcuda.so.590.* still
# wins. Harmless for `docker run` (libcuda.so.590.* would have won anyway).
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat
