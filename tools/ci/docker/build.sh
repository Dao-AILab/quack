#!/bin/bash
# Build quack CI images for CUDA 12.9 and CUDA 13.2.
# Usage: build.sh [cu129|cu132]   (default: build both)
set -e

DATE=$(date +%y.%m.%d)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

build_image() {
    local tag=$1 torch_cuda=$2 quack_extras=$3 target=$4
    local image="quack-kernels:${tag}-${DATE}"
    echo "=== Building $image (torch $torch_cuda, extras [$quack_extras], target $target) ==="
    docker build \
        --target "$target" \
        --build-arg "TORCH_CUDA=$torch_cuda" \
        --build-arg "QUACK_EXTRAS=$quack_extras" \
        -t "$image" \
        -f "$SCRIPT_DIR/Dockerfile" \
        "$REPO_ROOT"
    echo "Done: $image"
}

# cu12.9 image pins torch to cu129 wheels now that PyTorch 2.13 ships a cu129
# wheel. This keeps the cu12.9 image aligned with its CUDA label while still
# being runnable on driver 575+ unaided.
#
# cu13.2 image pins torch to cu132 wheels and adds the CUDA 13.x forward-
# compatibility libs (the `cu13` Dockerfile target). The user-mode
# libcuda.so.590.* shim from /usr/local/cuda/compat lets cu13 torch + cu13
# cute-dsl JIT successfully on the H100 runner's 575 kernel driver, so
# cu13.2 is a fully testable image — not driver-gated. Bonus: torch's cu13
# wheel bundles all nvidia libs under a single nvidia/cu13/ tree (~1.5 GB
# smaller than cu129's per-lib layout).
case "${1:-all}" in
    cu129)  build_image cu12.9 cu129 dev base ;;
    cu132)  build_image cu13.2 cu132 cu13,dev cu13 ;;
    all)    build_image cu12.9 cu129 dev base
            build_image cu13.2 cu132 cu13,dev cu13 ;;
    *)      echo "Unknown variant: $1 (expected cu129, cu132, or all)" >&2; exit 1 ;;
esac
