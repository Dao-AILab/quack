#!/bin/bash
# Build/push the quack CI docker images and their Apptainer SIFs.
#
# Usage: build.sh [cu129|cu132|all] [--image] [--push] [--sif]
#
# Step flags select what to produce, in pipeline order (default: --image):
#   --image  docker-build the image locally (quack-kernels:<tag>-<date>)
#   --push   tag as ${REGISTRY_REPO:-tridao/quack-kernels} and push to Docker
#            Hub (one-time login: `docker login -u tridao`, or set
#            DOCKERHUB_TOKEN to log in unattended)
#   --sif    build the Apptainer SIF into ${CI_WORK_DIR:-$HOME}, named exactly
#            as the gpu-test action's cache (tridao-quack-kernels-<tag>-<date>.sif)
#            so the runner skips the Docker Hub pull on the first CI run with
#            the new pin. The SIF is sourced from the local docker daemon when
#            the image exists there, else pulled from Docker Hub — so on a
#            docker-less runner (e.g. the b300 machine) `build.sh --sif` alone
#            pre-warms the CI cache from an already-pushed image.
#
# Typical invocations:
#   build.sh                       # build both images (build box)
#   build.sh --image --push --sif  # full pipeline on the h100 runner
#   DATE=26.08.07 build.sh --sif   # pre-warm SIFs on the b300 runner
#
# DATE (YY.MM.DD) defaults to today; override it when the images were built or
# pushed on an earlier day.
#
# Note: the gpu-test action prunes any *quack-kernels*.sif that does not match
# the tags pinned in .github/workflows/_test.yml. If you pre-build SIFs before
# the pin bump lands on the branch CI runs on, append .hold to the filenames
# (mv x.sif x.sif.hold) and strip it once the bump is pushed.
set -e

DATE="${DATE:-$(date +%y.%m.%d)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

DOCKERHUB_USER="${DOCKERHUB_USER:-tridao}"
REGISTRY_REPO="${REGISTRY_REPO:-${DOCKERHUB_USER}/quack-kernels}"

DO_IMAGE=0
DO_PUSH=0
DO_SIF=0
VARIANT=all
for arg in "$@"; do
    case "$arg" in
        --image)           DO_IMAGE=1 ;;
        --push)            DO_PUSH=1 ;;
        --sif)             DO_SIF=1 ;;
        cu129|cu132|all)   VARIANT=$arg ;;
        *)  echo "Unknown argument: $arg (expected cu129|cu132|all, --image, --push, --sif)" >&2
            exit 1 ;;
    esac
done
# No step flags = build the image.
if [ $((DO_IMAGE + DO_PUSH + DO_SIF)) -eq 0 ]; then
    DO_IMAGE=1
fi

run_variant() {
    local tag=$1 torch_cuda=$2 quack_extras=$3 target=$4
    local image="quack-kernels:${tag}-${DATE}"
    local remote_image="${REGISTRY_REPO}:${tag}-${DATE}"

    if [ "$DO_IMAGE" = 1 ]; then
        echo "=== Building $image (torch $torch_cuda, extras [$quack_extras], target $target) ==="
        docker build \
            --target "$target" \
            --build-arg "TORCH_CUDA=$torch_cuda" \
            --build-arg "QUACK_EXTRAS=$quack_extras" \
            -t "$image" \
            -f "$SCRIPT_DIR/Dockerfile" \
            "$REPO_ROOT"
        echo "Done: $image"
    fi

    if [ "$DO_PUSH" = 1 ]; then
        echo "=== Pushing $remote_image ==="
        docker tag "$image" "$remote_image"
        docker push "$remote_image"
    fi

    if [ "$DO_SIF" = 1 ]; then
        build_sif "$image" "$remote_image"
    fi
}

# Build a SIF for $2 (the remote image name the CI action would pull — its
# slug determines the SIF filename the action looks for:
# $WORK_DIR/$(tr '/: ' '---').sif). Source from the local docker daemon when
# $1 exists there, else from Docker Hub. Build to .tmp and rename atomically,
# mirroring the action, so an interrupted build never leaves a partial SIF
# that a later run would silently reuse.
build_sif() {
    local image=$1 remote_image=$2
    local work_dir="${CI_WORK_DIR:-$HOME}"
    local sif="$work_dir/$(echo "$remote_image" | tr '/: ' '---').sif"
    local source="docker://$remote_image"
    if command -v docker >/dev/null 2>&1 && docker image inspect "$image" >/dev/null 2>&1; then
        source="docker-daemon://$image"
    fi
    echo "=== Building $sif from $source ==="
    # Staging needs room for the uncompressed image; default to /tmp like the
    # gpu-test action, but let APPTAINER_TMPDIR point somewhere roomier when
    # /tmp is tight.
    local tmpdir="${APPTAINER_TMPDIR:-/tmp/apptainer_tmp}"
    mkdir -p "$tmpdir" "$work_dir/apptainer_cache"
    APPTAINER_TMPDIR="$tmpdir" \
    APPTAINER_CACHEDIR="$work_dir/apptainer_cache" \
        apptainer build --force "$sif.tmp" "$source"
    mv "$sif.tmp" "$sif"
    echo "Done: $sif"
}

hub_login() {
    if [ -n "${DOCKERHUB_TOKEN:-}" ]; then
        echo "Logging in to Docker Hub as $DOCKERHUB_USER (via DOCKERHUB_TOKEN)..."
        echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin
    fi
}

if [ "$DO_PUSH" = 1 ]; then
    hub_login
fi

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
case "$VARIANT" in
    cu129)  run_variant cu12.9 cu129 dev base ;;
    cu132)  run_variant cu13.2 cu132 cu13,dev cu13 ;;
    all)    run_variant cu12.9 cu129 dev base
            run_variant cu13.2 cu132 cu13,dev cu13 ;;
esac
