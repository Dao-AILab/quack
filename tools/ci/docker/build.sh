#!/bin/bash
# Build/push the quack CI docker image (CUDA 13) and import it for Apptainer or Enroot.
#
# Usage: build.sh [--image] [--push] [--sif] [--sqsh]
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
#   --sqsh   import the same image as an Enroot SquashFS into
#            ${SQSH_DIR:-${CI_WORK_DIR:-$HOME}} for use with Pyxis.
#
# Typical invocations:
#   build.sh                       # build the image (build box)
#   build.sh --image --push --sif  # full pipeline on a runner with Apptainer
#   DATE=26.08.07 build.sh --sif   # pre-warm the SIF on the b300 runner
#   DATE=26.08.07 build.sh --sqsh  # pre-build the image on an Enroot host
#
# DATE (YY.MM.DD) defaults to today; override it when the image was built or
# pushed on an earlier day.
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
ZSTD_LEVEL="${ZSTD_LEVEL:-9}"

DO_IMAGE=0
DO_PUSH=0
DO_SIF=0
DO_SQSH=0
for arg in "$@"; do
    case "$arg" in
        --image)  DO_IMAGE=1 ;;
        --push)   DO_PUSH=1 ;;
        --sif)    DO_SIF=1 ;;
        --sqsh)   DO_SQSH=1 ;;
        *)  echo "Unknown argument: $arg (expected --image, --push, --sif, --sqsh)" >&2
            exit 1 ;;
    esac
done
# No step flags = build the image.
if [ $((DO_IMAGE + DO_PUSH + DO_SIF + DO_SQSH)) -eq 0 ]; then
    DO_IMAGE=1
fi

run_variant() {
    local tag=$1 torch_cuda=$2
    local image="quack-kernels:${tag}-${DATE}"
    local remote_image="${REGISTRY_REPO}:${tag}-${DATE}"

    if [ "$DO_IMAGE" = 1 ]; then
        echo "=== Building $image (torch $torch_cuda) ==="
        docker build \
            --platform linux/amd64 \
            --build-arg "TORCH_CUDA=$torch_cuda" \
            -t "$image" \
            -f "$SCRIPT_DIR/Dockerfile" \
            "$REPO_ROOT"
        echo "Done: $image"
    fi

    if [ "$DO_PUSH" = 1 ]; then
        # Push OCI zstd-compressed layers via BuildKit (regular `docker push`
        # always re-pushes gzip). Reuses BuildKit's cache when --image ran first.
        echo "=== Pushing $remote_image (OCI zstd level $ZSTD_LEVEL) ==="
        docker buildx build \
            --platform linux/amd64 \
            --build-arg "TORCH_CUDA=$torch_cuda" \
            --output "type=image,name=${remote_image},push=true,oci-mediatypes=true,compression=zstd,compression-level=${ZSTD_LEVEL},force-compression=true" \
            -f "$SCRIPT_DIR/Dockerfile" \
            "$REPO_ROOT"
    fi

    if [ "$DO_SIF" = 1 ]; then
        build_sif "$image" "$remote_image"
    fi

    if [ "$DO_SQSH" = 1 ]; then
        local sqsh_dir="${SQSH_DIR:-${CI_WORK_DIR:-$HOME}}"
        local slug
        slug=$(echo "$remote_image" | tr '/: ' '---')
        build_sqsh "$image" "$remote_image" "$sqsh_dir/$slug.sqsh"
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
        apptainer build --force --mksquashfs-args="$ENROOT_SQUASH_OPTIONS" "$sif.tmp" "$source"
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

if [ "$DO_SIF" = 1 ]; then
    source "$REPO_ROOT/tools/docker/sqsh.env"
fi

if [ "$DO_SQSH" = 1 ]; then
    source "$REPO_ROOT/tools/docker/build_sqsh.sh"
fi

# One image: torch cu132 wheels with cute-dsl[cu13]. The cu13.2 tag names the
# wheel index, which is also what pyproject.toml's `cu13` extra installs. Torch's
# cu13 wheels bundle all nvidia libs under a single nvidia/cu13/ tree. The image
# carries no forward-compat libcuda, so runners need a CUDA 13 driver.
run_variant cu13.2 cu132
