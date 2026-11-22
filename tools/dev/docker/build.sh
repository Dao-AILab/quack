#!/bin/bash
# Build/push the kernel-dev docker image and import it for Apptainer or Enroot.
#
# Usage: build.sh [--image] [--push] [--sif] [--sqsh]
#
# Step flags select what to produce, in pipeline order (default: --image):
#   --image  docker-build ${REGISTRY_REPO:-tridao/quack-kernels-dev}:<date>
#   --push   push it to Docker Hub (one-time login: `docker login -u tridao`,
#            or set DOCKERHUB_TOKEN to log in unattended)
#   --sif    build ${SIF_DIR:-$HOME}/kernel-dev-<date>.sif, sourced from the
#            local docker daemon when the image exists there, else pulled from
#            Docker Hub — so on a docker-less machine `build.sh --sif` alone
#            fetches an already-pushed image.
#   --sqsh   import the same image as ${SQSH_DIR:-$HOME}/kernel-dev-<date>.sqsh
#            for use with Enroot/Pyxis.
#
# Typical invocations:
#   build.sh --image --push --sif  # full pipeline on the build box
#   DATE=26.08.07 build.sh --sif   # fetch that day's image as a SIF elsewhere
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
REGISTRY_REPO="${REGISTRY_REPO:-${DOCKERHUB_USER}/quack-kernels-dev}"
IMAGE="${REGISTRY_REPO}:${DATE}"
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

if [ "$DO_IMAGE" = 1 ]; then
    echo "=== Building $IMAGE ==="
    docker build --platform linux/amd64 -f "$SCRIPT_DIR/Dockerfile" -t "$IMAGE" "$REPO_ROOT"
    echo "Done: $IMAGE"
fi

if [ "$DO_PUSH" = 1 ]; then
    if [ -n "${DOCKERHUB_TOKEN:-}" ]; then
        echo "Logging in to Docker Hub as $DOCKERHUB_USER (via DOCKERHUB_TOKEN)..."
        echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin
    fi
    # Push OCI zstd-compressed layers via BuildKit (regular `docker push`
    # always re-pushes gzip). Reuses BuildKit's cache when --image ran first.
    echo "=== Pushing $IMAGE (OCI zstd level $ZSTD_LEVEL) ==="
    docker buildx build \
        --platform linux/amd64 \
        --output "type=image,name=${IMAGE},push=true,oci-mediatypes=true,compression=zstd,compression-level=${ZSTD_LEVEL},force-compression=true" \
        -f "$SCRIPT_DIR/Dockerfile" \
        "$REPO_ROOT"
fi

if [ "$DO_SIF" = 1 ]; then
    source "$REPO_ROOT/tools/docker/sqsh.env"
    SIF="${SIF_DIR:-$HOME}/kernel-dev-${DATE}.sif"
    SOURCE="docker://$IMAGE"
    if command -v docker >/dev/null 2>&1 && docker image inspect "$IMAGE" >/dev/null 2>&1; then
        SOURCE="docker-daemon://$IMAGE"
    fi
    echo "=== Building $SIF from $SOURCE ==="
    # Staging needs room for the uncompressed image (~15 GB for this one);
    # default to /tmp but let APPTAINER_TMPDIR point somewhere roomier when
    # /tmp is tight.
    TMPDIR_SIF="${APPTAINER_TMPDIR:-/tmp/apptainer_tmp}"
    mkdir -p "$TMPDIR_SIF" "${SIF_DIR:-$HOME}/apptainer_cache"
    # Build to .tmp and rename atomically so an interrupted build never leaves
    # a partial SIF that would be silently reused.
    APPTAINER_TMPDIR="$TMPDIR_SIF" \
    APPTAINER_CACHEDIR="${SIF_DIR:-$HOME}/apptainer_cache" \
        apptainer build --force --mksquashfs-args="$ENROOT_SQUASH_OPTIONS" "$SIF.tmp" "$SOURCE"
    mv "$SIF.tmp" "$SIF"
    echo "Done: $SIF"
fi

if [ "$DO_SQSH" = 1 ]; then
    source "$REPO_ROOT/tools/docker/build_sqsh.sh"
    build_sqsh "$IMAGE" "$IMAGE" "${SQSH_DIR:-$HOME}/kernel-dev-${DATE}.sqsh"
fi
