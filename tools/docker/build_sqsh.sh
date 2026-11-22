# Shared by the CI and dev image builders. Enroot can import directly from a
# local Docker daemon or from a registry on a machine without Docker.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sqsh.env"

build_sqsh() (
    local local_image=$1 remote_image=$2 sqsh=$3
    local source="docker://$remote_image" tmp_dir
    if ! command -v enroot >/dev/null 2>&1; then
        echo "--sqsh requires Enroot on the machine building the SquashFS image" >&2
        exit 1
    fi
    if command -v docker >/dev/null 2>&1 && docker image inspect "$local_image" >/dev/null 2>&1; then
        source="dockerd://$local_image"
    fi

    # Stage on the destination filesystem so a failed import never replaces
    # an existing image, and parallel imports don't share a temporary file.
    mkdir -p "$(dirname "$sqsh")" || exit 1
    tmp_dir=$(mktemp -d "$sqsh.tmp.XXXXXXXX") || exit 1
    trap 'rm -rf -- "$tmp_dir"' EXIT

    echo "=== Building $sqsh from $source ==="
    enroot import --output "$tmp_dir/image.sqsh" "$source" || exit 1
    mv -f -- "$tmp_dir/image.sqsh" "$sqsh" || exit 1
    echo "Done: $sqsh"
)
