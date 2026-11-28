#!/usr/bin/env bash
# Run on the Docker host. Only `register` takes a short-lived repo registration
# token on stdin. The Docker API is used by the operator, never by the listener.
set -euo pipefail
name=quack-b300
image=${2:?usage: listener.sh build\|register\|start CI_IMAGE}
tag=quack-ci-runner:${image##*:}
opts=(--hostname "$name" --cap-drop ALL --security-opt no-new-privileges
      --volume "$name-home:/home/ubuntu")
case ${1:-} in
  build)
    docker build --build-arg "CI_IMAGE=$image" -t "$tag" - < "$(dirname "$0")/Dockerfile"
    ;;
  register)
    docker run --rm -i "${opts[@]}" "$tag" bash -ec '
      test ! -f .runner
      read -r ACTIONS_RUNNER_INPUT_TOKEN
      test -n "$ACTIONS_RUNNER_INPUT_TOKEN"
      export ACTIONS_RUNNER_INPUT_TOKEN
      exec ./config.sh --unattended --url https://github.com/Dao-AILab/quack \
        --name quack-b300 --labels b300'
    ;;
  start)
    docker run -d --init --restart unless-stopped --name "$name" "${opts[@]}" \
      --gpus '"device=0,1,2,3"' --cpus 32 --memory 200g --shm-size 16g "$tag"
    ;;
  *) echo 'expected build, register or start' >&2; exit 2 ;;
esac
