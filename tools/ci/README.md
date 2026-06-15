# QuACK CI

CI runs on self-hosted GPU runners (h100, b300, h100/sm120) inside an Apptainer
SIF pulled from a Docker image on Docker Hub. Triggered on every push to `main`
and on PRs.

## Files

| File | Purpose |
|------|---------|
| `tools/ci/docker/Dockerfile` | image recipe (one Dockerfile, two variants via build args) |
| `tools/ci/docker/build.sh` | build cu129 and/or cu132 docker image locally |
| `tools/ci/docker/tag_and_push.sh` | tag and push to `tridao/quack-kernels` on Docker Hub |
| `.github/workflows/_test.yml` | reusable workflow with lint/changes/test jobs and the matrix; **the image tag pins live here** |
| `.github/workflows/ci.yml`, `ci-pr.yml` | thin shells that call `_test.yml` on push / PR |
| `.github/actions/gpu-test/action.yml` | composite action — pulls SIF, runs two-pass pytest |

## Image variants

| Variant | Docker image (latest) | Notes |
|---------|------------------------|-------|
| `cu129` | `tridao/quack-kernels:cu12.9-DATE` | base cute-dsl |
| `cu132` | `tridao/quack-kernels:cu13.2-DATE` | cute-dsl[cu13] |

The cu12.9 variant uses torch cu126 wheels (newest CUDA 12.x PyTorch 2.12 index).
The cu13.2 variant uses torch cu132 wheels plus the Dockerfile's CUDA 13 forward-
compatibility libcuda shim so it remains runnable on 575-series kernel drivers.

## Test matrix

`_test.yml` runs the full cross product (6 jobs per push):

| GPU | Arch override | cu129 | cu132 |
|-----|----------------|-------|-------|
| h100 | (none, sm90) | ✓ | ✓ |
| b300 | (none, sm100) | — | ✓ |
| h100 | sm120 | ✓ | ✓ |

## Two-pass test strategy

Per `gpu-test/action.yml`:

- **Pass 1** — `pytest tests/ --compile-only -n 24 --dist worksteal` (compile-only
  flag, no GPU memory needed; warms the persistent kernel cache).
- **Pass 2** — `CUDA_VISIBLE_DEVICES=$FREE_GPUS pytest tests/ -n $NUM_GPUS --dist worksteal`
  on real GPUs (free-memory threshold 50 GB).

## SIF caching on runners

The action pulls `docker://$IMAGE` into `${CI_WORK_DIR:-$HOME}/<tagslug>.sif`
on first use, then reuses the cached file on subsequent jobs with the same
tag. After each pull, **stale SIFs from previous image bumps are auto-deleted**;
the cleanup whitelist keeps both currently-pinned variants (`IMAGE_CU129` and
`IMAGE_CU132`), so cu129 and cu132 don't thrash each other's caches.

## Cutting a new image

The image tags are **pinned in `.github/workflows/_test.yml`** (used by both
ci.yml and ci-pr.yml). Three steps:

```bash
# 1. Build & push from a box that has docker (one-time Hub login: `docker login -u tridao`)
./tools/ci/docker/build.sh
./tools/ci/docker/tag_and_push.sh
# unattended variant: DOCKERHUB_TOKEN=hub_xxx ./tools/ci/docker/tag_and_push.sh
```

```yaml
# 2. Bump IMAGE_CU129 and IMAGE_CU132 in .github/workflows/_test.yml:
env:
  IMAGE_CU129: tridao/quack-kernels:cu12.9-NEW_DATE
  IMAGE_CU132: tridao/quack-kernels:cu13.2-NEW_DATE
```

```bash
# 3. Commit and push.
git commit -am "Bump CI images to cu*-NEW_DATE"
git push
```

That's it — runners auto-pull the new SIFs on the next CI run and prune the old ones. No manual runner steps.

## Manual / local SIF testing (off-CI)

For ad-hoc debugging on a runner, pull the same image CI uses:

```bash
apptainer pull ~/quack.sif docker://tridao/quack-kernels:cu12.9-DATE
apptainer exec --nv --writable-tmpfs ~/quack.sif bash
```

For private images, set `APPTAINER_DOCKER_USERNAME=tridao` and
`APPTAINER_DOCKER_PASSWORD=$DOCKERHUB_TOKEN` before running.

## Public vs private image (Docker Hub)

The current `tridao/quack-kernels` repo is intended to be public, so CI needs no
Docker Hub secrets. If you flip it to private, add `DOCKERHUB_USERNAME` and
`DOCKERHUB_TOKEN` as repo secrets and prepend a login step before the
gpu-test action:

```yaml
- name: Log in to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}
```

Then export `APPTAINER_DOCKER_USERNAME` / `APPTAINER_DOCKER_PASSWORD` for the
apptainer pull step (or set them on the runner host).
