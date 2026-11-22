# QuACK CI

CI runs on a self-hosted H100 runner (sm90 and an sm120-override leg) inside a
container built from a Docker image on Docker Hub: through Slurm/Pyxis from an
Enroot SquashFS on the cluster, or in an Apptainer SIF on runners without Slurm.
Triggered on every push to `main` and on PRs.

## Files

| File | Purpose |
|------|---------|
| `tools/ci/docker/Dockerfile` | image recipe (one CUDA 13 image) |
| `tools/ci/docker/build.sh` | `--image` / `--push` / `--sif` / `--sqsh`: build and publish the Docker image, or convert it for Apptainer or Enroot |
| `.github/workflows/_test.yml` | reusable workflow with lint/changes/test jobs and the matrix; **the image tag pins live here** |
| `.github/workflows/ci.yml`, `ci-pr.yml` | thin shells that call `_test.yml` on push / PR |
| `.github/actions/gpu-test/action.yml` | composite action — fetches the image (sqsh or SIF), runs single-pass pytest |

## Image

| Docker image (latest) | Notes |
|------------------------|-------|
| `tridao/quack-kernels:cu13.2-DATE` | PyTorch 2.15 cu132 wheels, cute-dsl[cu13] |

The image is CUDA 13 only and bundles no forward-compatibility libcuda, so a
runner's driver must itself support CUDA 13. The `cu13.2` tag names torch's wheel
index, the same one `pyproject.toml`'s `cu13` extra installs.

The Dockerfile no longer builds a CUDA 12 image, and `_test.yml` no longer runs
`cu129` legs.

## Test matrix

`_test.yml` runs 2 jobs per push, both on the H100 runner:

| GPU | Arch override | cu132 |
|-----|----------------|-------|
| h100 | (none, sm90) | ✓ |
| h100 | sm120 | ✓ |

The h100/sm120 job selects QuACK's SM120 implementations with `QUACK_ARCH=120`
but compiles and executes on H100 hardware. The B300 leg is disabled for now
(see the comment in `_test.yml`).

## Runners

The `h100` runner lives on the cluster login node in
`~/actions-runner-quack`; `gpu-test` detects Slurm + Pyxis there and runs each
job as `srun --gres=gpu:4 --container-image=<sqsh>` on a compute node, so the
login node only hosts the listener. The image is imported once per tag into
`$CI_WORK_DIR` (`~/quack-ci`, shared storage) as an Enroot SquashFS. The login
pod restarting kills the listener; restart it on the login node with
`bash ~/actions-runner-quack/start.sh`. Runners without Slurm
keep using an Apptainer SIF.

The listener runs as a cluster user on the login node, and a job's checkout
(including this action) executes there before any container starts. The test
container gets only the checkout and `$CI_WORK_DIR/container-home`, not that
user's home, but this is not a sandbox for untrusted code: keep PR runs behind
the `gpu-ci` environment approval and read a fork's diff before approving.

**TODO: Enable distributed GEMM coverage in CI.** The slim image stubs NCCL,
and each xdist worker sees only one GPU. Replace the stub with NCCL and run
`tests/test_distributed_gemm.py` and `tests/test_blockscaled_distributed_gemm.py`
in a separate test step with multiple GPUs visible.

Four GPUs is deliberate: wall time scales almost linearly with xdist workers
(warm cache 2.7 / 4.7 / 9.8 min and cold 8 / 16 / 32 min on 4 / 2 / 1 GPUs),
and several workers cannot share one GPU because the largest tests need most
of its memory.

## Test strategy

Per `gpu-test/action.yml`: a single pass with async kernel compilation —

- `CUDA_VISIBLE_DEVICES=$FREE_GPUS pytest tests/ -n $NUM_GPUS --dist worksteal --async-compile=24`
  On Slurm runners the GPUs are the ones Slurm allocates. On Apptainer runners
  the action picks GPUs with at least 50 GB free, waiting up to 5 minutes and
  polling every 15 seconds when runner assignment races with transient GPU
  contention. Cold kernel-compile misses are shipped to a pool of 24 CPU
  workers (forkserver sidecar, GPU-blind) while the affected tests defer and
  retry once their `.o` lands; warm runs pay nothing. The
  persistent kernel cache (`QUACK_CACHE_DIR`) carries `.o` files across runs
  on the same runner. CI prunes QuACK source-fingerprint cache directories
  older than 7 days before each test run, plus interrupted `.o.tmp.*` exports
  older than 1 day.

## Image caching on runners

The action pulls `docker://$IMAGE` into `${CI_WORK_DIR:-$HOME}/<tagslug>.sif`
(`.sqsh` on Slurm runners) on first use, then reuses the cached file on
subsequent jobs with the same tag. Each cache hit refreshes the file's mtime.
During image setup, CI prunes other tags of the same repository and format
that have gone unused for at least seven days.

`tools/docker/sqsh.env` sets zstd level 3 and 1 MiB blocks for CI cache misses
and manual `--sqsh` / `--sif` builds. Existing cached images are not repacked.

## Cutting a new image

The image tags are **pinned in `.github/workflows/_test.yml`** (used by both
ci.yml and ci-pr.yml). Three steps:

```bash
# 1. Build & push from a box that has docker (one-time Hub login: `docker login -u tridao`)
./tools/ci/docker/build.sh --image --push
# On a runner, add --sif to also pre-build the SIF the gpu-test action caches.
# On a docker-less runner (b300), pre-warm from the pushed image with:
#   DATE=YY.MM.DD ./tools/ci/docker/build.sh --sif
# unattended variant: DOCKERHUB_TOKEN=hub_xxx ./tools/ci/docker/build.sh --image --push
```

```yaml
# 2. Bump IMAGE_CU132 in .github/workflows/_test.yml:
env:
  IMAGE_CU132: tridao/quack-kernels:cu13.2-NEW_DATE
```

```bash
# 3. Commit and push.
git commit -am "Bump CI image to cu13.2-NEW_DATE"
git push
```

Runners auto-fetch the new image on the next CI run. No manual runner steps.

## Manual / local SIF testing (off-CI)

For ad-hoc debugging on a runner, pull the same image CI uses:

```bash
apptainer pull ~/quack.sif docker://tridao/quack-kernels:cu13.2-DATE
apptainer exec --nv --writable-tmpfs ~/quack.sif bash
```

For private images, set `APPTAINER_DOCKER_USERNAME=tridao` and
`APPTAINER_DOCKER_PASSWORD=$DOCKERHUB_TOKEN` before running.

## Prebuild a SquashFS image for Slurm/Pyxis

Pyxis can use the registry tag directly. To avoid rebuilding the SquashFS on
each node, run `--sqsh` once per tag on a compute node and store it in shared
storage. From the repo checkout on the login node, submit the import as a CPU job:

```bash
# Replace DATE with the date of an already-pushed image.
DATE=26.09.12 SQSH_DIR="$HOME/quack-images" \
  sbatch --cpus-per-task=8 --mem=16G --time=30 \
  --wrap='bash tools/ci/docker/build.sh --sqsh'

# Once the import completes:
srun --gres=gpu:1 --time=10 \
  --container-image="$HOME/quack-images/tridao-quack-kernels-cu13.2-26.09.12.sqsh" \
  python3 -c 'import torch; print(torch.cuda.device_count())'
```

For the dev image, use `tools/dev/docker/build.sh --sqsh` instead with the
dev tag's date. It writes `kernel-dev-<date>.sqsh` under `$SQSH_DIR` (or `$HOME`
if unset). The `gpu-test` action does this import itself on first use of a tag.

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
