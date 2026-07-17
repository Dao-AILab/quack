# CLC `try_cancel` spurious-invalid: the July 2026 varlen corruption

**TL;DR.** On B300 (sm_103a, driver 595, CUDA 13.2), `clusterlaunchcontrol.try_cancel`
can return invalid ("no pending cluster") while hundreds of the current grid's
clusters are still pending. It happens when (a) a second process is
time-slicing the GPU and (b) several CLC grids are queued back-to-back on a
stream — ~2.5% of launches under those conditions, zero otherwise. Any CLC
scheduler that treats an invalid response as "pool empty" retires early; that
is harmless for exact grids, but quack's varlen padding drain (`afe2ef3`)
fired blind cancels at exactly these false retirements and killed **real**
pending tiles, whose output rows silently kept stale allocator memory.
Fixed by gating the drain on a *decoded phantom* (a valid grant whose work
index delinearizes to padding) — never on an invalid response. Standalone
reproducer: `AI/repro_clc_spurious_invalid.py`.

## Symptom

CI on the b300 leg went flaky after `b164cd1`/`33ae67f` (July 5–6):

- `test_gemm_varlen_m` / `test_gemm_add_varlen_m` (bf16, k=8192) failing with
  `max_err` 5–14 — plausible-looking but wrong values in whole rectangles of
  output tiles, often entire trailing batches.
- Once, an Xid 31 MMU fault (wild read) mid-run poisoned an xdist worker's
  CUDA context and cascaded into 35 `illegal memory access` test failures.
- Only under co-tenant GPU load (the shared runner); never on a quiet GPU,
  never in single-test repeats — which is why it read as "CI flakiness" at
  first.

Reproduction recipe for the *test-level* failure: the k=8192 varlen sweep
plus a same-GPU contender process (6144³ bf16 matmul, 3 s on / 2 s off).
Fails 1–7 of 768 tests per run on bad code; 0 on fixed code.

## Root cause

Commit `afe2ef3` "[Sched] Impl spray and pray CLC canceling for varlen
scheduler" added a retirement drain: when the scheduler warp's steal decodes
into the padding region, fire a burst of fire-and-forget `try_cancel`s to
drain the padding tail so phantom clusters never launch. Two stacked defects:

1. **The trigger was wrong.** The drain ran at *every* retirement, including
   retirements caused by an **invalid steal response** — and on B300 under
   contention, `try_cancel` returns invalid **spuriously**, long before the
   pool is empty. A cluster retiring on a spurious invalid still has hundreds
   of real siblings pending.
2. **The budget/baseline were garbage on that path.** `_current_work_idx` was
   updated from the response's `bidx` even when the response was invalid
   (bidx is undefined then, observed ≈1), so the drain computed
   `base≈1, budget=256` — maximum spray volume with no justification.

Net effect: hundreds of cancels granted **real** work indices. Canceled
clusters never execute; their output tiles keep whatever the allocator block
held (the silent max_err flavor), and downstream consumers of garbage decode
occasionally produced the wild reads (the Xid 31 cascade flavor).

Smoking gun (in-situ capture build — drain re-enabled with the exit
response's valid bit added to a stats printf):

```
QUACK-DRAIN: pr=0 base=155 n=256 min=287 max=747  gridtot=1336
QUACK-DRAIN: pr=0 base=379 n=256 min=587 max=1065 gridtot=1336
```

`pr=0` = the exit response was invalid; `base` = last real grant (mid-pool);
`n/min/max` = the drain then received up to 256 **valid grants** at real work
indices. 24/24 corrupting drains showed this signature; the corrupt output
region matched the granted index range exactly. The grants *are* successful
retries after the invalid — hardware-attested proof the invalid was spurious.

## What it was NOT (exonerated by A/B, all with per-arm isolated caches)

The hunt took three wrong turns before landing; each was killed by a
controlled experiment, and several produced independently useful facts:

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Grant non-monotonicity (cancels steal real tiles out of order) | **refuted** | 576M observed cancels, zero out-of-order grants (`repro_clc_grant_monotonicity.py`) |
| Orphaned response write-backs at CTA exit | real hazard, not this bug | fire-and-forget exit fail-stops with Xid 43 in minimal kernels; but a response-drained spray still corrupted |
| Cross-grid cancels (killing the next launch's clusters) | **refuted** | 0/800 back-to-back-PDL trials; grants never cross grid boundaries |
| PDL, multicast queries, pipelined 2-in-flight queries | clean | direct probe arms |
| TMEM allocation, tcgen05 MMA execution | clean / clean | `--tmem` arm; `QUACK_SKIP_MMA` kernel ablation *fires harder* (226 vs 23) |
| TMA/LDG bandwidth or tx-barrier pressure | clean | `--tma-traffic` (2-deep 32KB cp.async.bulk ×3 warps), `--traffic` |
| Kernel duration / preemption straddling | clean | 36 ms kernels with the retry detector |
| Cluster shape, sched_stage depth | clean | `QUACK_FORCE_1CTA` fires (30), `QUACK_SCHED_STAGE=1` fires (84) |
| Contender type | irrelevant | elementwise `mul_` loop fires as hard as matmul |

Production-side scale sweep (signal = `pr=0` drain lines, not test failures):
fires at L=5/k=8192/n=2048 (~5–15%/launch); L=1, n=256, k=1024 all zero;
L=2 intermediate (19). Every "small" axis kills it — which pointed at
duration for a while, until skip-MMA (fast tiles, fires harder) broke that
theory.

## The missing ingredient: queued launches

Every clean standalone probe configuration synchronized per launch. The
firing production loop never did — it queued hundreds of gemm launches
back-to-back. Adding `--stream-depth` (N launches per sync) to the probe:

| Configuration (all + contender) | retry-proven spurious invalids / 3000 launches |
|---|---|
| depth 1 (per-launch sync, all prior modes) | 0 (150k+ retirements total) |
| depth 2 | 0 |
| **depth 10** | **78** |
| **depth 300** | **73** |
| depth 300, quiet GPU | 0 |
| depth 300, grid 256 (fits ~1 wave) | 0 |

**Minimal ingredient set** (each necessary, jointly sufficient):

1. a co-tenant process time-slicing the GPU (any kernels);
2. ≥ ~10 CLC grids queued back-to-back on one stream (threshold between 2
   and 10);
3. grid ≫ resident capacity (deep pending pool; 1344 clusters vs ~74
   resident).

The reproducing kernel is a **plain serial try_cancel steal loop** — no
multicast, no PDL, no TMEM, no TMA, 32 threads/CTA. The verdict is
self-proving: on an invalid response the probe retries; a grant on the retry
proves the pool was non-empty. Run it:

```bash
# terminal 1: any contender, e.g. a matmul loop 3s-on/2s-off
# terminal 2:
python AI/repro_clc_spurious_invalid.py --trials 3000 --stream-depth 300
# -> RESULT streaming grid=1344 depth=300 launches=3000: 73 RETRY-PROVEN spurious invalids
```

Bug-report one-liner: *with several `clusterlaunchcontrol` grids queued
back-to-back on a stream and a second process time-slicing the GPU,
`try_cancel` returns "no pending cluster" while hundreds of the current
grid's clusters remain pending (~2.5% of launches; B300, driver 595).*

## The fix (shipped in `quack/tile_scheduler.py::cancel_pending_tail`)

Three invariants, replacing the fire-and-forget spray:

1. **Phantom gate** (the correctness linchpin). `_current_work_idx` updates
   only from *valid* responses, and the drain gets a non-zero budget only
   when the cluster retired on a **decoded phantom** — a valid grant whose
   work index delinearized to padding. Grant monotonicity (576M-sample
   validated) then guarantees no real work remains in the pool. An invalid
   response proves nothing and drains nothing.
2. **Serial-observed cancels.** Issue → wait → decode, one at a time, so no
   CLC state is ever in flight at CTA exit (fire-and-forget exit fail-stops
   with Xid 43), and a below-baseline grant aborts the drain with a loud
   `QUACK-SPRAY-ANOMALY` printf instead of silently eating a tile.
3. **Private mailbox.** Responses land in a dedicated slot + mbarrier after
   the sched response ring (+8 Int32 in `sched_smem_size`), never touching
   live ring slots or barriers.

Validation:

- 84 runs / 64,512 tests under the reproducer contention: **0 failures,
  0 anomalies**, with the drain genuinely active (521 phantom-gated
  activations).
- Full CI-style suite: 16,288 passed, 0 failed.
- Perf recovers the no-spray regression entirely: L256/s128 tail-heavy shape
  132 µs (vs 173 µs with no drain; original spray was 132 µs); typical
  shapes unchanged.

Controls from the same campaign (all still corrupt, confirming the gate is
the load-bearing piece): unconditional serial-observed drain, burst-waited
drain, 300 µs-delayed drain. Delay-only (no cancels): clean.

## Guidance for any CLC scheduler

- **Never infer pool-empty from an invalid `try_cancel` response.** Retire on
  it if you like (the un-granted clusters simply launch later — correct for
  exact grids), but do not take any action premised on "no work remains".
  The only sound emptiness proof available is a granted index that decodes
  beyond the real work range, combined with grant monotonicity.
- Do not leave CLC responses in flight at CTA exit (Xid 43 fail-stop class).
- `bidx` of an invalid response is garbage; never fold it into scheduler
  state.

## Artifacts

- `AI/repro_clc_spurious_invalid.py` — the minimal reproducer + full
  ingredient/exoneration matrix in its docstring (`--stream-depth`,
  `--multicast`, `--pdl-attr`, `--xgrid`, `--tmem`, `--traffic`,
  `--tma-traffic`, `--pipelined` modes).
- `AI/repro_clc_grant_monotonicity.py` — grant-order validation (576M
  cancels, monotone) and the Xid-43 fire-and-forget-exit fail-stop repro.
- `quack/tile_scheduler.py::cancel_pending_tail` — the fixed drain, with the
  invariants documented in its docstring.
- Test-level repro: `pytest tests/test_linear_varlen_m.py -k
  "test_gemm_add_varlen_m and 8192"` + same-GPU contender (bad code: 1–7 of
  768 fail per run; fixed: 0).
