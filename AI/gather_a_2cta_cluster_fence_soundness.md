# gather_A 2-CTA relay: cluster-scope fence soundness (issue #63)

Context: https://github.com/Dao-AILab/quack/issues/63 reported that the cp.async gather_A
+ 2-CTA MMA path (`gather_A and use_2cta_instrs and not use_tma_gather`) relayed the
"AB buffer full" signal from the non-leader CTA to the leader with a plain
`mbarrier.arrive` (default `.release.cta`), which cannot form a cluster-scope
release-acquire relation with the leader's wait. The fix (quack/pipeline.py:
`mbarrier_arrive_release_cluster`, `mbarrier_acquire_cluster`; used in
`gemm_sm100.py::mma`) uses the issue's cheaper variant 2:

```
Relay (non-leader CTA, after local wait on its own full barrier):
    fence.release.sync_restrict::shared::cta.cluster
    mbarrier.arrive.relaxed.cluster.shared::cluster   [leader's full barrier]

Leader (MMA warp, after the regular cta-scope consumer_wait):
    mbarrier.test_wait.parity.acquire.cluster          [same barrier, same phase]
    (result feeds a never-taken fallback wait so ptxas cannot DCE it)
```

Measured cost on B300 (bf16, tile 256x256, cluster 2x1): ~0% on varlen_m gather
configs, ~1.3% on varlen_k 4096,4096,512,L=64 (short K => relay closer to critical
path). All 10,326 gather/varlen tests pass. This note records why the construction is
sound, from the PTX memory model and from the hardware's point of view.
All PTX quotes from PTX ISA 9.3.

## View 1: PTX memory model

Name the operations:

- `W_data` — cp.async writes into CTA1's (non-leader's) smem
- `Wt`     — relay warp's local wait on CTA1's own barrier (fed by `cp.async.mbarrier.arrive`)
- `F1`     — `fence.release.sync_restrict::shared::cta.cluster` (relay thread)
- `W_arr`  — `mbarrier.arrive.relaxed.cluster.shared::cluster` on leader's barrier B0
- `R_acq`  — leader's `mbarrier.test_wait.parity.acquire.cluster` on B0
- `R_mma`  — `tcgen05.mma.cta_group::2` reads of CTA1's smem over DSMEM

### Link 1: W_data -> relay warp (intra-CTA, mbarrier contract)

Not generic release-acquire; it is the mbarrier's own guarantee. From the
test_wait/try_wait section:

> "All cp.async operations requested prior, in program order, to
> cp.async.mbarrier.arrive during the completed phase by the participating threads
> **of the CTA** are performed and made visible to the executing thread."

Every one of the enumerated mbarrier guarantees is CTA-local ("of the CTA"). This is
also the formal confirmation that the *original* code was wrong: no mbarrier magic
carries cp.async data across CTAs; anything cross-CTA must be built from generic
release/acquire patterns.

### Link 2: relay -> leader (the cross-CTA edge)

Three spec definitions compose:

1. **Release pattern** (§8.9): "a release ... memory fence followed by a strong write
   on M in program order — e.g. `fence.release; atom.relaxed [M]`". `F1; W_arr` is
   literally this form (an mbarrier arrive is a strong RMW on the barrier word). The
   sync_restrict clause says the release effect "only applies to operations performed
   on objects in `.shared::cta` state space" — `W_data` lives in CTA1's shared::cta, so
   it is inside the released class. Cumulativity (the release pattern's effect "is
   further extended to operations in other threads through the transitive nature of
   causality order") carries `W_data` — written by the load warps / cp.async engine and
   observed by the relay via Link 1 — into the relay's release.
2. **Acquire pattern**: "When mbarrier.test_wait ... with .acquire qualifier returns
   True, they form the acquire pattern" — `R_acq` verbatim.
3. **Synchronizes-with** (§8.9.4): a release pattern X synchronizes with an acquire
   pattern Y if "a write operation in X precedes a read operation in Y in observation
   order, and the first operation in X and the last operation in Y are morally strong."
   - Moral strength of (F1, R_acq): both strong, both `.cluster` scope (each scope set
     includes the other thread — exactly what `.cta` scope failed), both generic proxy;
     complete-overlap is vacuous for a fence. OK.
   - Observation order of (W_arr, R_acq): §8.9.2 has the RMW-chain rule ("for some
     atomic operation Z, W precedes Z and Z precedes R in observation order"). So it
     does not matter that `W_arr` is not the arrive that completes the phase (TMA
     complete-tx arrives land on B0 too); the chain of barrier RMWs carries it.
     W_arr and R_acq are themselves morally strong (cluster/cluster, same proxy,
     same b64 word). OK.

### Link 3: leader wait -> MMA

"Any memory synchronization established by an acquire pattern only affects operations
occurring in program order **after** the last instruction in that pattern" — the MMA is
issued after `R_acq`. The earlier cta-scope try_wait in front of it contributes nothing
and needs to contribute nothing; it is just the cheap poll that guarantees `R_acq`
returns true on the first try. The split (cta-scope wait loop, then one cluster-scope
acquire test_wait on the completed phase) is not weaker than a single
`try_wait.acquire.cluster` loop: the spec's own examples use exactly this
"relaxed/weak observation, then a scoped acquire" decomposition, e.g.
`mbarrier.try_wait.relaxed.cluster ...; fence.acquire.sync_restrict::shared::cluster.cluster`.

### Conclusion

`W_data -(Link1)-> F1;W_arr -(sync-with)-> R_acq -(po)-> R_mma` is a causality-order
path; the Causality axiom forbids `R_mma` from reading anything older than `W_data`.

### Caveats (none introduced by the fix)

1. **Async proxy.** The spec says tcgen05.mma's smem accesses are performed in the
   *async proxy*, while cp.async "is treated as a weak memory operation performed in
   the **generic** proxy", and cross-proxy access "needs a cross-proxy fence". Strictly,
   W_data(generic) -> R_mma(async) wants `fence.proxy.async` on the path — and the
   model chapter's *proxy-preserved causality* definition has no case for it (only
   generic-generic, same-proxy-same-CTA, alias). Mitigations: (a) our fence lowers to
   `MEMBAR.ALL.CTA ; FENCE.VIEW.ASYNC.S` — the second instruction *is* the async-proxy
   fence, and the letter-of-spec
   `fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster` assembles to
   **bit-identical SASS** (verified with ptxas sm_103a), so switching the helper to
   `_nvvm.fence_proxy_sync_restrict` is a zero-cost one-liner if we want the PTX to say
   it explicitly. (b) The leader's own cp.async data, the single-CTA gather path, and
   every cp.async->UMMA pipeline NVIDIA ships have no proxy fence either — cp.async
   deposits appear to be async-proxy-visible by construction (see View 2); the
   proxy-fence hazard demonstrably belongs to `st.shared`-written data feeding TMA/UMMA.
2. The tcgen05 chapter's canonical consumer pattern is
   `mbarrier.try_wait.relaxed.cluster` -> **`tcgen05.fence::after_thread_sync`** ->
   `tcgen05.mma`. Neither the stock DSL TMA pipelines, nor quack, nor the issue's
   proposed consumer emits that fence in the mainloop. Whatever makes the omission safe
   (in practice: UTCHMMA issue is program-ordered behind the wait's predicate branch)
   applies identically before and after this fix.
3. Link 1 rests on the mbarrier prose ("performed and made visible"), which the formal
   axioms never fully integrate for async ops. Same foundation as every cp.async
   pipeline in existence.

## View 2: physical / hardware

**Memory structures.** SMEM is a physical array inside each SM, cached nowhere. A DSMEM
access (tcgen05.mma.2cta's peer-operand reads; the relay's remote arrive) is a request
routed over the GPC's SM-to-SM interconnect and serviced by the *owning* SM's array.
One copy of the data, no reader-side caching — **there is nothing to invalidate
anywhere in this path**. Coherence is trivial; the entire problem is *ordering*: could
the arrive signal reach SM0 while some A-tile bytes are still in flight inside SM1's
store path, letting SM0's read requests overtake them.

**What each SASS instruction does** (from the fixed kernel, sm_103a):

- `MEMBAR.ALL.CTA` — drains the SM-local write path (LSU queues / store buffering)
  until every prior write by this thread is committed to the SM's SMEM array. Local,
  tens of cycles, and near-zero here because there is rarely anything left to drain.
- `FENCE.VIEW.ASYNC.S` — reconciles the *generic* view of SMEM with the *async-proxy*
  view. The async engines (TMA, the UTC tensor pipe) access SMEM through their own
  port/path and the hardware tracks the two views separately — a real effect, not a
  paper distinction: on Hopper, TMA-storing `st.shared`-written data without
  `fence.proxy.async` observably reads stale bytes. This is the piece the buggy
  `.release.cta` arrive never emitted.
- `MEMBAR.ALL.GPU` (the cost of the naive `mbarrier.arrive.release.cluster`, avoided
  here) — an *unrestricted* cluster-scope release must assume prior **global** writes
  are in the released set. Global stores become visible to other SMs only at **L2**
  (L1s are not hardware-coherent; strong reads at scope > cta bypass L1 instead — the
  reader half of the protocol). So the writer-side membar drains to L2 and waits for
  the ack — hundreds of ns, once per k-tile. There is no cheaper "MEMBAR.CLUSTER" for
  global data because the ordering point between two SMs *is* L2; cluster ~= gpu for
  global traffic. `sync_restrict::shared::cta` is precisely the escape hatch: it
  declares "only my local SMEM is released", whose ordering point is the local array,
  so ptxas emits the SM-local drain + async-view sync instead. That is why the
  qualifier exists (PTX ISA 8.6).
- Consumer `SYNCS.PHASECHK` with `.acquire.cluster` — same opcode as the cta-scope
  check; a standalone `fence.acquire.sync_restrict::shared::cluster.cluster` emits no
  SASS at all. Physically sensible: the reader has no stale state to discard (barrier
  word is in its own SMEM; peer data will be fetched from the peer's array, uncached),
  so the only obligation is keeping UTCHMMA dispatch behind the phase check, which the
  predicate branch already enforces. Hence the consumer side measures free; the whole
  fix costs only the producer-side drain (0% hidden by the mainloop; ~1.3% on short-K
  varlen_k).

**Why the sequence closes the race.** cp.async data is deposited into SM1's array by
the LDGSTS/DMA path, and the mbarrier tracking fires *on commit* — by the time the
relay warp's local wait succeeds, the bytes are physically in the array. (This is also
why cp.async->UMMA plausibly works without proxy fences: those writes never sit in the
SM's generic store buffers; the proxy-fence hazard belongs to `st.shared` data.) The
fence then drains anything still local and syncs the async view; only then does the
warp issue the remote `SYNCS.ARRIVE...RED`, which crosses the interconnect; SM0's
PHASECHK can only observe it after it lands; and SM0's UTCHMMA read requests travel
back to SM1 and are serviced by an array that committed the data strictly earlier.
Every hop is a physical message ordered behind the previous one.

**Why the buggy version never visibly failed:** on current silicon that chain is
already ordered without the fences — data committed before the relay wakes, arrive
issued after by control dependency, reader uncached. The exposure is (a) ptxas
reordering around a fence it is told is only cta-scope, (b) future parts adding store
buffering where a local thread sees its write before the array commits it for remote
consumption, (c) async-view divergence for generically-written smem. The memory model
refuses to promise any of this precisely so NVIDIA can build such hardware — a real
bug worth ~1%, not pedantry.

## Design note: reuse the ab_pipeline barrier vs a dedicated relay mbarrier

Considered: giving the non-leader -> leader relay its own mbarrier instead of arriving
on the leader's ab_pipeline full barrier. Rejected — reuse is equally sound and
strictly cheaper on the consumer side.

**Soundness.** Reuse is the one place the proof is less obvious: with a dedicated
barrier the leader's acquire reads the very word the relay's arrive wrote (direct
release-acquire pairing), while with the shared barrier the relay's arrive is generally
*not* the arrive that completes the phase (TMA complete-tx and the local cp.async
arrive land on it too). That is exactly what the observation-order RMW-chain rule
(§8.9.2, used in Link 2 above) covers: all arrives are morally-strong RMWs on the same
b64 word, so `W_arr` chains through subsequent arrives to the value the test_wait
reads. Dedicated barrier = trivial proof; shared barrier = one extra spec-sanctioned
hop. Equally sound. Liveness requires the shared barrier's arrive count to include the
relay — the asymmetric producer_cnt in `make_ab_pipeline` (+2 leader / +0 non-leader)
does this; a dedicated barrier would restore uniform counts (its one real aesthetic
win).

**SASS.** Producer/relay side: identical either way —
`MEMBAR.ALL.CTA ; FENCE.VIEW.ASYNC.S ; SYNCS.ARRIVE.TRANS64.RED.A1T0` on a mapa'd
address doesn't care which barrier word it targets. Consumer/leader side: strictly
worse with a dedicated barrier. Today, full-barrier completion already implies the
relay arrived (it is counted in), so the cluster-acquire test_wait is a single
guaranteed-true non-looping `SYNCS.PHASECHK.TRANS64` — pure ordering annotation. With
a dedicated barrier that implication is lost, so the acquire becomes a second genuine
`PHASECHK.TRYWAIT / NANOSLEEP / PHASECHK` wait loop per k-tile, plus its own
parity/phase register state, `ab_stage x 8` bytes of smem, and another mbarrier_init.
Steady-state wall clock is a wash (leader waits for max(own data, peer data) either
way), but code size, state, and the hot-path wait all get worse.

One honest advantage of the dedicated barrier: its acquire wait *must* loop, so ptxas
cannot DCE it — the never-taken-fallback trick in `mbarrier_acquire_cluster` exists
only because our test_wait's result is otherwise unused. That ugliness is contained in
one documented helper and SASS-verified, so the trade goes to reuse.

## Verification recipe

- PTX: `CUTE_DSL_KEEP=ptx CUTE_DSL_DUMP_DIR=<dir> python ...` then grep for
  `fence.release.sync_restrict`, `arrive.relaxed.cluster`,
  `test_wait.parity.acquire.cluster`.
- SASS: carve the cubin from the newest quack_cache `.o` (payload after the second
  `\x7fELF` magic), `nvdisasm -c`. Relay =
  `MEMBAR.ALL.CTA ; FENCE.VIEW.ASYNC.S ; @P0 SYNCS.ARRIVE.TRANS64.RED` on the mapa'd
  remote barrier; leader = unconditional `SYNCS.PHASECHK.TRANS64` + branch between its
  wait and the UTCHMMA.2CTAs.
- Gotcha: the consumer-side test_wait's result must feed control flow (never-taken
  fallback wait) — ptxas dead-code-eliminates a test_wait whose predicate is unused,
  which silently removes the acquire. Caught by SASS inspection.
- Gotcha: `cute.arch.mbarrier_arrive(bar, peer_rank)` in cutlass-dsl 4.6.0 hardcodes
  scope=CTA even for the mapa'd remote pointer — arguably an upstream DSL bug.
