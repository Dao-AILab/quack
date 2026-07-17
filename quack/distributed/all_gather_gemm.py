# Copyright (c) 2026, QuACK team.
"""Overlapped distributed GEMMs for NVLink-connected ranks (SM90/SM100/SM110).

This docstring is the design record, measurements included — there is no
separate design doc.

Abstraction: from the GEMM kernel's perspective, AG-capability is exactly
(a) the shard-major rotated scheduler decode and (b) one flag gate in the
AB-load warp before a tile's first TMA — both in shared code, so ANY quack
GEMM variant participates via one host-side ``ag_args`` kwarg. The driver
half (:class:`AllGatherRunner`) owns buffers/flags/transport and wraps the
GEMM(s) in a context::

    with runner.gather(a_shard) as (a_full, ag_args):
        d = gemm(a_full, b, ..., ag_args=ag_args)  # any quack GEMM(s)

Contrast with the ecosystem: TransformerEngine's ub overlap lives INSIDE
its own GEMM wrapper (cuBLAS can't gate mid-kernel, so TE must chunk the
GEMM itself — nothing user-supplied can run against a partially-arrived
buffer), and PyTorch async-TP bakes one fused op per pattern
(fused_all_gather_matmul). The
in-kernel gate is what lets the body be arbitrary user code here — several
GEMMs can share one gather (q/k/v) with no dedicated fused op.

One persistent GEMM launch consumes the gathered A as it arrives. The
stable contract between the kernel and the transport is "resident buffer +
monotonic per-shard(-chunk) sequence flags": the tile scheduler decodes work
shard-major and the AB-load warp spins (relaxed, sys) on a flag before a
tile's first TMA.

The ONE built-in transport is ce_push (CE = Copy Engine, the GPU's DMA
units that execute ``cudaMemcpyAsync`` without occupying any SMs): the
OWNER CE-sends its shard to each peer, ``cudaMemcpyAsync`` in REVERSE ring
order, then remote-writes that peer's flag (a 4-byte copy of the device
epoch). Sender stream order carries the producer dependency — no
readiness handshake at all — and the reverse ring hands every receiver its
next-needed shard first under the ring-ROTATED consumption order. Chunked
arrival flags (opt-in, keep >= ~64 MB per CE op) refine gate granularity.

Measured transport physics the design rests on (B300, July 2026;
benchmarks/benchmark_p2p_* and session probes):

- Peer reads are never cached in local L2, so remote A must be
  materialized in local HBM exactly once — a resident gather buffer, not
  remote TMA (remote-A reads also amplify per N-tile reuse; NO-GO).
- CE peaks ~740 GB/s per copy but restarts its bandwidth ramp at EVERY
  descriptor (335/560/695 GB/s effective at 4/16/64 MB ops, ~2-6 us fixed
  overhead each, no warmth carry-over). Remote writes beat remote reads,
  idle and under load.
- The send loop below is measured-OPTIMAL at the API level: one egress
  path serializes a rank's peer-writes regardless of stream count (extra
  send streams change nothing); cudaMemcpyBatchAsync saves ~2 us/op but
  cannot order flags against data within a batch (uniform-late gates
  lose more); flags on an event-chained side stream starve behind the
  queued sends (CE serves its queue FIFO). The one live lever on top is
  outer CUDA-graph capture (-10..-17 us/iter of host enqueue at small
  many-rank shapes).
- SM copy kernels collapse to 35-97 GB/s under a full persistent grid
  while CE runs at 100% with ~zero GEMM interference — comm must ride CE
  or be fused into a kernel that already owns SMs.
- CE bytes transit L2 as evict_normal; access-policy windows are IGNORED
  by CE. No software control of its eviction class exists (the GEMM-side
  L2-hint chapter is closed too — tombstones in gemm_sm100/gemm_base).

Flags-contract warning for anyone hand-rolling a transport (all probed on
this stack against a spinning gated GEMM): publishing a flag CONCURRENTLY
with the GEMM must not be an SM kernel — tensor fill_ starves behind the
persistent grid and DEADLOCKS, even pre-warmed. cudaMemcpyAsync (intra- or
cross-device, 4 B through 8 MB probed), tensor copy_ (lowers to memcpy),
and cuMemsetD32Async all release the gate fine. Separate first-use trap:
LAZY MODULE LOADING — the first-ever launch of any kernel while the
persistent GEMM spins can wedge the context (a cold normal_ poisoned even
subsequent plain memcpys; pre-warming it fixed everything) — warm every
kernel your transport path touches before entering the spinning regime.

Two other transports were built, measured, and PARKED on branch
park/resolved-0715 behind the same flags contract (re-examined head-to-head
post-commit, July 2026 — these numbers are why; do not rebuild without a
new regime):

- CE ring pull (receiver-driven; staged-signal handshake + serial-ring
  pulls): edges ce_push by only ~8-13 us/iter (~1-1.5pp) at
  world_size == 2 with >= 128 MB shards — identically under CUDA graphs —
  and loses everywhere else. Not a wire effect (writes beat reads); the
  residual is ce_push's ~5.7 us remote flag memset per chunk vs pull's
  local flag. Disqualifier: the receiver-side WAIT on the peer's staged
  signal — stream-level waits compare against HOST immediates only,
  irreconcilable with the device-epoch capture story. ce_push has no
  receiver-side wait anywhere; the one device-valued waiter in the
  system is the GEMM's load-warp gate (an SM we already own).
- SM100 TMA-multicast push (smem-bounce kernel + multimem.st): wins ONLY
  world_size >= 8 AND shard <= 24 MB (TP8 17 MB: 42.6% vs 52.4% over
  roof), loses everywhere else including all of ws == 6. Three permanent
  taxes (reserve_clusters grid carve-out = +8-14% GEMM at wave-sensitive
  shapes; ~490 GB/s multicast plateau vs 600-740 unicast; host-sequenced
  staggering) against one benefit (1x sender egress via in-switch
  replication) that only pays when (ws-1) ramp-bound CE ops overrun the
  GEMM window; also the only transport reaching into the GEMM
  (reserve_clusters) and scheduler (own-first order) surfaces, and
  SM100-only. Its corner is better served by producer-push (below).

Parked with pull: the symmetric-source producer mode (make_source /
call_from_source). Considered, not built: log-depth tree forwarding —
distributes fan-out egress, but forwards are receiver-driven = pull's
host-value gating problem again.

Producer coupling: pass ``next_local_slot()`` for zero-copy producer writes
into the rotating buffer (inference-shaped lifetime). The gathered A is a
real all-gather output (``gathered_a()``) — transient, see below.

Training contract: ALWAYS REGATHER IN BACKWARD. Forward saves only the
local shard (a plain autograd tensor); backward issues fresh runner calls.
Nothing in the runner distinguishes forward/backward calls, and one runner
is shared across all layers of the same (shard, dtype) shape (persistent
symmetric buffers are collective+ms to create and peers/graphs bake their
addresses — the Megatron GlobalMemoryBuffer / PyTorch async-TP pattern).
Consequences: gathered_a() is transient (next call clobbers it) and
zero-copy next_local_slot() is inference-only by construction. The
backward chapter this opens (not built): wgrad (dW = dD^T @ A_full)
consumes the gathered dim as the REDUCTION dim, so its gating moves
per-K-TILE and the ring-rotation trick moves to the k-loop start
(reduction commutes — each rank starts at its own shard and wraps);
dgrad's dA is the ReduceScatter sibling's job.

Pipeline vocabulary (this is a distributed double-buffered producer/
consumer pipeline; sync objects map onto full/empty roles):

  ============================  =========================================
  arrival flags (+ epoch)       fine-grained FULL, per (shard, chunk);
                                consumer_wait = the in-kernel load-warp
                                gate (ag_wait_m_tile)
  ev_reuse[buffer]              collective EMPTY, one per buffer (EMPTY
                                is a per-buffer property); waiting it =
                                producer_acquire (staging + remote sends)
  ev_shard_staged               producer-internal edge, stage->transport
                                (not a pipeline barrier)
  ev_gemm_end                   my local consumer_release
  handle.barrier(ch=parity)     aggregates consumer_release across ranks
                                (EMPTY is a cross-rank fact: peers write
                                my buffer / read my slot)
  2 buffers                     pipeline depth (fixed; see below)
  ============================  =========================================

Per-call stream timeline (call i, parity p = i % 2, buffer B = bufs[p];
time flows DOWN; ──► = event edge record-to-wait; ~~► = cross-RANK write):

  ambient/compute stream         push_stream            barrier_stream
  ======================         ===========            ==============
  wait ev_reuse[p] ..................................... (recorded at call
  | (producer_acquire:                                    i-2 -- bottom row)
  |  B is EMPTY everywhere)
  g += 1; epoch[p] <- g (4B)
  stage a_shard -> B[me]
  flags[me] <- epoch[p]
  record ev_shard_staged ──────► wait ev_shard_staged
  |                              wait ev_reuse[p]
  |                              | (peers' B replicas EMPTY)
  GEMM, in-kernel gate:          reverse-ring, per peer:
  | spin until                   | CE copy B[me] ~~► peer's B[me]
  | flags[s] >= epoch[p]         | 4B epoch[p] ~~► peer's flags[me]
  | ▲                            |   (releases that peer's gate)
  | '~~ peers' sends fill
  |     MY B + flags
  record ev_gemm_end ───────────────────────────────► wait ev_gemm_end
  (host returns; call i+1                             barrier(channel=p)
   runs on parity 1-p)                                | (all ranks' GEMM
                                                      |  on B finished)
                                                      record ev_reuse[p]
                                                        (consumed by call
                                                         i+2 -- top row)

FULL is fine-grained + point-to-point (latency-critical: gates consumption
now); EMPTY is coarse + collective (throughput-only: gates a refill two
calls later) — that asymmetry is the design, not an accident. The counter
is an EPOCH (unbounded generation, MPI-RMA sense), deliberately NOT an
mbarrier-style 0/1 phase: iteration i+1's FULL signals legally land while
iteration i's consumers still spin (one-deep overlap), which monotonic GEQ
absorbs and parity toggling would deadlock.

Reuse safety: 2 rotating buffers + an off-critical-path cross-rank barrier
gating the staging of call i+2. Epochs are monotonic — no flag resets.
NOTE: cross-rank pacing (per-peer signals, the barrier) measures FASTER
than removing it: unpaced comm drifts into peers' compute windows and
loses more to HBM contention than the sync costs.

Overhead model (regret vs the pure-GEMM roof; all measured): SCHEDULE =
B re-read once per shard (arrival-ordered consumption sweeps all N per
shard; grows with TP); CONTENTION = the irreducible floor, ~0.1 us per MB
of comm traffic (diffuse L2/latency interference, NOT DRAM bandwidth);
GATING is binary — ~0 wherever NVLink delivery fits the GEMM window
(N_local >= ~4096, or 2048 at TP2), dominant in the comm-bound corner
(small N_local x large TP), where TP2-or-no-TP is the saner deployment
anyway. Typical totals at compute-bound shapes: ~3.5% TP2 / ~7% TP4 /
~11-17% TP8. B prefetch was refuted by NCU (stall timeline flat — misses
already pipeline-hidden); A needs none (the inbound CE writes ARE one).

CUDA graphs: whole calls are OUTER-capture-safe — torch.cuda.graph around
ANY number of gather() calls. Capture records enqueued work as a DAG
(edges = stream order + event record->wait pairs); Python runs ONCE, at
capture; replay executes the DAG without re-entering Python. Ordinary
single-stream tensor code captures with no special handling; this file
breaks that mold in three ways, and every is_current_stream_capturing()
check below maps to one of them:

1. WAITS ON EVENTS RECORDED BEFORE THE CAPTURE are illegal, and
   ev_reuse[p] was recorded two calls ago. _wait_reuse skips such waits;
   its docstring has the proof that the skip is redundant, not unsafe.
2. FORKED SIDE STREAMS LEFT DANGLING: capture_end requires every forked
   branch rejoined, but a plain per-call join bakes barrier_i ->
   compute_{i+1} edges — the 1-buffer lockstep schedule. gather() joins
   and then RESTORES the pre-join capture dependency set, making the
   side-branch tails graph LEAVES (still inside the graph — replay N+1
   waits for every node of replay N — but off the captured critical
   path). capture_lockstep=True keeps the join edges instead; both
   schedules and their measurements are documented at the join site in
   gather().
3. HOST VALUES BAKE at capture. Designed away rather than checked: the
   epoch lives in DEVICE memory and flag writes are 4-byte copies FROM
   it — values are read at node execution, fresh and monotone for ANY
   replayed call count (why a global row instead of a per-parity chain:
   see the epoch comment in __init__). Pointers do bake, which is safe:
   persistent symmetric buffers.

Capture contract: replays stream-ordered with each other; all ranks run
identical eager/replay call sequences; switch between EAGER calls and
replays only through runner.quiesce() — a full-quiescence fence +
last-buffer-parity resync (see its docstring; measured without it:
eager->replay under rank skew computes wrong GEMMs, and gathered_a()
after odd-count replays picks the wrong buffer). Capture itself needs no
boundary: torch syncs the device at capture start.

Why TE needs none of this: its overlap is INTRA-call (chunked GEMM,
output complete at return), so every call unconditionally forks and
joins its side streams — a capture sees a closed subgraph. Our checks
are the price of CROSS-CALL pipelining (call i's barrier/sends finish
during call i+1; ev_reuse spans i -> i+2). There is deliberately no
internal per-burst graph layer: measured ~neutral at TP2/TP4 and outer
capture subsumes its one projected win.

Future levers, in rough order of expected value: (1) producer-push north
star — fuse the AG into the producing kernel's epilogue (multimem.st
from CTAs that already own their SMs), making the GEMM side flags-only
and attacking the small-shard corner at its root; (2) table-driven
frontier schedule — a work-id permutation with arrival-aware diagonal
order cutting B sweeps from num_shards to ~2 (batch_idx_permute
precedent); (3) producer chunk-flag early delivery (producers flag
row-chunks so transport rides the producer's window); (4) process groups
beyond WORLD, torch-library custom op for compile-compat.
"""

from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cuda.bindings import runtime

from quack.gemm import AllGatherArguments

__all__ = ["AllGatherRunner"]

_SET_CAPTURE_DEPS = (
    runtime.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies
)


def _check(err_ret):
    err, *ret = err_ret if isinstance(err_ret, tuple) else (err_ret,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return ret[0] if len(ret) == 1 else tuple(ret)


class AllGatherRunner:
    """Transport half of overlapped AllGather+GEMM: owns the rotating
    symmetric gather buffers, arrival flags, streams/events, and the
    CE schedule — and knows NOTHING about the GEMM it feeds.

    The GEMM half is whatever the caller runs inside :meth:`gather`, as
    long as it

    - launches all of its work on the CURRENT stream (that stream-orders the
      runner's end-of-iteration event and reuse barrier; multi-kernel gemms
      are fine),
    - reads A only from the yielded ``a_full`` buffer,
    - forwards ``ag_args`` (a :class:`quack.gemm.AllGatherArguments`, or
      None on the world_size == 1 fast path) verbatim to a persistent dense
      quack GEMM — ``ag_args`` is a universal kwarg, any variant works
      (plain, act/epilogue, blockscaled, ...).

    From the kernel's perspective AG-capability is exactly (a) the
    shard-major rotated scheduler decode and (b) that one flag gate before a
    tile's first TMA — both live in shared code (TileScheduler /
    the arch base class), so enabling a new GEMM variant is one host-side
    ``ag_args`` kwarg. The single geometry fact the two halves share —
    shard and chunk boundaries must land on whole scheduler clusters
    (multiples of tile_M * cluster_M) — is asserted at every launch by the
    shared plan_scheduler_args choke point (validate_ag_geometry), which
    knows the exact tile config; the runner never guesses it.
    """

    def __init__(
        self,
        shard_m: int,
        k: int,
        dtype: torch.dtype,
        group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        # Sub-shard arrival granularity: each send op should stay >= ~64 MB
        # to amortize the CE per-op ramp (measured: 2 chunks at 134 MB shards
        # = -5.4%; 4 chunks at 34 MB shards = +39% REGRESSION — do not chunk
        # small shards). shard_m / arrival_chunks must be a multiple of the
        # consuming GEMM's tile_M * cluster_M; every launch asserts this with
        # its exact config (validate_ag_geometry).
        arrival_chunks: int = 1,
        # Captured-schedule choice, read at CAPTURE time (bakes per graph;
        # flip the attribute between captures to mix). False = dep-restore:
        # captured calls keep the eager schedule's 2-buffer skew slack.
        # True = every captured call paced behind the cross-rank barrier
        # (the 1-buffer-style schedule) — measured FASTER in uniform
        # steady-state replay loops (TP4 1.5-5%), worse in theory under
        # per-rank jitter (each call waits the fleet max). See gather().
        capture_lockstep: bool = False,
    ):
        self.group = group if group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        self.shard_m = shard_m
        self.k = k
        self.m_total = shard_m * self.world_size
        self.dtype = dtype
        self.device = (
            device if device is not None else torch.device("cuda", torch.cuda.current_device())
        )
        assert arrival_chunks >= 1
        self.arrival_chunks = arrival_chunks
        self.capture_lockstep = capture_lockstep

        torch.cuda.set_device(self.device)
        # Exactly 2 rotating buffers, permanently: consumers never need
        # gathered A past the next call (training REGATHERS in backward —
        # the saved tensor is the shard, not A_full), so deeper rotation
        # buys nothing and 2 x (M_total, K) is the whole footprint. ONE
        # symmetric (2, M_total, K) tensor, one collective rendezvous, one
        # handle: bufs[parity] indexes a buffer, peer addresses are
        # handle.buffer_ptrs[r] + parity * buffer_bytes, and the two
        # independent barrier states the old per-buffer handles provided are
        # barrier(channel=parity) instead.
        self.bufs = symm_mem.empty((2, self.m_total, k), dtype=dtype, device=self.device)
        self.handle = symm_mem.rendezvous(self.bufs, self.group.group_name)
        self.buffer_bytes = self.m_total * k * dtype.itemsize
        self.shard_bytes = shard_m * k * dtype.itemsize
        # Arrival flags, chunk-major within a shard: flag[shard * chunks + c]
        # is set (a 4-byte copy of the call's device epoch) once chunk c of
        # that shard is in local HBM. The kernel gate releases per chunk
        # (ag_wait_m_tile). Written REMOTELY (to the peer VA after each
        # send) -> symmetric.
        self.flags = symm_mem.empty(
            self.world_size * arrival_chunks, dtype=torch.int32, device=self.device
        )
        self.flags.zero_()
        self.flags_handle = symm_mem.rendezvous(self.flags, self.group.group_name)
        self.push_stream = torch.cuda.Stream(self.device)
        # THE THREE LANES (proven minimum — see docstring): the AMBIENT
        # stream (the caller's current stream) runs epoch-bump/stage/GEMM;
        # push_stream runs the send burst (must overlap the GEMM);
        # barrier_stream runs ONLY the empty-side barrier — merging it into
        # push_stream would queue sends(i+1) behind barrier(i) = all ranks'
        # gemm(i), re-introducing the one-iteration over-sync the
        # per-buffer ev_reuse removed; merging into the ambient stream
        # would put a cross-rank rendezvous on the critical path.
        self.barrier_stream = torch.cuda.Stream(self.device)
        # DEVICE-resident epoch: rows 0/1 = per-parity SNAPSHOTS, row 2 = the
        # GLOBAL counter. The bump is g += 1 (one captured kernel) followed
        # by a 4-byte snapshot copy g -> epoch[parity]; the snapshot is what
        # this call's gate reads (through a pointer) and what its flag
        # writers copy out at EXECUTION time — so the value stays stable for
        # the whole call even after the next call bumps g, and nothing
        # sync-related is ever host-baked (whole calls are graph-capturable
        # for ANY call count: monotonicity lives in the single global row,
        # not in parity alternation — a chained per-parity bump would reuse
        # an epoch value whenever a replayed ODD capture put two same-parity
        # calls back to back). Slot p is rewritten at call i+2 only after
        # ev_reuse[p] (all of call i's gates passed => its flag copies
        # executed), so in-flight readers never see a torn snapshot. int32
        # wrap is a non-event: the gate compares with modular GEQ (TE's
        # CHECK_IDS trick) — no resync path exists. (The kernel declares
        # the epoch tensor 4-byte aligned — one scalar load — so rows at 4-byte
        # offsets from the base are fine.)
        self.epoch = torch.zeros(3, 1, dtype=torch.int32, device=self.device)
        # Outer-capture bookkeeping: gather() detects a NEW capture via the
        # capture-sequence id and resets the per-buffer "reuse recorded in
        # this capture" flags (consumed by _wait_reuse's skip logic).
        self._capture_id = None
        self._reuse_in_capture = [False] * len(self.bufs)
        self._ev_join = torch.cuda.Event()  # capture-only side-branch join
        self.ev_shard_staged = torch.cuda.Event()
        self.ev_gemm_end = torch.cuda.Event()
        # Per-BUFFER empty events (EMPTY is a per-buffer property): the
        # event for buffer b is recorded after iteration i's barrier (i%2==b)
        # and waited by iteration i+2 — so iteration i+1's transport can run
        # a full iteration ahead of the slowest rank's gemm_i instead of
        # serializing behind it (a single shared event over-synchronizes by
        # one iteration and forfeits exactly the skew slack 2 buffers buy).
        self.ev_reuse = [torch.cuda.Event() for _ in range(len(self.bufs))]
        # First iteration has no prior reuse barrier; record it satisfied.
        for ev in self.ev_reuse:
            ev.record(torch.cuda.current_stream(self.device))
        # Captured calls use their OWN reuse events: a capture-recorded
        # event stays invalid for eager cudaStreamWaitEvent even after
        # replays, so sharing one pair would poison eager use after the
        # first capture (see _wait_reuse).
        self.ev_reuse_capture = [torch.cuda.Event() for _ in range(len(self.bufs))]
        # Parity (0/1) of the buffer the LAST EXECUTED call used — the ONLY
        # host-side state the runner needs (buffer/event/channel selection
        # and the accessors all key off WHICH buffer, never off a count;
        # a count cannot name a buffer: replayed graphs end on their BAKED
        # final parity regardless of how many times they replayed).
        self.last_parity = 0
        self._capture_parity = 0  # capture-local parity sequencer (see gather())
        self._in_gather = False  # reentrancy guard for gather()
        dist.barrier(self.group)  # flags zeroed everywhere before first use

    def gathered_a(self) -> torch.Tensor:
        """The gathered A of the last call — VALID ONLY UNTIL THE NEXT CALL
        (buffers rotate every 2 calls; with one runner shared across layers
        that is the next layer). Consume immediately (e.g. a fused wgrad) or
        don't: training regathers in backward instead of saving A_full.

        Tracks last_parity, which replays do NOT advance: after graph
        replays, call quiesce() first or this may select the wrong buffer.
        """
        return self.bufs[self.last_parity]

    def next_local_slot(self) -> torch.Tensor:
        """The local-shard slice of the buffer the NEXT gather() will use.

        Zero-copy producer fusion: have the op producing A write directly into
        this view, then pass it as ``a_shard`` — gather() detects the aliasing
        and skips the staging copy. NOTE: writing here races with lagging peers
        of the PREVIOUS call unless the producer runs after ev_reuse; when in
        doubt, pass a plain tensor and let gather() stage it. Tracks
        last_parity (see gathered_a): after graph replays, quiesce() first.
        """
        buf = self.bufs[self.last_parity ^ 1]
        return buf[self.rank * self.shard_m : (self.rank + 1) * self.shard_m]

    @property
    def last_capture_parity(self) -> int:
        """Baked parity of the LAST call of the most recent capture —
        constant across replays of that graph (buffer choices bake).
        Snapshot this right after capturing and hand it to
        quiesce(last_parity=...) after replaying that graph, for a
        sync-free boundary."""
        return self._capture_parity

    def quiesce(
        self,
        last_parity: int | None = None,
        replay_stream: torch.cuda.Stream | None = None,
    ):
        """THE sanctioned boundary for switching between eager gather()
        calls and graph replays (either direction). Call it on every rank at
        the same point in the call sequence (standard collective congruence
        — ranks must run identical eager/replay sequences), on the stream
        the replays were launched on — or pass that stream as
        ``replay_stream`` and quiesce() establishes the ordering edge
        itself (without it, everything below is ordered only against the
        CURRENT stream and misses replays on other streams entirely: the
        device-epoch read lands one call behind, measured). It does two
        things:

        1. Stream-orders full reuse quiescence onto the CURRENT stream —
           waits both buffers' EMPTY events (each certifies EVERY rank's
           outstanding GEMMs via the aggregated barrier) and joins the
           send/barrier side streams. This is the edge an eager->replay
           transition otherwise lacks: a replay's first captured calls
           skipped their reuse waits, and a bare replay is ordered only
           behind MY compute tail, which says nothing about a lagging
           peer's previous GEMM. No host or dist sync involved for this
           part. (After replays these event waits are stale-but-redundant:
           whole-graph completion already certified the graph work.)
        2. Resyncs last_parity — identifies WHICH PHYSICAL BUFFER the last
           executed call left newest, which replays change without the host
           seeing. A call COUNT cannot answer this (measured: a 1-call graph
           replayed twice always ends on its BAKED parity while any count's
           parity flips — count-based resync picked the wrong buffer), so:
           pass ``last_parity`` = the replayed graph's baked final parity
           (snapshot runner.last_capture_parity right after capturing) for
           a SYNC-FREE boundary; for a pure fence with no graph work since
           the last eager call, pass the runner's own ``last_parity``.
           With ``last_parity=None`` the buffer is DISCOVERED from the
           device: read both parity snapshots and take the modular-newest
           slot — a genuine GPU->CPU sync (the host blocks until the stream
           reaches the fence), unavoidable only because the runner must
           learn something it never observed. A wrong caller-supplied
           parity is not detectable without the sync — it silently
           mis-points the accessors.
        """
        assert not torch.cuda.is_current_stream_capturing(), "quiesce() is an eager-only boundary"
        compute_stream = torch.cuda.current_stream(self.device)
        if replay_stream is not None:
            compute_stream.wait_stream(replay_stream)
        for parity in range(len(self.bufs)):
            compute_stream.wait_event(self.ev_reuse[parity])
        self._ev_join.record(self.push_stream)
        compute_stream.wait_event(self._ev_join)
        self._ev_join.record(self.barrier_stream)
        compute_stream.wait_event(self._ev_join)
        if last_parity is None:
            # Which snapshot slot is newest? Modular signed compare (slots
            # can differ by more than 1: same-parity calls re-bump one
            # slot), robust to int32 wrap. Equal only before any call.
            snapshots = self.epoch[:2].cpu()  # blocking D2H = the sync
            difference = (int(snapshots[1, 0]) - int(snapshots[0, 0])) & 0xFFFFFFFF
            if difference != 0:
                self.last_parity = 1 if difference < 0x80000000 else 0
        else:
            assert last_parity in (0, 1)
            self.last_parity = last_parity

    def _ag_args(self, parity: int) -> AllGatherArguments:
        return AllGatherArguments(
            flags=self.flags,
            epoch=self.epoch[parity],
            num_shards=self.world_size,
            first_shard=self.rank,
            num_chunks=self.arrival_chunks,
        )

    def _wait_reuse(self, stream: torch.cuda.Stream, parity: int):
        """producer_acquire on the reuse event, capture-aware. Inside an
        outer CUDA-graph capture, an event last recorded BEFORE the capture
        cannot be waited (illegal dependency on uncaptured work) — and does
        not need to be: torch.cuda.graph synchronizes the device at capture
        start, so pre-capture reuse is satisfied by construction. Across
        REPLAYS the guarantee comes from graph-completion semantics: a
        launched graph is a SINGLE stream operation, so replay N+1 (and
        anything else on the launch stream) starts only after EVERY node of
        replay N completed — all barriers included — hence stream-ordered
        replays cannot race buffer reuse. We therefore skip waits until the
        reuse event has been recorded within THIS capture.

        Captured calls use a SEPARATE event pair (ev_reuse_capture): an
        event whose last record is a capture node stays invalid for EAGER
        cudaStreamWaitEvent even after replays execute the node — one
        shared pair would poison eager use of the runner forever after its
        first capture. Keeping the eager pair capture-untouched, an eager
        call after graph work waits the last EAGER record (satisfied —
        redundant-but-valid under the sync-boundary contract in the module
        docstring)."""
        # (new-capture detection/reset happens once per call, in gather())
        if torch.cuda.is_current_stream_capturing():
            if self._reuse_in_capture[parity]:
                stream.wait_event(self.ev_reuse_capture[parity])
        else:
            stream.wait_event(self.ev_reuse[parity])

    def _record_reuse(self, parity: int):
        # publish EMPTY
        if torch.cuda.is_current_stream_capturing():
            self.ev_reuse_capture[parity].record(self.barrier_stream)
            self._reuse_in_capture[parity] = True
        else:
            self.ev_reuse[parity].record(self.barrier_stream)

    def _write_flag(self, flags_base_ptr: int, parity: int, chunk: int, stream: torch.cuda.Stream):
        """producer_commit for one (my shard, chunk) arrival flag, in the
        flag replica at ``flags_base_ptr`` (mine, or a peer's for remote
        commits): a 4-byte CE copy of device epoch[parity], so the VALUE is
        read from device memory at execution time (graph-replay- and
        capture-safe).

        Why CE and not a torch op: for PEER commits it is load-bearing — a
        torch op is an SM kernel, and the peer's persistent GEMM owns every
        SM while spinning on exactly this flag (SM writers starve behind
        it); CE needs no SMs, and same-engine stream order after the data
        memcpy makes flag-implies-data free. Locally an SM op would be
        correct (ordered before the GEMM), but the raw memcpy enqueue is
        sub-us vs the torch dispatcher's ~3.5 us in the window that delays
        the GEMM launch — and one mechanism means one visibility argument.
        (cuStreamWriteValue32, the purpose-built API, measured ~10 us per
        call — rejected.)"""
        _check(
            runtime.cudaMemcpyAsync(
                flags_base_ptr + 4 * (self.rank * self.arrival_chunks + chunk),
                self.epoch[parity].data_ptr(),
                4,
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                stream.cuda_stream,
            )
        )

    def _bump_epoch(self, parity: int, stream: torch.cuda.Stream):
        """Epoch bump: g += 1 on the global row (captured kernel, no host
        value), then a 4-byte snapshot copy g -> epoch[parity] for this
        call's gate, flag writers, and bare-quiesce buffer discovery.
        Monotone for ANY replayed call count — see the epoch comment in
        __init__. Runs on EVERY call, including world_size == 1 (no gate or
        flags there, but quiesce()'s device discovery must still see which
        buffer a replay wrote last)."""
        torch.add(self.epoch[2], 1, out=self.epoch[2])
        _check(
            runtime.cudaMemcpyAsync(
                self.epoch[parity].data_ptr(),
                self.epoch[2].data_ptr(),
                4,
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                stream.cuda_stream,
            )
        )

    def _stage_local(self, a_shard, buf, parity: int):
        """Shared per-iteration prologue, on the ambient (compute) stream:
        wait for buffer reuse safety, stage the local shard into its slot,
        set its arrival flag, record ev_shard_staged.

        SM-kernel copy, NOT tensor.copy_: a contiguous same-dtype copy_
        lowers to a CE memcpy, which has ~20-30 us of ramp at shard sizes and
        delays everything gated on ev_shard_staged (peer pulls / our sends)
        by that much. The elementwise kernel runs at HBM rate.
        """
        stream = torch.cuda.current_stream(self.device)
        # producer_acquire: THIS buffer EMPTY everywhere (barrier two calls
        # ago). Must precede the epoch bump: that call's late flag copies
        # read epoch[parity] at execution, and this wait proves they are done.
        self._wait_reuse(stream, parity)
        self._bump_epoch(parity, stream)
        local_slot = buf[self.rank * self.shard_m : (self.rank + 1) * self.shard_m]
        if a_shard.data_ptr() != local_slot.data_ptr():
            torch.add(a_shard, 0, out=local_slot)
        # else: producer already wrote the symmetric slot (next_local_slot).
        # producer_commit: local shard FULL — ALL chunk flags at once, since
        # the one staging kernel above lands the whole shard together.
        for c in range(self.arrival_chunks):
            self._write_flag(self.flags.data_ptr(), parity, c, stream)
        # intra-producer edge (stage -> transport), not a pipeline barrier
        self.ev_shard_staged.record()

    @contextmanager
    def gather(self, a_shard: torch.Tensor):
        """Overlapped all-gather around any GEMM(s) — the primary API::

            with runner.gather(a_shard) as (a_full, ag_args):
                d = gemm(a_full, b, ..., ag_args=ag_args)

        On ENTER: acquire the rotating buffer, stage the local shard, commit
        its arrival flags. a_full is NOT gathered yet — the body's GEMM
        consumes remote shards as they arrive, via the in-kernel gate.

        The BODY must enqueue every consumer of a_full on the current stream
        and forward ag_args (an AllGatherArguments; None on the
        world_size == 1 fast path) to the quack GEMM(s) — ag_args is a
        universal kwarg, any variant works. ("Gate" in this file always
        means the AG flag gate in the AB-load warp, ag_wait_m_tile — not a
        gated activation.) Several GEMMs may share one gather (e.g. q/k/v
        projections) — they all wait on the same flags.

        On EXIT: record consumer_release, enqueue the CE sends and the reuse
        barrier. Transport is enqueued AFTER the body on purpose — the GEMM
        launch isn't delayed by the sends' host enqueue cost, and the sends
        still land while it runs. Under capture, exit also handles the
        side-stream joins — schedule chosen by capture_lockstep, see the
        comment at the join site. If the body raises, the exit transport is
        skipped and peers stall on this rank's shard until job teardown (an
        exception inside a collective is fatal to the step, as usual).

        CE-push transport (on exit): the OWNER sends its shard to each peer
        with cudaMemcpyAsync in REVERSE ring order (to rank-1 first, rank-2
        next, ...), then remote-writes that peer's flag. Reverse order makes
        every receiver get its next-needed shard first under the ring-
        ROTATED consumption order (one incoming transfer per rank per step —
        the send schedule is a permutation each step), at CE's full unicast
        rate — while the producer dependency is plain sender stream order:
        no readiness handshake at all.
        """
        assert a_shard.shape == (self.shard_m, self.k) and a_shard.dtype == self.dtype
        assert a_shard.stride(-1) == 1, "A shard must be K-major (row-contiguous)"
        assert not self._in_gather, "gather() calls cannot nest or overlap"
        self._in_gather = True
        try:
            # last_parity tracks the buffer of the last EXECUTED call.
            # Captured calls do NOT advance it — capture runs Python but
            # executes nothing; the buffer their replays leave newest is the
            # capture's BAKED final parity, which the caller reports at
            # quiesce(). Captured parities sequence from a capture-local
            # bit seeded from the eager state, so a capture continues the
            # eager buffer alternation.
            if torch.cuda.is_current_stream_capturing():
                # GetCaptureInfo returns (status, id, graph, deps, edgeData,
                # numDeps); [1] is the unique capture-sequence id int.
                _, cap_id, *_ = _check(
                    runtime.cudaStreamGetCaptureInfo(
                        torch.cuda.current_stream(self.device).cuda_stream
                    )
                )
                if cap_id != self._capture_id:  # new capture begins
                    self._capture_id = cap_id
                    self._reuse_in_capture = [False] * len(self.bufs)
                    self._capture_parity = self.last_parity
                self._capture_parity ^= 1
                parity = self._capture_parity
            else:
                self.last_parity ^= 1
                parity = self.last_parity
            buf = self.bufs[parity]

            if self.world_size == 1:
                # Degenerate single-rank path: no transport, flags, or gate
                # (ag_args=None) — but the epoch bump still runs so bare
                # quiesce()'s buffer discovery sees replays here too
                # (measured: without it, discovery picked the wrong buffer;
                # cost is noise next to the resident-buffer copy below).
                self._bump_epoch(parity, torch.cuda.current_stream(self.device))
                buf[:].copy_(a_shard)
                yield buf, None
                return

            # 1. Local staging + local flag.
            self._stage_local(a_shard, buf, parity)

            # 2. The body: GEMM(s) immediately (rotated order, per-shard
            #    flags); consumer_wait happens IN-KERNEL (gate).
            yield buf, self._ag_args(parity)
            self.ev_gemm_end.record()  # my consumer_release (local half)

            # 3. Sends, reverse ring order, on the push stream. Reuse safety:
            #    these sends write PEERS' buffers, so they must wait until
            #    all ranks' gemm two calls ago released this buffer —
            #    ev_reuse carries that (recorded after the post-gemm barrier
            #    below). Stream order after ev_shard_staged carries the
            #    producer dependency for free. This loop is measured-OPTIMAL
            #    at the API level — multi-stream fan-out, cudaMemcpyBatchAsync,
            #    and flag-offload all measured worse ("Measured transport
            #    physics" in the module docstring); don't "optimize" it.
            self.push_stream.wait_event(self.ev_shard_staged)  # stage->transport edge
            # producer_acquire (PEERS' replicas of THIS buffer)
            self._wait_reuse(self.push_stream, parity)

            src_base = buf.data_ptr() + self.rank * self.shard_bytes  # the staged slot
            my_off = self.rank * self.shard_bytes
            chunk_bytes = self.shard_bytes // self.arrival_chunks
            for step in range(1, self.world_size):
                dst_rank = (self.rank - step) % self.world_size
                dst_base = self.handle.buffer_ptrs[dst_rank] + parity * self.buffer_bytes + my_off
                for c in range(self.arrival_chunks):
                    _check(
                        runtime.cudaMemcpyAsync(
                            dst_base + c * chunk_bytes,
                            src_base + c * chunk_bytes,
                            chunk_bytes,
                            runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                            self.push_stream.cuda_stream,
                        )
                    )
                    # producer_commit on the PEER's replica, right after that
                    # chunk's data memcpy — their consumer_wait releases. CE
                    # is LOAD-BEARING here (see _write_flag: the peer's
                    # persistent GEMM owns every SM while spinning on exactly
                    # this flag).
                    self._write_flag(
                        self.flags_handle.buffer_ptrs[dst_rank], parity, c, self.push_stream
                    )

            # 4. Reuse barrier off the critical path (all gemms done =>
            #    buffer safe for everyone's staging and sends two calls from
            #    now).
            self.barrier_stream.wait_event(self.ev_gemm_end)
            with torch.cuda.stream(self.barrier_stream):
                # aggregate consumer_release across ranks; channel=parity
                # keeps the two buffer generations' barrier states
                # independent (what per-buffer handles used to provide)
                self.handle.barrier(channel=parity)
            self._record_reuse(parity)
            if torch.cuda.is_current_stream_capturing():
                # capture_end requires the forked side streams rejoined
                # (cudaErrorStreamCaptureUnjoined). Two schedules, chosen by
                # capture_lockstep (read at CAPTURE time — bakes per graph):
                # - False (dep-restore): join for legality, then RESTORE the
                #   pre-join capture dependency set. The side-branch tails
                #   become graph LEAVES — still in the graph (replay N+1
                #   waits for EVERY node of replay N) but unreachable from
                #   the next captured call: the eager dangling-branch shape,
                #   in graph form (the documented library pattern for
                #   cudaStreamUpdateCaptureDependencies: "nodes removed from
                #   the dependency set do not result in
                #   cudaErrorStreamCaptureUnjoined"). Preserves the
                #   2-buffer skew slack.
                # - True (lockstep): leave the join edges in — barrier_i ->
                #   compute_{i+1}, every captured call paced behind the
                #   cross-rank barrier. MEASURED FASTER in uniform
                #   steady-state loops (TP4 1.5-5%, TP2 big shapes ~1-4%:
                #   barrier pacing keeps send bursts out of peers' compute
                #   windows, and with no skew to absorb the slack buys
                #   nothing). Prefer under jitter-free replay loops; the
                #   dep-restore default may flip after a quiet-window TP8
                #   A/B.
                compute_stream = torch.cuda.current_stream(self.device)
                if not self.capture_lockstep:
                    _, _, _, deps, _, num_deps = _check(
                        runtime.cudaStreamGetCaptureInfo(compute_stream.cuda_stream)
                    )
                self._ev_join.record(self.push_stream)
                compute_stream.wait_event(self._ev_join)
                self._ev_join.record(self.barrier_stream)
                compute_stream.wait_event(self._ev_join)
                if not self.capture_lockstep:
                    _check(
                        runtime.cudaStreamUpdateCaptureDependencies(
                            compute_stream.cuda_stream, deps, None, num_deps, _SET_CAPTURE_DEPS
                        )
                    )
        finally:
            self._in_gather = False
