"""MINIMAL REPRODUCER (solved): clusterlaunchcontrol.try_cancel returns
INVALID ("no pending cluster") spuriously while the grid's pending pool holds
hundreds of clusters — B300 (sm_103a), driver 595, CUDA 13.2.

REPRO RECIPE (fires ~2.5% of launches, ~75 proven events per 3000):
  1. co-tenant process on the same GPU (any kernels, e.g. a matmul loop
     3s-on/2s-off — cross-context time-slicing is REQUIRED; quiet GPU: zero)
  2. this probe:  python AI/repro_clc_spurious_invalid.py \
                      --trials 3000 --stream-depth 300
     i.e. a plain serial try_cancel steal loop (no multicast, no PDL, no
     TMEM, no TMA — none needed), grid 1344 clusters >> ~74 resident, with
     launches queued BACK-TO-BACK on one stream (sync only per batch).
  3. detection is self-proving: on an invalid response the probe simply
     RETRIES; a grant on the retry proves the pool was non-empty. No
     counters or margins involved in the verdict.

INGREDIENT MATRIX (each necessary, jointly sufficient):
  * co-tenant contender:        required (quiet: 0/3000)
  * queued launch depth:        required, threshold between 2 and 10
                                (depth 2: 0; depth 10: 78; depth 300: 73)
  * grid >> resident capacity:  required (grid 256: 0; grid 1344: fires)
  Exonerated by A/B (all clean over 150k+ per-launch-synced retirements):
  multicast queries, PDL, back-to-back PDL grids, pipelined 2-in-flight
  queries, TMEM allocation, tcgen05 MMA execution, deep TMA-tx pressure,
  LDG saturation, kernel duration (36ms), cluster shape, contender type.
  Cross-grid grants: never observed (0/800 xgrid) — the spurious invalid is
  a fail-negative, not a mis-routed grant.

IMPACT: any CLC scheduler treating invalid-response as pool-empty retires
early. Harmless for exact grids (un-granted clusters just launch), but it
was the root cause of the July 2026 quack varlen corruption: the padding
drain (removed/fixed in quack/tile_scheduler.py cancel_pending_tail) fired
cancels at such retirements, canceling REAL pending tiles whose output rows
then silently kept stale memory.

In-situ variant (fires in 1-2 sweep iterations): quack varlen gemm loop +
contender + exit-valid printf in the drain; 24/24 corrupting drains showed
an invalid exit at mid-pool position followed by 79-256 valid grants.
"""

import argparse

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass import Int32, Boolean, const_expr

CLUSTER_M = 2
HIST_BINS = 16  # log2-ish bins of pending_est at invalid


class SpuriousInvalidProbe:
    def __init__(
        self,
        spin_ns: int,
        smem_pad_kb: int,
        use_pdl: bool = False,
        multicast: bool = False,
        traffic: bool = False,
        tma_traffic: bool = False,
        pipelined: bool = False,
        tmem: bool = False,
    ):
        self.spin_ns = spin_ns
        self.smem_pad_kb = smem_pad_kb
        self.use_pdl = use_pdl
        self.multicast = multicast
        # traffic=True: 3 companion warps per CTA hammer gmem while the
        # scheduler thread steals — mimics the GEMM's memory pressure on the
        # SM<->CWD path.
        self.traffic = traffic
        # tma_traffic=True: companion warps issue continuous cp.async.bulk
        # G2S copies with mbarrier::complete_tx tracking — saturating the
        # async queue + tx machinery the CLC response also rides.
        self.tma_traffic = tma_traffic
        # pipelined=True: keep TWO try_cancels in flight (production's sched
        # pipeline shape) — consume stage s while s^1 is already issued.
        # Retry-detector applies at each invalid.
        self.pipelined = pipelined
        # tmem=True: allocate full tensor memory for the CTA lifetime (like a
        # GEMM), dealloc at exit. Tests whether TMEM-holding CTAs perturb CLC.
        self.tmem = tmem

    @cute.kernel
    def kernel(
        self,
        mCounters: cute.Tensor,  # (4,) or (K,4): [0] started, [1] granted, [2] spurious/xgrid, [3] max_pending
        mHist: cute.Tensor,  # (HIST_BINS,) histogram of pending_est at invalid
        mTraffic: cute.Tensor,  # big gmem buffer for traffic warps
        grid_clusters: Int32,
        margin: Int32,
        slot: Int32,  # which counter row (back-to-back mode); 0 in single mode
        over_spray: Int32,  # extra observed cancels at retirement (xgrid mode)
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        if const_expr(self.smem_pad_kb > 0):
            smem.allocate_tensor(Int32, cute.make_layout(self.smem_pad_kb * 256), byte_alignment=16)
        resp = smem.allocate_tensor(Int32, cute.make_layout(4), byte_alignment=16)
        mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1), byte_alignment=8)
        # staging + per-warp tx-mbarriers for tma traffic. Allocated
        # unconditionally: the DSL traces branch bodies even under a False
        # const_expr when the condition also has dynamic terms.
        stage = smem.allocate_tensor(Int32, cute.make_layout(4096), byte_alignment=128)
        tmbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(3), byte_alignment=8)
        tmbar2 = smem.allocate_tensor(cutlass.Int64, cute.make_layout(3), byte_alignment=8)
        # v2 tma staging: 6 x 32KB slots (192KB) — pass --smem-pad-kb 0 with
        # --tma-traffic or the CTA won't fit.
        tstage = smem.allocate_tensor(
            Int32,
            cute.make_layout(49152 if const_expr(self.tma_traffic) else 4),
            byte_alignment=128,
        )

        cta_rank = cute.arch.block_idx_in_cluster()
        base = mCounters.iterator + 4 * slot
        if tidx == 0:
            cute.arch.mbarrier_init(mbar.iterator, 1)
            for w in cutlass.range_constexpr(3):
                cute.arch.mbarrier_init(tmbar.iterator + w, 1)
                cute.arch.mbarrier_init(tmbar2.iterator + w, 1)
            cute.arch.mbarrier_init_fence()
        if const_expr(self.tma_traffic):
            cute.arch.barrier()  # tmbar visible to traffic warps
        if const_expr(self.tmem):
            # Hold a full TMEM allocation for the CTA's lifetime, like a GEMM.
            if tidx == 0:
                cute.arch.alloc_tmem(512, stage.iterator, is_two_cta=False)
        if const_expr(self.multicast):
            # All CTAs must have initialized barriers before the first
            # multicast query can arm them (all threads arrive).
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        if cta_rank == 0 and tidx == 0 and const_expr(self.pipelined):
            # ---- pipelined variant: 2 queries in flight (production shape) ----
            cute.arch.atomic_add(base, Int32(1))  # started
            mb0 = mbar.iterator
            mb1 = tmbar.iterator + 2
            r0 = resp.iterator
            r1 = stage.iterator  # 16B of the staging buffer as resp slot 1
            cute.arch.mbarrier_arrive_and_expect_tx(mb0, 16)
            cute.arch.issue_clc_query(mb0, r0, multicast=False)
            cute.arch.mbarrier_arrive_and_expect_tx(mb1, 16)
            cute.arch.issue_clc_query(mb1, r1, multicast=False)
            cur = Int32(0)
            ph0, ph1 = Int32(0), Int32(0)
            consec_invalid = Int32(0)
            going = Boolean(True)
            while going:
                if cur == 0:
                    cute.arch.mbarrier_wait(mb0, ph0)
                    ph0 = ph0 ^ 1
                else:
                    cute.arch.mbarrier_wait(mb1, ph1)
                    ph1 = ph1 ^ 1
                v = Int32(0)
                if cur == 0:
                    _, _, _, v = cute.arch.clc_response(r0)
                else:
                    _, _, _, v = cute.arch.clc_response(r1)
                cute.arch.fence_view_async_shared()
                if v != 0:
                    if consec_invalid > 0:
                        cute.arch.atomic_add(base + 2, Int32(1))  # retry-proven
                    consec_invalid = Int32(0)
                    cute.arch.atomic_add(base + 1, Int32(1))
                    if const_expr(self.spin_ns > 0):
                        for _ in cutlass.range_constexpr(max(1, self.spin_ns // 1_000_000)):
                            cute.arch.inline_ptx(
                                "nanosleep.u32 {$r0};",
                                read_only_args=[Int32(min(self.spin_ns, 1_000_000))],
                            )
                else:
                    consec_invalid = consec_invalid + 1
                    if consec_invalid >= 4:
                        going = Boolean(False)
                # re-issue the stage we just consumed (even after invalid: this
                # doubles as the retry)
                if going:
                    if cur == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(mb0, 16)
                        cute.arch.issue_clc_query(mb0, r0, multicast=False)
                    else:
                        cute.arch.mbarrier_arrive_and_expect_tx(mb1, 16)
                        cute.arch.issue_clc_query(mb1, r1, multicast=False)
                    cur = cur ^ 1
            # drain the other in-flight query before exit (DSL scoping: define
            # v2 outside the branch)
            other = cur ^ 1
            v2 = Int32(0)
            if other == 0:
                cute.arch.mbarrier_wait(mb0, ph0)
                _, _, _, v2a = cute.arch.clc_response(r0)
                v2 = v2a
            else:
                cute.arch.mbarrier_wait(mb1, ph1)
                _, _, _, v2b = cute.arch.clc_response(r1)
                v2 = v2b
            cute.arch.fence_view_async_shared()
            if v2 != 0:
                cute.arch.atomic_add(base + 1, Int32(1))
                cute.arch.atomic_add(base + 2, Int32(1))  # grant after invalid-exit

        elif cta_rank == 0 and tidx == 0:
            cute.arch.atomic_add(base, Int32(1))  # started
            if const_expr(self.use_pdl):
                # Let the dependent (next back-to-back) grid start launching:
                # its clusters enter the CWD pending pool while we still run.
                cute.arch.griddepcontrol_launch_dependents()
            phase = Int32(0)
            going = Boolean(True)
            consec_invalid = Int32(0)
            while going:
                if const_expr(self.multicast):
                    for r in cutlass.range_constexpr(CLUSTER_M):
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar.iterator, 16, r)
                else:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar.iterator, 16)
                cute.arch.issue_clc_query(mbar.iterator, resp.iterator, multicast=self.multicast)
                cute.arch.mbarrier_wait(mbar.iterator, phase)
                phase = phase ^ 1
                _, _, _, valid = cute.arch.clc_response(resp.iterator)
                cute.arch.fence_view_async_shared()
                if valid != 0:
                    if consec_invalid > 0:
                        # GRANT AFTER INVALID: the pool cannot refill, so the
                        # earlier invalid was SPURIOUS. Counter-lag-immune.
                        cute.arch.atomic_add(base + 2, Int32(1))
                    consec_invalid = Int32(0)
                    cute.arch.atomic_add(base + 1, Int32(1))  # granted
                    if const_expr(self.spin_ns > 0):  # per-"tile" work
                        for _ in cutlass.range_constexpr(max(1, self.spin_ns // 1_000_000)):
                            cute.arch.inline_ptx(
                                "nanosleep.u32 {$r0};",
                                read_only_args=[Int32(min(self.spin_ns, 1_000_000))],
                            )
                else:
                    # INVALID: the hardware claims the pool is empty. Check.
                    # atomic_add(0) = coherent read of the global counters.
                    s = cute.arch.atomic_add(base, Int32(0))
                    g = cute.arch.atomic_add(base + 1, Int32(0))
                    pending = grid_clusters - s - g
                    if pending < 0:
                        pending = Int32(0)
                    # histogram bin: 0, 1-2, 3-4, 5-8, ... (log2)
                    b = Int32(0)
                    v = pending
                    while v > 0:
                        b = b + 1
                        v = v // 2
                    if b > HIST_BINS - 1:
                        b = Int32(HIST_BINS - 1)
                    cute.arch.atomic_add(mHist.iterator + b, Int32(1))
                    # (margin-based estimate retired: superseded by the
                    # retry test, which is immune to counter lag)
                    # track max pending seen at any invalid (racy max via CAS-free
                    # best-effort: add only when strictly larger is fine to skip;
                    # use atomic_max semantics via red if unavailable, just store)
                    old = cute.arch.atomic_add(base + 3, Int32(0))
                    if pending > old:
                        # benign race: last-writer-wins is fine for a report
                        cute.arch.atomic_add(base + 3, pending - old)
                    # Retry test: retire only after 3 consecutive invalids
                    # (~100us apart). A grant on any retry proves the previous
                    # invalid was spurious.
                    consec_invalid = consec_invalid + 1
                    if consec_invalid >= 3:
                        going = Boolean(False)
                    else:
                        cute.arch.inline_ptx("nanosleep.u32 {$r0};", read_only_args=[Int32(100000)])

            # xgrid mode: at retirement (own pool believed empty), keep issuing
            # observed cancels. With PDL, the NEXT grid's clusters are pending
            # in the CWD — any grant here is a CROSS-GRID cancel. Count grants
            # into THIS slot's granted; the victim grid's closure then shows
            # started+granted < G, and this grid shows an excess.
            ks = Int32(0)
            while ks < over_spray:
                cute.arch.mbarrier_arrive_and_expect_tx(mbar.iterator, 16)
                cute.arch.issue_clc_query(mbar.iterator, resp.iterator, multicast=False)
                cute.arch.mbarrier_wait(mbar.iterator, phase)
                phase = phase ^ 1
                _, _, _, valid2 = cute.arch.clc_response(resp.iterator)
                cute.arch.fence_view_async_shared()
                if valid2 != 0:
                    cute.arch.atomic_add(base + 1, Int32(1))  # granted (possibly x-grid)
                ks = ks + 1
        elif const_expr(self.tma_traffic) and tidx >= 32 and tidx % 32 == 0:
            # TMA-traffic warps v2: per warp TWO 32KB cp.async.bulk G2S in
            # flight (6 ops / 192KB outstanding per CTA), each tracked by its
            # own tx-mbarrier — GEMM-like pressure on the async/tx machinery.
            w = tidx // 32 - 1
            mb_a = tmbar.iterator + w
            mb_b = tmbar2.iterator + w
            dst_a = tstage.iterator + w * 8192  # Int32 elements: 32KB slot
            dst_b = tstage.iterator + (w + 3) * 8192
            src_off = Int32((Int32(bidx) * 3 + w) * 65536 % (cute.size(mTraffic) - 16384))
            pa, pb = Int32(0), Int32(0)
            cute.arch.mbarrier_arrive_and_expect_tx(mb_a, 32768)
            cute.arch.inline_ptx(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                "[{$r0}], [{$r1}], {$r2}, [{$r3}];",
                read_only_args=[
                    dst_a.toint(),
                    (mTraffic.iterator + src_off).llvm_ptr,
                    Int32(32768),
                    mb_a.toint(),
                ],
            )
            for _p in cutlass.range(8192, unroll=1):
                # issue into B while A is in flight, then wait A; swap.
                cute.arch.mbarrier_arrive_and_expect_tx(mb_b, 32768)
                cute.arch.inline_ptx(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                    "[{$r0}], [{$r1}], {$r2}, [{$r3}];",
                    read_only_args=[
                        dst_b.toint(),
                        (mTraffic.iterator + src_off).llvm_ptr,
                        Int32(32768),
                        mb_b.toint(),
                    ],
                )
                cute.arch.mbarrier_wait(mb_a, pa)
                pa = pa ^ 1
                src_off = src_off + 32768
                if src_off >= cute.size(mTraffic) - 16384:
                    src_off = Int32(0)
                # swap roles (unrolled 2x to keep names simple)
                cute.arch.mbarrier_arrive_and_expect_tx(mb_a, 32768)
                cute.arch.inline_ptx(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                    "[{$r0}], [{$r1}], {$r2}, [{$r3}];",
                    read_only_args=[
                        dst_a.toint(),
                        (mTraffic.iterator + src_off).llvm_ptr,
                        Int32(32768),
                        mb_a.toint(),
                    ],
                )
                cute.arch.mbarrier_wait(mb_b, pb)
                pb = pb ^ 1
                src_off = src_off + 32768
                if src_off >= cute.size(mTraffic) - 16384:
                    src_off = Int32(0)
            cute.arch.mbarrier_wait(mb_a, pa)  # drain final in-flight

        elif const_expr(self.traffic) and tidx >= 32:
            # Traffic warps (both CTAs): stream the big gmem buffer for a fixed
            # pass count (~outlasts the steal phase; smem-flag polling gets
            # loop-hoisted). Accumulate defeats DCE.
            n = Int32(cute.size(mTraffic))
            acc = Int32(0)
            i = Int32((Int32(bidx) * 97 + tidx) * 1031 % n)
            for _p in cutlass.range(2048, unroll=1):
                for _ in cutlass.range_constexpr(64):
                    acc = acc + mTraffic[i]
                    i = i + 4099
                    if i >= n:
                        i = i - n
            if acc == Int32(0x7FFFFFFF):  # never true; keeps acc live
                mTraffic[0] = acc

        elif tidx == 0:
            if const_expr(self.multicast):
                # Non-leader CTA mirrors the decode loop so multicast responses
                # always land in a live, waiting CTA.
                phase1 = Int32(0)
                going1 = Boolean(True)
                while going1:
                    cute.arch.mbarrier_wait(mbar.iterator, phase1)
                    phase1 = phase1 ^ 1
                    _, _, _, v1 = cute.arch.clc_response(resp.iterator)
                    cute.arch.fence_view_async_shared()
                    if v1 == 0:
                        going1 = Boolean(False)
        if const_expr(self.multicast):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        if const_expr(self.tmem):
            if tidx == 0:
                cute.arch.barrier()  # not needed for 32-thread CTAs; harmless
                tptr = cute.arch.retrieve_tmem_ptr(
                    Int32, alignment=16, ptr_to_buffer_holding_addr=stage.iterator
                )
                cute.arch.dealloc_tmem(tptr, 512, is_two_cta=False)

    @cute.jit
    def __call__(
        self,
        mCounters: cute.Tensor,
        mHist: cute.Tensor,
        mTraffic: cute.Tensor,
        grid_clusters: Int32,
        margin: Int32,
        slot: Int32,
        over_spray: Int32,
        stream,
    ):
        self.kernel(mCounters, mHist, mTraffic, grid_clusters, margin, slot, over_spray).launch(
            grid=(grid_clusters * CLUSTER_M, 1, 1),
            block=(128 if const_expr(self.traffic or self.tma_traffic) else 32, 1, 1),
            cluster=(CLUSTER_M, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )


def main_xgrid(args):
    """K back-to-back PDL launches of the probe on one stream. Each launch L
    uses counter row L. Launch L's clusters, at retirement (own pool empty),
    issue `over_spray` more observed cancels; with PDL overlap those cancels
    race launch L+1's pending pool. Cross-grid detection is pure bookkeeping:
      victim launch:  started + granted < G   (its clusters vanished)
      sprayer launch: started + granted > G   (excess grants on its row)

    The retry-proven counter is reported separately. It proves same-grid
    spurious invalids, but it is not evidence of cross-grid grants.
    """
    dev = "cuda"
    G, K = args.grid, args.k_launches
    counters = torch.zeros((K, 4), device=dev, dtype=torch.int32)
    hist = torch.zeros(HIST_BINS, device=dev, dtype=torch.int32)
    mCounters, counters_gpu = cutlass_torch.cute_tensor_like(
        counters.flatten(), cutlass.Int32, True, 16
    )
    mHist, hist_gpu = cutlass_torch.cute_tensor_like(hist, cutlass.Int32, True, 16)
    stream = cutlass_torch.default_stream()
    probe = SpuriousInvalidProbe(spin_ns=args.spin_ns, smem_pad_kb=args.smem_pad_kb, use_pdl=True)
    traffic_buf = torch.zeros(1024, device=dev, dtype=torch.int32)
    mTraffic, _tg = cutlass_torch.cute_tensor_like(traffic_buf, cutlass.Int32, True, 16)
    compiled = cute.compile(
        probe, mCounters, mHist, mTraffic, Int32(G), Int32(args.margin), Int32(0), Int32(0), stream
    )
    cont = None
    if args.contender:
        ca = torch.randn(6144, 6144, device=dev, dtype=torch.bfloat16)
        cb = torch.randn(6144, 6144, device=dev, dtype=torch.bfloat16)
        cont = torch.cuda.Stream()

    closure_trials, retry_trials, tot_missing, tot_excess, tot_retry = 0, 0, 0, 0, 0
    closure_example = None
    retry_example = None
    for t in range(args.trials):
        counters_gpu.zero_()
        hist_gpu.zero_()
        if cont is not None:
            with torch.cuda.stream(cont):
                for _ in range(6):
                    ca @ cb
        GLs = [G if L % 2 == 0 else max(64, G // 4) for L in range(K)]
        for L in range(K):
            compiled(
                mCounters,
                mHist,
                mTraffic,
                Int32(GLs[L]),
                Int32(args.margin),
                Int32(L),
                Int32(args.over_spray),
                stream,
            )
        torch.cuda.synchronize()
        c = counters_gpu.view(K, 4).tolist()
        deficits = [GLs[L] - (c[L][0] + c[L][1]) for L in range(K)]
        retry_proven = sum(row[2] for row in c)
        missing = sum(d for d in deficits if d > 0)
        excess = sum(-d for d in deficits if d < 0)
        if missing or excess:
            closure_trials += 1
            tot_missing += missing
            tot_excess += excess
            if closure_example is None:
                closure_example = (t, [(row[0], row[1]) for row in c])
        if retry_proven:
            retry_trials += 1
            tot_retry += retry_proven
            if retry_example is None:
                retry_example = (t, retry_proven)
        if (t + 1) % 50 == 0:
            print(
                f"[{t + 1}/{args.trials}] closure_trials={closure_trials} "
                f"missing={tot_missing} excess={tot_excess} "
                f"retry_trials={retry_trials} retry_proven={tot_retry}",
                flush=True,
            )
    print(
        f"\nRESULT xgrid: grid={G} K={K} over_spray={args.over_spray} "
        f"contender={args.contender}: {closure_trials}/{args.trials} trials with "
        f"cross-grid closure breaks; {tot_missing} clusters vanished from victim "
        f"grids, {tot_excess} excess grants on sprayer grids; "
        f"{retry_trials}/{args.trials} trials with {tot_retry} retry-proven "
        f"same-grid spurious invalids"
    )
    if closure_example:
        t, rows = closure_example
        print(f"first closure break at trial {t}: (started, granted) per launch = {rows}, G={G}")
    if retry_example:
        t, n = retry_example
        print(f"first retry-proven same-grid invalid at trial {t}: {n} events")
    print(
        "PROOF: try_cancel crosses grid boundaries under PDL"
        if tot_missing or tot_excess
        else "no cross-grid grants observed"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--grid", type=int, default=1344, help="total clusters G (prod-like)")
    ap.add_argument("--spin-ns", type=int, default=10000)
    ap.add_argument("--smem-pad-kb", type=int, default=100)
    ap.add_argument(
        "--margin",
        type=int,
        default=150,
        help="pending_est above this = spurious (must exceed max in-flight launches "
        "~= resident clusters; ~74 resident at 100KB smem pad on B300)",
    )
    ap.add_argument("--contender", action="store_true", help="run co-tenant matmul in-process")
    ap.add_argument(
        "--pdl-attr", action="store_true", help="launch grid with PDL attribute (production does)"
    )
    ap.add_argument(
        "--multicast", action="store_true", help="use multicast steal queries (production does)"
    )
    ap.add_argument(
        "--traffic",
        action="store_true",
        help="3 companion warps per CTA hammer gmem during the steal loop",
    )
    ap.add_argument(
        "--tma-traffic",
        action="store_true",
        help="3 companion warps per CTA run continuous cp.async.bulk with tx barriers",
    )
    ap.add_argument(
        "--stream-depth",
        type=int,
        default=0,
        help="launch this many probe grids back-to-back per sync (production "
        "launches hundreds of CLC grids deep); retry-proven counter accumulates",
    )
    ap.add_argument(
        "--tmem",
        action="store_true",
        help="hold a full 512-col TMEM allocation for each CTA's lifetime",
    )
    ap.add_argument(
        "--pipelined",
        action="store_true",
        help="keep 2 try_cancels in flight per cluster (production sched-pipeline shape)",
    )
    ap.add_argument(
        "--queued-successors",
        type=int,
        default=0,
        help="enqueue N torch kernels behind the probe before syncing "
        "(production always has grids queued behind the CLC grid)",
    )
    ap.add_argument(
        "--xgrid",
        action="store_true",
        help="cross-grid mode: K back-to-back PDL launches; retiring clusters "
        "over-spray observed cancels. A grant then can only come from the NEXT "
        "grid's pending pool -> victim grid closure breaks (started+granted < G).",
    )
    ap.add_argument("--k-launches", type=int, default=4)
    ap.add_argument("--over-spray", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda"
    G = args.grid
    if args.xgrid:
        return main_xgrid(args)

    counters = torch.zeros(4, device=dev, dtype=torch.int32)
    hist = torch.zeros(HIST_BINS, device=dev, dtype=torch.int32)
    mCounters, counters_gpu = cutlass_torch.cute_tensor_like(counters, cutlass.Int32, True, 16)
    mHist, hist_gpu = cutlass_torch.cute_tensor_like(hist, cutlass.Int32, True, 16)
    stream = cutlass_torch.default_stream()
    probe = SpuriousInvalidProbe(
        spin_ns=args.spin_ns,
        smem_pad_kb=args.smem_pad_kb,
        use_pdl=args.pdl_attr,
        multicast=args.multicast,
        traffic=args.traffic,
        tma_traffic=args.tma_traffic,
        pipelined=args.pipelined,
        tmem=args.tmem,
    )
    traffic_buf = torch.zeros(64 * 1024 * 1024, device=dev, dtype=torch.int32)  # 256 MB
    mTraffic, _tg = cutlass_torch.cute_tensor_like(traffic_buf, cutlass.Int32, True, 16)
    compiled = cute.compile(
        probe, mCounters, mHist, mTraffic, Int32(G), Int32(args.margin), Int32(0), Int32(0), stream
    )

    cont = None
    if args.contender:
        ca = torch.randn(6144, 6144, device=dev, dtype=torch.bfloat16)
        cb = torch.randn(6144, 6144, device=dev, dtype=torch.bfloat16)
        cont = torch.cuda.Stream()

    if args.stream_depth:
        # STREAMING MODE: back-to-back launches, sync per batch. Closure can't
        # be checked per launch; the retry-proven counter accumulates.
        total_events = 0
        batches = max(1, args.trials // args.stream_depth)
        for b in range(batches):
            if cont is not None:
                with torch.cuda.stream(cont):
                    for _ in range(6):
                        ca @ cb
            for _ in range(args.stream_depth):
                compiled(
                    mCounters,
                    mHist,
                    mTraffic,
                    Int32(G),
                    Int32(args.margin),
                    Int32(0),
                    Int32(0),
                    stream,
                )
            torch.cuda.synchronize()
            ev = int(counters_gpu[2])
            if ev != total_events:
                print(f"[batch {b}] retry-proven events now {ev}", flush=True)
                total_events = ev
        print(
            f"\nRESULT streaming grid={G} depth={args.stream_depth} "
            f"launches={batches * args.stream_depth}: {total_events} RETRY-PROVEN "
            f"spurious invalids"
        )
        print("PROOF: spurious invalid" if total_events else "no spurious invalids observed")
        return

    tot_hist = torch.zeros(HIST_BINS, dtype=torch.long)
    spurious_trials, tot_spurious, max_pending, bad_closure = 0, 0, 0, 0
    for t in range(args.trials):
        counters_gpu.zero_()
        hist_gpu.zero_()
        if cont is not None:
            with torch.cuda.stream(cont):
                for _ in range(6):
                    ca @ cb
        compiled(
            mCounters, mHist, mTraffic, Int32(G), Int32(args.margin), Int32(0), Int32(0), stream
        )
        if args.queued_successors:
            # CWD sees successor grids queued behind the running CLC grid,
            # like a real workload (pytest enqueues refs right behind the gemm)
            for _ in range(args.queued_successors):
                traffic_buf[:4096].mul_(1)
        torch.cuda.synchronize()
        c = counters_gpu.tolist()
        tot_hist += hist_gpu.cpu().long()
        if c[0] + c[1] != G:  # closure: launched + granted == G
            bad_closure += 1
        if c[2] > 0:
            spurious_trials += 1
            tot_spurious += c[2]
        max_pending = max(max_pending, c[3])
        if (t + 1) % 50 == 0:
            print(
                f"[{t + 1}/{args.trials}] spurious_trials={spurious_trials} "
                f"events={tot_spurious} max_pending={max_pending}",
                flush=True,
            )

    labels = ["0"] + [f"{2 ** (b - 1)}-{2**b - 1}" for b in range(1, HIST_BINS)]
    print("\npending_est at INVALID (all retirements, all trials):")
    for lbl, n in zip(labels, tot_hist.tolist()):
        if n:
            print(f"  {lbl:>12}: {n}")
    print(
        f"\nRESULT grid={G} margin={args.margin} pdl={args.pdl_attr} mcast={args.multicast} "
        f"contender={args.contender}: "
        f"{spurious_trials}/{args.trials} trials with RETRY-PROVEN spurious invalids, "
        f"{tot_spurious} events, max pending at invalid = {max_pending}, "
        f"closure_bad={bad_closure}"
    )
    print(
        "PROOF: try_cancel returns invalid while pool is non-empty"
        if tot_spurious
        else "no spurious invalids observed"
    )


if __name__ == "__main__":
    main()
