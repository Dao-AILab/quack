"""CLC grant-monotonicity validation harness.

The varlen CLC drain fix relies on this assumption: after a valid try_cancel
grant decodes into the phantom/padding region, later grants from the same grid
are not for earlier real work indices. This probe observes the old
cancel_pending_tail protocol and reports any counterexample. The July 2026
campaign found none (576M observed cancels); a violation here would mean a real
pending cluster was canceled after the phantom signal.

Background (July 2026 varlen CI corruption). The varlen CLC scheduler's
retirement drain (afe2ef3, removed) relied on: "once my own steal decodes into
the phantom/padding region, every cluster still pending maps to padding, so
blind cancels are safe." This kernel replicates the production protocol with
the cancel responses OBSERVED instead of fire-and-forget:

  * Grid of G clusters (cluster shape (2,1,1) like production), first B
    ("real") / rest ("phantom") — a labeling only; the hardware doesn't know.
    A large smem pad limits residency so a deep pending pool exists.
  * Each launched cluster's CTA-rank-0 warp runs the production scheduler-warp
    protocol: a 2-stage pipelined MULTICAST steal loop (a query for stage s+1
    is in flight while stage s is decoded). Real grants mark processed[id]
    (production runs the tile); the first phantom grant exits the loop
    (production decodes an invalid tile).
  * At exit the in-flight query is drained and decoded (channel 2: production
    silently drops this grant — a real id here is ALSO a killed tile).
  * Then it sprays K non-multicast cancels — production's cancel_pending_tail
    — decoding every response (channel 1). Any grant with id < B would be a
    REAL cluster canceled after the phantom signal: work production would
    silently never compute.

Bookkeeping closure (every id launched xor granted exactly once) validates
the harness itself.

Run (in the CI container, ideally with a co-tenant process on the same GPU):
  python AI/repro_clc_grant_monotonicity.py --trials 3000
"""

import argparse

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass import Int32, Boolean, const_expr

CLUSTER_M = 2  # production varlen default config cluster shape
STAGES = 2  # in-flight multicast queries, like production sched_stage
VLOG_CAP = 256


class ClcMonotonicityProbe:
    def __init__(
        self,
        spin_ns: int,
        smem_pad_kb: int,
        multicast: bool,
        faithful: bool = False,
        exit_delay_ns: int = 0,
        burst: bool = False,
    ):
        self.spin_ns = spin_ns
        self.smem_pad_kb = smem_pad_kb
        self.multicast = multicast
        # exit_delay_ns emulates production CTA teardown time (epilogue drain,
        # tmem dealloc): the window in which in-flight CLC responses land
        # before the cluster exits.
        self.exit_delay_ns = exit_delay_ns
        # burst=True sprays production-style: arm expect_tx(16*K) once, K
        # back-to-back issues, ONE wait at the end (responses overwrite one
        # slot; only closure accounting can detect lost real clusters).
        # burst=False (default) is serial: issue+wait+decode each.
        self.burst = burst
        # faithful=True reproduces production exactly: fire-and-forget spray, no
        # in-flight drain, CTAs exit immediately — pair with victim_kernel to
        # detect CLC responses landing in the NEXT kernel's smem (channel 3).
        self.faithful = faithful

    @cute.kernel
    def kernel(
        self,
        mExecuted: cute.Tensor,  # (G,)
        mProcessed: cute.Tensor,  # (G,)
        mSprayed: cute.Tensor,  # (G,) granted to spray or dropped in-flight
        mViol: cute.Tensor,  # (4,): [0] spray-real, [1] pool-empty sprays,
        #                          [2] inflight-dropped-real, [3] unused
        mVlog: cute.Tensor,  # (2*VLOG_CAP,) (sprayer, victim) pairs
        real_boundary: Int32,
        spray_budget: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        if const_expr(self.smem_pad_kb > 0):  # residency limiter (GEMM-like footprint)
            smem.allocate_tensor(Int32, cute.make_layout(self.smem_pad_kb * 256), byte_alignment=16)
        # STAGES response slots + barriers (same static smem offsets in every
        # CTA of the cluster, which multicast delivery relies on).
        resp = smem.allocate_tensor(Int32, cute.make_layout((4, STAGES)), byte_alignment=16)
        mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(STAGES), byte_alignment=8)

        cta_rank = cute.arch.block_idx_in_cluster()
        my_id = Int32(bidx) // CLUSTER_M

        if tidx == 0:
            for s in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(mbar.iterator + s, 1)
            cute.arch.mbarrier_init_fence()
        # All CTAs must see initialized barriers before any multicast query.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        if cta_rank == 0 and tidx == 0:
            mExecuted[my_id] = 1
            p0, p1 = Int32(0), Int32(0)

            # arm + issue into both stages up front. Arm every CTA's stage
            # barrier (16B tx each) for multicast — the production
            # _issue_clc_query_multicast; non-multicast arms only locally.
            # (Inlined at each site: the DSL forbids closure capture in loops.)
            for s in cutlass.range_constexpr(STAGES):
                if const_expr(self.multicast):
                    for r in cutlass.range_constexpr(CLUSTER_M):
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar.iterator + s, 16, r)
                else:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar.iterator + s, 16)
                cute.arch.issue_clc_query(
                    mbar.iterator + s, resp[None, s].iterator, multicast=self.multicast
                )
            cur = Int32(0)
            phantom_seen = Boolean(False)
            going = Boolean(True)
            # --- main steal loop (production scheduler warp + consumer) ---
            while going:
                if cur == 0:
                    cute.arch.mbarrier_wait(mbar.iterator, p0)
                    p0 = p0 ^ 1
                else:
                    cute.arch.mbarrier_wait(mbar.iterator + 1, p1)
                    p1 = p1 ^ 1
                bx, _, _, valid = cute.arch.clc_response(resp[None, cur].iterator)
                cute.arch.fence_view_async_shared()
                if valid != 0:
                    gid = Int32(bx) // CLUSTER_M
                    mProcessed[gid] = 1  # production: execute this tile
                    if gid >= real_boundary:
                        phantom_seen = Boolean(True)  # production: invalid tile
                        going = Boolean(False)
                    else:
                        if const_expr(self.spin_ns > 0):  # emulate tile work
                            cute.arch.inline_ptx(
                                "nanosleep.u32 {$r0};", read_only_args=[Int32(self.spin_ns)]
                            )
                        # refill the stage we just consumed
                        for s in cutlass.range_constexpr(STAGES):
                            if cur == s:
                                if const_expr(self.multicast):
                                    for r in cutlass.range_constexpr(CLUSTER_M):
                                        cute.arch.mbarrier_arrive_and_expect_tx(
                                            mbar.iterator + s, 16, r
                                        )
                                else:
                                    cute.arch.mbarrier_arrive_and_expect_tx(mbar.iterator + s, 16)
                                cute.arch.issue_clc_query(
                                    mbar.iterator + s,
                                    resp[None, s].iterator,
                                    multicast=self.multicast,
                                )
                        cur = cur ^ 1
                else:
                    going = Boolean(False)  # pool empty

            # --- faithful mode: production's fire-and-forget spray, no drain ---
            if const_expr(self.faithful):
                if phantom_seen:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar.iterator, 16 * spray_budget)
                    k = Int32(0)
                    while k < spray_budget:
                        cute.arch.issue_clc_query(mbar.iterator, resp.iterator, multicast=False)
                        k = k + 1
                # exit with everything in flight, exactly like production

            # --- drain the in-flight stage (production silently DROPS this grant) ---
            if const_expr(not self.faithful):
                other = cur ^ 1
                if other == 0:
                    cute.arch.mbarrier_wait(mbar.iterator, p0)
                    p0 = p0 ^ 1
                else:
                    cute.arch.mbarrier_wait(mbar.iterator + 1, p1)
                    p1 = p1 ^ 1
                bx2, _, _, valid2 = cute.arch.clc_response(resp[None, other].iterator)
                cute.arch.fence_view_async_shared()
                if valid2 != 0:
                    gid2 = Int32(bx2) // CLUSTER_M
                    mSprayed[gid2] = 1
                    if gid2 < real_boundary:
                        # channel 2: real tile canceled by the dropped in-flight query
                        cute.arch.atomic_add(mViol.iterator + 2, Int32(1))

            # --- burst spray (production form): all issues back-to-back, one
            # wait for the sum of transactions, no per-response decode ---
            if const_expr(not self.faithful and self.burst):
                if phantom_seen:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar.iterator, 16 * spray_budget)
                    kb = Int32(0)
                    while kb < spray_budget:
                        cute.arch.issue_clc_query(mbar.iterator, resp.iterator, multicast=False)
                        kb = kb + 1
                    cute.arch.mbarrier_wait(mbar.iterator, p0)
                    p0 = p0 ^ 1

            # --- the retirement spray (production cancel_pending_tail), observed ---
            if const_expr(not self.faithful and not self.burst) and phantom_seen:
                k = Int32(0)
                while k < spray_budget:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar.iterator, 16)
                    cute.arch.issue_clc_query(mbar.iterator, resp.iterator, multicast=False)
                    cute.arch.mbarrier_wait(mbar.iterator, p0)
                    p0 = p0 ^ 1
                    bx, _, _, valid = cute.arch.clc_response(resp.iterator)
                    cute.arch.fence_view_async_shared()
                    if valid != 0:
                        gid = Int32(bx) // CLUSTER_M
                        mSprayed[gid] = 1
                        if gid < real_boundary:
                            # channel 1 VIOLATION: real tile canceled by the spray
                            slot = cute.arch.atomic_add(mViol.iterator, Int32(1))
                            if slot < VLOG_CAP:
                                mVlog[2 * slot] = my_id
                                mVlog[2 * slot + 1] = gid
                        k = k + 1
                    else:
                        cute.arch.atomic_add(mViol.iterator + 1, Int32(1))
                        k = spray_budget

        elif tidx == 0:
            # Non-leader CTA: mirror the consumer decode loop so multicast
            # responses always land in a live CTA (production consumers do the
            # same; being stricter than production here keeps the experiment
            # about grant CONTENT, not orphaned writes).
            if const_expr(self.multicast):
                p0, p1 = Int32(0), Int32(0)
                cur = Int32(0)
                going = Boolean(True)
                while going:
                    if cur == 0:
                        cute.arch.mbarrier_wait(mbar.iterator, p0)
                        p0 = p0 ^ 1
                    else:
                        cute.arch.mbarrier_wait(mbar.iterator + 1, p1)
                        p1 = p1 ^ 1
                    bx, _, _, valid = cute.arch.clc_response(resp[None, cur].iterator)
                    cute.arch.fence_view_async_shared()
                    if valid != 0:
                        gid = Int32(bx) // CLUSTER_M
                        if gid >= real_boundary:
                            going = Boolean(False)
                        else:
                            cur = cur ^ 1
                    else:
                        going = Boolean(False)
                if const_expr(not self.faithful):
                    # drain this CTA's in-flight multicast response before exiting
                    other = cur ^ 1
                    if other == 0:
                        cute.arch.mbarrier_wait(mbar.iterator, p0)
                    else:
                        cute.arch.mbarrier_wait(mbar.iterator + 1, p1)

        if const_expr(not self.faithful):
            # Cluster exits as a unit only after CTA0's spray fully drained.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        # Post-sync dwell = production CTA lifetime after the scheduler warp
        # exits (epilogue warps still storing D). The window in which spray
        # responses normally land before the CTA retires. nanosleep caps at
        # ~1ms, so loop 1ms chunks (constexpr count).
        if const_expr(self.exit_delay_ns > 0):
            if tidx == 0:
                for _ in cutlass.range_constexpr(max(1, self.exit_delay_ns // 1_000_000)):
                    cute.arch.inline_ptx(
                        "nanosleep.u32 {$r0};",
                        read_only_args=[Int32(min(self.exit_delay_ns, 1_000_000))],
                    )

    @cute.kernel
    def victim_kernel(self, mClob: cute.Tensor):
        """Launched right after a faithful-mode sprayer grid: fill the smem
        words where the sprayer kept its CLC response slots + barriers with a
        sentinel, dwell, and re-check. Any mismatch = a late async write from
        the PREVIOUS grid landed in this kernel's smem (channel 3: the
        production corruption mechanism candidate). Same allocation sequence
        as the sprayer kernel => same static smem offsets."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        if const_expr(self.smem_pad_kb > 0):
            smem.allocate_tensor(Int32, cute.make_layout(self.smem_pad_kb * 256), byte_alignment=16)
        resp = smem.allocate_tensor(Int32, cute.make_layout((4, STAGES)), byte_alignment=16)
        mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(STAGES), byte_alignment=8)
        mbar_i32 = cute.make_tensor(
            cute.recast_ptr(mbar.iterator, dtype=Int32), cute.make_layout(2 * STAGES)
        )
        SENT = Int32(0x5A5A5A5A)
        if tidx == 0:
            for i in cutlass.range_constexpr(4 * STAGES):
                resp[i % 4, i // 4] = SENT
            for i in cutlass.range_constexpr(2 * STAGES):
                mbar_i32[i] = SENT
            cute.arch.fence_view_async_shared()
            # dwell so a late in-flight write from the previous grid can land
            for _ in cutlass.range(20, unroll=1):
                cute.arch.inline_ptx("nanosleep.u32 {$r0};", read_only_args=[Int32(10000)])
            cute.arch.fence_view_async_shared()
            nbad = Int32(0)
            first = Int32(0)
            for i in cutlass.range_constexpr(4 * STAGES):
                v = resp[i % 4, i // 4]
                if v != SENT:
                    nbad = nbad + 1
                    if nbad == 1:
                        first = v
            for i in cutlass.range_constexpr(2 * STAGES):
                v = mbar_i32[i]
                if v != SENT:
                    nbad = nbad + 1
                    if nbad == 1:
                        first = v
            if nbad > 0:
                slot = cute.arch.atomic_add(mClob.iterator, Int32(1))
                if slot < VLOG_CAP // 3:
                    mClob[1 + 3 * slot] = Int32(bidx)
                    mClob[2 + 3 * slot] = nbad
                    mClob[3 + 3 * slot] = first

    @cute.jit
    def launch_victim(self, mClob: cute.Tensor, grid_ctas: Int32, stream):
        self.victim_kernel(mClob).launch(grid=(grid_ctas, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.jit
    def __call__(
        self,
        mExecuted: cute.Tensor,
        mProcessed: cute.Tensor,
        mSprayed: cute.Tensor,
        mViol: cute.Tensor,
        mVlog: cute.Tensor,
        real_boundary: Int32,
        spray_budget: Int32,
        grid_clusters: Int32,
        stream,
    ):
        self.kernel(
            mExecuted, mProcessed, mSprayed, mViol, mVlog, real_boundary, spray_budget
        ).launch(
            grid=(grid_clusters * CLUSTER_M, 1, 1),
            block=(32, 1, 1),
            cluster=(CLUSTER_M, 1, 1),
            stream=stream,
        )


def main_late_write(args):
    """Channel 3: launch the production-faithful sprayer, then immediately a
    sentinel victim kernel; any sentinel clobber = an async CLC write from the
    previous grid landed in the next kernel's smem."""
    G, B, K = args.grid, int(args.grid * args.real_frac), args.spray
    dev = "cuda"
    bufs = {
        name: torch.zeros(shape, device=dev, dtype=torch.int32)
        for name, shape in [
            ("executed", G),
            ("processed", G),
            ("sprayed", G),
            ("viol", 4),
            ("vlog", 2 * VLOG_CAP),
            ("clob", 1 + VLOG_CAP),
        ]
    }
    cute_bufs, gpu_bufs = {}, {}
    for name, t in bufs.items():
        cute_bufs[name], gpu_bufs[name] = cutlass_torch.cute_tensor_like(t, cutlass.Int32, True, 16)
    stream = cutlass_torch.default_stream()
    probe = ClcMonotonicityProbe(
        spin_ns=args.spin_ns,
        smem_pad_kb=args.smem_pad_kb,
        multicast=True,
        faithful=True,
        exit_delay_ns=args.exit_delay_ns,
    )
    order = ("executed", "processed", "sprayed", "viol", "vlog")
    compiled_a = cute.compile(
        probe, *(cute_bufs[n] for n in order), Int32(B), Int32(K), Int32(G), stream
    )
    compiled_b = cute.compile(probe.launch_victim, cute_bufs["clob"], Int32(G * CLUSTER_M), stream)
    clob_trials, tot_clob = 0, 0
    example = None
    for t in range(args.trials):
        for g in gpu_bufs.values():
            g.zero_()
        compiled_a(*(cute_bufs[n] for n in order), Int32(B), Int32(K), Int32(G), stream)
        compiled_b(cute_bufs["clob"], Int32(G * CLUSTER_M), stream)
        torch.cuda.synchronize()
        n = int(gpu_bufs["clob"][0])
        if n:
            clob_trials += 1
            tot_clob += n
            if example is None:
                c = gpu_bufs["clob"][1:10].tolist()
                example = (
                    t,
                    n,
                    [
                        (c[3 * i], c[3 * i + 1], hex(c[3 * i + 2] & 0xFFFFFFFF))
                        for i in range(min(n, 3))
                    ],
                )
        if (t + 1) % 250 == 0:
            print(
                f"[{t + 1}/{args.trials}] clob_trials={clob_trials} total_ctas_clobbered={tot_clob}",
                flush=True,
            )
    print(
        f"\nRESULT(late-write) grid={G} spray={K}: {clob_trials}/{args.trials} trials had "
        f"sentinel clobbers in the NEXT kernel's smem; {tot_clob} CTAs affected total"
    )
    if example:
        t, n, ex = example
        print(f"first at trial {t}: {n} CTAs; (cta, words, first_word) = {ex}")
    print("PROOF: late cross-kernel smem writes" if tot_clob else "no late writes observed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--grid", type=int, default=1344, help="total clusters G (prod-like)")
    ap.add_argument("--real-frac", type=float, default=0.95, help="prod varlen padding ~5%%")
    ap.add_argument("--spray", type=int, default=64)
    ap.add_argument("--spin-ns", type=int, default=10000, help="per-'tile' work emulation")
    ap.add_argument("--smem-pad-kb", type=int, default=100)
    ap.add_argument("--no-multicast", action="store_true", help="simple non-multicast loop")
    ap.add_argument("--late-write", action="store_true", help="channel-3 experiment")
    ap.add_argument("--exit-delay-ns", type=int, default=0, help="CTA teardown emulation")
    ap.add_argument("--burst", action="store_true", help="production-style burst spray")
    args = ap.parse_args()
    if args.late_write:
        return main_late_write(args)

    G, B, K = args.grid, int(args.grid * args.real_frac), args.spray
    dev = "cuda"

    bufs = {
        name: torch.zeros(shape, device=dev, dtype=torch.int32)
        for name, shape in [
            ("executed", G),
            ("processed", G),
            ("sprayed", G),
            ("viol", 4),
            ("vlog", 2 * VLOG_CAP),
        ]
    }
    cute_bufs, gpu_bufs = {}, {}
    for name, t in bufs.items():
        cute_bufs[name], gpu_bufs[name] = cutlass_torch.cute_tensor_like(t, cutlass.Int32, True, 16)
    stream = cutlass_torch.default_stream()
    probe = ClcMonotonicityProbe(
        spin_ns=args.spin_ns,
        smem_pad_kb=args.smem_pad_kb,
        multicast=not args.no_multicast,
        burst=args.burst,
    )
    order = ("executed", "processed", "sprayed", "viol", "vlog")
    compiled = cute.compile(
        probe, *(cute_bufs[n] for n in order), Int32(B), Int32(K), Int32(G), stream
    )

    spray_viol, inflight_viol, viol_trials, closure_bad = 0, 0, 0, 0
    example = None
    for t in range(args.trials):
        for g in gpu_bufs.values():
            g.zero_()
        compiled(*(cute_bufs[n] for n in order), Int32(B), Int32(K), Int32(G), stream)
        torch.cuda.synchronize()
        ex, pr, sp = (gpu_bufs[n] for n in ("executed", "processed", "sprayed"))
        v = gpu_bufs["viol"].tolist()
        covered = ex + pr + sp
        if not bool((covered >= 1).all()):
            closure_bad += 1
        if v[0] or v[2]:
            viol_trials += 1
            spray_viol += v[0]
            inflight_viol += v[2]
            if example is None and v[0]:
                vl = gpu_bufs["vlog"][: 2 * min(v[0], 4)].tolist()
                lost = int(((ex[:B] == 0) & (pr[:B] == 0)).sum())
                example = (t, v[0], lost, [(vl[2 * i], vl[2 * i + 1]) for i in range(min(v[0], 4))])
        if (t + 1) % 250 == 0:
            print(
                f"[{t + 1}/{args.trials}] viol_trials={viol_trials} "
                f"spray_real={spray_viol} inflight_real={inflight_viol}",
                flush=True,
            )

    print(
        f"\nRESULT grid={G} real={B} spray={K} spin={args.spin_ns}ns "
        f"multicast={not args.no_multicast}: "
        f"{viol_trials}/{args.trials} trials violated; "
        f"real tiles killed: {spray_viol} by spray, {inflight_viol} by dropped in-flight; "
        f"closure_bad={closure_bad}"
    )
    if example:
        t, nv, lost, pairs = example
        print(
            f"first spray violation at trial {t}: {nv} real grants after phantom "
            f"(lost real ids that trial: {lost}); (sprayer, victim): {pairs}"
        )
    print("PROOF: assumption broken" if (spray_viol or inflight_viol) else "no violation observed")


if __name__ == "__main__":
    main()
