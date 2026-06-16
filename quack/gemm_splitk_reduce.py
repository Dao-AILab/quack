# Copyright (c) 2026, QuACK contributors.
# Second kernel of parallel split-K GEMM: sums the per-split fp32 partial tiles written
# by the GEMM kernel (fixed ascending split order -> run-to-run deterministic) and
# applies the default epilogue (alpha, beta*C, rowvec/colvec bias) while converting to
# the output dtype. Mirrors cuBLAS's splitKreduce_kernel.

import math
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import make_ptr

import torch
from torch import Tensor

import quack.utils as utils
from quack.cache_utils import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.gemm_tvm_ffi_utils import div_for_dtype


class SplitKReduce:
    """One CTA per (output tile, chunk); each CTA sums its chunk across the tile's slots.

    Workspace layout: tile `tile_idx` owns `tile_count[tile_idx]` consecutive slots
    starting at `tile_first_slot[tile_idx]` (a host-built prefix sum); each slot holds a
    row-major (tile_m, tile_n) fp32 partial, where tile_idx =
    (tile_m_idx * ntile_n + tile_n_idx) * L + l. The fixed split-K path passes uniform
    tables (count = split_k, first_slot = tile_idx * split_k); Stream-K will pass a
    variable, data-dependent count per tile into the same kernel. Edge tiles are padded
    inside the slot; this kernel predicates the final D store by (M, N).
    """

    # The reduce was latency-bound, NOT bandwidth-bound: at one thread per output element
    # (vec_width=1) the total thread count is pinned at tile_mn, which for a single 128x128
    # tile is only 256 CTAs of 64 threads -> waves=0.07, warps_active~5%, ~14 warps/SM max.
    # With ~14 warps/SM and `count` serial HBM loads per thread, the GPU cannot hide the
    # HBM latency, so ncu shows DRAM at ~8% of peak and issue_active ~45%. Repartitioning a
    # FIXED thread count (num_threads / vec_width) never raises occupancy -- the only lever
    # that fills the GPU is adding parallelism over the K (slot/contributor) dimension.
    #
    # `kgroups` (R) is that lever: each output element's `count` contributors are split into
    # R contiguous groups, R threads in the same CTA each reduce one group (ascending), and
    # the R partials are combined in shared memory in ascending group order. This multiplies
    # the thread/warp count by R (one kernel, no extra HBM traffic for partials), lifting
    # warps_active from ~5% to ~50-70% and the reduce's measured ncu kernel time on a single
    # 128x128 sk=32 tile from ~13us to ~9us. The combine is a fixed blocked-ascending sum:
    # run-to-run deterministic, and within <=1 bf16 ULP of the flat ascending sum (a few
    # elements per tile, all within the bf16 test tolerance). kgroups is picked per call by
    # choose_reduce_kgroups() so few-tile launches fill the GPU while many-tile launches
    # (already full at R=1) stay on the exact flat-ascending path.
    num_threads = 256
    # Per-thread memory-level parallelism for the R=1 path: pre-stage up to `stage_slots`
    # slot loads into registers (all outstanding at once) before the ordered accumulation
    # consumes them. Only used when kgroups==1 (the many-tile / Stream-K path, which is
    # already GPU-full and keeps bit-exact flat-ascending order).
    stage_slots = 8

    def __init__(self, tile_m: int, tile_n: int, vec_width: int = 1, kgroups: int = 1):
        self.tile_m, self.tile_n, self.vec_width, self.kgroups = (
            tile_m,
            tile_n,
            vec_width,
            kgroups,
        )
        # The CTA covers num_threads/kgroups distinct output elements (each handled by
        # kgroups threads that each reduce one contiguous group of contributors).
        assert self.num_threads % kgroups == 0
        self.elems_per_cta = self.num_threads // kgroups
        # A vector never straddles a row of the slot, and chunks tile the slot exactly
        assert tile_n % self.vec_width == 0
        assert (tile_m * tile_n) % (self.elems_per_cta * self.vec_width) == 0

    @cute.jit
    def __call__(
        self,
        mWS: cute.Tensor,  # (total_contributors * tile_m * tile_n,) f32
        mD: cute.Tensor,  # (M, N, L)
        mC: Optional[cute.Tensor],  # (M, N, L)
        alpha: Optional[Float32 | cute.Pointer],
        beta: Optional[Float32 | cute.Pointer],
        mRowVec: Optional[cute.Tensor],  # (L, N)
        mColVec: Optional[cute.Tensor],  # (L, M)
        mTileFirstSlot: cute.Tensor,  # (num_tiles,) i32: first slot owned by each tile
        mTileCount: cute.Tensor,  # (num_tiles,) i32: contributors per tile
        stream: cuda.CUstream,
    ):
        ntile_m = cute.ceil_div(cute.size(mD, mode=[0]), self.tile_m)
        ntile_n = cute.ceil_div(cute.size(mD, mode=[1]), self.tile_n)
        # One CTA per (output tile, chunk): a tile's chunks reduce on separate CTAs
        # so the kernel fills the GPU instead of serializing on one CTA per tile. Each CTA
        # covers elems_per_cta = num_threads/kgroups elements (kgroups threads per element).
        # block_idx.z packs (l, chunk) -> grid.z = L * num_chunks.
        num_chunks = (self.tile_m * self.tile_n) // (self.elems_per_cta * self.vec_width)
        self.kernel(
            mWS, mD, mC, alpha, beta, mRowVec, mColVec, mTileFirstSlot, mTileCount, ntile_n
        ).launch(
            grid=[ntile_m, ntile_n, cute.size(mD, mode=[2]) * num_chunks],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mWS: cute.Tensor,
        mD: cute.Tensor,
        mC: Optional[cute.Tensor],
        alpha: Optional[Float32 | cute.Pointer],
        beta: Optional[Float32 | cute.Pointer],
        mRowVec: Optional[cute.Tensor],
        mColVec: Optional[cute.Tensor],
        mTileFirstSlot: cute.Tensor,
        mTileCount: cute.Tensor,
        ntile_n: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, bid_lc = cute.arch.block_idx()
        tile_m = const_expr(self.tile_m)
        tile_n = const_expr(self.tile_n)
        tile_mn = const_expr(tile_m * tile_n)
        V = const_expr(self.vec_width)
        R = const_expr(self.kgroups)
        Tw = const_expr(self.elems_per_cta)
        num_chunks = const_expr(tile_mn // (Tw * V))
        # block_idx.z packs (l, chunk): this CTA reduces one chunk of one output tile.
        bid_l = bid_lc // num_chunks
        chunk = bid_lc % num_chunks
        # Thread split: kgroups threads cooperate on the same output element. `e` is the
        # element-in-chunk (consecutive within a warp -> coalesced loads), `r` the k-group.
        # When R==1, Tw==num_threads so r==0 and e==tidx (the original one-thread-per-elem).
        r = tidx // Tw
        e = tidx % Tw

        len_m = cute.size(mD, mode=[0])
        len_n = cute.size(mD, mode=[1])
        num_l = cute.size(mD, mode=[2])
        tile_idx = (bid_m * ntile_n + bid_n) * num_l + bid_l
        # Variable per-tile contributor count + first-slot offset (a host-built prefix
        # sum) generalize the fixed `tile_idx * split_k` of the uniform split-K path, so
        # Stream-K (data-dependent contributors per tile) can reuse this reduce kernel.
        first_slot = mTileFirstSlot[tile_idx]
        count = mTileCount[tile_idx]
        slot_base = mWS.iterator + first_slot * tile_mn
        m0, n0 = bid_m * tile_m, bid_n * tile_n

        alpha_v, beta_v = Float32(1.0), Float32(1.0)
        if const_expr(alpha is not None):
            alpha_v = utils.load_scalar_or_pointer(alpha)
        if const_expr(beta is not None):
            beta_v = utils.load_scalar_or_pointer(beta)

        rAcc = cute.make_rmem_tensor(V, Float32)
        flat = (chunk * Tw + e) * V
        if const_expr(R == 1):
            # R==1 (many-tile / Stream-K) path: already GPU-full at one thread per element.
            # Sum the split slices in fixed flat ascending order (bit-exact, deterministic):
            # stage a window of slot loads into registers (all outstanding at once -> mlp),
            # then add them in ascending order. slot 0 seeds rAcc; slots [1, W] are
            # pre-staged; any remainder (count > W) falls to an ordered loop.
            W = const_expr(self.stage_slots)
            tWS = cute.make_tensor(slot_base + flat, cute.make_layout(V))
            cute.autovec_copy(tWS, rAcc)
            staged = cute.make_rmem_tensor(W * V, Float32)
            for s in cutlass.range_constexpr(1, W + 1):
                if s < count:
                    src = cute.make_tensor(slot_base + s * tile_mn + flat, cute.make_layout(V))
                    dst = cute.make_tensor(staged.iterator + (s - 1) * V, cute.make_layout(V))
                    cute.autovec_copy(src, dst)
            for s in cutlass.range_constexpr(1, W + 1):
                if s < count:
                    staged_s = cute.make_tensor(
                        staged.iterator + (s - 1) * V, cute.make_layout(V)
                    )
                    rAcc.store(rAcc.load() + staged_s.load())
            for s in cutlass.range(W + 1, count, unroll=8):
                tWS_s = cute.make_tensor(slot_base + s * tile_mn + flat, cute.make_layout(V))
                rAcc.store(rAcc.load() + tWS_s.load())
        else:
            # R>1 intra-CTA K-split: this thread reduces its contiguous group of contributors
            # [s0, s1) in ascending order; the R partials are combined below in shared memory
            # in ascending group order (a fixed blocked-ascending sum: run-to-run
            # deterministic, <=1 bf16 ULP from the flat sum). spg rounds up so groups tile
            # [0, count); the last group(s) may be empty (s0 >= count) when R does not divide
            # count, and contribute 0 -- matching the blocked reference.
            spg = (count + R - 1) // R
            s0 = r * spg
            s1 = cutlass.min(s0 + spg, count)
            for i in cutlass.range_constexpr(V):
                rAcc[i] = Float32(0.0)
            if s0 < s1:
                t0 = cute.make_tensor(slot_base + s0 * tile_mn + flat, cute.make_layout(V))
                cute.autovec_copy(t0, rAcc)
            for s in cutlass.range(s0 + 1, s1, unroll=8):
                tWS_s = cute.make_tensor(slot_base + s * tile_mn + flat, cute.make_layout(V))
                rAcc.store(rAcc.load() + tWS_s.load())
            # Combine R partials in smem: sBuf[r, e*V + i]. r==0 sums in ascending r order.
            smem = cutlass.utils.SmemAllocator()
            sBuf = smem.allocate_tensor(
                Float32, cute.make_layout((R, Tw * V)), byte_alignment=16
            )
            for i in cutlass.range_constexpr(V):
                sBuf[r, e * V + i] = rAcc[i]
            cute.arch.sync_threads()
            if r == 0:
                for i in cutlass.range_constexpr(V):
                    acc = sBuf[0, e * V + i]
                    for rr in cutlass.range_constexpr(1, R):
                        acc += sBuf[rr, e * V + i]
                    rAcc[i] = acc
        # The vector spans one row of the slot (tile_n % V == 0). Only the r==0 thread of
        # each element does the epilogue + store; for R==1 every thread is r==0.
        do_epi = const_expr(True) if const_expr(R == 1) else (r == 0)
        m = m0 + flat // tile_n
        n_base = n0 + flat % tile_n
        if do_epi and m < len_m:
            colvec_val = Float32(0.0)
            if const_expr(mColVec is not None):
                colvec_val = Float32(mColVec[bid_l, m])
            # Epilogue + predicated store, elementwise (D/C layout-agnostic).
            # Same op order as GemmDefaultEpiMixin.epi_visit_subtile:
            # alpha * acc, then (+ beta * C | + C), then rowvec/colvec bias.
            for i in cutlass.range_constexpr(V):
                n = n_base + i
                if n < len_n:
                    val = Float32(rAcc[i])
                    if const_expr(alpha is not None):
                        val = val * alpha_v
                    if const_expr(mC is not None):
                        c_val = Float32(mC[m, n, bid_l])
                        if const_expr(beta is not None):
                            val += beta_v * c_val
                        else:
                            val += c_val
                    if const_expr(mRowVec is not None):
                        val += Float32(mRowVec[bid_l, n])
                    if const_expr(mColVec is not None):
                        val += colvec_val
                    mD[m, n, bid_l] = mD.element_type(val)


@jit_cache
def _compile_splitk_reduce(
    d_dtype,
    c_dtype,
    d_major,
    c_major,
    tile_m,
    tile_n,
    vec_width,
    kgroups,
    alpha_mode,
    beta_mode,
    rowvec_dtype,
    colvec_dtype,
):
    m, n, l = cute.sym_int(), cute.sym_int(), cute.sym_int()

    def fake_scalar(mode):
        if mode == 0:
            return None
        elif mode == 1:
            return Float32(1.0)
        else:
            return make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=4)

    mWS = fake_tensor(Float32, (cute.sym_int(),), leading_dim=0, divisibility=4)
    mD = fake_tensor(
        d_dtype,
        (m, n, l),
        leading_dim=1 if d_major == "n" else 0,
        divisibility=div_for_dtype(d_dtype),
    )
    mC = (
        fake_tensor(
            c_dtype,
            (m, n, l),
            leading_dim=1 if c_major == "n" else 0,
            divisibility=div_for_dtype(c_dtype),
        )
        if c_dtype is not None
        else None
    )
    mRowVec = (
        fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4)
        if rowvec_dtype is not None
        else None
    )
    mColVec = (
        fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
        if colvec_dtype is not None
        else None
    )
    mTileFirstSlot = fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
    mTileCount = fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
    return cute.compile(
        SplitKReduce(tile_m, tile_n, vec_width, kgroups),
        mWS,
        mD,
        mC,
        fake_scalar(alpha_mode),
        fake_scalar(beta_mode),
        mRowVec,
        mColVec,
        mTileFirstSlot,
        mTileCount,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _scalar_modes(alpha, beta):
    alpha_mode = 2 if isinstance(alpha, Tensor) else (1 if alpha != 1.0 else 0)
    beta_mode = 2 if isinstance(beta, Tensor) else (1 if beta != 1.0 else 0)
    return alpha_mode, beta_mode


def compile_splitk_reduce(
    D, C, alpha, beta, rowvec, colvec, tile_m, tile_n, vec_width, kgroups=1
):
    """Compile (or fetch from cache) the reduce kernel for these tensor properties."""
    alpha_mode, beta_mode = _scalar_modes(alpha, beta)
    return _compile_splitk_reduce(
        torch2cute_dtype_map[D.dtype],
        torch2cute_dtype_map[C.dtype] if C is not None else None,
        "n" if D.stride(1) == 1 else "m",
        ("n" if C.stride(1) == 1 else "m") if C is not None else None,
        tile_m,
        tile_n,
        vec_width,
        kgroups,
        alpha_mode,
        beta_mode,
        torch2cute_dtype_map[rowvec.dtype] if rowvec is not None else None,
        torch2cute_dtype_map[colvec.dtype] if colvec is not None else None,
    )


def auto_split_k(m, n, k, l, tile_m, tile_n, num_sms, max_split_k: int = 40) -> int:
    """Heuristic split_k for parallel split-K. The GEMM launches `tiles * split_k` work units
    across up to `num_sms` persistent CTAs, so it runs in ceil(tiles*split_k/num_sms) waves,
    and a separate reduce then sums `split_k` fp32 partials per output element (its cost grows
    with split_k). split_k thus trades GEMM fill against reduce cost; the optimum (validated by
    a cold-L2 do_bench sweep on B200, where the reduce reads the workspace the GEMM just
    evicted) splits into two regimes:

    Multi-tile (tiles >= 2): the output tiles already supply parallelism, so split only far
    enough to round the GPU out to ~0.7 of a wave -- NOT a full wave. The old `ceil(num_sms/
    tiles)` overshot into a near-empty 2nd wave (512x512x16384, 16 tiles: ceil = 10 -> 160
    units = 1.08 waves, a tail the GPU eats in full -> 0.69x of cuBLAS), and even exactly one
    wave wastes reduce work past saturation (256x256x32768, 4 tiles: a full-wave split_k=32 is
    0.84x vs split_k=24 at 0.65 wave = 0.97x). `floor(0.70*num_sms/tiles)` lands on the
    measured optimum for both (512 -> 6, 256 -> 24).

    Single tile (tiles == 1): the split funnels through one tile's accumulators, so GEMM
    latency falls as ~k_tiles/split_k while reduce cost rises as ~split_k; their product is
    minimized near split_k ~ sqrt(k_tiles). `round(1.2*sqrt(k_tiles))` matches the sweep
    (128x128x16384, k_tiles=256 -> 19~=opt 20 -> 1.00x; 128x128x65536, k_tiles=1024 -> 38~=opt
    36 -> 0.95x), whereas filling the GPU (split_k>=64) is 0.82x (all reduce, no extra GEMM).

    Capped so each split keeps enough K to amortize the reduce (k_tiles // 8) and below
    `max_split_k` (past which reduce cost outgrows the extra GEMM fill). Returns 1 when the
    output tiles already fill the GPU (split-K won't help). Assumes a 64-deep K tile
    (bf16/fp16, the dtypes parallel split-K supports)."""
    tiles = ((m + tile_m - 1) // tile_m) * ((n + tile_n - 1) // tile_n) * l
    k_tiles = max(1, k // 64)
    if tiles >= 2:
        fill = (7 * max(1, num_sms) // 10) // tiles  # floor(0.70 * num_sms / tiles): ~0.7 wave
    else:
        fill = round(1.2 * math.sqrt(k_tiles))  # single tile: balance GEMM latency vs reduce
    return max(1, min(fill, k_tiles // 8, max_split_k))


def choose_reduce_vec_width(num_tiles: int, tile_m: int, tile_n: int, num_sms: int) -> int:
    """fp32 elements per reduce thread. The reduce is latency-bound, so fewer elements ->
    more threads (tile_mn/V per tile) -> better latency hiding; but too few over-decomposes
    into many tiny CTAs (num_tiles * tile_mn / (num_threads*V)) whose launch/scheduling
    overhead shows when the reduce isn't the bottleneck (many output tiles). Pick the
    smallest valid V whose total reduce-CTA count stays within ~8 GPU waves; else the
    largest valid V. (Few tiles -> V=1, max threads; many tiles -> larger V, fewer CTAs.)
    Note: in the real GEMM->reduce pipeline the reduce reads a workspace the GEMM just
    evicted from L2, so it is HBM-latency-bound and V (load width) barely moves the cold
    reduce time. The dominant reduce levers are the split_k that auto_split_k picks (slots to
    sum) and, for few-tile launches, the intra-CTA K-split R from choose_reduce_kgroups (which
    fills the GPU) -- not V, which here just sets CTA granularity for many-tile shapes."""
    nt = SplitKReduce.num_threads
    tile_mn = tile_m * tile_n
    cta_cap = max(1, num_sms) * 8
    best = 1
    for V in (1, 2, 4):
        if tile_n % V != 0 or tile_mn % (nt * V) != 0:
            continue
        best = V  # largest valid so far (fallback when none fit the cap)
        if num_tiles * (tile_mn // (nt * V)) <= cta_cap:
            return V
    return best


def choose_reduce_kgroups(
    num_tiles: int, tile_m: int, tile_n: int, split_k: int, vec_width: int, num_sms: int
) -> int:
    """Intra-CTA K-split factor R for the reduce. The reduce is latency-bound, so the goal
    is enough resident warps to hide HBM latency. Without K-split the reduce launches
    num_tiles * tile_mn / (num_threads * V) CTAs; for few output tiles that is far below
    the GPU's capacity (a single 128x128 tile is 64 CTAs at num_threads=256 -> ~5% warps
    active). R multiplies the CTA/warp count by R (each element handled by R threads that
    each reduce 1/R of the contributors, combined in smem). Pick the smallest R (power of
    two) that brings the CTA count to ~`target_waves` GPU waves, capped so each k-group
    still has work (R <= split_k) and the per-element thread group stays a warp-friendly
    size (elems_per_cta = num_threads/R divides tile_mn/V and is a multiple of 32). R=1
    when the launch is already GPU-full (many tiles) -> exact flat-ascending path."""
    nt = SplitKReduce.num_threads
    tile_mn = tile_m * tile_n
    target_ctas = max(1, num_sms) * 4
    base_ctas = num_tiles * tile_mn // (nt * vec_width)  # CTAs at R=1
    if base_ctas >= target_ctas:
        return 1  # already GPU-full at R=1 -> keep the exact flat-ascending (bit-exact) path
    best = 1
    R = 2
    while R <= split_k and R <= nt:
        elems = nt // R
        if elems % 32 != 0 or (tile_mn // vec_width) % elems != 0:
            R *= 2
            continue
        best = R  # largest valid so far
        if base_ctas * R >= target_ctas:
            return R
        R *= 2
    return best


_uniform_tables_cache: dict = {}


def uniform_splitk_tables(num_tiles: int, split_k: int, device) -> tuple[Tensor, Tensor]:
    """(tile_first_slot, tile_count) for the fixed split-K layout: every tile has exactly
    `split_k` contributors at slots [t*split_k, (t+1)*split_k). Cached per
    (num_tiles, split_k, device) so the parallel split-K path pays no per-call build cost.
    Stream-K will instead build non-uniform tables and feed the same reduce kernel."""
    dev_key = device.index if device.type == "cuda" else -1
    key = (num_tiles, split_k, dev_key)
    cached = _uniform_tables_cache.get(key)
    if cached is None:
        first = torch.arange(0, num_tiles * split_k, split_k, dtype=torch.int32, device=device)
        count = torch.full((num_tiles,), split_k, dtype=torch.int32, device=device)
        cached = (first, count)
        _uniform_tables_cache[key] = cached
    return cached


def splitk_reduce(
    ws: Tensor,  # (total_contributors * tile_m * tile_n,) f32 partials
    D: Tensor,  # (M, N, L), post-perm3d layout
    C: Optional[Tensor],  # (M, N, L), post-perm3d layout
    alpha: float | Tensor,
    beta: float | Tensor,
    rowvec: Optional[Tensor],  # (L, N)
    colvec: Optional[Tensor],  # (L, M)
    tile_first_slot: Tensor,  # (num_tiles,) i32
    tile_count: Tensor,  # (num_tiles,) i32
    tile_m: int,
    tile_n: int,
    vec_width: int = 1,
    kgroups: int = 1,
) -> None:
    compiled_fn = compile_splitk_reduce(
        D, C, alpha, beta, rowvec, colvec, tile_m, tile_n, vec_width, kgroups
    )

    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    alpha_mode, beta_mode = _scalar_modes(alpha, beta)

    def scalar_arg(scalar, mode):
        if mode == 0:
            return None
        elif mode == 1:
            return float(scalar)
        else:
            return scalar.data_ptr()

    compiled_fn(
        ws,
        D,
        C,
        scalar_arg(alpha, alpha_mode),
        scalar_arg(beta, beta_mode),
        rowvec,
        colvec,
        tile_first_slot,
        tile_count,
    )
