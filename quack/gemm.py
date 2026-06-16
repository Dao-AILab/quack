# Copyright (c) 2025-2026, Tri Dao.
# GEMM compilation via TVM-FFI with fake tensors and NamedTuple args.

from typing import Optional

import torch
from torch import Tensor

import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.runtime import make_ptr

from quack.cache_utils import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    get_device_capacity,
    get_max_active_clusters,
    get_num_sms,
    torch2cute_dtype_map,
)
from quack.gemm_default_epi import (
    GemmDefaultEpiMixin,
    GemmDefaultSm90,
    GemmDefaultSm100,
    GemmDefaultSm120,
)
from quack.rounding import RoundingMode
from quack.gemm_splitk_reduce import (
    auto_split_k,
    choose_reduce_kgroups,
    choose_reduce_vec_width,
    compile_splitk_reduce,
    splitk_reduce,
    uniform_splitk_tables,
)
from quack.gemm_tvm_ffi_utils import (
    get_majors,
    get_dtypes,
    perm3d,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_fake_gemm_tensors,
    compile_gemm_kernel,
)


@jit_cache
def _compile_gemm(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    rowvec_dtype,
    colvec_dtype,
    colvec_ndim,
    alpha_mode,
    beta_mode,
    add_to_output,
    concat_layout,
    varlen_m,
    varlen_k,
    gather_A,
    use_tma_gather,
    has_batch_idx_permute,
    device_capacity,
    rounding_mode,
    sr_seed_mode,
    has_trace_ptr,
    has_splitk=False,
    splitk_parallel=False,
):
    sm_to_cls = {
        9: GemmDefaultSm90,
        10: GemmDefaultSm100,
        11: GemmDefaultSm100,
        12: GemmDefaultSm120,
    }
    GemmCls = sm_to_cls[device_capacity[0]]
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        varlen_m=varlen_m,
        varlen_k=varlen_k,
        gather_A=gather_A,
    )

    def fake_scalar(mode, dtype=Float32):
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(1.0 if dtype == Float32 else 0)
        else:
            return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=4)

    mRowVec = fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4)
    if colvec_ndim == 2:
        mColVec = fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
    elif colvec_ndim == 1:  # m is total_m in this case
        mColVec = fake_tensor(colvec_dtype, (m,), leading_dim=0, divisibility=4)
    else:
        mColVec = None

    epi_args = GemmCls.EpilogueArguments(
        alpha=fake_scalar(alpha_mode),
        beta=fake_scalar(beta_mode),
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        add_to_output=add_to_output,
        rounding_mode=rounding_mode,
        sr_seed=fake_scalar(sr_seed_mode, dtype=Int32),
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9),
        has_batch_idx_permute,
        l,
        has_split_k=has_splitk,
        splitk_parallel=splitk_parallel,
    )
    aidx_len = m if varlen_m else (k if varlen_k else None)
    varlen_args = make_fake_varlen_args(varlen_m, varlen_k, gather_A, aidx_len)
    return compile_gemm_kernel(
        GemmCls,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        gather_A,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        has_trace_ptr=has_trace_ptr,
        use_tma_gather=use_tma_gather,
        concat_layout=concat_layout or None,
    )


def gemm(
    # (l, m, k) or (total_m, k) if varlen_m or (m, total_k) if varlen_k or (whatever, k) if gather_A_varlen_m or (m, whatever) if gather_A_varlen_k
    A: Tensor,
    B: Tensor,  # (l, n, k) or (n, total_k) if varlen_k
    D: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,  # (l, n)
    colvec_bias: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    cu_seqlens_k: Optional[Tensor] = None,  # (l+1,) cumulative sum of k values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) or (total_k,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (l,) permutation of batch indices for scheduler
    add_to_output: bool = False,
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
    use_tma_gather: bool = False,
    concat_layout: dict | None = None,
    split_k: int = 1,  # K splits per output tile (SM100 only); 0 = auto (auto_split_k)
    # "parallel": every split writes its own fp32 workspace slice, a second kernel
    # reduces them (cuBLAS-style; split_k can scale to fill the GPU). "serial": fused
    # in-kernel turnstile reduction (single kernel, but reduction cost grows with
    # split_k). Both are run-to-run deterministic.
    split_k_mode: str = "parallel",
    trace_ptr=None,  # Optional Int64 from TraceSession.ptr
) -> None:
    varlen_m = cu_seqlens_m is not None
    varlen_k = cu_seqlens_k is not None
    varlen = varlen_m or varlen_k
    gather_A = A_idx is not None
    assert not (varlen_m and varlen_k), "Only one of cu_seqlens_m and cu_seqlens_k"
    if gather_A:
        assert varlen, "gather_A requires varlen"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    if varlen:
        assert persistent, "varlen requires persistent=True"
    if add_to_output:
        assert not varlen_m, "Add to output not supported with varlen_m"
    if varlen_m:
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
    if varlen_k:
        assert A.stride(-2) == 1, "varlen_k requires A to be m-major"
        assert B.stride(-2) == 1, "varlen_k requires B to be n-major"

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10, 11, 12], "Only SM90, SM100, SM110, and SM120 are supported"
    if use_tma_gather:
        assert device_capacity[0] in [10, 11], "TMA gather currently requires SM100/SM110"
    if rounding_mode == RoundingMode.RS:
        assert device_capacity[0] == 10, "Stochastic rounding (RoundingMode.RS) requires SM100"
    if is_dynamic_persistent and device_capacity[0] == 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler in SM90 requires a semaphore in GMEM"
        )

    if split_k == 0:  # auto: pick a split_k for this shape (1 when split-K won't help)
        split_k = 1
        if (
            split_k_mode == "parallel"
            and device_capacity[0] in (10, 11)
            and not varlen
            and not gather_A
            and not add_to_output
            and tile_M != 256
        ):
            num_sms = get_num_sms(A.device)
            l_auto = A.shape[0] if A.ndim == 3 else 1
            split_k = auto_split_k(
                A.shape[-2], B.shape[-2], A.shape[-1], l_auto, tile_M, tile_N, num_sms
            )
    assert split_k >= 1, "split_k must be >= 1"
    assert split_k_mode in ("serial", "parallel"), "split_k_mode must be 'serial' or 'parallel'"
    splitk_parallel = split_k > 1 and split_k_mode == "parallel"
    if split_k > 1:
        assert device_capacity[0] in (10, 11), "split_k > 1 is only supported on SM100"
        assert not varlen and not gather_A, "split_k > 1 does not support varlen/gather_A"
        assert not add_to_output, "split_k > 1 does not support add_to_output"
        assert tile_M != 256, "split_k > 1 does not support 2-CTA tiles (tile_M=256)"
        assert persistent, "split_k > 1 requires the persistent scheduler"
        # The split work-index expansion is only wired for the static/dynamic persistent
        # scheduler (and the serial turnstile additionally needs its ordered work handout).
        is_dynamic_persistent = False
        if splitk_parallel:
            assert rounding_mode == RoundingMode.RN, (
                "parallel split-K converts in the reduce kernel (RN only)"
            )
            assert colvec_bias is None or colvec_bias.ndim == 2, (
                "parallel split-K supports only (L, M) colvec bias"
            )

    A_p, B_p, D_p, C_p = perm3d(A, B, D, C, varlen_m=varlen_m, varlen_k=varlen_k)
    a_major, b_major, d_major, c_major = get_majors(A_p, B_p, D_p, C_p)
    a_dtype, b_dtype, d_dtype, c_dtype = get_dtypes(A, B, D, C)

    alpha_mode = 2 if isinstance(alpha, Tensor) else (1 if alpha != 1.0 else 0)
    beta_mode = 2 if isinstance(beta, Tensor) else (1 if beta != 1.0 else 0)
    colvec_ndim = colvec_bias.ndim if colvec_bias is not None else 0
    concat_layout = tuple(sorted(concat_layout)) if concat_layout else ()

    sr_seed_mode = (
        2 if isinstance(sr_seed, Tensor) else (1 if rounding_mode == RoundingMode.RS else 0)
    )
    # Parallel split-K: the GEMM kernel only writes fp32 partials; D/C and every
    # epilogue op (alpha/beta/bias) move to the reduce kernel, so the GEMM is compiled
    # without them (no D TMA atoms / epi smem -> more mainloop stages).
    compiled_fn = _compile_gemm(
        a_dtype,
        b_dtype,
        None if splitk_parallel else d_dtype,
        None if splitk_parallel else c_dtype,
        a_major,
        b_major,
        None if splitk_parallel else d_major,
        None if splitk_parallel else c_major,
        (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        torch2cute_dtype_map[rowvec_bias.dtype]
        if rowvec_bias is not None and not splitk_parallel
        else None,
        torch2cute_dtype_map[colvec_bias.dtype]
        if colvec_bias is not None and not splitk_parallel
        else None,
        0 if splitk_parallel else colvec_ndim,
        0 if splitk_parallel else alpha_mode,
        0 if splitk_parallel else beta_mode,
        add_to_output,
        concat_layout,
        varlen_m,
        varlen_k,
        gather_A,
        use_tma_gather,
        batch_idx_permute is not None,
        device_capacity,
        rounding_mode,
        sr_seed_mode,
        trace_ptr is not None,
        split_k > 1,
        splitk_parallel,
    )
    splitk_num_tiles, splitk_reduce_vw, splitk_reduce_kg = None, 1, 1
    if split_k > 1:
        _l = A.shape[0] if A.ndim == 3 else 1
        splitk_num_tiles = (
            ((A.shape[-2] + tile_M - 1) // tile_M) * ((B.shape[-2] + tile_N - 1) // tile_N) * _l
        )
        if splitk_parallel:
            _num_sms = get_num_sms(A.device)
            # Reduce vec_width (threads/CTA granularity) depends on the tile count.
            splitk_reduce_vw = choose_reduce_vec_width(
                splitk_num_tiles, tile_M, tile_N, _num_sms
            )
            # Intra-CTA K-split factor: fills the GPU when there are few output tiles (the
            # reduce's recurring latency-bound case). R=1 (exact flat-ascending) otherwise.
            splitk_reduce_kg = choose_reduce_kgroups(
                splitk_num_tiles, tile_M, tile_N, split_k, splitk_reduce_vw, _num_sms
            )
    from quack.cache_utils import COMPILE_ONLY

    if splitk_parallel and COMPILE_ONLY:
        # AOT only: pre-compile the reduce so COMPILE_ONLY flows cache both kernels before
        # the early return. The hot path compiles it lazily inside splitk_reduce() below,
        # so doing it here too would just be a redundant cache lookup every call.
        compile_splitk_reduce(
            D_p,
            C_p,
            alpha,
            beta,
            rowvec_bias,
            colvec_bias,
            tile_M,
            tile_N,
            splitk_reduce_vw,
            splitk_reduce_kg,
        )

    if COMPILE_ONLY:
        return

    def scalar_arg(scalar, mode, dtype=Float32):
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(scalar)
        else:
            return scalar.data_ptr()

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0

    splitk_flags, splitk_ws, splitk_tables = None, None, None
    if split_k > 1:
        num_tiles = splitk_num_tiles
        # Parallel mode: one slot per (tile, split), summed by the reduce kernel.
        # Serial mode: one accumulation slot per tile, plus zeroed turnstile counters.
        # The workspace itself never needs initialization in either mode.
        ws_numel = num_tiles * tile_M * tile_N * (split_k if splitk_parallel else 1)
        assert ws_numel < 2**31, "split-K workspace indexing is 32-bit"
        if not splitk_parallel:
            splitk_flags = torch.zeros(num_tiles, dtype=torch.int32, device=A.device)
        splitk_ws = torch.empty(ws_numel, dtype=torch.float32, device=A.device)
        if splitk_parallel:
            # Table-driven reduce: each tile owns `count` slots from `first_slot`. The
            # fixed split-K layout is the uniform case (count=split_k); the reduce kernel
            # is otherwise agnostic to how the partials were partitioned (Stream-K-ready).
            splitk_tables = uniform_splitk_tables(num_tiles, split_k, A.device)

    # In parallel split-K, D/C/alpha/beta/bias belong to the reduce kernel, not the GEMM
    epi_args = GemmDefaultEpiMixin.EpilogueArguments(
        alpha=scalar_arg(alpha, alpha_mode) if not splitk_parallel else None,
        beta=scalar_arg(beta, beta_mode) if not splitk_parallel else None,
        mRowVecBroadcast=rowvec_bias if not splitk_parallel else None,
        mColVecBroadcast=colvec_bias if not splitk_parallel else None,
        add_to_output=None,
        rounding_mode=None,
        sr_seed=scalar_arg(sr_seed, sr_seed_mode, dtype=Int32),
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
        batch_idx_permute,
        split_k=split_k,
        splitk_flags=splitk_flags,
        splitk_ws=splitk_ws,
    )
    varlen_args = make_varlen_args(cu_seqlens_m, cu_seqlens_k, A_idx)

    gemm_D_p = D_p if not splitk_parallel else None
    gemm_C_p = C_p if not splitk_parallel else None
    if device_capacity[0] in [10, 11]:
        compiled_fn(
            A_p,
            B_p,
            gemm_D_p,
            gemm_C_p,
            epi_args,
            scheduler_args,
            varlen_args,
            None,
            None,
            trace_ptr,
        )
    else:
        compiled_fn(A_p, B_p, gemm_D_p, gemm_C_p, epi_args, scheduler_args, varlen_args, trace_ptr)
    if splitk_parallel:
        # Deterministic parallel reduction of the per-split partials + epilogue
        tile_first_slot, tile_count = splitk_tables
        splitk_reduce(
            splitk_ws,
            D_p,
            C_p,
            alpha,
            beta,
            rowvec_bias,
            colvec_bias,
            tile_first_slot,
            tile_count,
            tile_M,
            tile_N,
            splitk_reduce_vw,
            splitk_reduce_kg,
        )
