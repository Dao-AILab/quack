# Copyright (c) 2025-2026, QuACK team.
# GEMM compilation via TVM-FFI with fake tensors and NamedTuple args.

from typing import Optional

import torch
from torch import Tensor

import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.runtime import make_ptr

from quack.cache import jit_cache
from quack.split_k_reduce import split_k_reduce
from quack.gemm_config import SplitKMode
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.gemm_default_epi import (
    GemmDefaultEpiMixin,
    GemmDefaultSm80,
    GemmDefaultSm90,
    GemmDefaultSm100,
    GemmDefaultSm120,
)
from quack.rounding import RoundingMode
from quack.gemm_tvm_ffi_utils import (
    get_majors,
    get_dtypes,
    perm3d,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_fake_gemm_tensors,
    make_fake_sf_tensor,
    compile_gemm_kernel,
    validate_blockscaled_sf,
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
    num_warps,
    sf_dtype=None,
    sf_vec_size=None,
    split_k=1,
    split_k_mode=SplitKMode.SERIAL,
):
    sm_to_cls = {
        8: GemmDefaultSm80,
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
    if split_k > 1 and split_k_mode == SplitKMode.SEPARATE:
        # D is the f32 partials workspace; its batch extent is l * split_k, not l,
        # so it needs its own symbolic batch dim.
        mD = fake_tensor(
            d_dtype,
            (m, n, cute.sym_int()),
            leading_dim=1 if d_major == "n" else 0,
            divisibility=128 // d_dtype.width,
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
        split_k_semaphore=(
            # (ntile_m, ntile_n, L) per-tile completion flag; tile counts are
            # runtime-symbolic. SERIAL: turnstile; PARALLEL: arrival counter.
            fake_tensor(Int32, (cute.sym_int(), cute.sym_int(), cute.sym_int()), leading_dim=1)
            if split_k > 1 and split_k_mode != SplitKMode.SEPARATE
            else None
        ),
        split_k_workspace=(
            # (cta_tile_m * cta_tile_n, ntile_m, ntile_n, L) raw-f32-partials regions,
            # one flat fragment stripe per output tile.
            fake_tensor(
                Float32,
                (cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()),
                leading_dim=0,
                divisibility=4,
            )
            if split_k > 1 and split_k_mode != SplitKMode.SEPARATE
            else None
        ),
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] <= 9), has_batch_idx_permute, l
    )
    aidx_len = m if varlen_m else (k if varlen_k else None)
    varlen_args = make_fake_varlen_args(varlen_m, varlen_k, gather_A, aidx_len)
    if sf_dtype is not None:
        # Padded SF buffers have a static batch dim of exactly 1 (not l): SFA for
        # varlen_m (M-padded) and varlen_k (K-padded); SFB is K-padded too for
        # varlen_k but stays per-batch (l, rn, rk, ...) for varlen_m.
        mSFA = make_fake_sf_tensor(sf_dtype, 1 if (varlen_m or varlen_k) else l)
        mSFB = make_fake_sf_tensor(sf_dtype, 1 if varlen_k else l)
    else:
        mSFA, mSFB = None, None
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
        mSFA=mSFA,
        mSFB=mSFB,
        use_tma_gather=use_tma_gather,
        concat_layout=concat_layout or None,
        num_warps=num_warps,
        sf_vec_size=sf_vec_size,
        split_k=split_k,
        split_k_mode=split_k_mode,
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
    cluster_K: int = 1,
    tile_K: int | None = None,
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
    num_warps: Optional[int] = None,
    # SFA/SFB: (l, rm/rn, rk, 32, 4, 4) blocked scale factors. For varlen_m, SFA is
    # M-padded (1, total_padded_rm, rk, 32, 4, 4) while SFB stays per-batch. For
    # varlen_k, BOTH are K-padded (1, rm/rn, total_padded_rk, 32, 4, 4); pad bytes
    # may be arbitrary (the kernel skips the MMA instructions covering them).
    # See AI/varlen_blockscaled_sf_layout.md.
    SFA: Optional[Tensor] = None,
    SFB: Optional[Tensor] = None,
    split_k: int = 1,
    split_k_mode: int = SplitKMode.SERIAL,
) -> None:
    varlen_m = cu_seqlens_m is not None
    varlen_k = cu_seqlens_k is not None
    varlen = varlen_m or varlen_k
    gather_A = A_idx is not None
    blockscaled = SFA is not None
    assert not (varlen_m and varlen_k), "Only one of cu_seqlens_m and cu_seqlens_k"
    if gather_A:
        assert varlen, "gather_A requires varlen"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    if add_to_output:
        assert not varlen_m, "Add to output not supported with varlen_m"
    assert split_k >= 1, "split_k must be >= 1"
    if split_k > 1:
        if split_k_mode not in tuple(SplitKMode):
            raise ValueError(
                f"split_k_mode must be a SplitKMode (SERIAL, PARALLEL, or SEPARATE), "
                f"got {split_k_mode!r}"
            )
        split_k_mode = SplitKMode(split_k_mode)
        if varlen or gather_A:
            raise ValueError("split_k requires a dense GEMM (no varlen_m/varlen_k/gather_A)")
        if rounding_mode != RoundingMode.RN:
            raise ValueError("split_k does not support stochastic rounding")
        if blockscaled and split_k_mode == SplitKMode.SEPARATE:
            raise NotImplementedError(
                "block-scaled split_k does not support SEPARATE yet; use SERIAL or PARALLEL"
            )
    if varlen_m:
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
    if varlen_k:
        assert A.stride(-2) == 1, "varlen_k requires A to be m-major"
        assert B.stride(-2) == 1, "varlen_k requires B to be n-major"

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )
    sf_dtype, sf_vec_size = None, None
    if blockscaled:
        assert not gather_A, "Blockscaled GEMM does not support gather_A yet"
        assert not concat_layout, "Blockscaled GEMM does not support concat_layout"
        assert tile_K is None, "Blockscaled GEMM derives tile_K from the MMA instruction"
        if varlen_m:
            num_batches = cu_seqlens_m.shape[0] - 1
        elif varlen_k:
            num_batches = cu_seqlens_k.shape[0] - 1
        else:
            num_batches = None
        sf_dtype, sf_vec_size = validate_blockscaled_sf(
            A, B, SFA, SFB, device_capacity, num_batches=num_batches, varlen_k=varlen_k
        )
    if split_k > 1 and device_capacity[0] not in [9, 10, 11, 12]:
        raise ValueError("split_k > 1 requires SM90, SM100, SM110, or SM120")
    if use_tma_gather:
        assert device_capacity[0] in [10, 11], "TMA gather currently requires SM100/SM110"
    if rounding_mode == RoundingMode.RS:
        assert device_capacity[0] == 10, "Stochastic rounding (RoundingMode.RS) requires SM100"
    if is_dynamic_persistent and device_capacity[0] <= 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler for SM8x and SM90 requires a semaphore in GMEM"
        )
    if device_capacity[0] == 8:
        if add_to_output:
            C = D
            add_to_output = False

    # Split-K: the full epilogue (alpha, beta*C, bias) runs exactly once per output tile,
    # on the entity that owns the completed f32 sum; non-finalizing splits emit raw f32
    # partials only.
    # SEPARATE: the GEMM stores raw f32 partials into a workspace whose batch mode is the
    # combined (l * split_k + split) index (majorness matches D, contiguous dim padded to
    # 16 bytes for TMA); the separate reduction kernel sums the splits in fixed order and
    # applies the epilogue — so all epi operands are routed there, not into the GEMM.
    D_gemm = D
    C_gemm = C
    alpha_gemm, beta_gemm = alpha, beta
    rowvec_gemm, colvec_gemm = rowvec_bias, colvec_bias
    split_k_workspace = None
    gemm_add_to_output = add_to_output
    if split_k > 1 and split_k_mode == SplitKMode.SEPARATE:
        num_l, len_m, len_n = D.shape
        if D.stride(-1) == 1:  # n-major
            n_pad = (len_n + 3) // 4 * 4
            split_k_workspace = torch.empty(
                (num_l * split_k, len_m, n_pad), dtype=torch.float32, device=D.device
            )[..., :len_n]
        else:  # m-major
            m_pad = (len_m + 3) // 4 * 4
            split_k_workspace = torch.empty(
                (num_l * split_k, len_n, m_pad), dtype=torch.float32, device=D.device
            )[..., :len_m].mT
        D_gemm = split_k_workspace
        # The epilogue (and the prior value of D for add_to_output) is applied once by
        # the reduction kernel instead.
        C_gemm = None
        alpha_gemm, beta_gemm = 1.0, 1.0
        rowvec_gemm, colvec_gemm = None, None
        gemm_add_to_output = False
    # SERIAL/PARALLEL: one Int32 completion flag per output tile (SERIAL: turnstile in
    # split order, deterministic; PARALLEL: arrival counter, not deterministic) plus a
    # flat f32 region per tile holding the non-finalizing splits' raw accumulator
    # fragments. The scheduler's CTA tile ids are cluster-rounded (a boundary cluster can
    # contain CTAs whose tile is fully out of bounds but which still run the protocol),
    # so tile counts are rounded up to cluster multiples. The per-CTA tile M must match
    # the kernel exactly (the workspace region stride is cta_tile_m * tile_N): SM100/110
    # 2-CTA MMA (cluster_M even, tile_M 128/256) halves it.
    split_k_semaphore = None
    if split_k > 1 and split_k_mode != SplitKMode.SEPARATE:
        num_l, len_m, len_n = D.shape
        use_2cta = device_capacity[0] in (10, 11) and cluster_M % 2 == 0 and tile_M in (128, 256)
        cta_tile_m = tile_M // 2 if use_2cta else tile_M
        ntile_m = (len_m + cta_tile_m - 1) // cta_tile_m
        ntile_n = (len_n + tile_N - 1) // tile_N
        ntile_m = (ntile_m + cluster_M - 1) // cluster_M * cluster_M
        ntile_n = (ntile_n + cluster_N - 1) // cluster_N * cluster_N
        # Kernel-facing layouts are (ntile_m, ntile_n, L) and (E, ntile_m, ntile_n, L):
        # the CuTe layouts own the address computation, so the kernel just slices them
        # at (pid_m, pid_n, batch).
        split_k_semaphore = torch.zeros(
            (num_l, ntile_m, ntile_n), dtype=torch.int32, device=D.device
        )
        tile_elems = cta_tile_m * tile_N
        if split_k_mode == SplitKMode.SERIAL:
            # Split 0's in-order plain store initializes each region.
            split_k_workspace = torch.empty(
                (num_l, ntile_m, ntile_n, tile_elems), dtype=torch.float32, device=D.device
            )
        else:
            # PARALLEL: every split red.adds in arrival order; no initializing store.
            split_k_workspace = torch.zeros(
                (num_l, ntile_m, ntile_n, tile_elems), dtype=torch.float32, device=D.device
            )

    A_p, B_p, D_p, C_p = perm3d(A, B, D_gemm, C_gemm, varlen_m=varlen_m, varlen_k=varlen_k)
    a_major, b_major, d_major, c_major = get_majors(A_p, B_p, D_p, C_p)
    a_dtype, b_dtype, d_dtype, c_dtype = get_dtypes(A, B, D_gemm, C_gemm)

    alpha_mode = 2 if isinstance(alpha_gemm, Tensor) else (1 if alpha_gemm != 1.0 else 0)
    beta_mode = 2 if isinstance(beta_gemm, Tensor) else (1 if beta_gemm != 1.0 else 0)
    colvec_ndim = colvec_gemm.ndim if colvec_gemm is not None else 0
    concat_layout = tuple(sorted(concat_layout)) if concat_layout else ()

    sr_seed_mode = (
        2 if isinstance(sr_seed, Tensor) else (1 if rounding_mode == RoundingMode.RS else 0)
    )
    tile_shape_mnk = (tile_M, tile_N) if tile_K is None else (tile_M, tile_N, tile_K)
    compiled_fn = _compile_gemm(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        tile_shape_mnk,
        (cluster_M, cluster_N, cluster_K),
        pingpong,
        persistent,
        is_dynamic_persistent,
        torch2cute_dtype_map[rowvec_gemm.dtype] if rowvec_gemm is not None else None,
        torch2cute_dtype_map[colvec_gemm.dtype] if colvec_gemm is not None else None,
        colvec_ndim,
        alpha_mode,
        beta_mode,
        gemm_add_to_output,
        concat_layout,
        varlen_m,
        varlen_k,
        gather_A,
        use_tma_gather,
        batch_idx_permute is not None,
        device_capacity,
        rounding_mode,
        sr_seed_mode,
        num_warps,
        sf_dtype,
        sf_vec_size,
        split_k,
        split_k_mode,
    )

    def reduce_staged_split_k():
        # The reduction kernel wants the contiguous dim last; transpose the views for
        # m-major outputs (the kernel is elementwise in (m, n), so this is free; the
        # row/col vector roles swap in the transposed space).
        ws_v, out_v, c_v = split_k_workspace, D, C
        rowvec_v, colvec_v = rowvec_bias, colvec_bias
        if D.stride(-1) != 1:
            ws_v, out_v = ws_v.transpose(1, 2), out_v.transpose(1, 2)
            c_v = c_v.transpose(1, 2) if c_v is not None else None
            rowvec_v, colvec_v = colvec_v, rowvec_v
        if c_v is not None and c_v.stride(-1) != 1:
            # The reduce kernel requires C contiguous along out's contiguous dim; a C
            # whose majorness differs from D's (legal in the GEMM epilogue, which
            # tracks c_major independently) is materialized once here.
            c_v = c_v.contiguous()
        split_k_reduce(
            ws_v,
            out_v,
            split_k,
            alpha=alpha,
            beta=beta,
            C=c_v,
            rowvec_bias=rowvec_v,
            colvec_bias=colvec_v,
            add_to_output=add_to_output,
        )

    staged_split_k = split_k > 1 and split_k_mode == SplitKMode.SEPARATE

    def scalar_arg(scalar, mode, dtype=Float32):
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(scalar)
        else:
            return scalar.data_ptr()

    cluster_size = cluster_M * cluster_N * cluster_K
    max_active_clusters = (
        get_max_active_clusters(cluster_size, device_capacity=device_capacity) if persistent else 0
    )

    epi_args = GemmDefaultEpiMixin.EpilogueArguments(
        alpha=scalar_arg(alpha_gemm, alpha_mode),
        beta=scalar_arg(beta_gemm, beta_mode),
        mRowVecBroadcast=rowvec_gemm,
        mColVecBroadcast=colvec_gemm,
        add_to_output=None,
        rounding_mode=None,
        sr_seed=scalar_arg(sr_seed, sr_seed_mode, dtype=Int32),
        split_k_semaphore=(
            split_k_semaphore.permute(1, 2, 0) if split_k_semaphore is not None else None
        ),
        split_k_workspace=(
            split_k_workspace.permute(3, 1, 2, 0)
            if split_k_workspace is not None and split_k_mode != SplitKMode.SEPARATE
            else None
        ),
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        # Must mirror make_fake_scheduler_args in _compile_gemm: only the SM8x/SM90
        # dynamic scheduler consumes the semaphore; SM100 uses CLC instead, and the
        # compiled signature has None there.
        tile_count_semaphore if (is_dynamic_persistent and device_capacity[0] <= 9) else None,
        batch_idx_permute,
    )
    varlen_args = make_varlen_args(cu_seqlens_m, cu_seqlens_k, A_idx)

    if device_capacity[0] in [10, 11]:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, SFA, SFB)
    else:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args)

    if staged_split_k:
        reduce_staged_split_k()
