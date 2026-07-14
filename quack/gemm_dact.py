# Copyright (c) 2025-2026, Tri Dao.
"""GEMM + activation backward (dact / dgated): thin wrappers over the
epilogue-mod path (quack.epilogues.dact_mod / dgated_mod)."""

from __future__ import annotations
from typing import Optional

from torch import Tensor

from quack.activation import dact_fn_map, dgate_fn_map
from quack.rounding import RoundingMode  # noqa: F401  (re-exported for callers)


def gemm_dact(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    Out: Tensor,  # (l, m, n) or (total_m, n) if varlen_m; or (l, m, 2*n)/(total_m, 2*n) if dgated
    PreAct: Tensor,  # same shape as Out
    PostAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    tile_K: int | None = None,
    pingpong: bool = True,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    colvec_scale: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m (dgated only)
    # (l, m, ceildiv(n, tile_n)), or (total_m, ceildiv(n, tile_n)) if varlen_m (dgated only)
    colvec_reduce: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
    use_tma_gather: bool = False,
    b_kn: bool = False,  # B passed (k, n) / (l, k, n), transposed at trace time (dense SM90+)
):
    """GEMM + activation backward, on the epilogue-mod path (quack.epilogues)."""
    from quack.epilogues import dact_mod, dgated_mod

    is_dgated = activation in dgate_fn_map
    if not is_dgated:
        assert activation in dact_fn_map, f"Unsupported activation {activation}"
        assert colvec_scale is None, "colvec_scale is only supported for gated activations"
        assert colvec_reduce is None, "colvec_reduce is only supported for gated activations"
        mod = dact_mod(activation)
        epi_args = dict(mAuxOut=PostAct)
    else:
        mod = dgated_mod(
            activation,
            has_scale=colvec_scale is not None,
            has_reduce=colvec_reduce is not None,
        )
        epi_args = dict(mAuxOut=PostAct)
        if colvec_scale is not None:
            epi_args["mColVecBroadcast"] = colvec_scale
        if colvec_reduce is not None:
            epi_args["mColVecReduce"] = colvec_reduce
    return mod.gemm(
        A,
        B,
        Out,
        PreAct,
        epi_args=epi_args,
        tile_M=tile_M,
        tile_N=tile_N,
        cluster_M=cluster_M,
        cluster_N=cluster_N,
        tile_K=tile_K,
        pingpong=pingpong,
        persistent=persistent,
        is_dynamic_persistent=is_dynamic_persistent,
        max_swizzle_size=max_swizzle_size,
        tile_count_semaphore=tile_count_semaphore,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        use_tma_gather=use_tma_gather,
        b_kn=b_kn,
    )


def run_gemm_dact_plan(
    plan,
    A: Tensor,
    B: Tensor,
    Out: Tensor,
    PreAct: Tensor,
    PostAct: Tensor,
    *,
    tile_count_semaphore: Optional[Tensor] = None,
    colvec_scale: Optional[Tensor] = None,
    colvec_reduce: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
) -> None:
    """Launch a resolved mod plan: only per-call pointers here (packed dgated
    C/D cross raw — the f32 recast is compiled into the trace, cd_packed)."""
    from quack.gemm_host import run_gemm_epi_plan

    run_gemm_epi_plan(
        plan,
        A,
        B,
        Out,
        PreAct,
        dict(mAuxOut=PostAct, mColVecBroadcast=colvec_scale, mColVecReduce=colvec_reduce),
        tile_count_semaphore=tile_count_semaphore,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )


gemm_dgated = gemm_dact
