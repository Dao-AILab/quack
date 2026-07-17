# Copyright (c) 2025-2026, Tri Dao.
"""GEMM + column sq-sum reduce + optional rowvec scaling: thin wrapper over
the epilogue-mod path (quack.epilogues.sq_reduce_mod)."""

from __future__ import annotations
from typing import Optional

from torch import Tensor


def gemm_sq_reduce(
    A: Tensor,  # (l, m, k)
    B: Tensor,  # (l, n, k)
    D: Tensor,  # (l, m, n)
    C: Optional[Tensor],  # (l, m, n)
    colvec_reduce: Tensor,  # (l, m, ceildiv(n, tile_n))
    tile_count_semaphore: Optional[Tensor],  # (1,)
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    tile_K: int | None = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec: Optional[Tensor] = None,  # (l, n) — norm_weight
    aux_out: Optional[Tensor] = None,  # (l, m, n) — pre-rowvec output snapshot
    b_kn: bool = False,  # B passed (k, n) / (l, k, n), transposed at trace time (SM90+)
):
    """GEMM + sq_reduce + optional rowvec scaling, on the epilogue-mod path.

    D_raw = A @ B (+ C), colvec_reduce[m] = sum_n(D_raw[m,n]^2), D = D_raw * rowvec.
    """
    from quack.epilogues import sq_reduce_mod

    mod = sq_reduce_mod(
        has_c=C is not None, has_rowvec=rowvec is not None, has_aux=aux_out is not None
    )
    epi_args = dict(mColVecReduce=colvec_reduce)
    if rowvec is not None:
        epi_args["mRowVecBroadcast"] = rowvec
    if aux_out is not None:
        epi_args["mAuxOut"] = aux_out
    return mod.gemm(
        A,
        B,
        D,
        C,
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
        b_kn=b_kn,
    )


def run_gemm_sq_reduce_plan(
    plan,
    A: Tensor,
    B: Tensor,
    D: Tensor,
    C: Optional[Tensor],
    colvec_reduce: Tensor,
    *,
    tile_count_semaphore: Optional[Tensor] = None,
    rowvec: Optional[Tensor] = None,
    aux_out: Optional[Tensor] = None,
) -> None:
    """Launch a resolved mod plan: only per-call pointers here."""
    from quack.gemm_host import run_gemm_epi_plan

    run_gemm_epi_plan(
        plan,
        A,
        B,
        D,
        C,
        dict(mRowVecBroadcast=rowvec, mColVecReduce=colvec_reduce, mAuxOut=aux_out),
        tile_count_semaphore=tile_count_semaphore,
    )
