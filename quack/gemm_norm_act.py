# Copyright (c) 2025-2026, Tri Dao.
"""GEMM + normalize (colvec/rowvec multiply) + activation: thin wrappers over
the epilogue-mod path (quack.epilogues.norm_act_mod)."""

from __future__ import annotations
from typing import Optional

from torch import Tensor

from quack.activation import act_fn_map, gate_fn_map
from quack.rounding import RoundingMode


def gemm_norm_act_fn(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m
    B: Tensor,  # (l, n, k)
    D: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    PostAct: Tensor,  # (l, m, n) or (total_m, n//2) if gated
    tile_count_semaphore: Optional[Tensor],
    activation: Optional[str],
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
    colvec: Optional[Tensor] = None,  # (l, m) or (total_m,) — rstd
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
    b_kn: bool = False,  # B passed (k, n) / (l, k, n), transposed at trace time (dense SM90+)
):
    """GEMM + normalize + activation, on the epilogue-mod path (quack.epilogues)."""
    from quack.epilogues import norm_act_mod

    gated = activation in gate_fn_map
    if not gated:
        assert activation in act_fn_map, f"Unsupported activation {activation}"
    sr = rounding_mode == RoundingMode.RS or isinstance(sr_seed, Tensor)
    sr_seed_mode = (
        2 if isinstance(sr_seed, Tensor) else (1 if rounding_mode == RoundingMode.RS else 0)
    )
    mod = norm_act_mod(
        activation,
        gated=gated,
        has_c=C is not None,
        has_rowvec=rowvec is not None,
        has_colvec=colvec is not None,
        sr=sr,
    )
    epi_args = dict(mAuxOut=PostAct)
    if rowvec is not None:
        epi_args["mRowVecBroadcast"] = rowvec
    if colvec is not None:
        epi_args["mColVecBroadcast"] = colvec
    if sr:
        epi_args["sr_seed"] = sr_seed
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
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        rounding_mode=rounding_mode,
        epi_key_overrides={"sr_seed": sr_seed_mode} if sr else None,
        b_kn=b_kn,
    )


def run_gemm_norm_act_plan(
    plan,
    A: Tensor,
    B: Tensor,
    D: Optional[Tensor],
    C: Optional[Tensor],
    PostAct: Tensor,
    *,
    tile_count_semaphore: Optional[Tensor] = None,
    rowvec: Optional[Tensor] = None,
    colvec: Optional[Tensor] = None,
    sr_seed: int | Tensor = 0,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
) -> None:
    """Launch a resolved mod plan: only per-call pointers and scalar values."""
    from quack.gemm_host import run_gemm_epi_plan

    run_gemm_epi_plan(
        plan,
        A,
        B,
        D,
        C,
        dict(
            mAuxOut=PostAct,
            mRowVecBroadcast=rowvec,
            mColVecBroadcast=colvec,
            sr_seed=sr_seed,
        ),
        tile_count_semaphore=tile_count_semaphore,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
