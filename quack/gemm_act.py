# Copyright (c) 2025-2026, Wentao Guo, Tri Dao.
"""GEMM + activation (optionally gated): thin wrappers over the epilogue-mod
path (quack.epilogues.linear_act_mod)."""

from __future__ import annotations
from typing import Optional

from torch import Tensor

from quack.activation import act_fn_map, gate_fn_map
from quack.gemm_host import GemmEpiPlan, run_gemm_epi_plan
from quack.gemm_tvm_ffi_utils import scalar_mode
from quack.rounding import RoundingMode


def gemm_act(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    D: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    PostAct: Tensor,  # (l, m, n) or (total_m, n//2) if gated
    tile_count_semaphore: Optional[Tensor],  # (1,)
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
    rowvec_bias: Optional[Tensor] = None,  # (l, n)
    colvec_bias: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    alpha: float | Tensor | None = None,  # pre-activation accumulator scale
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
    use_tma_gather: bool = False,
    concat_layout: tuple | None = None,
    SFA: Optional[Tensor] = None,  # (l, rm, rk, 32, 4, 4) blocked scale factors, or 5-D if 2D A
    SFB: Optional[Tensor] = None,  # (l, rn, rk, 32, 4, 4)
    # BlockScaledFormat names for A / B (independent); required when SFA/SFB are
    # passed (see quack.gemm.gemm).
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    b_kn: bool = False,  # B passed (k, n) / (l, k, n), transposed at trace time (dense SM90+)
    ag_args: Optional[tuple] = None,  # AllGather+GEMM flags contract (see quack/distributed/)
) -> GemmEpiPlan:
    """GEMM + activation (optionally gated), on the epilogue-mod path
    (quack.epilogues.linear_act_mod). ``alpha`` scales the accumulator BEFORE
    the activation (``aux = act(alpha * A @ B + ...)``) -- an output alpha
    cannot produce that through the nonlinearity. Neutral values (None or
    1.0) keep the alpha-free epilogue."""
    from quack.epilogues import linear_act_mod

    gated = activation in gate_fn_map
    if not gated:
        assert activation in act_fn_map, f"Unsupported activation {activation}"
    sr_seed_mode = (
        2 if isinstance(sr_seed, Tensor) else (1 if rounding_mode == RoundingMode.RS else 0)
    )
    sr = sr_seed_mode != 0
    if alpha is not None and scalar_mode(alpha) == 0:
        alpha = None  # neutral 1.0: keep the alpha-free epilogue
    mod = linear_act_mod(
        activation,
        gated=gated,
        has_c=C is not None,
        has_rowvec=rowvec_bias is not None,
        has_colvec=colvec_bias is not None,
        sr=sr,
        has_alpha=alpha is not None,
    )
    epi_args = dict(mAuxOut=PostAct)
    if rowvec_bias is not None:
        epi_args["mRowVecBroadcast"] = rowvec_bias
    if colvec_bias is not None:
        epi_args["mColVecBroadcast"] = colvec_bias
    if alpha is not None:
        epi_args["alpha"] = alpha
    if sr:
        epi_args["sr_seed"] = sr_seed
    return mod.gemm(
        A,
        B,
        D,
        C,
        epi_args=epi_args,
        ag_args=ag_args,
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
        use_tma_gather=use_tma_gather,
        concat_layout=tuple(sorted(concat_layout)) if concat_layout else None,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        b_kn=b_kn,
    )


def run_gemm_act_plan(
    plan: GemmEpiPlan,
    A: Tensor,
    B: Tensor,
    D: Optional[Tensor],
    C: Optional[Tensor],
    PostAct: Tensor,
    *,
    tile_count_semaphore: Optional[Tensor] = None,
    rowvec_bias: Optional[Tensor] = None,
    colvec_bias: Optional[Tensor] = None,
    alpha: float | Tensor | None = None,
    sr_seed: int | Tensor = 0,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    SFA: Optional[Tensor] = None,
    SFB: Optional[Tensor] = None,
) -> None:
    """Launch a resolved plan: only per-call pointers and scalar values here.

    The tensors must match the metadata the plan was built from (``gemm_act``
    guarantees that via its key; an outer layer holding the returned plan must
    guarantee it with its own key).
    """
    run_gemm_epi_plan(
        plan,
        A,
        B,
        D,
        C,
        dict(
            mAuxOut=PostAct,
            mRowVecBroadcast=rowvec_bias,
            mColVecBroadcast=colvec_bias,
            alpha=alpha if alpha is None or scalar_mode(alpha) != 0 else None,
            sr_seed=sr_seed,
        ),
        tile_count_semaphore=tile_count_semaphore,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        SFA=SFA,
        SFB=SFB,
    )


gemm_gated = gemm_act
