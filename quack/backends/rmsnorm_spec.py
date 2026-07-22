"""Backend-neutral RMSNorm call specification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class RMSNormSpec:
    input_dtype: Any
    weight_dtype: Any
    out_dtype: Any
    residual_dtype: Any
    has_weight: bool
    has_bias: bool
    has_residual: bool
    per_head: bool
    prenorm: bool
    weight_offset: float
    eps: float
    normalized_size: Any


def normalize_rmsnorm_spec(
    x,
    weight=None,
    bias=None,
    residual=None,
    out_dtype=None,
    residual_dtype=None,
    eps: float = 1e-6,
    prenorm: bool = False,
    weight_offset: float = 0.0,
) -> RMSNormSpec:
    """Normalize Quack's complete RMSNorm signature without selecting a backend."""

    weight_rank: Optional[int] = weight.dim() if weight is not None else None
    bias_rank: Optional[int] = bias.dim() if bias is not None else None
    per_head = weight_rank == 2 or bias_rank == 2
    normalized_size = x.shape[-1] if x.dim() else 0
    return RMSNormSpec(
        input_dtype=x.dtype,
        weight_dtype=weight.dtype if weight is not None else None,
        out_dtype=out_dtype,
        residual_dtype=residual_dtype,
        has_weight=weight is not None,
        has_bias=bias is not None,
        has_residual=residual is not None,
        per_head=per_head,
        prenorm=bool(prenorm),
        weight_offset=float(weight_offset),
        eps=float(eps),
        normalized_size=normalized_size,
    )


__all__ = ["RMSNormSpec", "normalize_rmsnorm_spec"]
