"""Phased ``scaled_grouped_mm`` interface (mirrors the proposed ``torch._scaled_grouped_mm``).

Phase 1 (this file): a clean, stable *local* MXFP8 grouped GEMM. The fused expert-parallel
prologue/epilogue descriptors (NVLink gather/scatter, SwiGLU) are *defined* here for forward
compatibility but raise ``NotImplementedError`` until their phases land. The Phase-1 signature is
fixed; later phases only add ``prologue`` / ``epilogue`` descriptors.

Mapping onto quack's kernel (``quack/mxfp8_grouped_gemm.py``):

    a            (M_total, K) fp8                 -> a
    b            (G, N, K) fp8 stacked weights    -> b.transpose(-2, -1)  == (G, K, N)
    scale_a      (M_total, K//32) e8m0 unpacked   -> sfa   (packed internally)
    scale_b      (G, N, K//32) e8m0 unpacked      -> sfb   (packed internally)
    group_sizes  (G,) int32 tokens-per-group      -> offs = cumsum(group_sizes)   (on device)

``group_sizes`` is the *per-group* count (``num_tokens_per_expert``); quack consumes cumulative
end offsets, so we ``cumsum`` on device -- no host sync, so the call stays CUDA-graph capturable.
Computes, per group ``g``: ``out[rows of g] = a[rows of g] @ b[g].t()`` (i.e. ``out = input @
weight.t()``).
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

from quack.mxfp8_grouped_gemm import MXFP8GroupedGemm, mxfp8_grouped_gemm

# Fusion descriptors for Phases 2+; Phase 1 accepts only ``None`` (see scaled_grouped_mm).


@dataclass(frozen=True)
class GatherPrologue:
    """Phase 2 -- fused NVLink gather + block-scaled quant. When active, ``scale_a`` MUST be
    ``None`` (the kernel produces it). ``buffer`` is SymmMem (cross-GPU) or a local Tensor."""

    gather_ptrs: Tensor  # [M] int64 absolute pointer per output row
    buffer: object  # SymmMem | Tensor source memory
    input_dtype: torch.dtype = torch.bfloat16


@dataclass(frozen=True)
class OffsetPrologue:
    """Phase 5 -- read A rows from arbitrary offsets in an overallocated local buffer."""

    row_offsets: Tensor  # [M] int64 byte offset per row into a


@dataclass(frozen=True)
class ScatterEpilogue:
    """Phase 3 -- fused NVLink scatter: each output row is written to ``buffer`` at
    ``scatter_ptrs``. Used for the EP combine phase."""

    scatter_ptrs: Tensor  # [M] int64 absolute pointer per output row
    buffer: object  # SymmMem | Tensor destination memory


@dataclass(frozen=True)
class ScatterSwiGLUEpilogue(ScatterEpilogue):
    """Phase 4 -- fused SwiGLU backward + quant + scatter (COMBINE_SWIGLU_BWD)."""

    fast_math: bool = False
    return_col_quant: bool = True


@dataclass(frozen=True)
class OffsetEpilogue:
    """Phase 5 -- write output rows to arbitrary offsets in an overallocated local buffer."""

    row_offsets: Tensor  # [M] int64 byte offset per row into out


@dataclass(frozen=True)
class KernelConfig:
    """Resource / tuning knobs. Phase 1 ignores these except ``num_sms``, which is rejected
    (the kernel uses all SMs)."""

    num_sms: Optional[int] = None
    config: Optional[str] = None
    m_multiple_of: int = 128


Prologue = Union[GatherPrologue, OffsetPrologue]
Epilogue = Union[ScatterEpilogue, ScatterSwiGLUEpilogue, OffsetEpilogue]


class PreparedGroupedWeights:
    """Fixed expert weights with the B operand + B-scale pre-packed once, so repeated calls skip
    the per-call B-scale pack. Pass it to :func:`scaled_grouped_mm` in place of ``(b, scale_b)``
    (with ``scale_b=None``), or call it directly::

        w = prepare_weights(b, scale_b)        # b=(G,N,K), scale_b=(G,N,K//32); the fixed weights
        y = w(a, scale_a, group_sizes)         # per step (a / scale_a / group_sizes vary)
        y = scaled_grouped_mm(a, w, scale_a, None, group_sizes)   # equivalent
    """

    def __init__(self, b: Tensor, scale_b: Tensor):
        assert b.dim() == 3, f"b must be (G, N, K), got {tuple(b.shape)}"
        # MXFP8GroupedGemm takes b as (G, K, N) and pre-packs scale_b in __init__.
        self._gemm = MXFP8GroupedGemm(b.transpose(-2, -1), scale_b, varlen_nonaligned=True)
        self.shape = tuple(b.shape)  # (G, N, K)

    def __call__(
        self,
        a: Tensor,
        scale_a: Tensor,
        group_sizes: Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        *,
        kernel_config: Optional["KernelConfig"] = None,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        return scaled_grouped_mm(
            a, self, scale_a, None, group_sizes, out_dtype, kernel_config=kernel_config, out=out
        )


def prepare_weights(b: Tensor, scale_b: Tensor) -> PreparedGroupedWeights:
    """Pre-pack the fixed expert weights + block scales for repeated ``scaled_grouped_mm`` calls
    (see :class:`PreparedGroupedWeights`). ``b`` is ``(G, N, K)`` fp8, ``scale_b`` is
    ``(G, N, K//32)`` e8m0."""
    return PreparedGroupedWeights(b, scale_b)


def scaled_grouped_mm(
    a: Tensor,
    b: Union[Tensor, PreparedGroupedWeights],
    scale_a: Optional[Tensor],
    scale_b: Optional[Tensor],
    group_sizes: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    prologue: Optional[Prologue] = None,
    epilogue: Optional[Epilogue] = None,
    kernel_config: Optional[KernelConfig] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Phase 1: local MXFP8 grouped GEMM. See the module docstring for the shape/layout
    contract. Returns the GEMM output ``C`` of shape ``(M_total, N)``.

    ``prologue`` / ``epilogue`` fusion is Phase 2+ and raises ``NotImplementedError``.
    """
    if prologue is not None:
        raise NotImplementedError(
            f"prologue={type(prologue).__name__} is a Phase 2+ fusion and is not implemented yet; "
            "Phase 1 supports only the local grouped GEMM (prologue=None)."
        )
    if epilogue is not None:
        raise NotImplementedError(
            f"epilogue={type(epilogue).__name__} is a Phase 3+ fusion and is not implemented yet; "
            "Phase 1 supports only the local grouped GEMM (epilogue=None)."
        )
    if scale_a is None:
        raise ValueError(
            "scale_a is required in Phase 1; it may be None only with a GatherPrologue (Phase 2)."
        )
    if out_dtype != torch.bfloat16:
        raise NotImplementedError(f"out_dtype={out_dtype} is not supported; only torch.bfloat16.")
    if kernel_config is not None and kernel_config.num_sms is not None:
        raise NotImplementedError(
            "kernel_config.num_sms (SM budgeting) is not supported in Phase 1; the kernel uses "
            "all available SMs."
        )
    assert a.dim() == 2, f"a must be (M_total, K), got {tuple(a.shape)}"
    assert group_sizes.dim() == 1, f"group_sizes must be (G,), got {tuple(group_sizes.shape)}"

    # tokens-per-group -> cumulative end offsets, on device (no host sync -> capturable).
    offs = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)

    if isinstance(b, PreparedGroupedWeights):
        if scale_b is not None:
            raise ValueError(
                "scale_b must be None when b is PreparedGroupedWeights (B-scale is pre-packed)."
            )
        return b._gemm(a, offs, scale_a, out=out, varlen_nonaligned=True)

    assert b.dim() == 3, f"b must be (G, N, K), got {tuple(b.shape)}"
    # (G, N, K) -> (G, K, N): quack's b operand is K-contiguous per expert.
    b_gkn = b.transpose(-2, -1)
    return mxfp8_grouped_gemm(a, b_gkn, offs, scale_a, scale_b, out=out, varlen_nonaligned=True)


__all__ = [
    "scaled_grouped_mm",
    "prepare_weights",
    "PreparedGroupedWeights",
    "GatherPrologue",
    "OffsetPrologue",
    "ScatterEpilogue",
    "ScatterSwiGLUEpilogue",
    "OffsetEpilogue",
    "KernelConfig",
]
