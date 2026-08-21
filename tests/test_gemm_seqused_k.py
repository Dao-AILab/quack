# Copyright (C) 2025, Tri Dao.
"""Tests for the optional decoupled real-K-length `seqused_k` on varlen_k GEMM.

The strong test poisons every padded row with NaN and asserts the output is
both finite and equal to a real-rows-only reference — so it fails unless BOTH
mechanisms work: the reduced per-batch K-tile count (`len_k` -> seqused_k[b])
AND the TMA OOB-fill bound (`offset_batch_A/B` length -> seqused_k[b]) that
zero-fills the straddling last K-tile.
"""

import math
import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm as quack_gemm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] != 9,
    reason="seqused_k varlen_k GEMM test targets SM90",
)

# Proven SM90 varlen_k config (see run_lowlevel_varlen_k_gemm in
# test_linear_varlen_k.py). tile_K auto-derives to 64 for bf16.
TILE_M, TILE_N = 256, 256
CLUSTER_M, CLUSTER_N = 2, 1
DTYPE = torch.bfloat16
ATOL, RTOL = 3e-2, 1e-3
K_PAD = 128


def _run_gemm(A, B, D, **kwargs):
    quack_gemm(
        A,
        B,
        D,
        C=None,
        tile_count_semaphore=None,
        tile_M=TILE_M,
        tile_N=TILE_N,
        cluster_M=CLUSTER_M,
        cluster_N=CLUSTER_N,
        persistent=True,
        **kwargs,
    )


def _make_padded_operands(m, n, real_k, pad_fill):
    """Build m-major A (m, total_pad_k) and n-major B (n, total_pad_k) whose
    per-group padded columns [cu[i] + real_k[i] : cu[i+1]) are set to `pad_fill`.

    Returns A, B, cu_seqlens_k (int32, padded offsets), real_k_t (int32).
    """
    device = "cuda"
    pad_k = [((rk + K_PAD - 1) // K_PAD) * K_PAD for rk in real_k]  # tile-aligned slot
    cu = [0]
    for p in pad_k:
        cu.append(cu[-1] + p)
    total_pad_k = cu[-1]
    avg_k = total_pad_k / len(real_k)

    # m-major A: contiguous (total_pad_k, m) then .T => stride(-2)==1.
    A = (torch.randn(total_pad_k, m, device=device, dtype=DTYPE) / math.sqrt(avg_k)).T
    # n-major B: contiguous (total_pad_k, n) then .T => stride(-2)==1.
    B = (torch.randn(total_pad_k, n, device=device, dtype=DTYPE) / math.sqrt(avg_k)).T
    for i, rk in enumerate(real_k):
        pad_cols = slice(cu[i] + rk, cu[i + 1])
        A[:, pad_cols] = pad_fill
        B[:, pad_cols] = pad_fill
    cu_seqlens_k = torch.tensor(cu, dtype=torch.int32, device=device)
    real_k_t = torch.tensor(real_k, dtype=torch.int32, device=device)
    return A, B, cu_seqlens_k, real_k_t


def _real_rows_ref(A, B, cu_seqlens_k, real_k):
    """Reference: per group, contract only over the real rows of each slot."""
    cu = cu_seqlens_k.tolist()
    return torch.stack(
        [
            A[:, cu[i] : cu[i] + rk].float() @ B[:, cu[i] : cu[i] + rk].float().T
            for i, rk in enumerate(real_k)
        ]
    ).to(DTYPE)


# real_k deliberately not tile_K(64)-aligned so the last real K-tile straddles
# real/pad. Mix: group 0 keeps the same tile count as the padded slot (only the
# OOB-fill length differs); groups 1,2 drop whole K-tiles (seqused reduces the
# count). pad slots become [256, 256, 384].
REAL_K = [200, 137, 264]


@pytest.mark.parametrize("n", [512])
@pytest.mark.parametrize("m", [1024])
def test_seqused_k_excludes_nan_padding(m, n):
    """NaN-poisoned padded rows + seqused_k => finite output matching real-only ref."""
    torch.manual_seed(0)
    A, B, cu_seqlens_k, real_k = _make_padded_operands(m, n, REAL_K, pad_fill=float("nan"))
    D = torch.empty(len(REAL_K), m, n, dtype=DTYPE, device="cuda")

    _run_gemm(A, B, D, cu_seqlens_k=cu_seqlens_k, seqused_k=real_k)

    assert torch.isfinite(D).all(), "NaN leaked from padded rows -> seqused_k OOB-fill broken"
    ref = _real_rows_ref(A, B, cu_seqlens_k, REAL_K)
    torch.testing.assert_close(D, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("n", [512])
@pytest.mark.parametrize("m", [1024])
def test_seqused_k_none_zero_pad_matches(m, n):
    """Backward-compat: seqused_k=None over zero-padded slots contracts the full
    slot but the zero pads contribute nothing, so it equals the real-only ref.
    Confirms the existing coupled-varlen_k path is unchanged."""
    torch.manual_seed(0)
    A, B, cu_seqlens_k, _ = _make_padded_operands(m, n, REAL_K, pad_fill=0.0)
    D = torch.empty(len(REAL_K), m, n, dtype=DTYPE, device="cuda")

    _run_gemm(A, B, D, cu_seqlens_k=cu_seqlens_k)  # no seqused_k

    ref = _real_rows_ref(A, B, cu_seqlens_k, REAL_K)
    torch.testing.assert_close(D, ref, atol=ATOL, rtol=RTOL)
