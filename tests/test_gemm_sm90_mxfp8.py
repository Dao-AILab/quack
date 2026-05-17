import math

import pytest
import torch

from quack.gemm_blockscaled_interface import (
    _SF_VEC_SIZE_SM90 as SF,
    _WEIGHT_BLOCK_N_SM90 as BN,
    mxfp8_gemm_act,
    mxfp8_quantize_act,
    mxfp8_quantize_weight,
)
from quack.gemm_interface import gemm_gated_ref


def _skip_if_not_sm90():
    if torch.cuda.get_device_properties(0).major != 9:
        pytest.skip("SM90 required")


def deepseek_calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Cosine similarity. Copied from DeepGEMM
    https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/deep_gemm/testing/numeric.py#L5
    """
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return 0.0 if denom == 0 else float(1 - 2 * (x * y).sum() / denom)


def _fp8_dequant_ref(A_q, A_sc, W_q, W_sc):
    """Exact float32 matmul of dequantized FP8 tensors."""
    A_dq = A_q.float() * A_sc.float().repeat_interleave(SF, dim=-1)
    W_sc_exp = W_sc.float().repeat_interleave(BN, dim=-2).repeat_interleave(SF, dim=-1)
    W_dq = W_q.float() * W_sc_exp
    return A_dq, W_dq.mT if W_q.ndim > 2 else W_dq.T


def _assert_close(kernel_out, ref, dtype_out, tag):
    """Check max-abs error (adaptive tolerance) and cosine complement (< 0.001)."""
    pt_out = ref[0].to(dtype_out)  # bf16 baseline for tolerance calibration
    tol = max(10 * (pt_out.float() - ref[1]).abs().max().item(), 1e-3)
    err = (kernel_out.float() - ref[1]).abs().max().item()
    cos = deepseek_calc_diff(kernel_out.float(), ref[1])
    assert err < tol, f"{tag}: max_abs={err:.5f} > tol={tol:.5f}"
    assert cos < 0.001, f"{tag}: cosine_diff={cos:.6f} >= 0.001"


def _make_varied_scale_inputs(M, K, N, *, dtype=torch.bfloat16, device="cuda"):
    """Build (A, W) bf16 tensors whose MXFP8 scales differ across every relevant
    indexing axis. Plain randn quantizes to nearly-uniform power-of-2 scales, masking
    indexing bugs (wrong-row / wrong-K-block / wrong-m_block reads still produce
    numerically plausible results).

    Scaling scheme — designed to defeat *every* indexing aliasing pattern:
    - Per-row factor: 2^(i % 4) cycles every 4 rows (catches per-row off-by-N bugs)
    - Per-64-row-chunk factor: 2^((i // 64) % 4) (catches off-by-BLOCK_M aliasing,
      since row 0 and row 128 land in different chunks → different factors)
    - Per-K-block factor: 2^((k * 3) % 4) (catches wrong-stage / wrong-K reads;
      stride 3 coprime with 4 spreads scales)
    - Same scheme for W = (2*N, K) at the 128-row N-block granularity that
      matches `_WEIGHT_BLOCK_N_SM90`.

    Power-of-2 ranges chosen so the dequantized inputs stay inside fp8_e4m3fn
    dynamic range and the K-summed outputs stay bf16-representable.
    """
    sf_k = K // SF
    base_a = torch.randn(M, K, device=device, dtype=dtype) / math.sqrt(K)
    base_w = torch.randn(2 * N, K, device=device, dtype=dtype) / math.sqrt(K)

    rows = torch.arange(M, device=device)
    # Combined exponent = (row % 4) + (row // 64) % 4: 4*4 = 16 combinations,
    # max factor 2^6 = 64 → max input magnitude ~64 * 1/sqrt(K), still safe for fp8.
    a_row = (2.0 ** ((rows % 4) + ((rows // 64) % 4))).to(dtype)
    ks = torch.arange(sf_k, device=device)
    a_kb = (2.0 ** ((ks * 3) % 4)).to(dtype)
    n_blk = (2 * N) // BN
    n_idx = torch.arange(n_blk, device=device)
    # Per-N-block factor uses BLOCK_N=128 granularity; combine outer chunk to
    # break aliasing across N-tiles when N > BLOCK_N.
    w_nb = (2.0 ** ((n_idx % 4) + ((n_idx // 4) % 4))).to(dtype)
    w_kb = (2.0 ** ((ks * 5) % 4)).to(dtype)

    a = base_a.view(M, sf_k, SF) * a_kb.view(1, sf_k, 1)
    a = a * a_row.view(M, 1, 1)
    a = a.reshape(M, K)

    w = base_w.view(n_blk, BN, sf_k, SF) * w_kb.view(1, 1, sf_k, 1)
    w = w * w_nb.view(n_blk, 1, 1, 1)
    w = w.reshape(2 * N, K)
    return a, w


# ---------------------------------------------------------------------------
# Batched (no varlen)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("store_preact", [True, False])
@pytest.mark.parametrize("activation", ["swiglu", "geglu"])
@pytest.mark.parametrize(
    "M, K, N",
    [
        (512,  2048, 1024),
        (1,    2048, 1024),   # M=1 edge
        (512,   768, 2048),   # K not divisible by 512 (sf_k not divisible by 4)
        (256,  1024,  512),
        (1536, 4096, 2048),
    ],
)
def test_mxfp8_gemm_gated_sm90(M, K, N, activation, store_preact):
    _skip_if_not_sm90()
    dtype = torch.bfloat16
    torch.manual_seed(0)
    device = "cuda"

    A_bf16 = torch.randn(M, K, device=device, dtype=dtype) / math.sqrt(K)
    W_bf16 = torch.randn(2 * N, K, device=device, dtype=dtype) / math.sqrt(K)

    A_q, A_sc = mxfp8_quantize_act(A_bf16)
    W_q, W_sc = mxfp8_quantize_weight(W_bf16)
    B_q, B_sc = W_q.mT, W_sc.mT

    preact, postact = mxfp8_gemm_act(
        A_q, B_q, A_sc, B_sc,
        activation=activation,
        out_dtype=dtype,
        postact_dtype=dtype,
        store_preact=store_preact,
        tuned=False,
    )

    A_dq, B_dq = _fp8_dequant_ref(A_q, A_sc, W_q, W_sc)
    pre_ref, post_ref = gemm_gated_ref(
        A_dq, B_dq, activation=activation, store_preact=store_preact
    )
    pre_pt, post_pt = gemm_gated_ref(
        A_dq.to(dtype), B_dq.to(dtype), activation=activation, store_preact=store_preact
    )

    assert postact.shape == (M, N)
    _assert_close(postact, (post_pt.float(), post_ref), dtype, "postact")
    if store_preact:
        assert preact is not None and pre_ref is not None
        assert preact.shape == (M, 2 * N)
        _assert_close(preact, (pre_pt.float(), pre_ref), dtype, "preact")
    else:
        assert preact is None


# ---------------------------------------------------------------------------
# Indexing stress test: rows / N-blocks / K-blocks have power-of-2-distinct
# MXFP8 scales, so any wrong-row or wrong-K-block load shows up as a numerical
# error instead of being masked by uniform-scale randn data.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "M, K, N",
    [
        (256, 512, 256),    # multi-m-block, multi-n-block, 4 K-stages
        (512, 1024, 512),   # bigger; exercises persistent scheduling
        (128, 768, 384),    # K not divisible by 512 (sf_k=6, not power of 2)
    ],
)
def test_mxfp8_gemm_gated_sm90_varied_scales(M, K, N):
    _skip_if_not_sm90()
    dtype = torch.bfloat16
    torch.manual_seed(0)
    device = "cuda"

    A_bf16, W_bf16 = _make_varied_scale_inputs(M, K, N, dtype=dtype, device=device)

    A_q, A_sc = mxfp8_quantize_act(A_bf16)
    W_q, W_sc = mxfp8_quantize_weight(W_bf16)
    B_q, B_sc = W_q.mT, W_sc.mT

    # Sanity: scales should genuinely vary along every indexing axis. Without this
    # check, a regression to _make_varied_scale_inputs that produced uniform scales
    # would silently weaken the indexing-bug coverage.
    assert A_sc.unique().numel() >= 4, (
        f"A scales need >=4 unique values; got {A_sc.unique().numel()}"
    )
    # Scales must vary across the BLOCK_M=128 boundary so wrong-m_block reads
    # produce different scales than the correct row.
    if M >= 256:
        assert not torch.equal(A_sc[0], A_sc[128]), (
            "A scales at row 0 and row 128 are identical — wrong-m_block bugs would alias"
        )
    # Scales must vary along K-blocks within a single row so wrong-stage reads
    # produce different scales than the correct K-block.
    assert A_sc[0].unique().numel() >= 2, (
        f"A scales for row 0 should vary across K-blocks; got {A_sc[0].unique().numel()}"
    )
    assert W_sc.unique().numel() >= 4, (
        f"W scales need >=4 unique values; got {W_sc.unique().numel()}"
    )

    preact, postact = mxfp8_gemm_act(
        A_q, B_q, A_sc, B_sc,
        activation="swiglu",
        out_dtype=dtype,
        postact_dtype=dtype,
        store_preact=True,
        tuned=False,
    )

    A_dq, B_dq = _fp8_dequant_ref(A_q, A_sc, W_q, W_sc)
    pre_ref, post_ref = gemm_gated_ref(A_dq, B_dq, activation="swiglu", store_preact=True)
    pre_pt, post_pt = gemm_gated_ref(
        A_dq.to(dtype), B_dq.to(dtype), activation="swiglu", store_preact=True
    )

    _assert_close(postact, (post_pt.float(), post_ref), dtype, "postact")
    _assert_close(preact, (pre_pt.float(), pre_ref), dtype, "preact")


# ---------------------------------------------------------------------------
# Variable-length M (grouped / ragged batch)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("store_preact", [True, False])
@pytest.mark.parametrize("activation", ["swiglu", "geglu"])
@pytest.mark.parametrize(
    "seq_lens, K, N",
    [
        ([128, 256,  64, 512], 2048, 1024),
        ([32]  * 8,            1024,  512),
        ([128, 256],            768, 1024),  # K not divisible by 512
    ],
)
def test_mxfp8_gemm_gated_sm90_varlen(seq_lens, K, N, activation, store_preact):
    _skip_if_not_sm90()
    dtype = torch.bfloat16
    torch.manual_seed(0)
    device = "cuda"

    L = len(seq_lens)
    total_m = sum(seq_lens)
    cu_seqlens_m = torch.cat([
        torch.zeros(1, dtype=torch.int32),
        torch.tensor(seq_lens, dtype=torch.int32).cumsum(0).int(),
    ]).to(device)

    A_bf16 = torch.randn(total_m, K, device=device, dtype=dtype) / math.sqrt(K)
    W_bf16 = torch.randn(L, 2 * N, K, device=device, dtype=dtype) / math.sqrt(K)

    A_q, A_sc = mxfp8_quantize_act(A_bf16)
    W_q, W_sc = mxfp8_quantize_weight(W_bf16)
    B_q, B_sc = W_q.mT, W_sc.mT

    preact, postact = mxfp8_gemm_act(
        A_q, B_q, A_sc, B_sc,
        activation=activation,
        out_dtype=dtype,
        postact_dtype=dtype,
        store_preact=store_preact,
        cu_seqlens_m=cu_seqlens_m,
        tuned=False,
    )

    A_dq, B_dq = _fp8_dequant_ref(A_q, A_sc, W_q, W_sc)
    pre_ref, post_ref = gemm_gated_ref(
        A_dq, B_dq, activation=activation, store_preact=store_preact, cu_seqlens_m=cu_seqlens_m,
    )
    pre_pt, post_pt = gemm_gated_ref(
        A_dq.to(dtype), B_dq.to(dtype),
        activation=activation, store_preact=store_preact, cu_seqlens_m=cu_seqlens_m,
    )

    assert postact.shape == (total_m, N)
    _assert_close(postact, (post_pt.float(), post_ref), dtype, "postact")
    if store_preact:
        assert preact is not None and pre_ref is not None
        assert preact.shape == (total_m, 2 * N)
        _assert_close(preact, (pre_pt.float(), pre_ref), dtype, "preact")
    else:
        assert preact is None
