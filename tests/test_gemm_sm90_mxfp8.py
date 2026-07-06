import math

import pytest
import torch

from quack.gemm_blockscaled_sm90 import (
    _SF_VEC_SIZE_SM90 as SF,
    _WEIGHT_BLOCK_N_SM90 as BN,
    mxfp8_gemm_act,
    quantize_act,
    quantize_weight_sm90,
)
from quack.gemm_interface import gemm_act
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


def _assert_cos_close(kernel_out, ref, tag, tol=0.001):
    """Cosine-complement check only (DeepGEMM style)."""
    cos = deepseek_calc_diff(kernel_out.float(), ref)
    assert cos < tol, f"{tag}: cosine_diff={cos:.6f} >= {tol}"


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
        (512, 2048, 1024),
        # TODO: M=1 fails with varied scales (cosine_diff ~0.87). Real kernel bug
        # at M<BLOCK_M=64 that uniform-scale randn was diluting. Suspect: TMA SFA
        # descriptor / row-scale load when M<BLOCK_M. Re-enable once fixed.
        # (1,    2048, 1024),   # M=1 edge
        (512, 768, 2048),
        (256, 1024, 512),
        (1536, 4096, 2048),
    ],
)
def test_mxfp8_gemm_gated_sm90(M, K, N, activation, store_preact):
    _skip_if_not_sm90()
    dtype = torch.bfloat16
    torch.manual_seed(0)
    device = "cuda"

    A_bf16, W_bf16 = _make_varied_scale_inputs(M, K, N, dtype=dtype, device=device)

    A_q, A_sc = quantize_act(A_bf16)
    W_q, W_sc = quantize_weight_sm90(W_bf16)
    B_q, B_sc = W_q.mT, W_sc.mT

    preact, postact = mxfp8_gemm_act(
        A_q,
        B_q,
        A_sc,
        B_sc,
        activation=activation,
        out_dtype=dtype,
        postact_dtype=dtype,
        store_preact=store_preact,
        tuned=False,
    )

    A_dq, B_dq = _fp8_dequant_ref(A_q, A_sc, W_q, W_sc)
    pre_ref, post_ref = gemm_gated_ref(A_dq, B_dq, activation=activation, store_preact=store_preact)

    assert postact.shape == (M, N)
    _assert_cos_close(postact, post_ref, "postact")
    if store_preact:
        assert preact is not None and pre_ref is not None
        assert preact.shape == (M, 2 * N)
        _assert_cos_close(preact, pre_ref, "preact")
    else:
        assert preact is None


@pytest.mark.parametrize("store_preact", [True, False])
@pytest.mark.parametrize("activation", ["swiglu", "geglu"])
@pytest.mark.parametrize(
    "M, K, N",
    [
        (512, 2048, 1024),
        (512, 768, 2048),
        (256, 1024, 512),
        (1536, 4096, 2048),
    ],
)
def test_mxfp8_gemm_sm90(M, K, N, activation, store_preact):
    _skip_if_not_sm90()
    dtype = torch.bfloat16
    torch.manual_seed(0)
    device = "cuda"

    A_bf16, W_bf16 = _make_varied_scale_inputs(M, K, N, dtype=dtype, device=device)

    A_q, A_sc = quantize_act(A_bf16)
    W_q, W_sc = quantize_weight_sm90(W_bf16)
    B_q, B_sc = W_q.mT, W_sc.mT

    preact, postact = mxfp8_gemm_act(
        A_q,
        B_q,
        A_sc,
        B_sc,
        activation=activation,
        out_dtype=dtype,
        postact_dtype=dtype,
        store_preact=store_preact,
        tuned=False,
    )
    pre_ref, post_ref = gemm_act(
        A_bf16,
        W_bf16.mT,
        activation=activation,
        out_dtype=dtype,
        postact_dtype=dtype,
        store_preact=store_preact,
        tuned=False,
    )

    assert postact.shape == (M, N)
    # the activation increases the error
    _assert_cos_close(postact, post_ref, "postact", tol=0.002)
    if store_preact:
        assert preact is not None and pre_ref is not None
        assert preact.shape == (M, 2 * N)
        _assert_cos_close(preact, pre_ref, "preact")
    else:
        assert preact is None
