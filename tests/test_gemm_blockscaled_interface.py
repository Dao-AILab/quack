# Copyright (c) 2026, Tri Dao.
"""Tests for the unified blockscaled GEMM interface: gemm((A, SFA), (B, SFB)).

Layout contract (see quack/gemm_interface.py):
  A:   (M, K) or (L, M, K)   fp8 e4m3 (mxfp8) or packed fp4x2 (mxfp4/nvfp4, K/2 bytes)
  B:   (K, N) or (L, K, N)   same dtype as A, K-contiguous (pass W.mT of an (N, K) weight)
  SF:  (rm, rk, 32, 4, 4) or (L, rm, rk, 32, 4, 4) with rm = ceil(rows / 128),
       rk = ceil(K / VEC / 4); inner block strides (16, 4, 1) (one contiguous 512 B atom).
       e8m0 (VEC=32, mx formats) or e4m3 (VEC=16, nvfp4).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from quack.blockscaled.utils import blockscaled_quantize, scale_blocked_for_cublas
from quack.gemm_interface import (
    gemm,
    gemm_act,
    gemm_add,
    gemm_add_inplace,
    gemm_blockscaled_ref,
)


def _skip_if_not_sm100():
    major = torch.cuda.get_device_properties(0).major
    if major < 10:
        pytest.skip("SM100+ required")


def _quantized_operands(fmt, m, n, k, batched, seed=0):
    torch.manual_seed(seed)
    L = 2 if batched else 1
    shape_a = (L, m, k) if batched else (m, k)
    shape_w = (L, n, k) if batched else (n, k)
    a_hp = torch.randn(*shape_a, device="cuda", dtype=torch.bfloat16) * k**-0.5
    w_hp = torch.randn(*shape_w, device="cuda", dtype=torch.bfloat16) * k**-0.5
    qa, sfa = blockscaled_quantize(a_hp, fmt)
    qw, sfw = blockscaled_quantize(w_hp, fmt)
    return (qa, sfa), (qw.mT, sfw)  # B = (K, N) K-contig view (or (K/2, N) for fp4)


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "shape_mnk",
    [
        (256, 256, 256),
        (512, 512, 512),
        (128, 128, 256),
        (448, 320, 512),  # M, N not multiples of 128 (padded SF rows)
        (1024, 256, 8192),
    ],
)
def test_blockscaled_gemm(fmt, batched, shape_mnk):
    _skip_if_not_sm100()
    m, n, k = shape_mnk
    A, B = _quantized_operands(fmt, m, n, k, batched)
    out = gemm(A, B, tuned=False)
    ref = gemm_blockscaled_ref(A, B)
    expected_shape = (2, m, n) if batched else (m, n)
    assert out.shape == expected_shape and out.dtype == torch.bfloat16
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    assert rel < 5e-3, f"{fmt} {shape_mnk} batched={batched}: rel_err={rel}"


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_blockscaled_gemm_out_dtype(out_dtype):
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    A, B = _quantized_operands("mxfp8", m, n, k, batched=False)
    out = gemm(A, B, out_dtype=out_dtype, tuned=False)
    ref = gemm_blockscaled_ref(A, B, out_dtype=out_dtype)
    assert out.dtype == out_dtype
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    assert rel < 5e-3, f"out_dtype={out_dtype}: rel_err={rel}"


def test_blockscaled_gemm_alpha_and_preallocated_out():
    _skip_if_not_sm100()
    # nvfp4's per-tensor global scales fold into alpha
    m, n, k = 512, 256, 512
    A, B = _quantized_operands("nvfp4", m, n, k, batched=False)
    alpha = 0.125
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    ret = gemm(A, B, out=out, alpha=alpha, tuned=False)
    assert ret is out
    ref = gemm_blockscaled_ref(A, B, alpha=alpha)
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    assert rel < 5e-3, f"alpha={alpha}: rel_err={rel}"


def test_blockscaled_gemm_sf_slice():
    """Atom-aligned slices of larger SF buffers work (outer strides are free)."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    (qa, sfa), (B, sfw) = _quantized_operands("mxfp8", m, n, k, batched=False)
    out_ref = gemm((qa, sfa), (B, sfw), tuned=False)
    # SFA living inside a larger buffer (extra rk columns), sliced back out
    rm, rk = sfa.shape[0], sfa.shape[1]
    big = torch.zeros(rm, rk + 3, 32, 4, 4, device="cuda", dtype=sfa.dtype)
    big[:, :rk] = sfa
    out_slice = gemm((qa, big[:, :rk]), (B, sfw), tuned=False)
    assert torch.equal(out_ref, out_slice)


@pytest.mark.parametrize("a_major", ["k", "m"])
@pytest.mark.parametrize("b_major", ["k", "n"])
def test_blockscaled_gemm_major_modes(a_major, b_major):
    """MXFP8 with A in {k,m}-major x B in {k,n}-major through the public interface.

    Majorness is inferred from operand strides; the SF tensors are identical in
    all four cases (the (rm, rk, 32, 4, 4) hardware format does not depend on
    how the operand values are stored).
    """
    _skip_if_not_sm100()
    m, n, k = 256, 320, 512  # n not a multiple of 128 (padded SF rows)
    (qa, sfa), (B, sfw) = _quantized_operands("mxfp8", m, n, k, batched=False)
    ref = gemm_blockscaled_ref((qa, sfa), (B, sfw))
    if a_major == "m":
        qa = qa.t().contiguous().t()  # (m, k) with M contiguous
        assert qa.stride() == (1, m)
    if b_major == "n":
        B = B.contiguous()  # (k, n) with N contiguous -> (n, k) n-major after .mT
        assert B.stride() == (n, 1)
    out = gemm((qa, sfa), (B, sfw), tuned=False)
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    assert rel < 5e-3, f"A={a_major}-major B={b_major}-major: rel_err={rel}"


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("seqlens_m", [[128, 128, 128], [100, 200, 150], [1, 128, 127, 129]])
def test_blockscaled_gemm_varlen_m(seqlens_m, fmt):
    """Grouped (varlen_m) blockscaled GEMM through the unified interface. SFA is a
    single M-padded buffer (tile-aligned per-batch padding, batch dim 1); SFB stays per-expert."""
    _skip_if_not_sm100()
    import cutlass

    from quack.blockscaled.utils import create_blockscaled_varlen_m_operands

    fmt_map = {
        "mxfp8": (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
        "mxfp4": (cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
        "nvfp4": (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
    }
    ab_dtype, sf_dtype, sf_vec = fmt_map[fmt]
    num_experts = len(seqlens_m)
    n, k = 256, 256
    torch.manual_seed(0)
    a_ref_dq, b_ref_dq, qa, qb, a_sc_contig, b_sc_contig, cu_seqlens_m = (
        create_blockscaled_varlen_m_operands(
            num_experts, 0, n, k, sf_vec, ab_dtype, sf_dtype, seqlens_m=seqlens_m
        )
    )
    SFA, SFB = a_sc_contig, b_sc_contig  # (1, total_padded_rm, rk, 32, 4, 4), (L, rn, rk, 32, 4, 4)
    B = qb.permute(2, 1, 0)  # (n, k[/2], L) -> (L, K[/2], N) with K contiguous
    out = gemm((qa, SFA), (B, SFB), cu_seqlens_m=cu_seqlens_m, tuned=False)

    cu = cu_seqlens_m.tolist()
    ref = torch.cat([a_ref_dq[cu[i] : cu[i + 1]] @ b_ref_dq[i].T for i in range(num_experts)])
    err = (out.float() - ref).abs().max().item()
    assert err < 5e-3, f"varlen_m {fmt} seqlens_m={seqlens_m} max_err={err}"


@pytest.mark.parametrize("seqlens_k", [[128, 128, 128], [96, 160, 128], [100, 220, 65]])
def test_blockscaled_gemm_varlen_k(seqlens_k):
    """Grouped varlen_k blockscaled GEMM through the unified interface (MXFP8 only:
    fp4 must be K-major, varlen_k needs m-major A / n-major B). Both SFA and SFB
    are single K-padded buffers (tile-aligned per-batch padding, batch dim 1).
    Per-expert k_i is arbitrary — not even sf_vec(32)-aligned."""
    _skip_if_not_sm100()
    from quack.blockscaled.utils import create_blockscaled_varlen_k_operands

    num_experts = len(seqlens_k)
    m, n, sf_vec = 256, 256, 32
    torch.manual_seed(0)
    a_ref_list, b_ref_list, qa, qb, SFA, SFB, cu_seqlens_k = create_blockscaled_varlen_k_operands(
        num_experts, 0, m, n, sf_vec, seqlens_k=seqlens_k
    )
    # Interface takes B as (total_K, N); qb is (n, total_k) n-major so .t() is
    # the (total_k, n) row-major view and gemm_tuned's .mT restores n-major.
    out = gemm((qa, SFA), (qb.t(), SFB), cu_seqlens_k=cu_seqlens_k, tuned=False)
    assert out.shape == (num_experts, m, n)

    ref = torch.stack([a_ref_list[i] @ b_ref_list[i].T for i in range(num_experts)])
    err = (out.float() - ref).abs().max().item()
    assert err < 5e-3, f"varlen_k seqlens_k={seqlens_k} max_err={err}"


def test_blockscaled_gemm_vs_cublas():
    """Bit-exact comparison against torch._scaled_mm (cuBLAS MXFP8 path)."""
    _skip_if_not_sm100()
    m, n, k = 512, 512, 512
    (qa, sfa), (B, sfw) = _quantized_operands("mxfp8", m, n, k, batched=False)
    out = gemm((qa, sfa), (B, sfw), tuned=False)
    sfa_flat = scale_blocked_for_cublas(sfa.unsqueeze(0), m, k // 32)
    sfw_flat = scale_blocked_for_cublas(sfw.unsqueeze(0), n, k // 32)
    out_cublas = torch._scaled_mm(
        qa, B, scale_a=sfa_flat, scale_b=sfw_flat, out_dtype=torch.bfloat16
    )
    assert torch.equal(out, out_cublas), (
        f"quack != cuBLAS: max_err={(out.float() - out_cublas.float()).abs().max().item()}"
    )


def test_blockscaled_gemm_bad_sf_layout_rejected():
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    (qa, sfa), (B, sfw) = _quantized_operands("mxfp8", m, n, k, batched=False)
    # non-contiguous inner block (transposed (4, 4) tail) must be rejected
    bad = sfa.transpose(-1, -2)
    with pytest.raises(AssertionError, match="inner"):
        gemm((qa, bad), (B, sfw), tuned=False)
    # a flattened (rm, rk, 512) form is not part of the interface contract
    with pytest.raises(AssertionError, match="32, 4, 4"):
        gemm((qa, sfa.flatten(-3)), (B, sfw.flatten(-3)), tuned=False)
    # SF on only one operand must be rejected
    with pytest.raises(AssertionError, match="both"):
        gemm((qa, sfa), B, tuned=False)


def test_blockscaled_gemm_torch_compile():
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    A, B = _quantized_operands("mxfp8", m, n, k, batched=False)
    ref = gemm(A, B, tuned=False)

    @torch.compile(dynamic=False)
    def f(qa, sfa, b, sfw):
        return gemm((qa, sfa), (b, sfw), tuned=False)

    out = f(A[0], A[1], B[0], B[1])
    assert torch.equal(out, ref)


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("batched", [False, True])
def test_blockscaled_gemm_add(fmt, batched):
    """D = alpha * A@B + beta * C with blockscaled A/B."""
    _skip_if_not_sm100()
    m, n, k = 512, 256, 512
    A, B = _quantized_operands(fmt, m, n, k, batched)
    ref_mm = gemm_blockscaled_ref(A, B, out_dtype=torch.float32)
    c_shape = (2, m, n) if batched else (m, n)
    C = torch.randn(c_shape, dtype=torch.bfloat16, device="cuda")

    # The C addend makes |out| ~ 4-5, so bf16 resolution (1 ulp at max magnitude)
    # exceeds a max-normalized 5e-3; bound the error by 2 bf16 ulp at max instead.
    def _max_err_within_2ulp(out, ref, what):
        err = (out.float() - ref.float()).abs().max().item()
        ulp = 2.0 ** (math.floor(math.log2(ref.float().abs().max().item())) - 7)
        assert err <= 2 * ulp, f"{fmt} {what} batched={batched}: max_err={err} > 2*ulp={2 * ulp}"

    alpha, beta = 0.5, 2.0
    out = gemm_add(A, B, C, alpha=alpha, beta=beta, tuned=False)
    ref = (alpha * ref_mm + beta * C.float()).to(torch.bfloat16)
    _max_err_within_2ulp(out, ref, "gemm_add")
    # In-place accumulate (add_to_output path, beta=1)
    acc = C.clone()
    gemm_add_inplace(A, B, acc, tuned=False)
    ref2 = (ref_mm + C.float()).to(torch.bfloat16)
    _max_err_within_2ulp(acc, ref2, "gemm_add_inplace")


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("activation", ["relu_sq", "gelu_tanh_approx", "swiglu"])
def test_blockscaled_gemm_act(fmt, activation):
    """gemm_act / gemm_gated with blockscaled A/B, checked against the dequant reference."""
    _skip_if_not_sm100()
    m, n, k = 512, 512, 512
    A, B = _quantized_operands(fmt, m, n, k, batched=False)
    ref_mm = gemm_blockscaled_ref(A, B, out_dtype=torch.float32)
    preact, postact = gemm_act(A, B, activation=activation, tuned=False)
    assert preact.dtype == torch.bfloat16 and postact.dtype == torch.bfloat16
    if activation == "swiglu":
        gate, up = ref_mm[..., ::2], ref_mm[..., 1::2]
        ref_post = (F.silu(gate) * up).to(torch.bfloat16)
        assert postact.shape == (m, n // 2)
    else:
        act_fn = {
            "relu_sq": lambda x: F.relu(x).square(),
            "gelu_tanh_approx": lambda x: F.gelu(x, approximate="tanh"),
        }[activation]
        ref_post = act_fn(ref_mm).to(torch.bfloat16)
        assert postact.shape == (m, n)
    rel_pre = (preact.float() - ref_mm).abs().max().item() / ref_mm.abs().max().item()
    denom = ref_post.float().abs().max().item() + 1e-9
    rel_post = (postact.float() - ref_post.float()).abs().max().item() / denom
    assert rel_pre < 5e-3, f"{fmt} {activation}: preact rel_err={rel_pre}"
    assert rel_post < 1e-2, f"{fmt} {activation}: postact rel_err={rel_post}"


def test_blockscaled_gemm_act_bias_no_preact():
    """Bias broadcast + store_preact=False on the blockscaled act path."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    A, B = _quantized_operands("mxfp8", m, n, k, batched=False)
    bias = torch.randn(n, dtype=torch.float32, device="cuda")
    ref_mm = gemm_blockscaled_ref(A, B, out_dtype=torch.float32)
    preact, postact = gemm_act(A, B, bias=bias, activation="relu", store_preact=False, tuned=False)
    assert preact is None
    ref_post = F.relu(ref_mm + bias).to(torch.bfloat16)
    denom = ref_post.float().abs().max().item() + 1e-9
    rel = (postact.float() - ref_post.float()).abs().max().item() / denom
    assert rel < 1e-2, f"bias+relu: rel_err={rel}"


@pytest.mark.parametrize("fmt", ["mxfp8", "nvfp4"])
def test_blockscaled_gemm_tuned(fmt):
    """Autotuned path: config pruning + sweep must produce a correct result."""
    _skip_if_not_sm100()
    m, n, k = 512, 512, 512
    A, B = _quantized_operands(fmt, m, n, k, batched=False)
    out = gemm(A, B, tuned=True)
    ref = gemm_blockscaled_ref(A, B)
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    assert rel < 5e-3, f"{fmt} tuned: rel_err={rel}"
