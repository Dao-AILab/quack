# Copyright (c) 2026, Tri Dao.
"""Tests for the unified blockscaled GEMM interface.

Operands are BlockScaledOperand containers - the only accepted blockscaled
operand form ((data, scale_factor) tuples are rejected, see
test_tuple_operand_rejected).

Layout contract (see quack/gemm_interface.py and AI/blockscaled_api.md):
  A:   (M, K) or (L, M, K)   fp8 e4m3 (mxfp8) or packed fp4x2 (mxfp4/nvfp4, K/2 bytes)
  B:   (K, N) or (L, K, N)   same dtype as A, K-contiguous (pass W.mT of an (N, K) weight)
  SF:  (rm, rk, 32, 4, 4) or (L, rm, rk, 32, 4, 4) with rm = ceil(rows / 128),
       rk = ceil(K / VEC / 4); inner block strides (16, 4, 1) (one contiguous 512 B atom).
       All format properties come from the BlockScaledFormat descriptor.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from quack.blockscaled.quantize import nvfp4_per_tensor_scale
from quack.blockscaled.operand import BlockScaledFormat, BlockScaledOperand
from quack.blockscaled.utils import blockscaled_quantize, scale_blocked_for_cublas
from quack.blockscaled.utils import blockscaled_quantize_dim0
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
    fmt_obj = BlockScaledFormat.from_name(fmt)
    A = BlockScaledOperand.from_parts(qa, sfa, fmt_obj)
    W = BlockScaledOperand.from_parts(qw, sfw, fmt_obj)
    return A, W.mT  # B = (K, N) logical view; qdata stride-swap, scale unchanged


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
    A, B = _quantized_operands("mxfp8", m, n, k, batched=False)
    out_ref = gemm(A, B, tuned=False)
    # SFA living inside a larger buffer (extra rk columns), sliced back out
    sfa = A.scale
    rm, rk = sfa.shape[0], sfa.shape[1]
    big = torch.zeros(rm, rk + 3, 32, 4, 4, device="cuda", dtype=sfa.dtype)
    big[:, :rk] = sfa
    A_slice = BlockScaledOperand.from_parts(A.qdata, big[:, :rk], A.format)
    out_slice = gemm(A_slice, B, tuned=False)
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
    A, B = _quantized_operands("mxfp8", m, n, k, batched=False)
    ref = gemm_blockscaled_ref(A, B)
    qa, bq = A.qdata, B.qdata
    if a_major == "m":
        qa = qa.t().contiguous().t()  # (m, k) with M contiguous
        assert qa.stride() == (1, m)
        A = BlockScaledOperand.from_parts(qa, A.scale, A.format)
    if b_major == "n":
        bq = bq.contiguous()  # (k, n) with N contiguous -> (n, k) n-major after .mT
        assert bq.stride() == (n, 1)
        B = BlockScaledOperand.from_parts(bq, B.scale, B.format, quant_dim=-2)
    out = gemm(A, B, tuned=False)
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
    A_op = BlockScaledOperand.from_parts(qa, SFA, fmt)
    B_op = BlockScaledOperand.from_parts(B, SFB, fmt, quant_dim=-2)
    out = gemm(A_op, B_op, cu_seqlens_m=cu_seqlens_m, tuned=False)

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
    A_op = BlockScaledOperand.from_parts(qa, SFA, "mxfp8")
    B_op = BlockScaledOperand.from_parts(qb.t(), SFB, "mxfp8", quant_dim=-2)
    out = gemm(A_op, B_op, cu_seqlens_k=cu_seqlens_k, tuned=False)
    assert out.shape == (num_experts, m, n)

    ref = torch.stack([a_ref_list[i] @ b_ref_list[i].T for i in range(num_experts)])
    err = (out.float() - ref).abs().max().item()
    assert err < 5e-3, f"varlen_k seqlens_k={seqlens_k} max_err={err}"


def test_blockscaled_gemm_vs_cublas():
    """Bit-exact comparison against torch._scaled_mm (cuBLAS MXFP8 path)."""
    _skip_if_not_sm100()
    m, n, k = 512, 512, 512
    A, B = _quantized_operands("mxfp8", m, n, k, batched=False)
    out = gemm(A, B, tuned=False)
    sfa_flat = scale_blocked_for_cublas(A.scale.unsqueeze(0), m, k // 32)
    sfw_flat = scale_blocked_for_cublas(B.scale.unsqueeze(0), n, k // 32)
    out_cublas = torch._scaled_mm(
        A.qdata, B.qdata, scale_a=sfa_flat, scale_b=sfw_flat, out_dtype=torch.bfloat16
    )
    assert torch.equal(out, out_cublas), (
        f"quack != cuBLAS: max_err={(out.float() - out_cublas.float()).abs().max().item()}"
    )


def test_blockscaled_gemm_bad_sf_layout_rejected():
    """Bad SF layouts cannot reach the GEMM: the container rejects them at
    construction, and one-sided scale factors are rejected at dispatch."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    A, B = _quantized_operands("mxfp8", m, n, k, batched=False)
    # non-contiguous inner block (transposed (4, 4) tail) must be rejected
    with pytest.raises(ValueError, match="inner"):
        BlockScaledOperand.from_parts(A.qdata, A.scale.transpose(-1, -2), A.format)
    # a flattened (rm, rk, 512) form is not part of the contract
    with pytest.raises(ValueError, match="32, 4, 4"):
        BlockScaledOperand.from_parts(A.qdata, A.scale.flatten(-3), A.format)
    # SF on only one operand must be rejected
    with pytest.raises(AssertionError, match="both"):
        gemm(A, B.qdata, tuned=False)


def test_blockscaled_gemm_torch_compile():
    """Raw parts as graph inputs, with from_parts staying in one full graph."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    A, B = _quantized_operands("mxfp8", m, n, k, batched=False)
    ref = gemm(A, B, tuned=False)

    @torch.compile(dynamic=False, fullgraph=True)
    def f(qa, sfa, b, sfw):
        a_op = BlockScaledOperand.from_parts(qa, sfa, "mxfp8_e4m3")
        b_op = BlockScaledOperand.from_parts(b, sfw, "mxfp8_e4m3", quant_dim=-2)
        return gemm(a_op, b_op, tuned=False)

    out = f(A.qdata, A.scale, B.qdata, B.scale)
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


def test_blockscaled_gemm_training_orientations():
    """The three GEMMs of a training linear (Y = X W^T), with the backward
    operands quantized along their reduction dims via blockscaled_quantize_dim0
    and consumed MN-major — data stays row-major throughout, only scales differ:
      fwd:   Y  = X W^T   (X, W rowwise)
      dgrad: dX = dY W    (dY rowwise, W dim0; B (N, K) n-major)
      wgrad: dW = dY^T X  (dY, X dim0; A (N, M) m-major, B (M, K) n-major)
    """
    _skip_if_not_sm100()
    torch.manual_seed(0)
    m, n, k = 256, 320, 512  # tokens, out-features, in-features
    X = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    W = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    dY = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)

    def check(A, B, truth, label, quant_tol):
        out = gemm(A, B, tuned=False)
        ref = gemm_blockscaled_ref(A, B)  # exact dequant reference
        rel_kernel = (out.float() - ref.float()).abs().max() / ref.float().abs().max()
        assert rel_kernel < 5e-3, f"{label}: kernel vs dequant-ref rel={rel_kernel}"
        rel_quant = (out.float() - truth).norm() / truth.norm()
        assert rel_quant < quant_tol, f"{label}: vs bf16 truth rel={rel_quant}"

    # Rowwise (dim -1) quantizations: scales run along K of the (rows, K) view.
    rowwise = lambda t: BlockScaledOperand.from_parts(*blockscaled_quantize(t), "mxfp8")
    # dim0 quantizations: scales run along dim 0 (the reduction dim of the
    # MN-major consumption), i.e. quant_dim=-2 on the (dim0, dim1) view; pass
    # directly as a B operand or as .mT for an A operand (.mT flips quant_dim).
    dim0 = lambda t: BlockScaledOperand.from_parts(
        *blockscaled_quantize_dim0(t), "mxfp8", quant_dim=-2
    )

    check(rowwise(X), rowwise(W).mT, X.float() @ W.float().T, "fwd", 0.04)
    check(rowwise(dY), dim0(W), dY.float() @ W.float(), "dgrad", 0.04)
    check(dim0(dY).mT, dim0(X), dY.float().T @ X.float(), "wgrad", 0.04)


# -- BlockScaledOperand-specific regressions (see AI/blockscaled_api.md section 10) --


def test_tuple_operand_rejected():
    """(data, scale_factor) tuple/list operands are removed: TypeError pointing
    at BlockScaledOperand, raised before any kernel work."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 256
    A, B = _quantized_operands("mxfp8", m, n, k, batched=False)
    with pytest.raises(TypeError, match="BlockScaledOperand"):
        gemm((A.qdata, A.scale), (B.qdata, B.scale), tuned=False)
    with pytest.raises(TypeError, match="no longer accepted"):
        gemm(A, [B.qdata, B.scale], tuned=False)
    with pytest.raises(TypeError, match="no longer accepted"):
        gemm_add((A.qdata, A.scale), B, torch.zeros(m, n, device="cuda"), tuned=False)


def test_nvfp4_per_tensor_scale_folding():
    """NVFP4 quantized with non-unit per-tensor scales, default alpha: the pts
    product must be folded into alpha automatically. Encodes the silent-accuracy
    trap where a dropped pts leaves every output off by pts_A * pts_B (all
    unit-scale tests stay green)."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    torch.manual_seed(0)
    # Large magnitudes so the per-tensor scales are far from 1.
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 30.0
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 30.0
    xq = BlockScaledOperand.quantize(
        x, "nvfp4", per_tensor_scale=nvfp4_per_tensor_scale(x.float().abs().amax())
    )
    wq = BlockScaledOperand.quantize(
        w, "nvfp4", per_tensor_scale=nvfp4_per_tensor_scale(w.float().abs().amax())
    )
    assert xq.per_tensor_scale is not None and xq.per_tensor_scale.item() != 1.0
    out = gemm(xq, wq.mT, tuned=False)
    ref = gemm_blockscaled_ref(xq, wq.mT)
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    assert rel < 5e-3, f"pts folding: rel_err={rel}"
    # The result must track the hp product (pts applied), not be off by pts_A*pts_B.
    hp = x.float() @ w.float().T
    rel_hp = (out.float() - hp).abs().max().item() / hp.abs().max().item()
    assert rel_hp < 0.2, f"pts appears dropped: rel_err vs hp = {rel_hp}"
    # Explicit alpha composes multiplicatively with the folded pts.
    out_a = gemm(xq, wq.mT, alpha=0.5, tuned=False)
    ref_a = gemm_blockscaled_ref(xq, wq.mT, alpha=0.5)
    rel_a = (out_a.float() - ref_a.float()).abs().max().item() / ref_a.float().abs().max().item()
    assert rel_a < 5e-3, f"pts+alpha: rel_err={rel_a}"


def test_square_weight_orientation():
    """K == N square weight: the .mT idiom computes x @ w^T correctly, and the
    wrong-axis forms - shape-legal on square weights, previously silent garbage -
    are rejected via the operand's quant_dim (per-slot contraction-axis check)."""
    _skip_if_not_sm100()
    m, k = 256, 512  # n == k (square weight)
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    w = torch.randn(k, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    xq = BlockScaledOperand.quantize(x, "mxfp8_e4m3")
    wq = BlockScaledOperand.quantize(w, "mxfp8_e4m3")
    out = gemm(xq, wq.mT, tuned=False)
    ref = xq.dequantize(torch.float32) @ wq.dequantize(torch.float32).T
    rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 5e-3, f"square-weight orientation: rel_err={rel}"
    # forgetting the .mT: B arrives quantized along its last dim (not K) -> rejected
    with pytest.raises(ValueError, match="B must be quantized along dim -2"):
        gemm(xq, wq, tuned=False)
    # stray transpose on A: quantized along dim -2 (not K) -> rejected
    with pytest.raises(ValueError, match="A must be quantized along its last dim"):
        gemm(BlockScaledOperand.quantize(x.mT.contiguous(), "mxfp8_e4m3").mT, wq.mT, tuned=False)
    # quantize(dim=-2) builds the (K, N) B operand directly
    b_direct = BlockScaledOperand.quantize(w, "mxfp8_e4m3", dim=-2)
    out2 = gemm(xq, b_direct, tuned=False)
    ref2 = xq.dequantize(torch.float32) @ b_direct.dequantize(torch.float32)
    rel2 = (out2.float() - ref2).abs().max().item() / ref2.abs().max().item()
    assert rel2 < 5e-3, f"quantize(dim=-2) B operand: rel_err={rel2}"


def test_uint8_scale_view_construction():
    """An NVFP4 scale round-tripped through a uint8 view (e.g. a collective)
    must still compute as vec-16 e4m3. Encodes the old dtype-sniffing decode
    trap where uint8 was assumed to mean e8m0 (vec 32)."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    A, B = _quantized_operands("nvfp4", m, n, k, batched=False)
    ref = gemm(A, B, tuned=False)
    A_u8 = BlockScaledOperand.from_parts(A.qdata, A.scale.view(torch.uint8), "nvfp4")
    assert A_u8.scale.dtype == torch.float8_e4m3fn  # canonicalized at construction
    out = gemm(A_u8, B, tuned=False)
    assert torch.equal(out, ref)


def _e5m2_operand(rows, k, seed=0):
    """mxfp8_e5m2 has no quantizer; build one by value-casting an e4m3
    quantization (kernel and dequant reference then see identical stored data)."""
    torch.manual_seed(seed)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    t = BlockScaledOperand.quantize(x, "mxfp8_e4m3")
    q5 = t.qdata.float().to(torch.float8_e5m2)
    return BlockScaledOperand.from_parts(q5, t.scale, "mxfp8_e5m2")


def _mixed_operand(fmt, rows, k, seed=0):
    if fmt == "mxfp8_e5m2":
        return _e5m2_operand(rows, k, seed)
    torch.manual_seed(seed)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    return BlockScaledOperand.quantize(x, fmt)


@pytest.mark.parametrize("fmt_pair", [("mxfp8_e4m3", "mxfp8_e5m2"), ("mxfp8_e5m2", "mxfp8_e4m3")])
def test_blockscaled_gemm_mixed_fp8(fmt_pair):
    """Mixed A/B element types on SM100 (tcgen05 kind::mxf8f6f4 takes independent
    a/b dtypes): e4m3 x e5m2 in both orders, checked against the dequant
    reference."""
    _skip_if_not_sm100()
    fmt_a, fmt_b = fmt_pair
    m, n, k = 256, 512, 512
    A = _mixed_operand(fmt_a, m, k, seed=0)
    W = _mixed_operand(fmt_b, n, k, seed=1)
    out = gemm(A, W.mT, tuned=False)
    ref = gemm_blockscaled_ref(A, W.mT)
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    assert rel < 5e-3, f"{fmt_a} x {fmt_b}: rel_err={rel}"
    ref_dq = A.dequantize(torch.float32) @ W.dequantize(torch.float32).T
    rel_dq = (out.float() - ref_dq).abs().max().item() / ref_dq.abs().max().item()
    assert rel_dq < 5e-3, f"{fmt_a} x {fmt_b}: rel_err vs dequant={rel_dq}"


def test_mixed_format_pairs():
    """A and B carry independent formats. Unrepresentable pairs raise ValueError
    at the interface: nvfp4 pairs only with itself (mxf4nvf4), and packed-fp4x2
    storage cannot feed the byte-container SMEM of the mixed mxf8f6f4 kind (a
    byte-container fp4 format is backlog). Working mixed pairs are covered by
    test_blockscaled_gemm_mixed_fp8."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    A8, B8 = _quantized_operands("mxfp8", m, n, k, batched=False)
    A4, B4 = _quantized_operands("nvfp4", m, n, k, batched=False)
    # nvfp4's mxf4nvf4 kind requires nvfp4 on both operands
    with pytest.raises(ValueError, match="cannot pair"):
        gemm(A8, B4, tuned=False)
    # packed fp4x2 cannot join a mixed (mxf8f6f4, byte-container) pair
    Am4, Bm4 = _quantized_operands("mxfp4", m, n, k, batched=False)
    with pytest.raises(ValueError, match="8-bit"):
        gemm(Am4, B8, tuned=False)
    with pytest.raises(ValueError, match="8-bit"):
        gemm(A8, Bm4, tuned=False)


def test_sm100_dtype_gate():
    """Kernel dtype support is a per-architecture assert (GemmSm100), not a
    registry property: fp6 (uint8 byte containers) is constructible and flows
    through interface + layout validation, then dies at the SM100 gate with a
    message naming the unsupported combination."""
    _skip_if_not_sm100()
    m, k = 256, 512
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    sf = BlockScaledOperand.quantize(x, "mxfp8_e4m3").scale
    t6 = BlockScaledOperand.from_parts(
        torch.zeros(m, k, dtype=torch.uint8, device="cuda"), sf, "mxfp6_e2m3"
    )
    with pytest.raises(AssertionError, match="GemmSm100 blockscaled does not support"):
        gemm(t6, t6.mT, tuned=False)
    # byte-container fp4 in a mixed pair: hardware-legal (kind::mxf8f6f4), but
    # CuTe-DSL 4.6 lacks the unpacksmem/TMA-expansion machinery - same gate.
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    a8 = BlockScaledOperand.quantize(x, "mxfp8_e4m3")
    b4 = BlockScaledOperand.quantize(x, "mxfp4_byte")
    with pytest.raises(AssertionError, match="GemmSm100 blockscaled does not support"):
        gemm(a8, b4.mT, tuned=False)


def test_blockscaled_out_dtype_reserved():
    """out_dtype accepting a BlockScaledFormat (blockscaled D output) is a
    reserved API: it validates the format and raises pointing at the
    SF-generation epilogue milestone."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    A, B = _quantized_operands("mxfp8", m, n, k, batched=False)
    from quack.blockscaled.operand import MXFP8_E4M3

    for out_dtype in (MXFP8_E4M3, "mxfp8_e4m3"):
        with pytest.raises(NotImplementedError, match="SF-generation"):
            gemm(A, B, out_dtype=out_dtype, tuned=False)
    with pytest.raises(NotImplementedError, match="SF-generation"):
        gemm_act(A, B, activation="relu", postact_dtype="mxfp8_e4m3", tuned=False)
    with pytest.raises(ValueError, match="unknown blockscaled format"):
        gemm(A, B, out_dtype="fp8", tuned=False)


def test_e5m2_from_parts_kernel():
    """mxfp8_e5m2 has no in-repo quantizer but the kernel accepts it; construct
    via from_parts with unit e8m0 scales and check against the dequant reference.
    (First e5m2 blockscaled kernel coverage in-repo.)"""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    from quack.blockscaled.quantize import pack_scale_2d_to_blocked_contig

    # e8m0 biased exponent 127 == 1.0; from_parts canonicalizes the uint8 view
    unit_sf = lambda rows: pack_scale_2d_to_blocked_contig(
        torch.full((rows, k // 32), 127, dtype=torch.uint8, device="cuda")
    ).squeeze(0)
    xq = BlockScaledOperand.from_parts(x.to(torch.float8_e5m2), unit_sf(m), "mxfp8_e5m2")
    wq = BlockScaledOperand.from_parts(w.to(torch.float8_e5m2), unit_sf(n), "mxfp8_e5m2")
    out = gemm(xq, wq.mT, tuned=False)
    ref = gemm_blockscaled_ref(xq, wq.mT)
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    assert rel < 5e-3, f"e5m2: rel_err={rel}"


def test_gemm_act_pts_rejected():
    """gemm_act has no alpha to fold the NVFP4 per-tensor scale into."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    xq = BlockScaledOperand.quantize(
        x, "nvfp4", per_tensor_scale=nvfp4_per_tensor_scale(x.float().abs().amax())
    )
    wq = BlockScaledOperand.quantize(
        w, "nvfp4", per_tensor_scale=nvfp4_per_tensor_scale(w.float().abs().amax())
    )
    with pytest.raises(NotImplementedError, match="per-tensor scale"):
        gemm_act(xq, wq.mT, activation="relu", tuned=False)


def test_blockscaled_gemm_torch_compile_container():
    """BlockScaledOperand as a torch.compile graph input, fullgraph=True
    (attribute reads trace through the frozen dataclass; exercises the preserved
    _sf_encode e8m0->uint8 Inductor workaround)."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    A, B = _quantized_operands("mxfp8", m, n, k, batched=False)
    ref = gemm(A, B, tuned=False)

    @torch.compile(dynamic=False, fullgraph=True)
    def f(a, b):
        return gemm(a, b, tuned=False)

    out = f(A, B)
    assert torch.equal(out, ref)


def test_blockscaled_quantize_in_graph():
    """Quantize + gemm in one full graph, including container construction."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5

    def f(x, w):
        xq = BlockScaledOperand.quantize(x, "mxfp8_e4m3")
        wq = BlockScaledOperand.quantize(w, "mxfp8_e4m3")
        return gemm(xq, wq.mT, tuned=False)

    ref = f(x, w)
    out = torch.compile(f, dynamic=False, fullgraph=True)(x, w)
    assert torch.equal(out, ref)
