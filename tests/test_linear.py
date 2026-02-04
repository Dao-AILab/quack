# Copyright (C) 2025, Tri Dao.
import math
import pytest
import torch
import torch.nn.functional as F

from quack.linear import linear_func, linear_act_func
from quack.gemm_interface import (
    gemm_add_inplace,
    gemm_dact,
    gemm_gated,
    gemm_dgated,
    gemm_ref,
    gemm_act_ref,
    gemm_dact_ref,
    gemm_gated_ref,
    gemm_dgated_ref,
)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("out_features", [1504, 2048])
@pytest.mark.parametrize("in_features", [736, 4096])
# @pytest.mark.parametrize("out_features", [2048])
# @pytest.mark.parametrize("in_features", [4096])
def test_linear(in_features, out_features, has_bias, input_dtype):
    device = "cuda"
    torch.random.manual_seed(0)
    m = 1920
    x = torch.randn((m, in_features), device=device, dtype=input_dtype)
    x = x[::2].requires_grad_(True)  # Testing non-contiguous
    w = (
        torch.randn((out_features, in_features), device=device, dtype=input_dtype)
        / math.sqrt(in_features)
    ).requires_grad_()
    bias = torch.randn(out_features, device=device, requires_grad=True) if has_bias else None
    x_ref, w_ref, bias_ref = [
        t.detach().clone().float().requires_grad_(True) if t is not None else None
        for t in (x, w, bias)
    ]
    x_pt, w_pt, bias_pt = [
        t.detach().clone().to(x.dtype).requires_grad_(True) if t is not None else None
        for t in (x, w, bias)
    ]
    out = linear_func(x, w, bias, tuned=False)  # Disable tuning for faster test
    out_ref = F.linear(x_ref, w_ref, bias_ref)
    out_pt = F.linear(x_pt, w_pt, bias_pt)
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-6
    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout.float())
    out_pt.backward(dout)
    assert (x.grad - x_ref.grad).abs().max() < 2 * (x_pt.grad - x_ref.grad).abs().max() + 1e-6
    assert (w.grad - w_ref.grad).abs().max() < 2 * (w_pt.grad - w_ref.grad).abs().max() + 1e-6
    if bias is not None:
        assert (bias.grad - bias_ref.grad).abs().max() < 2 * (
            bias_pt.grad - bias_ref.grad
        ).abs().max() + 1e-6


@pytest.mark.parametrize("store_preact", [False, True])
@pytest.mark.parametrize("activation", ["relu", "relu_sq", "gelu_tanh_approx"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("out_features", [1504, 2048])
@pytest.mark.parametrize("in_features", [736, 4096])
# @pytest.mark.parametrize("out_features", [2048])
# @pytest.mark.parametrize("in_features", [4096])
def test_linear_act(in_features, out_features, has_bias, input_dtype, activation, store_preact):
    device = "cuda"
    torch.random.manual_seed(0)
    m = 1920
    x = torch.randn((m, in_features), device=device, dtype=input_dtype)
    x = x[::2].requires_grad_(True)  # Testing non-contiguous
    w = (
        torch.randn((out_features, in_features), device=device, dtype=input_dtype)
        / math.sqrt(in_features)
    ).requires_grad_()
    bias = torch.randn(out_features, device=device, requires_grad=True) if has_bias else None
    # Disable tuning for faster test
    preact, postact = linear_act_func(
        x, w, activation, bias=bias, store_preact=store_preact, tuned=False
    )
    preact_ref, postact_ref = gemm_act_ref(
        x.float(), w.float().T, activation=activation, bias=bias, store_preact=store_preact
    )
    preact_pt, postact_pt = gemm_act_ref(
        x, w.T, activation=activation, bias=bias, store_preact=store_preact
    )
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-6
    if store_preact:
        assert preact is not None and preact_ref is not None
        assert (preact - preact_ref).abs().max() < 2 * (preact_pt - preact_ref).abs().max() + 1e-6


@pytest.mark.parametrize("activation", ["relu", "relu_sq", "gelu_tanh_approx"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("k", [736, 1024])
@pytest.mark.parametrize("n", [1504, 2048])
def test_gemm_dact(n, k, input_dtype, activation):
    """Test GEMM with activation gradient computation."""
    device = "cuda"
    torch.random.manual_seed(0)
    m = 960
    dout_input = torch.randn((m, k), device=device, dtype=input_dtype)
    weight = torch.randn((n, k), device=device, dtype=input_dtype) / math.sqrt(k)
    preact = torch.randn((m, n), device=device, dtype=input_dtype, requires_grad=True)
    # Disable tuning for faster test
    dx, postact = gemm_dact(dout_input, weight.T, preact, activation=activation, tuned=False)
    dx_ref, postact_ref = gemm_dact_ref(
        dout_input.float(), weight.float().T, preact.float(), activation=activation
    )
    dx_pt, postact_pt = gemm_dact_ref(dout_input, weight.T, preact, activation=activation)
    assert (dx - dx_ref).abs().max() < 2 * (dx_pt - dx_ref).abs().max() + 1e-5
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-5


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1504, 2048])
@pytest.mark.parametrize("k", [736, 1024])
@pytest.mark.parametrize("m", [960, 1920])
def test_gemm_add_inplace(m, k, n, input_dtype):
    """Test in-place GEMM with addition: C += A @ B."""
    device = "cuda"
    torch.random.manual_seed(0)
    A = torch.randn((m, k), device=device, dtype=input_dtype)
    B = torch.randn((k, n), device=device, dtype=input_dtype)
    C = torch.randn((m, n), device=device, dtype=input_dtype)
    # Save original C for reference computation
    C_og = C.clone()
    gemm_add_inplace(A, B, C, tuned=False)
    C_ref = C_og.float() + torch.mm(A.float(), B.float())
    C_pt = C_og + torch.mm(A, B)
    assert (C - C_ref).abs().max() < 2 * (C_pt - C_ref).abs().max() + 1e-5


@pytest.mark.parametrize("alpha_beta_type", ["float", "tensor"])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [512, 1024])
@pytest.mark.parametrize("k", [256, 768])
@pytest.mark.parametrize("m", [480, 960])
def test_gemm_add_inplace_alpha_beta(m, k, n, input_dtype, alpha, beta, alpha_beta_type):
    """Test in-place GEMM with alpha/beta scaling: C = alpha * A @ B + beta * C."""
    device = "cuda"
    torch.random.manual_seed(42)
    A = torch.randn((m, k), device=device, dtype=input_dtype)
    B = torch.randn((k, n), device=device, dtype=input_dtype)
    C = torch.randn((m, n), device=device, dtype=input_dtype)
    if alpha_beta_type == "tensor":
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
        beta = torch.tensor(beta, device=device, dtype=torch.float32)
    C_og = C.clone()
    gemm_add_inplace(A, B, C, alpha=alpha, beta=beta, tuned=False)
    alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
    beta_val = beta.item() if torch.is_tensor(beta) else beta
    C_ref = alpha_val * torch.mm(A.float(), B.float()) + beta_val * C_og.float()
    C_pt = alpha_val * torch.mm(A, B) + beta_val * C_og
    assert (C - C_ref).abs().max() < 2 * (C_pt - C_ref).abs().max() + 1e-4


@pytest.mark.parametrize("store_preact", [True, False])
@pytest.mark.parametrize("activation", ["swiglu", "swiglu_oai", "reglu", "geglu", "glu"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("out_features", [1504, 2048])
@pytest.mark.parametrize("in_features", [736, 4096])
def test_gemm_gated(in_features, out_features, has_bias, input_dtype, activation, store_preact):
    """Test GEMM with gated activation forward computation."""
    device = "cuda"
    torch.random.manual_seed(0)
    m = 1920
    x = torch.randn((m, in_features), device=device, dtype=input_dtype, requires_grad=True)
    x = x[::2]  # Testing non-contiguous
    # Weight has 2*out_features columns for gated activation
    w = (
        torch.randn((2 * out_features, in_features), device=device, dtype=input_dtype)
        / math.sqrt(in_features)
    ).requires_grad_()
    bias = torch.randn(2 * out_features, device=device) if has_bias else None
    preact, postact = gemm_gated(
        x, w.T, bias=bias, activation=activation, store_preact=store_preact, tuned=False
    )
    preact_ref, postact_ref = gemm_gated_ref(
        x.float(), w.float().T, bias=bias, activation=activation, store_preact=store_preact
    )
    preact_pt, postact_pt = gemm_gated_ref(
        x, w.T, bias=bias, activation=activation, store_preact=store_preact
    )
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-6
    if store_preact:
        assert preact is not None and preact_ref is not None
        assert (preact - preact_ref).abs().max() < 2 * (preact_pt - preact_ref).abs().max() + 1e-5


@pytest.mark.parametrize("activation", ["swiglu", "swiglu_oai", "reglu", "geglu", "glu"])
# @pytest.mark.parametrize("activation", ["swiglu"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("colvec_reduce", [False, True])
# @pytest.mark.parametrize("colvec_reduce", [True])
@pytest.mark.parametrize("has_colvec_scale", [False, True])
# @pytest.mark.parametrize("has_colvec_scale", [True])
@pytest.mark.parametrize("k", [736, 1024])
@pytest.mark.parametrize("n", [1504, 2048])
# @pytest.mark.parametrize("k", [1024])
# @pytest.mark.parametrize("n", [2048])
def test_gemm_dgated(n, k, has_colvec_scale, colvec_reduce, input_dtype, activation):
    """Test GEMM with gated activation gradient computation."""
    device = "cuda"
    torch.random.manual_seed(0)
    m = 960
    dout_input = torch.randn((m, k), device=device, dtype=input_dtype)
    weight = torch.randn((n, k), device=device, dtype=input_dtype) / math.sqrt(k)
    # PreAct has 2*n columns for gated activation (gate and up projections interleaved)
    preact = torch.randn((m, 2 * n), device=device, dtype=input_dtype, requires_grad=True)
    colvec_scale = torch.randn(m, device=device) if has_colvec_scale else None
    dx, postact, *rest = gemm_dgated(
        dout_input,
        weight.T,
        preact,
        colvec_scale=colvec_scale,
        activation=activation,
        colvec_reduce=colvec_reduce,
        tuned=False,
    )
    if colvec_reduce:
        colvec_reduce_out = rest[0]
    dx_ref, postact_ref = gemm_dgated_ref(
        dout_input.float(), weight.float().T, preact.float(), activation=activation
    )
    dx_pt, postact_pt = gemm_dgated_ref(dout_input, weight.T, preact, activation=activation)
    if colvec_reduce:
        colvec_reduce_ref = (postact_ref * gemm_ref(dout_input.float(), weight.float().T)).sum(
            dim=-1
        )
        colvec_reduce_pt = (postact_pt * gemm_ref(dout_input, weight.T)).sum(dim=-1)
    if has_colvec_scale:
        dx_ref *= colvec_scale.float()[:, None]
        postact_ref *= colvec_scale.float()[:, None]
        dx_pt *= colvec_scale[:, None]
        postact_pt *= colvec_scale[:, None]
    assert (dx - dx_ref).abs().max() < 2 * (dx_pt - dx_ref).abs().max() + 1e-5
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-5
    if colvec_reduce:
        assert (colvec_reduce_out - colvec_reduce_ref).abs().max() < 2 * (
            colvec_reduce_pt - colvec_reduce_ref
        ).abs().max() + 1e-5
