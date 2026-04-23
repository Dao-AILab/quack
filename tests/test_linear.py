# Copyright (C) 2025, Tri Dao.
from dataclasses import replace
import math
import pytest
import torch
import torch.nn.functional as F

from quack.linear import linear_func, linear_act_func, act_linear_func, gated_linear_func
from quack.linear import linear_gated_func
from quack.mlp import mlp_func
from quack.gemm_interface import (
    gemm,
    gemm_add,
    gemm_add_inplace,
    gemm_tuned,
    default_config,
    gemm_dact,
    gemm_gated,
    gemm_dgated,
    gemm_ref,
    gemm_add_ref,
    gemm_act_ref,
    gemm_dact_ref,
    gemm_gated_ref,
    gemm_dgated_ref,
    gemm_rms,
    gemm_rms_ref,
    gemm_norm_act,
    gemm_norm_act_ref,
)
from quack.gemm_config import GemmConfig
from quack.rounding import RoundingMode
from quack.rms_final_reduce import rms_final_reduce

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024


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


@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("out_major", ["m", "n"])
@pytest.mark.parametrize("B_major", ["k", "n"])
@pytest.mark.parametrize("A_major", ["k", "m"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1504, 2048])
@pytest.mark.parametrize("k", [736, 1024])
@pytest.mark.parametrize("m", [960])
def test_gemm(m, k, n, input_dtype, A_major, B_major, out_major, swap_ab):
    device = "cuda"
    torch.random.manual_seed(0)
    A = torch.randn((m, k), device=device, dtype=input_dtype)
    if A_major == "m":
        A = A.T.contiguous().T
    B = torch.randn((k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    if B_major == "k":
        B = B.T.contiguous().T
    out = torch.empty((m, n), device=device, dtype=input_dtype)
    if out_major == "m":
        out = out.T.contiguous().T
    config = replace(default_config(torch.device(device)), swap_ab=swap_ab)
    gemm_tuned.fn(A, B, out, config=config)
    out_ref = gemm_ref(A.float(), B.float())
    out_pt = gemm_ref(A, B)
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-5


@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("out_major", ["m", "n"])
@pytest.mark.parametrize("B_major", ["k", "n"])
@pytest.mark.parametrize("A_major", ["k", "m"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("bias_dtype", [None, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("concat_tensor", ["A", "B", "out"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1504, 2048])
@pytest.mark.parametrize("k", [736, 1024])
@pytest.mark.parametrize("m", [960])
def test_gemm_concat_layout(
    m, k, n, input_dtype, concat_tensor, bias_dtype, batched, A_major, B_major, out_major, swap_ab
):
    """Test concat_layout on each tensor with all major/stride combinations."""
    if bias_dtype is not None and batched:
        pytest.skip("batched + bias not supported")
    device = "cuda"
    torch.random.manual_seed(0)
    concat = (concat_tensor,)
    if bias_dtype is not None and concat_tensor == "B":
        concat = ("B", "bias")
    batch_shape = (3,) if batched else ()
    A = torch.randn((*batch_shape, m, k), device=device, dtype=input_dtype) / math.sqrt(k)
    if A_major == "m":
        A = A.mT.contiguous().mT
    B = torch.randn((*batch_shape, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    if B_major == "k":
        B = B.mT.contiguous().mT
    out = torch.empty((*batch_shape, m, n), device=device, dtype=input_dtype)
    if out_major == "m":
        out = out.mT.contiguous().mT
    bias = torch.randn(n, device=device, dtype=bias_dtype) if bias_dtype is not None else None
    config = replace(default_config(torch.device(device)), swap_ab=swap_ab)
    gemm_tuned.fn(A, B, out, config=config, bias=bias, concat_layout=concat)
    # For ref: gemm_ref always produces n-major output and interleaves rows for concat="out".
    # When the kernel's out is m-major, the kernel interleaves columns instead.
    # Match by creating the ref with matching out major.
    if concat_tensor == "out" and out_major == "m":
        # Kernel interleaves columns. Ref: compute flat, then interleave columns.
        out_ref_flat = gemm_ref(A.float(), B.float(), bias=bias)
        out_ref = torch.cat([out_ref_flat[..., ::2], out_ref_flat[..., 1::2]], dim=-1)
        out_pt_flat = gemm_ref(A, B, bias=bias)
        out_pt = torch.cat([out_pt_flat[..., ::2], out_pt_flat[..., 1::2]], dim=-1)
    else:
        out_ref = gemm_ref(A.float(), B.float(), bias=bias, concat_layout=concat)
        out_pt = gemm_ref(A, B, bias=bias, concat_layout=concat)
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-5


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
@pytest.mark.parametrize("is_concat_layout_out", [False, True])
def test_gemm_add_inplace(m, k, n, input_dtype, is_concat_layout_out):
    """Test in-place GEMM with addition: C += A @ B."""
    device = "cuda"
    torch.random.manual_seed(0)
    A = torch.randn((m, k), device=device, dtype=input_dtype)
    B = torch.randn((k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    concat = ("C", "out") if is_concat_layout_out else None
    C = torch.randn((m, n), device=device, dtype=input_dtype)
    # Save original C for reference computation
    C_og = C.clone()
    gemm_add_inplace(A, B, C, tuned=False, concat_layout=concat)
    C_ref = gemm_add_ref(
        A.float(),
        B.float(),
        C_og.float(),
        alpha=1.0,
        beta=1.0,
        out_dtype=torch.float32,
        concat_layout=concat,
    )
    C_pt = gemm_add_ref(
        A,
        B,
        C_og,
        alpha=1.0,
        beta=1.0,
        out_dtype=input_dtype,
        concat_layout=concat,
    )
    assert (C - C_ref).abs().max() < 2 * (C_pt - C_ref).abs().max() + 1e-5


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1504, 2048])
@pytest.mark.parametrize("k", [736, 1024])
@pytest.mark.parametrize("m", [960, 1920])
def test_gemm_add_out_reuses_c_storage(m, k, n, input_dtype):
    """Regression test for gemm_add(..., out=C) dispatching to the in-place path."""
    device = "cuda"
    torch.random.manual_seed(123)
    A = torch.randn((m, k), device=device, dtype=input_dtype)
    B = torch.randn((k, n), device=device, dtype=input_dtype)
    C = torch.randn((m, n), device=device, dtype=input_dtype)
    alpha = torch.tensor(0.5, device=device, dtype=torch.float32)
    C_og = C.clone()
    out = gemm_add(A, B, C, out=C, alpha=alpha, beta=1.0, tuned=False)
    alpha_val = alpha.item()
    C_ref = alpha_val * torch.mm(A.float(), B.float()) + C_og.float()
    C_pt = alpha_val * torch.mm(A, B) + C_og
    assert out is C
    assert (C - C_ref).abs().max() < 2 * (C_pt - C_ref).abs().max() + 1e-5


@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize(
    "tile_m, cluster_m, tile_n",
    [
        (128, 1, 160),
        (128, 1, 192),
        (128, 1, 224),
        (128, 1, 240),
        (256, 2, 160),
        (256, 2, 224),
    ],
)
def test_gemm_add_sm100_unaligned_epilogue_tile_n(tile_m, cluster_m, tile_n, swap_ab):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-specific TMA store regression")

    # Bug repro: with source C present, SM100 may choose a 64-wide epilogue TMA store tile.
    # For CTA-N sizes such as 160/224/240, later CTA-N tiles start at a non-64-aligned
    # offset, and TMA store skipped the prefix before the next 64-element boundary.
    device = "cuda"
    dtype = torch.float16
    k, tail_n = 8, tile_n + 8
    base_cfg = dict(
        tile_m=tile_m,
        tile_n=tile_n,
        pingpong=False,
        is_dynamic_persistent=True,
        cluster_m=cluster_m,
        cluster_n=1,
        swap_ab=swap_ab,
        max_swizzle_size=8,
        device_capacity=10,
    )

    if not swap_ab:
        A = torch.zeros((1, 1, k), device=device, dtype=dtype)
        A[0, 0, 0] = 1
        B = torch.zeros((1, k, tail_n), device=device, dtype=dtype)
        B[0, 0] = torch.arange(tail_n, device=device, dtype=dtype) / 1000
        C = torch.zeros((1, 1, tail_n), device=device, dtype=dtype)
    else:
        A = torch.zeros((1, tail_n, k), device=device, dtype=dtype)
        A[0, :, 0] = torch.arange(tail_n, device=device, dtype=dtype) / 1000
        B = torch.zeros((1, k, 1), device=device, dtype=dtype)
        B[0, 0, 0] = 1
        C = torch.zeros((1, tail_n, 1), device=device, dtype=dtype)
    out = torch.empty_like(C)
    gemm_tuned.fn(A, B, out, C, beta=0.0, config=GemmConfig(**base_cfg))
    ref = torch.baddbmm(C.float(), A.float(), B.float(), beta=0.0)
    torch.testing.assert_close(out.float(), ref, atol=0, rtol=0)


@pytest.mark.parametrize("alpha_beta_type", ["float", "tensor"])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [512, 1024])
@pytest.mark.parametrize("k", [256, 768])
@pytest.mark.parametrize("m", [480, 960])
@pytest.mark.parametrize("is_concat_layout_out", [False, True])
def test_gemm_add_inplace_alpha_beta(
    m, k, n, input_dtype, alpha, beta, alpha_beta_type, is_concat_layout_out
):
    """Test in-place GEMM with alpha/beta scaling: C = alpha * A @ B + beta * C."""
    device = "cuda"
    torch.random.manual_seed(42)
    A = torch.randn((m, k), device=device, dtype=input_dtype)
    B = torch.randn((k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    C = torch.randn((m, n), device=device, dtype=input_dtype)
    concat = ("C", "out") if is_concat_layout_out else None
    if alpha_beta_type == "tensor":
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
        beta = torch.tensor(beta, device=device, dtype=torch.float32)
    C_og = C.clone()
    gemm_add_inplace(A, B, C, alpha=alpha, beta=beta, tuned=False, concat_layout=concat)
    C_ref = gemm_add_ref(
        A.float(),
        B.float(),
        C_og.float(),
        alpha=alpha.item() if torch.is_tensor(alpha) else alpha,
        beta=beta.item() if torch.is_tensor(beta) else beta,
        out_dtype=torch.float32,
        concat_layout=concat,
    )
    C_pt = gemm_add_ref(
        A,
        B,
        C_og,
        alpha=alpha,
        beta=beta,
        concat_layout=concat,
        out_dtype=input_dtype,
    )
    assert (C - C_ref).abs().max() < 2 * (C_pt - C_ref).abs().max() + 1e-4


@pytest.mark.parametrize("store_preact", [True, False])
@pytest.mark.parametrize(
    "activation", ["swiglu", "swiglu_oai", "swiglu_oai-tanh", "reglu", "geglu", "glu"]
)
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("bias_dtype", [None, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("out_features", [1504, 2048])
@pytest.mark.parametrize("in_features", [736, 4096])
@pytest.mark.parametrize("is_concat_layout_B", [False, True])
def test_gemm_gated(
    in_features, out_features, bias_dtype, input_dtype, activation, store_preact, is_concat_layout_B
):
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
    B = w.T
    bias = (
        torch.randn(2 * out_features, device=device, dtype=bias_dtype)
        if bias_dtype is not None
        else None
    )
    concat = (
        ("B", "bias")
        if is_concat_layout_B and bias_dtype is not None
        else ("B",)
        if is_concat_layout_B
        else None
    )
    preact, postact = gemm_gated(
        x,
        B,
        bias=bias,
        activation=activation,
        store_preact=store_preact,
        tuned=False,
        concat_layout=concat,
    )
    preact_ref, postact_ref = gemm_gated_ref(
        x.float(),
        B.float(),
        bias=bias,
        activation=activation,
        store_preact=store_preact,
        concat_layout=concat,
    )
    preact_pt, postact_pt = gemm_gated_ref(
        x,
        B,
        bias=bias,
        activation=activation,
        store_preact=store_preact,
        concat_layout=concat,
    )
    assert postact.shape == (x.shape[0], out_features)
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-6
    if store_preact:
        assert preact is not None and preact_ref is not None
        assert preact.shape == (x.shape[0], 2 * out_features)
        assert (preact - preact_ref).abs().max() < 2 * (preact_pt - preact_ref).abs().max() + 1e-5


@pytest.mark.parametrize(
    "activation", ["swiglu", "swiglu_oai", "swiglu_oai-tanh", "reglu", "geglu", "glu"]
)
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
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
        pytest.skip("SM120 gated dactivation GEMM epilogue is not yet supported")
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


@pytest.mark.parametrize("activation", ["gelu_tanh_approx", "swiglu"])
@pytest.mark.parametrize("freeze", ["x", "weight"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
def test_dact_linear_partial_grad(input_dtype, freeze, activation):
    """DActLinearFunc / DGatedLinearFunc backward with one input frozen.

    Regression test: gemm_dact recomputes postact from preact, which is needed for
    both dpreact and dweight. Previously, freezing weight caused preact to not be saved.
    """
    device = "cuda"
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
        pytest.skip("SM120 (d)activation GEMM epilogues not yet supported")
    torch.random.manual_seed(0)
    m, in_features, out_features = 256, 512, 512
    gated = activation in ("swiglu", "swiglu_oai", "swiglu_oai-tanh", "reglu", "geglu", "glu")
    freeze_x = freeze == "x"
    x = torch.randn(m, in_features, device=device, dtype=input_dtype, requires_grad=not freeze_x)
    w1_out = 2 * out_features if gated else out_features
    w1 = (
        torch.randn(w1_out, in_features, device=device, dtype=input_dtype) / math.sqrt(in_features)
    ).requires_grad_(not freeze_x)
    w2 = (
        torch.randn(in_features, out_features, device=device, dtype=input_dtype)
        / math.sqrt(out_features)
    ).requires_grad_(freeze_x)
    # fc1 forward
    fc1_fn = linear_gated_func if gated else linear_act_func
    preact, postact = fc1_fn(x, w1, activation, store_preact=True, tuned=False)
    # fc2 forward + backward
    fc2_fn = gated_linear_func if gated else act_linear_func
    out = fc2_fn(preact, w2, postact, activation=activation, tuned=False)
    dout = torch.randn_like(out)
    out.backward(dout)
    if freeze_x:
        # x and w1 frozen, only w2 gets grad
        assert x.grad is None
        assert w1.grad is None
        assert w2.grad is not None
    else:
        # w2 frozen, x and w1 get grad
        assert x.grad is not None
        assert w1.grad is not None
        assert w2.grad is None


@pytest.mark.parametrize("activation", ["gelu_tanh_approx", "swiglu"])
@pytest.mark.parametrize("freeze", ["x", "weight"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
def test_linear_act_partial_grad(input_dtype, freeze, activation):
    """LinearActFunc / LinearGatedFunc backward with one input frozen.

    Regression test: ensure gradient flows correctly when x or weight is frozen.
    """
    device = "cuda"
    gated = activation in ("swiglu", "swiglu_oai", "swiglu_oai-tanh", "reglu", "geglu", "glu")
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12 and not gated:
        pytest.skip("SM120 non-gated activation GEMM epilogue is not yet supported")
    torch.random.manual_seed(0)
    m, in_features, out_features = 256, 512, 512
    freeze_x = freeze == "x"
    x = torch.randn(m, in_features, device=device, dtype=input_dtype, requires_grad=not freeze_x)
    fc1_out = 2 * out_features if gated else out_features
    w = (
        torch.randn(fc1_out, in_features, device=device, dtype=input_dtype) / math.sqrt(in_features)
    ).requires_grad_(freeze_x)
    fc1_fn = linear_gated_func if gated else linear_act_func
    preact, postact = fc1_fn(x, w, activation, store_preact=True, tuned=False)
    # preact is the differentiable output; postact is marked non-differentiable
    dout = torch.randn_like(preact)
    preact.backward(dout)
    if freeze_x:
        assert x.grad is None
        assert w.grad is not None
    else:
        assert x.grad is not None
        assert w.grad is None


@pytest.mark.parametrize(
    "fn_name", ["linear_func", "linear_act_func", "linear_gated_func", "mlp_func"]
)
def test_autocast(fn_name):
    """Autocast: float32 inputs are cast to bfloat16 for the kernel.

    Regression test for https://github.com/Dao-AILab/quack/issues/54.
    """
    device = "cuda"
    if (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 12
        and fn_name in ("linear_act_func", "mlp_func")
    ):
        pytest.skip("SM120 non-gated activation GEMM epilogue is not yet supported")
    torch.random.manual_seed(0)
    m, in_features, out_features = 256, 512, 512

    x = torch.randn(m, in_features, device=device, dtype=torch.float32, requires_grad=True)
    w = torch.randn(out_features, in_features, device=device, dtype=torch.float32)
    w /= math.sqrt(in_features)
    w.requires_grad_(True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if fn_name == "linear_func":
            out = linear_func(x, w, tuned=False)
        elif fn_name == "linear_act_func":
            out, _postact = linear_act_func(x, w, activation="gelu_tanh_approx", tuned=False)
        elif fn_name == "linear_gated_func":
            w_gated = (
                torch.randn(2 * out_features, in_features, device=device, dtype=torch.float32)
                / math.sqrt(in_features)
            ).requires_grad_(True)
            out, _postact = linear_gated_func(x, w_gated, activation="swiglu", tuned=False)
            w = w_gated  # for grad check below
        elif fn_name == "mlp_func":
            w2 = torch.randn(
                in_features,
                out_features,
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            ) / math.sqrt(out_features)
            out = mlp_func(x, w, w2, activation="gelu_tanh_approx", tuned=False)

    assert out.dtype == torch.bfloat16, f"expected bfloat16 output, got {out.dtype}"
    out.sum().backward()
    assert x.grad is not None
    assert w.grad is not None


@pytest.mark.parametrize(
    "fn_name", ["linear_func", "linear_act_func", "linear_gated_func", "mlp_func"]
)
def test_autocast_compile(fn_name):
    """Autocast under torch.compile(fullgraph=True).

    Regression test for https://github.com/Dao-AILab/quack/issues/54.
    """
    device = "cuda"
    if (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 12
        and fn_name in ("linear_act_func", "mlp_func")
    ):
        pytest.skip("SM120 non-gated activation GEMM epilogue is not yet supported")
    torch.random.manual_seed(0)
    m, in_features, out_features = 256, 512, 512

    x = torch.randn(m, in_features, device=device, dtype=torch.float32, requires_grad=True)
    w = torch.randn(out_features, in_features, device=device, dtype=torch.float32)
    w /= math.sqrt(in_features)
    w.requires_grad_(True)

    if fn_name == "linear_func":

        @torch.compile(fullgraph=True)
        def fn(x, w):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return linear_func(x, w, tuned=False)

        out = fn(x, w)
    elif fn_name == "linear_act_func":

        @torch.compile(fullgraph=True)
        def fn(x, w):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return linear_act_func(x, w, activation="gelu_tanh_approx", tuned=False)

        out, _postact = fn(x, w)
    elif fn_name == "linear_gated_func":
        w_gated = (
            torch.randn(2 * out_features, in_features, device=device, dtype=torch.float32)
            / math.sqrt(in_features)
        ).requires_grad_(True)

        @torch.compile(fullgraph=True)
        def fn(x, w):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return linear_gated_func(x, w, activation="swiglu", tuned=False)

        out, _postact = fn(x, w_gated)
        w = w_gated
    elif fn_name == "mlp_func":
        w2 = (
            torch.randn(in_features, out_features, device=device, dtype=torch.float32)
            / math.sqrt(out_features)
        ).requires_grad_(True)

        @torch.compile(fullgraph=True)
        def fn(x, w, w2):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return mlp_func(x, w, w2, activation="gelu_tanh_approx", tuned=False)

        out = fn(x, w, w2)

    assert out.dtype == torch.bfloat16, f"expected bfloat16 output, got {out.dtype}"
    out.sum().backward()
    assert x.grad is not None
    assert w.grad is not None


# =============================================================================
# Stochastic Rounding Tests
# =============================================================================


@pytest.mark.parametrize("sr_seed", [0, 42])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [512, 1024])
@pytest.mark.parametrize("k", [256, 768])
@pytest.mark.parametrize("m", [480, 960])
def test_gemm_stochastic_rounding(m, k, n, input_dtype, sr_seed):
    """Test GEMM with stochastic rounding on SM100/SM110.

    Validates that SR produces results close to RNE (within BF16 tolerance)
    and that the output has correct shape and dtype.
    """
    device = "cuda"
    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        pytest.skip("Stochastic rounding requires SM100")
    torch.random.manual_seed(0)
    A = torch.randn((m, k), device=device, dtype=input_dtype)
    B = torch.randn((k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    out_sr = gemm(A, B, tuned=False, rounding_mode=RoundingMode.RS, sr_seed=sr_seed)
    out_rn = gemm(A, B, tuned=False, rounding_mode=RoundingMode.RN)
    out_ref = torch.mm(A.float(), B.float())
    assert out_sr.shape == out_rn.shape
    assert out_sr.dtype == input_dtype
    # SR should be close to reference; may differ by up to 1 ULP more than RNE
    # so use a looser multiplier and atol
    assert (out_sr - out_ref).abs().max() < 3 * (out_rn - out_ref).abs().max() + 5e-3


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
def test_gemm_sr_requires_sm100(input_dtype):
    """Assert that SR raises on non-SM100 hardware."""
    device = "cuda"
    cap = torch.cuda.get_device_capability()
    if cap[0] == 10:
        pytest.skip("This test is for non-SM100 hardware")
    torch.random.manual_seed(0)
    A = torch.randn((128, 256), device=device, dtype=input_dtype)
    B = torch.randn((256, 128), device=device, dtype=input_dtype)
    with pytest.raises(AssertionError, match="SM100"):
        gemm(A, B, tuned=False, rounding_mode=RoundingMode.RS)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [512])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("m", [480])
def test_gemm_sr_different_seeds(m, k, n, input_dtype):
    """Different SR seeds should produce different results (non-deterministic rounding)."""
    device = "cuda"
    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        pytest.skip("Stochastic rounding requires SM100")
    torch.random.manual_seed(0)
    A = torch.randn((m, k), device=device, dtype=input_dtype)
    B = torch.randn((k, n), device=device, dtype=input_dtype)
    out1 = gemm(A, B, tuned=False, rounding_mode=RoundingMode.RS, sr_seed=1)
    out2 = gemm(A, B, tuned=False, rounding_mode=RoundingMode.RS, sr_seed=2)
    # Different seeds should give different outputs (with high probability)
    assert not torch.equal(out1, out2)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N", [1, 4, 8, 16, 64, 128])
@pytest.mark.parametrize("M", [1, 37, 960, 1920])
def test_rms_final_reduce(M, N, input_dtype):
    """Test rms_final_reduce: rstd[m] = rsqrt(sum_n(x[m,n]) * scale + eps)."""
    device = "cuda"
    torch.random.manual_seed(0)
    total_cols = 4096  # pretend the GEMM had this many columns
    scale = 1.0 / total_cols
    eps = 1e-6
    x = torch.randn((M, N), device=device, dtype=input_dtype).abs()  # partial sums are non-negative
    rstd = rms_final_reduce(x, scale=scale, eps=eps)
    rstd_ref = torch.rsqrt(x.float().sum(dim=-1) * scale + eps)
    assert (rstd - rstd_ref).abs().max() < 1e-5


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("has_norm_weight", [False, True])
@pytest.mark.parametrize("has_C", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1504, 2048])
@pytest.mark.parametrize("k", [736, 1024])
@pytest.mark.parametrize("m", [960, 1920])
def test_gemm_rms(m, k, n, input_dtype, has_C, has_norm_weight, use_compile):
    """Test GEMM + RMS + optional rowvec scaling.

    D_raw = A @ B (+ C), rstd = rsqrt(mean(D_raw^2) + eps), D_out = D_raw * norm_weight.
    """
    device = "cuda"
    torch.random.manual_seed(0)
    eps = 1e-6
    A = torch.randn((m, k), device=device, dtype=input_dtype)
    B = torch.randn((k, n), device=device, dtype=input_dtype)
    C = torch.randn((m, n), device=device, dtype=input_dtype) if has_C else None
    norm_weight = torch.randn(n, device=device, dtype=input_dtype) if has_norm_weight else None
    fn = gemm_rms if not use_compile else torch.compile(gemm_rms, fullgraph=True)
    D, rstd = fn(A, B, C=C, norm_weight=norm_weight, eps=eps, tuned=False)
    D_ref, rstd_ref = gemm_rms_ref(
        A.float(),
        B.float(),
        C=C.float() if C is not None else None,
        norm_weight=norm_weight,
        eps=eps,
    )
    D_pt, rstd_pt = gemm_rms_ref(A, B, C=C, norm_weight=norm_weight, eps=eps)
    assert (D - D_ref).abs().max() < 2 * (D_pt - D_ref).abs().max() + 1e-5
    assert (rstd - rstd_ref).abs().max() < 2 * (rstd_pt - rstd_ref).abs().max() + 1e-3


@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "relu", "gelu_tanh_approx"])
@pytest.mark.parametrize("has_C", [False, True])
@pytest.mark.parametrize("n", [2048])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
def test_gemm_norm_act(input_dtype, k, n, has_C, activation, use_compile, swap_ab):
    from quack.gemm_interface import gemm_norm_act_tuned

    device = "cuda"
    torch.random.manual_seed(0)
    m = 1024
    A = torch.randn(m, k, device=device, dtype=input_dtype)
    B = torch.randn(k, n, device=device, dtype=input_dtype) / math.sqrt(k)
    C = torch.randn(m, n, device=device, dtype=input_dtype) if has_C else None
    rstd = torch.randn(m, device=device, dtype=torch.float32)
    if not swap_ab:
        fn = gemm_norm_act if not use_compile else torch.compile(gemm_norm_act, fullgraph=True)
        preact, postact = fn(
            A,
            B,
            rstd=rstd,
            C=C,
            activation=activation,
            store_preact=True,
            tuned=False,
        )
    else:
        preact = torch.empty(m, n, device=device, dtype=input_dtype)
        postact = torch.empty(m, n, device=device, dtype=input_dtype)
        gemm_norm_act_tuned.fn(
            A,
            B,
            preact,
            postact,
            C,
            rstd,
            activation,
            False,
            config=GemmConfig(swap_ab=True),
        )
    preact_ref, postact_ref = gemm_norm_act_ref(
        A.float(),
        B.float(),
        rstd=rstd,
        C=C.float() if C is not None else None,
        activation=activation,
        store_preact=True,
    )
    preact_pt, postact_pt = gemm_norm_act_ref(
        A,
        B,
        rstd=rstd,
        C=C,
        activation=activation,
        store_preact=True,
    )
    assert (preact - preact_ref).abs().max() < 2 * (preact_pt - preact_ref).abs().max() + 1e-5
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-5


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("activation", ["swiglu", "reglu", "geglu"])
@pytest.mark.parametrize("n", [2048])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
def test_gemm_norm_gated(input_dtype, k, n, activation, use_compile):
    device = "cuda"
    torch.random.manual_seed(0)
    m = 1024
    A = torch.randn(m, k, device=device, dtype=input_dtype)
    B = torch.randn(k, n, device=device, dtype=input_dtype) / math.sqrt(k)
    rstd = torch.randn(m, device=device, dtype=torch.float32)
    fn = gemm_norm_act if not use_compile else torch.compile(gemm_norm_act, fullgraph=True)
    preact, postact = fn(
        A,
        B,
        rstd=rstd,
        activation=activation,
        store_preact=True,
        tuned=False,
    )
    preact_ref, postact_ref = gemm_norm_act_ref(
        A.float(),
        B.float(),
        rstd=rstd,
        activation=activation,
        store_preact=True,
    )
    preact_pt, postact_pt = gemm_norm_act_ref(
        A,
        B,
        rstd=rstd,
        activation=activation,
        store_preact=True,
    )
    assert (preact - preact_ref).abs().max() < 2 * (preact_pt - preact_ref).abs().max() + 1e-5
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-5


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu_tanh_approx"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
def test_gemm_rms_then_norm_act(input_dtype, activation, use_compile):
    """End-to-end: gemm_rms + gemm_norm_act vs PyTorch activation(rmsnorm(x @ W2 + res) @ W1).

    Transformer pattern:
      D = x @ W2 + residual                       ← gemm_rms step 1
      rstd = rsqrt(mean(D^2) + eps)               ← gemm_rms step 2
      rmsnorm(D) = D * rstd * w                   ← element-wise
      postact = activation(rmsnorm(D) @ W1)        ← GEMM + activation

    Since rmsnorm(D) @ W1 = (D * rstd * w) @ W1 = (D @ diag(w) @ W1) * rstd,
    we pre-fuse w into W1 and use only rstd as the colvec:
      W1_fused = diag(w) @ W1
      postact = gemm_norm_act(D, W1_fused, rstd=rstd, activation=act)
    """
    device = "cuda"
    torch.random.manual_seed(0)
    m, k, n = 1024, 4096, 2048
    eps = 1e-6

    x = torch.randn(m, k, device=device, dtype=input_dtype)
    W2 = torch.randn(k, k, device=device, dtype=input_dtype) / math.sqrt(k)
    residual = torch.randn(m, k, device=device, dtype=input_dtype)
    W1 = torch.randn(k, n, device=device, dtype=input_dtype) / math.sqrt(k)
    w = torch.randn(k, device=device, dtype=input_dtype)

    # Step 1: fused gemm_rms with norm_weight → D_normed = (x @ W2 + residual) * w, rstd
    fn_rms = gemm_rms if not use_compile else torch.compile(gemm_rms, fullgraph=True)
    D_normed, rstd = fn_rms(x, W2, C=residual, norm_weight=w, eps=eps, tuned=False)

    # Step 2: fused gemm_norm_act → postact = activation((D_normed @ W1) * rstd)
    fn_norm_act = gemm_norm_act if not use_compile else torch.compile(gemm_norm_act, fullgraph=True)
    _, postact = fn_norm_act(D_normed, W1, rstd=rstd, activation=activation, tuned=False)

    # Reference: PyTorch fp32
    D_ref = x.float() @ W2.float() + residual.float()
    rstd_ref = torch.rsqrt(D_ref.square().mean(dim=-1) + eps)
    normed = D_ref * rstd_ref.unsqueeze(-1) * w.float()
    preact = normed @ W1.float()
    act_map = {
        None: lambda x: x,
        "silu": F.silu,
        "gelu_tanh_approx": lambda x: F.gelu(x, approximate="tanh"),
    }
    postact_ref = act_map[activation](preact).to(input_dtype)

    # PyTorch bf16 baseline (for error comparison)
    D_pt = x @ W2 + residual
    rstd_pt = torch.rsqrt(D_pt.float().square().mean(dim=-1) + eps)
    normed_pt = D_pt.float() * rstd_pt.unsqueeze(-1) * w.float()
    preact_pt = normed_pt @ W1.float()
    postact_pt = act_map[activation](preact_pt.float()).to(input_dtype)

    err = (postact - postact_ref).abs().max().item()
    err_pt = (postact_pt - postact_ref).abs().max().item()
    print(
        f"act={activation}: err={err:.4f}, err_pt={err_pt:.4f}, ratio={err / (err_pt + 1e-10):.2f}"
    )
    assert err < 2 * err_pt + 1e-4
