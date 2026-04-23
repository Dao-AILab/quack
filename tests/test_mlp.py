# Copyright (C) 2025, Tri Dao.
import pytest
import torch
import torch.nn.functional as F
import torch._dynamo

from quack.mlp import MLP
from quack.gemm_interface import act_to_pytorch_fn_map, gated_to_pytorch_fn_map

torch._dynamo.config.cache_size_limit = 64


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("activation", ["gelu_tanh_approx", "relu", "swiglu", "reglu", "geglu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_features", [512])
@pytest.mark.parametrize("in_features", [512])
def test_mlp(in_features, hidden_features, dtype, activation, use_compile):
    device = "cuda"
    gated = activation in gated_to_pytorch_fn_map
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12 and gated:
        pytest.skip("SM120 gated dactivation GEMM epilogue is not yet supported")
    torch.random.manual_seed(0)
    batch = 256
    mlp = MLP(
        in_features, hidden_features, activation=activation, device=device, dtype=dtype, tuned=False
    )
    if gated:
        assert mlp.fc1.out_features == 2 * hidden_features
        assert mlp.fc2.in_features == hidden_features
    w1_ref = mlp.fc1.weight.detach().clone().float()
    w2_ref = mlp.fc2.weight.detach().clone().float()
    if use_compile:
        mlp = torch.compile(mlp, fullgraph=True)
    x = torch.randn(batch, in_features, device=device, dtype=dtype, requires_grad=True)
    out = mlp(x)
    assert out.shape == (batch, in_features)
    # Reference
    x_ref = x.detach().clone().float().requires_grad_(True)
    y_ref = F.linear(x_ref, w1_ref)
    if gated:
        y_ref = gated_to_pytorch_fn_map[activation](y_ref[..., ::2], y_ref[..., 1::2])
    else:
        y_ref = act_to_pytorch_fn_map[activation](y_ref)
    out_ref = F.linear(y_ref, w2_ref)
    assert (out.float() - out_ref).abs().max() < 1e-2
    # Backward
    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout.float())
    assert x.grad is not None
    assert (x.grad.float() - x_ref.grad).abs().max() < 1e-2


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("activation", ["gelu_tanh_approx", "swiglu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mlp_zero_stride_grad(dtype, activation, use_compile):
    """out.sum().backward() produces expanded gradient with zero strides."""
    device = "cuda"
    gated = activation in gated_to_pytorch_fn_map
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12 and gated:
        pytest.skip("SM120 gated dactivation GEMM epilogue is not yet supported")
    torch.random.manual_seed(0)
    mlp = MLP(512, 512, activation=activation, device=device, dtype=dtype, tuned=False)
    w1_ref = mlp.fc1.weight.detach().clone().float()
    w2_ref = mlp.fc2.weight.detach().clone().float()
    if use_compile:
        mlp = torch.compile(mlp, fullgraph=True)
    x = torch.randn(256, 512, device=device, dtype=dtype, requires_grad=True)
    out = mlp(x)
    out.sum().backward()
    assert x.grad is not None
    # Reference
    x_ref = x.detach().clone().float().requires_grad_(True)
    y_ref = F.linear(x_ref, w1_ref)
    if gated:
        y_ref = gated_to_pytorch_fn_map[activation](y_ref[..., ::2], y_ref[..., 1::2])
    else:
        y_ref = act_to_pytorch_fn_map[activation](y_ref)
    out_ref = F.linear(y_ref, w2_ref)
    out_ref.sum().backward()
    assert (x.grad.float() - x_ref.grad).abs().max() < 1e-2


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("activation", ["gelu_tanh_approx", "swiglu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mlp_recompute(dtype, activation, use_compile):
    """recompute=True should match normal mode and float32 reference."""
    device = "cuda"
    gated = activation in gated_to_pytorch_fn_map
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12 and gated:
        pytest.skip("SM120 gated dactivation GEMM epilogue is not yet supported")
    torch.random.manual_seed(0)
    batch, dim, hidden = 256, 512, 512
    mlp = MLP(dim, hidden, activation=activation, device=device, dtype=dtype, tuned=False)
    x = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(batch, dim, device=device, dtype=dtype)
    # Standard forward/backward
    mlp_std = torch.compile(mlp, fullgraph=True) if use_compile else mlp
    out_std = mlp_std(x)
    out_std.backward(dout)
    dx_std = x.grad.clone()
    dw1_std = mlp.fc1.weight.grad.clone()
    dw2_std = mlp.fc2.weight.grad.clone()
    # Recompute forward/backward (toggle flag on same model)
    x.grad = None
    mlp.zero_grad()
    mlp.recompute = True
    mlp_rec = torch.compile(mlp, fullgraph=True) if use_compile else mlp
    out_rec = mlp_rec(x)
    out_rec.backward(dout)
    dx_rec = x.grad.clone()
    dw1_rec = mlp.fc1.weight.grad.clone()
    dw2_rec = mlp.fc2.weight.grad.clone()
    mlp.recompute = False
    # vs normal mode
    assert (out_std - out_rec).abs().max() < 1e-6, "Output mismatch"
    assert (dx_std - dx_rec).abs().max() < 1e-2, "dx mismatch"
    assert (dw1_std - dw1_rec).abs().max() < 1e-2, "dW1 mismatch"
    assert (dw2_std - dw2_rec).abs().max() < 1e-2, "dW2 mismatch"
    # vs float32 reference
    x_ref = x.detach().clone().float().requires_grad_(True)
    w1_ref = mlp.fc1.weight.detach().clone().float()
    w2_ref = mlp.fc2.weight.detach().clone().float()
    y_ref = F.linear(x_ref, w1_ref)
    if gated:
        y_ref = gated_to_pytorch_fn_map[activation](y_ref[..., ::2], y_ref[..., 1::2])
    else:
        y_ref = act_to_pytorch_fn_map[activation](y_ref)
    out_ref = F.linear(y_ref, w2_ref)
    out_ref.backward(dout.float())
    assert (out_rec.float() - out_ref).abs().max() < 1e-2
    assert (dx_rec.float() - x_ref.grad).abs().max() < 1e-2


@pytest.mark.parametrize("activation", ["gelu_tanh_approx", "swiglu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "freeze", ["x", "weight1", "weight2", "x_weight1", "x_weight2", "weight1_weight2"]
)
def test_mlp_recompute_partial_grad(dtype, activation, freeze):
    """Recompute mode with some inputs frozen (requires_grad=False).

    Tests that needs_input_grad indexing is correct: x=[0], weight1=[1], weight2=[2].
    """
    device = "cuda"
    gated = activation in gated_to_pytorch_fn_map
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12 and gated:
        pytest.skip("SM120 gated dactivation GEMM epilogue is not yet supported")
    torch.random.manual_seed(0)
    batch, dim, hidden = 256, 512, 512
    mlp = MLP(
        dim,
        hidden,
        activation=activation,
        device=device,
        dtype=dtype,
        tuned=False,
        recompute=True,
    )
    x = torch.randn(batch, dim, device=device, dtype=dtype)
    dout = torch.randn(batch, dim, device=device, dtype=dtype)
    frozen = set(freeze.split("_")) if "_" in freeze else {freeze}
    freeze_x = "x" in frozen
    freeze_w1 = "weight1" in frozen
    freeze_w2 = "weight2" in frozen
    x.requires_grad_(not freeze_x)
    mlp.fc1.weight.requires_grad_(not freeze_w1)
    mlp.fc2.weight.requires_grad_(not freeze_w2)
    # Normal (non-recompute) forward/backward as reference
    mlp.recompute = False
    out_ref = mlp(x)
    out_ref.backward(dout)
    dx_ref = x.grad.clone() if x.grad is not None else None
    dw1_ref = mlp.fc1.weight.grad.clone() if mlp.fc1.weight.grad is not None else None
    dw2_ref = mlp.fc2.weight.grad.clone() if mlp.fc2.weight.grad is not None else None
    # Recompute forward/backward
    x.grad = None
    mlp.zero_grad()
    mlp.recompute = True
    out = mlp(x)
    out.backward(dout)
    dx = x.grad.clone() if x.grad is not None else None
    dw1 = mlp.fc1.weight.grad.clone() if mlp.fc1.weight.grad is not None else None
    dw2 = mlp.fc2.weight.grad.clone() if mlp.fc2.weight.grad is not None else None
    # Forward should match exactly
    assert (out - out_ref).abs().max() < 1e-6, "Output mismatch"
    # Gradients: recompute should match normal mode
    if freeze_x:
        assert dx is None and dx_ref is None
    else:
        assert (dx - dx_ref).abs().max() < 1e-2, "dx mismatch"
    if freeze_w1:
        assert dw1 is None and dw1_ref is None
    else:
        assert (dw1 - dw1_ref).abs().max() < 1e-2, "dW1 mismatch"
    if freeze_w2:
        assert dw2 is None and dw2_ref is None
    else:
        assert (dw2 - dw2_ref).abs().max() < 1e-2, "dW2 mismatch"


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("activation", ["swiglu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mlp_concat_layout(dtype, activation, recompute, use_compile):
    device = "cuda"
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
        pytest.skip("SM120 gated dactivation GEMM epilogue is not yet supported")
    torch.random.manual_seed(0)
    batch, dim, hidden = 256, 512, 512
    mlp = MLP(
        dim,
        hidden,
        activation=activation,
        device=device,
        dtype=dtype,
        tuned=False,
        recompute=recompute,
        concat_layout=True,
    )
    # Reference uses concat weight directly. F.linear gives concat output [gate; up].
    # The kernel interleaves, so we interleave the ref's preact before gated activation.
    w1_ref = mlp.fc1.weight.detach().clone().float().requires_grad_(True)
    w2_ref = mlp.fc2.weight.detach().clone().float().requires_grad_(True)
    w1_pt = mlp.fc1.weight.detach().clone().requires_grad_(True)
    w2_pt = mlp.fc2.weight.detach().clone().requires_grad_(True)
    if use_compile:
        mlp = torch.compile(mlp, fullgraph=True)
    x = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    out = mlp(x)
    x_ref = x.detach().clone().float().requires_grad_(True)
    x_pt = x.detach().clone().requires_grad_(True)
    # F.linear(x, w1_concat) gives [gate_cols, up_cols]. Interleave before gated activation.
    y_ref = F.linear(x_ref, w1_ref)
    y_ref = gated_to_pytorch_fn_map[activation](*y_ref.chunk(2, dim=-1))
    out_ref = F.linear(y_ref, w2_ref)
    y_pt = F.linear(x_pt, w1_pt)
    y_pt = gated_to_pytorch_fn_map[activation](*y_pt.chunk(2, dim=-1))
    out_pt = F.linear(y_pt, w2_pt)
    assert (out.float() - out_ref).abs().max() < 1e-2
    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout.float())
    out_pt.backward(dout)
    assert x.grad is not None
    assert x_pt.grad is not None
    assert (x.grad.float() - x_ref.grad).abs().max() < 2 * (
        x_pt.grad.float() - x_ref.grad
    ).abs().max() + 1e-2
    assert mlp.fc1.weight.grad is not None and w1_ref.grad is not None and w1_pt.grad is not None
    assert mlp.fc2.weight.grad is not None and w2_ref.grad is not None and w2_pt.grad is not None
    assert (mlp.fc1.weight.grad.float() - w1_ref.grad).abs().max() < 2 * (
        w1_pt.grad.float() - w1_ref.grad
    ).abs().max() + 1e-2
    assert (mlp.fc2.weight.grad.float() - w2_ref.grad).abs().max() < 2 * (
        w2_pt.grad.float() - w2_ref.grad
    ).abs().max() + 1e-2


@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_bias2", [False, True])
@pytest.mark.parametrize("has_bias1", [False, True])
@pytest.mark.parametrize("activation", ["swiglu", "reglu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mlp_concat_layout_bias_fwd(dtype, activation, has_bias1, has_bias2, bias_dtype):
    """Test MLP concat layout with bias in forward-only (inference) mode."""
    device = "cuda"
    torch.random.manual_seed(0)
    batch, dim, hidden = 256, 512, 512
    mlp = MLP(
        dim,
        hidden,
        activation=activation,
        bias1=has_bias1,
        bias2=has_bias2,
        device=device,
        dtype=dtype,
        tuned=False,
        concat_layout=True,
    )
    # Override bias dtype to test both bf16 (Python permute) and fp32 (kernel layout permute)
    if has_bias1:
        mlp.fc1.bias = torch.nn.Parameter(mlp.fc1.bias.to(bias_dtype))
    if has_bias2:
        mlp.fc2.bias = torch.nn.Parameter(mlp.fc2.bias.to(bias_dtype))
    x = torch.randn(batch, dim, device=device, dtype=dtype)
    with torch.no_grad():
        out = mlp(x)
    # fp32 reference
    w1, w2 = mlp.fc1.weight.float(), mlp.fc2.weight.float()
    b1 = mlp.fc1.bias.float() if has_bias1 else None
    b2 = mlp.fc2.bias.float() if has_bias2 else None
    y_ref = F.linear(x.float(), w1, b1)
    y_ref = gated_to_pytorch_fn_map[activation](*y_ref.chunk(2, dim=-1))
    out_ref = F.linear(y_ref, w2, b2)
    # bf16 baseline (cast bias to input dtype for F.linear compatibility)
    b1_pt = mlp.fc1.bias.to(dtype) if has_bias1 else None
    b2_pt = mlp.fc2.bias.to(dtype) if has_bias2 else None
    y_pt = F.linear(x, mlp.fc1.weight, b1_pt)
    y_pt = gated_to_pytorch_fn_map[activation](*y_pt.chunk(2, dim=-1))
    out_pt = F.linear(y_pt, mlp.fc2.weight, b2_pt)
    assert (out.float() - out_ref).abs().max() < 2 * (out_pt.float() - out_ref).abs().max() + 1e-5


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("activation", ["swiglu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mlp_concat_layout_fuse_grad_accum(dtype, activation, recompute):
    """Test MLP concat layout with fused gradient accumulation (in-place dweight)."""
    device = "cuda"
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
        pytest.skip("SM120 gated dactivation GEMM epilogue is not yet supported")
    torch.random.manual_seed(0)
    batch, dim, hidden = 256, 512, 512
    mlp = MLP(
        dim,
        hidden,
        activation=activation,
        device=device,
        dtype=dtype,
        tuned=False,
        recompute=recompute,
        concat_layout=True,
        fuse_grad_accum=True,
    )
    # Pre-initialize grads with random values to test accumulation into existing grads
    mlp.fc1.weight.grad = torch.randn_like(mlp.fc1.weight)
    mlp.fc2.weight.grad = torch.randn_like(mlp.fc2.weight)
    w1_grad_init = mlp.fc1.weight.grad.clone()
    w2_grad_init = mlp.fc2.weight.grad.clone()
    # Reference without fuse_grad_accum
    mlp_ref = MLP(
        dim,
        hidden,
        activation=activation,
        device=device,
        dtype=dtype,
        tuned=False,
        recompute=recompute,
        concat_layout=True,
        fuse_grad_accum=False,
    )
    mlp_ref.fc1.weight.data.copy_(mlp.fc1.weight.data)
    mlp_ref.fc2.weight.data.copy_(mlp.fc2.weight.data)
    x = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    dout = torch.randn(batch, mlp.fc2.out_features, device=device, dtype=dtype)
    out = mlp(x)
    out.backward(dout)
    out_ref = mlp_ref(x_ref)
    out_ref.backward(dout)
    # fuse_grad_accum adds to existing grad; non-fused replaces.
    # Check that accumulated = init + new_grad, with tolerance relative to grad magnitude.
    w1_expected = w1_grad_init + mlp_ref.fc1.weight.grad
    w2_expected = w2_grad_init + mlp_ref.fc2.weight.grad
    w1_atol = 1e-2 * w1_expected.abs().mean()
    w2_atol = 1e-2 * w2_expected.abs().mean()
    assert (mlp.fc1.weight.grad - w1_expected).abs().max() < w1_atol
    assert (mlp.fc2.weight.grad - w2_expected).abs().max() < w2_atol
