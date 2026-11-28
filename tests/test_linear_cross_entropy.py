# Copyright (c) 2025, Tri Dao.

import pytest
import torch

from quack.linear_cross_entropy import (
    chunked_linear_cross_entropy,
    linear_cross_entropy_func_ref,
    scaled_exp_lce_supported,
    scaled_exp_linear_cross_entropy,
)


def _make_token_weights(target, dtype, ignore_index=-100):
    if dtype is None:
        return None
    weights = torch.rand(*target.shape, 2, device=target.device, dtype=dtype)[..., 0]
    weights[..., ::7] = 0
    weights[target == ignore_index] = float("nan")
    return weights.detach().requires_grad_()


def _linear_ce_ref(x, weight, target, reduction="mean", ignore_index=-100, token_weights=None):
    if token_weights is None:
        return linear_cross_entropy_func_ref(
            x, weight, None, target, ignore_index=ignore_index, reduction=reduction
        )
    loss = linear_cross_entropy_func_ref(
        x, weight, None, target, ignore_index=ignore_index, reduction="none"
    ).float()
    loss = (loss * token_weights.detach().float().masked_fill(target == ignore_index, 0)).sum()
    return loss if reduction == "sum" else loss / (target != ignore_index).sum()


def _require_scaled_exp(x, weight, chunk_size, reduction="mean"):
    if not scaled_exp_lce_supported(x, weight, chunk_size, reduction):
        pytest.skip("scaled-exp LCE unsupported here (needs SM90 + bf16 + V % 128 == 0)")


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("V", [32000, 50264, 128256])
# @pytest.mark.parametrize("V", [32000])
@pytest.mark.parametrize("d", [768, 1024])
# @pytest.mark.parametrize("d", [768])
@pytest.mark.parametrize("B_L", [8, 16, 24])
@pytest.mark.parametrize("chunk_size", [16])
@pytest.mark.parametrize("token_weights_dtype", [None, torch.float32])
def test_chunked_linear_cross_entropy(
    B_L, d, V, chunk_size, reduction, input_dtype, token_weights_dtype
):
    """Test chunked linear cross entropy against reference implementation."""
    device = "cuda"
    atol, rtol = 1e-3, 1e-3
    torch.random.manual_seed(0)
    x = (torch.randn(B_L, d, device=device, dtype=input_dtype) * 0.1).requires_grad_()
    weight = (torch.randn(V, d, device=device, dtype=input_dtype) / (d**0.5)).requires_grad_()
    target = torch.randint(0, V, (B_L,), device=device, dtype=torch.int64)
    token_weights = _make_token_weights(target, token_weights_dtype)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    x_pt = x.detach().clone().requires_grad_(True)
    weight_pt = weight.detach().clone().requires_grad_(True)
    loss_ref = _linear_ce_ref(
        x_ref.float(),
        weight_ref.float(),
        target,
        reduction=reduction,
        token_weights=token_weights,
    )
    loss_pt = _linear_ce_ref(
        x_pt, weight_pt, target, reduction=reduction, token_weights=token_weights
    )
    # Chunked implementation
    loss = chunked_linear_cross_entropy(
        x,
        weight,
        target,
        chunk_size=chunk_size,
        reduction=reduction,
        tuned=False,
        token_weights=token_weights,
    )
    assert (loss - loss_ref).abs().max() < 3 * (loss_pt - loss_ref).abs().max() + 1e-5
    loss.backward()
    loss_ref.backward()
    loss_pt.backward()
    assert (x.grad - x_ref.grad).abs().max() < 2 * (x_pt.grad - x_ref.grad).abs().max() + 1e-4
    assert (weight.grad - weight_ref.grad).abs().max() < 2 * (
        weight_pt.grad - weight_ref.grad
    ).abs().max() + 1e-4

    if token_weights is not None:
        assert token_weights.grad is None


@pytest.mark.parametrize("use_scaled_exp", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("V", [1536, 2048])  # tile_n1 192 / 256
@pytest.mark.parametrize("d", [512, 768])  # dx/dw tile_N 256 / 192
@pytest.mark.parametrize("B_L", [128, 600, 1024])  # single chunk / ragged padded / multi-chunk
@pytest.mark.parametrize(
    "token_weights_dtype", [None, torch.float16, torch.bfloat16, torch.float32]
)
def test_scaled_exp_linear_cross_entropy(B_L, d, V, reduction, use_scaled_exp, token_weights_dtype):
    """Scaled-exp pipeline vs fp32 reference at bf16-baseline-relative
    tolerance, across chunk shapes (incl. the padded ragged last chunk) and
    both gemm1 / grad-GEMM tile classes, with ignored targets mixed in.
    use_scaled_exp=False pins the base pipeline at the same shapes."""
    device = "cuda"
    chunk_size = 256
    ignore_index = -100
    torch.random.manual_seed(0)
    x = (torch.randn(B_L, d, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_()
    weight = (torch.randn(V, d, device=device, dtype=torch.bfloat16) / (d**0.5)).requires_grad_()
    target = torch.randint(0, V, (B_L,), device=device, dtype=torch.int64)
    target[torch.rand(B_L, device=device) < 0.15] = ignore_index
    token_weights = _make_token_weights(target, token_weights_dtype)
    if use_scaled_exp:
        _require_scaled_exp(x, weight, chunk_size, reduction)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    x_pt = x.detach().clone().requires_grad_(True)
    weight_pt = weight.detach().clone().requires_grad_(True)
    loss_ref = _linear_ce_ref(
        x_ref.float(),
        weight_ref.float(),
        target,
        reduction=reduction,
        token_weights=token_weights,
    )
    loss_pt = _linear_ce_ref(
        x_pt, weight_pt, target, reduction=reduction, token_weights=token_weights
    )
    loss = chunked_linear_cross_entropy(
        x,
        weight,
        target,
        chunk_size=chunk_size,
        reduction=reduction,
        tuned=False,
        use_scaled_exp=use_scaled_exp,
        token_weights=token_weights,
    )
    assert (loss - loss_ref).abs().max() < 3 * (loss_pt - loss_ref).abs().max() + 1e-5
    loss.backward()
    loss_ref.backward()
    loss_pt.backward()
    assert (x.grad - x_ref.grad).abs().max() < 2 * (x_pt.grad - x_ref.grad).abs().max() + 1e-4
    assert (weight.grad - weight_ref.grad).abs().max() < 2 * (
        weight_pt.grad - weight_ref.grad
    ).abs().max() + 1e-4
    if token_weights is not None:
        assert token_weights.grad is None


@pytest.mark.parametrize("token_weights_dtype", [None, torch.float32])
def test_scaled_exp_linear_cross_entropy_batched(token_weights_dtype):
    """(B, L, d) input through the public fn: grads keep the batch shape."""
    device = "cuda"
    B, L, d, V = 3, 100, 768, 1536
    torch.random.manual_seed(2)
    x = (torch.randn(B, L, d, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_()
    weight = (torch.randn(V, d, device=device, dtype=torch.bfloat16) / (d**0.5)).requires_grad_()
    target = torch.randint(0, V, (B, L), device=device, dtype=torch.int64)
    token_weights = _make_token_weights(target, token_weights_dtype)
    _require_scaled_exp(x, weight, 128)
    x_ref = x.detach().float().requires_grad_(True)
    weight_ref = weight.detach().float().requires_grad_(True)
    loss_ref = _linear_ce_ref(
        x_ref.reshape(-1, d),
        weight_ref,
        target.reshape(-1),
        reduction="mean",
        token_weights=token_weights.reshape(-1) if token_weights is not None else None,
    )
    loss = scaled_exp_linear_cross_entropy(
        x, weight, target, chunk_size=128, token_weights=token_weights
    )
    assert (loss - loss_ref).abs().max() < 1e-3 * loss_ref.abs().max() + 1e-5
    loss.backward()
    loss_ref.backward()
    assert x.grad.shape == x.shape
    assert (x.grad.float() - x_ref.grad).abs().max() < 1e-2
    assert (weight.grad.float() - weight_ref.grad).abs().max() < 1e-2
    if token_weights is not None:
        assert token_weights.grad is None


@pytest.mark.parametrize("token_weights_dtype", [None, torch.float32])
def test_chunked_linear_cross_entropy_torch_compile_dispatch(token_weights_dtype):
    device = "cuda"
    B_L, d, V = 256, 512, 2048
    torch.random.manual_seed(3)
    x0 = torch.randn(B_L, d, device=device, dtype=torch.bfloat16) * 0.1
    w0 = torch.randn(V, d, device=device, dtype=torch.bfloat16) / (d**0.5)
    target = torch.randint(0, V, (B_L,), device=device)
    target[::7] = -100
    token_weights = _make_token_weights(target, token_weights_dtype)
    _require_scaled_exp(x0, w0, 128)

    def f(x, w, token_weights):
        return chunked_linear_cross_entropy(
            x,
            w,
            target,
            chunk_size=128,
            reduction="sum",
            tuned=False,
            token_weights=token_weights,
        )

    compiled = torch.compile(f, fullgraph=True)
    for iteration in range(2 if token_weights is not None else 1):
        if iteration:
            with torch.no_grad():
                token_weights.uniform_()
        x_e, w_e = x0.clone().requires_grad_(), w0.clone().requires_grad_()
        loss_e = f(x_e, w_e, token_weights)
        loss_e.backward()
        x_c, w_c = x0.clone().requires_grad_(), w0.clone().requires_grad_()
        loss_c = compiled(x_c, w_c, token_weights)
        loss_c.backward()
        torch.testing.assert_close(loss_c, loss_e.detach(), atol=1e-6, rtol=1e-6)
        assert torch.equal(x_c.grad, x_e.grad), "compiled dx != eager scaled-exp dx"
        assert torch.equal(w_c.grad, w_e.grad), "compiled dw != eager scaled-exp dw"

    def f_mean(x, w, token_weights):
        return chunked_linear_cross_entropy(
            x, w, target, chunk_size=128, tuned=False, token_weights=token_weights
        )

    # Inductor and aten reciprocals may differ by one fp32 ulp for mean reduction.
    compiled_mean = torch.compile(f_mean, fullgraph=True)
    for iteration in range(2 if token_weights is not None else 1):
        if iteration:
            with torch.no_grad():
                token_weights.uniform_()
        x_m, w_m = x0.clone().requires_grad_(), w0.clone().requires_grad_()
        f_mean(x_m, w_m, token_weights).backward()
        x_mc, w_mc = x0.clone().requires_grad_(), w0.clone().requires_grad_()
        compiled_mean(x_mc, w_mc, token_weights).backward()
        torch.testing.assert_close(x_mc.grad, x_m.grad, atol=1e-6, rtol=1e-2)
        torch.testing.assert_close(w_mc.grad, w_m.grad, atol=1e-6, rtol=1e-2)
    if token_weights is not None:
        assert token_weights.grad is None


@pytest.mark.parametrize("use_scaled_exp", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("chunk_size", [256, 1024])
@pytest.mark.parametrize("token_weights_dtype", [None, torch.float32])
def test_chunked_linear_cross_entropy_ignore_index(
    input_dtype, reduction, chunk_size, use_scaled_exp, token_weights_dtype
):
    """Test chunked linear cross entropy with ignore_index."""
    device = "cuda"
    B_L, d, V = 1024, 512, 2048
    ignore_index = V - 1
    atol, rtol = 1e-3, 1e-3
    torch.random.manual_seed(42)
    x = (torch.randn(B_L, d, device=device, dtype=input_dtype) * 0.1).requires_grad_()
    weight = (torch.randn(V, d, device=device, dtype=input_dtype) / (d**0.5)).requires_grad_()
    target = torch.randint(0, V, (B_L,), device=device, dtype=torch.int64)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    x_pt = x.detach().clone().requires_grad_(True)
    weight_pt = weight.detach().clone().requires_grad_(True)
    # Set some targets to ignore_index
    ignore_mask = torch.rand(B_L, device=device) < 0.2
    target[ignore_mask] = ignore_index
    token_weights = _make_token_weights(target, token_weights_dtype, ignore_index)
    loss_ref = _linear_ce_ref(
        x_ref.float(),
        weight_ref.float(),
        target,
        ignore_index=ignore_index,
        reduction=reduction,
        token_weights=token_weights,
    )
    loss_pt = _linear_ce_ref(
        x_pt,
        weight_pt,
        target,
        ignore_index=ignore_index,
        reduction=reduction,
        token_weights=token_weights,
    )
    if use_scaled_exp:
        _require_scaled_exp(x, weight, chunk_size, reduction)
    # Chunked implementation
    loss = chunked_linear_cross_entropy(
        x,
        weight,
        target,
        chunk_size=chunk_size,
        ignore_index=ignore_index,
        reduction=reduction,
        tuned=False,
        use_scaled_exp=use_scaled_exp,
        token_weights=token_weights,
    )
    assert (loss - loss_ref).abs().max() < 3 * (loss_pt - loss_ref).abs().max() + 1e-5
    loss.backward()
    loss_ref.backward()
    loss_pt.backward()
    assert (x.grad - x_ref.grad).abs().max() < 2 * (x_pt.grad - x_ref.grad).abs().max() + 1e-4
    assert (weight.grad - weight_ref.grad).abs().max() < 2 * (
        weight_pt.grad - weight_ref.grad
    ).abs().max() + 1e-4

    if token_weights is not None:
        assert token_weights.grad is None


@pytest.mark.parametrize("use_scaled_exp", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("chunk_size", [256, 1024])
@pytest.mark.parametrize(
    "token_weights_dtype", [None, torch.float16, torch.bfloat16, torch.float32]
)
def test_chunked_linear_cross_entropy_no_grad(
    reduction, chunk_size, use_scaled_exp, token_weights_dtype
):
    """Loss-only path (no_grad / nothing requires grad): must match the
    training-path loss while skipping the dx/dw GEMMs and fp32 accumulator."""
    device = "cuda"
    B_L, d, V = 1024, 512, 2048
    torch.random.manual_seed(0)
    x = (torch.randn(B_L, d, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_()
    weight = (torch.randn(V, d, device=device, dtype=torch.bfloat16) / (d**0.5)).requires_grad_()
    target = torch.randint(0, V, (B_L,), device=device, dtype=torch.int64)
    token_weights = _make_token_weights(target, token_weights_dtype)
    if use_scaled_exp:
        _require_scaled_exp(x, weight, chunk_size, reduction)
    loss_train = chunked_linear_cross_entropy(
        x,
        weight,
        target,
        chunk_size=chunk_size,
        reduction=reduction,
        tuned=False,
        use_scaled_exp=use_scaled_exp,
        token_weights=token_weights,
    )
    with torch.no_grad():
        loss_eval = chunked_linear_cross_entropy(
            x,
            weight,
            target,
            chunk_size=chunk_size,
            reduction=reduction,
            tuned=False,
            use_scaled_exp=use_scaled_exp,
            token_weights=token_weights,
        )
    assert not loss_eval.requires_grad
    assert torch.allclose(loss_eval, loss_train.detach(), atol=1e-5, rtol=1e-5)
    loss_frozen = chunked_linear_cross_entropy(
        x.detach(),
        weight.detach(),
        target,
        chunk_size=chunk_size,
        reduction=reduction,
        tuned=False,
        use_scaled_exp=use_scaled_exp,
        token_weights=token_weights,
    )
    assert not loss_frozen.requires_grad
    torch.testing.assert_close(loss_frozen, loss_eval, atol=1e-5, rtol=1e-5)
    loss_ref = _linear_ce_ref(
        x.float(),
        weight.float(),
        target,
        reduction=reduction,
        token_weights=token_weights,
    )
    torch.testing.assert_close(loss_eval, loss_ref, atol=0.003, rtol=0.003)


@pytest.mark.parametrize("use_scaled_exp", [False, True])
@pytest.mark.parametrize("frozen", ["x", "weight"])
@pytest.mark.parametrize("chunk_size", [256, 1024])  # 1024 = single chunk (deferred-dw edge)
@pytest.mark.parametrize(
    "token_weights_dtype", [None, torch.float16, torch.bfloat16, torch.float32]
)
def test_chunked_linear_cross_entropy_partial_grad(
    frozen, chunk_size, use_scaled_exp, token_weights_dtype
):
    """One input frozen: its gradient GEMMs are skipped, the other's gradient
    is unchanged vs the both-require-grad run."""
    device = "cuda"
    B_L, d, V = 1024, 512, 2048
    torch.random.manual_seed(0)
    x0 = torch.randn(B_L, d, device=device, dtype=torch.bfloat16) * 0.1
    w0 = torch.randn(V, d, device=device, dtype=torch.bfloat16) / (d**0.5)
    target = torch.randint(0, V, (B_L,), device=device, dtype=torch.int64)
    token_weights = _make_token_weights(target, token_weights_dtype)

    if use_scaled_exp:
        _require_scaled_exp(x0, w0, chunk_size)
    x_full, w_full = x0.clone().requires_grad_(), w0.clone().requires_grad_()
    chunked_linear_cross_entropy(
        x_full,
        w_full,
        target,
        chunk_size=chunk_size,
        tuned=False,
        use_scaled_exp=use_scaled_exp,
        token_weights=token_weights,
    ).backward()

    x = x0.clone().requires_grad_(frozen != "x")
    w = w0.clone().requires_grad_(frozen != "weight")
    chunked_linear_cross_entropy(
        x,
        w,
        target,
        chunk_size=chunk_size,
        tuned=False,
        use_scaled_exp=use_scaled_exp,
        token_weights=token_weights,
    ).backward()
    if frozen == "x":
        assert x.grad is None
        assert torch.allclose(w.grad, w_full.grad, atol=1e-5, rtol=1e-3)
    else:
        assert w.grad is None
        assert torch.allclose(x.grad, x_full.grad, atol=1e-5, rtol=1e-3)

    if token_weights is not None:
        assert token_weights.grad is None


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_scaled_exp_unit_weights_square_dx(reduction):
    torch.manual_seed(23)
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) / 8).requires_grad_()
    if not scaled_exp_lce_supported(x, w, 128, reduction):
        pytest.skip("scaled-exp LCE unsupported here (needs SM90 + bf16 + V % 128 == 0)")
    target = torch.randint(256, (128,), device="cuda")
    target[::11] = -100
    loss = scaled_exp_linear_cross_entropy(x, w, target, chunk_size=128, reduction=reduction)
    weighted_loss = scaled_exp_linear_cross_entropy(
        x,
        w,
        target,
        chunk_size=128,
        reduction=reduction,
        token_weights=torch.ones_like(target, dtype=torch.float32),
    )
    torch.testing.assert_close(weighted_loss, loss, rtol=0, atol=0)
    grads = torch.autograd.grad(loss, (x, w))
    weighted_grads = torch.autograd.grad(weighted_loss, (x, w))
    for actual, expected in zip(weighted_grads, grads):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
