# Copyright (c) 2025, Tri Dao.

import pytest
import torch

from quack.linear_cross_entropy import (
    chunked_linear_cross_entropy,
    linear_cross_entropy_func_ref,
    scaled_exp_lce_supported,
    scaled_exp_linear_cross_entropy,
)


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
def test_chunked_linear_cross_entropy(B_L, d, V, chunk_size, reduction, input_dtype):
    """Test chunked linear cross entropy against reference implementation."""
    device = "cuda"
    atol, rtol = 1e-3, 1e-3
    torch.random.manual_seed(0)
    x = (torch.randn(B_L, d, device=device, dtype=input_dtype) * 0.1).requires_grad_()
    weight = (torch.randn(V, d, device=device, dtype=input_dtype) / (d**0.5)).requires_grad_()
    target = torch.randint(0, V, (B_L,), device=device, dtype=torch.int64)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    x_pt = x.detach().clone().requires_grad_(True)
    weight_pt = weight.detach().clone().requires_grad_(True)
    loss_ref = linear_cross_entropy_func_ref(
        x_ref.float(), weight_ref.float(), None, target, reduction=reduction
    )
    loss_pt = linear_cross_entropy_func_ref(x_pt, weight_pt, None, target, reduction=reduction)
    # Chunked implementation
    loss = chunked_linear_cross_entropy(
        x, weight, target, chunk_size=chunk_size, reduction=reduction, tuned=False
    )
    assert (loss - loss_ref).abs().max() < 3 * (loss_pt - loss_ref).abs().max() + 1e-5
    loss.backward()
    loss_ref.backward()
    loss_pt.backward()
    assert (x.grad - x_ref.grad).abs().max() < 2 * (x_pt.grad - x_ref.grad).abs().max() + 1e-4
    assert (weight.grad - weight_ref.grad).abs().max() < 2 * (
        weight_pt.grad - weight_ref.grad
    ).abs().max() + 1e-4


@pytest.mark.parametrize("use_scaled_exp", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("V", [1536, 2048])  # tile_n1 192 / 256
@pytest.mark.parametrize("d", [512, 768])  # dx/dw tile_N 256 / 192
@pytest.mark.parametrize("B_L", [128, 600, 1024])  # single chunk / ragged padded / multi-chunk
def test_scaled_exp_linear_cross_entropy(B_L, d, V, reduction, use_scaled_exp):
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
    if use_scaled_exp:
        _require_scaled_exp(x, weight, chunk_size, reduction)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    x_pt = x.detach().clone().requires_grad_(True)
    weight_pt = weight.detach().clone().requires_grad_(True)
    loss_ref = linear_cross_entropy_func_ref(
        x_ref.float(), weight_ref.float(), None, target, reduction=reduction
    )
    loss_pt = linear_cross_entropy_func_ref(x_pt, weight_pt, None, target, reduction=reduction)
    loss = chunked_linear_cross_entropy(
        x,
        weight,
        target,
        chunk_size=chunk_size,
        reduction=reduction,
        tuned=False,
        use_scaled_exp=use_scaled_exp,
    )
    assert (loss - loss_ref).abs().max() < 3 * (loss_pt - loss_ref).abs().max() + 1e-5
    loss.backward()
    loss_ref.backward()
    loss_pt.backward()
    assert (x.grad - x_ref.grad).abs().max() < 2 * (x_pt.grad - x_ref.grad).abs().max() + 1e-4
    assert (weight.grad - weight_ref.grad).abs().max() < 2 * (
        weight_pt.grad - weight_ref.grad
    ).abs().max() + 1e-4


def test_scaled_exp_linear_cross_entropy_batched():
    """(B, L, d) input through the public fn: grads keep the batch shape."""
    device = "cuda"
    B, L, d, V = 3, 100, 768, 1536
    torch.random.manual_seed(2)
    x = (torch.randn(B, L, d, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_()
    weight = (torch.randn(V, d, device=device, dtype=torch.bfloat16) / (d**0.5)).requires_grad_()
    target = torch.randint(0, V, (B, L), device=device, dtype=torch.int64)
    _require_scaled_exp(x, weight, 128)
    x_ref = x.detach().float().requires_grad_(True)
    weight_ref = weight.detach().float().requires_grad_(True)
    loss_ref = linear_cross_entropy_func_ref(
        x_ref.reshape(-1, d), weight_ref, None, target.reshape(-1), reduction="mean"
    )
    loss = scaled_exp_linear_cross_entropy(x, weight, target, chunk_size=128)
    assert (loss - loss_ref).abs().max() < 1e-3 * loss_ref.abs().max() + 1e-5
    loss.backward()
    loss_ref.backward()
    assert x.grad.shape == x.shape
    assert (x.grad.float() - x_ref.grad).abs().max() < 1e-2
    assert (weight.grad.float() - weight_ref.grad).abs().max() < 1e-2


def test_chunked_linear_cross_entropy_torch_compile_dispatch():
    """Under torch.compile the scaled-exp path records the single
    quack::lce_scaled_exp_fwd custom op — the host chunk loop (mod.gemm plan
    machinery, jit-cache probes, Triton launches) never gets traced. Same
    kernels either way, so compiled grads match eager scaled-exp bitwise."""
    device = "cuda"
    B_L, d, V = 256, 512, 2048
    torch.random.manual_seed(3)
    x0 = torch.randn(B_L, d, device=device, dtype=torch.bfloat16) * 0.1
    w0 = torch.randn(V, d, device=device, dtype=torch.bfloat16) / (d**0.5)
    target = torch.randint(0, V, (B_L,), device=device)
    target[::7] = -100
    if not scaled_exp_lce_supported(x0, w0, 128, "mean"):
        pytest.skip("shape not scaled-exp eligible; dispatch test needs an eligible shape")

    # sum reduction: no grad_scale scalar, so compiled and eager run byte-
    # identical inputs through identical kernels -> bitwise grads. (mean's
    # 1/num_valid is Inductor-codegen'd under compile and can differ from
    # aten's reciprocal by one fp32 ulp — covered at tolerance below.)
    def f(x, w):
        return chunked_linear_cross_entropy(
            x, w, target, chunk_size=128, reduction="sum", tuned=False
        )

    x_e, w_e = x0.clone().requires_grad_(), w0.clone().requires_grad_()
    loss_e = f(x_e, w_e)  # eager: auto routes to scaled-exp
    loss_e.backward()
    x_c, w_c = x0.clone().requires_grad_(), w0.clone().requires_grad_()
    loss_c = torch.compile(f, fullgraph=True)(x_c, w_c)
    loss_c.backward()
    torch.testing.assert_close(loss_c, loss_e.detach(), atol=1e-6, rtol=1e-6)
    assert torch.equal(x_c.grad, x_e.grad), "compiled dx != eager scaled-exp dx"
    assert torch.equal(w_c.grad, w_e.grad), "compiled dw != eager scaled-exp dw"

    def f_mean(x, w):
        return chunked_linear_cross_entropy(x, w, target, chunk_size=128, tuned=False)

    x_m, w_m = x0.clone().requires_grad_(), w0.clone().requires_grad_()
    f_mean(x_m, w_m).backward()
    x_mc, w_mc = x0.clone().requires_grad_(), w0.clone().requires_grad_()
    torch.compile(f_mean, fullgraph=True)(x_mc, w_mc).backward()
    torch.testing.assert_close(x_mc.grad, x_m.grad, atol=1e-6, rtol=1e-2)
    torch.testing.assert_close(w_mc.grad, w_m.grad, atol=1e-6, rtol=1e-2)


@pytest.mark.parametrize("use_scaled_exp", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("chunk_size", [256, 1024])
def test_chunked_linear_cross_entropy_ignore_index(
    input_dtype, reduction, chunk_size, use_scaled_exp
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
    loss_ref = linear_cross_entropy_func_ref(
        x_ref.float(),
        weight_ref.float(),
        None,
        target,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    loss_pt = linear_cross_entropy_func_ref(
        x_pt, weight_pt, None, target, ignore_index=ignore_index, reduction=reduction
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
    )
    assert (loss - loss_ref).abs().max() < 3 * (loss_pt - loss_ref).abs().max() + 1e-5
    loss.backward()
    loss_ref.backward()
    loss_pt.backward()
    assert (x.grad - x_ref.grad).abs().max() < 2 * (x_pt.grad - x_ref.grad).abs().max() + 1e-4
    assert (weight.grad - weight_ref.grad).abs().max() < 2 * (
        weight_pt.grad - weight_ref.grad
    ).abs().max() + 1e-4


@pytest.mark.parametrize("use_scaled_exp", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("chunk_size", [256, 1024])
def test_chunked_linear_cross_entropy_no_grad(reduction, chunk_size, use_scaled_exp):
    """Loss-only path (no_grad / nothing requires grad): must match the
    training-path loss while skipping the dx/dw GEMMs and fp32 accumulator."""
    device = "cuda"
    B_L, d, V = 1024, 512, 2048
    torch.random.manual_seed(0)
    x = (torch.randn(B_L, d, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_()
    weight = (torch.randn(V, d, device=device, dtype=torch.bfloat16) / (d**0.5)).requires_grad_()
    target = torch.randint(0, V, (B_L,), device=device, dtype=torch.int64)
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
        )
    assert not loss_eval.requires_grad
    assert torch.allclose(loss_eval, loss_train.detach(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_scaled_exp", [False, True])
@pytest.mark.parametrize("frozen", ["x", "weight"])
@pytest.mark.parametrize("chunk_size", [256, 1024])  # 1024 = single chunk (deferred-dw edge)
def test_chunked_linear_cross_entropy_partial_grad(frozen, chunk_size, use_scaled_exp):
    """One input frozen: its gradient GEMMs are skipped, the other's gradient
    is unchanged vs the both-require-grad run."""
    device = "cuda"
    B_L, d, V = 1024, 512, 2048
    torch.random.manual_seed(0)
    x0 = torch.randn(B_L, d, device=device, dtype=torch.bfloat16) * 0.1
    w0 = torch.randn(V, d, device=device, dtype=torch.bfloat16) / (d**0.5)
    target = torch.randint(0, V, (B_L,), device=device, dtype=torch.int64)

    if use_scaled_exp:
        _require_scaled_exp(x0, w0, chunk_size)
    x_full, w_full = x0.clone().requires_grad_(), w0.clone().requires_grad_()
    chunked_linear_cross_entropy(
        x_full, w_full, target, chunk_size=chunk_size, tuned=False, use_scaled_exp=use_scaled_exp
    ).backward()

    x = x0.clone().requires_grad_(frozen != "x")
    w = w0.clone().requires_grad_(frozen != "weight")
    chunked_linear_cross_entropy(
        x, w, target, chunk_size=chunk_size, tuned=False, use_scaled_exp=use_scaled_exp
    ).backward()
    if frozen == "x":
        assert x.grad is None
        assert torch.allclose(w.grad, w_full.grad, atol=1e-5, rtol=1e-3)
    else:
        assert w.grad is None
        assert torch.allclose(x.grad, x_full.grad, atol=1e-5, rtol=1e-3)
