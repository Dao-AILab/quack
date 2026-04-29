import pytest
import torch
import torch.nn.functional as F

from quack.cross_entropy import cross_entropy_fwd, cross_entropy

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "N",
    [192, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 128256, 131072],
)
@pytest.mark.parametrize("M", [1, 77, 289])
@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_weight(M, N, input_dtype, use_compile):
    """Test weighted cross entropy forward and backward against PyTorch reference."""
    major, _ = torch.cuda.get_device_capability()
    if major == 12 and input_dtype == torch.float32 and N > 131072:
        pytest.skip("SM12x: fp32 exceeds 99 KB SMEM")
    device = "cuda"
    atol, rtol = 5e-5, 1e-5
    torch.random.manual_seed(0)

    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    weight = torch.rand(N, device=device, dtype=torch.float32) + 0.1  # avoid zero weights

    x_ref = x.detach().clone().float().requires_grad_()
    target_ref = target.detach().clone()

    function = torch.compile(cross_entropy, fullgraph=True) if use_compile else cross_entropy
    loss = function(x, target, weight=weight, reduction="none")
    loss_ref = F.cross_entropy(x_ref, target_ref, weight=weight, reduction="none")

    assert loss.shape == (M,)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, loss_ref, atol=atol, rtol=rtol)
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()

    # Test backward
    dloss = torch.randn_like(loss)
    torch.cuda.synchronize()
    (dx,) = torch.autograd.grad(loss, x, grad_outputs=dloss)
    (dx_ref,) = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss)
    assert dx.shape == x.shape
    torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("N", [192, 1024, 32768])
@pytest.mark.parametrize("M", [1, 77, 289])
@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_weight_with_ignore_index(M, N, input_dtype, use_compile):
    """Test weighted cross entropy with ignore_index."""
    device = "cuda"
    atol, rtol = 5e-5, 1e-5
    torch.random.manual_seed(0)

    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    weight = torch.rand(N, device=device, dtype=torch.float32) + 0.1
    ignore_index = N - 1
    ignore_mask = torch.rand(M, device=device) < 0.3
    target[ignore_mask] = ignore_index

    x_ref = x.detach().clone().float().requires_grad_()
    target_ref = target.detach().clone()

    function = torch.compile(cross_entropy, fullgraph=True) if use_compile else cross_entropy
    loss = function(x, target, weight=weight, reduction="none", ignore_index=ignore_index)
    loss_ref = F.cross_entropy(
        x_ref, target_ref, weight=weight, reduction="none", ignore_index=ignore_index
    )

    assert (loss[ignore_mask] == 0).all(), "Loss should be 0 for ignored indices"
    if (~ignore_mask).any():
        torch.testing.assert_close(loss[~ignore_mask], loss_ref[~ignore_mask], atol=atol, rtol=rtol)

    # Test backward
    dloss = torch.randn_like(loss)
    torch.cuda.synchronize()
    (dx,) = torch.autograd.grad(loss, x, grad_outputs=dloss)
    (dx_ref,) = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss)
    torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("N", [1024, 32768, 128256])
@pytest.mark.parametrize("M", [77, 289])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_weight_reduction(M, N, input_dtype, reduction, use_compile):
    """Test weighted cross entropy with different reduction modes."""
    major, _ = torch.cuda.get_device_capability()
    if major == 12 and input_dtype == torch.float32 and N > 131072:
        pytest.skip("SM12x: fp32 exceeds 99 KB SMEM")
    device = "cuda"
    atol, rtol = 1e-4, 1e-4
    torch.random.manual_seed(0)

    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    weight = torch.rand(N, device=device, dtype=torch.float32) + 0.1

    x_ref = x.detach().clone().float().requires_grad_()
    target_ref = target.detach().clone()

    function = torch.compile(cross_entropy, fullgraph=True) if use_compile else cross_entropy
    loss = function(x, target, weight=weight, reduction=reduction)
    loss_ref = F.cross_entropy(x_ref, target_ref, weight=weight, reduction=reduction)

    torch.testing.assert_close(loss, loss_ref, atol=atol, rtol=rtol)

    # Test backward for scalar losses
    if reduction in ("mean", "sum"):
        loss.backward()
        loss_ref.backward()
        torch.testing.assert_close(x.grad, x_ref.grad.to(input_dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("N", [1024, 32768])
@pytest.mark.parametrize("M", [77, 289])
def test_cross_entropy_weight_uniform(M, N, input_dtype):
    """Uniform weights should produce the same result as no weight."""
    device = "cuda"
    atol, rtol = 1e-5, 1e-5
    torch.random.manual_seed(0)

    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    weight_uniform = torch.ones(N, device=device, dtype=torch.float32)

    x_no_weight = x.detach().clone().requires_grad_()

    loss_weighted = cross_entropy(x, target, weight=weight_uniform, reduction="none")
    loss_unweighted = cross_entropy(x_no_weight, target, reduction="none")

    torch.testing.assert_close(loss_weighted, loss_unweighted, atol=atol, rtol=rtol)

    dloss = torch.randn_like(loss_weighted)
    (dx_weighted,) = torch.autograd.grad(loss_weighted, x, grad_outputs=dloss)
    (dx_unweighted,) = torch.autograd.grad(loss_unweighted, x_no_weight, grad_outputs=dloss)
    torch.testing.assert_close(dx_weighted, dx_unweighted, atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("N", [192, 1024, 32768, 128256])
@pytest.mark.parametrize("M", [1, 77, 289])
@pytest.mark.parametrize("inplace_backward", [False, True])
def test_cross_entropy_fwd_with_grad_weight(M, N, input_dtype, inplace_backward):
    """Test fused forward+gradient path with weights."""
    major, _ = torch.cuda.get_device_capability()
    if major == 12 and input_dtype == torch.float32 and N > 131072:
        pytest.skip("SM12x: fp32 exceeds 99 KB SMEM")
    device = "cuda"
    atol, rtol = 1e-4, 1e-4
    torch.random.manual_seed(0)

    x = 0.1 * torch.randn(M, N, device=device, dtype=input_dtype)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    weight = torch.rand(N, device=device, dtype=torch.float32) + 0.1

    if inplace_backward:
        x_copy = x.detach().clone()
        loss, dx = cross_entropy_fwd(
            x_copy,
            target,
            weight=weight,
            return_dx=True,
            inplace_backward=True,
        )
        assert dx is x_copy
    else:
        loss, dx = cross_entropy_fwd(
            x,
            target,
            weight=weight,
            return_dx=True,
            inplace_backward=False,
        )
        assert dx is not x

    # Reference: the fused fwd computes dx = w_target * (softmax(x) - one_hot(target))
    x_ref = x.detach().clone().float().requires_grad_()
    loss_ref = F.cross_entropy(x_ref, target, weight=weight, reduction="none")
    dloss_ones = torch.ones_like(loss_ref)
    (dx_ref,) = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss_ones)

    torch.testing.assert_close(loss, loss_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)
