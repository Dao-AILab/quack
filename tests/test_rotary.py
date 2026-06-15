import math

import pytest
import torch

from quack.rotary import apply_rotary, apply_rotary_emb, apply_rotary_emb_kv_, apply_rotary_emb_qkv_
from quack.testing.fake_compat import assert_aliased

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

# Marker alias for tests whose entire point is a runtime invariant that
# FakeTensorMode cannot represent:
#   * data_ptr() aliasing (FakeTensor has no real storage; .data_ptr() emits a
#     DeprecationWarning and will raise in PyTorch 2.5).
#   * torch.profiler kernel counts (no real kernel launches under --compile-only).
#   * torch.compile / Dynamo dispatch traces (Dynamo skips frames when an outer
#     FakeTensorMode dispatch is active, producing "no compiled frames" warnings).
# Skipping in phase 1 (--compile-only / cache warming) is safe: the underlying
# rotary kernel signatures are still warmed by other parametrized tests in this
# file. Phase 2 (real GPU) runs them normally.
#
# Backed by the ``compile_only_skip`` marker registered in
# ``quack.testing.pytest_plugin``; evaluated at test-setup time so it is
# robust to xdist worksteal item-fetch ordering.
_skip_under_compile_only = pytest.mark.compile_only_skip(
    "runtime invariant (data_ptr aliasing / profiler kernel count / "
    "torch.compile dispatch) cannot be validated under FakeTensorMode"
)


def _rotary_grid_dim_pairs():
    return [
        (headdim, rotary_dim)
        for headdim in [32, 64, 96, 128]
        for rotary_dim in range(8, headdim + 1, 8)
    ] + [(256, 256), (512, 512)]


def generate_cos_sin(seqlen, rotary_dim, device, dtype):
    assert rotary_dim % 2 == 0
    angle = torch.rand(seqlen * 2, rotary_dim // 2, device=device) * 2 * math.pi
    cos = torch.cos(angle).to(dtype=dtype)
    sin = torch.sin(angle).to(dtype=dtype)
    return cos, sin


def generate_seqlen_offsets(seqlen_offsets_type, batch_size, seqlen, device):
    if seqlen_offsets_type is None:
        return None
    if seqlen_offsets_type is torch.Tensor:
        return torch.randint(0, seqlen + 1, (batch_size,), dtype=torch.int32, device=device)
    raise ValueError(f"Unsupported seqlen_offsets_type: {seqlen_offsets_type}")


def index_cos_sin(cos, sin, seqlen_offsets, seqlen):
    if seqlen_offsets is None:
        return cos[:seqlen], sin[:seqlen]
    if isinstance(seqlen_offsets, torch.Tensor):
        arange = torch.arange(seqlen, device=cos.device).view(1, seqlen)
        idx = seqlen_offsets.view(-1, 1) + arange
        return cos[idx], sin[idx]
    raise TypeError("seqlen_offsets must be None or a tensor")


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    if not interleaved:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    else:
        cos = cos.unsqueeze(-1).expand(*cos.shape, 2).reshape(*cos.shape[:-1], ro_dim)
        sin = sin.unsqueeze(-1).expand(*sin.shape, 2).reshape(*sin.shape[:-1], ro_dim)
    if cos.dim() == 3:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    x_ro = x[..., :ro_dim]
    out_ro = x_ro * cos + rotate_half(x_ro, interleaved) * sin
    return torch.cat([out_ro, x[..., ro_dim:]], dim=-1)


def unpad_input(x, padding_mask):
    from quack.testing.fake_compat import fake_safe_nonzero

    batch, seqlen = padding_mask.shape
    indices = fake_safe_nonzero(padding_mask)
    x_unpad = x.reshape(batch * seqlen, *x.shape[2:])[indices]
    lengths = padding_mask.sum(dim=1, dtype=torch.int32)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=x.device),
            torch.cumsum(lengths, dim=0, dtype=torch.int32),
        ]
    )
    return x_unpad, indices, cu_seqlens, int(lengths.max().item())


def pad_input(x_unpad, indices, batch, seqlen):
    out = torch.zeros(
        batch * seqlen, *x_unpad.shape[1:], device=x_unpad.device, dtype=x_unpad.dtype
    )
    out[indices] = x_unpad
    return out.reshape(batch, seqlen, *x_unpad.shape[1:])


def cuda_event_names(prof):
    return [
        event.name
        for event in prof.events()
        if str(getattr(event, "device_type", "")).endswith("CUDA")
    ]


_profiler_cuda_kernels_visible = None


def _profiler_can_see_cuda_kernels():
    # Some CI environments (e.g. CUPTI mismatched with the device, or no
    # CAP_SYS_ADMIN) load kineto fine but never report CUDA activity events.
    # Probe once and cache so we can skip kernel-count tests there.
    global _profiler_cuda_kernels_visible
    if _profiler_cuda_kernels_visible is None:
        try:
            a = torch.randn(8, device="cuda")
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                (a + 1).sum()
                torch.cuda.synchronize()
            _profiler_cuda_kernels_visible = bool(cuda_event_names(prof))
        except Exception:
            _profiler_cuda_kernels_visible = False
    return _profiler_cuda_kernels_visible


def assert_one_cuda_kernel_no_memcpy(prof):
    if not _profiler_can_see_cuda_kernels():
        pytest.skip(
            "torch.profiler reports no CUDA kernels in this environment "
            "(CUPTI unavailable); cannot verify kernel-count / no-memcpy."
        )
    names = cuda_event_names(prof)
    if not names:
        # Probe confirmed CUPTI works in this process, but kineto intermittently
        # delivers zero CUDA events for an individual profile block under load
        # (observed in containerized CI). The "kernel removed entirely" regression
        # this would mask is still caught by the numerical asserts in these tests.
        pytest.skip("torch.profiler captured no CUDA events for this run (kineto flake)")
    kernels = [name for name in names if not name.startswith("Memcpy")]
    memcpys = [name for name in names if name.startswith("Memcpy")]
    assert len(kernels) == 1, names
    assert memcpys == [], names


def assert_packed_qkv_reshape_is_view(qkv):
    batch_size, seqlen, three, nheads, headdim = qkv.shape
    assert three == 3
    qk = qkv[:, :, :2].reshape(batch_size, seqlen, 2 * nheads, headdim)
    assert qk.untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr()
    assert qk.storage_offset() == qkv.storage_offset()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seqlen_offsets_type", [None, torch.Tensor])
# @pytest.mark.parametrize("seqlen_offsets_type", [None])
@pytest.mark.parametrize("rotary_fraction", [1.0, 0.5])
# @pytest.mark.parametrize("rotary_fraction", [1.0])
@pytest.mark.parametrize("interleaved", [False, True])
# @pytest.mark.parametrize("interleaved", [False])
@pytest.mark.parametrize("inplace", [False, True])
def test_rotary_emb_func(inplace, interleaved, rotary_fraction, seqlen_offsets_type, dtype):
    torch.manual_seed(42)
    device = "cuda"
    batch_size, seqlen, nheads, headdim = 8, 67, 4, 128
    rotary_dim = int(rotary_fraction * headdim)
    x = torch.randn(
        batch_size, seqlen, nheads, headdim, dtype=dtype, device=device, requires_grad=True
    )
    x_pt = x.detach().clone().requires_grad_()
    cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)
    seqlen_offsets = generate_seqlen_offsets(seqlen_offsets_type, batch_size, seqlen, device)

    out = apply_rotary_emb(
        x, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=inplace
    )
    cos_pt, sin_pt = index_cos_sin(cos, sin, seqlen_offsets, seqlen)
    out_pt = apply_rotary_emb_torch(
        x_pt.float(), cos_pt.float(), sin_pt.float(), interleaved=interleaved
    ).to(dtype=dtype)

    grad = torch.randn_like(out)
    grad_pt = grad.clone()
    out.backward(grad)
    out_pt.backward(grad_pt)

    if not inplace:
        assert torch.equal(x, x_pt)
    torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(x.grad, x_pt.grad, atol=1e-2, rtol=1e-3)


@_skip_under_compile_only
def test_rotary_emb_compile():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    batch_size, seqlen, nheads, headdim, rotary_dim = 4, 61, 4, 128, 64
    x = torch.randn(
        batch_size, seqlen, nheads, headdim, dtype=dtype, device=device, requires_grad=True
    )
    x_pt = x.detach().clone().requires_grad_()
    cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)
    seqlen_offsets_storage = torch.randint(
        0, seqlen + 1, (batch_size * 2,), dtype=torch.int32, device=device
    )
    seqlen_offsets = seqlen_offsets_storage[::2]
    assert seqlen_offsets.stride(0) != 1

    fn = torch.compile(apply_rotary_emb, fullgraph=True)
    out = fn(x, cos, sin, interleaved=True, seqlen_offsets=seqlen_offsets)
    cos_pt, sin_pt = index_cos_sin(cos, sin, seqlen_offsets, seqlen)
    out_pt = apply_rotary_emb_torch(x_pt.float(), cos_pt.float(), sin_pt.float(), True).to(
        dtype=dtype
    )
    grad = torch.randn_like(out)
    grad_pt = grad.clone()
    out.backward(grad)
    out_pt.backward(grad_pt)

    torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(x.grad, x_pt.grad, atol=1e-2, rtol=1e-3)


@_skip_under_compile_only
@pytest.mark.parametrize("use_compile", [False, True])
def test_rotary_emb_inplace_backward_no_copy(use_compile):
    torch._dynamo.reset()
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim, rotary_dim = 4, 64, 4, 128, 64
    fn = torch.compile(apply_rotary_emb, fullgraph=True) if use_compile else apply_rotary_emb

    def make_case():
        x = torch.randn(
            batch_size, seqlen, nheads, headdim, dtype=dtype, device=device, requires_grad=True
        )
        x_pt = x.detach().clone().requires_grad_()
        cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)
        return x, x_pt, cos, sin

    def reference(x_pt, cos, sin):
        cos_pt, sin_pt = index_cos_sin(cos, sin, None, seqlen)
        return apply_rotary_emb_torch(x_pt.float(), cos_pt.float(), sin_pt.float()).to(dtype=dtype)

    # Warm both the CuTe JIT and the torch.compile forward/backward graphs.
    x, _, cos, sin = make_case()
    out = fn(x, cos, sin, inplace=True)
    torch.autograd.grad(out, x, torch.randn_like(out))
    torch.cuda.synchronize()

    x, x_pt, cos, sin = make_case()
    out = fn(x, cos, sin, inplace=True)
    out_pt = reference(x_pt, cos, sin)
    grad = torch.randn_like(out)
    grad_pt = grad.clone()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        (dx,) = torch.autograd.grad(out, x, grad)
        torch.cuda.synchronize()
    out_pt.backward(grad_pt)

    assert out.data_ptr() == x.data_ptr()
    torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)
    assert_one_cuda_kernel_no_memcpy(prof)
    assert dx.data_ptr() == grad.data_ptr()
    torch.testing.assert_close(dx, x_pt.grad, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("headdim", "rotary_dim", "x_offset"),
    [
        (104, 64, 8),
        (96, 72, 0),  # x/out use 128-bit copies, cos/sin use 64-bit copies
    ],
)
def test_rotary_emb_vector_width_selection(headdim, rotary_dim, x_offset, dtype):
    torch.manual_seed(42)
    device = "cuda"
    batch_size, seqlen, nheads = 3, 37, 2
    x_storage = torch.randn(
        batch_size, seqlen, nheads, headdim + x_offset, dtype=dtype, device=device
    )
    x = x_storage[..., x_offset:].detach().requires_grad_()
    x_pt = x.detach().clone().requires_grad_()
    cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)

    out = apply_rotary_emb(x, cos, sin)
    cos_pt, sin_pt = index_cos_sin(cos, sin, None, seqlen)
    out_pt = apply_rotary_emb_torch(x_pt.float(), cos_pt.float(), sin_pt.float()).to(dtype=dtype)

    grad = torch.randn_like(out)
    grad_pt = grad.clone()
    out.backward(grad)
    out_pt.backward(grad_pt)

    torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(x.grad, x_pt.grad, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
# @pytest.mark.parametrize("x_dtype", [torch.bfloat16])
@pytest.mark.parametrize("cossin_dtype", [torch.bfloat16, torch.float32])
# @pytest.mark.parametrize("cossin_dtype", [torch.bfloat16])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize(("headdim", "rotary_dim"), _rotary_grid_dim_pairs())
# @pytest.mark.parametrize(("headdim", "rotary_dim"), [(32, 16)])
def test_rotary_emb_dim_dtype_grid(
    headdim, rotary_dim, conjugate, interleaved, x_dtype, cossin_dtype
):
    torch.manual_seed(42)
    device = "cuda"
    # batch_size, seqlen, nheads = 2, 17, 3
    batch_size, seqlen, nheads = 1, 17, 1
    x = torch.randn(
        batch_size, seqlen, nheads, headdim, dtype=x_dtype, device=device, requires_grad=True
    )
    x_pt = x.detach().clone().requires_grad_()
    cos, sin = generate_cos_sin(seqlen, rotary_dim, device, cossin_dtype)

    out = apply_rotary(x, cos, sin, interleaved=interleaved, conjugate=conjugate)
    cos_pt, sin_pt = index_cos_sin(cos, sin, None, seqlen)
    if conjugate:
        sin_pt = -sin_pt
    out_pt = apply_rotary_emb_torch(
        x_pt.float(), cos_pt.float(), sin_pt.float(), interleaved=interleaved
    ).to(dtype=x_dtype)

    torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)


def test_rotary_emb_dim_multiple_of_8_validation():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    batch_size, seqlen, nheads = 2, 17, 2

    cos, sin = generate_cos_sin(seqlen, 64, device, dtype)
    x_bad_headdim = torch.randn(batch_size, seqlen, nheads, 100, dtype=dtype, device=device)
    with pytest.raises(AssertionError, match="headdim must be divisible by 8"):
        apply_rotary_emb(x_bad_headdim, cos, sin)

    x = torch.randn(batch_size, seqlen, nheads, 104, dtype=dtype, device=device)
    cos_bad_rotary, sin_bad_rotary = generate_cos_sin(seqlen, 68, device, dtype)
    with pytest.raises(AssertionError, match="rotary_dim must be divisible by 8"):
        apply_rotary_emb(x, cos_bad_rotary, sin_bad_rotary)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_emb_strided_x_qkv_view(interleaved, inplace, dtype):
    torch.manual_seed(42)
    device = "cuda"
    batch_size, seqlen, nheads, headdim, rotary_dim = 3, 47, 4, 128, 64
    qkv = torch.randn(
        batch_size,
        seqlen,
        3,
        nheads,
        headdim,
        dtype=dtype,
        device=device,
        requires_grad=not inplace,
    )
    qkv_pt = qkv.detach().clone().requires_grad_(not inplace)
    x = qkv[:, :, 0]
    x_pt = qkv_pt[:, :, 0]
    assert not x.is_contiguous()
    assert x.stride(-1) == 1
    cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)

    out = apply_rotary_emb(x, cos, sin, interleaved=interleaved, inplace=inplace)
    cos_pt, sin_pt = index_cos_sin(cos, sin, None, seqlen)
    out_pt = apply_rotary_emb_torch(
        x_pt.float(), cos_pt.float(), sin_pt.float(), interleaved=interleaved
    ).to(dtype=dtype)
    if inplace:
        qkv_expected = qkv_pt.clone()
        qkv_expected[:, :, 0] = out_pt
        torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)
        torch.testing.assert_close(qkv, qkv_expected, atol=1e-2, rtol=1e-3)
    else:
        grad = torch.randn_like(out)
        grad_pt = grad.clone()
        out.backward(grad)
        out_pt.backward(grad_pt)

        torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)
        assert torch.equal(qkv, qkv_pt)
        torch.testing.assert_close(qkv.grad, qkv_pt.grad, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_emb_strided_cos_sin(interleaved, dtype):
    torch.manual_seed(42)
    device = "cuda"
    batch_size, seqlen, nheads, headdim, rotary_dim = 2, 41, 3, 128, 64
    x = torch.randn(
        batch_size, seqlen, nheads, headdim, dtype=dtype, device=device, requires_grad=True
    )
    x_pt = x.detach().clone().requires_grad_()
    cos_storage, sin_storage = generate_cos_sin(seqlen, rotary_dim + 16, device, dtype)
    cos = cos_storage[:, 8 : 8 + rotary_dim // 2]
    sin = sin_storage[:, 8 : 8 + rotary_dim // 2]
    assert not cos.is_contiguous()
    assert not sin.is_contiguous()
    assert cos.stride(-1) == 1
    assert sin.stride(-1) == 1

    out = apply_rotary_emb(x, cos, sin, interleaved=interleaved)
    cos_pt, sin_pt = index_cos_sin(cos, sin, None, seqlen)
    out_pt = apply_rotary_emb_torch(
        x_pt.float(), cos_pt.float(), sin_pt.float(), interleaved=interleaved
    ).to(dtype=dtype)

    grad = torch.randn_like(out)
    grad_pt = grad.clone()
    out.backward(grad)
    out_pt.backward(grad_pt)

    torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(x.grad, x_pt.grad, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rotary_emb_odd_nheads_block_h(dtype):
    torch.manual_seed(42)
    device = "cuda"
    batch_size, seqlen, nheads, headdim, rotary_dim = 3, 39, 3, 128, 32
    x = torch.randn(
        batch_size, seqlen, nheads, headdim, dtype=dtype, device=device, requires_grad=True
    )
    x_pt = x.detach().clone().requires_grad_()
    cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)

    out = apply_rotary_emb(x, cos, sin)
    cos_pt, sin_pt = index_cos_sin(cos, sin, None, seqlen)
    out_pt = apply_rotary_emb_torch(x_pt.float(), cos_pt.float(), sin_pt.float()).to(dtype=dtype)

    grad = torch.randn_like(out)
    grad_pt = grad.clone()
    out.backward(grad)
    out_pt.backward(grad_pt)

    torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(x.grad, x_pt.grad, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seqlen_offsets_type", [None, torch.Tensor])
@pytest.mark.parametrize("rotary_fraction", [1.0, 0.5])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_rotary_emb_varlen_func(inplace, interleaved, rotary_fraction, seqlen_offsets_type, dtype):
    torch.manual_seed(42)
    device = "cuda"
    batch_size, seqlen, nheads, headdim = 8, 71, 4, 128
    rotary_dim = int(rotary_fraction * headdim)
    x = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    x_pt = x.detach().clone().requires_grad_()
    lengths = torch.randint(max(1, seqlen - 20), seqlen + 1, (batch_size, 1), device=device)
    padding_mask = torch.arange(seqlen, device=device).view(1, seqlen) < lengths
    x_unpad, indices, cu_seqlens, max_seqlen = unpad_input(x, padding_mask)
    x_unpad_clone = x_unpad.clone()
    x_unpad = x_unpad.requires_grad_()
    cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)
    seqlen_offsets = generate_seqlen_offsets(seqlen_offsets_type, batch_size, seqlen, device)

    out_unpad = apply_rotary_emb(
        x_unpad,
        cos,
        sin,
        seqlen_offsets=seqlen_offsets,
        interleaved=interleaved,
        inplace=inplace,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    out = pad_input(out_unpad, indices, batch_size, seqlen)
    cos_pt, sin_pt = index_cos_sin(cos, sin, seqlen_offsets, seqlen)
    out_pt = apply_rotary_emb_torch(
        x_pt.float(), cos_pt.float(), sin_pt.float(), interleaved=interleaved
    ).to(dtype=dtype)
    out_pt = out_pt.masked_fill(~padding_mask[:, :, None, None], 0.0)

    grad = torch.randn_like(out)
    grad_pt = grad.clone()
    out.backward(grad)
    out_pt.backward(grad_pt)
    x_grad = pad_input(x_unpad.grad, indices, batch_size, seqlen)

    if not inplace:
        assert torch.equal(x_unpad, x_unpad_clone)
    torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(x_grad, x_pt.grad, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("seqlen_offsets_type", [None, torch.Tensor])
@pytest.mark.parametrize("rotary_fraction", [1.0, 0.5])
@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_emb_qkv(interleaved, rotary_fraction, seqlen_offsets_type, gqa, use_compile, dtype):
    torch.manual_seed(42)
    device = "cuda"
    batch_size, nheads, seqlen, headdim = 4, 4, 64, 128
    rotary_dim = int(rotary_fraction * headdim)
    if not gqa:
        qkv = torch.randn(
            batch_size,
            seqlen,
            3,
            nheads,
            headdim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
    else:
        nheads_k = nheads // 2
        qkv = torch.randn(
            batch_size,
            seqlen,
            nheads + 2 * nheads_k,
            headdim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
    qkv_pt = qkv.detach().clone().requires_grad_()
    cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)
    seqlen_offsets = generate_seqlen_offsets(seqlen_offsets_type, batch_size, seqlen, device)
    fn = (
        torch.compile(apply_rotary_emb_qkv_, fullgraph=True)
        if use_compile
        else apply_rotary_emb_qkv_
    )

    out = fn(
        qkv,
        cos,
        sin,
        seqlen_offsets=seqlen_offsets,
        interleaved=interleaved,
        num_heads_q=None if not gqa else nheads,
    )
    cos_pt, sin_pt = index_cos_sin(cos, sin, seqlen_offsets, seqlen)
    if not gqa:
        q_pt, k_pt, v_pt = qkv_pt.unbind(2)
        q_pt = apply_rotary_emb_torch(
            q_pt.float(), cos_pt.float(), sin_pt.float(), interleaved=interleaved
        ).to(dtype=dtype)
        k_pt = apply_rotary_emb_torch(
            k_pt.float(), cos_pt.float(), sin_pt.float(), interleaved=interleaved
        ).to(dtype=dtype)
        out_pt = torch.stack([q_pt, k_pt, v_pt], dim=2)
    else:
        nheads_k = nheads // 2
        q_pt, k_pt, v_pt = qkv_pt.split([nheads, nheads_k, nheads_k], dim=2)
        q_pt = apply_rotary_emb_torch(
            q_pt.float(), cos_pt.float(), sin_pt.float(), interleaved=interleaved
        ).to(dtype=dtype)
        k_pt = apply_rotary_emb_torch(
            k_pt.float(), cos_pt.float(), sin_pt.float(), interleaved=interleaved
        ).to(dtype=dtype)
        out_pt = torch.cat([q_pt, k_pt, v_pt], dim=2)

    grad = torch.randn_like(out)
    grad_pt = grad.clone()
    out.backward(grad)
    out_pt.backward(grad_pt)

    # data_ptr() comparison is deprecated on FakeTensor; under --compile-only the
    # Aliasing assertion only meaningful with real storage; the helper
    # short-circuits under compile-only mode.
    assert_aliased(out, qkv)
    torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(qkv.grad, qkv_pt.grad, atol=1e-2, rtol=1e-3)


@_skip_under_compile_only
@pytest.mark.parametrize("use_compile", [False, True])
def test_rotary_emb_qkv_compile_packed_then_gqa(use_compile):
    """Regression test for Dynamo dispatch across packed 5D QKV then 4D GQA QKV."""
    torch._dynamo.reset()
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, nheads, seqlen, headdim, rotary_dim = 2, 4, 37, 128, 128
    fn = (
        torch.compile(apply_rotary_emb_qkv_, fullgraph=True)
        if use_compile
        else apply_rotary_emb_qkv_
    )

    def run_case(gqa):
        if not gqa:
            qkv = torch.randn(
                batch_size,
                seqlen,
                3,
                nheads,
                headdim,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        else:
            nheads_k = nheads // 2
            qkv = torch.randn(
                batch_size,
                seqlen,
                nheads + 2 * nheads_k,
                headdim,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        qkv_pt = qkv.detach().clone().requires_grad_()
        cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)

        out = fn(
            qkv,
            cos,
            sin,
            seqlen_offsets=None,
            interleaved=False,
            num_heads_q=None if not gqa else nheads,
        )
        cos_pt, sin_pt = index_cos_sin(cos, sin, None, seqlen)
        if not gqa:
            q_pt, k_pt, v_pt = qkv_pt.unbind(2)
            q_pt = apply_rotary_emb_torch(q_pt.float(), cos_pt.float(), sin_pt.float()).to(
                dtype=dtype
            )
            k_pt = apply_rotary_emb_torch(k_pt.float(), cos_pt.float(), sin_pt.float()).to(
                dtype=dtype
            )
            out_pt = torch.stack([q_pt, k_pt, v_pt], dim=2)
        else:
            nheads_k = nheads // 2
            q_pt, k_pt, v_pt = qkv_pt.split([nheads, nheads_k, nheads_k], dim=2)
            q_pt = apply_rotary_emb_torch(q_pt.float(), cos_pt.float(), sin_pt.float()).to(
                dtype=dtype
            )
            k_pt = apply_rotary_emb_torch(k_pt.float(), cos_pt.float(), sin_pt.float()).to(
                dtype=dtype
            )
            out_pt = torch.cat([q_pt, k_pt, v_pt], dim=2)

        grad = torch.randn_like(out)
        grad_pt = grad.clone()
        out.backward(grad)
        out_pt.backward(grad_pt)

        assert out.data_ptr() == qkv.data_ptr()
        torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)
        torch.testing.assert_close(qkv.grad, qkv_pt.grad, atol=1e-2, rtol=1e-3)

    run_case(gqa=False)
    run_case(gqa=True)


@_skip_under_compile_only
@pytest.mark.parametrize("use_compile", [False, True])
def test_rotary_emb_qkv_inplace_kernel_count(use_compile):
    torch._dynamo.reset()
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, nheads, seqlen, headdim, rotary_dim = 4, 4, 64, 128, 128
    fn = (
        torch.compile(apply_rotary_emb_qkv_, fullgraph=True)
        if use_compile
        else apply_rotary_emb_qkv_
    )

    def make_case():
        qkv = torch.randn(
            batch_size,
            seqlen,
            3,
            nheads,
            headdim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        qkv_pt = qkv.detach().clone().requires_grad_()
        cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)
        return qkv, qkv_pt, cos, sin

    def reference(qkv_pt, cos, sin):
        q_pt, k_pt, v_pt = qkv_pt.unbind(2)
        cos_pt, sin_pt = index_cos_sin(cos, sin, None, seqlen)
        q_pt = apply_rotary_emb_torch(q_pt.float(), cos_pt.float(), sin_pt.float()).to(dtype=dtype)
        k_pt = apply_rotary_emb_torch(k_pt.float(), cos_pt.float(), sin_pt.float()).to(dtype=dtype)
        return torch.stack([q_pt, k_pt, v_pt], dim=2)

    # Warm both the CuTe JIT and the torch.compile forward/backward graphs.
    qkv, _, cos, sin = make_case()
    out = fn(qkv, cos, sin, seqlen_offsets=None, interleaved=False)
    torch.autograd.grad(out, qkv, torch.randn_like(out))
    torch.cuda.synchronize()

    qkv, qkv_pt, cos, sin = make_case()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        out = fn(qkv, cos, sin, seqlen_offsets=None, interleaved=False)
        torch.cuda.synchronize()
    assert_one_cuda_kernel_no_memcpy(prof)
    assert out.data_ptr() == qkv.data_ptr()
    torch.testing.assert_close(out, reference(qkv_pt, cos, sin), atol=1e-2, rtol=1e-3)

    qkv, qkv_pt, cos, sin = make_case()
    out = fn(qkv, cos, sin, seqlen_offsets=None, interleaved=False)
    out_pt = reference(qkv_pt, cos, sin)
    grad = torch.randn_like(out)
    grad_pt = grad.clone()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        (dqkv,) = torch.autograd.grad(out, qkv, grad)
        torch.cuda.synchronize()
    assert_one_cuda_kernel_no_memcpy(prof)
    assert dqkv.data_ptr() == grad.data_ptr()
    out_pt.backward(grad_pt)
    torch.testing.assert_close(dqkv, qkv_pt.grad, atol=1e-2, rtol=1e-3)


@_skip_under_compile_only
@pytest.mark.parametrize("use_compile", [False, True])
def test_rotary_emb_qkv_packed_reshape_backward_no_copy(use_compile):
    torch._dynamo.reset()
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, nheads, seqlen, headdim, rotary_dim = 4, 4, 64, 128, 128
    fn = (
        torch.compile(apply_rotary_emb_qkv_, fullgraph=True)
        if use_compile
        else apply_rotary_emb_qkv_
    )

    def make_case():
        qkv = torch.randn(
            batch_size,
            seqlen,
            3,
            nheads,
            headdim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        qkv_pt = qkv.detach().clone().requires_grad_()
        cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)
        return qkv, qkv_pt, cos, sin

    def reference(qkv_pt, cos, sin):
        q_pt, k_pt, v_pt = qkv_pt.unbind(2)
        cos_pt, sin_pt = index_cos_sin(cos, sin, None, seqlen)
        q_pt = apply_rotary_emb_torch(q_pt.float(), cos_pt.float(), sin_pt.float()).to(dtype=dtype)
        k_pt = apply_rotary_emb_torch(k_pt.float(), cos_pt.float(), sin_pt.float()).to(dtype=dtype)
        return torch.stack([q_pt, k_pt, v_pt], dim=2)

    # This is the exact packed-QKV reshape used by rotary.py.
    # It must stay a view; otherwise backward would need an extra copy.
    qkv, _, cos, sin = make_case()
    assert_packed_qkv_reshape_is_view(qkv)

    # Warm both the CuTe JIT and the torch.compile forward/backward graphs.
    out = fn(qkv, cos, sin, seqlen_offsets=None, interleaved=False)
    torch.autograd.grad(out, qkv, torch.randn_like(out))
    torch.cuda.synchronize()

    qkv, qkv_pt, cos, sin = make_case()
    out = fn(qkv, cos, sin, seqlen_offsets=None, interleaved=False)
    out_pt = reference(qkv_pt, cos, sin)
    grad = torch.randn_like(out)
    grad_pt = grad.clone()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        (dqkv,) = torch.autograd.grad(out, qkv, grad)
        torch.cuda.synchronize()
    assert_one_cuda_kernel_no_memcpy(prof)
    assert dqkv.data_ptr() == grad.data_ptr()
    out_pt.backward(grad_pt)
    torch.testing.assert_close(dqkv, qkv_pt.grad, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seqlen_offsets_type", [None, torch.Tensor])
@pytest.mark.parametrize("rotary_fraction", [1.0, 0.5])
@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_emb_kv(interleaved, rotary_fraction, seqlen_offsets_type, dtype):
    torch.manual_seed(42)
    device = "cuda"
    batch_size, nheads, seqlen, headdim = 4, 4, 73, 64
    rotary_dim = int(rotary_fraction * headdim)
    kv = torch.randn(
        batch_size, seqlen, 2, nheads, headdim, dtype=dtype, device=device, requires_grad=True
    )
    kv_pt = kv.detach().clone().requires_grad_()
    cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)
    seqlen_offsets = generate_seqlen_offsets(seqlen_offsets_type, batch_size, seqlen, device)

    out = apply_rotary_emb_kv_(kv, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved)
    cos_pt, sin_pt = index_cos_sin(cos, sin, seqlen_offsets, seqlen)
    k_pt = apply_rotary_emb_torch(
        kv_pt[:, :, 0].float(), cos_pt.float(), sin_pt.float(), interleaved=interleaved
    ).to(dtype=dtype)
    out_pt = torch.stack([k_pt, kv_pt[:, :, 1]], dim=2)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_pt.backward(grad.clone())

    torch.testing.assert_close(out, out_pt, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(kv.grad, kv_pt.grad, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("inplace", [False, True])
def test_apply_rotary_empty(inplace):
    """apply_rotary must handle zero-batch inputs without launching a kernel."""
    dtype = torch.bfloat16
    seqlen, nheads, headdim = 64, 4, 64
    rotary_dim = 32
    x = torch.empty(0, seqlen, nheads, headdim, device="cuda", dtype=dtype)
    cos = torch.randn(seqlen, rotary_dim // 2, device="cuda", dtype=dtype)
    sin = torch.randn(seqlen, rotary_dim // 2, device="cuda", dtype=dtype)
    out = apply_rotary(x, cos, sin, inplace=inplace)
    assert out.shape == x.shape and out.numel() == 0


def test_apply_rotary_bwd_empty():
    """Backward path of apply_rotary_emb (autograd) on zero-batch inputs."""
    dtype = torch.bfloat16
    seqlen, nheads, headdim = 64, 4, 64
    rotary_dim = 32
    x = torch.empty(0, seqlen, nheads, headdim, device="cuda", dtype=dtype, requires_grad=True)
    cos = torch.randn(seqlen, rotary_dim // 2, device="cuda", dtype=dtype)
    sin = torch.randn(seqlen, rotary_dim // 2, device="cuda", dtype=dtype)
    out = apply_rotary_emb(x, cos, sin)
    grad = torch.empty_like(out)
    out.backward(grad)
    assert x.grad.shape == x.shape and x.grad.numel() == 0
