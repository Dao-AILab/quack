import pytest
import torch


_SF_VEC = 32


def _skip_if_not_sm100():
    if torch.cuda.get_device_properties(0).major < 10:
        pytest.skip("SM100+ required")


def _dequant(q: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    return q.float() * sc.float().repeat_interleave(_SF_VEC, dim=-1)


def _gated_ref(A_q, A_sc, W_q, W_sc, activation):
    from quack.gemm_interface import gated_to_pytorch_fn_map

    a_dq = _dequant(A_q, A_sc)  # (..., m, k)
    b_dq = _dequant(W_q, W_sc)  # (..., n_full, k)
    full = a_dq @ b_dq.transpose(-1, -2)  # (..., m, n_full)
    gate, up = full[..., ::2], full[..., 1::2]
    return gated_to_pytorch_fn_map[activation](gate, up)


@pytest.mark.parametrize("activation", ["swiglu", "reglu", "geglu"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "m,n_out,k",
    [
        (256, 64, 256),  # n_full=128, single N-tile
        (256, 128, 256),  # n_full=256
        (512, 256, 512),  # n_full=512, larger K
    ],
)
def test_mxfp8_gemm_gated(m, n_out, k, batched, activation):
    _skip_if_not_sm100()
    from quack.gemm_blockscaled_interface import mxfp8_gemm_gated, mxfp8_quantize

    n_full = 2 * n_out
    L = 2 if batched else 1
    torch.manual_seed(0)
    shape_A = (L, m, k) if batched else (m, k)
    # Fused gate/up weight stored nn.Linear-style (n_full, k) row-major.
    shape_W = (L, n_full, k) if batched else (n_full, k)
    A_hp = torch.randn(*shape_A, device="cuda", dtype=torch.bfloat16) * k**-0.5
    W_hp = torch.randn(*shape_W, device="cuda", dtype=torch.bfloat16) * k**-0.5

    A_q, A_sc = mxfp8_quantize(A_hp)  # (..., m, k), (..., m, k/32)
    W_q, W_sc = mxfp8_quantize(W_hp)  # (..., n_full, k), (..., n_full, k/32)

    # Pass .mT to reach the (K, N) K-contig layout the interface expects.
    out = mxfp8_gemm_gated(A_q, W_q.mT, A_sc, W_sc.mT, activation)
    assert out.shape == ((L, m, n_out) if batched else (m, n_out))
    assert out.dtype == torch.bfloat16

    ref = _gated_ref(A_q, A_sc, W_q, W_sc, activation)
    err = (out.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item() + 1e-6
    # fp8 inputs + gate activation + bf16 output rounding.
    assert err < 2e-2 or err / scale < 5e-2, f"max_err={err}, rel={err / scale}"


def test_mxfp8_gemm_gated_preallocated_out():
    _skip_if_not_sm100()
    from quack.gemm_blockscaled_interface import mxfp8_gemm_gated, mxfp8_quantize

    m, n_out, k = 256, 128, 256
    torch.manual_seed(0)
    A_hp = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    W_hp = torch.randn(2 * n_out, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    A_q, A_sc = mxfp8_quantize(A_hp)
    W_q, W_sc = mxfp8_quantize(W_hp)

    out_alloc = mxfp8_gemm_gated(A_q, W_q.mT, A_sc, W_sc.mT, "swiglu")
    out_pre = torch.empty(m, n_out, device="cuda", dtype=torch.bfloat16)
    mxfp8_gemm_gated(A_q, W_q.mT, A_sc, W_sc.mT, "swiglu", out=out_pre)
    assert torch.equal(out_alloc, out_pre)


@pytest.mark.parametrize(
    "mma_tiler_mn,cluster_shape_mn,m,n_out,k",
    [
        ((128, 128), (1, 1), 256, 128, 256),
        ((256, 128), (2, 1), 512, 128, 256),  # 2-CTA path (M=256 tile)
    ],
)
def test_mxfp8_gemm_gated_tiler_override(mma_tiler_mn, cluster_shape_mn, m, n_out, k):
    _skip_if_not_sm100()
    from quack.gemm_blockscaled_interface import mxfp8_gemm_gated, mxfp8_quantize

    n_full = 2 * n_out
    torch.manual_seed(0)
    A_hp = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    W_hp = torch.randn(n_full, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    A_q, A_sc = mxfp8_quantize(A_hp)
    W_q, W_sc = mxfp8_quantize(W_hp)

    out = mxfp8_gemm_gated(
        A_q,
        W_q.mT,
        A_sc,
        W_sc.mT,
        "swiglu",
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )
    ref = _gated_ref(A_q, A_sc, W_q, W_sc, "swiglu")
    err = (out.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item() + 1e-6
    assert err < 2e-2 or err / scale < 5e-2, f"max_err={err}, rel={err / scale}"


# ---------------------------------------------------------------------------
# Backward gated (dSwiGLU / dGeGLU / dReGLU)
# ---------------------------------------------------------------------------
import torch.nn.functional as F  # noqa: E402

_FWD_ACT = {
    "swiglu": lambda g: F.silu(g),
    "reglu": lambda g: F.relu(g),
    "geglu": lambda g: F.gelu(g, approximate="tanh"),
}


def _dgated_ref(A_q, A_sc, W_q, W_sc, PreAct, activation):
    dy = (_dequant(A_q, A_sc) @ _dequant(W_q, W_sc).transpose(-1, -2)).float()
    gate = PreAct[..., ::2].float().detach().clone().requires_grad_()
    up = PreAct[..., 1::2].float().detach().clone().requires_grad_()
    y = _FWD_ACT[activation](gate) * up
    y.backward(dy)
    return gate.grad, up.grad, y.detach()


@pytest.mark.parametrize("activation", ["swiglu", "reglu", "geglu"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("m,n,k", [(256, 128, 256), (512, 256, 512)])
def test_mxfp8_gemm_dgated(m, n, k, batched, activation):
    _skip_if_not_sm100()
    from quack.gemm_blockscaled_interface import mxfp8_gemm_dgated, mxfp8_quantize

    L = 2 if batched else 1
    torch.manual_seed(0)
    # A @ W.T = dy  (m, n).  A=(m,k) is e.g. quantized dout; W=(n,k) is e.g. W2.
    shape_A = (L, m, k) if batched else (m, k)
    shape_W = (L, n, k) if batched else (n, k)
    A_hp = torch.randn(*shape_A, device="cuda", dtype=torch.bfloat16) * k**-0.5
    W_hp = torch.randn(*shape_W, device="cuda", dtype=torch.bfloat16) * k**-0.5
    A_q, A_sc = mxfp8_quantize(A_hp)
    W_q, W_sc = mxfp8_quantize(W_hp)

    # Forward pre-activation [gate; up] interleaved, (..., m, 2n).
    pre_shape = (L, m, 2 * n) if batched else (m, 2 * n)
    PreAct = torch.randn(*pre_shape, device="cuda", dtype=torch.bfloat16)

    dPreAct, PostAct = mxfp8_gemm_dgated(A_q, W_q.mT, A_sc, W_sc.mT, PreAct, activation)
    assert dPreAct.shape == PreAct.shape
    assert PostAct.shape == ((L, m, n) if batched else (m, n))
    assert dPreAct.dtype == torch.bfloat16 and PostAct.dtype == torch.bfloat16

    dgate_ref, dup_ref, y_ref = _dgated_ref(A_q, A_sc, W_q, W_sc, PreAct, activation)

    def _close(got, ref):
        err = (got.float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item() + 1e-6
        return err < 2e-2 or err / scale < 5e-2, err, err / scale

    ok_y, e_y, r_y = _close(PostAct, y_ref)
    ok_g, e_g, r_g = _close(dPreAct[..., ::2], dgate_ref)
    ok_u, e_u, r_u = _close(dPreAct[..., 1::2], dup_ref)
    assert ok_y, f"PostAct max_err={e_y}, rel={r_y}"
    assert ok_g, f"dgate max_err={e_g}, rel={r_g}"
    assert ok_u, f"dup max_err={e_u}, rel={r_u}"
