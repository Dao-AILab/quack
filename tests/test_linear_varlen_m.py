# Copyright (C) 2025, Tri Dao.
import math
import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm as quack_gemm
from quack.gemm_interface import (
    gemm,
    gemm_ref,
    gemm_add,
    gemm_add_ref,
    gemm_add_inplace,
    gemm_act,
    gemm_dact,
    gemm_act_ref,
    gemm_dact_ref,
    gemm_gated,
    gemm_dgated,
    gemm_gated_ref,
    gemm_dgated_ref,
)

sm100_tma_gather_only = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (10, 11),
    reason="TMA gather tests require SM100/SM110",
)


def generate_A_with_gather(total_m, k, device, dtype, gather_A=False):
    """Generate A matrix and optionally A_idx for gather_A case.

    Args:
        total_m: Number of rows needed
        k: Number of columns
        device: Device to create tensors on
        dtype: Data type of tensors
        gather_A: Whether to create gather indices

    Returns:
        A: Matrix of shape (larger_m, k) if gather_A else (total_m, k)
        A_idx: Index tensor of shape (total_m,) if gather_A else None
    """
    if gather_A:
        # Create random indices for gathering from a larger A matrix
        larger_m = total_m * 2  # Make A larger than needed
        A = torch.randn((larger_m, k), device=device, dtype=dtype)
        # Create random indices to gather from A
        A_idx = torch.randperm(larger_m, device=device, dtype=torch.int32)[:total_m]
    else:
        A = torch.randn((total_m, k), device=device, dtype=dtype)
        A_idx = None
    return A, A_idx


def run_lowlevel_varlen_m_gemm(
    A,
    B,
    out,
    cu_seqlens_m,
    A_idx,
    *,
    dynamic_persistent=False,
    use_tma_gather=False,
):
    device_capacity = get_device_capacity(A.device)[0]
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_persistent and device_capacity == 9
        else None
    )
    quack_gemm(
        A,
        B,
        out,
        C=None,
        tile_count_semaphore=tile_count_semaphore,
        tile_M=256,
        tile_N=256,
        cluster_M=2,
        cluster_N=1,
        persistent=True,
        is_dynamic_persistent=dynamic_persistent,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        use_tma_gather=use_tma_gather,
    )


@sm100_tma_gather_only
@pytest.mark.parametrize("dynamic_persistent", [False, True])
@pytest.mark.parametrize("B_major", ["k", "n"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [512])
def test_gemm_varlen_m_tma_gather_matches_cpasync(
    k,
    n,
    input_dtype,
    B_major,
    dynamic_persistent,
):
    device = "cuda"
    torch.random.manual_seed(42)
    num_groups = 4
    seq_lens = torch.randint(96, 320, (num_groups,), device=device)
    total_m = seq_lens.sum().item()
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A, A_idx = generate_A_with_gather(total_m, k, device, input_dtype, gather_A=True)
    if B_major == "k":
        B = torch.randn((num_groups, n, k), device=device, dtype=input_dtype) / math.sqrt(k)
    else:
        B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype).permute(0, 2, 1)
        B /= math.sqrt(k)

    out_cpasync = torch.empty((total_m, n), device=device, dtype=input_dtype)
    out_tma = torch.empty_like(out_cpasync)

    run_lowlevel_varlen_m_gemm(
        A,
        B,
        out_cpasync,
        cu_seqlens_m,
        A_idx,
        dynamic_persistent=dynamic_persistent,
        use_tma_gather=False,
    )
    run_lowlevel_varlen_m_gemm(
        A,
        B,
        out_tma,
        cu_seqlens_m,
        A_idx,
        dynamic_persistent=dynamic_persistent,
        use_tma_gather=True,
    )

    out_ref = gemm_ref(
        A.float(),
        B.mT.float(),
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
    out_pt = gemm_ref(A, B.mT, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx)

    assert out_tma.shape == (total_m, n)
    assert (out_tma - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-5
    assert (out_cpasync - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-5
    torch.testing.assert_close(out_tma, out_cpasync, atol=3e-2, rtol=1e-3)


@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("alpha_is_tensor", [False, True])
@pytest.mark.parametrize("alpha", [1.0, 1.7])
# @pytest.mark.parametrize("alpha_is_tensor", [False])
# @pytest.mark.parametrize("alpha", [1.0])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [True])
@pytest.mark.parametrize("B_major", ["k", "n"])
# @pytest.mark.parametrize("B_major", ["k"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1504, 2048, 4096])
@pytest.mark.parametrize("k", [736, 1024, 8192])
# @pytest.mark.parametrize("n", [2048])
# @pytest.mark.parametrize("k", [736])
@pytest.mark.parametrize("num_groups", [3, 5])
# @pytest.mark.parametrize("num_groups", [4])
def test_gemm_varlen_m(
    num_groups,
    k,
    n,
    input_dtype,
    B_major,
    dynamic_scheduler,
    alpha,
    alpha_is_tensor,
    has_bias,
    gather_A,
):
    """Test GEMM with variable length M dimension using cu_seqlens_m."""
    device = "cuda"
    torch.random.manual_seed(0)
    seq_lens = torch.randint(8192 - 1024, 8192 + 1024, (num_groups,), device=device)
    total_m = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A, A_idx = generate_A_with_gather(total_m, k, device, input_dtype, gather_A)
    B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    if B_major == "k":
        B = B.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    if alpha_is_tensor:
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    bias = torch.randn(num_groups, n, device=device) if has_bias else None
    out = gemm(
        A,
        B,
        bias=bias,
        alpha=alpha,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    A_f, B_f = A.float(), B.float()
    out_ref = gemm_ref(A_f, B_f, bias=bias, alpha=alpha, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx)
    del A_f, B_f
    out_pt = gemm_ref(A, B, bias=bias, alpha=alpha, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx)
    assert out.shape == (total_m, n), (
        f"Output shape mismatch: {out.shape} vs expected ({total_m}, {n})"
    )
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-5


@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("beta_is_tensor", [False, True])
@pytest.mark.parametrize("alpha_is_tensor", [False, True])
@pytest.mark.parametrize("beta", [0.5, 1.0])
@pytest.mark.parametrize("alpha", [1.0, 1.5])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [False])
@pytest.mark.parametrize("B_major", ["k", "n"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1504, 2048, 4096])
@pytest.mark.parametrize("k", [736, 1024, 8192])
@pytest.mark.parametrize("num_groups", [3, 5])
def test_gemm_add_varlen_m(
    num_groups,
    k,
    n,
    input_dtype,
    B_major,
    dynamic_scheduler,
    alpha,
    beta,
    alpha_is_tensor,
    beta_is_tensor,
    gather_A,
):
    """Test GEMM with addition and variable length M dimension using cu_seqlens_m."""
    device = "cuda"
    torch.random.manual_seed(0)
    seq_lens = torch.randint(8192 - 1024, 8192 + 1024, (num_groups,), device=device)
    total_m = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A, A_idx = generate_A_with_gather(total_m, k, device, input_dtype, gather_A)
    B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    C = torch.randn((total_m, n), device=device, dtype=input_dtype)
    if B_major == "k":
        B = B.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    if alpha_is_tensor:
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    if beta_is_tensor:
        beta = torch.tensor(beta, device=device, dtype=torch.float32)
    out = gemm_add(
        A,
        B,
        C,
        alpha=alpha,
        beta=beta,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
    beta_val = beta.item() if torch.is_tensor(beta) else beta
    A_f, B_f, C_f = A.float(), B.float(), C.float()
    out_ref = gemm_add_ref(
        A_f,
        B_f,
        C_f,
        alpha=alpha_val,
        beta=beta_val,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
    del A_f, B_f, C_f
    out_pt = gemm_add_ref(
        A, B, C, alpha=alpha_val, beta=beta_val, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
    )
    assert out.shape == (total_m, n), (
        f"Output shape mismatch: {out.shape} vs expected ({total_m}, {n})"
    )
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-5


@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("beta_is_tensor", [False, True])
@pytest.mark.parametrize("alpha_is_tensor", [False, True])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [False])
@pytest.mark.parametrize("B_major", ["k", "n"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024, 1504])
@pytest.mark.parametrize("k", [512, 768])
@pytest.mark.parametrize("num_groups", [2, 4])
def test_gemm_add_inplace_varlen_m(
    num_groups,
    k,
    n,
    input_dtype,
    B_major,
    dynamic_scheduler,
    alpha,
    beta,
    alpha_is_tensor,
    beta_is_tensor,
    gather_A,
):
    """Test in-place GEMM with addition and variable length M dimension: out = alpha * A @ B + beta * out."""
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.randint(100, 500, (num_groups,), device=device)
    total_m = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A, A_idx = generate_A_with_gather(total_m, k, device, input_dtype, gather_A)
    B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    out = torch.randn((total_m, n), device=device, dtype=input_dtype)
    if B_major == "k":
        B = B.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    if alpha_is_tensor:
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    if beta_is_tensor:
        beta = torch.tensor(beta, device=device, dtype=torch.float32)
    # Save original out for reference computation
    out_og = out.clone()
    gemm_add_inplace(
        A,
        B,
        out,
        alpha=alpha,
        beta=beta,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
    beta_val = beta.item() if torch.is_tensor(beta) else beta
    A_f, B_f, out_og_f = A.float(), B.float(), out_og.float()
    out_ref = gemm_add_ref(
        A_f,
        B_f,
        out_og_f,
        out=None,
        alpha=alpha_val,
        beta=beta_val,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
    del A_f, B_f, out_og_f
    out_pt = gemm_add_ref(
        A,
        B,
        out_og,
        out=None,
        alpha=alpha_val,
        beta=beta_val,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
    assert out.shape == (total_m, n), (
        f"Output shape mismatch: {out.shape} vs expected ({total_m}, {n})"
    )
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-4


@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("activation", [None, "relu", "gelu_tanh_approx"])
# @pytest.mark.parametrize("activation", [None])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [False])
@pytest.mark.parametrize("B_major", ["k", "n"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024, 1504])
@pytest.mark.parametrize("k", [512, 768])
@pytest.mark.parametrize("num_groups", [2, 4])
def test_gemm_act_varlen_m(
    num_groups,
    k,
    n,
    input_dtype,
    B_major,
    dynamic_scheduler,
    activation,
    gather_A,
):
    """Test GEMM with activation and variable length M dimension."""
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.randint(100, 500, (num_groups,), device=device)
    total_m = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A, A_idx = generate_A_with_gather(total_m, k, device, input_dtype, gather_A)
    B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    C = torch.randn((total_m, n), device=device, dtype=input_dtype) * 0.1
    if B_major == "k":
        B = B.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    # Test with kernel
    preact, postact = gemm_act(
        A,
        B,
        C,
        activation=activation,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    assert preact.shape == (total_m, n)
    assert postact.shape == (total_m, n)
    # Compare with reference
    A_f, B_f, C_f = A.float(), B.float(), C.float()
    preact_ref, postact_ref = gemm_act_ref(
        A_f,
        B_f,
        C_f,
        activation=activation,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
    del A_f, B_f, C_f
    preact_pt, postact_pt = gemm_act_ref(
        A, B, C, activation=activation, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
    )
    assert (preact - preact_ref).abs().max() < 2 * (preact_pt - preact_ref).abs().max() + 1e-5
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-5


@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("activation", [None, "relu", "gelu_tanh_approx"])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [False])
@pytest.mark.parametrize("B_major", ["k", "n"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024, 1504])
@pytest.mark.parametrize("k", [512, 768])
@pytest.mark.parametrize("num_groups", [2, 4])
def test_gemm_dact_varlen_m(
    num_groups,
    k,
    n,
    input_dtype,
    B_major,
    dynamic_scheduler,
    activation,
    gather_A,
):
    """Test GEMM with activation gradient and variable length M dimension."""
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.randint(100, 500, (num_groups,), device=device)
    total_m = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A, A_idx = generate_A_with_gather(total_m, k, device, input_dtype, gather_A)
    B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    PreAct = torch.randn((total_m, n), device=device, dtype=input_dtype) * 0.1
    if B_major == "k":
        B = B.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    # Test with kernel
    dx, postact = gemm_dact(
        A,
        B,
        PreAct,
        activation=activation,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    assert dx.shape == (total_m, n)
    assert postact.shape == (total_m, n)
    # Compare with reference
    A_f, B_f, P_f = A.float(), B.float(), PreAct.float()
    dx_ref, postact_ref = gemm_dact_ref(
        A_f,
        B_f,
        P_f,
        activation=activation,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
    del A_f, B_f, P_f
    dx_pt, postact_pt = gemm_dact_ref(
        A, B, PreAct, activation=activation, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
    )
    assert (dx - dx_ref).abs().max() < 2 * (dx_pt - dx_ref).abs().max() + 1e-5
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-5


@pytest.mark.parametrize("gather_A", [False, True])
@pytest.mark.parametrize("activation", ["swiglu"])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
@pytest.mark.parametrize("B_major", ["k", "n"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("n", [1024, 1504])
@pytest.mark.parametrize("k", [512, 768])
@pytest.mark.parametrize("num_groups", [2, 4])
def test_gemm_gated_varlen_m(
    num_groups,
    k,
    n,
    has_bias,
    input_dtype,
    B_major,
    dynamic_scheduler,
    activation,
    gather_A,
):
    """Test GEMM with gated activation and variable length M dimension."""
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.randint(50, 300, (num_groups,), device="cpu")
    total_m = seq_lens.sum().item()
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_m = cu_seqlens_m.to(device)
    A, A_idx = generate_A_with_gather(total_m, k, device, input_dtype, gather_A)
    B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    C = torch.randn((total_m, n), device=device, dtype=input_dtype) * 0.1
    bias = torch.randn(num_groups, n, device=device) if has_bias else None
    if B_major == "k":
        B = B.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    # Test with kernel
    preact, postact = gemm_gated(
        A,
        B,
        C,
        bias=bias,
        activation=activation,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    assert preact.shape == (total_m, n)
    assert postact.shape == (total_m, n // 2)
    # Compare with reference
    A_f, B_f, C_f = A.float(), B.float(), C.float()
    preact_ref, postact_ref = gemm_gated_ref(
        A_f,
        B_f,
        C_f,
        bias=bias,
        activation=activation,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
    del A_f, B_f, C_f
    preact_pt, postact_pt = gemm_gated_ref(
        A, B, C, bias=bias, activation=activation, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
    )
    assert (preact - preact_ref).abs().max() < 2 * (preact_pt - preact_ref).abs().max() + 1e-5
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-5


@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("activation", ["swiglu"])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [False])
@pytest.mark.parametrize("B_major", ["k", "n"])
# @pytest.mark.parametrize("B_major", ["k"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("colvec_reduce", [False, True])
@pytest.mark.parametrize("has_colvec_scale", [False, True])
# @pytest.mark.parametrize("has_colvec_scale", [True])
@pytest.mark.parametrize("n", [1024, 1504])
@pytest.mark.parametrize("k", [512, 768])
# @pytest.mark.parametrize("n", [1024])
# @pytest.mark.parametrize("k", [512])
@pytest.mark.parametrize("num_groups", [2, 4])
# @pytest.mark.parametrize("num_groups", [2])
def test_gemm_dgated_varlen_m(
    num_groups,
    k,
    n,
    has_colvec_scale,
    colvec_reduce,
    input_dtype,
    B_major,
    dynamic_scheduler,
    activation,
    gather_A,
):
    """Test GEMM with gated activation gradient and variable length M dimension."""
    device = "cuda"
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
        pytest.skip("SM120 gated dactivation GEMM epilogue is not yet supported")
    torch.random.manual_seed(42)
    seq_lens = torch.randint(50, 300, (num_groups,), device="cpu")
    total_m = seq_lens.sum().item()
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_m = cu_seqlens_m.to(device)
    A, A_idx = generate_A_with_gather(total_m, k, device, input_dtype, gather_A)
    B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    PreAct = torch.randn((total_m, 2 * n), device=device, dtype=input_dtype) * 0.1
    if B_major == "k":
        B = B.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    colvec_scale = torch.randn(total_m, device=device) if has_colvec_scale else None
    # Test with kernel
    dx, postact, *rest = gemm_dgated(
        A,
        B,
        PreAct,
        colvec_scale=colvec_scale,
        activation=activation,
        colvec_reduce=colvec_reduce,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    if colvec_reduce:
        colvec_reduce_out = rest[0]
    assert dx.shape == (total_m, 2 * n)
    assert postact.shape == (total_m, n)
    # Compare with reference
    A_f, B_f, P_f = A.float(), B.float(), PreAct.float()
    dx_ref, postact_ref = gemm_dgated_ref(
        A_f,
        B_f,
        P_f,
        activation=activation,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
    del P_f
    dx_pt, postact_pt = gemm_dgated_ref(
        A, B, PreAct, activation=activation, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
    )
    if colvec_reduce:
        colvec_reduce_ref = (
            postact_ref * gemm_ref(A_f, B_f, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx)
        ).sum(dim=-1)
        colvec_reduce_pt = (
            postact_pt * gemm_ref(A, B, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx)
        ).sum(dim=-1)
    del A_f, B_f
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


@pytest.mark.parametrize("gather_A", [False, True])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1504])
@pytest.mark.parametrize("k", [736])
@pytest.mark.parametrize("num_groups", [3])
def test_gemm_varlen_m_concat(num_groups, k, n, input_dtype, has_bias, dynamic_scheduler, gather_A):
    """Test GEMM varlen_m with concat_layout=("B",) for MoE forward/backward."""
    device = "cuda"
    torch.random.manual_seed(0)
    seq_lens = torch.randint(100, 200, (num_groups,), device=device)
    total_m = seq_lens.sum().item()
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A, A_idx = generate_A_with_gather(total_m, k, device, input_dtype, gather_A)
    B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    bias = torch.randn(num_groups, n, device=device) if has_bias else None
    concat = ("B",)
    out = gemm(
        A,
        B,
        bias=bias,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
        concat_layout=concat,
    )
    out_ref = gemm_ref(
        A.float(),
        B.float(),
        bias=bias,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        concat_layout=concat,
    )
    out_pt = gemm_ref(A, B, bias=bias, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx, concat_layout=concat)
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-5


@pytest.mark.parametrize("gather_A", [False, True])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1504])
@pytest.mark.parametrize("k", [736])
@pytest.mark.parametrize("num_groups", [3])
def test_gemm_gated_varlen_m_concat(
    num_groups, k, n, input_dtype, has_bias, dynamic_scheduler, gather_A
):
    """Test gated GEMM varlen_m with concat_layout=("B",) for MoE forward."""
    device = "cuda"
    torch.random.manual_seed(0)
    seq_lens = torch.randint(100, 200, (num_groups,), device=device)
    total_m = seq_lens.sum().item()
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A, A_idx = generate_A_with_gather(total_m, k, device, input_dtype, gather_A)
    B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    bias = torch.randn(num_groups, n, device=device) if has_bias else None
    concat = ("B",)
    preact, postact = gemm_gated(
        A,
        B,
        bias=bias,
        activation="swiglu",
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
        concat_layout=concat,
    )
    preact_ref, postact_ref = gemm_gated_ref(
        A.float(),
        B.float(),
        bias=bias,
        activation="swiglu",
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        concat_layout=concat,
    )
    preact_pt, postact_pt = gemm_gated_ref(
        A,
        B,
        bias=bias,
        activation="swiglu",
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        concat_layout=concat,
    )
    assert (preact - preact_ref).abs().max() < 2 * (preact_pt - preact_ref).abs().max() + 1e-5
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-5
