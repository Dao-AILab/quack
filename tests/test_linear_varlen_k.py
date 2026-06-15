# Copyright (C) 2025, Tri Dao.
import math
import pytest
import torch

from quack.testing.fake_compat import assert_aliased
from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm as quack_gemm
from quack.gemm_interface import (
    gemm,
    gemm_ref,
    gemm_add,
    gemm_add_ref,
    gemm_add_inplace,
)

sm100_tma_gather_only = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (10, 11),
    reason="TMA gather tests require SM100/SM110",
)


def generate_A_with_gather(m, total_k, device, dtype, gather_A=False):
    """Generate A matrix and optionally A_idx for gather_A case with varlen_k.

    Args:
        m: Number of rows
        total_k: Number of columns needed
        device: Device to create tensors on
        dtype: Data type of tensors
        gather_A: Whether to create gather indices

    Returns:
        A: Matrix of shape (m, larger_k) if gather_A else (m, total_k)
        A_idx: Index tensor of shape (total_k,) if gather_A else None
    """
    if gather_A:
        # Create random indices for gathering from a larger A matrix
        larger_k = total_k * 2  # Make A larger than needed
        A = torch.randn((m, larger_k), device=device, dtype=dtype)
        # Make A m-major
        A = A.T.contiguous().T
        # Create random indices to gather from A
        A_idx = torch.randperm(larger_k, device=device, dtype=torch.int32)[:total_k]
    else:
        A = torch.randn((m, total_k), device=device, dtype=dtype)
        # Make A m-major
        A = A.T.contiguous().T
        A_idx = None
    return A, A_idx


def run_lowlevel_varlen_k_gemm(
    A,
    B,
    out,
    cu_seqlens_k,
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
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        use_tma_gather=use_tma_gather,
    )


@sm100_tma_gather_only
@pytest.mark.parametrize("dynamic_persistent", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("m", [2048])
def test_gemm_varlen_k_tma_gather_matches_cpasync(
    m,
    n,
    input_dtype,
    dynamic_persistent,
):
    """Compare TMA gather vs cp.async gather for varlen_k."""
    device = "cuda"
    torch.random.manual_seed(42)
    num_groups = 4
    # Use K values divisible by tile_K (64 for bf16) to avoid partial-tile edge cases
    seq_lens = torch.randint(2, 6, (num_groups,), device="cpu") * 64
    total_k = seq_lens.sum().item()
    cu_seqlens_k = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    ).to(device)
    A, A_idx = generate_A_with_gather(m, total_k, device, input_dtype, gather_A=True)
    # B for quack_gemm varlen_k: 2D (n, total_k), n-major (stride(-2)==1)
    B_ref = torch.randn((total_k, n), device=device, dtype=input_dtype) / math.sqrt(
        total_k / num_groups
    )
    B = B_ref.T  # (n, total_k) with n contiguous — stride(-2)==1

    out_cpasync = torch.empty((num_groups, m, n), device=device, dtype=input_dtype)
    out_tma = torch.empty_like(out_cpasync)

    run_lowlevel_varlen_k_gemm(
        A,
        B,
        out_cpasync,
        cu_seqlens_k,
        A_idx,
        dynamic_persistent=dynamic_persistent,
        use_tma_gather=False,
    )
    run_lowlevel_varlen_k_gemm(
        A,
        B,
        out_tma,
        cu_seqlens_k,
        A_idx,
        dynamic_persistent=dynamic_persistent,
        use_tma_gather=True,
    )

    # gemm_ref expects B as (total_K, N)
    out_ref = gemm_ref(
        A.float(),
        B_ref.float(),
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
    )
    out_pt = gemm_ref(A, B_ref, cu_seqlens_k=cu_seqlens_k, A_idx=A_idx)

    assert out_tma.shape == (num_groups, m, n)
    assert (out_tma - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-5
    assert (out_cpasync - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-5
    torch.testing.assert_close(out_tma, out_cpasync, atol=3e-2, rtol=1e-3)


@pytest.mark.parametrize("permute_batch", [False, True])
@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [False])
@pytest.mark.parametrize("alpha_is_tensor", [False, True])
# @pytest.mark.parametrize("alpha_is_tensor", [False])
@pytest.mark.parametrize("alpha", [1.0, 0.93])
# @pytest.mark.parametrize("alpha", [1.0])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024, 1504, 4096])
@pytest.mark.parametrize("m", [2048, 1064, 8192])
# @pytest.mark.parametrize("n", [1024])
# @pytest.mark.parametrize("m", [2048])
@pytest.mark.parametrize("num_groups", [2, 4])
# @pytest.mark.parametrize("num_groups", [2])
def test_gemm_varlen_k(
    num_groups,
    m,
    n,
    input_dtype,
    alpha,
    alpha_is_tensor,
    dynamic_scheduler,
    gather_A,
    permute_batch,
):
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.randint(50, 300, (num_groups,), device="cpu")
    total_k = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_k = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_k = cu_seqlens_k.to(device)
    A, A_idx = generate_A_with_gather(m, total_k, device, input_dtype, gather_A)
    avg_k = total_k / num_groups
    B = torch.randn((total_k, n), device=device, dtype=input_dtype) / math.sqrt(avg_k)
    if alpha_is_tensor:
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
    if permute_batch:
        batch_idx_permute = torch.randperm(num_groups, device=device).to(torch.int32)
    else:
        batch_idx_permute = None
    out = gemm(
        A,
        B,
        alpha=alpha,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    assert out.shape == (num_groups, m, n)
    out_ref = gemm_ref(
        A.float(),
        B.float(),
        alpha=alpha_val,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
    )
    out_pt = gemm_ref(A, B, alpha=alpha_val, cu_seqlens_k=cu_seqlens_k, A_idx=A_idx)
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-4


@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("m", [2048])
def test_gemm_varlen_k_with_zero_lengths(
    m,
    n,
    input_dtype,
    gather_A,
):
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.tensor([150, 64, 0, 200, 0], device="cpu", dtype=torch.int32)
    num_groups = seq_lens.shape[0]
    total_k = seq_lens.sum().item()
    cu_seqlens_k = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_k = cu_seqlens_k.to(device)
    A, A_idx = generate_A_with_gather(m, total_k, device, input_dtype, gather_A)
    avg_k = total_k / num_groups
    B = torch.randn((total_k, n), device=device, dtype=input_dtype) / math.sqrt(avg_k)
    out = gemm(
        A,
        B,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        dynamic_scheduler=False,
        tuned=False,
    )
    assert out.shape == (num_groups, m, n)
    out_ref = gemm_ref(
        A.float(),
        B.float(),
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
    )
    out_pt = gemm_ref(A, B, cu_seqlens_k=cu_seqlens_k, A_idx=A_idx)
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-4


@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [False])
@pytest.mark.parametrize("C_major", ["m", "n"])
@pytest.mark.parametrize("alpha_is_tensor", [False, True])
@pytest.mark.parametrize("beta_is_tensor", [False, True])
@pytest.mark.parametrize("beta", [0.0, 1.17])
@pytest.mark.parametrize("alpha", [1.0, 0.93])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024, 1504])
@pytest.mark.parametrize("m", [2048, 1024])
@pytest.mark.parametrize("num_groups", [2, 4])
def test_gemm_add_varlen_k(
    num_groups,
    m,
    n,
    input_dtype,
    alpha,
    beta,
    alpha_is_tensor,
    beta_is_tensor,
    C_major,
    dynamic_scheduler,
    gather_A,
):
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.randint(50, 300, (num_groups,), device="cpu")
    total_k = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_k = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_k = cu_seqlens_k.to(device)
    A, A_idx = generate_A_with_gather(m, total_k, device, input_dtype, gather_A)
    # Make A m-major
    A = A.T.contiguous().T
    avg_k = total_k / num_groups
    B = torch.randn((total_k, n), device=device, dtype=input_dtype) / math.sqrt(avg_k)
    C = torch.randn((num_groups, m, n), device=device, dtype=input_dtype)
    if C_major == "m":
        C = C.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    if alpha_is_tensor:
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    if beta_is_tensor:
        beta = torch.tensor(beta, device=device, dtype=torch.float32)
    alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
    beta_val = beta.item() if torch.is_tensor(beta) else beta
    out = gemm_add(
        A,
        B,
        C,
        alpha=alpha,
        beta=beta,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    assert out.shape == (num_groups, m, n)
    out_ref = gemm_add_ref(
        A.float(),
        B.float(),
        C.float(),
        alpha=alpha_val,
        beta=beta_val,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
    )
    out_pt = gemm_add_ref(
        A, B, C, alpha=alpha_val, beta=beta_val, cu_seqlens_k=cu_seqlens_k, A_idx=A_idx
    )
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-4


@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [False])
@pytest.mark.parametrize("alpha_is_tensor", [False, True])
@pytest.mark.parametrize("beta_is_tensor", [False, True])
@pytest.mark.parametrize("beta", [0.0, 1.17])
@pytest.mark.parametrize("alpha", [1.0, 0.93])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024, 1504])
@pytest.mark.parametrize("m", [2048, 1024])
@pytest.mark.parametrize("num_groups", [2, 4])
def test_gemm_add_inplace_varlen_k(
    num_groups,
    m,
    n,
    input_dtype,
    alpha,
    beta,
    alpha_is_tensor,
    beta_is_tensor,
    dynamic_scheduler,
    gather_A,
):
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.randint(50, 300, (num_groups,), device="cpu")
    total_k = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_k = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_k = cu_seqlens_k.to(device)
    A, A_idx = generate_A_with_gather(m, total_k, device, input_dtype, gather_A)
    # Make A m-major
    A = A.T.contiguous().T
    avg_k = total_k / num_groups
    B = torch.randn((total_k, n), device=device, dtype=input_dtype) / math.sqrt(avg_k)
    out = torch.randn((num_groups, m, n), device=device, dtype=input_dtype)
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
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
    beta_val = beta.item() if torch.is_tensor(beta) else beta
    out_ref = gemm_add_ref(
        A.float(),
        B.float(),
        out_og.float(),
        out=None,  # Don't use in-place for reference
        alpha=alpha_val,
        beta=beta_val,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
    )
    out_pt = gemm_add_ref(
        A,
        B,
        out_og,
        out=None,
        alpha=alpha_val,
        beta=beta_val,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
    )
    assert out.shape == (num_groups, m, n), (
        f"Output shape mismatch: {out.shape} vs expected ({num_groups}, {m}, {n})"
    )
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-4


@pytest.mark.parametrize("pre_allocate_out", [False, True])
@pytest.mark.parametrize("gather_A", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [512])
@pytest.mark.parametrize("m", [1024])
@pytest.mark.parametrize("num_groups", [3])
def test_gemm_varlen_k_concat_out_m(num_groups, m, n, input_dtype, gather_A, pre_allocate_out):
    """Test varlen_k GEMM with concat_layout={"out"} (MoE dweight backward).

    Covers sonic-moe's down_projection_backward_weight call: cu_seqlens_k + A_idx
    (gather) + concat_layout=("out",) + pre-allocated out= buffer.
    """
    device = "cuda"
    torch.random.manual_seed(0)
    seq_lens = torch.randint(100, 200, (num_groups,), device=device)
    total_k = seq_lens.sum().item()
    cu_seqlens_k = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A, A_idx = generate_A_with_gather(m, total_k, device, input_dtype, gather_A)
    A = A / math.sqrt(total_k)
    B = torch.randn((total_k, n), device=device, dtype=input_dtype) / math.sqrt(total_k)
    concat_layout = ("out",)
    out_buf = (
        torch.empty((num_groups, m, n), device=device, dtype=input_dtype)
        if pre_allocate_out
        else None
    )
    out = gemm(
        A,
        B,
        out=out_buf,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        tuned=False,
        concat_layout=concat_layout,
    )
    out_ref = gemm_ref(
        A.float(),
        B.float(),
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        concat_layout=concat_layout,
    )
    out_pt = gemm_ref(A, B, cu_seqlens_k=cu_seqlens_k, A_idx=A_idx, concat_layout=concat_layout)
    if pre_allocate_out:
        assert_aliased(out, out_buf)
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-5


# ---- Empty-input tests for varlen_k. total_k=0 means each batch's contraction
# dim is empty (mathematically, output is per-batch zero matrix, then summed).
def _zero_cu_seqlens_k(L, device="cuda"):
    return torch.zeros(L + 1, dtype=torch.int32, device=device)


def _make_cu_seqlens_k(L, total_k, device="cuda"):
    cu = _zero_cu_seqlens_k(L, device)
    if total_k > 0:
        per = total_k // L
        cu[1:] = torch.arange(per, total_k + 1, per, dtype=torch.int32, device=device)
    return cu


@pytest.mark.parametrize("zero_dim", ["total_k", "M", "N"])
def test_gemm_varlen_k_empty(zero_dim):
    L, M, total_k, N = 4, 4096, 4096, 4096
    if zero_dim == "total_k":
        total_k = 0
    if zero_dim == "M":
        M = 0
    if zero_dim == "N":
        N = 0
    cu_seqlens_k = _make_cu_seqlens_k(L, total_k)
    A = torch.randn(M, total_k, device="cuda", dtype=torch.bfloat16)
    A = A.T.contiguous().T  # m-major as required by varlen_k
    B = torch.randn(total_k, N, device="cuda", dtype=torch.bfloat16)
    out = gemm(A, B, cu_seqlens_k=cu_seqlens_k, tuned=False)
    assert out.shape == (L, M, N)
    if total_k == 0:
        assert torch.all(out == 0)


@pytest.mark.parametrize("zero_dim", ["total_k", "M", "N"])
def test_gemm_add_varlen_k_empty(zero_dim):
    L, M, total_k, N = 4, 4096, 4096, 4096
    if zero_dim == "total_k":
        total_k = 0
    if zero_dim == "M":
        M = 0
    if zero_dim == "N":
        N = 0
    cu_seqlens_k = _make_cu_seqlens_k(L, total_k)
    A = torch.randn(M, total_k, device="cuda", dtype=torch.bfloat16)
    A = A.T.contiguous().T
    B = torch.randn(total_k, N, device="cuda", dtype=torch.bfloat16)
    C = torch.randn(L, M, N, device="cuda", dtype=torch.bfloat16)
    out = gemm_add(A, B, C, cu_seqlens_k=cu_seqlens_k, tuned=False)
    assert out.shape == (L, M, N)
    if total_k == 0:
        assert torch.equal(out, C)
