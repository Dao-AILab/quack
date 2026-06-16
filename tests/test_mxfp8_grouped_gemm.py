# Copyright (c) 2026, Tri Dao.
"""Parity tests for the MXFP8 grouped GEMM dispatcher (quack/mxfp8_grouped_gemm.py).

Exercises all three routes (uniform-batched, ragged-varlen-M, ragged-padded) and
checks each against a per-group dequantized matmul reference.
"""

import itertools

import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.mx_utils import to_mx_compiled

SF_VEC = 32


def _skip_if_not_sm100():
    if not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (
        10,
        11,
    ):
        pytest.skip("MXFP8 grouped GEMM requires SM100 (B200/B300) or SM110")


def _make_grouped_mxfp8(group_sizes, k, n, seed=0):
    """Build a grouped MXFP8 problem in the dispatcher's input layout + fp32 dequant refs."""
    torch.manual_seed(seed)
    e = len(group_sizes)
    total_m = sum(group_sizes)
    sf_k = k // SF_VEC
    dev = torch.device("cuda")
    std = k**-0.5
    a_hp = (torch.randn(total_m, k, dtype=torch.bfloat16, device=dev) * std).contiguous()
    qa, sa = to_mx_compiled(a_hp, SF_VEC)  # (total_m, k) e4m3, (total_m, sf_k) e8m0
    b_hp = (torch.randn(e, n, k, dtype=torch.bfloat16, device=dev) * std).contiguous()
    qb_flat, sb = to_mx_compiled(b_hp.view(e * n, k), SF_VEC)
    qb = qb_flat.view(e, n, k)
    sb = sb.view(e, n, sf_k)
    b_disp = qb.transpose(1, 2)  # (E, K, N) K-contig (b.transpose(-2,-1) is contiguous)
    a_ref = qa.float() * sa.float().repeat_interleave(SF_VEC, dim=-1)  # (total_m, k)
    b_ref = qb_flat.float().view(e, n, k) * sb.float().repeat_interleave(SF_VEC, dim=-1)  # (e,n,k)
    offs = torch.tensor(list(itertools.accumulate(group_sizes)), dtype=torch.int32, device=dev)
    return qa, b_disp, offs, sa, sb, a_ref, b_ref


def _ref_grouped(a_ref, b_ref, group_sizes):
    outs, start = [], 0
    for g, gs in enumerate(group_sizes):
        outs.append(a_ref[start : start + gs] @ b_ref[g].T)  # (gs, n)
        start += gs
    return torch.cat(outs)


@pytest.mark.parametrize("api", ["eager", "prepared"])
@pytest.mark.parametrize("k,n", [(256, 256), (512, 512), (6144, 4096)])
@pytest.mark.parametrize(
    "route,group_sizes",
    [
        ("uniform", (256, 256, 256, 256)),  # uniform 128-aligned -> batched-L
        ("varlen", (128, 256, 384, 256)),  # ragged 128-aligned   -> varlen-M
        ("padded", (100, 300, 200, 424)),  # ragged non-128       -> padded
    ],
)
def test_mxfp8_grouped_gemm(api, route, group_sizes, k, n):
    _skip_if_not_sm100()
    from quack.mxfp8_grouped_gemm import make_mxfp8_grouped_gemm_runner, mxfp8_grouped_gemm

    qa, b_disp, offs, sa, sb, a_ref, b_ref = _make_grouped_mxfp8(group_sizes, k, n)
    if api == "eager":
        out = mxfp8_grouped_gemm(qa, b_disp, offs, sa, sb)
    else:
        out = make_mxfp8_grouped_gemm_runner(qa, b_disp, offs, sa, sb)()
    assert out.shape == (sum(group_sizes), n)
    ref = _ref_grouped(a_ref, b_ref, group_sizes)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("k,n", [(512, 512), (6144, 4096)])
@pytest.mark.parametrize(
    "route,group_sizes",
    [
        ("uniform", (256, 256, 256, 256)),
        ("varlen", (128, 256, 384, 256)),
        ("padded", (100, 300, 200, 424)),
    ],
)
def test_mxfp8_grouped_gemm_eager_prepared_bitwise(route, group_sizes, k, n):
    """The eager and prepared APIs compile the SAME GemmSm100 kernel, so their
    outputs must be bit-identical -- not merely within tolerance. This pins the
    scale-pack / reshape plumbing: a subtly-wrong prepared path could still pass
    the 1e-2 dequant check above but would diverge here. (Unlike a cross-kernel
    bitwise check vs torch._scaled_grouped_mm, this is robust -- it does not
    depend on neither side picking split-K.)"""
    _skip_if_not_sm100()
    from quack.mxfp8_grouped_gemm import make_mxfp8_grouped_gemm_runner, mxfp8_grouped_gemm

    qa, b_disp, offs, sa, sb, _, _ = _make_grouped_mxfp8(group_sizes, k, n)
    eager = mxfp8_grouped_gemm(qa, b_disp, offs, sa, sb)
    prepared = make_mxfp8_grouped_gemm_runner(qa, b_disp, offs, sa, sb)()
    assert torch.equal(eager, prepared), (
        f"eager vs prepared not bit-identical: "
        f"{(eager != prepared).sum().item()}/{eager.numel()} elems differ"
    )


@pytest.mark.parametrize("k,n", [(512, 512), (4096, 4096)])
@pytest.mark.parametrize(
    "group_sizes",
    [
        (1792, 128, 128, 128),  # 128-aligned, heavy load imbalance -> varlen-M
        (512, 0, 0, 0),  # 128-aligned, single non-empty expert
        (256, 0, 256, 256),  # 128-aligned, empty middle expert
        (0, 256, 256, 256),  # 128-aligned, empty first expert
        (256, 256, 256, 0),  # 128-aligned, empty last expert
        (100, 0, 200, 224),  # non-128, empty expert -> padded route
        (500, 12, 8, 4),  # non-128, heavy imbalance -> padded route
    ],
)
def test_mxfp8_grouped_gemm_skew_and_empty(group_sizes, k, n):
    """Realistic MoE routing: heavy load imbalance and empty experts (size-0 groups).
    Exercises the varlen-M route's dQaccum-padded SFA build and the padded route with
    m_i == 0. Checks both APIs vs the dequant reference and that they agree bitwise."""
    _skip_if_not_sm100()
    from quack.mxfp8_grouped_gemm import make_mxfp8_grouped_gemm_runner, mxfp8_grouped_gemm

    qa, b_disp, offs, sa, sb, a_ref, b_ref = _make_grouped_mxfp8(group_sizes, k, n)
    eager = mxfp8_grouped_gemm(qa, b_disp, offs, sa, sb)
    prepared = make_mxfp8_grouped_gemm_runner(qa, b_disp, offs, sa, sb)()
    assert eager.shape == (sum(group_sizes), n)
    torch.testing.assert_close(
        eager.float(), _ref_grouped(a_ref, b_ref, group_sizes), atol=1e-2, rtol=1e-2
    )
    assert torch.equal(eager, prepared)
