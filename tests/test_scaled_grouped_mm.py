# Copyright (c) 2026, Tri Dao.
"""Tests for the phased ``scaled_grouped_mm`` interface (quack/scaled_grouped_mm.py).

Phase 1 is a local MXFP8 grouped GEMM. Covers numerical correctness against a per-group
dequantized matmul reference (including hot-expert skew), the Phase-1 contract (fusion
descriptors / SM budgeting / out_dtype rejected), and CUDA-graph capturability (the
group_sizes -> offsets cumsum must stay on device).
"""

import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.mx_utils import to_mx_compiled
from quack.scaled_grouped_mm import (
    GatherPrologue,
    KernelConfig,
    OffsetPrologue,
    PreparedGroupedWeights,
    ScatterEpilogue,
    ScatterSwiGLUEpilogue,
    prepare_weights,
    scaled_grouped_mm,
)

SF_VEC = 32


def _skip_if_not_sm100():
    if not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (
        10,
        11,
    ):
        pytest.skip("MXFP8 grouped GEMM requires SM100 (B200/B300) or SM110")


def _make_inputs(group_sizes, k, n, seed=0):
    """Build a Phase-1 problem in the public layout: a=(M,K), b=(G,N,K), scale_a=(M,sf_k),
    scale_b=(G,N,sf_k), group_sizes=(G,) int32; plus fp32 dequant refs."""
    torch.manual_seed(seed)
    e, total_m, sf_k = len(group_sizes), sum(group_sizes), k // SF_VEC
    dev = torch.device("cuda")
    std = k**-0.5
    a_hp = (torch.randn(total_m, k, dtype=torch.bfloat16, device=dev) * std).contiguous()
    qa, sa = to_mx_compiled(a_hp, SF_VEC)  # (M,K) e4m3, (M,sf_k) e8m0
    b_hp = (torch.randn(e, n, k, dtype=torch.bfloat16, device=dev) * std).contiguous()
    qb_flat, sb = to_mx_compiled(b_hp.view(e * n, k), SF_VEC)
    qb = qb_flat.view(e, n, k)  # (G,N,K) -- the public ``b`` layout
    sb = sb.view(e, n, sf_k)
    a_ref = qa.float() * sa.float().repeat_interleave(SF_VEC, dim=-1)
    b_ref = qb.float() * sb.float().repeat_interleave(SF_VEC, dim=-1)  # (G,N,K)
    gs = torch.tensor(list(group_sizes), dtype=torch.int32, device=dev)
    return qa, qb, sa, sb, gs, a_ref, b_ref


def _ref_grouped(a_ref, b_ref, group_sizes):
    outs, start = [], 0
    for g, gs in enumerate(group_sizes):
        outs.append(a_ref[start : start + gs] @ b_ref[g].T)  # (gs, N)
        start += gs
    return torch.cat(outs)


@pytest.mark.parametrize(
    "group_sizes",
    [
        (256, 256, 256, 256),  # uniform 128-aligned
        (128, 256, 384),  # ragged 128-aligned
        (100, 300, 200, 424),  # ragged non-128
        (8192, 704, 704, 704, 704, 704, 704, 704),  # hot-expert skew
        (256, 0, 256, 256),  # empty expert
    ],
)
def test_scaled_grouped_mm_correct(group_sizes):
    """Phase 1 output matches a per-group dequantized matmul (out = a @ b.t() per group)."""
    _skip_if_not_sm100()
    k, n = 512, 512
    qa, qb, sa, sb, gs, a_ref, b_ref = _make_inputs(group_sizes, k, n)
    out = scaled_grouped_mm(qa, qb, sa, sb, gs)
    assert tuple(out.shape) == (sum(group_sizes), n)
    torch.testing.assert_close(
        out.float(), _ref_grouped(a_ref, b_ref, group_sizes), atol=1e-2, rtol=1e-2
    )


def test_scaled_grouped_mm_phase1_contract():
    """Phase 2+ fusions / SM budgeting / non-bf16 out are rejected; scale_a is required."""
    _skip_if_not_sm100()
    qa, qb, sa, sb, gs, _, _ = _make_inputs((128, 256), 256, 256)
    m = qa.shape[0]
    ptrs = torch.zeros(m, dtype=torch.int64, device=qa.device)

    with pytest.raises(NotImplementedError):
        scaled_grouped_mm(qa, qb, sa, sb, gs, prologue=GatherPrologue(ptrs, None))
    with pytest.raises(NotImplementedError):
        scaled_grouped_mm(qa, qb, sa, sb, gs, prologue=OffsetPrologue(ptrs))
    with pytest.raises(NotImplementedError):
        scaled_grouped_mm(qa, qb, sa, sb, gs, epilogue=ScatterEpilogue(ptrs, None))
    with pytest.raises(NotImplementedError):
        scaled_grouped_mm(qa, qb, sa, sb, gs, epilogue=ScatterSwiGLUEpilogue(ptrs, None))
    with pytest.raises(NotImplementedError):
        scaled_grouped_mm(qa, qb, sa, sb, gs, kernel_config=KernelConfig(num_sms=108))
    with pytest.raises(NotImplementedError):
        scaled_grouped_mm(qa, qb, sa, sb, gs, out_dtype=torch.float32)
    with pytest.raises(ValueError):
        scaled_grouped_mm(qa, qb, None, sb, gs)


def test_scaled_grouped_mm_capturable():
    """group_sizes -> offsets is a device cumsum, so the whole call is CUDA-graph capturable."""
    _skip_if_not_sm100()
    group_sizes = (100, 300, 200, 424)  # non-128 -> dQaccum path
    k, n = 512, 512
    qa, qb, sa, sb, gs, a_ref, b_ref = _make_inputs(group_sizes, k, n)
    out = scaled_grouped_mm(qa, qb, sa, sb, gs)  # warm up / compile
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        scaled_grouped_mm(qa, qb, sa, sb, gs, out=out)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        scaled_grouped_mm(qa, qb, sa, sb, gs, out=out)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.float(), _ref_grouped(a_ref, b_ref, group_sizes), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize(
    "group_sizes",
    [(128, 256, 384), (100, 300, 200, 424), (8192, 704, 704, 704, 704, 704, 704, 704)],
)
def test_prepare_weights_matches_stateless(group_sizes):
    """Pre-packed weights give bit-identical output to the stateless path (same kernel); both the
    handle call and the free-function-with-handle form agree."""
    _skip_if_not_sm100()
    k, n = 512, 512
    qa, qb, sa, sb, gs, _, _ = _make_inputs(group_sizes, k, n)
    stateless = scaled_grouped_mm(qa, qb, sa, sb, gs)
    w = prepare_weights(qb, sb)
    assert isinstance(w, PreparedGroupedWeights) and w.shape == tuple(qb.shape)
    assert torch.equal(w(qa, sa, gs), stateless)  # handle call
    assert torch.equal(scaled_grouped_mm(qa, w, sa, None, gs), stateless)  # free fn + handle


def test_prepare_weights_scale_b_must_be_none():
    """Passing scale_b alongside prepared weights is an error -- B-scale is already packed."""
    _skip_if_not_sm100()
    qa, qb, sa, sb, gs, _, _ = _make_inputs((128, 256), 256, 256)
    w = prepare_weights(qb, sb)
    with pytest.raises(ValueError):
        scaled_grouped_mm(qa, w, sa, sb, gs)


def test_prepare_weights_capturable():
    """The prepared path (pre-packed B, per-call A-build) is also CUDA-graph capturable."""
    _skip_if_not_sm100()
    group_sizes = (100, 300, 200, 424)
    k, n = 512, 512
    qa, qb, sa, sb, gs, a_ref, b_ref = _make_inputs(group_sizes, k, n)
    w = prepare_weights(qb, sb)
    out = w(qa, sa, gs)  # warm up / compile
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        w(qa, sa, gs, out=out)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        w(qa, sa, gs, out=out)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.float(), _ref_grouped(a_ref, b_ref, group_sizes), atol=1e-2, rtol=1e-2
    )
