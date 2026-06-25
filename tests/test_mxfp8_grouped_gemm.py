# Copyright (c) 2026, Tri Dao.
"""Tests for the MXFP8 grouped GEMM dispatcher (quack/mxfp8_grouped_gemm.py).

Covers route selection, per-route numerical correctness against a per-group
dequantized matmul reference, and CUDA-graph capturability of the sync-free paths.
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


def test_choose_route_pure():
    """Route selection from cumulative group offsets (no GPU)."""
    from quack.mxfp8_grouped_gemm import _choose_route

    assert _choose_route((256, 512, 768, 1024)) == ("uniform", 256)  # equal, 128-aligned
    assert _choose_route((128, 384, 768, 1024)) == ("varlen", None)  # ragged, 128-aligned
    assert _choose_route((100, 400, 600, 1024)) == ("varlen_dqaccum", None)  # non-128
    assert _choose_route((100, 100, 200, 224)) == ("varlen_dqaccum", None)  # non-128 + empty


@pytest.mark.parametrize(
    "group_sizes",
    [
        (256, 256, 256, 256),  # uniform 128-aligned -> batched-L dense
        (128, 256, 384, 256),  # ragged 128-aligned  -> varlen-M (natural SFA)
        (100, 300, 200, 424),  # non-128             -> varlen-M (dQaccum-padded SFA)
        (256, 0, 256, 256),  # empty expert
    ],
)
def test_mxfp8_grouped_gemm_correct(group_sizes):
    """Each route matches a per-group fp32 dequant reference, and the eager and prepared
    APIs (same kernel, two entry points) are bit-identical."""
    _skip_if_not_sm100()
    from quack.mxfp8_grouped_gemm import MXFP8GroupedGemm, mxfp8_grouped_gemm

    qa, b_disp, offs, sa, sb, a_ref, b_ref = _make_grouped_mxfp8(group_sizes, 512, 512)
    ref = _ref_grouped(a_ref, b_ref, group_sizes)
    eager = mxfp8_grouped_gemm(qa, b_disp, offs, sa, sb)
    prepared = MXFP8GroupedGemm(b_disp, sb)(qa, offs, sa)
    torch.testing.assert_close(eager.float(), ref, atol=1e-2, rtol=1e-2)
    assert torch.equal(eager, prepared)


@pytest.mark.parametrize("mode", ["uniform", "varlen"])
def test_mxfp8_grouped_gemm_cuda_graph_capturable(mode):
    """The sync-free paths (uniform=True / varlen=True) capture and replay under a CUDA
    graph and stay correct -- guards against reintroducing a host<->device sync."""
    _skip_if_not_sm100()
    from quack.mxfp8_grouped_gemm import MXFP8GroupedGemm

    group_sizes = (256,) * 4 if mode == "uniform" else (128, 256, 384, 256)
    qa, b_disp, offs, sa, sb, a_ref, b_ref = _make_grouped_mxfp8(group_sizes, 512, 512)
    gemm = MXFP8GroupedGemm(b_disp, sb)
    out = torch.empty(sum(group_sizes), 512, dtype=torch.bfloat16, device=qa.device)
    kw = {"uniform": True} if mode == "uniform" else {"varlen": True}
    call = lambda: gemm(qa, offs, sa, out=out, **kw)

    call()  # compile outside capture
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):  # raises if the path does any host<->device copy / sync
        call()
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.float(), _ref_grouped(a_ref, b_ref, group_sizes), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("builder", ["device", "triton"])
@pytest.mark.parametrize("k", [256, 160])  # 160 -> sf_k=5 (not a multiple of 4): partial K-block
@pytest.mark.parametrize(
    "group_sizes",
    [
        (100, 200, 156),  # ragged non-128
        (128, 256, 384),  # all 128-aligned
        (256, 0, 256, 256),  # empty expert (middle)
        (0, 128, 100, 28),  # empty expert (first) + non-128
        (100, 300, 200, 400),  # non-128, total_m % 128 != 0
        (128, 128),  # total_m exact multiple of 128
    ],
)
def test_dqaccum_padded_sfa_byte_identical(group_sizes, k, builder):
    """Both device SFA builders must be byte-identical to the host scatter -- the unchanged
    kernel reads this exact layout. Guards the cu//128+i placement and the blocked swizzle."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from quack.mxfp8_grouped_gemm import (
        _dqaccum_padded_sfa,
        _dqaccum_padded_sfa_device,
        _dqaccum_padded_sfa_triton,
    )

    build = {"device": _dqaccum_padded_sfa_device, "triton": _dqaccum_padded_sfa_triton}[builder]
    torch.manual_seed(0)
    e = len(group_sizes)
    total_m = sum(group_sizes)
    sf_k = k // SF_VEC
    dev = torch.device("cuda")
    sfa = torch.randint(0, 256, (total_m, sf_k), dtype=torch.uint8, device=dev).view(
        torch.float8_e8m0fnu
    )
    offs = torch.tensor(list(itertools.accumulate(group_sizes)), dtype=torch.int32, device=dev)
    host = _dqaccum_padded_sfa(sfa, list(group_sizes), sf_k, e)
    got = build(sfa, offs, sf_k, e)
    assert got.shape == host.shape, f"shape {tuple(got.shape)} != {tuple(host.shape)}"
    assert torch.equal(got.view(torch.uint8), host.view(torch.uint8))


@pytest.mark.parametrize(
    "group_sizes",
    [
        (100, 300, 200, 424),  # non-128 boundaries, total_m % 128 == 0
        (256, 0, 256, 256),  # empty expert
        (100, 300, 200, 400),  # non-128 boundaries AND total_m % 128 != 0
    ],
)
def test_mxfp8_grouped_gemm_varlen_nonaligned(group_sizes):
    """varlen_nonaligned=True: arbitrary (non-128) ragged groups via the device-built
    dQaccum SFA, matching a per-group dequant reference; eager and prepared are identical."""
    _skip_if_not_sm100()
    from quack.mxfp8_grouped_gemm import MXFP8GroupedGemm, mxfp8_grouped_gemm

    qa, b_disp, offs, sa, sb, a_ref, b_ref = _make_grouped_mxfp8(group_sizes, 512, 512)
    ref = _ref_grouped(a_ref, b_ref, group_sizes)
    eager = mxfp8_grouped_gemm(qa, b_disp, offs, sa, sb, varlen_nonaligned=True)
    prepared = MXFP8GroupedGemm(b_disp, sb, varlen_nonaligned=True)(qa, offs, sa)
    torch.testing.assert_close(eager.float(), ref, atol=1e-2, rtol=1e-2)
    assert torch.equal(eager, prepared)


@pytest.mark.parametrize("group_sizes", [(100, 300, 200, 424), (256, 0, 256, 256)])
def test_mxfp8_grouped_gemm_varlen_nonaligned_capturable(group_sizes):
    """The non-128 path with varlen_nonaligned=True captures + replays under a CUDA graph.
    torch.cuda.graph raises on any host<->device sync -- the empirical syncless proof
    (a source grep can't see a sync hidden inside an ATen op)."""
    _skip_if_not_sm100()
    from quack.mxfp8_grouped_gemm import MXFP8GroupedGemm

    qa, b_disp, offs, sa, sb, a_ref, b_ref = _make_grouped_mxfp8(group_sizes, 512, 512)
    gemm = MXFP8GroupedGemm(b_disp, sb, varlen_nonaligned=True)
    out = torch.empty(sum(group_sizes), 512, dtype=torch.bfloat16, device=qa.device)
    call = lambda: gemm(qa, offs, sa, out=out)

    call()  # compile outside capture
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):  # raises if the path does any host<->device copy / sync
        call()
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.float(), _ref_grouped(a_ref, b_ref, group_sizes), atol=1e-2, rtol=1e-2
    )
