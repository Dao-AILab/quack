# Copyright (c) 2026, Tri Dao.
"""Torch-facing epilogue-object interface (Tier 4 surface): eager __call__,
plan()/run(), out= buffers, reduce finalization. Numeric pins are against
torch references, independent of the variant wrappers."""

import pytest
import torch

from quack.epi_ops import ColVecReduce, RowVecReduce
from quack.gemm_config import GemmConfig
from quack.gemm_epilogue import gemm_epilogue

torch.manual_seed(0)

CFG = GemmConfig(
    tile_m=128,
    tile_n=256,
    tile_k=None,
    pingpong=False,
    is_dynamic_persistent=False,
    cluster_m=2,
    cluster_n=1,
)


@gemm_epilogue(outputs=("doubled",))
def _bias_double(acc, bias):
    y = acc + bias
    return {"D": y, "doubled": y * 2.0}


@gemm_epilogue(reduces={"sq": ColVecReduce("sq", scaled=True)})
def _sq_reduce(acc):
    return {"D": acc, "sq": (acc, acc)}


@gemm_epilogue(reduces={"colsum": RowVecReduce("colsum")})
def _colsum(acc):
    return {"D": acc, "colsum": acc}


def _inputs(m=512, n=768, k=256, dtype=torch.bfloat16):
    A = torch.randn(m, k, device="cuda", dtype=dtype) / 8
    B = torch.randn(k, n, device="cuda", dtype=dtype) / 8
    return A, B


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_eager_basic(dtype):
    A, B = _inputs(dtype=dtype)
    bias = torch.randn(1, B.shape[-1], device="cuda", dtype=torch.float32)
    res = _bias_double(A, B, config=CFG, bias=bias)
    ref = (A.float() @ B.float()) + bias
    assert set(res) == {"D", "doubled"}
    torch.testing.assert_close(res["D"].float(), ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(res["doubled"].float(), 2 * ref, atol=4e-2, rtol=2e-2)


def test_eager_out_buffers_and_store_d():
    A, B = _inputs()
    bias = torch.randn(1, B.shape[-1], device="cuda", dtype=torch.float32)
    dbl = torch.empty(A.shape[0], B.shape[-1], device="cuda", dtype=torch.float32)
    # provided buffer (mixed dtype!) is used as-is; store_d=False compiles no D store
    res = _bias_double(A, B, config=CFG, bias=bias, out={"doubled": dbl}, store_d=False)
    assert "D" not in res and res["doubled"] is dbl
    ref = 2 * ((A.float() @ B.float()) + bias)
    torch.testing.assert_close(dbl, ref, atol=4e-2, rtol=2e-2)


def test_reduce_finalized():
    A, B = _inputs()
    res = _sq_reduce(A, B, config=CFG)
    ref_d = A.float() @ B.float()
    torch.testing.assert_close(res["D"].float(), ref_d, atol=2e-2, rtol=2e-2)
    # sq comes back finalized: rowwise sum of squares (fp32 accumulator)
    torch.testing.assert_close(res["sq"], ref_d.square().sum(-1), rtol=1e-3, atol=1e-2)


# RowVecReduce requires cluster_M == 1: partials silently corrupt under
# cluster_M=2 (see the xfail below + HANDOFF known issues; likely the same
# SM100 cluster-coordinate bug as the gated aux-store corruption).
CFG_CM1 = CFG.__class__(**{**CFG.__dict__, "cluster_m": 1})


def test_rowvec_reduce_finalized():
    A, B = _inputs()
    res = _colsum(A, B, config=CFG_CM1)
    ref_d = A.float() @ B.float()
    torch.testing.assert_close(res["colsum"], ref_d.sum(0), rtol=1e-3, atol=1e-1)


@pytest.mark.xfail(
    strict=True,
    reason="RowVecReduce partials corrupt with cluster_M=2 on SM100 — open bug, "
    "see HANDOFF known issues (gated cluster_M=2 family)",
)
def test_rowvec_reduce_cluster_m2():
    A, B = _inputs()
    res = _colsum(A, B, config=CFG)
    ref_d = A.float() @ B.float()
    torch.testing.assert_close(res["colsum"], ref_d.sum(0), rtol=1e-3, atol=1e-1)


def test_plan_run_matches_eager():
    A, B = _inputs()
    bias = torch.randn(1, B.shape[-1], device="cuda", dtype=torch.float32)
    eager = _bias_double(A, B, config=CFG, bias=bias)
    out = {
        "D": torch.empty_like(eager["D"]),
        "doubled": torch.empty_like(eager["doubled"]),
    }
    plan = _bias_double.plan(A, B, out=out, config=CFG, bias=bias)
    plan.run(A, B, out=out, bias=bias)
    assert torch.equal(out["D"], eager["D"])
    assert torch.equal(out["doubled"], eager["doubled"])
    # plan() itself must not have written the buffers: rerun with fresh data
    A2, B2 = _inputs()
    plan.run(A2, B2, out=out, bias=bias)
    eager2 = _bias_double(A2, B2, config=CFG, bias=bias)
    assert torch.equal(out["D"], eager2["D"])


def test_plan_never_launches():
    A, B = _inputs()
    out = {
        "D": torch.full((512, 768), 7.0, device="cuda", dtype=torch.bfloat16),
        "doubled": torch.full((512, 768), 7.0, device="cuda", dtype=torch.bfloat16),
    }
    bias = torch.zeros(1, 768, device="cuda", dtype=torch.float32)
    _bias_double.plan(A, B, out=out, config=CFG, bias=bias)
    torch.cuda.synchronize()
    assert (out["D"] == 7.0).all() and (out["doubled"] == 7.0).all()


def test_plan_run_with_reduce_scratch():
    A, B = _inputs()
    eager = _sq_reduce(A, B, config=CFG)
    out = {"D": torch.empty_like(eager["D"])}
    plan = _sq_reduce.plan(A, B, out=out, config=CFG)
    plan.run(A, B, out=out)
    # plan-attached scratch carries the partials; finalize by hand
    sq = plan.scratch["sq"].sum(-1)
    torch.testing.assert_close(sq, eager["sq"], rtol=1e-5, atol=1e-5)
    assert torch.equal(out["D"], eager["D"])


def test_batched_3d():
    L, m, n, k = 3, 256, 512, 256
    A = torch.randn(L, m, k, device="cuda", dtype=torch.bfloat16) / 8
    B = torch.randn(L, k, n, device="cuda", dtype=torch.bfloat16) / 8
    bias = torch.randn(L, n, device="cuda", dtype=torch.float32)
    res = _bias_double(A, B, config=CFG, bias=bias)
    ref = torch.bmm(A.float(), B.float()) + bias[:, None, :]
    torch.testing.assert_close(res["D"].float(), ref, atol=2e-2, rtol=2e-2)


def test_eager_tuned_smoke():
    """tuned=True rides quack.epi_autotune (sweep once per metadata class,
    warm replay after); numerics must match the pinned-config call."""
    A, B = _inputs(m=256, n=512, k=256)
    bias = torch.randn(1, B.shape[-1], device="cuda", dtype=torch.float32)
    res = _bias_double(A, B, bias=bias)  # tuned
    ref = _bias_double(A, B, config=CFG, bias=bias)
    torch.testing.assert_close(res["D"].float(), ref["D"].float(), atol=2e-2, rtol=2e-2)
    # warm replay hits the tuner cache + mod.gemm plan cache
    res2 = _bias_double(A, B, bias=bias)
    assert torch.equal(res2["D"], res["D"])


def test_eager_tuned_reduce_sinks():
    """Worst-case sink buffers through the tuned path come back finalized."""
    A, B = _inputs(m=256, n=512, k=256)
    res = _sq_reduce(A, B)  # tuned
    ref_d = A.float() @ B.float()
    torch.testing.assert_close(res["sq"], ref_d.square().sum(-1), rtol=1e-3, atol=1e-2)


def test_from_class_static():
    """Rung-3 escape hatch: a hand-written class (the default-epi GEMM) gets
    the plan/run interface without the fn frontend. Power API: B arrives
    dispatch-shaped (n, k), epi_args explicit (all default-epi ops absent =
    plain D = A @ B)."""
    from quack.gemm import GemmDefaultSm100
    from quack.gemm_epilogue import epilogue_from_class

    if torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("SM100 class")
    A, B = _inputs()
    D = torch.empty(A.shape[0], B.shape[-1], device="cuda", dtype=torch.bfloat16)
    epi = epilogue_from_class(GemmDefaultSm100)
    plan = epi.plan(A, B.mT, D, epi_args={}, config=CFG)
    plan.run(A, B.mT, out={"D": D})
    torch.testing.assert_close(D.float(), A.float() @ B.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("fullgraph", [True])
def test_torch_compile_single_op(fullgraph):
    """The single quack::gemm_epi custom op: any epilogue object under
    torch.compile, numerics matching eager."""
    A, B = _inputs(m=256, n=512, k=256)
    bias = torch.randn(1, B.shape[-1], device="cuda", dtype=torch.float32)

    def f(A, B, bias):
        res = _bias_double(A, B, config=CFG, bias=bias)
        return res["D"], res["doubled"]

    eager_d, eager_dbl = f(A, B, bias)
    cf = torch.compile(f, fullgraph=fullgraph, dynamic=False)
    comp_d, comp_dbl = cf(A, B, bias)
    assert torch.equal(comp_d, eager_d)
    assert torch.equal(comp_dbl, eager_dbl)
    # second call: compiled-graph replay, fresh data
    A2, B2 = _inputs(m=256, n=512, k=256)
    comp_d2, _ = cf(A2, B2, bias)
    eager_d2, _ = f(A2, B2, bias)
    assert torch.equal(comp_d2, eager_d2)


def test_torch_compile_reduce_sinks():
    """Reduce sinks under compile: config pinned, partials graph-allocated,
    finalization traced."""
    A, B = _inputs(m=256, n=512, k=256)

    def f(A, B):
        res = _sq_reduce(A, B, config=CFG)
        return res["D"], res["sq"]

    eager_d, eager_sq = f(A, B)
    comp_d, comp_sq = torch.compile(f, fullgraph=True, dynamic=False)(A, B)
    assert torch.equal(comp_d, eager_d)
    torch.testing.assert_close(comp_sq, eager_sq, rtol=1e-5, atol=1e-5)


def test_varlen_m():
    """varlen through the object surface; B rides the (k, n) trace relabel
    (varlen b_kn, 2026-07-14). varlen operands are per-sequence indexed and
    3D BY CONTRACT: shared weights are the caller's zero-copy stride-0
    expand (2D B raises — see test_varlen_m_2d_b_rejected)."""
    total_m, n, k = 768, 512, 256
    cu = torch.tensor([0, 128, 448, 768], device="cuda", dtype=torch.int32)
    A = torch.randn(total_m, k, device="cuda", dtype=torch.bfloat16) / 8
    B = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / 8
    B = B.unsqueeze(0).expand(3, -1, -1)
    bias = torch.randn(1, n, device="cuda", dtype=torch.float32).expand(3, -1)
    res = _bias_double(A, B, config=CFG, bias=bias, cu_seqlens_m=cu)
    ref = (A[..., :].float() @ B[0].float()) + bias[0]
    torch.testing.assert_close(res["D"].float(), ref, atol=2e-2, rtol=2e-2)


def test_varlen_m_2d_b_rejected():
    """2D B under varlen_m is a loud error, not a batch-of-one (the old
    silent-OOB hazard, found 2026-07-14)."""
    total_m, n, k = 768, 512, 256
    cu = torch.tensor([0, 128, 448, 768], device="cuda", dtype=torch.int32)
    A = torch.randn(total_m, k, device="cuda", dtype=torch.bfloat16) / 8
    B = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / 8
    with pytest.raises(ValueError, match="per-sequence"):
        _bias_double(A, B, config=CFG, cu_seqlens_m=cu)


CFG_SWAP = CFG.__class__(**{**CFG.__dict__, "swap_ab": True})


def test_swap_at_trace():
    """swap_ab configs ride trace-time relabels (a/cd_transposed): no host
    views, SASS byte-identical to the host-view swap (AI/swap_sass_ab.py)."""
    A, B = _inputs()
    bias = torch.randn(1, B.shape[-1], device="cuda", dtype=torch.float32)
    res = _bias_double(A, B, config=CFG_SWAP, bias=bias)
    ref = (A.float() @ B.float()) + bias
    torch.testing.assert_close(res["D"].float(), ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(res["doubled"].float(), 2 * ref, atol=4e-2, rtol=2e-2)
    # plan/run with a swapped config: zero-decision replay, bitwise vs eager
    out = {"D": torch.empty_like(res["D"]), "doubled": torch.empty_like(res["doubled"])}
    plan = _bias_double.plan(A, B, out=out, config=CFG_SWAP, bias=bias)
    plan.run(A, B, out=out, bias=bias)
    assert torch.equal(out["D"], res["D"])


def test_swap_at_trace_colvec():
    """A caller colvec becomes the kernel rowvec under swap (kind inference
    in kernel coordinates)."""

    @gemm_epilogue()
    def _scale_epi(acc, scale):
        return {"D": acc * scale}

    A, B = _inputs()
    scale = torch.randn(1, A.shape[0], device="cuda", dtype=torch.float32).abs() + 0.5
    res = _scale_epi(A, B, config=CFG_SWAP, scale=scale)
    ref = (A.float() @ B.float()) * scale[0, :, None]
    torch.testing.assert_close(res["D"].float(), ref, atol=2e-2, rtol=2e-2)


def test_swap_requires_contract():
    """swap_ab is element-mode, sink-less, dense-only at mod.gemm level."""
    A, B = _inputs()
    with pytest.raises(ValueError, match="element-mode"):
        _sq_reduce(A, B, config=CFG_SWAP)
