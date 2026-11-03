# Copyright (c) 2026, Tri Dao.
"""Torch-facing epilogue-object interface (Tier 4 surface): eager __call__,
plan()/run(), out= buffers, reduce finalization. Numeric pins are against
torch references, independent of the variant wrappers."""

import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.epilogue.ops import ColVecReduce, RowVecReduce
from quack.gemm_config import GemmConfig, cta_tile_shape_m
from quack.epilogue.frontend import gemm_epilogue

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


CFG_CM1 = CFG.__class__(**{**CFG.__dict__, "cluster_m": 1})


def test_rowvec_reduce_finalized():
    A, B = _inputs()
    res = _colsum(A, B, config=CFG_CM1)
    ref_d = A.float() @ B.float()
    torch.testing.assert_close(res["colsum"], ref_d.sum(0), rtol=1e-3, atol=1e-1)


# ── Reduce-sink partials are sized by the DEVICE, never by the config tag ──
# This used to be a strict xfail blamed on the kernel ("the sink buffer is
# sized by the MMA tile but indexed per CTA tile, so the back half of M is
# dropped"). That diagnosis was wrong on both counts — there was no kernel
# defect and nothing was silent. GemmConfig.device_capacity is PROVENANCE
# ("tuned for this arch"), it defaults to 9, and a caller-supplied config is
# used verbatim; _alloc_sinks read that tag as if it were the running device:
#
#   cta_tile_shape_m(tile_m=128, cluster_m=2, 9)  -> 128  (allocation)
#   cta_tile_shape_m(tile_m=128, cluster_m=2, 10) ->  64  (validation, kernel)
#
# so on SM100 a cluster_m=2 call allocated ceil(M/128) partial rows where the
# kernel writes ceil(M/64), and EpiMod.gemm raised, loudly, before any launch:
#   ValueError: colsum must have shape (8, 768), got (4, 768)
# The kernel-side claim is pinned false by test_rowvec_reduce_cluster_m2_partials.
# CFG deliberately leaves device_capacity at its default so this call keeps
# exercising the stale-tag path on SM100.


def test_rowvec_reduce_cluster_m2():
    A, B = _inputs()
    res = _colsum(A, B, config=CFG)
    ref_d = A.float() @ B.float()
    torch.testing.assert_close(res["colsum"], ref_d.sum(0), rtol=1e-3, atol=1e-1)


def _mismatched_capacity(real_cap, tile_m, cluster_m):
    """A device_capacity tag that is wrong for the running device AND changes
    the per-CTA M tile, so any site still reading the tag mis-sizes the sink."""
    want = cta_tile_shape_m(tile_m, cluster_m, real_cap)
    return next((cap for cap in (9, 10) if cta_tile_shape_m(tile_m, cluster_m, cap) != want), None)


def test_sink_partials_sized_by_device_not_config_tag():
    """Arch-general form of the bug above: a config whose device_capacity tag
    disagrees with the running device must not change the sink buffer. Revert
    the fix and this raises ValueError from EpiMod.gemm on every arch (on
    SM100 the tag under-counts rows, on SM90/SM120 it over-counts)."""
    cap = get_device_capacity(torch.device("cuda"))[0]
    bogus = _mismatched_capacity(cap, CFG.tile_m, CFG.cluster_m)
    assert bogus is not None and bogus != cap
    cfg = CFG.__class__(**{**CFG.__dict__, "device_capacity": bogus})
    A, B = _inputs(m=391)  # ragged against both candidate per-CTA tiles
    res = _colsum(A, B, config=cfg)
    ref_d = A.float() @ B.float()
    torch.testing.assert_close(res["colsum"], ref_d.sum(0), rtol=1e-3, atol=1e-1)


def test_rowvec_reduce_cluster_m2_partials():
    """Pins the kernel half of the old (wrong) diagnosis: under cluster_M=2
    every raw partial row is exactly its per-CTA M tile's column sum, with a
    ragged tail. Checked per row, not just through the fold, so a dropped or
    duplicated M tile cannot cancel out inside the sum."""
    cap = get_device_capacity(torch.device("cuda"))[0]
    cta = cta_tile_shape_m(CFG.tile_m, CFG.cluster_m, cap)
    m, n = 5 * cta + 13, 768
    A, B = _inputs(m=m, n=n)
    ref_d = A.float() @ B.float()
    # caller-owned partials: NaN-filled, so an unwritten row is a failure
    buf = torch.full((-(-m // cta), n), float("nan"), device="cuda", dtype=torch.float32)
    _colsum(A, B, config=CFG, colsum=buf)
    assert not buf.isnan().any(), "kernel left reduce-sink partial rows unwritten"
    for t in range(buf.shape[0]):
        torch.testing.assert_close(
            buf[t], ref_d[t * cta : (t + 1) * cta].sum(0), rtol=1e-3, atol=1e-1
        )


def test_torch_op_sink_partials_sized_by_device(monkeypatch):
    """Same rule as test_sink_partials_sized_by_device_not_config_tag, for the
    torch-op route: quack::gemm_epi_f's graph-owned allocation
    (_alloc_outs_from_meta) and compile_call's caller-owned partials must both
    size from the device, not cfg["device_capacity"].

    That route is reachable only under torch.compiler.is_compiling(), and the
    two torch.compile tests that reach it are red on torch 2.11.0 for an
    unrelated Dynamo defect (obj.__dict__ traces to a GetAttrVariable) — so
    without this test those lines have no coverage on 2.11 at all. Driving the
    gate directly keeps the check on every torch version.

    is_compiling is armed for exactly one query: the op body asks again while
    executing, and a permanently-True gate makes it re-enter compile_call until
    the stack overflows.
    """
    armed = [False]

    def _once():
        if armed[0]:
            armed[0] = False
            return True
        return False

    monkeypatch.setattr(torch.compiler, "is_compiling", _once)
    cap = get_device_capacity(torch.device("cuda"))[0]
    bogus = _mismatched_capacity(cap, CFG.tile_m, CFG.cluster_m)
    assert bogus is not None and bogus != cap
    cfg = CFG.__class__(**{**CFG.__dict__, "device_capacity": bogus})
    m, n = 391, 768  # ragged M; n keeps B's leading stride 16B-aligned
    A, B = _inputs(m=m, n=n)
    ref = (A.float() @ B.float()).sum(0)

    armed[0] = True  # -> quack::gemm_epi_f -> _alloc_outs_from_meta
    res = _colsum(A, B, config=cfg)
    torch.testing.assert_close(res["colsum"], ref, rtol=1e-3, atol=1e-1)

    armed[0] = True  # out= makes it caller-owned -> compile_call's own alloc
    out = {"D": torch.empty(m, n, device="cuda", dtype=torch.bfloat16)}
    res = _colsum(A, B, config=cfg, out=out)
    torch.testing.assert_close(res["colsum"], ref, rtol=1e-3, atol=1e-1)


@gemm_epilogue(reduces={"bfly_sum": RowVecReduce("bfly_sum")})
def _bfly_sum(acc):
    return {"D": acc, "bfly_sum": acc}


@gemm_epilogue(reduces={"bfly_max": RowVecReduce("bfly_max", combine="max")})
def _bfly_max(acc):
    return {"D": acc, "bfly_max": acc}


def test_rowvec_butterfly_fallback(monkeypatch):
    """RowVecReduce has two intra-warp reductions and real layouts only ever
    compile one of them. Measured over 38 configs (bf16/f16 x tile_m 128/256 x
    tile_n 128/192/256 x cluster_m 1/2/4 x pingpong, SM90 and SM100): every one
    took the swap-shuffle path, because E and the M-lane stride are powers of 2
    in every layout the epilogue builds. The butterfly fallback at
    RowVecReduce.end_loop_stage is therefore live code that nothing executes.

    _mode_flat_stride -> None makes use_swap_shuffle False on those same real
    layouts, so the fallback compiles and is checked numerically. Without this,
    the first epilogue tiler to produce an odd E would be the first time the
    branch ever ran.

    The epilogue fns are private to this test so the jit cache can only ever
    hold the forced variant for this key (a shared fn could serve a cubin
    compiled on the swap path and the check would silently pass).
    """
    import quack.epilogue.ops as ops

    traced = {}
    real_swap = ops.swap_shuffle_reduce

    def _spy_swap(*a, **k):
        traced["swap"] = True
        return real_swap(*a, **k)

    monkeypatch.setattr(ops, "swap_shuffle_reduce", _spy_swap)
    monkeypatch.setattr(ops, "_mode_flat_stride", lambda layout, mode: None)

    cap = get_device_capacity(torch.device("cuda"))[0]
    cta = cta_tile_shape_m(CFG.tile_m, CFG.cluster_m, cap)
    m, n = 4 * cta + 13, 704
    A, B = _inputs(m=m, n=n)
    ref = A.float() @ B.float()
    torch.testing.assert_close(
        _bfly_sum(A, B, config=CFG)["bfly_sum"], ref.sum(0), rtol=1e-3, atol=1e-1
    )
    torch.testing.assert_close(
        _bfly_max(A, B, config=CFG)["bfly_max"], ref.amax(0), rtol=1e-3, atol=1e-2
    )
    # A warm jit cache serves the (already forced) cubin without tracing, so the
    # spy stays silent; it can only ever prove the swap path was NOT taken.
    assert not traced.get("swap"), "override failed: still compiled the swap-shuffle path"


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
    """tuned=True rides quack.gemm_runtime.autotune (sweep once per metadata class,
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
    from quack.cute_dsl_utils import get_device_capacity
    from quack.gemm import GemmDefaultSm100
    from quack.epilogue.frontend import epilogue_from_class

    # quack capability, not torch's: QUACK_ARCH overrides (e.g. the CI sm120
    # legs) dispatch non-SM100 classes on SM100 hardware.
    if get_device_capacity(torch.device("cuda"))[0] not in (10, 11):
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
