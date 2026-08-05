# Copyright (c) 2026, QuACK team.
# Split-K GEMM: correctness vs fp32 reference, run-to-run bitwise determinism, exact
# equivalence of split_k=1 with the baseline kernel, and serial/staged agreement.

import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm as quack_gemm
from quack.gemm_interface import SplitKMode, gemm, gemm_add, gemm_add_inplace, gemm_ref

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_device_capacity(torch.device("cuda"))[0] not in (9, 10, 11, 12),
    reason="split_k requires SM90/SM100/SM120",
)

DETERMINISM_RUNS = 3


def _assert_close_double_baseline(out, out_ref, out_pt, mult=2, atol=1e-4):
    assert (out.float() - out_ref).abs().max().item() < mult * (
        out_pt.float() - out_ref
    ).abs().max().item() + atol


def _make_inputs(m, n, k, L, dtype, seed=0):
    torch.manual_seed(seed)
    shape_a = (m, k) if L is None else (L, m, k)
    shape_b = (k, n) if L is None else (L, k, n)
    A = torch.randn(shape_a, dtype=dtype, device="cuda") / 4
    B = torch.randn(shape_b, dtype=dtype, device="cuda") / 4
    return A, B


@pytest.mark.parametrize("split_k_mode", list(SplitKMode), ids=lambda m: m.name.lower())
@pytest.mark.parametrize("split_k", [2, 5])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("L", [None, 3])
def test_gemm_split_k(L, out_dtype, split_k, split_k_mode):
    # Small M/N, large ragged K (65 k-tiles of 64 -> uneven splits for split_k=2 and 5)
    m, n, k = 192, 480, 4160
    A, B = _make_inputs(m, n, k, L, torch.bfloat16)
    out_ref = gemm_ref(A.float(), B.float())
    out_pt = (A.float() @ B.float()).to(out_dtype)
    out = gemm(A, B, out_dtype=out_dtype, tuned=False, split_k=split_k, split_k_mode=split_k_mode)
    # All modes accumulate raw partials in f32 and convert exactly once (the epilogue
    # runs only on the finalizing entity), so the baseline tolerance applies as-is.
    _assert_close_double_baseline(out, out_ref, out_pt, mult=2)
    if split_k_mode == SplitKMode.PARALLEL:
        return  # arrival-order reduction: not deterministic run to run by design
    # Run-to-run bitwise determinism with fresh output buffers
    runs = [
        gemm(A, B, out_dtype=out_dtype, tuned=False, split_k=split_k, split_k_mode=split_k_mode)
        for _ in range(DETERMINISM_RUNS)
    ]
    for r in runs:
        assert torch.equal(out, r), "split_k result is not bitwise deterministic"


@pytest.mark.parametrize(
    "persistent,dynamic_persistent", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("split_k_mode", list(SplitKMode), ids=lambda m: m.name.lower())
def test_gemm_split_k_schedulers(split_k_mode, persistent, dynamic_persistent):
    """Exercise NONE (non-persistent), STATIC, and CLC/DYNAMIC persistent schedulers."""
    device_capacity = get_device_capacity(torch.device("cuda"))[0]
    if not persistent and device_capacity in (10, 11):
        pytest.skip("SM100 GEMM is always persistent")
    m, n, k, L, split_k = 256, 512, 8192, 2, 4
    A, B = _make_inputs(m, n, k, L, torch.bfloat16)
    D = torch.empty((L, m, n), dtype=torch.bfloat16, device="cuda")
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device="cuda")
        if dynamic_persistent and device_capacity == 9
        else None
    )

    def run(out):
        quack_gemm(
            A,
            B.mT,  # quack.gemm takes B as (L, N, K)
            out,
            None,
            tile_count_semaphore,
            tile_M=128,
            tile_N=128,
            cluster_M=1,
            cluster_N=1,
            persistent=persistent,
            is_dynamic_persistent=dynamic_persistent,
            split_k=split_k,
            split_k_mode=split_k_mode,
        )
        if tile_count_semaphore is not None:
            tile_count_semaphore.zero_()
        return out

    run(D)
    out_ref = gemm_ref(A.float(), B.float())
    out_pt = A @ B
    _assert_close_double_baseline(D, out_ref, out_pt, mult=2)
    if split_k_mode == SplitKMode.PARALLEL:
        return  # arrival-order reduction: not deterministic run to run by design
    for _ in range(DETERMINISM_RUNS):
        D2 = run(torch.empty_like(D))
        assert torch.equal(D, D2), "split_k result is not bitwise deterministic"


@pytest.mark.parametrize("split_k_mode", list(SplitKMode), ids=lambda m: m.name.lower())
def test_gemm_split_k_bias_alpha(split_k_mode):
    """The full epilogue (alpha, bias) runs exactly once, on the finalizing entity
    (the last split in serial/parallel, the reduction kernel in staged)."""
    m, n, k, split_k = 192, 512, 6144, 4
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    bias = torch.randn(n, dtype=torch.float32, device="cuda")
    alpha = 0.5
    out_ref = alpha * (A.float() @ B.float()) + bias
    out_pt = (alpha * (A @ B).float() + bias).to(torch.bfloat16)
    out = gemm(
        A, B, bias=bias, alpha=alpha, tuned=False, split_k=split_k, split_k_mode=split_k_mode
    )
    _assert_close_double_baseline(out, out_ref, out_pt, mult=2)


@pytest.mark.parametrize("split_k_mode", list(SplitKMode), ids=lambda m: m.name.lower())
def test_gemm_add_split_k(split_k_mode):
    """beta * C is applied exactly once, by the finalizing entity, on the full sum."""
    m, n, k, split_k = 192, 512, 6144, 4
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    C = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    alpha, beta = 1.0, 0.5
    out_ref = alpha * (A.float() @ B.float()) + beta * C.float()
    out_pt = alpha * (A @ B) + beta * C
    out = gemm_add(
        A, B, C, alpha=alpha, beta=beta, tuned=False, split_k=split_k, split_k_mode=split_k_mode
    )
    _assert_close_double_baseline(out, out_ref, out_pt, mult=2)


@pytest.mark.parametrize("split_k_mode", list(SplitKMode), ids=lambda m: m.name.lower())
def test_gemm_add_inplace_split_k(split_k_mode):
    """add_to_output: the prior value of D is added exactly once — by the finalizer's
    TMA reduce-add store (serial/parallel, one extra D-dtype rounding) or in f32 by the
    reduction kernel (staged)."""
    m, n, k, split_k = 192, 512, 6144, 4
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    C0 = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    out_ref = C0.float() + A.float() @ B.float()
    out_pt = C0 + A @ B
    out = C0.clone()
    gemm_add_inplace(A, B, out, tuned=False, split_k=split_k, split_k_mode=split_k_mode)
    _assert_close_double_baseline(out, out_ref, out_pt, mult=3)
    if split_k_mode == SplitKMode.PARALLEL:
        return  # arrival-order reduction: not deterministic run to run by design
    out2 = C0.clone()
    gemm_add_inplace(A, B, out2, tuned=False, split_k=split_k, split_k_mode=split_k_mode)
    assert torch.equal(out, out2), "split_k result is not bitwise deterministic"


def test_gemm_split_k_staged_grid_not_inflated():
    """The scheduler must see the true L in staged mode (the workspace batch extent is
    L*split_k and the scheduler scales by split_k itself): the staged GEMM grid must
    equal the serial one (regression for split_k x phantom work units)."""
    m, n, k, L, split_k = 256, 512, 8192, 2, 4
    A, B = _make_inputs(m, n, k, L, torch.bfloat16)

    def max_kernel_grid_z(mode):
        return _max_gemm_grid_z(lambda: gemm(A, B, tuned=False, split_k=split_k, split_k_mode=mode))

    serial_z = max_kernel_grid_z(SplitKMode.SERIAL)
    staged_z = max_kernel_grid_z(SplitKMode.SEPARATE)
    if serial_z is None or staged_z is None:
        pytest.skip("profiler did not expose the QuACK GEMM grid")
    # The GEMM dominates grid z in both modes (the reduce kernel's z is just L)
    assert staged_z <= serial_z, f"staged GEMM grid z inflated: {staged_z} vs serial {serial_z}"


def _max_gemm_grid_z(fn):
    """Grid.z of the QuACK GEMM launched by fn().

    grid.z is NOT split_k directly: the default config runs a persistent (pingpong)
    scheduler, so grid.z == num_persistent_clusters == n_clusters * split_k when the
    problem fits in one wave. Callers must normalize by the split_k=1 launch to recover
    the split factor (z(split_k) / z(1) == split_k)."""
    import json
    import os
    import tempfile

    fn()  # warm compile
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        trace_path = f.name
    prof.export_chrome_trace(trace_path)
    try:
        with open(trace_path) as f:
            events = json.load(f)["traceEvents"]
    finally:
        os.unlink(trace_path)
    # Kineto can intermittently omit an individual CUDA launch under CI load. Do not
    # substitute an unrelated reference/assertion kernel's grid for the GEMM in that
    # case: report no observation and let the caller skip the profiler-only assertion.
    zs = [
        e["args"]["grid"][2]
        for e in events
        if e.get("cat") == "kernel"
        and "grid" in e.get("args", {})
        and "quack" in e.get("name", "").lower()
        and "gemm" in e.get("name", "").lower()
    ]
    return max(zs) if zs else None


def test_gemm_split_k_config_autotuner_surface():
    """split_k is a GemmConfig field on the autotuner surface: split_k=None defers to
    config.split_k, an explicit split_k overrides the config, and the prune hook expands
    split-k variants for occupancy-starved shapes (and only when split_k is None)."""
    from dataclasses import replace

    from quack.autotuner import AutotuneConfig
    from quack.gemm_interface import default_config, gemm_tuned, prune_invalid_gemm_configs

    m, n, k = 256, 512, 16384
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    out_ref = gemm_ref(A.float(), B.float())
    cfg = replace(default_config(A.device), split_k=4)

    def run(split_k):
        out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        resolved_config, resolved_split_k, _, dispatch_plan = gemm_tuned.fn(
            A, B, out, config=cfg, split_k=split_k
        )
        _assert_close_double_baseline(out, out_ref, A @ B, mult=2)
        assert resolved_config == cfg
        assert dispatch_plan.split_k == resolved_split_k
        return resolved_split_k

    assert run(split_k=1) == 1
    assert run(split_k=None) == 4, "config.split_k not honored"
    assert run(split_k=2) == 2, "explicit split_k must override config"

    # Prune-hook expansion: split_k=None on a starved shape adds split-k variants...
    base = [AutotuneConfig(config=default_config(A.device))]
    expanded = prune_invalid_gemm_configs(base, {"A": A, "B": B, "split_k": None})
    assert any(c.kwargs["config"].split_k > 1 for c in expanded), "no split-k variants expanded"
    # ...but a forced factor (or an unexposed knob) expands nothing.
    forced = prune_invalid_gemm_configs(base, {"A": A, "B": B, "split_k": 2})
    assert all(c.kwargs["config"].split_k == 1 for c in forced)
    no_knob = prune_invalid_gemm_configs(base, {"A": A, "B": B})
    assert all(c.kwargs["config"].split_k == 1 for c in no_knob)


def test_gemm_split_k_one_matches_baseline():
    """split_k=1 must compile to the exact baseline kernel: bitwise-identical output."""
    m, n, k = 256, 512, 4096
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    out_base = gemm(A, B, tuned=False)
    for mode in SplitKMode:
        out1 = gemm(A, B, tuned=False, split_k=1, split_k_mode=mode)
        assert torch.equal(out_base, out1)


def test_gemm_split_k_serial_staged_f32_bitwise_equal():
    """Both modes sum raw f32 partials in ascending split order (serial: turnstile-ordered
    red.add.f32 into the workspace + the finalizer's register add; staged: the reduction
    kernel's register adds) and run the identical shared epilogue once, so f32 outputs
    must agree bitwise. (red.add.f32 flushes subnormals where FADD does not — randn
    inputs keep everything normal.)"""
    m, n, k, split_k = 192, 480, 8192, 4
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    out_serial = gemm(
        A, B, out_dtype=torch.float32, tuned=False, split_k=split_k, split_k_mode=SplitKMode.SERIAL
    )
    out_staged = gemm(
        A,
        B,
        out_dtype=torch.float32,
        tuned=False,
        split_k=split_k,
        split_k_mode=SplitKMode.SEPARATE,
    )
    assert torch.equal(out_serial, out_staged)


def test_gemm_split_k_cluster_rounded_tiles():
    """Tile count not divisible by the cluster shape: boundary clusters contain CTAs
    whose tile is fully out of bounds, but which still run the serial lock protocol.
    Lock indexing must use cluster-rounded tile counts or CTAs alias locks and deadlock
    (regression for the under-sized/mis-indexed semaphore array)."""
    # default SM100 config: mma tiler (256, 256), cluster (2, 1) -> CTA tile m = 128.
    # m = 320 gives 3 CTA tiles in M, rounded up to 4 by the cluster.
    m, n, k, split_k = 320, 512, 8192, 4
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    out_ref = gemm_ref(A.float(), B.float())
    out_pt = A @ B
    for mode in SplitKMode:
        out = gemm(A, B, tuned=False, split_k=split_k, split_k_mode=mode)
        _assert_close_double_baseline(out, out_ref, out_pt, mult=2)
        if mode != SplitKMode.PARALLEL:  # arrival-order reduction is not deterministic
            out2 = gemm(A, B, tuned=False, split_k=split_k, split_k_mode=mode)
            assert torch.equal(out, out2)


def test_gemm_split_k_ragged_n_strided_out():
    """N a multiple of 8 (the bf16 16-byte contract) but not of the tile width, with the
    output a slice of a larger buffer: boundary tiles/vectors must not clobber adjacent
    user data (regression for per-vector predication in split_k_reduce)."""
    m, n_alloc, n, k, split_k = 64, 128, 104, 2048, 2
    torch.manual_seed(0)
    A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / 4
    # B's non-contiguous stride must satisfy the 16-byte contract: slice a padded buffer
    B = (torch.randn(k, n_alloc, dtype=torch.bfloat16, device="cuda") / 4)[:, :n]
    out_ref = gemm_ref(A.float(), B.float())
    for mode in SplitKMode:
        big = torch.full((m, n_alloc), 7.0, dtype=torch.bfloat16, device="cuda")
        out = big[:, :n]
        gemm(A, B, out=out, tuned=False, split_k=split_k, split_k_mode=mode)
        assert (big[:, n:] == 7.0).all(), f"{mode} split-K clobbered memory next to out"
        _assert_close_double_baseline(out, out_ref, A @ B, mult=2)


def test_gemm_split_k_larger_than_k_tiles():
    """split_k > number of k-tiles: trailing splits are empty but must still
    participate in the reduction protocol."""
    m, n, k, split_k = 192, 256, 128, 6  # at most 2 k-tiles
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    out_ref = gemm_ref(A.float(), B.float())
    out_pt = A @ B
    for mode in SplitKMode:
        out = gemm(A, B, tuned=False, split_k=split_k, split_k_mode=mode)
        _assert_close_double_baseline(out, out_ref, out_pt, mult=2)


def test_gemm_split_k_m_major_out():
    """m-major output exercises the transposed staged workspace path."""
    m, n, k, split_k = 192, 512, 6144, 4
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    out_ref = gemm_ref(A.float(), B.float())
    out_pt = A @ B
    for mode in SplitKMode:
        out = torch.empty((n, m), dtype=torch.bfloat16, device="cuda").mT
        gemm(A, B, out=out, tuned=False, split_k=split_k, split_k_mode=mode)
        _assert_close_double_baseline(out, out_ref, out_pt, mult=2)


def test_gemm_split_k_rejects_unsupported():
    m, n, k = 128, 256, 1024
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    cu_seqlens_m = torch.tensor([0, m], dtype=torch.int32, device="cuda")
    with pytest.raises(Exception, match="split_k"):
        gemm(A, B, cu_seqlens_m=cu_seqlens_m, tuned=False, split_k=2)
    with pytest.raises(Exception, match="split_k"):
        gemm(A, B, tuned=False, split_k=2, split_k_mode="bogus")
    from quack.gemm_interface import gemm_act, gemm_symmetric

    with pytest.raises(NotImplementedError, match="split_k"):
        gemm_act(A, B, activation="relu", split_k=2)
    with pytest.raises(NotImplementedError, match="split_k"):
        gemm_symmetric(A, A.mT.contiguous().mT, split_k=2)


@pytest.mark.parametrize("split_k_mode", list(SplitKMode), ids=lambda m: m.name.lower())
def test_gemm_add_inplace_beta_split_k(split_k_mode):
    """C aliasing the output (gemm_add_inplace with beta != 1) is legal in every mode:
    only the finalizing entity reads C and D is written exactly once, after the full
    reduction (this was rejected for parallel mode under the old reduce-into-D design)."""
    m, n, k = 192, 512, 6144
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    C0 = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    out_ref = 0.5 * C0.float() + A.float() @ B.float()
    out_pt = 0.5 * C0 + A @ B
    out = C0.clone()
    gemm_add_inplace(A, B, out, beta=0.5, tuned=False, split_k=4, split_k_mode=split_k_mode)
    _assert_close_double_baseline(out, out_ref, out_pt, mult=2)


@pytest.mark.parametrize("split_k_mode", list(SplitKMode), ids=lambda m: m.name.lower())
def test_gemm_split_k_full_linear_epilogue(split_k_mode):
    """alpha, beta*C, rowvec AND colvec bias together, applied once on the full sum —
    exercises every operand path of the staged reduction kernel's shared epilogue and
    of the serial/parallel finalizer."""
    m, n, k, L, split_k = 256, 512, 8192, 2, 4
    A, B = _make_inputs(m, n, k, L, torch.bfloat16)
    C = torch.randn(L, m, n, dtype=torch.bfloat16, device="cuda")
    rowvec = torch.randn(L, n, dtype=torch.float32, device="cuda")
    colvec = torch.randn(L, m, dtype=torch.float32, device="cuda")
    alpha, beta = 0.5, 2.0
    acc = gemm_ref(A.float(), B.float())
    out_ref = alpha * acc + beta * C.float() + rowvec[:, None, :] + colvec[:, :, None]
    D_base = torch.empty(L, m, n, dtype=torch.bfloat16, device="cuda")

    def run(out, **kw):
        quack_gemm(
            A,
            B.mT,  # quack.gemm takes B as (L, N, K)
            out,
            C,
            None,
            tile_M=128,
            tile_N=128,
            cluster_M=1,
            cluster_N=1,
            persistent=True,
            alpha=alpha,
            beta=beta,
            rowvec_bias=rowvec,
            colvec_bias=colvec,
            **kw,
        )
        return out

    run(D_base)
    D = run(torch.empty_like(D_base), split_k=split_k, split_k_mode=split_k_mode)
    _assert_close_double_baseline(D, out_ref, D_base, mult=2)
    if split_k_mode != SplitKMode.PARALLEL:
        D2 = run(torch.empty_like(D_base), split_k=split_k, split_k_mode=split_k_mode)
        assert torch.equal(D, D2), "split_k result is not bitwise deterministic"


@pytest.mark.parametrize("split_k_mode", list(SplitKMode), ids=lambda m: m.name.lower())
def test_gemm_split_k_mixed_c_majorness(split_k_mode):
    """C m-major with an n-major output: legal in the GEMM epilogue (c_major is tracked
    independently); the staged reduce kernel materializes a contiguous C view
    (regression: the staged path used to raise a TVM-FFI stride mismatch)."""
    m, n, k = 192, 512, 6144
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    C = torch.randn(n, m, dtype=torch.bfloat16, device="cuda").mT  # m-major
    out_ref = A.float() @ B.float() + 0.5 * C.float()
    out_pt = A @ B + 0.5 * C
    out = gemm_add(A, B, C, beta=0.5, tuned=False, split_k=4, split_k_mode=split_k_mode)
    _assert_close_double_baseline(out, out_ref, out_pt, mult=2)


@pytest.mark.parametrize("split_k_mode", list(SplitKMode), ids=lambda m: m.name.lower())
def test_gemm_split_k_m_major_out_with_epilogue(split_k_mode):
    """m-major output WITH rowvec+colvec+C: the staged host view transposes m/n and
    must SWAP the vector roles (a swapped-but-not-transposed bug passes the plain
    m-major test)."""
    m, n, k, L = 192, 512, 6144, 2
    A, B = _make_inputs(m, n, k, L, torch.bfloat16)
    C = torch.randn(L, m, n, dtype=torch.bfloat16, device="cuda")
    rowvec = torch.randn(L, n, dtype=torch.float32, device="cuda")
    colvec = torch.randn(L, m, dtype=torch.float32, device="cuda")
    acc = gemm_ref(A.float(), B.float())
    out_ref = 0.5 * acc + 2.0 * C.float() + rowvec[:, None, :] + colvec[:, :, None]
    base = torch.empty(L, n, m, dtype=torch.bfloat16, device="cuda").mT

    def run(out, **kw):
        quack_gemm(
            A,
            B.mT,
            out,
            C,
            None,
            tile_M=128,
            tile_N=128,
            cluster_M=1,
            cluster_N=1,
            persistent=True,
            alpha=0.5,
            beta=2.0,
            rowvec_bias=rowvec,
            colvec_bias=colvec,
            **kw,
        )
        return out

    run(base)
    out = run(
        torch.empty(L, n, m, dtype=torch.bfloat16, device="cuda").mT,
        split_k=4,
        split_k_mode=split_k_mode,
    )
    _assert_close_double_baseline(out, out_ref, base, mult=2)


@pytest.mark.parametrize("split_k_mode", list(SplitKMode), ids=lambda m: m.name.lower())
def test_gemm_split_k_tensor_alpha_beta(split_k_mode):
    """Tensor-valued (pointer-mode) alpha/beta through both finalizers."""
    m, n, k = 192, 512, 6144
    A, B = _make_inputs(m, n, k, None, torch.bfloat16)
    C = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    alpha = torch.tensor(0.5, dtype=torch.float32, device="cuda")
    beta = torch.tensor(2.0, dtype=torch.float32, device="cuda")
    out_ref = 0.5 * (A.float() @ B.float()) + 2.0 * C.float()
    out_pt = 0.5 * (A @ B) + 2.0 * C
    out = gemm_add(
        A, B, C, alpha=alpha, beta=beta, tuned=False, split_k=4, split_k_mode=split_k_mode
    )
    _assert_close_double_baseline(out, out_ref, out_pt, mult=2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (10, 11),
    reason="tf32 (float32 inputs -> TFloat32 MMA) is SM100/SM110 only",
)
@pytest.mark.parametrize("split_k_mode", list(SplitKMode), ids=lambda m: m.name.lower())
@pytest.mark.parametrize("split_k", [2, 5])
def test_gemm_split_k_tf32(split_k, split_k_mode):
    """tf32 GEMM (float32 inputs, TFloat32 MMA): split-K must match the plain tf32 kernel
    at the baseline tolerance and stay deterministic for serial/staged. The partials
    workspace is f32 regardless of input dtype, so tf32 needs no special handling."""
    m, n, k, L = 256, 512, 8192, 2
    A, B = _make_inputs(m, n, k, L, torch.float32)
    out_ref = gemm_ref(A.float(), B.float())  # true f32 matmul
    # The plain (split_k=1) tf32 kernel is the same-precision baseline.
    out_pt = gemm(A, B, out_dtype=torch.float32, tuned=False)
    out = gemm(
        A, B, out_dtype=torch.float32, tuned=False, split_k=split_k, split_k_mode=split_k_mode
    )
    _assert_close_double_baseline(out, out_ref, out_pt, mult=2)
    if split_k_mode == SplitKMode.PARALLEL:
        return  # arrival-order reduction: not deterministic run to run by design
    for _ in range(DETERMINISM_RUNS):
        out2 = gemm(
            A, B, out_dtype=torch.float32, tuned=False, split_k=split_k, split_k_mode=split_k_mode
        )
        assert torch.equal(out, out2), "split_k tf32 result is not bitwise deterministic"


@pytest.mark.parametrize("split_k_mode", list(SplitKMode), ids=lambda m: m.name.lower())
@pytest.mark.parametrize("split_k", [2, 5])
@pytest.mark.parametrize(
    "in_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3fn", "e5m2"]
)
def test_gemm_split_k_fp8(in_dtype, split_k, split_k_mode):
    """fp8 (e4m3fn / e5m2) GEMM with bf16 output: split-K matches the plain fp8 kernel at
    the baseline tolerance and stays deterministic for serial/staged. fp8 needs k-major
    A and B; the f32 partials workspace makes the finalizer dtype-agnostic."""
    if get_device_capacity(torch.device("cuda"))[0] == 12:
        pytest.skip("SM120 GEMM is warp-level MmaF16BF16Op: no fp8 support (fp16/bf16 only)")
    m, n, k, L = 256, 512, 8192, 2
    torch.manual_seed(0)
    # fp8 requires k-major A (m, k) and B (k, n): k contiguous in both.
    A = (torch.randn(L, m, k, device="cuda") / 4).to(in_dtype)
    B = (torch.randn(L, n, k, device="cuda") / 4).to(in_dtype).mT  # (L, k, n), k-major
    out_ref = gemm_ref(A.float(), B.float())
    out_pt = gemm(A, B, out_dtype=torch.bfloat16, tuned=False)  # plain fp8 baseline
    out = gemm(
        A, B, out_dtype=torch.bfloat16, tuned=False, split_k=split_k, split_k_mode=split_k_mode
    )
    _assert_close_double_baseline(out, out_ref, out_pt, mult=2)
    if split_k_mode == SplitKMode.PARALLEL:
        return  # arrival-order reduction: not deterministic run to run by design
    for _ in range(DETERMINISM_RUNS):
        out2 = gemm(
            A, B, out_dtype=torch.bfloat16, tuned=False, split_k=split_k, split_k_mode=split_k_mode
        )
        assert torch.equal(out, out2), "split_k fp8 result is not bitwise deterministic"


def test_gemm_split_k_odd_epi_subtile_count():
    """A tile shape whose epi-subtile count is not a multiple of the epi smem stage
    count: skipped (non-finalizing) tiles jump the smem buffer rotation by a
    non-multiple of the stage count, so the finalizer must drain its TMA stores before
    the next tile reuses an arbitrary buffer (regression for the rotation/drain
    desync — silent D corruption without the producer_tail in epilogue_split_k)."""
    m, n, k, L, split_k = 512, 448, 8192, 2, 4
    A, B = _make_inputs(m, n, k, L, torch.bfloat16)
    out_ref = gemm_ref(A.float(), B.float())
    D_base = torch.empty(L, m, n, dtype=torch.bfloat16, device="cuda")

    def run(out, **kw):
        quack_gemm(
            A,
            B.mT,
            out,
            None,
            None,
            tile_M=128,
            tile_N=224,
            cluster_M=1,
            cluster_N=1,
            persistent=True,
            **kw,
        )
        return out

    run(D_base)
    for mode in (SplitKMode.SERIAL, SplitKMode.PARALLEL):
        D = run(torch.empty_like(D_base), split_k=split_k, split_k_mode=mode)
        _assert_close_double_baseline(D, out_ref, D_base, mult=2)
    D = run(torch.empty_like(D_base), split_k=split_k, split_k_mode=SplitKMode.SERIAL)
    for _ in range(DETERMINISM_RUNS):
        D2 = run(torch.empty_like(D_base), split_k=split_k, split_k_mode=SplitKMode.SERIAL)
        assert torch.equal(D, D2), "split_k result is not bitwise deterministic"
