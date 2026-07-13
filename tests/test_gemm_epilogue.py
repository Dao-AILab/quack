"""Numeric tests for the @gemm_epilogue fn-authoring frontend.

Each minted epilogue is checked against a float32 torch reference, and the
norm_gelu case additionally against the hand-written GemmNormAct kernel — the
fn frontend must produce the same math the mixin produces.
"""

import pytest
from quack.epi_ops import ColVecReduce, OnlineLSEReduce, Scalar, TileLoad
from quack.gemm_epilogue import gemm_epilogue, pack, unpack
from quack.gemm_host import resolve_gemm_class
import math
import pickle
import torch
import cutlass.cute as cute
from cutlass import Float32, Int32
from quack.gemm_act import gemm_act
from quack.gemm_dact import gemm_dact
from quack.gemm_norm_act import gemm_norm_act_fn
from quack.gemm_sq_reduce import gemm_sq_reduce
from quack.epilogue.head_rmsnorm import HeadRMSNormStats  # noqa: F401
from quack.epilogue.rotary import rope_table_epi, rope_table_ldg_epi
from quack.epilogues import (
    amax_epi,
    dgelu_mod,
    dswiglu_mod,
    dswiglu_norm_mod,
    linear_epi,
    lse_partial_epi,
    norm_gelu,
    norm_swiglu_mod,
    qk_rope_epi,
    qk_rope_ldg_epi,
    qknorm_epi,
    relu_mod,
    relu_sq_mod,
    residual_epi,
    rms_bwd_partial_epi,
    rms_fused,
    rms_partial_epi,
    rope_epi,
    rstd_swiglu_epi,
    scaled_residual,
    swiglu_mod,
)


def _rel_check(out, ref, name, tol=2e-2):
    err = (out.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err < tol * scale + 1e-2, f"{name}: err {err} vs scale {scale}"


def _cache_config_epi_fn(acc):
    return {"D": acc, "stat": acc}


_cache_add_mod = gemm_epilogue(reduces={"stat": ColVecReduce("stat")})(_cache_config_epi_fn)
_cache_max_mod = gemm_epilogue(reduces={"stat": ColVecReduce("stat", combine="max")})(
    _cache_config_epi_fn
)


def test_epi_mod_semantic_cache_key_and_resolver():
    """Static op config changes identity; a pickled class recipe remints locally."""
    from quack.rounding import RoundingMode

    assert _cache_add_mod.semantic_digest != _cache_max_mod.semantic_digest
    mint_key = ((), 10, False, False, (), RoundingMode.RN)
    ref = pickle.loads(pickle.dumps(_cache_add_mod._class_ref(mint_key)))
    cls = resolve_gemm_class(ref)
    assert cls._epi_mod_class_semantic_key == (_cache_add_mod.semantic_digest, mint_key)


def test_epi_mod_local_payload_identity_and_consumption():
    cloudpickle = pytest.importorskip("cloudpickle")  # noqa: F841
    from quack.cache.async_compile import PoolPayload
    from quack.gemm_host import _LOCAL_EPI_MODS, install_epi_mod_payload

    def build():
        @gemm_epilogue()
        def local_epi(acc):
            return {"D": acc}

        return local_epi

    mod = build()
    mint_key = ((), 10, False, False, ())
    ref = mod._class_ref(mint_key)
    payload = ref.__quack_pool_payload__()
    assert isinstance(payload, PoolPayload)
    assert payload.identity == mod.semantic_digest

    # Model the separate worker: it starts without the submitter's registry.
    _LOCAL_EPI_MODS.pop(mod.semantic_digest)
    with pytest.raises(ValueError, match="digest mismatch"):
        install_epi_mod_payload("wrong-digest", payload.data)
    assert mod.semantic_digest not in _LOCAL_EPI_MODS

    install_epi_mod_payload(payload.identity, payload.data)
    assert mod.semantic_digest in _LOCAL_EPI_MODS
    resolve_gemm_class(ref)
    assert mod.semantic_digest not in _LOCAL_EPI_MODS


def test_epi_scalar_fixed_abi():
    f32 = Scalar("scale")
    assert f32.host_arg_key(1.5) == ("immediate", Float32)
    x = torch.ones(1, device="cuda", dtype=torch.float32)
    assert f32.host_arg_key(x) == ("pointer", Float32)
    with pytest.raises(TypeError, match="must have dtype"):
        f32.host_arg_key(x.half())
    with pytest.raises(ValueError, match="exactly one"):
        f32.host_arg_key(torch.ones(2, device="cuda", dtype=torch.float32))
    seed = Scalar("seed", dtype=Int32)
    assert seed.host_arg_key(torch.ones(1, device="cuda", dtype=torch.int32)) == (
        "pointer",
        Int32,
    )


def test_epi_mode_validates_shapes_without_inference():
    A = torch.empty((1, 8, 8), device="cuda", dtype=torch.bfloat16)
    B = torch.empty((1, 16, 8), device="cuda", dtype=torch.bfloat16)
    D = torch.empty((1, 8, 16), device="cuda", dtype=torch.bfloat16)
    half_aux = torch.empty((1, 8, 8), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must have shape"):
        relu_mod.gemm(
            A,
            B,
            D,
            epi_args=dict(postact=half_aux),
            tile_M=8,
            tile_N=8,
            cluster_M=1,
            cluster_N=1,
        )

    B_odd = torch.empty((1, 15, 8), device="cuda", dtype=torch.bfloat16)
    odd_aux = torch.empty((1, 8, 7), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="even GEMM N"):
        swiglu_mod.gemm(
            A,
            B_odd,
            None,
            epi_args=dict(postact=odd_aux),
            tile_M=8,
            tile_N=8,
            cluster_M=1,
            cluster_N=1,
        )

    semaphore = torch.zeros(1, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="requires is_dynamic_persistent"):
        relu_mod.gemm(
            A,
            B,
            D,
            epi_args=dict(postact=torch.empty_like(D)),
            tile_M=8,
            tile_N=8,
            cluster_M=1,
            cluster_N=1,
            tile_count_semaphore=semaphore,
        )


@pytest.mark.parametrize("batched", [True, False])
def test_epi_mod_norm_gelu(batched):
    device = "cuda"
    torch.random.manual_seed(0)
    l, m, n, k = 2, 512, 1024, 736
    shape_a = (l, m, k) if batched else (m, k)
    shape_b = (l, n, k) if batched else (n, k)
    shape_d = (l, m, n) if batched else (m, n)
    A = torch.randn(shape_a, device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn(shape_b, device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    D = torch.empty(shape_d, device=device, dtype=torch.bfloat16)
    postact = torch.empty(shape_d, device=device, dtype=torch.bfloat16)
    # Dense calls take (l, dim) broadcast vectors; l == 1 for the unbatched case.
    rstd = torch.rand((l if batched else 1, m), device=device, dtype=torch.float32) + 0.5
    weight = torch.randn((l if batched else 1, n), device=device, dtype=torch.float32)

    norm_gelu.gemm(
        A,
        B,
        D,
        epi_args=dict(rstd=rstd, weight=weight, postact=postact),
        tile_M=128,
        tile_N=256,
        cluster_M=2,
        cluster_N=1,
    )

    x_ref = torch.einsum("...mk,...nk->...mn", A.float(), B.float())
    rstd_b = rstd.unsqueeze(-1) if batched else rstd[0].unsqueeze(-1)
    weight_b = weight.unsqueeze(-2) if batched else weight[0]
    x_ref = x_ref * rstd_b * weight_b
    _rel_check(D, x_ref, "D")
    _rel_check(postact, torch.nn.functional.gelu(x_ref, approximate="tanh"), "postact")

    # Cross-check against the hand-written GemmNormAct mixin: same math, same
    # config — outputs must agree to the same tolerance as vs the reference.
    if batched:
        D2 = torch.empty_like(D)
        postact2 = torch.empty_like(postact)
        gemm_norm_act_fn(
            A,
            B,
            D2,
            None,
            postact2,
            None,
            "gelu_tanh_approx",
            128,
            256,
            2,
            1,
            rowvec=weight,
            colvec=rstd,
        )
        _rel_check(D, D2.float(), "D vs handwritten", tol=1e-3)
        _rel_check(postact, postact2.float(), "postact vs handwritten", tol=1e-3)


# Module-level probe used ONLY by test_epi_mod_async_compile: its digest (and
# so its jit-cache sha) must not collide with any other test's keys, so the
# test can assert its compiles went through the pool it is watching.
@gemm_epilogue()
def async_probe_epi(acc, alpha):
    return {"D": acc * alpha}


def test_epi_mod_async_compile(tmp_path, monkeypatch):
    """Async-compile pool end-to-end for minted epilogue kernels: a
    module-anchored mod resolves by import in the worker, a factory-local mod
    ships by value (cloudpickle payload side-channel). A fresh CACHE_DIR
    forces cold misses through the pool; the test fails if the pool was
    bypassed (no submission) or a worker failed (in-process fallback warns)."""
    import time
    import warnings

    import quack.cache as cache_state
    from quack.cache import async_compile

    pytest.importorskip("cloudpickle")
    device = "cuda"
    torch.random.manual_seed(25)

    def build(shift):
        @gemm_epilogue()
        def local_probe(acc, alpha):
            return {"D": acc * alpha + shift}

        return local_probe

    l, m, n, k = 1, 256, 384, 512
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    ref = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    cases = [
        ("by-import", async_probe_epi, dict(alpha=2.0), 2.0 * ref),
        ("payload", build(0.25), dict(alpha=3.0), 3.0 * ref + 0.25),
    ]

    monkeypatch.setattr(cache_state, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_state, "CACHE_ENABLED", True)
    with async_compile.pool_scope() as pool:
        submitted_before = pool.n_submitted
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for label, mod, eargs, expected in cases:
                D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
                deadline = time.monotonic() + 180
                while True:
                    try:
                        mod.gemm(
                            A,
                            B,
                            D,
                            epi_args=eargs,
                            tile_M=128,
                            tile_N=128,
                            cluster_M=1,
                            cluster_N=1,
                        )
                        break
                    except async_compile.CompilePending:
                        assert time.monotonic() < deadline, f"{label}: compile never completed"
                        time.sleep(0.2)
                _rel_check(D, expected, label)
        fallbacks = [str(w.message) for w in caught if "async compile failed" in str(w.message)]
        assert not fallbacks, f"pool worker failed, compiled in-process instead: {fallbacks}"
        assert pool.n_submitted - submitted_before == 2, (
            f"expected both epilogue compiles to go through the pool, "
            f"got {pool.n_submitted - submitted_before} submissions"
        )


def test_epi_mod_factory_local():
    """An EpiMod never bound to a module global (factory-local, closure-
    carrying) must still plan/compile: it gets an ``epi_mod_local`` ref
    resolved through the process registry, and under --async-compile the
    pool ships it by value (cloudpickle payload side-channel)."""
    device = "cuda"
    torch.random.manual_seed(24)

    def build(beta):
        @gemm_epilogue()
        def scale_shift(acc, alpha):
            return {"D": acc * alpha + beta}

        return scale_shift

    l, m, n, k = 1, 384, 512, 512
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    build(1.5).gemm(
        A, B, D, epi_args=dict(alpha=2.0), tile_M=128, tile_N=256, cluster_M=1, cluster_N=1
    )
    ref = 2.0 * torch.einsum("lmk,lnk->lmn", A.float(), B.float()) + 1.5
    _rel_check(D, ref, "D")
    # Same factory, different closure value: distinct digest, distinct kernel.
    D2 = torch.empty_like(D)
    build(-4.0).gemm(
        A, B, D2, epi_args=dict(alpha=2.0), tile_M=128, tile_N=256, cluster_M=1, cluster_N=1
    )
    _rel_check(D2, ref - 5.5, "D (different closure)")


@pytest.mark.parametrize("alpha", [0.5, 2.0])
def test_epi_mod_scalar_and_c(alpha):
    device = "cuda"
    torch.random.manual_seed(1)
    l, m, n, k = 2, 512, 768, 512
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    C = torch.randn((l, m, n), device=device, dtype=torch.bfloat16)
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)

    scaled_residual.gemm(
        A,
        B,
        D,
        C,
        epi_args=dict(alpha=alpha),
        tile_M=128,
        tile_N=192,
        cluster_M=1,
        cluster_N=1,
    )

    ref = alpha * torch.einsum("lmk,lnk->lmn", A.float(), B.float()) + C.float()
    _rel_check(D, ref, "D")


def test_epi_mod_linear():
    """Default linear epilogue: alpha*acc + beta*C + rowvec + colvec."""
    device = "cuda"
    torch.random.manual_seed(2)
    l, m, n, k = 2, 384, 512, 736
    alpha, beta = 1.5, 0.5
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    C = torch.randn((l, m, n), device=device, dtype=torch.bfloat16)
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    bias_n = torch.randn((l, n), device=device, dtype=torch.float32)
    bias_m = torch.randn((l, m), device=device, dtype=torch.float32)

    linear_epi.gemm(
        A,
        B,
        D,
        C,
        epi_args=dict(alpha=alpha, beta=beta, bias_n=bias_n, bias_m=bias_m),
        tile_M=128,
        tile_N=192,
        cluster_M=1,
        cluster_N=1,
    )

    ref = alpha * torch.einsum("lmk,lnk->lmn", A.float(), B.float()) + beta * C.float()
    ref = ref + bias_n.unsqueeze(-2) + bias_m.unsqueeze(-1)
    _rel_check(D, ref, "D")


def test_epi_mod_act_factory():
    """Two mods minted from one factory body must not share cache identity."""
    device = "cuda"
    torch.random.manual_seed(3)
    l, m, n, k = 1, 512, 768, 512
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    assert relu_mod._ident != relu_sq_mod._ident, "closure salt failed: idents collide"
    ref = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    for mod, act_ref in ((relu_mod, torch.relu), (relu_sq_mod, lambda x: torch.relu(x) ** 2)):
        D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
        postact = torch.empty_like(D)
        mod.gemm(
            A,
            B,
            D,
            epi_args=dict(postact=postact),
            tile_M=128,
            tile_N=192,
            cluster_M=1,
            cluster_N=1,
        )
        _rel_check(D, ref, "D")
        _rel_check(postact, act_ref(ref), "postact")


def test_epi_mod_dact():
    """GemmDAct as a mod, cross-checked against the hand-written kernel."""
    device = "cuda"
    torch.random.manual_seed(4)
    l, m, n, k = 2, 512, 1024, 736
    dout = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    W = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    preact = torch.randn((l, m, n), device=device, dtype=torch.bfloat16)
    dx = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    postact = torch.empty_like(dx)

    dgelu_mod.gemm(
        dout,
        W,
        dx,
        preact,
        epi_args=dict(postact=postact),
        tile_M=128,
        tile_N=256,
        cluster_M=2,
        cluster_N=1,
    )

    x = preact.float()
    g = torch.einsum("lmk,lnk->lmn", dout.float(), W.float())
    tanh_out = torch.nn.functional.gelu(x, approximate="tanh")
    xg = x.detach().requires_grad_()
    torch.nn.functional.gelu(xg, approximate="tanh").backward(g)
    _rel_check(dx, xg.grad, "dx")
    _rel_check(postact, tanh_out, "postact")

    dx2 = torch.empty_like(dx)
    postact2 = torch.empty_like(postact)
    gemm_dact(
        dout,
        W,
        dx2,
        preact,
        postact2,
        None,
        "gelu_tanh_approx",
        128,
        256,
        2,
        1,
        pingpong=False,
    )
    _rel_check(dx, dx2.float(), "dx vs handwritten", tol=1e-3)
    _rel_check(postact, postact2.float(), "postact vs handwritten", tol=1e-3)


@pytest.mark.parametrize("tile_N", [192, 256])
def test_epi_mod_rms_fused(tile_N):
    """GemmSqReduce as a mod (reduce output + aux + rowvec), cross-checked
    against the hand-written kernel."""
    device = "cuda"
    torch.random.manual_seed(5)
    l, m, n, k = 2, 512, 1536, 736
    tile_M, cluster_M, cluster_N = 128, 1, 1
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    weight = torch.randn((l, n), device=device, dtype=torch.float32)
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    premult = torch.empty_like(D)
    n_tiles = (n + tile_N - 1) // tile_N
    sqsum = torch.empty((l, m, n_tiles), device=device, dtype=torch.float32)

    rms_fused.gemm(
        A,
        B,
        D,
        epi_args=dict(weight=weight, premult=premult, sqsum=sqsum),
        tile_M=tile_M,
        tile_N=tile_N,
        cluster_M=cluster_M,
        cluster_N=cluster_N,
    )

    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    pad = n_tiles * tile_N - n
    sq = torch.nn.functional.pad(x**2, (0, pad)) if pad else x**2
    sq_ref = sq.unflatten(-1, (n_tiles, tile_N)).sum(dim=-1)
    _rel_check(D, x * weight.unsqueeze(-2), "D")
    _rel_check(premult, x, "premult")
    _rel_check(sqsum, sq_ref, "sqsum", tol=1e-3)

    D2 = torch.empty_like(D)
    premult2 = torch.empty_like(premult)
    sqsum2 = torch.empty_like(sqsum)
    gemm_sq_reduce(
        A,
        B,
        D2,
        None,
        sqsum2,
        None,
        tile_M,
        tile_N,
        cluster_M,
        cluster_N,
        rowvec=weight,
        aux_out=premult2,
    )
    _rel_check(D, D2.float(), "D vs handwritten", tol=1e-3)
    _rel_check(premult, premult2.float(), "premult vs handwritten", tol=1e-3)
    _rel_check(sqsum, sqsum2, "sqsum vs handwritten", tol=1e-4)


def test_epi_mod_gated_swiglu():
    """GemmGated as a mod, cross-checked against the hand-written kernel."""
    device = "cuda"
    torch.random.manual_seed(6)
    l, m, N, k = 2, 512, 2048, 736
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, N, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    postact = torch.empty((l, m, N // 2), device=device, dtype=torch.bfloat16)

    swiglu_mod.gemm(
        A,
        B,
        None,
        epi_args=dict(postact=postact),
        tile_M=128,
        tile_N=256,
        # cluster_M=1: gated + cluster_M=2 miscomputes ~27% of postact on SM100 —
        # pre-existing bug in the hand-written kernel (reproduces on main @8ff10ac
        # via the raw gemm_act dispatch; see dbg_gated6 minimized repro).
        cluster_M=1,
        cluster_N=1,
    )

    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    gate, up = x[..., 0::2], x[..., 1::2]
    ref = torch.nn.functional.silu(gate) * up
    _rel_check(postact, ref, "postact")

    postact2 = torch.empty_like(postact)
    gemm_act(A, B, None, None, postact2, None, "swiglu", 128, 256, 1, 1)
    _rel_check(postact, postact2.float(), "postact vs handwritten", tol=1e-3)


def test_epi_mod_gated_operands_and_d():
    """Gated mod with rowvec (per-lane tuple), colvec (scalar), and D writeback."""
    device = "cuda"
    torch.random.manual_seed(7)
    l, m, N, k = 2, 384, 1536, 512
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, N, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    D = torch.empty((l, m, N), device=device, dtype=torch.bfloat16)
    postact = torch.empty((l, m, N // 2), device=device, dtype=torch.bfloat16)
    rstd = torch.rand((l, m), device=device, dtype=torch.float32) + 0.5
    bias = torch.randn((l, N), device=device, dtype=torch.float32)  # interleaved gate/up

    norm_swiglu_mod.gemm(
        A,
        B,
        D,
        epi_args=dict(rstd=rstd, bias=bias, postact=postact),
        tile_M=128,
        tile_N=256,
        cluster_M=1,  # see cluster_M note in test_epi_mod_gated_swiglu
        cluster_N=1,
    )

    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    x = x * rstd.unsqueeze(-1) + bias.unsqueeze(-2)
    gate, up = x[..., 0::2], x[..., 1::2]
    _rel_check(postact, torch.nn.functional.silu(gate) * up, "postact")
    _rel_check(D, x, "D writeback")


def _dswiglu_torch_ref(x, y, dout):
    xg = x.detach().requires_grad_()
    yg = y.detach().requires_grad_()
    out = torch.nn.functional.silu(xg) * yg
    out.backward(dout)
    return xg.grad, yg.grad, (torch.nn.functional.silu(x) * y)


def test_epi_mod_dgated():
    """GemmDGated as a mod, cross-checked against the hand-written kernel."""
    device = "cuda"
    torch.random.manual_seed(8)
    l, m, n, k = 2, 512, 1024, 736  # n = pair count; PreAct/Out are (l, m, 2n)
    dout_in = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    W = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    preact = torch.randn((l, m, 2 * n), device=device, dtype=torch.bfloat16)
    out_mod = torch.empty_like(preact)
    postact = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)

    dswiglu_mod.gemm(
        dout_in,
        W,
        out_mod,
        preact,
        epi_args=dict(postact=postact),
        tile_M=128,
        tile_N=256,
        cluster_M=1,
        cluster_N=1,
    )

    dout = torch.einsum("lmk,lnk->lmn", dout_in.float(), W.float())
    x, y = preact.float()[..., 0::2], preact.float()[..., 1::2]
    dx_ref, dy_ref, out_ref = _dswiglu_torch_ref(x, y, dout)
    _rel_check(out_mod[..., 0::2], dx_ref, "dx")
    _rel_check(out_mod[..., 1::2], dy_ref, "dy")
    _rel_check(postact, out_ref, "postact")

    out_hand = torch.empty_like(preact)
    postact_hand = torch.empty_like(postact)
    gemm_dact(
        dout_in,
        W,
        out_hand,
        preact,
        postact_hand,
        None,
        "swiglu",
        128,
        256,
        1,
        1,
        pingpong=False,
    )
    _rel_check(out_mod.float(), out_hand.float(), "D vs handwritten", tol=1e-3)
    _rel_check(postact, postact_hand.float(), "postact vs handwritten", tol=1e-3)


@pytest.mark.parametrize("tile_N", [192, 256])
def test_epi_mod_dgated_norm_reduce(tile_N):
    """Full dgated (colvec scale + reduce), cross-checked against hand-written."""
    device = "cuda"
    torch.random.manual_seed(9)
    l, m, n, k = 2, 384, 1536, 512
    tile_M = 128
    dout_in = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    W = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    preact = torch.randn((l, m, 2 * n), device=device, dtype=torch.bfloat16)
    rstd = torch.rand((l, m), device=device, dtype=torch.float32) + 0.5
    out_mod = torch.empty_like(preact)
    postact = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    n_tiles = (n + tile_N - 1) // tile_N
    dsum = torch.empty((l, m, n_tiles), device=device, dtype=torch.float32)

    dswiglu_norm_mod.gemm(
        dout_in,
        W,
        out_mod,
        preact,
        epi_args=dict(rstd=rstd, postact=postact, dsum=dsum),
        tile_M=tile_M,
        tile_N=tile_N,
        cluster_M=1,
        cluster_N=1,
    )

    dout = torch.einsum("lmk,lnk->lmn", dout_in.float(), W.float())
    x, y = preact.float()[..., 0::2], preact.float()[..., 1::2]
    dx_ref, dy_ref, out_pre = _dswiglu_torch_ref(x, y, dout * rstd.unsqueeze(-1))
    _rel_check(out_mod[..., 0::2], dx_ref, "dx")
    _rel_check(out_mod[..., 1::2], dy_ref, "dy")
    _rel_check(postact, out_pre * rstd.unsqueeze(-1), "postact")
    prod = out_pre * dout
    pad = n_tiles * tile_N - n
    if pad:
        prod = torch.nn.functional.pad(prod, (0, pad))
    _rel_check(dsum, prod.unflatten(-1, (n_tiles, tile_N)).sum(dim=-1), "dsum", tol=1e-3)

    out_hand = torch.empty_like(preact)
    postact_hand = torch.empty_like(postact)
    dsum_hand = torch.empty_like(dsum)
    gemm_dact(
        dout_in,
        W,
        out_hand,
        preact,
        postact_hand,
        None,
        "swiglu",
        tile_M,
        tile_N,
        1,
        1,
        pingpong=False,
        colvec_scale=rstd,
        colvec_reduce=dsum_hand,
    )
    _rel_check(out_mod.float(), out_hand.float(), "D vs handwritten", tol=1e-3)
    _rel_check(postact, postact_hand.float(), "postact vs handwritten", tol=1e-3)
    _rel_check(dsum, dsum_hand, "dsum vs handwritten", tol=1e-4)


def test_epi_mod_residual_tileload():
    """Residual via the TileLoad epilogue-pipeline path (C absent)."""
    device = "cuda"
    torch.random.manual_seed(10)
    l, m, n, k = 2, 512, 768, 736
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    res = torch.randn((l, m, n), device=device, dtype=torch.bfloat16)
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)

    residual_epi.gemm(
        A,
        B,
        D,
        epi_args=dict(res=res),
        tile_M=128,
        tile_N=192,
        cluster_M=1,
        cluster_N=1,
    )

    ref = torch.einsum("lmk,lnk->lmn", A.float(), B.float()) + res.float()
    _rel_check(D, ref, "D")


def test_epi_mod_rope():
    """RoPE: paired accumulator with no aux buffer (explicit paired=)."""
    device = "cuda"
    torch.random.manual_seed(11)
    l, m, n, k = 2, 512, 256, 736  # n = head_dim * heads-in-tile-N sense; pairs = n // 2
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    # Interleaved cos/sin table, congruent with the D tile: standard RoPE
    # angles theta_j = base^(-2j/d) rotated by position.
    pos = torch.arange(m, device=device, dtype=torch.float32)
    inv_freq = 10000.0 ** (-torch.arange(n // 2, device=device, dtype=torch.float32) / (n // 2))
    ang = pos[:, None] * inv_freq[None, :]  # (m, n/2)
    table = torch.empty((l, m, n), device=device, dtype=torch.float32)
    table[..., 0::2] = ang.cos()
    table[..., 1::2] = ang.sin()

    rope_epi.gemm(
        A,
        B,
        D,
        epi_args=dict(table=table),
        tile_M=128,
        tile_N=128,
        cluster_M=1,
        cluster_N=1,
    )

    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = ang.cos(), ang.sin()
    ref = torch.empty_like(x)
    ref[..., 0::2] = x1 * cos - x2 * sin
    ref[..., 1::2] = x1 * sin + x2 * cos
    _rel_check(D, ref, "D")


def test_epi_mod_lse_partials():
    """LSE via per-tile sum-of-exp partials, host-finalized."""
    device = "cuda"
    torch.random.manual_seed(12)
    l, m, n, k = 2, 512, 1536, 512
    tile_N = 256
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    n_tiles = (n + tile_N - 1) // tile_N
    sexp = torch.empty((l, m, n_tiles), device=device, dtype=torch.float32)
    scale = 8.0  # logits stay small enough for the no-online-max variant

    lse_partial_epi.gemm(
        A,
        B,
        D,
        epi_args=dict(scale=scale, sexp=sexp),
        tile_M=128,
        tile_N=tile_N,
        cluster_M=1,
        cluster_N=1,
    )

    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    lse_ref = torch.logsumexp(x * scale, dim=-1)
    lse = sexp.sum(dim=-1).log()
    _rel_check(D, x, "D")
    err = (lse - lse_ref).abs().max().item()
    assert err < 1e-2, f"lse err {err}"


@pytest.mark.parametrize("n,tile_N", [(1536, 256), (1024, 192)])
def test_epi_mod_amax_reduce(n, tile_N):
    device = "cuda"
    torch.random.manual_seed(13)
    l, m, k = 2, 512, 736
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    n_tiles = (n + tile_N - 1) // tile_N
    amax = torch.empty((l, m, n_tiles), device=device, dtype=torch.float32)

    amax_epi.gemm(
        A,
        B,
        D,
        epi_args=dict(amax=amax),
        tile_M=128,
        tile_N=tile_N,
        cluster_M=1,
        cluster_N=1,
    )

    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    pad = n_tiles * tile_N - n
    xa = torch.nn.functional.pad(x.abs(), (0, pad)) if pad else x.abs()
    ref = xa.unflatten(-1, (n_tiles, tile_N)).amax(dim=-1)
    _rel_check(D, x, "D")
    _rel_check(amax, ref, "amax", tol=1e-3)


class RopeOp(TileLoad):
    """User-defined apply-port op (defined here, not in quack — that's the
    point): loads an interleaved cos/sin table through TileLoad's staged
    pipeline and rotates adjacent-N pairs wherever the fn calls it."""

    fn_port = "apply"

    @cute.jit
    def fn_prepare(self, gemm, state, paired):
        assert paired, "RopeOp rotates adjacent-N pairs: use a paired-acc epilogue"
        t = state.to(gemm.acc_dtype)
        p = cute.flat_divide(t, cute.make_layout(2))
        return (p[0, ...], p[1, ...])

    @cute.jit
    def fn_apply(self, gemm, pstate, i, v):
        x1, x2 = unpack(v)
        cos, sin = pstate[0][i], pstate[1][i]
        return pack(x1 * cos - x2 * sin, x1 * sin + x2 * cos)


@gemm_epilogue(
    ops={"rope": RopeOp("rope")},
    reduces={"rowsum": ColVecReduce("rowsum")},
    paired=("acc",),
)
def rope_rowsum_epi(acc, rope, alpha):
    """The composition ask: rope(acc) * alpha, then row sum — an apply-port op
    slotted into fn math, feeding an existing sink."""
    y = rope(acc) * alpha
    return {"D": y, "rowsum": y}


@gemm_epilogue(outs={"lse": OnlineLSEReduce("lse")})
def lse_epi(acc, scale):
    """Online LSE sink: the coupled (max, sum) accumulator combine= can't
    express. Stable at logit scales where naive sum-exp overflows f32."""
    return {"D": acc, "lse": acc * scale}


@gemm_epilogue(outs={"lse": OnlineLSEReduce("lse", check_oob=False)})
def lse_nocheck_epi(acc, scale):
    """check_oob=False variant: OOB predicate compiled out (CUTLASS
    VisitCheckOOB=false); the host rejects N not divisible by tile_N."""
    return {"D": acc, "lse": acc * scale}


def test_epi_mod_rope_apply_rowsum():
    device = "cuda"
    torch.random.manual_seed(14)
    l, m, n, k = 2, 512, 256, 736
    tile_N, alpha = 128, 1.7
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    pos = torch.arange(m, device=device, dtype=torch.float32)
    inv_freq = 10000.0 ** (-torch.arange(n // 2, device=device, dtype=torch.float32) / (n // 2))
    ang = pos[:, None] * inv_freq[None, :]
    table = torch.empty((l, m, n), device=device, dtype=torch.float32)
    table[..., 0::2] = ang.cos()
    table[..., 1::2] = ang.sin()
    n_tiles = n // tile_N
    rowsum = torch.empty((l, m, n_tiles), device=device, dtype=torch.float32)

    rope_rowsum_epi.gemm(
        A,
        B,
        D,
        epi_args=dict(rope=table, alpha=alpha, rowsum=rowsum),
        tile_M=128,
        tile_N=tile_N,
        cluster_M=1,
        cluster_N=1,
    )

    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y = torch.empty_like(x)
    y[..., 0::2] = x1 * ang.cos() - x2 * ang.sin()
    y[..., 1::2] = x1 * ang.sin() + x2 * ang.cos()
    y = y * alpha
    _rel_check(D, y, "D")
    _rel_check(rowsum, y.unflatten(-1, (n_tiles, tile_N)).sum(dim=-1), "rowsum", tol=1e-3)


# overflow: logits ~ +-1300, exp overflows f32 without the online max.
# ragged: last N tile is partial (1160 = 4*256 + 136; N stride must stay 8-divisible
# for TMA) with all logits pushed negative, so an unpredicated fold of the OOB
# accumulator zeros would dominate both the max (0 > all logits) and the sum.
# 1040 = 4*256 + 16: the boundary tile masks ENTIRE slots at init time on
# warp-split-N epi layouts (SM120: warp 1's first chunk starts at n=16), so
# the -inf fold identity meets itself — regression for the _guard_neg_inf
# subtrahend (unguarded, (-inf) - (-inf) = NaN poisons the row's sum).
@pytest.mark.parametrize(
    "n,regime,check_oob",
    [
        (1024, "overflow", True),
        (1160, "negative", True),
        (1160, "overflow", True),
        (1040, "negative", True),
        (1024, "overflow", False),  # divisible N: predicate compiled out, same math
    ],
)
def test_epi_mod_online_lse(n, regime, check_oob):
    """Logits far beyond f32 exp range: naive sum-exp is inf, online LSE exact.

    Ragged n exercises the OOB predication: the accumulator zeros in the last
    tile's OOB columns must not enter the (max, sum) fold. The negative regime
    is the sharp regression for that — with logits ~ -10, a single unpredicated
    zero shifts the tile LSE to ~log(#oob).
    """
    device = "cuda"
    torch.random.manual_seed(15)
    l, m, k = 2, 512, 512
    tile_N = 256
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    if regime == "overflow":
        scale = 64.0  # logits ~ +-1300
    else:
        # Anti-correlated signs: acc = -sum |a||b| ~ -10, strictly negative.
        A, B, scale = A.abs(), -B.abs(), 1.0
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    n_tiles = (n + tile_N - 1) // tile_N
    lse = torch.empty((l, m, n_tiles), device=device, dtype=torch.float32)

    mod = lse_epi if check_oob else lse_nocheck_epi
    mod.gemm(
        A,
        B,
        D,
        epi_args=dict(scale=scale, lse=lse),
        tile_M=128,
        tile_N=tile_N,
        cluster_M=1,
        cluster_N=1,
    )

    logits = torch.einsum("lmk,lnk->lmn", A.float(), B.float()) * scale
    if regime == "overflow":
        assert torch.isinf(logits.exp().sum(dim=-1)).any(), (
            "test regime should overflow naive sumexp"
        )
    else:
        assert logits.max().item() < 0, "test regime should make OOB zeros the (wrong) max"
    pad = n_tiles * tile_N - n
    logits_p = torch.nn.functional.pad(logits, (0, pad), value=-math.inf) if pad else logits
    ref_tiles = torch.logsumexp(logits_p.unflatten(-1, (n_tiles, tile_N)), dim=-1)
    err = (lse - ref_tiles).abs().max().item()
    assert err < 1e-2, f"per-tile lse err {err}"
    final = torch.logsumexp(lse, dim=-1)
    ref = torch.logsumexp(logits, dim=-1)
    assert (final - ref).abs().max().item() < 1e-2


@gemm_epilogue(ops={"sr_seed": Scalar("sr_seed", dtype=Int32)})
def sr_epi(acc, sr_seed):
    """Plain D store; the sr_seed scalar feeds the stochastic-rounding D
    conversion when the kernel is minted with rounding_mode=RS."""
    return {"D": acc}


def test_epi_mod_stochastic_rounding():
    """RS through the fn frontend: hw cvt.rs on SM100/SM103, sw emulation on
    SM90/SM120. Checks RS engages (differs from RN), stays within the usual
    SR error envelope, and is reproducible per seed."""
    from quack.rounding import RoundingMode

    device = "cuda"
    torch.random.manual_seed(19)
    l, m, n, k = 1, 512, 1024, 512
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    ref = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    cfg = dict(tile_M=128, tile_N=256, cluster_M=1, cluster_N=1)

    def run(mode, seed):
        D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
        sr_epi.gemm(A, B, D, epi_args=dict(sr_seed=seed), rounding_mode=mode, **cfg)
        return D

    D_rn = run(RoundingMode.RN, 42)
    D_rs = run(RoundingMode.RS, 42)
    D_rs_same = run(RoundingMode.RS, 42)
    D_rs_other = run(RoundingMode.RS, 43)
    assert not torch.equal(D_rs, D_rn), "RS should differ from RN somewhere"
    assert torch.equal(D_rs, D_rs_same), "same seed must reproduce bitwise"
    assert not torch.equal(D_rs, D_rs_other), "different seeds should differ"
    err_rs = (D_rs.float() - ref).abs().max().item()
    err_rn = (D_rn.float() - ref).abs().max().item()
    assert err_rs < 3 * err_rn + 5e-3, f"RS err {err_rs} vs RN err {err_rn}"


def test_epi_mod_online_lse_nocheck_rejects_ragged():
    """check_oob=False compiles the OOB predicate out, so the host must
    reject N not divisible by tile_N instead of silently corrupting the LSE."""
    device = "cuda"
    l, m, n, k, tile_N = 2, 512, 1160, 512, 256
    A = torch.empty((l, m, k), device=device, dtype=torch.bfloat16)
    B = torch.empty((l, n, k), device=device, dtype=torch.bfloat16)
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    lse = torch.empty((l, m, (n + tile_N - 1) // tile_N), device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="check_oob=False"):
        lse_nocheck_epi.gemm(
            A,
            B,
            D,
            epi_args=dict(scale=1.0, lse=lse),
            tile_M=128,
            tile_N=tile_N,
            cluster_M=1,
            cluster_N=1,
        )


@pytest.mark.parametrize("tma", [False, True])  # gmem->rmem op vs TMA-staged op
@pytest.mark.parametrize("tile_N", [64, 128, 256])  # < head_dim (slice path), ==, 2 heads/tile
def test_epi_mod_rope_table_op(tile_N, tma):
    device = "cuda"
    torch.random.manual_seed(16)
    l, m, k, head_dim, heads = 2, 384, 736, 128, 4  # m != n keeps bias inference unambiguous
    n = head_dim * heads
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    bias = torch.randn((l, n), device=device, dtype=torch.float32)
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    pos = torch.arange(m, device=device, dtype=torch.float32)
    inv_freq = 10000.0 ** (
        -torch.arange(head_dim // 2, device=device, dtype=torch.float32) / (head_dim // 2)
    )
    ang = pos[:, None] * inv_freq[None, :]  # (m, head_dim/2)
    table = torch.stack([ang.cos(), ang.sin()], dim=-1).reshape(m, head_dim).contiguous()

    mod = rope_table_epi if tma else rope_table_ldg_epi
    mod.gemm(
        A,
        B,
        D,
        epi_args=dict(cs=table, bias=bias),
        tile_M=128,
        tile_N=tile_N,
        cluster_M=1,
        cluster_N=1,
    )

    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float()) + bias.unsqueeze(-2)
    xp = x.unflatten(-1, (heads, head_dim // 2, 2))
    c = ang.cos()[None, :, None, :]
    s = ang.sin()[None, :, None, :]
    ref = torch.empty_like(xp)
    ref[..., 0] = xp[..., 0] * c - xp[..., 1] * s
    ref[..., 1] = xp[..., 0] * s + xp[..., 1] * c
    _rel_check(D, ref.reshape(l, m, n), "D")


def _qknorm_ref(x, w, eps):
    head_dim = w.shape[0]
    xh = x.unflatten(-1, (-1, head_dim))
    rstd = torch.rsqrt(xh.float().pow(2).mean(-1, keepdim=True) + eps)
    return (xh * rstd * w).reshape(x.shape)


def _quack_capability():
    # QUACK_ARCH-aware (CI runs e.g. SM120 dispatch on H100 runners); the
    # torch capability would report the physical GPU and mis-skip.
    from quack.cute_dsl_utils import get_device_capacity

    return get_device_capacity(torch.device("cuda"))[0]


def _skip_unless_acc_prepass():
    if _quack_capability() not in (9, 10, 11, 12):
        pytest.skip("acc prepass needs a re-readable accumulator (SM90/SM100/SM110/SM120)")


# (tile_N, head_dim): one head per tile, several heads per tile, and a
# head_dim below the SM90 epi-tile N extent — each exercises a different
# (row, head) smem indexing shape in HeadRMSNormStats. pingpong (SM90): the two
# warpgroups' epilogues are strictly exclusive (TMA drain before the epi
# barrier hand-off), so the temporally-shared stats smem must stay correct.
@pytest.mark.parametrize(
    "tile_N,head_dim,pingpong",
    [(128, 128, False), (256, 128, False), (128, 64, False), (128, 128, True), (128, 64, True)],
)
def test_epi_mod_qknorm_prepass(tile_N, head_dim, pingpong):
    _skip_unless_acc_prepass()
    if pingpong and _quack_capability() not in (9, 12):
        pytest.skip("pingpong is an SM90/SM120 schedule")
    device = "cuda"
    torch.random.manual_seed(17)
    l, m, k, heads = 2, 384, 736, 512 // head_dim
    n = head_dim * heads
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    w = torch.randn(head_dim, device=device, dtype=torch.float32).abs() + 0.5
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)

    qknorm_epi.gemm(
        A,
        B,
        D,
        epi_args=dict(qk=w),
        tile_M=128,
        tile_N=tile_N,
        cluster_M=1,
        cluster_N=1,
        pingpong=pingpong,
    )

    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    _rel_check(D, _qknorm_ref(x, w, 1e-6), "D")


@pytest.mark.parametrize("tma", [True, False])  # TMA-staged (default) vs gmem->rmem table
@pytest.mark.parametrize("pingpong", [False, True])
def test_epi_mod_qknorm_rope_prepass(pingpong, tma):
    _skip_unless_acc_prepass()
    if pingpong and _quack_capability() not in (9, 12):
        pytest.skip("pingpong is an SM90/SM120 schedule")
    device = "cuda"
    torch.random.manual_seed(18)
    l, m, k, head_dim, heads = 2, 384, 736, 128, 4
    n = head_dim * heads
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    w = torch.randn(head_dim, device=device, dtype=torch.float32).abs() + 0.5
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    pos = torch.arange(m, device=device, dtype=torch.float32)
    inv_freq = 10000.0 ** (
        -torch.arange(head_dim // 2, device=device, dtype=torch.float32) / (head_dim // 2)
    )
    ang = pos[:, None] * inv_freq[None, :]
    table = torch.stack([ang.cos(), ang.sin()], dim=-1).reshape(m, head_dim).contiguous()

    mod = qk_rope_epi if tma else qk_rope_ldg_epi
    mod.gemm(
        A,
        B,
        D,
        epi_args=dict(cs=table, qk=w),
        tile_M=128,
        tile_N=128,
        cluster_M=1,
        cluster_N=1,
        pingpong=pingpong,
    )

    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    y = _qknorm_ref(x, w, 1e-6)
    yp = y.unflatten(-1, (heads, head_dim // 2, 2))
    c = ang.cos()[None, :, None, :]
    s = ang.sin()[None, :, None, :]
    ref = torch.empty_like(yp)
    ref[..., 0] = yp[..., 0] * c - yp[..., 1] * s
    ref[..., 1] = yp[..., 0] * s + yp[..., 1] * c
    _rel_check(D, ref.reshape(l, m, n), "D")


def test_epi_mod_rms_block_pipeline():
    """(1)+(2): GEMM+residual+partial-rms -> host rstd -> GEMM+rstd+swiglu,
    validated end-to-end against a torch reference of the whole block."""
    device = "cuda"
    torch.random.manual_seed(19)
    l, m, k1, n1, pairs = 2, 384, 736, 1024, 512
    tile_N1, eps = 256, 1e-6
    x = torch.randn((l, m, k1), device=device, dtype=torch.bfloat16) / math.sqrt(k1) * 4
    W1 = torch.randn((l, n1, k1), device=device, dtype=torch.bfloat16) / math.sqrt(k1) * 4
    resid = torch.randn((l, m, n1), device=device, dtype=torch.bfloat16)
    w = torch.randn((l, n1), device=device, dtype=torch.float32).abs() + 0.5
    W2 = torch.randn((l, 2 * pairs, n1), device=device, dtype=torch.bfloat16) / math.sqrt(n1)

    # GEMM1: P = (x@W1 + resid) * w, resid_out, sqsum partials
    P = torch.empty((l, m, n1), device=device, dtype=torch.bfloat16)
    resid_out = torch.empty_like(P)
    n_tiles = n1 // tile_N1
    sqsum = torch.empty((l, m, n_tiles), device=device, dtype=torch.float32)
    rms_partial_epi.gemm(
        x,
        W1,
        P,
        resid,
        epi_args=dict(weight=w, resid_out=resid_out, sqsum=sqsum),
        tile_M=128,
        tile_N=tile_N1,
        cluster_M=1,
        cluster_N=1,
    )
    # host-side rstd finalize (stands in for rms_final_reduce)
    rstd = torch.rsqrt(sqsum.sum(-1) / n1 + eps)  # (l, m)

    # GEMM2: postact = swiglu((P @ W2^T) * rstd)
    postact = torch.empty((l, m, pairs), device=device, dtype=torch.bfloat16)
    rstd_swiglu_epi.gemm(
        P,
        W2,
        None,
        epi_args=dict(rstd=rstd, postact=postact),
        tile_M=128,
        tile_N=256,
        cluster_M=1,
        cluster_N=1,
    )

    # torch reference of the whole block
    y_ref = torch.einsum("lmk,lnk->lmn", x.float(), W1.float()) + resid.float()
    rstd_ref = torch.rsqrt(y_ref.pow(2).mean(-1) + eps)
    yhat = y_ref * rstd_ref.unsqueeze(-1) * w.unsqueeze(-2)
    h = torch.einsum("lmn,lpn->lmp", yhat, W2.float())
    ref = torch.nn.functional.silu(h[..., 0::2]) * h[..., 1::2]
    _rel_check(resid_out, y_ref, "resid_out")
    _rel_check(P, y_ref * w.unsqueeze(-2), "P (weighted, rstd deferred)")
    err = (rstd - rstd_ref).abs().max().item()
    assert err < 1e-3, f"rstd err {err}"
    # bf16 P + bf16 GEMM2 accumulate the block error; bound vs a bf16-P baseline
    h_bf = torch.einsum("lmn,lpn->lmp", (P.float() * rstd.unsqueeze(-1)), W2.float())
    ref_bf = torch.nn.functional.silu(h_bf[..., 0::2]) * h_bf[..., 1::2]
    _rel_check(postact, ref_bf, "postact vs bf16-chain ref")
    assert (ref_bf - ref).abs().max().item() < 0.3, "sanity: chain refs should be close"


def test_epi_mod_rms_bwd_link():
    """(3): the rmsnorm-backward link — dgrad GEMM + saved-prenorm TileLoad +
    rstd colvec + w rowvec + correction-dot partials; assembled dx checked
    against torch autograd of rmsnorm."""
    device = "cuda"
    torch.random.manual_seed(20)
    l, m, k, n = 2, 384, 512, 1024
    tile_N, eps = 256, 1e-6
    dz = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    W2t = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    y = torch.randn((l, m, n), device=device, dtype=torch.bfloat16)  # saved pre-norm
    w = torch.randn((l, n), device=device, dtype=torch.float32).abs() + 0.5
    rstd = torch.rsqrt(y.float().pow(2).mean(-1) + eps)  # (l, m)

    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    n_tiles = n // tile_N
    dots = torch.empty((l, m, n_tiles), device=device, dtype=torch.float32)
    rms_bwd_partial_epi.gemm(
        dz,
        W2t,
        D,
        epi_args=dict(y=y, rstd=rstd, w=w, dots=dots),
        tile_M=128,
        tile_N=tile_N,
        cluster_M=1,
        cluster_N=1,
    )

    dyhat = torch.einsum("lmk,lnk->lmn", dz.float(), W2t.float())
    t = dyhat * w.unsqueeze(-2)
    xhat = y.float() * rstd.unsqueeze(-1)
    _rel_check(D, t * rstd.unsqueeze(-1), "D (= rstd*t)")
    _rel_check(dots, (t * xhat).unflatten(-1, (n_tiles, tile_N)).sum(-1), "dots", tol=1e-3)

    # Assemble the final dx from the kernel outputs and check vs autograd.
    dx = D.float() - xhat * rstd.unsqueeze(-1) * (dots.sum(-1) / n).unsqueeze(-1)
    yg = y.float().detach().requires_grad_()
    out = yg * torch.rsqrt(yg.pow(2).mean(-1, keepdim=True) + eps) * w.unsqueeze(-2)
    out.backward(dyhat)
    err = (dx - yg.grad).abs().max().item()
    scale = yg.grad.abs().max().item()
    assert err < 2e-2 * scale + 1e-2, f"assembled dx vs autograd: {err} (scale {scale})"


# ── Varlen through the fn frontend ───────────────────────────────────────────


def _varlen_setup(seqlens, k, n, device, seed):
    torch.random.manual_seed(seed)
    total_m = sum(seqlens)
    cu = torch.tensor(
        [0, *torch.tensor(seqlens).cumsum(0).tolist()], device=device, dtype=torch.int32
    )
    A = torch.randn((total_m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((len(seqlens), n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    return total_m, cu, A, B


def _varlen_ref_x(A, B, cu):
    xs = []
    for b in range(B.shape[0]):
        xs.append(torch.einsum("mk,nk->mn", A[cu[b] : cu[b + 1]].float(), B[b].float()))
    return torch.cat(xs, dim=0)


def test_epi_mod_varlen_norm_gelu():
    """Varlen: rank-1 colvec (total_m,), per-segment rowvec, aux TileStore."""
    device = "cuda"
    seqlens, k, n = [200, 184], 736, 1024
    total_m, cu, A, B = _varlen_setup(seqlens, k, n, device, 21)
    D = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)
    postact = torch.empty_like(D)
    rstd = torch.rand(total_m, device=device, dtype=torch.float32) + 0.5  # rank-1 varlen colvec
    weight = torch.randn((len(seqlens), n), device=device, dtype=torch.float32)

    norm_gelu.gemm(
        A,
        B,
        D,
        epi_args=dict(rstd=rstd, weight=weight, postact=postact),
        tile_M=128,
        tile_N=256,
        cluster_M=1,
        cluster_N=1,
        cu_seqlens_m=cu,
    )

    x = _varlen_ref_x(A, B, cu)
    wfull = torch.cat([weight[b].expand(seqlens[b], n) for b in range(len(seqlens))], dim=0)
    ref = x * rstd.unsqueeze(-1) * wfull
    _rel_check(D, ref, "D")
    _rel_check(postact, torch.nn.functional.gelu(ref, approximate="tanh"), "postact")


def test_epi_mod_varlen_rms_partial():
    """Varlen: C residual + aux + ColVecReduce partials, all (total_m, ...)."""
    device = "cuda"
    seqlens, k, n, tile_N = [200, 184], 512, 768, 192
    total_m, cu, A, B = _varlen_setup(seqlens, k, n, device, 22)
    resid = torch.randn((total_m, n), device=device, dtype=torch.bfloat16)
    w = torch.randn((len(seqlens), n), device=device, dtype=torch.float32)
    D = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)
    resid_out = torch.empty_like(D)
    n_tiles = n // tile_N
    sqsum = torch.empty((total_m, n_tiles), device=device, dtype=torch.float32)

    rms_partial_epi.gemm(
        A,
        B,
        D,
        resid,
        epi_args=dict(weight=w, resid_out=resid_out, sqsum=sqsum),
        tile_M=128,
        tile_N=tile_N,
        cluster_M=1,
        cluster_N=1,
        cu_seqlens_m=cu,
    )

    y = _varlen_ref_x(A, B, cu) + resid.float()
    wfull = torch.cat([w[b].expand(seqlens[b], n) for b in range(len(seqlens))], dim=0)
    _rel_check(resid_out, y, "resid_out")
    _rel_check(D, y * wfull, "D")
    _rel_check(sqsum, (y * y).unflatten(-1, (n_tiles, tile_N)).sum(-1), "sqsum", tol=1e-3)


def test_epi_mod_varlen_rope():
    """Varlen RoPE — the epirope capability: table indexed by global flattened
    row (per the op's contract), pre-gathered here to per-segment positions.
    Uses the LDG table op (rope_table_ldg_epi): TMA loads have no varlen path."""
    device = "cuda"
    seqlens, k, head_dim, heads = [200, 184], 736, 128, 4
    n = head_dim * heads
    total_m, cu, A, B = _varlen_setup(seqlens, k, n, device, 23)
    D = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)
    inv_freq = 10000.0 ** (
        -torch.arange(head_dim // 2, device=device, dtype=torch.float32) / (head_dim // 2)
    )
    # per-token position restarting each segment, gathered into the table
    pos = torch.cat([torch.arange(s, device=device, dtype=torch.float32) for s in seqlens])
    ang = pos[:, None] * inv_freq[None, :]  # (total_m, head_dim/2)
    table = torch.stack([ang.cos(), ang.sin()], dim=-1).reshape(total_m, head_dim).contiguous()

    rope_table_ldg_epi.gemm(
        A,
        B,
        D,
        epi_args=dict(
            cs=table, bias=torch.zeros((len(seqlens), n), device=device, dtype=torch.float32)
        ),
        tile_M=128,
        tile_N=128,
        cluster_M=1,
        cluster_N=1,
        cu_seqlens_m=cu,
    )

    x = _varlen_ref_x(A, B, cu)
    xp = x.unflatten(-1, (heads, head_dim // 2, 2))
    c = ang.cos()[:, None, :]
    s = ang.sin()[:, None, :]
    ref = torch.empty_like(xp)
    ref[..., 0] = xp[..., 0] * c - xp[..., 1] * s
    ref[..., 1] = xp[..., 0] * s + xp[..., 1] * c
    _rel_check(D, ref.reshape(total_m, n), "D")


# ── Fail-closed semantic keying ──────────────────────────────────────────────


def test_semantic_key_fail_closed_and_protocol():
    """Unsupported captures are rejected loudly; __quack_semantic_key__ opts
    types in; partials/dataclasses key by content (the old best-effort walk
    keyed all partials identically — a silent-collision bug)."""
    import dataclasses
    from functools import partial

    def mint(captured):
        @gemm_epilogue()
        def epi(acc):
            _ = captured
            return {"D": acc}

        return epi

    # 1. Reject: a bare object has no stable semantic representation.
    with pytest.raises(TypeError, match="__quack_semantic_key__"):
        mint(object())

    # 2. Protocol: key changes with the returned value, not object identity.
    class TableCfg:
        def __init__(self, base):
            self.base = base

        def __quack_semantic_key__(self):
            return ("tablecfg", self.base)

    d1 = mint(TableCfg(10000.0)).semantic_digest
    d2 = mint(TableCfg(10000.0)).semantic_digest
    d3 = mint(TableCfg(500.0)).semantic_digest
    assert d1 == d2 and d1 != d3

    # 3. Partials key by (func, args, kwargs) — distinct funcs must differ.
    p_relu = mint(partial(torch.relu))
    p_tanh = mint(partial(torch.tanh))
    assert p_relu.semantic_digest != p_tanh.semantic_digest

    # 4. Dataclasses key by fields.
    @dataclasses.dataclass
    class Cfg:
        eps: float

    assert mint(Cfg(1e-6)).semantic_digest != mint(Cfg(1e-5)).semantic_digest

    # 5. EpiOps implement the protocol as their cache identity.
    op = ColVecReduce("s", combine="max")
    assert op.__quack_semantic_key__() == op.cache_key()


def test_semantic_digest_ignores_extern_library_state():
    """The digest must not depend on runtime-mutable state inside installed
    libraries. cutlass._mlir_helpers.op lazily materializes _DSL_PACKAGE_ROOT(S)
    on the first traced op, so a digest that recursed into cutlass function
    globals differed between "before any compile" and "after a compile" in the
    same process: async-compile workers (which typically compile other keys
    before their lazy first import of quack.epilogues) then rejected every
    module-global epilogue ref as "changed while resolving" and every one of
    those keys fell back to an in-process compile."""
    import cutlass._mlir_helpers.op as _op

    from quack.activation import gelu_tanh_approx

    def mint():
        @gemm_epilogue()
        def epi(acc):
            return {"D": gelu_tanh_approx(acc)}

        return epi

    saved = (_op._DSL_PACKAGE_ROOT, _op._DSL_PACKAGE_ROOTS)
    try:
        # State of a process that has never compiled anything.
        _op._DSL_PACKAGE_ROOT, _op._DSL_PACKAGE_ROOTS = "", None
        d_fresh = mint().semantic_digest
        # First framework-frame check (any traced op) populates the roots.
        _op._is_framework_frame.cache_clear()
        _op._is_framework_frame(__file__)
        assert _op._DSL_PACKAGE_ROOTS is not None
        d_after_compile = mint().semantic_digest
    finally:
        _op._DSL_PACKAGE_ROOT, _op._DSL_PACKAGE_ROOTS = saved
        _op._is_framework_frame.cache_clear()
    assert d_fresh == d_after_compile


def test_epi_mod_multi_output_mixed_dtype():
    """Tier-1 unlock: several TileStores from one epilogue, mixed dtypes —
    each op derives its own dtype/copy-atom (no singular aux_out_dtype)."""
    from quack.activation import gelu_tanh_approx, relu

    device = "cuda"
    torch.random.manual_seed(11)

    @gemm_epilogue(outputs=("y1", "y2"))
    def dual_out(acc):
        return {"D": acc, "y1": gelu_tanh_approx(acc), "y2": relu(acc)}

    l, m, n, k = 2, 512, 1024, 736
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    y1 = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    y2 = torch.empty((l, m, n), device=device, dtype=torch.float32)  # mixed dtype
    dual_out.gemm(
        A, B, D, epi_args=dict(y1=y1, y2=y2), tile_M=128, tile_N=256, cluster_M=2, cluster_N=1
    )
    ref = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    _rel_check(D, ref, "D")
    _rel_check(y1, torch.nn.functional.gelu(ref, approximate="tanh"), "y1")
    _rel_check(y2, torch.relu(ref), "y2")
