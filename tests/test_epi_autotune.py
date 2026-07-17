"""Tests for the generic @gemm_epilogue mod autotuner (quack.epi_autotune).

Small shapes, injected 2-3 config sweeps (monkeypatched _config_space) so the
suite stays fast; the full-space sweep is exercised by the llama block harness.
"""

import math

import pytest
import torch

import quack.epi_autotune as epi_autotune
from quack.cute_dsl_utils import get_device_capacity
from quack.epi_autotune import sink_arg_shapes, tuned_mod_gemm
from quack.epilogues import rms_fused, rstd_swiglu_epi
from quack.gemm_config import GemmConfig


def _cap():
    return get_device_capacity(torch.device("cuda"))[0]


def _cfg(tile_m, tile_n, cluster_m=1, pingpong=False):
    return GemmConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        cluster_m=cluster_m,
        cluster_n=1,
        pingpong=pingpong,
        device_capacity=_cap(),
    )


@pytest.fixture()
def small_space(monkeypatch, tmp_path):
    monkeypatch.setenv("QUACK_CACHE_DIR", str(tmp_path))  # hermetic disk cache
    cfgs = [_cfg(128, 128), _cfg(128, 256), _cfg(128, 192, pingpong=_cap() in (9, 12))]
    monkeypatch.setattr(epi_autotune, "_config_space", lambda mod, device: cfgs)
    monkeypatch.setattr(epi_autotune, "_MOD_TUNERS", {})
    return cfgs


def test_tuned_mod_gemm_matches_explicit(small_space):
    """Winner output is bitwise-equal to an explicit mod.gemm at the same
    config; sink buffers sized worst-case get sliced per config."""
    device = "cuda"
    torch.random.manual_seed(40)
    l, m, n, k = 2, 512, 1536, 736
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    weight = torch.randn((l, n), device=device, dtype=torch.float32)
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    premult = torch.empty_like(D)
    shapes = sink_arg_shapes(rms_fused, m, n, l=l)
    assert shapes == {"sqsum": (l, m, (n + 127) // 128)}  # min tile_N in the space
    sqsum = torch.empty(shapes["sqsum"], device=device, dtype=torch.float32)

    res = tuned_mod_gemm(
        rms_fused, A, B, D, epi_args=dict(weight=weight, premult=premult, sqsum=sqsum)
    )
    n_tiles = (n + res.config.tile_n - 1) // res.config.tile_n
    assert res.sinks["sqsum"].shape == (l, m, n_tiles)

    D2 = torch.empty_like(D)
    premult2 = torch.empty_like(premult)
    sqsum2 = torch.empty((l, m, n_tiles), device=device, dtype=torch.float32)
    c = res.config
    rms_fused.gemm(
        A,
        B,
        D2,
        epi_args=dict(weight=weight, premult=premult2, sqsum=sqsum2),
        tile_M=c.tile_m,
        tile_N=c.tile_n,
        cluster_M=c.cluster_m,
        cluster_N=c.cluster_n,
        pingpong=c.pingpong,
    )
    assert torch.equal(D, D2)
    assert torch.equal(premult, premult2)
    assert torch.equal(res.sinks["sqsum"], sqsum2)
    # And the math is right regardless of which config won.
    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    sq_ref = x.pow(2).unflatten(-1, (n_tiles, c.tile_n)).sum(-1)
    assert (res.sinks["sqsum"] - sq_ref).abs().max().item() < 1e-3 * sq_ref.abs().max().item()


def test_tuned_mod_gemm_cache_hit(small_space):
    """Second call with identical metadata skips the sweep (in-memory cache)."""
    device = "cuda"
    torch.random.manual_seed(41)
    l, m, n, k = 2, 384, 1024, 512
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16)
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16)
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    postact = torch.empty((l, m, n // 2), device=device, dtype=torch.bfloat16)
    rstd = torch.rand((l, m), device=device, dtype=torch.float32) + 0.5

    args = dict(rstd=rstd, postact=postact)
    tuned_mod_gemm(rstd_swiglu_epi, A, B, D, epi_args=args)
    tuner = next(iter(epi_autotune._MOD_TUNERS.values()))
    assert len(tuner.cache) == 1
    bench_time = tuner.bench_time
    tuned_mod_gemm(rstd_swiglu_epi, A, B, D, epi_args=args)
    assert len(tuner.cache) == 1 and tuner.bench_time == bench_time  # no re-bench


def test_prune_rules(small_space):
    """acc_pair mods on SM90 with aux outputs only keep tile_N % 32 == 0; a
    too-small caller sink buffer prunes the finer-tile configs."""
    if _cap() != 9:
        pytest.skip("prune-rule assertions are written for SM90")
    named = dict(
        A=torch.empty((384, 512), device="cuda", dtype=torch.bfloat16),
        B=torch.empty((1024, 512), device="cuda", dtype=torch.bfloat16),
        b_kn=False,
    )
    from quack.autotuner import AutotuneConfig

    confs = [AutotuneConfig(config=c) for c in small_space + [_cfg(128, 208)]]
    surv = epi_autotune._prune_for_mod(rstd_swiglu_epi, confs, named)
    assert all(c.kwargs["config"].tile_n % 32 == 0 for c in surv), [
        c.kwargs["config"].tile_n for c in surv
    ]
    # sink-buffer capacity rule: a buffer sized for tile_N=256 (4 tiles) drops
    # the 128/192-tile configs.
    small_buf = torch.empty((384, 4), device="cuda", dtype=torch.float32)
    surv2 = epi_autotune._prune_for_mod(rms_fused, confs, named | dict(sqsum=small_buf))
    assert {c.kwargs["config"].tile_n for c in surv2} == {256}


def test_mod_digest_in_disk_key(small_space):
    """The disk-cache directory hash includes the mod digest via key=; two
    mods with different fn bodies must not share tuning files."""
    from quack.epilogues import rms_fused as m1, rstd_swiglu_epi as m2

    t1 = epi_autotune._get_tuner(m1, ("weight",), False, torch.device("cuda"))
    t2 = epi_autotune._get_tuner(m2, ("rstd",), False, torch.device("cuda"))
    assert m1.semantic_digest != m2.semantic_digest
    assert t1.fn.__name__ != t2.fn.__name__
    assert "mod_digest" in t1.keys
