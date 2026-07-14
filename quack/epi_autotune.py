# Copyright (c) 2026, Tri Dao.
"""Generic autotuning for @gemm_epilogue mods.

``tuned_mod_gemm(mod, A, B, D, C, epi_args=...)`` sweeps the arch's GemmConfig
space directly through ``mod.gemm()`` with the existing Autotuner machinery
(CUDA-graph L2-cold bench, async-compile pool overlap, disk cache under
``$QUACK_CACHE_DIR``) — no per-variant torch interface layer involved. Any mod
gets tuning for free; ``mod.gemm_tuned(...)`` is the method form.

Mechanics that make the generic path work with the Autotuner:

* One ``Autotuner`` per (mod semantic digest, epi-arg name set, has-C): the
  Autotuner derives its cache key and its L2-rotate clone sets from TOP-LEVEL
  named tensor kwargs, so ``epi_args`` is flattened into explicit kwargs and
  the wrapper carries a synthetic ``__signature__`` naming them (a dict value
  would neither key nor clone — every bench replay would share one buffer).
* ``mod_digest`` rides ``key=`` so editing the epilogue fn body invalidates
  in-memory AND disk tuning caches; the wrapper ``__name__`` embeds it so the
  ``<fn>.autotune.json`` files stay human-attributable.
* Reduce-sink buffers are tile-shaped ((l, m, n_tiles) / (l, m_tiles, n)):
  callers allocate them at the sweep's worst case (``sink_arg_shapes``) and
  the wrapper slices per config, so one buffer serves every tile size. The
  winning slice is returned in ``TunedModGemm.sinks``.
* mod.gemm validation errors (ValueError/TypeError) are rewrapped into
  RuntimeError: the bench loop only converts RuntimeError/MemoryError into an
  inf timing, and a config a prune rule missed must not abort the sweep.
* Scalar epi args do not enter the tuning key (only tensor metadata does):
  tile choice is insensitive to scalar VALUES, and mod.gemm re-plans per
  metadata on the real call anyway.

varlen_m/gather_A tune through this path (cu_seqlens_m/A_idx are top-level
tensor kwargs, so they key and clone like any operand); swap_ab rides
swap-at-trace for element-mode sink-less mods; ``dynamic_scheduler=True``
forces dynamic-persistent scheduling on every candidate (matching the old
per-variant tuned wrappers); blockscaled SFA/SFB sweeps the
_blockscaled_ok-pruned space; ``concat_layout`` enters the tuner key (the
old per-variant tuners aliased concat/non-concat winners). Not supported
(yet): split_k.
"""

from __future__ import annotations

import inspect
from functools import partial
from typing import NamedTuple

import torch

from quack.autotuner import AutotuneConfig, Autotuner
from quack.cute_dsl_utils import get_device_capacity
from quack.gemm_config import config_supports, get_all_configs

__all__ = ["tuned_mod_gemm", "sink_arg_shapes", "TunedModGemm"]


class TunedModGemm(NamedTuple):
    plan: object  # GemmEpiPlan of the winning call (already executed)
    config: object  # winning GemmConfig
    sinks: dict  # name -> the winning config's slice of the caller's buffer


def _cdiv(a, b):
    return (a + b - 1) // b


def _config_space(mod, device):
    """Coarse per-arch config list for this mod (before per-call pruning)."""
    cap = get_device_capacity(device)[0]
    hint = "gated" if mod.mode in ("acc_pair", "packed_cd_b16x2") else None
    cfgs = [
        c
        for c in get_all_configs(epilogue=hint)
        if c.device_capacity == cap
        # swap_ab rides swap-at-trace (2026-07-14): element-mode sink-less
        # mods only, enforced in _prune_for_mod per call.
        and not (c.swap_ab and (mod.mode != "element" or mod.sinks))
        and not c.use_tma_gather  # gather_A untested through the fn frontend
        and (c.split_k is None or c.split_k == 1)  # split-K is default-epilogue-only
    ]
    if not cfgs:
        raise ValueError(f"no GemmConfigs for device capacity {cap}")
    return cfgs


def _gemm_mn(A, B, b_kn):
    n = B.shape[-1] if b_kn else B.shape[-2]
    m = A.shape[-2] if A.ndim == 3 else A.shape[0]
    return m, n


def sink_arg_shapes(mod, m, n_gemm, l=None, device="cuda"):
    """Worst-case (over the tuning sweep) buffer shapes for the mod's reduce
    sinks, keyed by sink name. Allocate these f32 and pass them in epi_args;
    the tuner slices per config and TunedModGemm.sinks returns the live view."""
    cfgs = _config_space(mod, torch.device(device))
    min_tile_n = min(c.tile_n for c in cfgs)
    min_tile_m = min(c.tile_m for c in cfgs)
    shapes = {}
    for name, op in mod.sinks.items():
        if not hasattr(op, "dim"):
            continue
        inner = (m, _cdiv(n_gemm, min_tile_n)) if op.dim == 0 else (_cdiv(m, min_tile_m), n_gemm)
        shapes[name] = inner if l is None else (l, *inner)
    return shapes


def _slice_sinks(mod, epi_args, config, m_gemm, n_gemm):
    views = {}
    for name, op in mod.sinks.items():
        dim = getattr(op, "dim", None)
        if dim is None or name not in epi_args:
            continue
        buf = epi_args[name]
        if dim == 0:
            views[name] = buf[..., : _cdiv(n_gemm, config.tile_n)]
        else:
            views[name] = buf[..., : _cdiv(m_gemm, config.tile_m), :]
    return views


def _prune_for_mod(mod, configs, named_args, **kwargs):
    kwargs = named_args | kwargs
    A, B = kwargs["A"], kwargs["B"]
    cap = get_device_capacity(A.device)[0]
    A_idx = kwargs.get("A_idx")
    m_gemm, n_gemm = _gemm_mn(A, B, kwargs.get("b_kn", False))
    if A_idx is not None:
        m_gemm = A_idx.shape[0]
    has_out = bool(mod.outputs)
    survivors = []
    b_kn_call = kwargs.get("b_kn", False)
    varlen_m = kwargs.get("cu_seqlens_m") is not None
    varlen_or_gather = varlen_m or A_idx is not None
    blockscaled = kwargs.get("SFA") is not None
    has_concat = bool(kwargs.get("concat_layout"))
    for conf in configs:
        c = conf.kwargs["config"]
        if c.device_capacity != cap:
            continue
        if not config_supports(c, gather_A=A_idx is not None, varlen_m=varlen_m):
            continue
        if blockscaled and not (
            # Mirrors prune_invalid_gemm_configs._blockscaled_ok (SM100
            # tcgen05 MMA constraints; SF tmem is 64-N granular).
            c.device_capacity in (10, 11)
            and not c.swap_ab
            and c.tile_k is None
            and c.tile_m in (128, 256)
            and c.tile_n % 64 == 0
            and 64 <= c.tile_n <= 256
            and c.cluster_m <= 4
            and c.cluster_n <= 4
        ):
            continue
        if c.swap_ab and (
            not b_kn_call or varlen_or_gather or has_concat or mod.mode != "element" or mod.sinks
        ):
            continue
        if mod.mode == "acc_pair":
            if c.tile_n % 2:
                continue
            if cap == 9 and has_out and c.tile_n % 32:
                continue
        ok = True
        for name, op in mod.sinks.items():
            if getattr(op, "check_oob", True) is False and n_gemm % c.tile_n:
                ok = False
            dim = getattr(op, "dim", None)
            buf = kwargs.get(name)
            if dim == 0 and buf is not None and buf.shape[-1] < _cdiv(n_gemm, c.tile_n):
                ok = False  # caller's partial buffer too small for this tile_N
            if dim == 1 and buf is not None and buf.shape[-2] < _cdiv(m_gemm, c.tile_m):
                ok = False
        if ok:
            survivors.append(conf)
    return survivors


def _make_tuned_fn(mod, epi_names):
    sink_dims = {n: op.dim for n, op in mod.sinks.items() if hasattr(op, "dim")}

    def fn(
        A=None,
        B=None,
        D=None,
        C=None,
        mod_digest=None,
        b_kn=False,
        cu_seqlens_m=None,
        A_idx=None,
        dynamic_scheduler=False,
        SFA=None,
        SFB=None,
        concat_layout=None,
        config=None,
        **epi_flat,
    ):
        c = config
        m_gemm, n_gemm = _gemm_mn(A, B, b_kn)
        if A_idx is not None:
            m_gemm = A_idx.shape[0]
        epi_args = {}
        for name in epi_names:
            v = epi_flat[name]
            dim = sink_dims.get(name)
            if dim == 0 and isinstance(v, torch.Tensor):
                v = v[..., : _cdiv(n_gemm, c.tile_n)]
            elif dim == 1 and isinstance(v, torch.Tensor):
                v = v[..., : _cdiv(m_gemm, c.tile_m), :]
            epi_args[name] = v
        dyn = c.is_dynamic_persistent or dynamic_scheduler
        # SM90 dynamic-persistent scheduling consumes a semaphore; a fresh
        # zeros(1) per call is the gemm_interface pattern (under the CUDA-graph
        # bench the captured memset re-zeros it on every replay).
        sem = None
        if dyn and get_device_capacity(A.device)[0] == 9:
            sem = torch.zeros(1, dtype=torch.int32, device=A.device)
        B_pass, bkn_pass = B, b_kn
        if c.swap_ab and not b_kn:
            B_pass, bkn_pass = B.mT, True  # swap_ab requires B given (k, n)
        try:
            return mod.gemm(
                A,
                B_pass,
                D,
                C,
                epi_args=epi_args,
                tile_M=c.tile_m,
                tile_N=c.tile_n,
                tile_K=None if SFA is not None else c.tile_k,
                cluster_M=c.cluster_m,
                cluster_N=c.cluster_n,
                pingpong=c.pingpong,
                is_dynamic_persistent=dyn,
                max_swizzle_size=c.max_swizzle_size,
                tile_count_semaphore=sem,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx,
                SFA=SFA,
                SFB=SFB,
                concat_layout=concat_layout,
                b_kn=b_kn,
                swap_ab=c.swap_ab,
            )
        except (ValueError, TypeError, AssertionError) as e:
            # The bench loop only maps RuntimeError/MemoryError to an inf
            # timing; a config the prune missed must not abort the sweep.
            raise RuntimeError(f"config {c} rejected: {e}") from e

    fn.__name__ = f"mod_{mod._ident}"
    kw = inspect.Parameter.KEYWORD_ONLY
    params = [
        inspect.Parameter(n, kw, default=None)
        for n in ("A", "B", "D", "C", "mod_digest", "cu_seqlens_m", "A_idx", "SFA", "SFB", "config")
    ]
    params.append(inspect.Parameter("b_kn", kw, default=False))
    params.append(inspect.Parameter("dynamic_scheduler", kw, default=False))
    params.append(inspect.Parameter("concat_layout", kw, default=None))
    params.extend(inspect.Parameter(n, kw, default=None) for n in epi_names)
    fn.__signature__ = inspect.Signature(params)
    return fn


_MOD_TUNERS: dict = {}


def _get_tuner(mod, epi_names, has_c, device):
    key = (mod.semantic_digest, epi_names, has_c, get_device_capacity(device)[0])
    tuner = _MOD_TUNERS.get(key)
    if tuner is None:
        tuner = Autotuner(
            _make_tuned_fn(mod, epi_names),
            key=["mod_digest", "b_kn", "dynamic_scheduler", "concat_layout"],
            configs=[AutotuneConfig(config=c) for c in _config_space(mod, device)],
            prune_configs_by={"early_config_prune": partial(_prune_for_mod, mod)},
            cache_results=True,
        )
        _MOD_TUNERS[key] = tuner
    return tuner


def tuned_mod_gemm(
    mod,
    A,
    B,
    D,
    C=None,
    *,
    epi_args,
    b_kn=False,
    cu_seqlens_m=None,
    A_idx=None,
    dynamic_scheduler=False,
    SFA=None,
    SFB=None,
    concat_layout=None,
):
    """Autotuned ``mod.gemm``: sweep the arch's config space on the first call
    per (mod, tensor metadata), then run the winner (warm calls replay through
    mod.gemm's own plan cache). Reduce-sink buffers in ``epi_args`` must be
    allocated at the sweep's worst case — see ``sink_arg_shapes``. Returns
    TunedModGemm(plan, config, sinks) with the winning config's sink views."""
    epi_names = tuple(sorted(epi_args))
    tuner = _get_tuner(mod, epi_names, C is not None, A.device)
    plan = tuner(
        A=A,
        B=B,
        D=D,
        C=C,
        mod_digest=mod.semantic_digest,
        b_kn=b_kn,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        SFA=SFA,
        SFB=SFB,
        concat_layout=concat_layout,
        **epi_args,
    )
    best = tuner.best_config.kwargs["config"]
    m_gemm, n_gemm = _gemm_mn(A, B, b_kn)
    if A_idx is not None:
        m_gemm = A_idx.shape[0]
    return TunedModGemm(plan, best, _slice_sinks(mod, epi_args, best, m_gemm, n_gemm))
