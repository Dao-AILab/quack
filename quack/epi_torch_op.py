# Copyright (c) 2026, Tri Dao.
"""The single torch custom op for all epilogue-GEMM objects (HANDOFF Tier 4).

``quack::gemm_epi(str digest, Tensor[] ins, Tensor(a!)[] outs, str meta)``:
one mutating op + one no-op fake covers every epilogue — including
user-defined ``@gemm_epilogue`` fns, which otherwise have no torch.compile
story (precedent: torch.compile's own triton_kernel_wrapper_mutation).
``digest`` resolves the epilogue through an in-process registry, falling back
to import by the module locator carried in ``meta`` (so compiled artifacts
survive process boundaries when the epilogue is bound to an importable name —
the same constraint the async-compile pool imposes).

``ins`` is a packed list of the non-None input tensors, named positionally by
``meta['ins_names']``; ``outs`` carries D + declared outputs + reduce-partial
buffers (all graph-owned, since the op only mutates). Host constants (config,
flags, scalar operands) ride ``meta`` as a repr'd dict — Dynamo guards on the
string, which is exactly right since they select compiled behavior.

Reduce sinks under torch.compile: the wrapper pins the config (the partial
buffers must be graph-allocated at exact shapes BEFORE the op runs, so
runtime autotuning inside the op cannot pick a different tiling) and
finalizes the partials with traced torch ops.
"""

from __future__ import annotations

import ast
from typing import Optional

import torch

from quack.gemm_config import GemmConfig, default_config
from quack.rounding import RoundingMode

# digest -> EpiMod, populated by the compile-path wrapper (same process) or
# lazily by import through the meta locator.
_EPI_REGISTRY: dict = {}


def _resolve(digest: str, locator):
    mod = _EPI_REGISTRY.get(digest)
    if mod is None and locator:
        import importlib

        module = importlib.import_module(locator[0])
        mod = getattr(module, locator[1])
        if mod.semantic_digest != digest:
            raise RuntimeError(
                f"epilogue {locator[0]}.{locator[1]} changed since this graph was compiled"
            )
        _EPI_REGISTRY[digest] = mod
    if mod is None:
        raise RuntimeError(
            "epilogue digest not resolvable in this process; bind the @gemm_epilogue "
            "object to a module-global name in an importable module"
        )
    return mod


@torch.library.custom_op("quack::gemm_epi", mutates_args={"outs"}, device_types="cuda")
def _gemm_epi(digest: str, ins: list[torch.Tensor], outs: list[torch.Tensor], meta: str) -> None:
    m = ast.literal_eval(meta)
    mod = _resolve(digest, m["locator"])
    named = dict(zip(m["ins_names"], ins))
    operands = {k[4:]: v for k, v in named.items() if k.startswith("op__")}
    operands.update(m["scalar_ops"])
    i = 0
    out = {}
    if m["store_d"]:
        out["D"] = outs[0]
        i = 1
    for name in m["out_names"]:
        out[name] = outs[i]
        i += 1
    for name in m["sink_names"]:  # exact-shape partials: finalized by the wrapper
        operands[name] = outs[i]
        i += 1
    cfg = GemmConfig(**m["config"]) if m["config"] is not None else None
    mod(
        named["A"],
        named["B"],
        named.get("C"),
        out=out,
        store_d=m["store_d"],
        config=cfg,
        tuned=m["tuned"],
        cu_seqlens_m=named.get("cu_seqlens_m"),
        A_idx=named.get("A_idx"),
        SFA=named.get("SFA"),
        SFB=named.get("SFB"),
        rounding_mode=m["rounding_mode"],
        **operands,
    )


@_gemm_epi.register_fake
def _gemm_epi_fake(digest, ins, outs, meta) -> None:
    # Pure no-op: the op only mutates ``outs``; compilation is owned by
    # jit_cache + the async pool at real execution time.
    return


def compile_call(
    mod,
    A,
    B,
    C,
    *,
    out,
    out_dtype,
    store_d,
    config,
    tuned,
    cu_seqlens_m,
    A_idx,
    SFA,
    SFB,
    rounding_mode,
    operands,
):
    """torch.compile-path body of ``EpiMod.__call__``: allocate graph-owned
    outputs + exact-shape reduce partials, record one ``quack::gemm_epi`` call,
    finalize reduces with traced ops. Returns the same dict as eager."""
    out = mod._alloc_outputs(out, A, B, C, store_d, out_dtype, cu_seqlens_m, A_idx)
    lead = mod._lead_shape(A, cu_seqlens_m, A_idx)
    n = B.shape[-1]
    cfg: Optional[GemmConfig] = config
    if mod.sinks and cfg is None:
        # Partials are graph-allocated before the op runs, so the tiling must
        # be fixed here — no runtime autotune under compile with sinks.
        cfg = default_config(A.device)
        tuned = False
    partials = {}
    for name, op in mod.sinks.items():
        if operands.get(name) is not None:
            continue  # caller-provided partial buffer: returned raw
        if op.dim == 0:
            shape = (*lead, -(-n // cfg.tile_n))
        else:
            shape = (*lead[:-1], -(-lead[-1] // cfg.tile_m), n)
        partials[name] = torch.empty(shape, dtype=torch.float32, device=A.device)

    ins_names, ins = [], []
    for name, t in (
        ("A", A),
        ("B", B),
        ("C", C),
        ("cu_seqlens_m", cu_seqlens_m),
        ("A_idx", A_idx),
        ("SFA", SFA),
        ("SFB", SFB),
        *((f"op__{k}", v) for k, v in operands.items() if isinstance(v, torch.Tensor)),
    ):
        if t is not None:
            ins_names.append(name)
            ins.append(t)
    scalar_ops = {k: v for k, v in operands.items() if not isinstance(v, torch.Tensor)}
    outs = []
    if store_d:
        outs.append(out["D"])
    outs.extend(out[name] for name in mod.outputs)
    outs.extend(partials.values())
    meta = repr(
        dict(
            ins_names=tuple(ins_names),
            out_names=tuple(mod.outputs),
            sink_names=tuple(partials),
            store_d=bool(store_d),
            tuned=bool(tuned),
            config=None if cfg is None else cfg.__dict__,
            rounding_mode=int(rounding_mode),
            scalar_ops=scalar_ops,
            locator=mod._module_locator(),
        )
    )
    _EPI_REGISTRY[mod.semantic_digest] = mod
    torch.ops.quack.gemm_epi(mod.semantic_digest, ins, outs, meta)
    result = dict(out) if store_d else {k: v for k, v in out.items() if k != "D"}
    for name, buf in partials.items():
        finalize = getattr(mod.sinks[name], "host_finalize", None)
        result[name] = finalize(buf) if finalize is not None else buf
    return result


_DEFAULT_RN = RoundingMode.RN  # re-export convenience for the __call__ branch
