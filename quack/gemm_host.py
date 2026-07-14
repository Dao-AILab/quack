# Copyright (c) 2026, Han Guo, Tri Dao.
"""Generic host-side plan/compile/launch layer for epilogue GEMM variants.

The per-variant host plumbing (fake-tensor construction, jit_cache'd compile
wrapper, plan NamedTuple + cache key + build/run pair) used to be ~400
near-identical lines per variant file. This module replaces it with one
generic implementation driven by the variant's EpiOp schema:

* each EpiOp describes its own host argument via ``host_arg_key`` (torch value
  -> picklable descriptor), ``host_fake_arg`` (descriptor -> fake trace-time
  tensor/scalar), and ``host_call_arg`` (torch value -> runtime argument);
* a reconstructable ``GemmClassRef`` is the jit_cache key component — static
  classes resolve by module+qualname, while dynamic epilogue classes resolve
  through a module-global EpiMod and mint locally in async workers.

A variant file keeps only: its mixin(s), the per-SM class stampings, its
validation asserts, and thin ``gemm_X`` / ``run_gemm_X_plan`` wrappers that
map the public signature onto ``epi_values`` dicts.
"""

from __future__ import annotations

import importlib
from typing import NamedTuple, Optional


from quack.cache import jit_cache
from quack.cache.async_compile import PoolPayload
from quack.cute_dsl_utils import get_max_active_clusters, torch2cute_dtype_map
from quack.gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    launch_gemm,
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_sf_tensor,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
    plan_scheduler_args,
    tensor_key,
)


class FakeArgCtx(NamedTuple):
    """Shared symbolic dims + flags handed to EpiOp.host_fake_arg.

    ``swapped`` (swap-at-trace): m/n are KERNEL dims (m = caller n); tile-
    shaped args cross the boundary caller-oriented, i.e. (l, n, m) in kernel
    labels, and are transposed at trace time (GemmBase.cd_transposed)."""

    m: object
    n: object
    k: object
    l: object  # noqa: E741
    batched: bool
    varlen_m: bool
    swapped: bool = False


class GemmClassRef(NamedTuple):
    """Picklable recipe for resolving a GEMM class in async workers.

    Dynamic epilogue classes must never cross the cache boundary directly:
    their module registration exists only in the creating process. Instead an
    epi_mod reference imports the module-global EpiMod and asks it to mint the
    same class from a semantic digest plus the runtime kind signature.

    ``epi_mod_local`` covers EpiMods with no importable anchor (defined in
    ``__main__`` — scripts, notebooks — or never bound to a module global):
    the semantic digest still keys the disk cache correctly and resolution
    goes through a process-local registry. To reach async workers, the ref
    ships the EpiMod by value as a side-channel payload (cloudpickle, see
    ``__quack_pool_payload__``) — the payload never enters the cache key, so
    shas stay deterministic. If the payload can't be serialized the pool
    refuses the key and the cold miss compiles in-process.
    """

    kind: str  # "static", "epi_mod", or "epi_mod_local"
    module: str
    qualname: str
    mint_key: tuple = ()
    semantic_digest: str = ""

    def __quack_pool_payload__(self):
        """Worker setup for a local EpiMod, or None for importable refs."""
        if self.kind != "epi_mod_local":
            return None
        import cloudpickle

        payload = cloudpickle.dumps(_LOCAL_EPI_MODS[self.semantic_digest])
        return PoolPayload(
            "quack.gemm_host",
            "install_epi_mod_payload",
            self.semantic_digest,
            payload,
        )


# semantic_digest -> EpiMod, for refs with no importable module anchor.
# Populated by EpiMod._class_ref before the compile that needs it (and by
# install_epi_mod_payload in async workers). Cold compile resolution consumes
# entries so long-lived workers do not retain user closures.
_LOCAL_EPI_MODS: dict[str, object] = {}


def register_local_epi_mod(digest: str, epi_mod) -> None:
    _LOCAL_EPI_MODS[digest] = epi_mod


def install_epi_mod_payload(expected_digest: str, data: bytes) -> None:
    """Worker-side installer for ``epi_mod_local`` payloads (see
    ``GemmClassRef.__quack_pool_payload__``)."""
    import cloudpickle

    epi_mod = cloudpickle.loads(data)
    if epi_mod.semantic_digest != expected_digest:
        raise ValueError(
            "local epilogue payload digest mismatch: "
            f"expected {expected_digest}, got {epi_mod.semantic_digest}"
        )
    register_local_epi_mod(expected_digest, epi_mod)


def _resolve_qualname(obj, qualname):
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def static_gemm_class_ref(GemmCls):
    return GemmClassRef("static", GemmCls.__module__, GemmCls.__qualname__)


def resolve_gemm_class(ref: GemmClassRef):
    if ref.kind == "epi_mod_local":
        # Consume the registration: worker payloads may close over sizeable
        # Python state and workers live for the whole test/autotune session.
        obj = _LOCAL_EPI_MODS.pop(ref.semantic_digest, None)
        if obj is None:
            raise RuntimeError(
                "process-local epilogue reference is not registered here (created in "
                "another process and its payload was not installed); bind the "
                "@gemm_epilogue object to a module-global name in an importable module "
                "to make it resolvable by import"
            )
        return obj._mint(*ref.mint_key)
    module = importlib.import_module(ref.module)
    obj = _resolve_qualname(module, ref.qualname)
    if ref.kind == "static":
        return obj
    if ref.kind != "epi_mod":
        raise ValueError(f"unknown GEMM class reference kind {ref.kind!r}")
    if obj.semantic_digest != ref.semantic_digest:
        raise RuntimeError(
            f"epilogue {ref.module}.{ref.qualname} changed while resolving a compile request"
        )
    return obj._mint(*ref.mint_key)


def _ops_by_name(GemmCls):
    return {op.name: op for op in GemmCls._epi_ops}


@jit_cache
def _compile_gemm_epi(
    gemm_cls_ref,
    device_capacity,
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    varlen_m,
    gather_A,
    batched,
    b_kn,
    epi_keys,  # ((op_name, op.host_arg_key(value)), ...) — name-sorted
    swap_ab=False,
    use_tma_gather=False,
    concat_layout=(),
    sf_dtype=None,
    sf_vec_size=None,
    sf_batched=True,
    post_init_attrs=(),  # ((attr, value), ...) setattr'd on the gemm object pre-trace
    packed_cd=None,  # "n" | "m": raw 16-bit D/C, f32-recast at trace (dgated)
):
    """Compile one epilogue-GEMM variant against fake symbolic tensors.

    Every argument is a picklable primitive (jit_cache pickles the tuple for
    the disk key and to ship cold misses to async-compile workers).
    """
    GemmCls = resolve_gemm_class(gemm_cls_ref)
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        varlen_m=varlen_m,
        gather_A=gather_A,
        batched=batched,
        b_kn=b_kn,
        swap_ab=swap_ab,
        packed_cd=packed_cd,
    )
    fctx = FakeArgCtx(m, n, k, l, batched, varlen_m, swap_ab)
    ops = _ops_by_name(GemmCls)
    fields = {}
    for name, key in epi_keys:
        fake = ops[name].host_fake_arg(key, fctx)
        if fake is not None:
            fields[name] = fake
    epi_args = GemmCls.EpilogueArguments(**fields)

    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    varlen_args = make_fake_varlen_args(varlen_m, False, gather_A, m if varlen_m else None)
    mSFA = make_fake_sf_tensor(sf_dtype, l if sf_batched else None) if sf_dtype else None
    mSFB = make_fake_sf_tensor(sf_dtype, l if sf_batched else None) if sf_dtype else None
    post_init = None
    if post_init_attrs:

        def post_init(gemm_obj):
            for attr, value in post_init_attrs:
                setattr(gemm_obj, attr, value)

    return compile_gemm_kernel(
        GemmCls,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        gather_A,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        post_init=post_init,
        mSFA=mSFA,
        mSFB=mSFB,
        use_tma_gather=use_tma_gather,
        concat_layout=concat_layout or None,
        sf_vec_size=sf_vec_size,
        b_transposed=b_kn,
        a_transposed=swap_ab,
        cd_transposed=swap_ab,
        cd_packed=packed_cd,
    )


class GemmEpiPlan(NamedTuple):
    """Launch plan derived purely from tensor metadata and config flags.

    Cached by the variant wrapper per metadata key, so warm calls skip
    validation, major/dtype derivation, and the compile-cache lookup.
    ``epi_arg_keys`` replays each op's compile-time descriptor at launch
    (host_call_arg needs e.g. the scalar mode); ``gemm_cls`` carries the op
    schema and EpilogueArguments type into run_gemm_epi_plan.
    """

    compiled_fn: object
    gemm_cls: type
    is_sm100_family: bool  # SM100/110 take trailing (SFA, SFB) args
    max_active_clusters: int
    max_swizzle_size: int
    scheduler_uses_semaphore: bool  # only the SM90 dynamic scheduler consumes the semaphore
    scheduler_static: Optional[object]  # TileSchedulerOptions when it has no per-call values
    epi_arg_keys: tuple  # ((op_name, key), ...) as compiled
    # Launch-overhead precomputation (host hot path): (name, converter, key)
    # triples — converter is the op's bound ``host_call_arg``, or None when the
    # op inherits the identity default (the value passes straight through) —
    # plus an all-None EpilogueArguments field template in field order, so warm
    # calls do a dict .copy() + positional ``_make`` instead of rebuilding both
    # dicts and parsing kwargs.
    call_ops: tuple = ()
    arg_template: dict = {}


def _get_major(t, m_label, n_label):
    return n_label if t.stride(-1) == 1 else m_label


def build_gemm_epi_plan(
    GemmCls,
    device_capacity,
    A,
    B,
    D,
    C,
    *,
    epi_values,  # {op_name: torch value or scalar}; missing/None = op inactive
    epi_key_overrides=None,  # {op_name: key} when the wrapper owns the key rule (scalar modes)
    tile_M,
    tile_N,
    cluster_M,
    cluster_N,
    tile_K=None,
    pingpong=False,
    persistent=True,
    is_dynamic_persistent=False,
    max_swizzle_size=8,
    varlen_m=False,
    gather_A=False,
    b_kn=False,
    swap_ab=False,  # swap-at-trace: slot tensors in, caller-oriented D/C
    use_tma_gather=False,
    concat_layout=(),
    sf_dtype=None,
    sf_vec_size=None,
    sf_batched=True,
    post_init_attrs=(),
    gemm_cls_ref=None,
    packed_cd=None,  # "n" | "m": D/C passed RAW 16-bit, f32-recast at trace (dgated)
) -> GemmEpiPlan:
    """Derive majors/dtypes/epi keys from tensor metadata and compile (or hit
    the jit cache). Variant wrappers call this after their validation asserts."""
    batched = A.ndim == 3 or varlen_m
    a_major = _get_major(A, "m", "k")
    b_major = _get_major(B, "n", "k")
    if b_kn:
        # Majors are logical (n, k) labels: with B stored (k, n), a contiguous
        # last dim means n-major.
        b_major = "n" if B.stride(-1) == 1 else "k"
    d_major = _get_major(D, "m", "n") if D is not None else None
    c_major = _get_major(C, "m", "n") if C is not None else None
    if swap_ab:
        # Slot tensors: A-slot = caller B (k, n) native (a_transposed relabels
        # at trace) — kernel-A is (m_k, k) with m_k the caller n, so the label
        # flips vs the standard derivation. B-slot = caller A (m, k) is
        # already kernel-ordered (n_k, k): the standard formula holds. D/C
        # cross caller-oriented, so their kernel labels flip like A's.
        a_major = "m" if A.stride(-1) == 1 else "k"
        d_major = ("m" if D.stride(-1) == 1 else "n") if D is not None else None
        c_major = ("m" if C.stride(-1) == 1 else "n") if C is not None else None
    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype] if D is not None else None
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None

    ops = _ops_by_name(GemmCls)
    overrides = epi_key_overrides or {}
    epi_keys = []
    for name, op in ops.items():
        key = overrides[name] if name in overrides else op.host_arg_key(epi_values.get(name))
        if key is not None:
            epi_keys.append((name, key))
    epi_keys = tuple(sorted(epi_keys, key=lambda nk: nk[0]))

    if gemm_cls_ref is None:
        gemm_cls_ref = static_gemm_class_ref(GemmCls)
    compiled_fn = _compile_gemm_epi(
        gemm_cls_ref,
        device_capacity,
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        (tile_M, tile_N, tile_K) if tile_K is not None else (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        varlen_m,
        gather_A,
        batched,
        b_kn,
        epi_keys,
        swap_ab=swap_ab,
        use_tma_gather=use_tma_gather,
        concat_layout=concat_layout,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        sf_batched=sf_batched,
        post_init_attrs=post_init_attrs,
        packed_cd=packed_cd,
    )

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    # Must mirror make_fake_scheduler_args above: only the SM90 dynamic
    # scheduler consumes the semaphore, so it's the only non-static case.
    scheduler_uses_semaphore = is_dynamic_persistent and device_capacity[0] == 9
    scheduler_static = (
        make_scheduler_args(max_active_clusters, max_swizzle_size, None)
        if not scheduler_uses_semaphore
        else None
    )
    from quack.epi_ops import EpiOp

    plan_ops = _ops_by_name(GemmCls)
    call_ops = tuple(
        (
            name,
            None
            if type(plan_ops[name]).host_call_arg is EpiOp.host_call_arg
            else plan_ops[name].host_call_arg,
            key,
        )
        for name, key in epi_keys
    )
    return GemmEpiPlan(
        compiled_fn=compiled_fn,
        gemm_cls=GemmCls,
        call_ops=call_ops,
        arg_template={name: None for name in GemmCls.EpilogueArguments._fields},
        is_sm100_family=device_capacity[0] in [10, 11],
        max_active_clusters=max_active_clusters,
        max_swizzle_size=max_swizzle_size,
        scheduler_uses_semaphore=scheduler_uses_semaphore,
        scheduler_static=scheduler_static,
        epi_arg_keys=epi_keys,
    )


def run_gemm_epi_plan(
    plan: GemmEpiPlan,
    A,
    B,
    D,
    C,
    epi_values,
    *,
    tile_count_semaphore=None,
    cu_seqlens_m=None,
    cu_seqlens_k=None,
    A_idx=None,
    SFA=None,
    SFB=None,
) -> None:
    """Launch a resolved plan: only per-call pointers and scalar values here.

    The tensors must match the metadata the plan was built from (the variant
    wrapper guarantees that via its plan-cache key). Constexpr fields are
    passed None — they are baked into the compiled kernel.
    """
    # arg_template preserves EpilogueArguments field order, so the values()
    # view feeds _make positionally (no kwargs parsing).
    fields = plan.arg_template.copy()
    for name, convert, key in plan.call_ops:
        value = epi_values.get(name)
        if convert is not None:
            value = convert(value, key)
        if value is not None:
            fields[name] = value
    epi_args = plan.gemm_cls.EpilogueArguments._make(fields.values())
    scheduler_args = plan_scheduler_args(plan, tile_count_semaphore)
    varlen_args = make_varlen_args(cu_seqlens_m, cu_seqlens_k, A_idx)
    launch_gemm(plan, A, B, D, C, epi_args, scheduler_args, varlen_args, SFA, SFB)


def gemm_epi_plan_key(A, B, D, C, epi_values, epi_key_overrides=None, *config) -> tuple:
    """Standard plan-cache key: full tensor metadata for the operands and every
    epilogue tensor (shapes and strides subsume the majors and the validation
    asserts), scalar-mode overrides for scalar epi args, plus the config tail.
    A cache hit is exactly a replay of a previously validated call with
    different data pointers."""
    epi_meta = tuple(
        (name, tensor_key(v) if hasattr(v, "stride") else v is not None)
        for name, v in sorted(epi_values.items(), key=lambda nv: nv[0])
    )
    overrides = tuple(sorted((epi_key_overrides or {}).items()))
    return (
        tensor_key(A),
        tensor_key(B),
        tensor_key(D),
        tensor_key(C),
        epi_meta,
        overrides,
        *config,
    )
