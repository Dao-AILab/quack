# Copyright (c) 2026, Tri Dao.
"""The ``transform_a=`` handle protocol: what every A-operand transform mod
(a value fn, a packed decode format, dropout) provides to the generic host
layers. Sits below both the fn frontend and the decode formats, which
subclass it; imports only the identity leaf, the rest lazily."""

from typing import NamedTuple

from quack.gemm_runtime.identity import GemmMod

__all__ = ["ResolvedOperands", "TransformModBase"]


class ResolvedOperands(NamedTuple):
    """EpiMod.gemm's slot resolution — ONE statement of the transform call
    contract: what crosses in the kernel's mA/mB slots at launch, the plain
    activations tensor for metadata/validation, the runtime-operand view
    (its metadata is a plan-key slot), the resolved tile_K, and the padded
    weight N for layout-owning transforms (None: derive N from B as
    usual)."""

    a_slot: object
    b_slot: object
    A: object
    arg_sf: object = None
    tile_k: object = None
    n_gemm: object = None


class TransformModBase(GemmMod):
    """Shared ``transform_a=`` handle protocol. The generic layers
    (gemm_runtime.host, EpiMod.gemm/__call__/plan, gemm_runtime.torch_op)
    never enumerate
    transform flavours: they normalize any handle to a mod
    (:func:`quack.operand_transform.host.as_transform_mod`) and call these
    methods — the layout-owning vs value branch below is its single
    statement."""

    args = ()
    consts = None
    regs = None

    @property
    def owned_fmt(self):
        """DecodeFormat when this transform owns A's layout (packed
        weights); None for value transforms (A stays a plain (M, K)
        operand)."""
        return None

    @property
    def needs_operands(self):
        """True when calls must carry runtime operand tensors (``args``)."""
        return bool(self.args)

    def plan_key(self):
        """Cheap plan-cache identity (the digest is precomputed)."""
        return ("mod", self.semantic_digest)

    def config_ok(self, cfg):
        """Cheap autotune prune: can this GemmConfig possibly run this
        transform? Geometry/kernel asserts still guard correctness (a
        mispruned config fails host validation and benches as inf) — this
        only avoids wasted compile attempts."""
        if cfg.swap_ab:
            return False  # transform_a + swap_ab is rejected at the call
        fmt = self.owned_fmt
        if fmt is None:
            return True
        return (
            not cfg.pingpong
            and cfg.cluster_m == 1
            and cfg.tile_m % 64 == 0
            and cfg.tile_k in (None, fmt.tile_k)
        )

    def default_config(self, A, B):
        """Transform-specific default GemmConfig, or None for the caller's
        per-arch default. Layout-owning formats get the measured W4
        coverage rule."""
        fmt = self.owned_fmt
        if fmt is None:
            return None
        from quack.cute_dsl_utils import get_device_capacity
        from quack.gemm_config import GemmConfig
        from quack.operand_transform.host import pick_w4_cfg, pick_w4a8_cfg

        if fmt.promote:
            tm, tn, _sk = pick_w4a8_cfg(A.shape[-2], self.padded_n(B))
        else:
            # W4 runs atom_n == 1 on every arch (SM120's (4,1,1)/(8,1,1)
            # decode layouts included): tile_n floor is the 16-wide warp span
            tm, tn, _sk = pick_w4_cfg(
                A.shape[-2],
                self.padded_n(B),
                A.shape[-1] // fmt.tile_k,
                sm120=get_device_capacity(A.device)[0] == 12,
                device=A.device,
            )
        return GemmConfig(
            tile_m=tm,
            tile_n=tn,
            cluster_m=1,
            cluster_n=1,
            pingpong=False,
            is_dynamic_persistent=False,
        )

    def padded_n(self, B):
        """Padded weight N when B is this transform's repacked blob; None
        when B is a plain operand."""
        if self.owned_fmt is None:
            return None
        from quack.operand_transform.host import w4_padded_n

        return w4_padded_n(B)

    def resolve_operands(self, A, B, transform_sf, tile_m, tile_k) -> ResolvedOperands:
        fmt = self.owned_fmt
        if fmt is None:
            if transform_sf is not None:
                raise ValueError("transform_sf without a layout-owning transform_a")
            if not self.args:
                return ResolvedOperands(A, B, A, tile_k=tile_k)
            from quack.operand_transform.transform import TransformAOperand

            if not isinstance(A, TransformAOperand):
                raise ValueError(
                    "transform_a with runtime operands: pass A as "
                    "transform_a_operand(mod, A, values, tile_M, tile_K)"
                )
            # blob = plain (m, k) A for keys/validation, sf = the operand
            # view (must have been built with THIS call's tile_M/tile_K; a
            # mismatch fails at trace against the kind's fake).
            return ResolvedOperands(A, B, A.blob, arg_sf=A.sf, tile_k=tile_k)
        # Layout-owning: caller A = activations, caller B = the repacked blob.
        from quack.operand_transform.host import w4_operand_views

        if A.ndim != 2 or A.stride(-1) != 1:
            raise ValueError("layout-owning transform_a: activations are (m, k), k-major")
        if tile_k is not None and tile_k != fmt.tile_k:
            raise ValueError(f"format {fmt.name!r} requires tile_K={fmt.tile_k}")
        if B.shape[1] * fmt.tile_k != A.shape[-1]:
            raise ValueError(
                f"K mismatch: activations K={A.shape[-1]}, blob K={B.shape[1] * fmt.tile_k}"
            )
        n_full = self.padded_n(B)
        if n_full % tile_m:
            raise ValueError(
                f"padded N ({n_full}) must be divisible by tile_M ({tile_m}): the blob's"
                " gmem view tiles kernel-M in whole CTA tiles"
            )
        bundle = w4_operand_views(fmt, B, transform_sf, tile_m)
        return ResolvedOperands(bundle, A, A, tile_k=fmt.tile_k, n_gemm=n_full)

    def bundle(self, A, operand_values, tile_m, tile_k):
        """TransformAOperand from RAW operand tensors (the __call__/plan
        convention; ``tile_k`` None resolves to the 16-bit kernel
        default)."""
        from quack.operand_transform.host import transform_a_operand

        return transform_a_operand(
            self, A, operand_values, tile_m, 64 if tile_k is None else tile_k
        )

    def fake_bundle(self, mA_fake, a_dtype, tile_m, tile_k):
        """Trace-time twin of :meth:`bundle` (same tile_k fallback; the
        kind's smem box shape is derived from the real tile_K, so a
        mismatch fails at trace)."""
        from quack.operand_transform.host import transform_a_fake_operand

        return transform_a_fake_operand(
            self, mA_fake, a_dtype, tile_m, 64 if tile_k is None else tile_k
        )

    def compile_dims(self, bundle):
        """(n_full, k) static problem geometry from a layout-owning bundle's
        blob view (256, wpt, tm64, Gt, Kt, 1); None for value transforms
        (their dims are symbolic)."""
        fmt = self.owned_fmt
        if fmt is None:
            return None
        blob = bundle.blob
        return (blob.shape[2] * blob.shape[3] * 64, blob.shape[4] * fmt.tile_k)

    def fake_operands(self, n_full, k, tile_m):
        """Trace-time fake bundle for a layout-owning transform."""
        from quack.operand_transform.host import w4_fake_operands

        return w4_fake_operands(self.owned_fmt, n_full, k, tile_m)
