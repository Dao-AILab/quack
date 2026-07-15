# Copyright (c) 2026, Tri Dao.
"""Block-scaled operand container and format descriptors.

Design doc: AI/blockscaled_api.md. In short:

- ``BlockScaledFormat`` is a frozen descriptor (storage dtype, element bits, scale
  dtype, scale vector size, ...) and the single source of truth for format
  properties; nothing may re-derive them from tensor dtypes.
- ``BlockScaledOperand`` is a plain frozen dataclass holding ``qdata`` (quantized
  values), ``scale`` (128x4-blocked scale factors), the format, and (NVFP4 only)
  an optional per-tensor scale. It is deliberately NOT a torch.Tensor subclass:
  quack GEMM entry points are its only consumers, and the simplest container that
  keeps (qdata, scale, format, pts) atomic wins. It exposes an explicit, honest
  surface (``shape``/``dtype``/``mT``/``to``/``dequantize``) with no aten
  interception and no dequantize fallback - torch ops on it fail loudly with a
  TypeError.
- Transpose (``.mT`` / ``.T`` / ``transpose``) is a qdata stride-swap view;
  ``scale`` is carried unchanged - it blocks along K in either orientation. The
  blocked scale atom is not view-transposable.
- ``quant_dim`` records which logical dim the scale vector runs along (-1 or
  -2) - the analogue of CUTLASS's ``UMMA::Major`` on its SF atoms. Storage alone
  cannot express it: a square fp8 canonical operand and its ``.mT`` view are
  byte-identical. fp4 packing pins it to the packed (unit-stride) dim; byte
  formats default to the last dim (quantize's convention); transpose flips it.
  Consumed as a GEMM operand, the quantized axis must be the contraction axis
  (the interface enforces this per operand slot).

This module must stay import-light (torch only at module level): it is imported
by ``quack.gemm_interface`` at module level and by the kernel-layer modules
lazily.
"""

import copy as _copy
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
import torch.utils._pytree as _pytree

from quack.blockscaled.quantize import (
    QUANTIZERS,
    check_blocked_scale_atom,
    pack_scale_2d_to_blocked_contig,
)

__all__ = [
    "BlockScaledFormat",
    "BlockScaledOperand",
    "BLOCKSCALED_FORMAT_REGISTRY",
    "mma_kind_for_pair",
    "MXFP8_E4M3",
    "MXFP8_E5M2",
    "MXFP4",
    "MXFP4_BYTE",
    "NVFP4",
    "MXFP6_E2M3",
    "MXFP6_E3M2",
]


@dataclass(frozen=True)
class BlockScaledFormat:
    """Descriptor for a block-scaled quantization format.

    Frozen (hashable, picklable) so instances can serve as dynamo guard context and
    kernel-cache key material; only ``name`` crosses torch.library op schemas.
    """

    name: str
    qdata_dtype: torch.dtype
    # CuTe-DSL MMA element type (may differ from storage: fp6). None marks a format
    # with no DSL element type at all — host-side complete (quantize/dequantize/
    # serialize) but consumable by a kernel only once that kernel declares its own
    # (copy dtype, MMA dtype, convert) triple. See AI/blockscaled_recipes.md.
    cutlass_dtype_name: Optional[str]
    elem_bits: int
    elems_per_container: int  # logical elements per qdata element (2 for fp4x2)
    scale_dtype: torch.dtype
    sf_vec_size: int  # logical K elements per scale factor (== min logical-K divisibility)
    has_per_tensor_scale: bool = False

    def to_cutlass_dtype(self):
        """The CuTe-DSL element type for the MMA instruction (not the storage/copy
        dtype: MXFP6 stores uint8 byte containers but computes as Float6*)."""
        if self.cutlass_dtype_name is None:
            raise ValueError(f"{self.name} has no CuTe-DSL element type")
        import cutlass  # lazy: keep this module importable without the DSL

        return getattr(cutlass, self.cutlass_dtype_name)

    @classmethod
    def from_name(cls, name: str) -> "BlockScaledFormat":
        key = _LEGACY_FORMAT_NAMES.get(name, name)
        try:
            return BLOCKSCALED_FORMAT_REGISTRY[key]
        except KeyError:
            raise ValueError(
                f"unknown blockscaled format {name!r}; known: {sorted(BLOCKSCALED_FORMAT_REGISTRY)}"
            ) from None

    @classmethod
    def from_cutlass_dtypes(cls, ab_dtype, sf_dtype, sf_vec_size: int) -> "BlockScaledFormat":
        """Identify the format from CuTe-DSL dtypes (replaces the old
        ``_blockscaled_format_of`` if-ladder). NVFP4 vs MXFP4 disambiguates on the
        scale dtype + vec size; fp6 disambiguates on the element type."""
        from quack.cute_dsl_utils import torch2cute_dtype_map  # lazy: needs the DSL

        for fmt in BLOCKSCALED_FORMAT_REGISTRY.values():
            if (
                fmt.cutlass_dtype_name is not None  # DSL-typeless formats can't match
                and fmt.to_cutlass_dtype() == ab_dtype
                and torch2cute_dtype_map[fmt.scale_dtype] == sf_dtype
                and fmt.sf_vec_size == sf_vec_size
            ):
                return fmt
        raise ValueError(
            f"no blockscaled format matches (ab={ab_dtype}, sf={sf_dtype}, vec={sf_vec_size})"
        )


MXFP8_E4M3 = BlockScaledFormat(
    "mxfp8_e4m3", torch.float8_e4m3fn, "Float8E4M3FN", 8, 1, torch.float8_e8m0fnu, 32
)
MXFP8_E5M2 = BlockScaledFormat(
    "mxfp8_e5m2", torch.float8_e5m2, "Float8E5M2", 8, 1, torch.float8_e8m0fnu, 32
)
MXFP4 = BlockScaledFormat(
    "mxfp4", torch.float4_e2m1fn_x2, "Float4E2M1FN", 4, 2, torch.float8_e8m0fnu, 32
)
NVFP4 = BlockScaledFormat(
    "nvfp4",
    torch.float4_e2m1fn_x2,
    "Float4E2M1FN",
    4,
    2,
    torch.float8_e4m3fn,
    16,
    has_per_tensor_scale=True,
)
# MXFP6 qdata is byte-per-element uint8 (one fp6 code in bits [5:0], bits [7:6] zero):
# PTX kind::mxf8f6f4 addresses every element as an 8-bit container in SMEM, matching
# CUDA __nv_fp6_storage_t and cuBLASLt CUDA_R_6F_*. Byte-per-element MXFP6 therefore
# has fp8's bandwidth; its value is accuracy headroom over MXFP4, not speed. Packed
# 6-bit gmem (CUTLASS 16U6 TMA) is a backlog item that only changes descriptor data.
# Byte-container fp4: one E2M1 code per uint8. This is fp4 in the form the
# mixed-capable kind::mxf8f6f4 consumes (the DSL has no TMA sub-byte->container
# expansion, so byte containers live in gmem too). Use packed `mxfp4` for pure
# fp4 x fp4 GEMMs (kind::mxf4, half the bandwidth).
MXFP4_BYTE = BlockScaledFormat(
    "mxfp4_byte", torch.uint8, "Float4E2M1FN", 4, 1, torch.float8_e8m0fnu, 32
)
MXFP6_E2M3 = BlockScaledFormat(
    "mxfp6_e2m3", torch.uint8, "Float6E2M3FN", 6, 1, torch.float8_e8m0fnu, 32
)
MXFP6_E3M2 = BlockScaledFormat(
    "mxfp6_e3m2", torch.uint8, "Float6E3M2FN", 6, 1, torch.float8_e8m0fnu, 32
)

BLOCKSCALED_FORMAT_REGISTRY = {
    fmt.name: fmt
    for fmt in (MXFP8_E4M3, MXFP8_E5M2, MXFP4, MXFP4_BYTE, NVFP4, MXFP6_E2M3, MXFP6_E3M2)
}

# Legacy short names used by blockscaled_quantize / BLOCKSCALED_FORMATS (the
# inverse is derived - this dict is the single source of the aliasing).
_LEGACY_FORMAT_NAMES = {"mxfp8": "mxfp8_e4m3"}
_CANONICAL_TO_LEGACY = {v: k for k, v in _LEGACY_FORMAT_NAMES.items()}


def legacy_format_name(fmt: "BlockScaledFormat") -> str:
    """The pre-descriptor short name ("mxfp8") where one exists, else fmt.name."""
    return _CANONICAL_TO_LEGACY.get(fmt.name, fmt.name)


def mma_kind_for_pair(fmt_a: "BlockScaledFormat", fmt_b: "BlockScaledFormat") -> str:
    """The tcgen05 MMA kind for an (A, B) format pair, or raise ValueError if the
    pair is not representable on the hardware. This is hardware LEGALITY only:
    whether a legal pair is implemented is enforced per-architecture by the
    gemm_smXXX kernel classes (each SM version supports different mx dtype
    combinations; see GemmSm100.is_valid_dtypes_and_scale_factor_vec_size,
    asserted in its blockscaled setup path).

    PTX rules: ``kind::mxf4nvf4`` (e4m3 scales, vec 16) requires fp4 on both
    operands; ``kind::mxf4`` covers both-fp4 with e8m0 scales; ``kind::mxf8f6f4``
    admits any mix of fp8/fp6/fp4 element types (e8m0 scales, vec 32).

    The kind rules are encoded in three places that must stay in sync (this
    function cannot be shared: the kernel layer keys on cutlass dtypes and must
    not import torch-level format descriptors): here (format names), instruction-K
    in ``GemmSm100._blockscaled_mma_inst_k`` (storage dtypes), and the per-arch
    subset in ``GemmSm100.is_valid_dtypes_and_scale_factor_vec_size`` (MMA
    dtypes). ``test_mma_kind_mirrors_kernel_inst_k`` pins the first mirror.
    """
    if fmt_a.name == "nvfp4" or fmt_b.name == "nvfp4":
        if fmt_a.name == fmt_b.name:
            return "mxf4nvf4"
        raise ValueError(
            f"nvfp4 (e4m3 scales, vec 16) cannot pair with "
            f"{fmt_b.name if fmt_a.name == 'nvfp4' else fmt_a.name}: "
            f"the mxf4nvf4 MMA kind requires nvfp4 on both operands"
        )
    if fmt_a.name == "mxfp4" and fmt_b.name == "mxfp4":
        return "mxf4"
    if fmt_a.name == "mxfp4_byte" and fmt_b.name == "mxfp4_byte":
        raise ValueError(
            "mxfp4_byte x mxfp4_byte has no MMA kind: use mxfp4 (packed, kind::mxf4) "
            "for pure fp4 pairs; byte-container fp4 exists for mixed mxf8f6f4 pairs"
        )
    if fmt_a.elems_per_container > 1 or fmt_b.elems_per_container > 1:
        packed = fmt_a.name if fmt_a.elems_per_container > 1 else fmt_b.name
        other = fmt_b.name if fmt_a.elems_per_container > 1 else fmt_a.name
        raise ValueError(
            f"{packed} (packed fp4x2 storage) cannot join a mixed pair with {other}: "
            f"the mixed-capable mxf8f6f4 MMA kind reads sub-byte elements from 8-bit "
            f"SMEM containers, which packed gmem storage cannot feed; requantize as "
            f"mxfp4_byte (byte-container fp4) for mixed pairs "
            f"(AI/blockscaled_api.md section 6)"
        )
    # fp8/fp6 byte-width element types mix freely under mxf8f6f4 (all e8m0 / vec 32).
    # Gate on the MMA element class explicitly: a format without a tcgen05-
    # representable element type (no DSL type, or a non-fp8/6/4 one) must fail
    # HERE, not by falling through to a kind its elements cannot join. Software
    # kinds (blockwise promotion: fp32-scale fp8, one-sided bf16, int4) are a
    # follow-up with their own kind names — see AI/blockscaled_recipes.md.
    for fmt in (fmt_a, fmt_b):
        if _mma_element_class(fmt) is None:
            raise ValueError(
                f"{fmt.name} has no tcgen05 MMA element type "
                f"(cutlass_dtype_name={fmt.cutlass_dtype_name}): no hardware "
                f"blockscaled MMA kind exists for this format"
            )
    return "mxf8f6f4"


def _mma_element_class(fmt: "BlockScaledFormat") -> Optional[str]:
    """The tcgen05 element class ("f8" | "f6" | "f4") of a format's MMA type, or
    None when the format has no hardware-representable element type. Derived from
    the DSL type name: PTX kind legality is a property of the MMA element type."""
    name = fmt.cutlass_dtype_name
    for prefix, cls in (("Float8", "f8"), ("Float6", "f6"), ("Float4", "f4")):
        if name is not None and name.startswith(prefix):
            return cls
    return None


def _coerce_format(format) -> BlockScaledFormat:
    if isinstance(format, BlockScaledFormat):
        return format
    if isinstance(format, str):
        return BlockScaledFormat.from_name(format)
    raise TypeError(f"format must be a BlockScaledFormat or str, got {type(format)}")


def _normalize_quant_dim(dim: int, ndim: int) -> int:
    """Normalize a quantized-dim argument to -1 or -2 (the only two directions a
    block-scaled operand can have - the batch dim is never quantized)."""
    d = dim if dim < 0 else dim - ndim
    if d not in (-1, -2):
        raise ValueError(f"quant_dim must be one of the last two dims, got {dim}")
    return d


def _packed_dim(qdata: torch.Tensor, fmt: BlockScaledFormat) -> int:
    """The qdata dim holding multiple logical elements per storage element.

    Convention: the unit-stride dim. Packing is always along the quantization
    (K) axis, and packed (fp4) formats are required K-major by the kernel, so
    unit-stride == packed == K. Scanned from the last dim so fully-contiguous
    qdata resolves to the innermost dim. Size-1 dims report arbitrary strides
    (a (Kp, 1) K-major operand can present stride 1 on BOTH dims), so dims with
    extent > 1 take precedence; if every unit-stride dim is size-1 the tensor is
    degenerate and any choice is equivalent - the innermost wins.
    """
    fallback = None
    for d in range(qdata.ndim - 1, -1, -1):
        if qdata.stride(d) == 1:
            if qdata.shape[d] != 1:
                return d
            fallback = d if fallback is None else fallback
    if fallback is not None:
        return fallback
    raise ValueError(
        f"qdata has no unit-stride dim: shape={tuple(qdata.shape)}, stride={qdata.stride()}"
    )


@dataclass(frozen=True, eq=False)  # eq=False: tensor fields do not compare with ==
class BlockScaledOperand:
    """Block-scaled quantized operand for quack GEMMs.

    A plain container - not a torch.Tensor. ``shape``/``dtype`` report the
    *logical* view (unpacked shape, original high-precision dtype); storage truth
    is ``qdata``. torch ops do not accept it (loud TypeError); the supported
    surface is quack.gemm* plus the explicit methods here.

    Construction validates mode-independent invariants only. The qdata-shape <->
    scale-shape relation is intentionally NOT checked here: varlen operands use
    padded scale buffers whose shape depends on cu_seqlens; the coupling is
    validated at GEMM dispatch time (``validate_blockscaled_sf``).
    """

    qdata: torch.Tensor
    scale: torch.Tensor
    format: BlockScaledFormat
    per_tensor_scale: Optional[torch.Tensor] = None  # scalar fp32, NVFP4 only
    orig_dtype: torch.dtype = torch.bfloat16
    # The logical dim the scale vector (sf_vec_size-sized blocks) runs along,
    # normalized to -1 or -2. The direct analogue of CUTLASS's UMMA::Major on
    # SfKMajorAtom/SfMNMajorAtom (sm100_blockscaled_layout.hpp) - the two atoms
    # are mode-swaps of one physical 512 B block, so a quant_dim=-2 operand is
    # byte-identical to the quant_dim=-1 operand of the transposed data. Resolved
    # in __post_init__ (fp4 packing pins it to the packed dim); None -> -1.
    quant_dim: Optional[int] = None

    def __post_init__(self):
        fmt = _coerce_format(self.format)
        object.__setattr__(self, "format", fmt)
        if self.qdata.dtype != fmt.qdata_dtype:
            raise TypeError(f"{fmt.name} qdata must be {fmt.qdata_dtype}, got {self.qdata.dtype}")
        check_blocked_scale_atom(self.scale, "scale")
        # Canonicalize the scale dtype from the format. A uint8 view is accepted
        # (scales round-tripped through collectives / the custom-op boundary arrive
        # as uint8) and re-viewed; anything else is a construction error. This kills
        # the old "uint8 implies e8m0" sniffing trap where an NVFP4 e4m3 scale seen
        # as uint8 would silently run as a vec-32 MX format.
        if self.scale.dtype == torch.uint8:
            object.__setattr__(self, "scale", self.scale.view(fmt.scale_dtype))
        elif self.scale.dtype != fmt.scale_dtype:
            raise TypeError(f"{fmt.name} scale must be {fmt.scale_dtype}, got {self.scale.dtype}")
        if self.per_tensor_scale is not None:
            if not fmt.has_per_tensor_scale:
                raise ValueError(f"{fmt.name} does not take a per_tensor_scale")
            if self.per_tensor_scale.numel() != 1 or self.per_tensor_scale.dtype != torch.float32:
                raise ValueError(
                    f"per_tensor_scale must be a scalar fp32 tensor, got "
                    f"{self.per_tensor_scale.dtype} with {self.per_tensor_scale.numel()} elements"
                )
            # Canonical (1,) shape: the GEMM's alpha folding then never reshapes.
            object.__setattr__(self, "per_tensor_scale", self.per_tensor_scale.reshape(1))
        if self.qdata.device != self.scale.device:
            raise ValueError(f"qdata on {self.qdata.device} but scale on {self.scale.device}")
        requested = (
            None if self.quant_dim is None else _normalize_quant_dim(self.quant_dim, self.ndim)
        )
        if fmt.elems_per_container > 1:
            # fp4 packing pins the quantized axis to the packed (unit-stride) dim.
            derived = -1 if _packed_dim(self.qdata, fmt) == self.qdata.ndim - 1 else -2
            if requested is not None and requested != derived:
                raise ValueError(
                    f"quant_dim={self.quant_dim} conflicts with the fp4 packed dim "
                    f"(packing runs along the quantized axis, here dim {derived})"
                )
            object.__setattr__(self, "quant_dim", derived)
        else:
            object.__setattr__(self, "quant_dim", -1 if requested is None else requested)

    # -- logical metadata (honest, computed - no advertised-metadata machinery) --

    @property
    def shape(self) -> Tuple[int, ...]:
        """Logical (unpacked) shape; fp4x2 doubles the packed dim, which
        ``quant_dim`` already tracks."""
        epc = self.format.elems_per_container
        if epc == 1:
            return tuple(self.qdata.shape)
        k_dim = self._mn_k_dims()[1]
        return tuple(s * epc if d == k_dim else s for d, s in enumerate(self.qdata.shape))

    @property
    def ndim(self) -> int:
        return self.qdata.ndim

    @property
    def dtype(self) -> torch.dtype:
        """The original high-precision dtype (storage truth is ``qdata.dtype``)."""
        return self.orig_dtype

    @property
    def device(self) -> torch.device:
        return self.qdata.device

    def _mn_k_dims(self) -> Tuple[int, int]:
        d0, d1 = self.ndim - 2, self.ndim - 1
        return (d0, d1) if self.quant_dim == -1 else (d1, d0)

    # -- views: qdata stride-swap, scale carried unchanged ------------------

    def _view(self, qdata: torch.Tensor, quant_dim: int) -> "BlockScaledOperand":
        """Unvalidated view constructor: a stride swap of valid qdata cannot
        invalidate any invariant __post_init__ established on ``self`` (and
        ``.mT`` runs once per GEMM call - skip the re-validation)."""
        obj = object.__new__(BlockScaledOperand)
        for name, value in (
            ("qdata", qdata),
            ("scale", self.scale),
            ("format", self.format),
            ("per_tensor_scale", self.per_tensor_scale),
            ("orig_dtype", self.orig_dtype),
            ("quant_dim", quant_dim),
        ):
            object.__setattr__(obj, name, value)
        return obj

    @property
    def mT(self) -> "BlockScaledOperand":
        """Swap the last two logical dims. The blocked scale atom is not
        view-transposable; scale blocks along K in either orientation."""
        return self._view(self.qdata.transpose(-2, -1), -3 - self.quant_dim)  # -1 <-> -2

    @property
    def T(self) -> "BlockScaledOperand":
        if self.ndim != 2:
            raise ValueError(f"BlockScaledOperand.T requires 2-D, got {self.ndim}-D")
        return self.mT

    def transpose(self, dim0: int, dim1: int) -> "BlockScaledOperand":
        dim0, dim1 = dim0 % self.ndim, dim1 % self.ndim
        if {dim0, dim1} != {self.ndim - 2, self.ndim - 1}:
            raise ValueError(
                f"BlockScaledOperand.transpose({dim0}, {dim1}): only the last two dims may "
                f"swap (scale blocks along K; other permutations are not representable)"
            )
        return self.mT

    # -- constructors / conversions ------------------------------------------

    @classmethod
    def from_parts(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        format,
        *,
        per_tensor_scale: Optional[torch.Tensor] = None,
        orig_dtype: torch.dtype = torch.bfloat16,
        quant_dim: int = -1,
    ) -> "BlockScaledOperand":
        """Wrap already-quantized storage (checkpoints, external quantizers).

        ``quant_dim`` is the logical dim the scales block along (default: last,
        the quantize convention). To express the (K, N) B-operand orientation,
        either construct the (N, K) operand and take its ``.mT`` view, or pass
        ``quant_dim=-2`` for data already laid out as (K, N).
        """
        return _construct(
            qdata, scale, _coerce_format(format), per_tensor_scale, orig_dtype, quant_dim
        )

    @classmethod
    def quantize(
        cls,
        x: torch.Tensor,
        format,
        *,
        dim: int = -1,
        per_tensor_scale: Optional[torch.Tensor] = None,
    ) -> "BlockScaledOperand":
        """Quantize a 2-D/3-D bf16/fp32 tensor along ``dim`` (one of the last two
        dims; default: last).

        ``dim=-2`` quantizes along the second-to-last dim via a transpose copy of
        ``x`` (the cost CUTLASS/torchao pay for their dim-1 casts too) and returns
        the ``quant_dim=-2`` view - useful to build a (K, N) B operand directly.

        The NVFP4 ``per_tensor_scale`` is stored on the result and folded into
        GEMM alpha automatically at dispatch (the tuple-era "caller folds
        pts_A*pts_B into alpha" contract does not apply to container operands).
        """
        fmt = _coerce_format(format)
        if _normalize_quant_dim(dim, x.ndim) == -2:
            xt = x.transpose(-1, -2).contiguous()
            return cls.quantize(xt, fmt, per_tensor_scale=per_tensor_scale).mT
        quantizer = QUANTIZERS.get(fmt.name)
        if quantizer is None:
            # Every registered format except mxfp8_e5m2 has a quantizer; to_mx has
            # no e5m2 encoder, so from_parts is the e5m2 construction path.
            raise NotImplementedError(
                f"no in-repo quantizer for {fmt.name}; construct pre-quantized data via from_parts"
            )
        if per_tensor_scale is not None and not fmt.has_per_tensor_scale:
            raise ValueError(f"{fmt.name} does not take a per_tensor_scale")
        assert x.ndim in (2, 3), f"expected (M, K) or (L, M, K), got shape {tuple(x.shape)}"
        assert x.shape[-1] % fmt.sf_vec_size == 0, (
            f"K ({x.shape[-1]}) must be divisible by {fmt.sf_vec_size} for {fmt.name}"
        )
        # Under dynamo, call the raw quantizer: the torch.compile'd wrapper would
        # nest compilation; the raw fn traces into the enclosing graph.
        fn = quantizer[0] if torch.compiler.is_compiling() else quantizer[1]
        batched = x.ndim == 3
        l, mn, k = x.shape if batched else (1, *x.shape)
        x_flat = x.reshape(l * mn, k)
        pts = None
        if fmt.has_per_tensor_scale:
            q, sc, pts_out = fn(x_flat, fmt.sf_vec_size, per_tensor_scale)
            pts = pts_out.to(torch.float32) if per_tensor_scale is not None else None
        else:
            q, sc = fn(x_flat, fmt.sf_vec_size)
        if fmt.elems_per_container > 1:  # quantizers emit packed uint8 codes for fp4
            q = q.view(torch.uint8).view(fmt.qdata_dtype)
        q = q.reshape(*x.shape[:-1], -1)
        sf = pack_scale_2d_to_blocked_contig(sc.view(l, mn, k // fmt.sf_vec_size))
        if not batched:
            sf = sf.squeeze(0)
        return _construct(q, sf, fmt, pts, x.dtype, -1)

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Explicit dequantization to a plain high-precision tensor."""
        fmt = self.format
        from quack.blockscaled.quantize import dequant_operand, unpack_scale_blocked_to_2d

        mn_dim, k_dim = self._mn_k_dims()
        qdata = self.qdata if k_dim == self.ndim - 1 else self.qdata.transpose(mn_dim, k_dim)
        batched = self.ndim == 3
        q3 = qdata if batched else qdata.unsqueeze(0)  # (l, mn, k_packed)
        vals = dequant_operand(q3, fmt)  # (l, mn, k) fp32
        l, mn, k = vals.shape
        scale = self.scale if self.scale.ndim == 6 else self.scale.unsqueeze(0)
        sf2d = unpack_scale_blocked_to_2d(scale, mn, k // fmt.sf_vec_size).float()
        out = vals * sf2d.view(l, mn, k // fmt.sf_vec_size).repeat_interleave(
            fmt.sf_vec_size, dim=-1
        )
        if self.per_tensor_scale is not None:
            out = out * self.per_tensor_scale
        if not batched:
            out = out.squeeze(0)
        if k_dim != self.ndim - 1:
            out = out.transpose(mn_dim, k_dim)
        return out.to(dtype if dtype is not None else self.dtype)

    def to(self, device, non_blocking: bool = False) -> "BlockScaledOperand":
        """Device move only. Implicit dtype conversion of quantized storage is not
        supported; call ``dequantize(dtype)`` explicitly."""
        if isinstance(device, torch.dtype):
            raise TypeError(
                f"BlockScaledOperand.to({device}): dtype conversion of quantized storage "
                f"is not supported; call .dequantize(dtype) explicitly"
            )
        move = lambda t: t.to(device, non_blocking=non_blocking) if t is not None else None
        return replace(
            self,
            qdata=move(self.qdata),
            scale=move(self.scale),
            per_tensor_scale=move(self.per_tensor_scale),
        )

    def clone(self) -> "BlockScaledOperand":
        c = lambda t: t.clone() if t is not None else None
        return replace(
            self,
            qdata=c(self.qdata),
            scale=c(self.scale),
            per_tensor_scale=c(self.per_tensor_scale),
        )

    def __deepcopy__(self, memo):
        d = lambda t: _copy.deepcopy(t, memo) if t is not None else None
        return replace(
            self,
            qdata=d(self.qdata),
            scale=d(self.scale),
            per_tensor_scale=d(self.per_tensor_scale),
        )

    def __repr__(self):
        pts = "" if self.per_tensor_scale is None else f", per_tensor_scale={self.per_tensor_scale}"
        return (
            f"BlockScaledOperand(format={self.format.name}, shape={self.shape}, "
            f"quant_dim={self.quant_dim}, dtype={self.dtype}, "
            f"qdata={self.qdata.dtype}{tuple(self.qdata.shape)}, "
            f"scale={tuple(self.scale.shape)}{pts}, device={self.device})"
        )


@torch._dynamo.allow_in_graph
def _construct(qdata, scale, fmt, per_tensor_scale, orig_dtype, quant_dim):
    return BlockScaledOperand(qdata, scale, fmt, per_tensor_scale, orig_dtype, quant_dim)


# pytree registration: lets the container cross torch.compile / functional-transform
# boundaries as a structure of tensor leaves (same mechanics as the legacy tuple).
def _flatten(op: BlockScaledOperand):
    children = (op.qdata, op.scale, op.per_tensor_scale)
    ctx = (op.format.name, op.orig_dtype, op.quant_dim)
    return children, ctx


def _unflatten(children, ctx):
    qdata, scale, pts = children
    name, orig_dtype, quant_dim = ctx
    return BlockScaledOperand(
        qdata, scale, BlockScaledFormat.from_name(name), pts, orig_dtype, quant_dim
    )


_pytree.register_pytree_node(
    BlockScaledOperand,
    _flatten,
    _unflatten,
    serialized_type_name="quack.blockscaled.operand.BlockScaledOperand",
)

# weights_only torch.load of pickled containers needs the classes allowlisted.
torch.serialization.add_safe_globals([BlockScaledOperand, BlockScaledFormat])
