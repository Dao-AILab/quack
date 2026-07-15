# Copyright (c) 2026, Tri Dao.
"""BlockScaledOperand container behavior (no GEMM kernels launched).

Covers construction validation, the logical-metadata surface, transpose
semantics, serialization/pytree, and the format registry - see
AI/blockscaled_api.md section 3 and section 8.
"""

import copy
import io

import pytest
import torch

from quack.blockscaled.operand import (
    BLOCKSCALED_FORMAT_REGISTRY,
    MXFP4,
    MXFP4_BYTE,
    MXFP6_E2M3,
    MXFP6_E2M3_PACKED,
    MXFP6_E3M2,
    MXFP6_E3M2_PACKED,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
    BlockScaledFormat,
    BlockScaledOperand,
)

FORMATS = [
    MXFP8_E4M3,
    MXFP4,
    NVFP4,
    MXFP6_E2M3,
    MXFP6_E3M2,
    MXFP6_E2M3_PACKED,
    MXFP6_E3M2_PACKED,
]


def _quantize(fmt, m=256, k=256, batched=False, seed=0):
    torch.manual_seed(seed)
    shape = (2, m, k) if batched else (m, k)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * k**-0.5
    return x, BlockScaledOperand.quantize(x, fmt)


# -- format registry ---------------------------------------------------------


def test_format_registry_and_legacy_names():
    import quack.blockscaled as blockscaled

    assert BlockScaledFormat.from_name("mxfp8") is MXFP8_E4M3  # legacy short name
    assert BlockScaledFormat.from_name("nvfp4") is NVFP4
    assert BlockScaledFormat.from_name("mxfp6_e2m3") is MXFP6_E2M3
    assert BlockScaledFormat.from_name("mxfp6_e2m3_packed") is MXFP6_E2M3_PACKED
    assert MXFP6_E2M3.is_byte_container and not MXFP6_E2M3_PACKED.is_byte_container
    assert blockscaled.MXFP4_BYTE is MXFP4_BYTE
    assert blockscaled.to_mxfp4_byte is not None
    with pytest.raises(ValueError, match="unknown blockscaled format"):
        BlockScaledFormat.from_name("fp8")
    for fmt in BLOCKSCALED_FORMAT_REGISTRY.values():
        assert BlockScaledFormat.from_name(fmt.name) is fmt


def test_format_k_mapping():
    """logical<->storage K mapping derives from elem_bits and the torch storage
    element width: fp8 identity, fp4x2 ratio 2, packed fp6 (uint8) ratio 4/3."""
    assert not MXFP8_E4M3.is_packed and MXFP4.is_packed and MXFP6_E2M3_PACKED.is_packed
    assert MXFP8_E4M3.storage_k(384) == 384 and MXFP8_E4M3.logical_k(384) == 384
    assert MXFP4.storage_k(384) == 192 and MXFP4.logical_k(192) == 384
    assert NVFP4.storage_k(384) == 192
    for fmt in (MXFP6_E2M3_PACKED, MXFP6_E3M2_PACKED):
        assert fmt.storage_k(384) == 288 and fmt.logical_k(288) == 384
    for fmt in (MXFP4_BYTE, MXFP6_E2M3, MXFP6_E3M2):
        assert not fmt.is_packed and fmt.is_byte_container
        assert fmt.storage_k(384) == 384 and fmt.logical_k(384) == 384
    # non-whole mappings are loud errors (fp6 rows are whole 3-byte groups)
    with pytest.raises(ValueError, match="whole storage"):
        MXFP6_E2M3_PACKED.storage_k(30)  # 180 bits: not whole bytes
    with pytest.raises(ValueError, match="whole packed groups"):
        MXFP6_E2M3_PACKED.logical_k(100)  # 100 bytes: not whole 3-byte groups


def test_format_legacy_positional_order_and_validation():
    """The public constructor keeps the origin/main positional field order."""
    fmt = BlockScaledFormat(
        "legacy_fp4", torch.uint8, "Float4E2M1FN", 4, 1, torch.float8_e8m0fnu, 32
    )
    assert fmt.elems_per_container == 1
    assert fmt.scale_dtype == torch.float8_e8m0fnu and fmt.sf_vec_size == 32
    assert fmt.storage_layout is None and fmt.is_byte_container
    # A call written against the short-lived signature without
    # elems_per_container must fail, not silently shift all later fields.
    with pytest.raises(TypeError, match="elems_per_container"):
        BlockScaledFormat(
            "shifted", torch.uint8, None, 4, torch.float8_e8m0fnu, 32, False
        )
    with pytest.raises(ValueError, match="do not fit"):
        BlockScaledFormat(
            "overfull", torch.uint8, None, 6, 2, torch.float8_e8m0fnu, 32
        )


def test_format_from_cutlass_dtypes():
    import cutlass

    assert (
        BlockScaledFormat.from_cutlass_dtypes(cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32)
        is MXFP8_E4M3
    )
    assert (
        BlockScaledFormat.from_cutlass_dtypes(cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16)
        is NVFP4
    )
    assert (
        BlockScaledFormat.from_cutlass_dtypes(cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32)
        is MXFP4
    )
    assert (
        BlockScaledFormat.from_cutlass_dtypes(cutlass.Float6E2M3FN, cutlass.Float8E8M0FNU, 32)
        is MXFP6_E2M3_PACKED
    )
    assert MXFP6_E2M3.is_byte_container  # byte descriptors are never inferred for GEMM
    with pytest.raises(ValueError, match="no blockscaled format"):
        BlockScaledFormat.from_cutlass_dtypes(cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, 32)


# -- construction & validation -----------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("batched", [False, True])
def test_quantize_dequantize_roundtrip(fmt, batched):
    x, t = _quantize(fmt, batched=batched)
    assert t.shape == tuple(x.shape) and t.dtype == x.dtype and t.device == x.device
    assert t.qdata.dtype == fmt.qdata_dtype and t.scale.dtype == fmt.scale_dtype
    # Numerical correctness of the round trip vs the fp32 original.
    err = (t.dequantize(torch.float32) - x.float()).abs()
    scale = x.float().abs().max()
    assert (err.max() / scale).item() < (0.1 if fmt.elem_bits == 8 else 0.6), (
        f"{fmt.name}: relative round-trip error too large"
    )


def test_construction_rejects():
    x, t = _quantize(MXFP8_E4M3)
    # qdata dtype must match the format
    with pytest.raises(TypeError, match="qdata must be"):
        BlockScaledOperand.from_parts(t.qdata.view(torch.uint8), t.scale, MXFP8_E4M3)
    # scale dtype must match (or be a uint8 view)
    with pytest.raises(TypeError, match="scale must be"):
        BlockScaledOperand.from_parts(t.qdata, t.scale.view(torch.int8), MXFP8_E4M3)
    # bad atom strides
    with pytest.raises(ValueError, match="atom"):
        BlockScaledOperand.from_parts(t.qdata, t.scale.transpose(-1, -2), MXFP8_E4M3)
    # bad scale rank / trailing shape
    with pytest.raises(ValueError, match="32, 4, 4"):
        BlockScaledOperand.from_parts(t.qdata, t.scale.flatten(-3), MXFP8_E4M3)
    # pts on a non-nvfp4 format
    pts = torch.tensor(2.0, device="cuda")
    with pytest.raises(ValueError, match="per_tensor_scale"):
        BlockScaledOperand.from_parts(t.qdata, t.scale, MXFP8_E4M3, per_tensor_scale=pts)
    # non-scalar / non-fp32 pts on nvfp4
    _, n = _quantize(NVFP4)
    with pytest.raises(ValueError, match="scalar fp32"):
        BlockScaledOperand.from_parts(
            n.qdata, n.scale, NVFP4, per_tensor_scale=torch.ones(2, device="cuda")
        )
    with pytest.raises(ValueError, match="per_tensor_scale on cpu"):
        BlockScaledOperand.from_parts(
            n.qdata, n.scale, NVFP4, per_tensor_scale=torch.tensor(1.0)
        )


def test_uint8_scale_canonicalization():
    """A uint8-viewed scale (collective round-trip) re-views to the format dtype;
    NVFP4 must never be mistaken for a vec-32 e8m0 format."""
    _, t = _quantize(NVFP4)
    t2 = BlockScaledOperand.from_parts(t.qdata, t.scale.view(torch.uint8), NVFP4)
    assert t2.scale.dtype == torch.float8_e4m3fn
    assert torch.equal(t2.scale.view(torch.uint8), t.scale.view(torch.uint8))


def test_varlen_padded_scale_constructible():
    """Constructor must NOT couple qdata and scale shapes: varlen operands use
    padded scale buffers whose shape depends on cu_seqlens (validated at GEMM
    dispatch, not at construction)."""
    _, t = _quantize(MXFP8_E4M3, m=256, k=256)
    rm, rk = t.scale.shape[0], t.scale.shape[1]
    padded = torch.zeros(1, rm + 3, rk, 32, 4, 4, dtype=t.scale.dtype, device="cuda")
    bst = BlockScaledOperand.from_parts(t.qdata, padded, MXFP8_E4M3)  # must not raise
    assert bst.scale.shape[1] == rm + 3
    with pytest.raises(ValueError, match="Padded varlen scale buffers"):
        bst.dequantize()

    # Exact-shape atom-aligned slices remain valid dense scale storage even
    # when their outer strides include padding.
    strided_storage = torch.zeros(rm, rk + 3, 32, 4, 4, dtype=t.scale.dtype, device="cuda")
    strided_storage[:, :rk] = t.scale
    dense_view = BlockScaledOperand.from_parts(t.qdata, strided_storage[:, :rk], MXFP8_E4M3)
    assert torch.equal(dense_view.dequantize(torch.float32), t.dequantize(torch.float32))


def test_pack_uint6_bit_layout():
    """The packed fp6 storage contract: element i occupies bits [6i, 6i+6) of a
    little-endian byte stream, so 4 codes (c0..c3) pack into 3 bytes as
    b0 = c0 | c1<<6; b1 = c1>>2 | c2<<4; b2 = c2>>4 | c3<<2 - the CUTLASS
    SubbyteReference bit order (verified bit-exact against the DSL fp6 fill)."""
    from quack.blockscaled.quantize import pack_uint6, unpack_uint6

    codes = torch.tensor([[0x21, 0x33, 0x2A, 0x3F]], dtype=torch.uint8, device="cuda")
    packed = pack_uint6(codes)
    assert packed.shape == (1, 3) and packed.cpu().tolist() == [[0xE1, 0xAC, 0xFE]]
    assert torch.equal(unpack_uint6(packed), codes)
    # round trip on random 6-bit codes
    rand = torch.randint(0, 64, (7, 5, 128), dtype=torch.uint8, device="cuda")
    assert torch.equal(unpack_uint6(pack_uint6(rand)), rand)
    with pytest.raises(AssertionError, match="divisible by 4"):
        pack_uint6(torch.zeros(6, dtype=torch.uint8, device="cuda"))
    with pytest.raises(AssertionError, match="3-byte groups"):
        unpack_uint6(torch.zeros(4, dtype=torch.uint8, device="cuda"))


def test_unversioned_fp6_quantizer_keeps_byte_container_contract():
    from quack.blockscaled.quantize import (
        to_mxfp6_e2m3,
        to_mxfp6_e2m3_packed,
        unpack_uint6,
    )

    x = torch.randn(7, 128, dtype=torch.bfloat16, device="cuda").contiguous()
    q_byte, sf_byte = to_mxfp6_e2m3(x)
    q_packed, sf_packed = to_mxfp6_e2m3_packed(x)
    assert q_byte.shape == x.shape
    assert q_packed.shape == (7, 96)
    assert torch.equal(unpack_uint6(q_packed), q_byte)
    assert torch.equal(sf_packed.view(torch.uint8), sf_byte.view(torch.uint8))


@pytest.mark.parametrize(
    "fmt_name,max_tol,norm_tol",
    [("mxfp6_e2m3_packed", 0.12, 0.05), ("mxfp6_e3m2_packed", 0.25, 0.10)],
)
def test_packed_fp6_quantize_roundtrip(fmt_name, max_tol, norm_tol):
    """Packed fp6 qdata is a uint8 6-bit bit stream: 4 codes per 3 bytes, so
    the storage K extent is 3*K/4 (what the TMA unpack tensormap consumes
    under kind::mxf8f6f4). Round-trip quality on randn: e2m3 ~3.7% rel, e3m2
    ~8% rel - both the eager and compiled quantizers must produce the same
    packed layout."""
    from quack.blockscaled.quantize import QUANTIZERS

    m, k = 256, 512
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    t = BlockScaledOperand.quantize(x, fmt_name)
    assert t.qdata.dtype == torch.uint8 and t.qdata.shape == (m, 3 * k // 4)
    assert t.shape == (m, k) and t.quant_dim == -1
    dq = t.dequantize(torch.float32)
    xf = x.float()
    rel_max = ((dq - xf).abs().max() / xf.abs().max()).item()
    rel_norm = ((dq - xf).norm() / xf.norm()).item()
    assert rel_max < max_tol, f"{fmt_name}: round-trip rel_max={rel_max}"
    assert rel_norm < norm_tol, f"{fmt_name}: round-trip rel_norm={rel_norm}"
    # eager and compiled quantizers agree bit-exactly on the packed layout
    eager_fn, compiled_fn = QUANTIZERS[fmt_name]
    q_e, s_e = eager_fn(x, 32)
    q_c, s_c = compiled_fn(x, 32)
    assert torch.equal(q_e, q_c) and torch.equal(s_e.view(torch.uint8), s_c.view(torch.uint8))
    assert torch.equal(q_e, t.qdata)
    # transpose semantics hold for packed fp6 (quant_dim pinned to the packed dim)
    assert t.mT.shape == (k, m) and t.mT.quant_dim == -2
    assert torch.equal(t.mT.dequantize(torch.float32), dq.mT)


@pytest.mark.parametrize(
    "fmt,max_tol",
    [(MXFP4_BYTE, 0.6), (MXFP6_E2M3, 0.12), (MXFP6_E3M2, 0.25)],
    ids=lambda f: f.name if isinstance(f, BlockScaledFormat) else str(f),
)
def test_byte_container_compatibility_roundtrip_and_gemm_rejection(fmt, max_tol):
    """Deprecated byte formats remain host-side lossless compatibility paths."""
    from quack.blockscaled.operand import mma_kind_for_pair

    m, k = 7, 96
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    t = BlockScaledOperand.quantize(x, fmt)
    assert t.qdata.dtype == torch.uint8 and t.qdata.shape == (m, k)
    assert t.shape == (m, k) and t.quant_dim == -1
    dq = t.dequantize(torch.float32)
    rel = ((dq - x.float()).abs().max() / x.float().abs().max()).item()
    assert rel < max_tol
    packed = t.to_packed()
    target = {
        MXFP4_BYTE: MXFP4,
        MXFP6_E2M3: MXFP6_E2M3_PACKED,
        MXFP6_E3M2: MXFP6_E3M2_PACKED,
    }[fmt]
    assert packed.format is target and packed.shape == t.shape
    assert packed.qdata.shape[-1] == target.storage_k(k)
    assert torch.equal(packed.dequantize(torch.float32), dq)
    with pytest.raises(ValueError, match="byte-per-element.*host-side compatibility"):
        mma_kind_for_pair(t.format, MXFP8_E4M3)


def test_to_packed_canonicalizes_interim_name_and_rejects_custom_recipe():
    x, byte = _quantize(MXFP6_E2M3, m=7, k=128)
    packed = byte.to_packed()
    interim_fmt = BlockScaledFormat(
        "mxfp6_e2m3",
        torch.uint8,
        "Float6E2M3FN",
        6,
        1,
        torch.float8_e8m0fnu,
        32,
        storage_layout="packed_lsb_v1",
    )
    interim = BlockScaledOperand.from_parts(packed.qdata, packed.scale, interim_fmt)
    canonical = interim.to_packed()
    assert canonical.format is MXFP6_E2M3_PACKED
    assert canonical.qdata is interim.qdata and canonical.scale is interim.scale
    assert torch.equal(canonical.dequantize(torch.float32), interim.dequantize(torch.float32))
    quantized = BlockScaledOperand.quantize(x, interim_fmt)
    explicit = BlockScaledOperand.quantize(x, MXFP6_E2M3_PACKED)
    assert quantized.format is MXFP6_E2M3_PACKED and quantized.qdata.shape == (7, 96)
    assert torch.equal(quantized.qdata, explicit.qdata)
    assert torch.equal(quantized.scale.view(torch.uint8), explicit.scale.view(torch.uint8))

    custom_vec64 = BlockScaledFormat(
        "customer_fp6_vec64",
        torch.uint8,
        "Float6E2M3FN",
        6,
        1,
        torch.float8_e8m0fnu,
        64,
    )
    custom = BlockScaledOperand.from_parts(byte.qdata, byte.scale, custom_vec64)
    with pytest.raises(ValueError, match="recipe.*does not match"):
        custom.to_packed()


def test_e5m2_from_parts_but_no_quantizer():
    _, t = _quantize(MXFP8_E4M3)
    q = t.qdata.view(torch.uint8).view(torch.float8_e5m2)
    t5 = BlockScaledOperand.from_parts(q, t.scale, MXFP8_E5M2)
    assert t5.format is MXFP8_E5M2
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="e5m2"):
        BlockScaledOperand.quantize(x, MXFP8_E5M2)


# -- logical metadata & transpose semantics ----------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_transpose_is_qdata_stride_swap(fmt):
    x, t = _quantize(fmt)
    tt = t.mT
    assert tt.shape == (t.shape[1], t.shape[0])
    assert tt.scale is t.scale  # scale carried unchanged: it blocks along K either way
    assert tt.qdata.shape == (t.qdata.shape[1], t.qdata.shape[0])
    assert torch.equal(tt.dequantize(torch.float32), t.dequantize(torch.float32).mT)
    # round trip
    ttt = tt.mT
    assert ttt.shape == t.shape
    assert torch.equal(ttt.dequantize(torch.float32), t.dequantize(torch.float32))
    # .T is the 2-D alias
    assert torch.equal(t.T.dequantize(torch.float32), tt.dequantize(torch.float32))


def test_batched_transpose_and_rejections():
    x, t = _quantize(MXFP8_E4M3, batched=True)
    tt = t.transpose(1, 2)
    assert tt.shape == (2, t.shape[2], t.shape[1]) and tt.scale is t.scale
    assert torch.equal(tt.dequantize(torch.float32), t.dequantize(torch.float32).transpose(1, 2))
    # only the last two dims may swap
    with pytest.raises(ValueError, match="last two"):
        t.transpose(0, 2)
    with pytest.raises(ValueError, match="2-D"):
        t.T  # noqa: B018


# -- explicit method surface (no aten interception) --------------------------


def test_clone_to_deepcopy():
    _, t = _quantize(NVFP4)
    c = t.clone()
    assert torch.equal(c.qdata.view(torch.uint8), t.qdata.view(torch.uint8))
    assert c.qdata.data_ptr() != t.qdata.data_ptr()
    # device round trip
    cpu = t.to("cpu")
    assert cpu.device.type == "cpu" and cpu.qdata.device.type == "cpu"
    back = cpu.to("cuda")
    assert torch.equal(back.qdata.view(torch.uint8), t.qdata.view(torch.uint8))
    # implicit dtype conversion must point at dequantize()
    with pytest.raises(TypeError, match="dequantize"):
        t.to(torch.float16)
    d = copy.deepcopy(t)
    assert d.format is NVFP4 and d.qdata.data_ptr() != t.qdata.data_ptr()
    assert torch.equal(d.dequantize(torch.float32), t.dequantize(torch.float32))


def test_torch_ops_fail_loudly():
    """The container is not a Tensor: torch ops reject it with a TypeError
    instead of silently dequantizing or computing on packed storage."""
    _, t = _quantize(MXFP8_E4M3)
    with pytest.raises(TypeError):
        torch.mm(t, t.mT)
    with pytest.raises(TypeError):
        t + t


def test_repr_is_metadata_only():
    _, t = _quantize(NVFP4)
    r = repr(t)
    assert "nvfp4" in r and "float4_e2m1fn_x2" in r and str(t.shape) in r


# -- serialization / pytree --------------------------------------------------


def test_save_load_weights_only():
    _, t = _quantize(NVFP4)
    buf = io.BytesIO()
    torch.save(t, buf)
    buf.seek(0)
    t2 = torch.load(buf, weights_only=True)
    assert isinstance(t2, BlockScaledOperand) and t2.format == t.format
    assert t2.shape == t.shape and t2.dtype == t.dtype
    assert torch.equal(t2.qdata.view(torch.uint8), t.qdata.view(torch.uint8))
    assert torch.equal(t2.scale.view(torch.uint8), t.scale.view(torch.uint8))


def test_origin_main_byte_fp6_pickle_migrates_without_reinterpretation():
    """An old descriptor has no storage_layout field and canonical fp6 name.

    Loading must materialize the missing default as legacy byte-container
    semantics. K=96 is deliberately divisible by three: blindly applying the
    new packed ratio would look superficially valid while changing logical K
    to 128.
    """
    import torch.utils._pytree as pytree

    from quack.blockscaled.operand import _unflatten, mma_kind_for_pair

    x = torch.randn(5, 96, device="cuda", dtype=torch.bfloat16)
    byte_op = BlockScaledOperand.quantize(x, MXFP6_E2M3)
    name_only = BlockScaledOperand.from_parts(
        byte_op.qdata, byte_op.scale, "mxfp6_e2m3", orig_dtype=torch.bfloat16
    )
    assert name_only.format is MXFP6_E2M3 and name_only.shape == (5, 96)
    assert torch.equal(name_only.dequantize(torch.float32), byte_op.dequantize(torch.float32))
    old_fmt = object.__new__(BlockScaledFormat)
    for name, value in (
        ("name", "mxfp6_e2m3"),
        ("qdata_dtype", torch.uint8),
        ("cutlass_dtype_name", "Float6E2M3FN"),
        ("elem_bits", 6),
        ("elems_per_container", 1),
        ("scale_dtype", torch.float8_e8m0fnu),
        ("sf_vec_size", 32),
        ("has_per_tensor_scale", False),
    ):
        object.__setattr__(old_fmt, name, value)
    assert "storage_layout" not in old_fmt.__dict__
    old_op = BlockScaledOperand.from_parts(
        byte_op.qdata, byte_op.scale, old_fmt, orig_dtype=torch.bfloat16
    )

    buf = io.BytesIO()
    torch.save(old_op, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=True)
    assert loaded.format.name == "mxfp6_e2m3"
    assert loaded.format.storage_layout is None
    assert "storage_layout" in loaded.format.__dict__
    assert loaded.format.is_byte_container and loaded.shape == (5, 96)
    assert loaded.format == MXFP6_E2M3
    assert hash(loaded.format) == hash(old_fmt)
    assert torch.equal(loaded.dequantize(torch.float32), byte_op.dequantize(torch.float32))
    migrated = loaded.to_packed()
    assert migrated.format is MXFP6_E2M3_PACKED and migrated.shape == (5, 96)
    assert migrated.qdata.shape == (5, 72)
    assert torch.equal(migrated.dequantize(torch.float32), loaded.dequantize(torch.float32))

    leaves, spec = pytree.tree_flatten(loaded)
    rt = pytree.tree_unflatten(leaves, spec)
    assert rt.format.storage_layout is None and rt.shape == (5, 96)
    assert torch.equal(rt.dequantize(torch.float32), loaded.dequantize(torch.float32))

    # Name-only TreeSpec contexts emitted by origin/main keep byte semantics.
    old_tree_rt = _unflatten(
        (loaded.qdata, loaded.scale, None),
        ("mxfp6_e2m3", loaded.orig_dtype, loaded.quant_dim),
    )
    assert old_tree_rt.format.storage_layout is None and old_tree_rt.shape == (5, 96)
    with pytest.raises(ValueError, match="byte-per-element.*host-side compatibility"):
        mma_kind_for_pair(loaded.format, MXFP8_E4M3)
    from quack.gemm_interface import gemm

    with pytest.raises(ValueError, match="byte-per-element.*host-side compatibility"):
        gemm(loaded, loaded.mT, tuned=False)


def test_feature_head_packed_fp6_pickle_migrates_to_versioned_name():
    """The feature schema omitted epc/layout and derived 6-bit packing by width."""
    x = torch.randn(5, 96, device="cuda", dtype=torch.bfloat16)
    packed = BlockScaledOperand.quantize(x, MXFP6_E2M3_PACKED)

    feature_fmt = object.__new__(BlockScaledFormat)
    for name, value in (
        ("name", "mxfp6_e2m3"),
        ("qdata_dtype", torch.uint8),
        ("cutlass_dtype_name", "Float6E2M3FN"),
        ("elem_bits", 6),
        ("scale_dtype", torch.float8_e8m0fnu),
        ("sf_vec_size", 32),
        ("has_per_tensor_scale", False),
    ):
        object.__setattr__(feature_fmt, name, value)
    assert "elems_per_container" not in feature_fmt.__dict__
    assert "storage_layout" not in feature_fmt.__dict__

    feature_op = object.__new__(BlockScaledOperand)
    for name, value in (
        ("qdata", packed.qdata),
        ("scale", packed.scale),
        ("format", feature_fmt),
        ("per_tensor_scale", None),
        ("orig_dtype", packed.orig_dtype),
        ("quant_dim", packed.quant_dim),
    ):
        object.__setattr__(feature_op, name, value)

    buf = io.BytesIO()
    torch.save(feature_op, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=True)
    assert loaded.format == MXFP6_E2M3_PACKED
    assert loaded.format.name == "mxfp6_e2m3_packed"
    assert loaded.format.elems_per_container == 1
    assert loaded.format.storage_layout == "packed_lsb_v1"
    assert loaded.qdata.shape == (5, 72) and loaded.shape == (5, 96)
    assert torch.equal(loaded.dequantize(torch.float32), packed.dequantize(torch.float32))

    feature_fp4 = object.__new__(BlockScaledFormat)
    feature_fp4.__setstate__(
        {
            "name": "mxfp4",
            "qdata_dtype": torch.float4_e2m1fn_x2,
            "cutlass_dtype_name": "Float4E2M1FN",
            "elem_bits": 4,
            "scale_dtype": torch.float8_e8m0fnu,
            "sf_vec_size": 32,
            "has_per_tensor_scale": False,
        }
    )
    assert feature_fp4 == MXFP4
    assert feature_fp4.elems_per_container == 2 and feature_fp4.storage_layout is None


def test_pytree_roundtrip():
    import torch.utils._pytree as pytree

    _, t = _quantize(NVFP4)
    leaves, spec = pytree.tree_flatten(t)
    assert all(isinstance(leaf, torch.Tensor) or leaf is None for leaf in leaves)
    rt = pytree.tree_unflatten(leaves, spec)
    assert isinstance(rt, BlockScaledOperand)
    assert rt.format is t.format and rt.shape == t.shape and rt.quant_dim == t.quant_dim
    # transposed views round-trip their orientation
    leaves, spec = pytree.tree_flatten(t.mT)
    rt = pytree.tree_unflatten(leaves, spec)
    assert rt.quant_dim == t.mT.quant_dim and rt.shape == t.mT.shape
    # Packed FP6 carries its explicit storage-version name through the context.
    _, t6 = _quantize(MXFP6_E2M3_PACKED, m=8, k=128)
    leaves, spec = pytree.tree_flatten(t6)
    rt6 = pytree.tree_unflatten(leaves, spec)
    assert rt6.format is MXFP6_E2M3_PACKED
    assert rt6.format.storage_layout == "packed_lsb_v1"
    assert rt6.shape == t6.shape


def test_pytree_treespec_json_roundtrip_preserves_custom_and_legacy_formats():
    import json

    import torch.utils._pytree as pytree

    _, packed = _quantize(MXFP6_E2M3_PACKED, m=8, k=128)
    custom_fmt = BlockScaledFormat(
        "customer_fp6_packed_v7",
        torch.uint8,
        "Float6E2M3FN",
        6,
        1,
        torch.float8_e8m0fnu,
        32,
        storage_layout="packed_lsb_v1",
    )
    custom = BlockScaledOperand.from_parts(packed.qdata, packed.scale, custom_fmt)

    for op in (custom, BlockScaledOperand.quantize(packed.dequantize(), MXFP6_E2M3)):
        leaves, spec = pytree.tree_flatten(op)
        dumped = pytree.treespec_dumps(spec)
        json.loads(dumped)  # the entire context must be JSON-safe
        restored = pytree.tree_unflatten(leaves, pytree.treespec_loads(dumped))
        assert restored.format == op.format
        assert restored.format.storage_layout == op.format.storage_layout
        assert restored.orig_dtype == op.orig_dtype and restored.quant_dim == op.quant_dim
        assert restored.shape == op.shape


# -- rejection guards at non-blockscaled entry points ------------------------


def test_rejection_guards():
    from quack.gemm_interface import gemm_dact, gemm_rms, gemm_symmetric

    _, t = _quantize(MXFP8_E4M3)
    y = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    for fn, name in (
        (lambda: gemm_dact(t, y.T, y), "gemm_dact"),
        (lambda: gemm_symmetric(t, t.mT), "gemm_symmetric"),
        (lambda: gemm_rms(t, y.T), "gemm_rms"),
    ):
        with pytest.raises(TypeError, match=name):
            fn()


def test_mma_kind_for_pair():
    from quack.blockscaled.operand import mma_kind_for_pair

    # equal pairs: both-fp4 is ONE mxf4nvf4 atom, scale config from the format
    # (mxfp4: vec 32 e8m0 - PTX spells that instantiation kind::mxf4)
    assert mma_kind_for_pair(NVFP4, NVFP4) == "mxf4nvf4"
    assert mma_kind_for_pair(MXFP4, MXFP4) == "mxf4nvf4"
    assert mma_kind_for_pair(MXFP8_E4M3, MXFP8_E4M3) == "mxf8f6f4"
    # fp8/fp6 mix freely under mxf8f6f4
    assert mma_kind_for_pair(MXFP8_E4M3, MXFP8_E5M2) == "mxf8f6f4"
    assert mma_kind_for_pair(MXFP8_E4M3, MXFP6_E2M3_PACKED) == "mxf8f6f4"
    assert mma_kind_for_pair(MXFP6_E3M2_PACKED, MXFP6_E2M3_PACKED) == "mxf8f6f4"
    # packed sub-byte operands join mixed pairs too: TMA unpack expands the
    # packed gmem into 8-bit smem containers under kind::mxf8f6f4
    assert mma_kind_for_pair(MXFP4, MXFP8_E4M3) == "mxf8f6f4"
    assert mma_kind_for_pair(MXFP8_E5M2, MXFP4) == "mxf8f6f4"
    assert mma_kind_for_pair(MXFP4, MXFP6_E2M3_PACKED) == "mxf8f6f4"
    with pytest.raises(ValueError, match="byte-per-element"):
        mma_kind_for_pair(MXFP6_E2M3, MXFP8_E4M3)
    # nvfp4 pairs only with itself (e4m3 scales / vec 16 are per-kind)
    with pytest.raises(ValueError, match="mxf4nvf4"):
        mma_kind_for_pair(NVFP4, MXFP8_E4M3)
    with pytest.raises(ValueError, match="mxf4nvf4"):
        mma_kind_for_pair(MXFP4, NVFP4)


# -- quantized-axis direction (CUTLASS SfK/MNMajorAtom analogue) --------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_quantize_dim(fmt):
    """quantize(dim=-2) equals quantizing the transposed data and viewing back:
    the two directions are mode-swaps of one physical scale atom."""
    torch.manual_seed(0)
    k, n = 256, 512
    w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * k**-0.5
    b = BlockScaledOperand.quantize(w, fmt, dim=-2)  # (K, N) quantized along K
    assert b.shape == (k, n) and b.quant_dim == -2
    ref = BlockScaledOperand.quantize(w.mT.contiguous(), fmt).mT
    assert torch.equal(b.dequantize(torch.float32), ref.dequantize(torch.float32))
    assert torch.equal(b.qdata.view(torch.uint8), ref.qdata.view(torch.uint8))
    # canonical default
    a = BlockScaledOperand.quantize(w, fmt)
    assert a.quant_dim == -1 and a.mT.quant_dim == -2 and a.mT.mT.quant_dim == -1
    # the batch dim of a 3-D tensor is never a quantization axis
    with pytest.raises(ValueError, match="last two dims"):
        BlockScaledOperand.quantize(
            torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16), fmt, dim=0
        )


def test_from_parts_quant_dim():
    _, t = _quantize(MXFP8_E4M3)
    # explicit quant_dim=-2 on byte formats
    b = BlockScaledOperand.from_parts(t.qdata, t.scale, MXFP8_E4M3, quant_dim=-2)
    assert b.quant_dim == -2 and b.mT.quant_dim == -1
    # fp4: quant_dim is pinned by the packing; a conflicting request raises
    _, t4 = _quantize(MXFP4)
    with pytest.raises(ValueError, match="packed dim"):
        BlockScaledOperand.from_parts(t4.qdata, t4.scale, MXFP4, quant_dim=-2)


def test_packed_dim_ignores_size_one_dims():
    """A (Kp, 1) K-major fp4 operand can present stride 1 on BOTH dims (size-1
    dims report arbitrary strides); the packed (quantized) dim is the extent>1
    one. Regression: the unit-stride scan used to pick the size-1 dim and
    reject quant_dim=-2."""
    _, t = _quantize(MXFP4, m=1, k=256)
    q = t.qdata.reshape(-1).unsqueeze(-1)  # (Kp, 1) with strides (1, 1)
    assert q.stride() == (1, 1)
    b = BlockScaledOperand.from_parts(q, t.scale, MXFP4, quant_dim=-2)
    assert b.quant_dim == -2 and b.shape == (256, 1)
    assert torch.equal(b.dequantize(torch.float32), t.dequantize(torch.float32).mT)


def test_format_without_dsl_element_type():
    """cutlass_dtype_name=None marks a host-side-only format (e.g. a future
    e3m4): it must fail loudly at the kernel seam (to_cutlass_dtype) and at
    kind selection (no silent mxf8f6f4 fall-through), and a registered
    DSL-typeless format must not break registry-wide dtype lookup for the
    formats that do have DSL types. See AI/blockscaled_api.md section 9."""
    from quack.blockscaled.operand import BLOCKSCALED_FORMAT_REGISTRY, mma_kind_for_pair

    weird = BlockScaledFormat("e3m4_test", torch.uint8, None, 8, 1, torch.float8_e8m0fnu, 32)
    with pytest.raises(ValueError, match="no CuTe-DSL element type"):
        weird.to_cutlass_dtype()
    with pytest.raises(ValueError, match="no tcgen05 MMA element type"):
        mma_kind_for_pair(weird, MXFP8_E4M3)
    with pytest.raises(ValueError, match="no tcgen05 MMA element type"):
        mma_kind_for_pair(MXFP8_E4M3, weird)
    import cutlass

    BLOCKSCALED_FORMAT_REGISTRY[weird.name] = weird
    try:
        assert (
            BlockScaledFormat.from_cutlass_dtypes(cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32)
            is MXFP8_E4M3
        )
    finally:
        del BLOCKSCALED_FORMAT_REGISTRY[weird.name]


def test_future_recipe_examples():
    """Executable form of the worked examples next to BLOCKSCALED_FORMAT_REGISTRY
    (AI/blockscaled_api.md section 10): recipes the descriptor already expresses
    as pure data, pinned to fail at the KIND layer with the right reason until
    their consumer kernels land. The scale-recipe axis mirrors cuBLASLt's
    cublasLtMatmulMatrixScale_t and torch._C._ScalingType."""
    from quack.blockscaled.operand import mma_kind_for_pair

    # DeepSeek-style 1x128 fp32-scale fp8: hardware elements, software recipe.
    ds1d = BlockScaledFormat(
        "fp8_e4m3_1x128", torch.float8_e4m3fn, "Float8E4M3FN", 8, 1, torch.float32, 128
    )
    with pytest.raises(ValueError, match="no hardware MMA kind"):
        mma_kind_for_pair(ds1d, ds1d)
    # ... and a software recipe cannot smuggle in next to a hardware one.
    with pytest.raises(ValueError, match="no hardware MMA kind"):
        mma_kind_for_pair(MXFP8_E4M3, ds1d)

    # kscale: bf16 elements + per-row fp32 K-block scales (one-sided customer).
    kscale = BlockScaledFormat(
        "bf16_1x128", torch.bfloat16, "BFloat16", 16, 1, torch.float32, 128
    )
    with pytest.raises(ValueError, match="no tcgen05 MMA element type"):
        mma_kind_for_pair(kscale, MXFP8_E4M3)

    # W4A16 int4-g128 (AWQ/GPTQ): fixed-offset elements are element decode;
    # bf16 scale dtype is just data; no integer tcgen05 blockscaled kind.
    int4 = BlockScaledFormat("int4_1x128", torch.uint8, "Int4", 4, 2, torch.bfloat16, 128)
    with pytest.raises(ValueError, match="no tcgen05 MMA element type"):
        mma_kind_for_pair(int4, int4)

    # e3m4: no DSL element type at all -> host-side only (see
    # test_format_without_dsl_element_type for the full behavior pin).
    e3m4 = BlockScaledFormat("mxfp8_e3m4", torch.uint8, None, 8, 1, torch.float8_e8m0fnu, 32)
    with pytest.raises(ValueError, match="no CuTe-DSL element type"):
        e3m4.to_cutlass_dtype()

    # All are hashable descriptor singletons like the registered formats.
    assert len({ds1d, kscale, int4, e3m4}) == 4

    # W4A16-nvfp4 needs no new format: the container spelling is the existing
    # NVFP4 operand paired with a plain bf16 tensor; sided-ness is per-kind
    # (a follow-up), so today the interface still requires both-sided SF.
    assert NVFP4.has_per_tensor_scale
