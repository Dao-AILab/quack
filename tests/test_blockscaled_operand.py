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
    MXFP6_E2M3,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
    BlockScaledFormat,
    BlockScaledOperand,
)

FORMATS = [MXFP8_E4M3, MXFP4, NVFP4]


def _quantize(fmt, m=256, k=256, batched=False, seed=0):
    torch.manual_seed(seed)
    shape = (2, m, k) if batched else (m, k)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * k**-0.5
    return x, BlockScaledOperand.quantize(x, fmt)


# -- format registry ---------------------------------------------------------


def test_format_registry_and_legacy_names():
    assert BlockScaledFormat.from_name("mxfp8") is MXFP8_E4M3  # legacy short name
    assert BlockScaledFormat.from_name("nvfp4") is NVFP4
    with pytest.raises(ValueError, match="unknown blockscaled format"):
        BlockScaledFormat.from_name("fp8")
    for fmt in BLOCKSCALED_FORMAT_REGISTRY.values():
        assert BlockScaledFormat.from_name(fmt.name) is fmt


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
        is MXFP6_E2M3
    )
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


@pytest.mark.parametrize(
    "fmt_name, tol", [("mxfp4_byte", 0.6), ("mxfp6_e2m3", 0.12), ("mxfp6_e3m2", 0.25)]
)
def test_byte_container_formats_quantize(fmt_name, tol):
    """Byte-container sub-byte formats (one code per uint8 - the form the
    mixed-capable kind::mxf8f6f4 consumes) quantize and dequantize today; the
    GEMM rejects them at the per-architecture gate until the DSL grows
    unpacksmem/TMA-expansion support (see gemm_sm100 validity comment)."""
    m, k = 256, 256
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    t = BlockScaledOperand.quantize(x, fmt_name)
    assert t.qdata.dtype == torch.uint8 and t.shape == (m, k) and t.quant_dim == -1
    err = (t.dequantize(torch.float32) - x.float()).abs().max() / x.float().abs().max()
    assert err.item() < tol, f"{fmt_name}: round-trip rel_err={err.item()}"
    # transpose semantics hold for byte formats too
    assert torch.equal(t.mT.dequantize(torch.float32), t.dequantize(torch.float32).mT)


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
    from quack.blockscaled.operand import MXFP6_E3M2, mma_kind_for_pair

    # equal pairs
    assert mma_kind_for_pair(NVFP4, NVFP4) == "mxf4nvf4"
    assert mma_kind_for_pair(MXFP4, MXFP4) == "mxf4"
    assert mma_kind_for_pair(MXFP8_E4M3, MXFP8_E4M3) == "mxf8f6f4"
    # fp8/fp6 byte-container formats mix freely under mxf8f6f4
    assert mma_kind_for_pair(MXFP8_E4M3, MXFP8_E5M2) == "mxf8f6f4"
    assert mma_kind_for_pair(MXFP8_E4M3, MXFP6_E2M3) == "mxf8f6f4"
    assert mma_kind_for_pair(MXFP6_E3M2, MXFP6_E2M3) == "mxf8f6f4"
    # nvfp4 pairs only with itself (e4m3 scales / vec 16 are per-kind)
    with pytest.raises(ValueError, match="mxf4nvf4"):
        mma_kind_for_pair(NVFP4, MXFP8_E4M3)
    with pytest.raises(ValueError, match="mxf4nvf4"):
        mma_kind_for_pair(MXFP4, NVFP4)
    # packed fp4x2 storage cannot join a mixed byte-container pair
    with pytest.raises(ValueError, match="8-bit"):
        mma_kind_for_pair(MXFP4, MXFP8_E4M3)
    # byte-container fp4 mixes under mxf8f6f4; a pure byte-fp4 pair has no kind
    from quack.blockscaled.operand import MXFP4_BYTE

    assert mma_kind_for_pair(MXFP4_BYTE, MXFP8_E4M3) == "mxf8f6f4"
    assert mma_kind_for_pair(MXFP4_BYTE, MXFP6_E2M3) == "mxf8f6f4"
    with pytest.raises(ValueError, match="use mxfp4"):
        mma_kind_for_pair(MXFP4_BYTE, MXFP4_BYTE)


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
