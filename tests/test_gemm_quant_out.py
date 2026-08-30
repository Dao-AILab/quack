# Copyright (c) 2026, Tri Dao.
"""Quantized-output GEMM (SFD epilogue): mxfp8 / mxfp4 / nvfp4 D on SM100/SM120.

The kernel quantizes the fp32 epilogue values per SF vector (32 elements along
N for e8m0 scales, 16 for e4m3), stores the scale bytes in the blocked
(rm, rk, 32, 4, 4) layout, and writes D scaled by the reciprocal of the
quantized scale. Rounding matches the hardware cvt semantics: f32->e8m0 rounds
toward +inf (cvt.rp), f32->e4m3 / f32->e2m1 round to nearest even, saturating.

Exactness tests use B = identity so the GEMM result (and hence every amax) is
bit-exact against the reference; random-B tests bound the dequantized error by
the quantization step.
"""

import pytest
import torch

from quack.gemm_interface import gemm
from quack.blockscaled import BlockScaledFormat, BlockScaledOperand
from quack.blockscaled.quantize import dequant_operand, unpack_scale_blocked_to_2d
from quack.cute_dsl_utils import get_compile_target_capacity, get_device_capacity


if get_device_capacity()[0] not in (10, 11, 12):
    pytest.skip(reason="Quantized-output GEMM requires SM100/SM110/SM120", allow_module_level=True)

# Dispatch arch (QUACK_ARCH) is not enough: the SFD epilogue's f32->e8m0 and
# f32->e2m1 cvts have no pre-SM100 encoding, so the H100 proxy legs
# (QUACK_ARCH=120 compiled for sm_90a) fail in NVVM, not at dispatch.
if get_compile_target_capacity()[0] < 10:
    pytest.skip(
        reason="Quantized-output cvts (f32->e8m0/e2m1) need an sm_100+/sm_120+ compile target",
        allow_module_level=True,
    )

# SM120 (SM90-style register epilogue): full parity — both directions and
# aux-target (minted-mod postact) quantization run.
IS_SM120 = get_device_capacity()[0] == 12


def skip_unsupported(fmt=None, col=False, aux=False):
    pass


DTYPE_MAX = {torch.float8_e4m3fn: 448.0, torch.float4_e2m1fn_x2: 6.0}


def fmt_props(fmt: str):
    f = BlockScaledFormat.from_name(fmt)
    return f.qdata_dtype, f.scale_dtype, f.sf_vec_size


def ceil_div(a, b):
    return (a + b - 1) // b


def quant_ref(x: torch.Tensor, fmt: str, norm_const: float = 1.0):
    """Reference quantization matching the kernel exactly.

    Returns (q_float, sf_bytes, scale_float): q_float holds the quantized
    values (already representable in the value dtype), sf_bytes the raw SF
    bytes, scale_float the dequantized scale.
    """
    val_dtype, scale_dtype, vec = fmt_props(fmt)
    dmax = DTYPE_MAX[val_dtype]
    m, n = x.shape
    xb = x.reshape(m, n // vec, vec).float()
    amax = xb.abs().amax(-1)
    scale = amax / dmax * norm_const
    if scale_dtype == torch.float8_e8m0fnu:
        # cvt.rp.satfinite.ue8m0x2.f32: round exponent up
        mant, exp = torch.frexp(scale)
        e = torch.where(mant == 0.5, exp - 1, exp)
        e = torch.where(scale == 0, torch.full_like(exp, -127), e).clamp(-127, 127)
        sf_bytes = (e + 127).to(torch.uint8)
        scale_q = torch.pow(2.0, e.float())
    else:
        sf_e4m3 = scale.clamp(max=448.0).to(torch.float8_e4m3fn)
        sf_bytes = sf_e4m3.view(torch.uint8)
        scale_q = sf_e4m3.float()
    rcp = torch.where(
        scale_q > 0,
        norm_const / scale_q,
        torch.full_like(scale_q, torch.finfo(torch.float32).max),
    )
    q = (xb * rcp.unsqueeze(-1)).clamp(-dmax, dmax)
    if val_dtype == torch.float4_e2m1fn_x2:
        # snap to the fp4 grid (RNE): {0, .5, 1, 1.5, 2, 3, 4, 6}
        q = q.to(torch.float32).reshape(m, n)
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
        sign = q.sign()
        mag = q.abs()
        # round-half-even between adjacent grid points
        idx = torch.searchsorted(grid, mag)
        lo = grid[(idx - 1).clamp(min=0)]
        hi = grid[idx.clamp(max=7)]
        pick_hi = (mag - lo > hi - mag) | ((mag - lo == hi - mag) & (idx % 2 == 0))
        q = sign * torch.where(pick_hi, hi, lo)
    else:
        q = q.reshape(m, n).to(val_dtype).float()
    return q, sf_bytes, scale_q


def out_values(out: torch.Tensor) -> torch.Tensor:
    return dequant_operand(out) if out.dtype == torch.float4_e2m1fn_x2 else out.float()


@pytest.mark.parametrize("fmt", ["mxfp8_e4m3", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("mnk", [(256, 256, 256), (128, 448, 64)])
def test_quant_out_exact(fmt, mnk):
    """B = identity => D == A exactly => SF bytes and D values must be exact."""
    skip_unsupported(fmt)
    torch.manual_seed(0)
    m, n, k = mnk
    device = "cuda"
    _, _, vec = fmt_props(fmt)
    A = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    B = torch.eye(k, n, dtype=torch.bfloat16, device=device)
    res = gemm(A, B, out_dtype=fmt, tuned=False)
    assert isinstance(res, BlockScaledOperand) and res.format.name == fmt
    ref_x = A.float() @ B.float()
    q_ref, sf_ref, scale_ref = quant_ref(ref_x, fmt)

    sf_2d = unpack_scale_blocked_to_2d(res.scale.unsqueeze(0), m, ceil_div(n, vec))[0]
    assert torch.equal(sf_2d.view(torch.uint8), sf_ref), "SF bytes mismatch"
    torch.testing.assert_close(out_values(res.qdata), q_ref, rtol=0, atol=0)


@pytest.mark.parametrize("fmt", ["mxfp8_e4m3", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("alpha,bias", [(1.0, False), (2.0, True)])
@pytest.mark.parametrize(
    "mnk",
    [
        (256, 512, 256),
        (257, 208, 128),  # ragged M and N (partial SF atoms)
        (100, 96, 64),
    ],
)
def test_quant_out_random(fmt, alpha, bias, mnk):
    """Random B: dequantized output within the per-block quantization step."""
    skip_unsupported(fmt)
    torch.manual_seed(0)
    m, n, k = mnk
    device = "cuda"
    val_dtype, _, vec = fmt_props(fmt)
    if val_dtype == torch.float4_e2m1fn_x2 and n % 32 != 0:
        pytest.skip("fp4 output requires N % 32 == 0")
    A = torch.randn(m, k, dtype=torch.bfloat16, device=device) / k**0.25
    B = torch.randn(k, n, dtype=torch.bfloat16, device=device) / k**0.25
    rowvec = torch.randn(n, dtype=torch.bfloat16, device=device) if bias else None
    res = gemm(A, B, bias=rowvec, alpha=alpha, out_dtype=fmt, tuned=False)

    ref = alpha * (A.float() @ B.float())
    if bias:
        ref = ref + rowvec.float()
    sf_2d = unpack_scale_blocked_to_2d(res.scale.unsqueeze(0), m, ceil_div(n, vec))[0].float()
    scale = sf_2d.repeat_interleave(vec, -1)[:, :n]
    deq = out_values(res.qdata) * scale
    # Worst-case rounding error is half the largest bin gap, in units of the
    # scale: e4m3 -> (448-416)/2 = 16, e2m1 -> (6-4)/2 = 1.
    half_gap = 16.0 if val_dtype == torch.float8_e4m3fn else 1.0
    bound = scale * half_gap * 1.05 + 1e-2
    assert ((deq - ref).abs() <= bound).all(), (
        f"max err {(deq - ref).abs().max().item()} vs bound {bound.max().item()}"
    )
    # no NaN scale bytes (poison) inside the covered region
    assert not torch.isnan(sf_2d[:, : ceil_div(n, vec)]).any()


def test_quant_out_batched():
    torch.manual_seed(0)
    l, m, n, k = 2, 256, 256, 128
    A = torch.randn(l, m, k, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(l, k, n, dtype=torch.bfloat16, device="cuda") / k**0.5
    res = gemm(A, B, out_dtype="mxfp8_e4m3", tuned=False)
    out, out_sf = res.qdata, res.scale
    assert out.shape == (l, m, n) and out_sf.shape[0] == l
    vec = 32
    for li in range(l):
        ref = A[li].float() @ B[li].float()
        sf_2d = unpack_scale_blocked_to_2d(out_sf[li : li + 1], m, n // vec)[0].float()
        deq = out[li].float() * sf_2d.repeat_interleave(vec, -1)
        bound = sf_2d.repeat_interleave(vec, -1) * 16.0 + 1e-2
        assert ((deq - ref).abs() <= bound).all()


def test_quant_out_nvfp4_per_tensor_scale():
    """nvfp4 with global (second-level) scale: dequant = out * sf * pts."""
    skip_unsupported("nvfp4")
    torch.manual_seed(0)
    m, n, k, vec = 256, 256, 128, 16
    A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    B = torch.eye(k, n, dtype=torch.bfloat16, device="cuda")
    ref = A.float() @ B.float()
    pts = (ref.abs().max() / (448.0 * 6.0)).item()
    res = gemm(A, B, out_dtype="nvfp4", out_per_tensor_scale=pts, tuned=False)
    assert res.per_tensor_scale is not None and res.per_tensor_scale.item() == pytest.approx(pts)
    q_ref, sf_ref, scale_ref = quant_ref(ref, "nvfp4", norm_const=1.0 / pts)
    sf_2d = unpack_scale_blocked_to_2d(res.scale.unsqueeze(0), m, n // vec)[0]
    assert torch.equal(sf_2d.view(torch.uint8), sf_ref)
    deq = out_values(res.qdata) * sf_2d.float().repeat_interleave(vec, -1) * pts
    bound = sf_2d.float().repeat_interleave(vec, -1) * pts * 1.01 + 1e-3
    assert ((deq - ref).abs() <= bound).all()


def test_quant_out_blockscaled_input():
    """nvfp4 inputs (SFA/SFB) with nvfp4 quantized output — the fc1->fc2 shape."""
    skip_unsupported("nvfp4")
    torch.manual_seed(0)
    m, n, k = 256, 512, 256
    A_hp = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / k**0.5
    B_hp = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / k**0.5
    opA = BlockScaledOperand.quantize(A_hp, "nvfp4")
    opB = BlockScaledOperand.quantize(B_hp, "nvfp4")
    res = gemm(opA, opB.mT, out_dtype="nvfp4", tuned=False)

    a_deq = dequant_operand(opA.qdata) * unpack_scale_blocked_to_2d(
        opA.scale.unsqueeze(0), m, k // 16
    )[0].float().repeat_interleave(16, -1)
    b_deq = dequant_operand(opB.qdata) * unpack_scale_blocked_to_2d(
        opB.scale.unsqueeze(0), n, k // 16
    )[0].float().repeat_interleave(16, -1)
    ref = a_deq @ b_deq.t()
    sf_2d = unpack_scale_blocked_to_2d(res.scale.unsqueeze(0), m, n // 16)[0].float()
    deq = out_values(res.qdata) * sf_2d.repeat_interleave(16, -1)
    bound = sf_2d.repeat_interleave(16, -1) * 1.01 + 1e-2
    assert ((deq - ref).abs() <= bound).all()


def test_quant_out_blockscaled_input_mx():
    """mxfp8 blockscaled inputs (real e8m0 SFA/SFB) with mxfp8 quantized
    output. On SM120 this exercises the widened 32-column warp run together
    with real SF operand fragments; the wide row magnitudes make any SF byte
    misplacement (operand or output side) exceed the quantization bound."""
    torch.manual_seed(0)
    m, n, k = 256, 512, 256
    row_a = torch.logspace(-3, 3, m, device="cuda").bfloat16()[:, None]
    row_b = torch.logspace(-2, 2, n, device="cuda").bfloat16()[:, None]
    A_hp = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * row_a / k**0.5
    B_hp = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * row_b / k**0.5
    opA = BlockScaledOperand.quantize(A_hp, "mxfp8_e4m3")
    opB = BlockScaledOperand.quantize(B_hp, "mxfp8_e4m3")
    res = gemm(opA, opB.mT, out_dtype="mxfp8_e4m3", tuned=False)

    a_deq = opA.qdata.float() * unpack_scale_blocked_to_2d(opA.scale.unsqueeze(0), m, k // 32)[
        0
    ].float().repeat_interleave(32, -1)
    b_deq = opB.qdata.float() * unpack_scale_blocked_to_2d(opB.scale.unsqueeze(0), n, k // 32)[
        0
    ].float().repeat_interleave(32, -1)
    ref = a_deq @ b_deq.t()
    sf_2d = unpack_scale_blocked_to_2d(res.scale.unsqueeze(0), m, n // 32)[0].float()
    scale = sf_2d.repeat_interleave(32, -1)
    deq = res.qdata.float() * scale
    assert ((deq - ref).abs() <= scale * 16.0 * 1.05 + 1e-2).all()


def test_quant_out_preallocated():
    torch.manual_seed(0)
    m, n, k = 256, 256, 128
    A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(m, n, dtype=torch.float8_e4m3fn, device="cuda")
    out_sf = torch.empty(m // 128, n // 128, 32, 4, 4, dtype=torch.float8_e8m0fnu, device="cuda")
    container = BlockScaledOperand.from_parts(out, out_sf, "mxfp8_e4m3")
    res = gemm(A, B, out=container, out_dtype="mxfp8_e4m3", tuned=False)
    assert res.qdata is out and res.scale is out_sf
    ref = gemm(A, B, out_dtype="mxfp8_e4m3", tuned=False)
    assert torch.equal(out.view(torch.uint8), ref.qdata.view(torch.uint8))
    assert torch.equal(out_sf.view(torch.uint8), ref.scale.view(torch.uint8))


def test_quant_out_varlen_k():
    """varlen_k with quantized output: (L, m, n) out, per-batch SFD."""
    torch.manual_seed(0)
    m, n = 256, 512
    seqk = [192, 64, 320]
    L = len(seqk)
    cu_k = torch.tensor([0] + list(torch.tensor(seqk).cumsum(0)), dtype=torch.int32, device="cuda")
    total_k = sum(seqk)
    A = (torch.randn(total_k, m, dtype=torch.bfloat16, device="cuda") / 4).T  # m-major
    B = torch.randn(total_k, n, dtype=torch.bfloat16, device="cuda") / 4  # n-major
    res = gemm(A, B, out_dtype="mxfp8_e4m3", cu_seqlens_k=cu_k, tuned=False)
    out, out_sf = res.qdata, res.scale
    assert out.shape == (L, m, n) and out_sf.shape == (L, 2, 4, 32, 4, 4)
    for b in range(L):
        ref = A[:, cu_k[b] : cu_k[b + 1]].float() @ B[cu_k[b] : cu_k[b + 1]].float()
        sf_2d = unpack_scale_blocked_to_2d(out_sf[b : b + 1], m, n // 32)[0].float()
        scale = sf_2d.repeat_interleave(32, -1)
        deq = out[b].float() * scale
        assert ((deq - ref).abs() <= scale * 16.0 * 1.05 + 1e-2).all()


@pytest.mark.parametrize("fmt", ["mxfp8_e4m3", "nvfp4"])
def test_quant_out_varlen_m(fmt):
    """varlen_m with quantized output: (total_m, n) out, one M-padded SFD buffer
    with tile-aligned per-batch padding (feeds the next varlen_m blockscaled
    GEMM's A input directly)."""
    skip_unsupported(fmt)
    torch.manual_seed(0)
    n, k = 512, 256
    seqm = [200, 128, 300]  # non-multiples of 128 included
    L = len(seqm)
    val_dtype, _, vec = fmt_props(fmt)
    cu_m = torch.tensor([0] + list(torch.tensor(seqm).cumsum(0)), dtype=torch.int32, device="cuda")
    total_m = sum(seqm)
    A = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda") / 4
    B = torch.randn(L, k, n, dtype=torch.bfloat16, device="cuda") / 4
    res = gemm(A, B, out_dtype=fmt, cu_seqlens_m=cu_m, tuned=False)
    out, out_sf = res.qdata, res.scale
    exp_rm = ceil_div(total_m, 128) + (L - 1)
    assert out_sf.shape == (1, exp_rm, ceil_div(n, 4 * vec), 32, 4, 4)
    sf_all = unpack_scale_blocked_to_2d(out_sf, exp_rm * 128, n // vec)[0].float()
    half_gap = 16.0 if val_dtype == torch.float8_e4m3fn else 1.0
    for b in range(L):
        lo, hi = cu_m[b].item(), cu_m[b + 1].item()
        ref = A[lo:hi].float() @ B[b].float()
        row0 = (lo // 128 + b) * 128  # tile-aligned padded row offset
        scale = sf_all[row0 : row0 + hi - lo].repeat_interleave(vec, -1)
        deq = out_values(out[lo:hi]) * scale
        bound = scale * half_gap * 1.05 + 1e-2
        if fmt == "nvfp4":
            bound = bound + 6.0 * 2.0**-10  # subnormal e4m3 SF rounding
        assert ((deq - ref).abs() <= bound).all(), f"batch {b}"


def test_quant_out_varlen_m_chain():
    """MoE-style chain: varlen_m bf16 -> nvfp4 quantized out, consumed directly
    as the blockscaled A input of a varlen_m nvfp4 GEMM."""
    skip_unsupported("nvfp4")
    torch.manual_seed(1)
    seqm = [200, 128, 300]
    L = len(seqm)
    cu_m = torch.tensor([0] + list(torch.tensor(seqm).cumsum(0)), dtype=torch.int32, device="cuda")
    total_m = sum(seqm)
    k1, n1, n2 = 256, 512, 256
    A = torch.randn(total_m, k1, dtype=torch.bfloat16, device="cuda") / 4
    W1 = torch.randn(L, k1, n1, dtype=torch.bfloat16, device="cuda") / 4
    x1_op = gemm(A, W1, out_dtype="nvfp4", cu_seqlens_m=cu_m, tuned=False)

    W2 = torch.randn(L, n1, n2, dtype=torch.bfloat16, device="cuda") / 16
    w2_ops = [BlockScaledOperand.quantize(W2[b].t().contiguous(), "nvfp4") for b in range(L)]
    qw2 = torch.stack([op.qdata for op in w2_ops])
    sfw2 = torch.stack([op.scale for op in w2_ops])
    w2_op = BlockScaledOperand.from_parts(qw2, sfw2, "nvfp4")
    out2 = gemm(x1_op, w2_op.mT, out_dtype=torch.bfloat16, cu_seqlens_m=cu_m, tuned=False)

    exp_rm = ceil_div(total_m, 128) + (L - 1)
    sf_all = unpack_scale_blocked_to_2d(x1_op.scale, exp_rm * 128, n1 // 16)[0].float()
    for b in range(L):
        lo, hi = cu_m[b].item(), cu_m[b + 1].item()
        row0 = (lo // 128 + b) * 128
        x1 = dequant_operand(x1_op.qdata[lo:hi]) * sf_all[row0 : row0 + hi - lo].repeat_interleave(
            16, -1
        )
        sfw_2d = unpack_scale_blocked_to_2d(sfw2[b : b + 1], n2, n1 // 16)[0].float()
        w2 = dequant_operand(qw2[b]) * sfw_2d.repeat_interleave(16, -1)
        ref2 = x1 @ w2.t()
        rel = (out2[lo:hi].float() - ref2).abs().max().item() / ref2.abs().max().item()
        assert rel < 0.03, f"batch {b}: {rel}"


def col_quant_ref(x, fmt):
    """Column quantize reference: SF per vec rows of each column.
    Returns (sf_bytes (n, m//vec), scale (m//vec, n))."""
    val_dtype, scale_dtype, vec = fmt_props(fmt)
    dmax = DTYPE_MAX[val_dtype]
    m, n = x.shape
    amax = x.reshape(m // vec, vec, n).abs().amax(1)
    scale = amax / dmax
    if scale_dtype == torch.float8_e8m0fnu:
        mant, e = torch.frexp(scale)
        e = torch.where(mant == 0.5, e - 1, e)
        e = torch.where(scale == 0, torch.full_like(e, -127), e).clamp(-127, 127)
        return (e + 127).to(torch.uint8).T.contiguous(), torch.pow(2.0, e.float())
    sf8 = scale.clamp(max=448.0).to(torch.float8_e4m3fn)
    return sf8.view(torch.uint8).T.contiguous(), sf8.float()


def test_quant_out_col_exact():
    """Column-direction SFD (SF vectors along M, for backward consumers):
    B = identity => SF bytes exact. fp8 values only (fp4 packs along N)."""
    skip_unsupported(col=True)
    torch.manual_seed(0)
    fmt = "mxfp8_e4m3"
    m, n, k = 256, 256, 128
    _, _, vec = fmt_props(fmt)
    A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    B = torch.eye(k, n, dtype=torch.bfloat16, device="cuda")
    res = gemm(A, B, out_dtype=fmt, out_quant_dim=-2, tuned=False)
    assert res.quant_dim == -2
    ref = A.float() @ B.float()
    sf_ref, scale_ref = col_quant_ref(ref, fmt)
    sf_2d = unpack_scale_blocked_to_2d(res.scale.unsqueeze(0), n, m // vec)[0]
    assert torch.equal(sf_2d.view(torch.uint8), sf_ref)
    scale_mn = sf_2d.float().T.repeat_interleave(vec, 0)
    deq = out_values(res.qdata) * scale_mn
    bound = scale_mn * 16.0 * 1.05 + 1e-2
    assert ((deq - ref).abs() <= bound).all()


def test_quant_out_col_random():
    """Column SFD with random B, ragged M (partial stripes) and batching."""
    skip_unsupported(col=True)
    torch.manual_seed(0)
    l, m, n, k = 2, 200, 512, 256
    A = torch.randn(l, m, k, dtype=torch.bfloat16, device="cuda") / 4
    B = torch.randn(l, k, n, dtype=torch.bfloat16, device="cuda") / 4
    res = gemm(A, B, out_dtype="mxfp8_e4m3", out_quant_dim=-2, tuned=False)
    out, out_sf = res.qdata, res.scale
    assert out_sf.shape == (l, n // 128, 2, 32, 4, 4)
    for li in range(l):
        ref = A[li].float() @ B[li].float()
        sf_2d = unpack_scale_blocked_to_2d(out_sf[li : li + 1], n, 8)[0].float()
        scale_mn = sf_2d.T.repeat_interleave(32, 0)[:m]
        deq = out[li].float() * scale_mn
        assert ((deq - ref).abs() <= scale_mn * 16.0 * 1.05 + 1e-2).all()


@pytest.mark.parametrize("fmt", ["mxfp8_e4m3", "nvfp4"])
def test_quant_out_transposed(fmt):
    """out_transposed (out_quant_dim=-2): D^T as a (N, M) row-quantized operand
    — the swapped-GEMM route through the ordinary row path, giving values
    contiguous along the consumer's K = M. B = identity makes SF bytes and
    values exact against row-quantizing ref^T. fp4 packs along M here, which
    the n-major col path cannot express; for fp8 the SF bytes and values must
    also be bit-identical to the col path's (same amax sets, same cvt)."""
    torch.manual_seed(0)
    m, n, k = 256, 320, 256
    _, _, vec = fmt_props(fmt)
    A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    B = torch.eye(k, n, dtype=torch.bfloat16, device="cuda")
    res = gemm(A, B, out_dtype=fmt, out_quant_dim=-2, out_transposed=True, tuned=False)
    assert isinstance(res, BlockScaledOperand) and res.qdata.shape[0] == n
    ref_t = (A.float() @ B.float()).T.contiguous()  # (n, m)
    q_ref, sf_ref, _ = quant_ref(ref_t, fmt)
    sf_2d = unpack_scale_blocked_to_2d(res.scale.unsqueeze(0), n, ceil_div(m, vec))[0]
    assert torch.equal(sf_2d.view(torch.uint8), sf_ref), "SF bytes mismatch"
    torch.testing.assert_close(out_values(res.qdata), q_ref, rtol=0, atol=0)
    if fmt == "mxfp8_e4m3":
        res_col = gemm(A, B, out_dtype=fmt, out_quant_dim=-2, tuned=False)
        assert torch.equal(res.scale.view(torch.uint8), res_col.scale.view(torch.uint8))
        assert torch.equal(
            res.qdata.view(torch.uint8).T.contiguous(), res_col.qdata.view(torch.uint8)
        )


def test_quant_out_16dp_row():
    """tile_m 64 (16dp256b tmem load): row SFD stays bit-exact; the layout-
    derived store width falls back to byte stores (thread slots interleave)."""
    from quack.gemm import gemm as gemm_lowlevel

    torch.manual_seed(0)
    m, n, k = 256, 256, 128
    A = torch.randn(1, m, k, dtype=torch.bfloat16, device="cuda")
    B = torch.eye(n, k, dtype=torch.bfloat16, device="cuda").unsqueeze(0)
    D = torch.empty(1, m, n, dtype=torch.float8_e4m3fn, device="cuda")
    SFD = torch.full((1, 2, 2, 32, 4, 4), 255, dtype=torch.uint8, device="cuda").view(
        torch.float8_e8m0fnu
    )
    gemm_lowlevel(A, B, D, None, None, 64, 128, 1, 1, SFD=SFD)
    torch.cuda.synchronize()
    ref = A[0].float() @ B[0].float().T
    q_ref, sf_ref, _ = quant_ref(ref, "mxfp8_e4m3")
    sf_2d = unpack_scale_blocked_to_2d(SFD, m, n // 32)[0]
    assert torch.equal(sf_2d.view(torch.uint8), sf_ref)
    torch.testing.assert_close(D[0].float(), q_ref, rtol=0, atol=0)


def test_quant_out_regression_no_sfd():
    """Adding the SFD epi op must not change the plain-output path."""
    torch.manual_seed(0)
    m, n, k = 256, 256, 128
    A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
    out = gemm(A, B, tuned=False)
    ref = A.float() @ B.float()
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("fmt", ["mxfp8_e4m3", "mxfp4", "nvfp4"])
def test_quant_out_stochastic_rounding(fmt):
    """RS quantized output: SF bytes stay RN/ceil (bit-equal to the RN run);
    values are stochastically rounded through cvt.rs (hw fp8x4/e2m1x4 on
    sm_100a/103a, sw emulation elsewhere) — deterministic per seed,
    seed-sensitive, within one full quantization bin of the reference, and
    unbiased (the seed-average converges to the reference, unlike RN whose
    error is parked at up to half a bin)."""
    from quack.rounding import RoundingMode

    torch.manual_seed(0)
    m, n, k = 256, 256, 256
    device = "cuda"
    val_dtype, _, vec = fmt_props(fmt)
    A = torch.randn(m, k, dtype=torch.bfloat16, device=device) / k**0.25
    B = torch.randn(k, n, dtype=torch.bfloat16, device=device) / k**0.25

    def run_rs(seed):
        return gemm(A, B, out_dtype=fmt, tuned=False, rounding_mode=RoundingMode.RS, sr_seed=seed)

    res_rn = gemm(A, B, out_dtype=fmt, tuned=False)
    res1, res1b, res2 = run_rs(7), run_rs(7), run_rs(8)
    # The SF/rescale path is rounding-mode invariant: bytes bit-equal to RN.
    assert torch.equal(res1.scale.view(torch.uint8), res_rn.scale.view(torch.uint8))
    assert torch.equal(res1.qdata.view(torch.uint8), res1b.qdata.view(torch.uint8))
    assert not torch.equal(res1.qdata.view(torch.uint8), res2.qdata.view(torch.uint8))
    assert not torch.equal(res1.qdata.view(torch.uint8), res_rn.qdata.view(torch.uint8))

    ref = A.float() @ B.float()
    sf_2d = unpack_scale_blocked_to_2d(res1.scale.unsqueeze(0), m, ceil_div(n, vec))[0].float()
    scale = sf_2d.repeat_interleave(vec, -1)[:, :n]
    # SR picks the floor or ceil neighbor: error within one FULL bin gap
    # (RN is within half), in units of the scale.
    full_gap = 32.0 if val_dtype == torch.float8_e4m3fn else 2.0
    bound = scale * full_gap * 1.05 + 1e-2
    deq1 = out_values(res1.qdata) * scale
    assert ((deq1 - ref).abs() <= bound).all(), (
        f"max err {(deq1 - ref).abs().max().item()} vs bound {bound.max().item()}"
    )
    # Unbiasedness: the mean over seeds converges to ref. Normalized by the
    # local bin gap, the mean of n_seeds samples has std <= 0.5/sqrt(n_seeds)
    # ~ 0.055; RN's deterministic error is uniform in [-0.5, 0.5] (mean |err|
    # ~ 0.25), so the thresholds separate the two cleanly. The local gap is
    # bounded below by ulp(q): for e4m3 that's q/8 at the wide end of a
    # binade, for e2m1 the {0.5, 1} gaps; use the reference-magnitude bin.
    n_seeds = 80
    acc = torch.zeros_like(ref)
    for s in range(n_seeds):
        acc += out_values(run_rs(1000 + s).qdata)
    mean_deq = (acc / n_seeds) * scale
    if val_dtype == torch.float8_e4m3fn:
        gap = (2.0 ** torch.floor(torch.log2((ref / scale).abs().clamp(min=2**-6)))) / 8
        gap = gap.clamp(min=2**-9)
    else:
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)
        idx = torch.searchsorted(grid, (ref / scale).abs().clamp(max=6.0)).clamp(1, 7)
        gap = grid[idx] - grid[idx - 1]
    bias = ((mean_deq - ref) / (gap * scale)).abs()
    assert bias.mean() < 0.12, f"mean bias {bias.mean():.3f} bins (SR should be ~0.045)"
    assert bias.max() < 0.6, f"max bias {bias.max():.3f} bins"


# --- Quantized gated postact (swiglu FC1 fusion) via the minted-mod path ------


@pytest.mark.parametrize("fmt", ["mxfp8_e4m3", "mxfp4", "nvfp4"])
def test_quant_postact_gated(fmt):
    """gemm + swiglu + quantized postact: SF slots live in acc space (one
    vector of vec postact values = 2*vec acc columns of interleaved gate/up)."""
    skip_unsupported(fmt, aux=True)
    from quack.epilogue.library import swiglu_quant_mod

    torch.manual_seed(0)
    l, m, N, k = 1, 256, 1024, 256
    n_post = N // 2
    val_dtype, sf_dtype, vec = fmt_props(fmt)
    n_stored = n_post // 2 if val_dtype == torch.float4_e2m1fn_x2 else n_post
    A = torch.randn(l, m, k, dtype=torch.bfloat16, device="cuda") / k**0.25
    B = torch.randn(l, N, k, dtype=torch.bfloat16, device="cuda") / k**0.25
    postact = torch.empty(l, m, n_stored, dtype=val_dtype, device="cuda")
    sf = torch.empty(
        l, ceil_div(m, 128), ceil_div(n_post, 4 * vec), 32, 4, 4, dtype=sf_dtype, device="cuda"
    )
    swiglu_quant_mod.gemm(
        A,
        B,
        None,
        epi_args=dict(postact=postact, postact_sf=sf),
        tile_M=128,
        tile_N=256,
        cluster_M=1,
        cluster_N=1,
    )
    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    post_ref = torch.nn.functional.silu(x[..., 0::2]) * x[..., 1::2]
    sf_2d = unpack_scale_blocked_to_2d(sf, m, n_post // vec).float()
    scale = sf_2d.repeat_interleave(vec, -1)
    deq = out_values(postact[0]) * scale
    half_gap = 16.0 if val_dtype == torch.float8_e4m3fn else 1.0
    bound = scale * half_gap * 1.05 + 1e-2
    if fmt == "nvfp4":
        bound = bound + 6.0 * 2.0**-10  # subnormal e4m3 SF rounding
    assert ((deq - post_ref[0]).abs() <= bound).all(), (
        f"max err {(deq - post_ref[0]).abs().max().item()}"
    )


def test_quant_postact_gated_exact():
    """reglu (relu(gate)*up) postact matches a torch fp32 reference bitwise,
    so the mxfp8 SF bytes and values must be exact."""
    skip_unsupported(aux=True)
    from quack.epilogue.library import gated_quant_mod

    torch.manual_seed(0)
    l, m, N, k = 1, 256, 512, 128
    n_post = N // 2
    A = torch.randn(l, m, k, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(l, N, k, dtype=torch.bfloat16, device="cuda")
    postact = torch.empty(l, m, n_post, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty(
        l,
        ceil_div(m, 128),
        ceil_div(n_post, 128),
        32,
        4,
        4,
        dtype=torch.float8_e8m0fnu,
        device="cuda",
    )
    gated_quant_mod("reglu").gemm(
        A,
        B,
        None,
        epi_args=dict(postact=postact, postact_sf=sf),
        tile_M=128,
        tile_N=256,
        cluster_M=1,
        cluster_N=1,
    )
    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    post_ref = (x[..., 0::2].relu() * x[..., 1::2])[0]
    q_ref, sf_ref, _ = quant_ref(post_ref, "mxfp8_e4m3")
    sf_2d = unpack_scale_blocked_to_2d(sf, m, n_post // 32)[0]
    assert torch.equal(sf_2d.view(torch.uint8), sf_ref)
    torch.testing.assert_close(postact[0].float(), q_ref, rtol=0, atol=0)


def test_quant_postact_nvfp4_norm_const():
    """nvfp4 postact with the per-tensor second level folded via sfd_norm_const."""
    skip_unsupported("nvfp4", aux=True)
    from quack.epilogue.library import swiglu_quant_mod

    torch.manual_seed(0)
    l, m, N, k = 1, 256, 1024, 256
    n_post = N // 2
    A = torch.randn(l, m, k, dtype=torch.bfloat16, device="cuda") / k**0.25
    B = torch.randn(l, N, k, dtype=torch.bfloat16, device="cuda") / k**0.25
    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    post_ref = (torch.nn.functional.silu(x[..., 0::2]) * x[..., 1::2])[0]
    pts = (post_ref.abs().max() / (448.0 * 6.0)).item()
    postact = torch.empty(l, m, n_post // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    sf = torch.empty(
        l,
        ceil_div(m, 128),
        ceil_div(n_post, 64),
        32,
        4,
        4,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    swiglu_quant_mod.gemm(
        A,
        B,
        None,
        epi_args=dict(postact=postact, postact_sf=sf, sfd_norm_const=1.0 / pts),
        tile_M=128,
        tile_N=256,
        cluster_M=1,
        cluster_N=1,
    )
    sf_2d = unpack_scale_blocked_to_2d(sf, m, n_post // 16).float()
    scale = sf_2d.repeat_interleave(16, -1) * pts
    deq = out_values(postact[0]) * scale
    bound = scale * 1.05 + 1e-3
    assert ((deq - post_ref).abs() <= bound).all()
