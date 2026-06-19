import pytest
import torch

import cutlass

from quack.blockscaled_gemm_utils import (
    FP4_E2M1FN_VALUES,
    blockscaled_gemm_reference,
    compile_blockscaled_gemm_tvm_ffi,
    create_blockscaled_operand_quantized,
    create_blockscaled_operand_tensor,
    create_blockscaled_scale_tensor,
    create_sm120_blockscaled_scale_tensor,
    create_blockscaled_varlen_k_operands,
    create_blockscaled_varlen_m_operands,
    scale_blocked_for_cublas,
    scale_view_for_kernel,
)
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.gemm_default_epi import GemmDefaultSm100, GemmDefaultSm120
from quack.varlen_utils import VarlenArguments
from quack.mx_utils import to_blocked


def _skip_if_not_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major = torch.cuda.get_device_properties(0).major
    if major not in (10, 11):
        pytest.skip("SM100/SM110 required")


def _skip_if_not_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major = torch.cuda.get_device_properties(0).major
    if major != 12:
        pytest.skip("SM120 required")


def _compile_blockscaled_gemm(
    ab_dtype,
    sf_dtype,
    sf_vec_size,
    d_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    m,
    n,
    k,
    l,
):
    a_ref, mA = create_blockscaled_operand_tensor(l, m, k, False, ab_dtype)
    b_ref, mB = create_blockscaled_operand_tensor(l, n, k, False, ab_dtype)
    _, mD = create_blockscaled_operand_tensor(l, m, n, False, d_dtype, init="empty")
    sfa_ref, mSFA = create_blockscaled_scale_tensor(l, m, k, sf_vec_size, sf_dtype)
    sfb_ref, mSFB = create_blockscaled_scale_tensor(l, n, k, sf_vec_size, sf_dtype)
    compiled = compile_blockscaled_gemm_tvm_ffi(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        d_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        mA,
        mB,
        mD,
        mSFA,
        mSFB,
    )
    return (
        compiled,
        (mA, mB, mD, mSFA, mSFB),
        (a_ref, b_ref, sfa_ref, sfb_ref, mD),
    )


def _run_blockscaled_gemm(compiled, args):
    mA, mB, mD, mSFA, mSFB = args
    compiled(mA, mB, mD, mSFA, mSFB)
    torch.cuda.synchronize()


def test_blockscaled_validation():
    assert GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (128, 64),
        (1, 1),
        256,
        64,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (128, 192),
        (1, 1),
        256,
        192,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (128, 128),
        (1, 1),
        256,
        256,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float4E2M1FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.Float32,
        (128, 128),
        (1, 1),
        256,
        256,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        cutlass.Float32,
        (128, 192),
        (1, 1),
        256,
        192,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (256, 384),
        (2, 1),
        256,
        512,
        256,
        1,
        "k",
        "k",
        "n",
    )


def test_sm120_blockscaled_validation():
    assert GemmDefaultSm120.can_implement_blockscaled(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        cutlass.BFloat16,
        (128, 128, 64),
        (1, 1),
        128,
        128,
        64,
        1,
        "k",
        "k",
        "n",
    )
    assert GemmDefaultSm120.can_implement_blockscaled(
        cutlass.Float4E2M1FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (128, 128, 64),
        (1, 1),
        256,
        256,
        128,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm120.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (128, 128, 64),
        (1, 1),
        128,
        128,
        64,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm120.can_implement_blockscaled(
        cutlass.Int8,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (128, 128, 64),
        (1, 1),
        128,
        128,
        64,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm120.can_implement_blockscaled(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        32,
        cutlass.BFloat16,
        (128, 128, 64),
        (1, 1),
        128,
        128,
        64,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm120.can_implement_blockscaled(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        cutlass.BFloat16,
        (128, 128, 64),
        (1, 1),
        128,
        128,
        96,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.Float32,
        (256, 224),
        (2, 1),
        256,
        448,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float4E2M1FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.Float32,
        (256, 384),
        (2, 1),
        256,
        512,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (64, 128),
        (1, 1),
        256,
        256,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        32,
        cutlass.Float32,
        (128, 128),
        (1, 1),
        256,
        256,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (256, 128),
        (1, 1),
        512,
        256,
        256,
        1,
        "k",
        "k",
        "n",
    )


def test_sm120_blockscaled_class_call_validation():
    m = n = 128
    k = 64
    l = 1
    gemm = GemmDefaultSm120(
        cutlass.Float32,
        cutlass.Float4E2M1FN,
        (128, 128, 64),
        (1, 1, 1),
        is_persistent=False,
        sf_vec_size=16,
        sf_dtype=cutlass.Float8E4M3FN,
    )
    mA = fake_tensor(cutlass.Float4E2M1FN, (m, k, l), leading_dim=1, divisibility=4)
    mB = fake_tensor(cutlass.Float4E2M1FN, (n, k, l), leading_dim=1, divisibility=4)
    mD = fake_tensor(cutlass.BFloat16, (m, n, l), leading_dim=1, divisibility=8)
    mSFA = fake_tensor(cutlass.Float8E4M3FN, (m, 16, l), leading_dim=1, divisibility=4)
    mSFB = fake_tensor(cutlass.Float8E4M3FN, (n, 16, l), leading_dim=1, divisibility=4)

    assert (
        gemm._validate_blockscaled_call(
            mA,
            mB,
            mD,
            None,
            mSFA,
            mSFB,
            gemm.EpilogueArguments(),
            None,
            None,
            None,
        )
        == VarlenArguments()
    )
    with pytest.raises(ValueError, match="requires SFA and SFB"):
        gemm._validate_blockscaled_call(
            mA, mB, mD, None, None, mSFB, gemm.EpilogueArguments(), None, None, None
        )
    with pytest.raises(NotImplementedError, match="C/beta"):
        gemm._validate_blockscaled_call(
            mA, mB, mD, mD, mSFA, mSFB, gemm.EpilogueArguments(), None, None, None
        )
    packed_k_a = fake_tensor(cutlass.Float4E2M1FN, (m, k // 2, l), leading_dim=1, divisibility=4)
    packed_k_b = fake_tensor(cutlass.Float4E2M1FN, (n, k // 2, l), leading_dim=1, divisibility=4)
    with pytest.raises(ValueError, match="expects logical Float4E2M1FN K extent"):
        gemm._validate_blockscaled_call(
            packed_k_a,
            packed_k_b,
            mD,
            None,
            mSFA,
            mSFB,
            gemm.EpilogueArguments(),
            None,
            None,
            None,
        )


@pytest.mark.parametrize(
    "ab_dtype,sf_dtype,sf_vec_size,d_dtype,mma_tiler_mn,cluster_shape_mn,m,n,k,l",
    [
        (
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (128, 64),
            (1, 1),
            256,
            64,
            256,
            1,
        ),
        (
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (128, 192),
            (1, 1),
            256,
            192,
            256,
            1,
        ),
        (
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (128, 128),
            (1, 1),
            256,
            256,
            256,
            1,
        ),
        (
            cutlass.Float8E5M2,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (256, 64),
            (2, 1),
            512,
            64,
            256,
            1,
        ),
        (
            cutlass.Float8E5M2,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (256, 192),
            (2, 1),
            512,
            192,
            256,
            1,
        ),
        (
            cutlass.Float8E5M2,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (256, 128),
            (2, 1),
            512,
            256,
            256,
            1,
        ),
        (
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            (256, 192),
            (2, 1),
            256,
            192,
            256,
            1,
        ),
        (
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            (256, 224),
            (2, 1),
            256,
            224,
            256,
            1,
        ),
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            (128, 128),
            (1, 1),
            256,
            256,
            256,
            1,
        ),
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            (256, 224),
            (2, 1),
            256,
            224,
            256,
            1,
        ),
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E8M0FNU,
            16,
            cutlass.Float32,
            (128, 64),
            (1, 1),
            256,
            64,
            256,
            1,
        ),
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.Float32,
            (256, 192),
            (2, 1),
            256,
            192,
            256,
            1,
        ),
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.Float32,
            (128, 192),
            (1, 1),
            256,
            192,
            256,
            1,
        ),
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.Float32,
            (256, 224),
            (2, 1),
            256,
            224,
            256,
            1,
        ),
    ],
)
def test_blockscaled_correctness(
    ab_dtype, sf_dtype, sf_vec_size, d_dtype, mma_tiler_mn, cluster_shape_mn, m, n, k, l
):
    _skip_if_not_sm100()

    (
        compiled,
        args,
        (a_ref, b_ref, sfa_ref, sfb_ref, _),
    ) = _compile_blockscaled_gemm(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        d_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
    )
    _run_blockscaled_gemm(compiled, args)

    _, _, d_torch, _, _ = args
    ref = blockscaled_gemm_reference(a_ref, b_ref, sfa_ref, sfb_ref)
    err = (d_torch.float() - ref).abs().max().item()
    tol = 5e-3 if d_dtype != cutlass.Float32 else 5e-4
    assert err < tol, f"max_err={err}"


# ---------------------------------------------------------------------------
# Scale layout invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mn,sf_k,l", [(128, 4, 1), (256, 16, 1), (384, 12, 2), (512, 8, 1)])
def test_scale_layout_matches_cublas(mn, sf_k, l):
    """The quack kernel scale-view and cuBLAS's to_blocked must share the
    same underlying byte layout (they both represent the PTX
    tcgen05 scale-factor atom, tiled in the same outer order)."""
    torch.manual_seed(0)
    # a 2D uint8 scale slice per batch
    scale_2d = torch.randint(0, 255, (l, mn, sf_k), device="cuda", dtype=torch.uint8)

    # Build our contiguous scale storage via create_blockscaled_operand_quantized's
    # rearrangement logic: pad + (l, rm, 128, rk, 4) -> (l, rm, rk, 512)
    rm = (mn + 127) // 128
    rk = (sf_k + 3) // 4
    mn_pad = rm * 128
    sf_k_pad = rk * 4
    padded = torch.zeros(l, mn_pad, sf_k_pad, device="cuda", dtype=torch.uint8)
    padded[:, :mn, :sf_k] = scale_2d
    blocks = padded.view(l, rm, 128, rk, 4).permute(0, 1, 3, 2, 4)
    blocks = blocks.reshape(l, rm, rk, 4, 32, 4).transpose(3, 4).contiguous()
    scale_contig = blocks.view(l, rm, rk, 512)  # (l, rm, rk, 512)

    # kernel view indexing: byte offset within tile = (m%32)*16 + ((m//32)%4)*4 + (k%4)
    kv = scale_view_for_kernel(scale_contig.view(torch.float8_e8m0fnu), mn, sf_k, l).view(
        torch.uint8
    )
    m_positions = sorted({0, 1, 17, 31, 33, 127, min(128, mn - 1), mn - 1} & set(range(mn)))
    k_positions = sorted({0, 1, 3, 4, 7, sf_k - 1} & set(range(sf_k)))
    for li in range(l):
        for mi in m_positions:
            for ki in k_positions:
                byte_off = (mi % 32) * 16 + ((mi // 32) % 4) * 4 + (ki % 4)
                assert kv[li, mi // 128, ki // 4, byte_off].item() == scale_2d[li, mi, ki].item(), (
                    f"mismatch at l={li} m={mi} k={ki}"
                )

    # cuBLAS slice must equal to_blocked(scale_2d[l])
    for li in range(l):
        ours = scale_blocked_for_cublas(scale_contig.view(torch.float8_e8m0fnu), mn, sf_k, li).view(
            torch.uint8
        )
        ref = to_blocked(scale_2d[li])
        assert torch.equal(ours, ref), f"to_blocked mismatch at l={li}"


@pytest.mark.parametrize(
    "k,sf_vec_size,expected_cols",
    [
        (64, 16, 16),
        (128, 16, 16),
        (256, 16, 16),
        (384, 16, 32),
        (64, 32, 16),
        (256, 32, 16),
        (576, 32, 32),
    ],
)
def test_sm120_blockscaled_padded_scale_layout(k, sf_vec_size, expected_cols):
    _skip_if_not_sm120()
    mn, l = 128, 1
    sf_dtype = cutlass.Float8E4M3FN if sf_vec_size == 16 else cutlass.Float8E8M0FNU
    ref, physical = create_sm120_blockscaled_scale_tensor(l, mn, k, sf_vec_size, sf_dtype)
    assert tuple(physical.shape) == (mn, expected_cols, l)
    assert tuple(ref.shape) == (mn, k, l)

    logical_cols = (k + sf_vec_size - 1) // sf_vec_size
    if expected_cols > logical_cols:
        padding = physical[:, logical_cols:, :].view(torch.uint8)
        assert torch.any(padding != 0)


def test_sm120_blockscaled_scale_helper_validation():
    _skip_if_not_sm120()
    with pytest.raises(ValueError, match="K divisible by 64"):
        create_sm120_blockscaled_scale_tensor(1, 128, 96, 16, cutlass.Float8E4M3FN)
    with pytest.raises(ValueError, match="sf_vec_size 16 or 32"):
        create_sm120_blockscaled_scale_tensor(1, 128, 64, 8, cutlass.Float8E4M3FN)
    with pytest.raises(ValueError, match="sf_vec_size=16 requires"):
        create_sm120_blockscaled_scale_tensor(1, 128, 64, 16, cutlass.Float8E8M0FNU)
    with pytest.raises(ValueError, match="sf_vec_size=32 requires"):
        create_sm120_blockscaled_scale_tensor(1, 128, 64, 32, cutlass.Float8E4M3FN)


def _pack_sm120_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    packed = torch.empty(
        (codes.shape[0], codes.shape[1] // 2, 1),
        device=codes.device,
        dtype=torch.float4_e2m1fn_x2,
    )
    packed.view(torch.uint8).copy_(codes[:, 0::2, None] | (codes[:, 1::2, None] << 4))
    return packed


def _sm120_fp4_blockscaled_reference(
    a_codes: torch.Tensor,
    b_codes: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    sf_vec_size: int,
) -> torch.Tensor:
    table = torch.tensor(FP4_E2M1FN_VALUES, dtype=torch.float32, device=a_codes.device)
    scale_k = torch.arange(a_codes.shape[1], device=a_codes.device) // sf_vec_size
    a = table[a_codes.long()] * sfa.float()[:, scale_k, 0]
    b = table[b_codes.long()] * sfb.float()[:, scale_k, 0]
    return torch.einsum("mk,nk->mn", a, b).unsqueeze(-1)


def _make_sm120_scales(mn, k, sf_vec_size, sf_dtype, row_or_col_sensitive=True):
    _, scales = create_sm120_blockscaled_scale_tensor(1, mn, k, sf_vec_size, sf_dtype)
    logical_cols = (k + sf_vec_size - 1) // sf_vec_size
    if sf_dtype == cutlass.Float8E8M0FNU:
        base = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
    else:
        base = torch.tensor([1.0, 2.0, 0.5, 1.5], device="cuda", dtype=torch.float32)
    for idx in range(mn):
        values = base[torch.arange(logical_cols, device="cuda") % base.numel()]
        if row_or_col_sensitive:
            values = values * (1.0 + 0.125 * (idx % 4))
        scales[idx, :logical_cols, 0] = values.to(scales.dtype)
    return scales


def _compile_sm120_blockscaled_gemm(ab_dtype, sf_dtype, sf_vec_size, m, n, k, mA, mB):
    l = 1
    _, mD = create_blockscaled_operand_tensor(l, m, n, False, cutlass.BFloat16, init="empty")
    mSFA = _make_sm120_scales(m, k, sf_vec_size, sf_dtype)
    mSFB = _make_sm120_scales(n, k, sf_vec_size, sf_dtype)
    compiled = compile_blockscaled_gemm_tvm_ffi(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        cutlass.BFloat16,
        (128, 128),
        (1, 1),
        mA,
        mB,
        mD,
        mSFA,
        mSFB,
    )
    return compiled, (mA, mB, mD, mSFA, mSFB)


@pytest.mark.parametrize(
    "sf_dtype,sf_vec_size,m,n,k",
    [
        (cutlass.Float8E4M3FN, 16, 128, 128, 64),
        (cutlass.Float8E4M3FN, 16, 256, 128, 128),
        (cutlass.Float8E4M3FN, 16, 128, 128, 320),
        (cutlass.Float8E8M0FNU, 32, 128, 128, 64),
    ],
)
def test_sm120_blockscaled_scale_correctness(sf_dtype, sf_vec_size, m, n, k):
    _skip_if_not_sm120()
    a_codes = torch.full((m, k), 0x2, device="cuda", dtype=torch.uint8)
    b_codes = torch.full((n, k), 0x2, device="cuda", dtype=torch.uint8)
    mA = _pack_sm120_fp4_codes(a_codes)
    mB = _pack_sm120_fp4_codes(b_codes)
    compiled, args = _compile_sm120_blockscaled_gemm(
        cutlass.Float4E2M1FN, sf_dtype, sf_vec_size, m, n, k, mA, mB
    )
    _run_blockscaled_gemm(compiled, args)

    _, _, d_torch, mSFA, mSFB = args
    ref = _sm120_fp4_blockscaled_reference(a_codes, b_codes, mSFA, mSFB, sf_vec_size).to(
        torch.bfloat16
    )
    err = (d_torch.float() - ref).abs().max().item()
    assert err < 1e-1, f"max_err={err}"


def test_sm120_blockscaled_k_loop_accumulates_before_bf16_store():
    _skip_if_not_sm120()
    m = n = 128
    k = 384
    sf_vec_size = 16
    sf_dtype = cutlass.Float8E4M3FN
    a_codes = torch.full((m, k), 0x2, device="cuda", dtype=torch.uint8)
    b_codes = torch.full((n, k), 0x2, device="cuda", dtype=torch.uint8)
    # Make the first K64 tile very large and the remaining K64 tiles small.
    # Storing BF16 after each K tile loses the later small partials; true FP32
    # accumulation keeps them until the final BF16 conversion.
    a_codes[:, :64] = 0x7
    b_codes[:, :64] = 0x7
    mA = _pack_sm120_fp4_codes(a_codes)
    mB = _pack_sm120_fp4_codes(b_codes)
    compiled, args = _compile_sm120_blockscaled_gemm(
        cutlass.Float4E2M1FN, sf_dtype, sf_vec_size, m, n, k, mA, mB
    )
    _, _, _, mSFA, mSFB = args
    logical_cols = k // sf_vec_size
    mSFA[:, :logical_cols, 0] = torch.tensor(1.0, device="cuda", dtype=mSFA.dtype)
    mSFB[:, :logical_cols, 0] = torch.tensor(1.0, device="cuda", dtype=mSFB.dtype)
    mSFA[:, :4, 0] = torch.tensor(3.0, device="cuda", dtype=mSFA.dtype)
    mSFB[:, :4, 0] = torch.tensor(3.0, device="cuda", dtype=mSFB.dtype)
    _run_blockscaled_gemm(compiled, args)

    _, _, d_torch, mSFA, mSFB = args
    ref = _sm120_fp4_blockscaled_reference(a_codes, b_codes, mSFA, mSFB, sf_vec_size).to(
        torch.bfloat16
    )
    torch.testing.assert_close(d_torch.float(), ref.float(), rtol=0, atol=0)


def test_sm120_blockscaled_asymmetric_fp4_and_scale_page_crossing():
    _skip_if_not_sm120()
    m = n = 128
    k = 320
    sf_vec_size = 16
    sf_dtype = cutlass.Float8E4M3FN
    ks = torch.arange(k, device="cuda")[None, :]
    a_codes = torch.where(
        (ks % 4) < 2,
        torch.tensor(0x2, device="cuda", dtype=torch.uint8),
        torch.tensor(0x4, device="cuda", dtype=torch.uint8),
    ).expand(m, k)
    b_codes = torch.where(
        (ks % 8) < 4,
        torch.tensor(0x3, device="cuda", dtype=torch.uint8),
        torch.tensor(0x5, device="cuda", dtype=torch.uint8),
    ).expand(n, k)
    mA = _pack_sm120_fp4_codes(a_codes)
    mB = _pack_sm120_fp4_codes(b_codes)
    compiled, args = _compile_sm120_blockscaled_gemm(
        cutlass.Float4E2M1FN, sf_dtype, sf_vec_size, m, n, k, mA, mB
    )
    _run_blockscaled_gemm(compiled, args)

    _, _, d_torch, mSFA, mSFB = args
    logical_cols = k // sf_vec_size
    assert mSFA.shape[1] == 32
    assert torch.any(mSFA[:, logical_cols:, :].float() != 1.0)
    ref = _sm120_fp4_blockscaled_reference(a_codes, b_codes, mSFA, mSFB, sf_vec_size).to(
        torch.bfloat16
    )
    err = (d_torch.float() - ref.float()).abs().max().item()
    assert err < 1e-1, f"max_err={err}"


def test_sm120_blockscaled_rejects_compact_scale_layout():
    _skip_if_not_sm120()
    l, m, n, k, sf_vec_size = 1, 128, 128, 64, 16
    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    _, mA = create_blockscaled_operand_tensor(l, m, k, False, ab_dtype)
    _, mB = create_blockscaled_operand_tensor(l, n, k, False, ab_dtype)
    _, mD = create_blockscaled_operand_tensor(l, m, n, False, cutlass.BFloat16, init="empty")
    mSFA = torch.empty((m, k // sf_vec_size, l), device="cuda", dtype=torch.float8_e4m3fn)
    mSFB = torch.empty((n, k // sf_vec_size, l), device="cuda", dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match="SFA shape"):
        runner = compile_blockscaled_gemm_tvm_ffi(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            cutlass.BFloat16,
            (128, 128),
            (1, 1),
            mA,
            mB,
            mD,
            mSFA,
            mSFB,
        )
        runner(mA, mB, mD, mSFA, mSFB)


# ---------------------------------------------------------------------------
# End-to-end: quantized MXFP8 inputs through quack kernel vs cuBLAS vs dequant ref
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mma_tiler_mn,cluster_shape_mn,m,n,k",
    [
        # All 5 supported blockscaled tile_n values (64, 128, 192, 224, 256).
        ((128, 64), (1, 1), 256, 64, 512),
        ((128, 128), (1, 1), 256, 256, 256),
        ((128, 128), (1, 1), 512, 512, 512),
        ((128, 192), (1, 1), 256, 192, 256),
        ((128, 256), (1, 1), 256, 256, 256),
        ((256, 128), (2, 1), 512, 256, 512),
        ((256, 192), (2, 1), 256, 192, 256),
        ((256, 192), (2, 1), 256, 384, 256),
        ((256, 192), (2, 1), 512, 192, 512),
        ((256, 224), (2, 1), 256, 224, 256),
        ((256, 224), (2, 1), 512, 224, 512),
        ((256, 256), (2, 1), 512, 256, 512),
    ],
)
def test_blockscaled_mxfp8_quantized(mma_tiler_mn, cluster_shape_mn, m, n, k):
    _skip_if_not_sm100()
    l, sf_vec = 1, 32

    torch.manual_seed(0)
    a_ref, mA, a_sc = create_blockscaled_operand_quantized(l, m, k, False, sf_vec)
    b_ref, mB, b_sc = create_blockscaled_operand_quantized(l, n, k, False, sf_vec)
    _, mD = create_blockscaled_operand_tensor(l, m, n, False, cutlass.BFloat16, init="empty")

    mSFA = scale_view_for_kernel(a_sc, m, k // sf_vec, l)
    mSFB = scale_view_for_kernel(b_sc, n, k // sf_vec, l)

    runner = compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        sf_vec,
        cutlass.BFloat16,
        mma_tiler_mn,
        cluster_shape_mn,
        mA,
        mB,
        mD,
        mSFA,
        mSFB,
    )
    runner(mA, mB, mD, mSFA, mSFB)
    torch.cuda.synchronize()

    # Reference: dequant matmul (a_ref/b_ref are already dequantized)
    d_ref = torch.einsum("mkl,nkl->mnl", a_ref, b_ref)
    err = (mD.float() - d_ref).abs().max().item()
    assert err < 5e-3, f"quack vs dequant max_err={err}"

    # cuBLAS: bit-exact match expected (same operand bits, same scale bytes, same hw MMA)
    from torch.nn.functional import scaled_mm as F_scaled_mm, ScalingType, SwizzleType

    a_cub = mA[:, :, 0].contiguous()
    b_cub = mB[:, :, 0].contiguous()
    a_sc_cub = scale_blocked_for_cublas(a_sc, m, k // sf_vec, 0)
    b_sc_cub = scale_blocked_for_cublas(b_sc, n, k // sf_vec, 0)
    out_cublas = F_scaled_mm(
        a_cub,
        b_cub.t(),
        scale_a=a_sc_cub,
        scale_recipe_a=ScalingType.BlockWise1x32,
        scale_b=b_sc_cub,
        scale_recipe_b=ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )
    assert torch.equal(mD.squeeze(-1), out_cublas), (
        f"quack != cuBLAS: max_err={(mD.squeeze(-1).float() - out_cublas.float()).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# High-level PyTorch interface
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape_mnk", [(256, 256, 256), (512, 256, 256), (128, 128, 256)])
@pytest.mark.parametrize("batched", [False, True])
def test_mxfp8_interface(shape_mnk, batched):
    _skip_if_not_sm100()
    from quack.gemm_blockscaled_interface import (
        mxfp8_gemm,
        mxfp8_gemm_cublas,
        mxfp8_gemm_ref,
        mxfp8_gemm_quantize,
        mxfp8_quantize,
    )

    M, N, K = shape_mnk
    L = 2 if batched else 1
    torch.manual_seed(0)
    shape_A = (L, M, K) if batched else (M, K)
    # Weight stored nn.Linear-style (N, K) row-major; pass .mT to get K-contig (K, N)
    shape_W = (L, N, K) if batched else (N, K)
    A_hp = torch.randn(*shape_A, device="cuda", dtype=torch.bfloat16) * K**-0.5
    W_hp = torch.randn(*shape_W, device="cuda", dtype=torch.bfloat16) * K**-0.5

    A_q, A_sc = mxfp8_quantize(A_hp)
    W_q, W_sc = mxfp8_quantize(W_hp)  # (..., N, K), (..., N, K/32)
    assert A_q.dtype == torch.float8_e4m3fn and A_sc.dtype == torch.float8_e8m0fnu

    B_q = W_q.mT  # (..., K, N) K-contig view
    B_sc = W_sc.mT  # (..., K/32, N) K-contig view

    out = mxfp8_gemm(A_q, B_q, A_sc, B_sc)
    assert out.shape == ((L, M, N) if batched else (M, N))
    assert out.dtype == torch.bfloat16

    ref = mxfp8_gemm_ref(A_q, B_q, A_sc, B_sc)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 5e-3, f"quack vs ref max_err={err}"

    # cuBLAS comparison only for 2D / L=1
    if not batched:
        out_cublas = mxfp8_gemm_cublas(A_q, B_q, A_sc, B_sc)
        assert torch.equal(out, out_cublas), "quack interface != cuBLAS"

    # High-level quantize+gemm convenience fn
    out2 = mxfp8_gemm_quantize(A_hp, W_hp)
    assert torch.equal(out, out2)


@pytest.mark.parametrize("a_major", ["k", "m"])
@pytest.mark.parametrize("b_major", ["k", "n"])
def test_blockscaled_mxfp8_major_modes(a_major, b_major):
    """MXFP8 with A in {k,m}-major × B in {k,n}-major. The SF tensor layout
    stays K-major (hardware convention); only A/B operand strides differ."""
    _skip_if_not_sm100()
    from quack.mx_utils import to_mx

    m, n, k, l = 256, 256, 256, 1
    sf_vec = 32

    def _make_operand(mn, major):
        hp = (torch.randn(l, mn, k, device="cuda", dtype=torch.bfloat16) * k**-0.5).contiguous()
        q_flat, sc_flat = to_mx(hp.view(l * mn, k), sf_vec)
        ref_mkl = (
            (
                q_flat.float().view(l, mn, k)
                * sc_flat.float().view(l, mn, k // sf_vec).repeat_interleave(sf_vec, dim=-1)
            )
            .permute(1, 2, 0)
            .contiguous()
        )
        if major == "k":
            # (l, mn, k) contig → permute to (mn, k, l) → stride (k, 1, mn*k)
            q_mkl = q_flat.view(l, mn, k).contiguous().permute(1, 2, 0)
        else:
            # (l, mn, k) contig → permute to (mn, k, l) with mn fastest → stride (1, mn, mn*k)
            q_mkl = (
                q_flat.view(l, mn, k).contiguous().permute(0, 2, 1).contiguous().permute(2, 1, 0)
            )
        return ref_mkl, q_mkl, sc_flat.view(l, mn, k // sf_vec)

    a_ref, mA, sa_2d = _make_operand(m, a_major)
    b_ref, mB, sb_2d = _make_operand(n, b_major)
    # Sanity: stride(0) == 1 iff mn-major.
    assert (mA.stride(0) == 1) == (a_major == "m"), f"mA stride: {mA.stride()}"
    assert (mB.stride(0) == 1) == (b_major == "n"), f"mB stride: {mB.stride()}"
    from quack.blockscaled_gemm_utils import pack_scale_2d_to_blocked_contig

    a_sc = pack_scale_2d_to_blocked_contig(sa_2d)
    b_sc = pack_scale_2d_to_blocked_contig(sb_2d)
    _, mD = create_blockscaled_operand_tensor(l, m, n, False, cutlass.BFloat16, init="empty")

    assert GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        sf_vec,
        cutlass.BFloat16,
        (128, 128),
        (1, 1),
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        "n",
    )
    runner = compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        sf_vec,
        cutlass.BFloat16,
        (128, 128),
        (1, 1),
        mA,
        mB,
        mD,
        a_sc,
        b_sc,
    )
    runner(mA, mB, mD, a_sc, b_sc)
    torch.cuda.synchronize()

    ref = torch.einsum("mkl,nkl->mnl", a_ref, b_ref)
    err = (mD.float() - ref).abs().max().item()
    assert err < 5e-3, f"A={a_major} B={b_major} max_err={err}"


@pytest.mark.parametrize("b_major", ["k", "n"])
@pytest.mark.parametrize(
    "seqlens_m",
    [
        [128, 128, 128],  # baseline: all aligned
        [100, 200, 150],  # none aligned to 128
        [30, 300, 64, 200],  # mix small + non-aligned
        [1, 128, 127, 129],  # boundary conditions
    ],
)
def test_blockscaled_mxfp8_varlen_m_nonaligned(seqlens_m, b_major):
    """varlen_m with per-expert seqlens not divisible by 128, plus k/n-major B.
    SFA is stored in dQaccum-style padded format; kernel reads it via
    offset_batch_SFA."""
    _skip_if_not_sm100()
    num_experts = len(seqlens_m)
    n, k = 256, 256
    sf_vec = 32
    mma_tiler_mn = (128, 128)
    cluster_shape_mn = (1, 1)

    torch.manual_seed(0)
    a_ref_dq, b_ref_dq, mA, mB, a_sc_contig, b_sc_contig, cu_seqlens_m = (
        create_blockscaled_varlen_m_operands(
            num_experts,
            0,
            n,
            k,
            sf_vec,
            seqlens_m=seqlens_m,
            b_major=b_major,
        )
    )
    expected_b_stride0 = 1 if b_major == "n" else k
    assert mB.stride(0) == expected_b_stride0, (
        f"b_major={b_major} → mB.stride(0) should be {expected_b_stride0}, got {mB.stride()}"
    )
    total_m = int(sum(seqlens_m))
    mSFA = a_sc_contig  # (1, total_padded_rm, rk, 512)
    mSFB = b_sc_contig  # (L, rn, rk, 512)

    mD = torch.empty(total_m, n, dtype=torch.bfloat16, device="cuda")
    runner = compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        sf_vec,
        cutlass.BFloat16,
        mma_tiler_mn,
        cluster_shape_mn,
        mA,
        mB,
        mD,
        mSFA,
        mSFB,
        varlen_m=True,
    )
    runner(mA, mB, mD, mSFA, mSFB, cu_seqlens_m)
    torch.cuda.synchronize()

    # Per-expert reference matmul on dequantized operands.
    cu = cu_seqlens_m.tolist()
    ref = torch.cat([a_ref_dq[cu[i] : cu[i + 1]] @ b_ref_dq[i].T for i in range(num_experts)])
    err = (mD.float() - ref).abs().max().item()
    assert err < 5e-3, f"varlen_m non-aligned seqlens_m={seqlens_m} max_err={err}"


@pytest.mark.parametrize(
    "seqlens_k",
    [
        [128, 128, 128],  # all aligned to 128
        [128, 256, 128],  # 128-aligned mixed sizes
        [96, 160, 128],  # not 128-aligned (but all sf_vec-aligned)
        [32, 256, 64, 128],  # small + varied
    ],
)
def test_blockscaled_mxfp8_varlen_k(seqlens_k):
    """varlen_k blockscaled: per-expert k_i (must be sf_vec-aligned; 128-alignment
    is NOT required). SFA/SFB use dQaccum-style K-padded storage and the kernel
    reads them via offset_batch_SFA/offset_batch_SFB padded-K formula."""
    _skip_if_not_sm100()
    num_experts = len(seqlens_k)
    m, n = 256, 256
    sf_vec = 32
    mma_tiler_mn = (128, 128)
    cluster_shape_mn = (1, 1)

    torch.manual_seed(0)
    a_ref_list, b_ref_list, mA, mB, a_sc_contig, b_sc_contig, cu_seqlens_k = (
        create_blockscaled_varlen_k_operands(num_experts, 0, m, n, sf_vec, seqlens_k=seqlens_k)
    )
    # (m, n, L) with stride 1 on N dim (compile expects leading_dim=1 on mD).
    mD = torch.empty(num_experts, m, n, dtype=torch.bfloat16, device="cuda").permute(1, 2, 0)
    runner = compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        sf_vec,
        cutlass.BFloat16,
        mma_tiler_mn,
        cluster_shape_mn,
        mA,
        mB,
        mD,
        a_sc_contig,
        b_sc_contig,
        varlen_k=True,
    )
    runner(mA, mB, mD, a_sc_contig, b_sc_contig, cu_seqlens_k)
    torch.cuda.synchronize()

    # Per-expert reference: for expert i, result = a_ref[i] @ b_ref[i].T.
    # mD has shape (m, n, L) N-major; each mD[:, :, i] is one expert's output.
    for i in range(num_experts):
        ref_i = a_ref_list[i] @ b_ref_list[i].T
        out_i = mD[:, :, i].float()
        err = (out_i - ref_i).abs().max().item()
        assert err < 5e-3, f"varlen_k seqlens_k={seqlens_k} expert={i} max_err={err}"


@pytest.mark.parametrize("rk_pad", [1, 3, 5])
def test_blockscaled_mxfp8_strided_sf(rk_pad):
    """Verify the kernel honors mSFA/mSFB's actual outer strides (doesn't
    require the full scale tensor to be contig — only the innermost 512-B
    tile). Allocates a larger scale buffer with extra rk padding and slices
    back to the valid rk, producing a non-packed rm stride."""
    _skip_if_not_sm100()
    m, n, k = 256, 256, 512  # k=512 → sf_k=16 → rk=4 (meaningful stride change)
    l, sf_vec = 1, 32

    torch.manual_seed(0)
    a_ref, mA, a_sc = create_blockscaled_operand_quantized(l, m, k, False, sf_vec)
    b_ref, mB, b_sc = create_blockscaled_operand_quantized(l, n, k, False, sf_vec)

    rm = (m + 127) // 128
    rn = (n + 127) // 128
    rk = ((k // sf_vec) + 3) // 4

    # Allocate padded scale buffers (rk + rk_pad along K-blocks), copy valid
    # tiles into the prefix, slice back to rk.  The slice is non-contig:
    # stride(1) = (rk + rk_pad) * 512 instead of rk * 512.
    a_sc_big = torch.zeros(l, rm, rk + rk_pad, 512, dtype=torch.float8_e8m0fnu, device="cuda")
    b_sc_big = torch.zeros(l, rn, rk + rk_pad, 512, dtype=torch.float8_e8m0fnu, device="cuda")
    a_sc_big[:, :, :rk, :] = a_sc
    b_sc_big[:, :, :rk, :] = b_sc
    mSFA_strided = a_sc_big[:, :, :rk, :]
    mSFB_strided = b_sc_big[:, :, :rk, :]
    assert not mSFA_strided.is_contiguous()
    assert mSFA_strided.stride(-1) == 1
    assert mSFA_strided.stride(1) == (rk + rk_pad) * 512, (
        f"expected non-packed rm stride {(rk + rk_pad) * 512}, got {mSFA_strided.stride(1)}"
    )

    # Validate our helper accepts the non-contig layout
    _ = scale_view_for_kernel(mSFA_strided, m, k // sf_vec, l)
    _ = scale_view_for_kernel(mSFB_strided, n, k // sf_vec, l)

    _, mD_strided = create_blockscaled_operand_tensor(
        l, m, n, False, cutlass.BFloat16, init="empty"
    )
    runner = compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        sf_vec,
        cutlass.BFloat16,
        (128, 128),
        (1, 1),
        mA,
        mB,
        mD_strided,
        mSFA_strided,
        mSFB_strided,
    )
    runner(mA, mB, mD_strided, mSFA_strided, mSFB_strided)

    # Compare bit-exactly against the same matmul with contig scales.
    _, mD_contig = create_blockscaled_operand_tensor(l, m, n, False, cutlass.BFloat16, init="empty")
    runner_contig = compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        sf_vec,
        cutlass.BFloat16,
        (128, 128),
        (1, 1),
        mA,
        mB,
        mD_contig,
        a_sc,
        b_sc,
    )
    runner_contig(mA, mB, mD_contig, a_sc, b_sc)
    torch.cuda.synchronize()

    assert torch.equal(mD_strided, mD_contig), (
        f"strided-SF output differs from contig-SF: "
        f"max_abs_err={(mD_strided.float() - mD_contig.float()).abs().max().item()}"
    )


def test_mxfp8_interface_preallocated_out():
    _skip_if_not_sm100()
    from quack.gemm_blockscaled_interface import mxfp8_gemm, mxfp8_quantize

    M, N, K = 256, 256, 256
    torch.manual_seed(0)
    A_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * K**-0.5
    W_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * K**-0.5
    A_q, A_sc = mxfp8_quantize(A_hp)
    W_q, W_sc = mxfp8_quantize(W_hp)
    B_q, B_sc = W_q.mT, W_sc.mT

    out_alloc = mxfp8_gemm(A_q, B_q, A_sc, B_sc)
    out_pre = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    mxfp8_gemm(A_q, B_q, A_sc, B_sc, out=out_pre)
    assert torch.equal(out_alloc, out_pre)
