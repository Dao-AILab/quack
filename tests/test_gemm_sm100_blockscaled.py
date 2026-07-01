import pytest
import torch

import cutlass

from quack.blockscaled_gemm_utils import (
    blockscaled_gemm_reference,
    compile_blockscaled_gemm_tvm_ffi,
    create_blockscaled_operand_quantized,
    create_blockscaled_operand_tensor,
    create_blockscaled_scale_tensor,
    create_blockscaled_varlen_k_operands,
    create_blockscaled_varlen_m_operands,
    scale_blocked_for_cublas,
    scale_view_for_kernel,
)
from quack.gemm_default_epi import GemmDefaultSm100
from quack.mx_utils import to_blocked


def _skip_if_not_sm100():
    major = torch.cuda.get_device_properties(0).major
    if major < 10:
        pytest.skip("SM100+ required")


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
    "seqlens_m",
    [
        [128, 128, 128, 128],  # uniform
        [128, 256, 384, 256],  # ragged, 128-aligned
        [256, 0, 256, 256],  # empty expert
        [1792, 128, 128, 128],  # heavy skew
    ],
)
def test_blockscaled_mxfp8_varlen_m_sfa_tile_aligned(seqlens_m):
    """sfa_tile_aligned=True: with 128-aligned per-expert boundaries, SFA is passed in
    NATURAL (unpadded) packed layout and the kernel offsets per expert by
    cu_seqlens[b] // 128 (no dQaccum +b padding -> no scatter). Must (a) match the
    dequant reference and (b) be bit-identical to the default dQaccum-padded path --
    same kernel math, only the SFA addressing differs."""
    _skip_if_not_sm100()
    from quack.blockscaled_gemm_utils import pack_scale_2d_to_blocked_contig
    from quack.mx_utils import to_mx_compiled

    e = len(seqlens_m)
    n, k, sf_vec = 256, 256, 32
    sf_k = k // sf_vec
    mma_tiler_mn, cluster_shape_mn = (128, 128), (1, 1)
    assert all(m % 128 == 0 for m in seqlens_m), "this path requires 128-aligned seqlens"
    total_m = sum(seqlens_m)
    dev = "cuda"

    torch.manual_seed(0)
    qa, sa = to_mx_compiled(
        (torch.randn(total_m, k, dtype=torch.bfloat16, device=dev) * k**-0.5).contiguous(), sf_vec
    )
    qbf, sb = to_mx_compiled(
        (torch.randn(e, n, k, dtype=torch.bfloat16, device=dev) * k**-0.5)
        .contiguous()
        .view(e * n, k),
        sf_vec,
    )
    qb = qbf.view(e, n, k)
    sb = sb.view(e, n, sf_k)
    mB = qb.transpose(1, 2).permute(2, 1, 0)  # (n, k, e)
    mSFB = pack_scale_2d_to_blocked_contig(sb)
    seq = torch.tensor(seqlens_m, dtype=torch.int32, device=dev)
    cu = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=dev), torch.cumsum(seq, 0).to(torch.int32)]
    )

    a_ref = qa.float() * sa.float().repeat_interleave(sf_vec, -1)
    b_ref = qbf.float().view(e, n, k) * sb.float().repeat_interleave(sf_vec, -1)
    cul = cu.tolist()
    ref = torch.cat(
        [a_ref[cul[i] : cul[i + 1]] @ b_ref[i].T for i in range(e) if cul[i + 1] > cul[i]]
    )

    # (a) natural (unpadded) SFA + sfa_tile_aligned=True
    mSFA_nat = pack_scale_2d_to_blocked_contig(sa.view(1, total_m, sf_k))
    out_nat = torch.empty(total_m, n, dtype=torch.bfloat16, device=dev)
    compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        sf_vec,
        cutlass.BFloat16,
        mma_tiler_mn,
        cluster_shape_mn,
        qa,
        mB,
        out_nat,
        mSFA_nat,
        mSFB,
        varlen_m=True,
        sfa_tile_aligned=True,
    )(qa, mB, out_nat, mSFA_nat, mSFB, cu)

    # (b) dQaccum-padded SFA + default path (per-expert +i*128 tile gap)
    tile = 128
    total_padded = ((total_m + tile - 1) // tile + (e - 1)) * tile
    sa_pad = sa.new_zeros(total_padded, sf_k)
    off = 0
    for i, m_i in enumerate(seqlens_m):
        dst = (off // tile + i) * tile
        sa_pad[dst : dst + m_i] = sa[off : off + m_i]
        off += m_i
    mSFA_pad = pack_scale_2d_to_blocked_contig(sa_pad.view(1, total_padded, sf_k))
    out_pad = torch.empty(total_m, n, dtype=torch.bfloat16, device=dev)
    compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        sf_vec,
        cutlass.BFloat16,
        mma_tiler_mn,
        cluster_shape_mn,
        qa,
        mB,
        out_pad,
        mSFA_pad,
        mSFB,
        varlen_m=True,
        sfa_tile_aligned=False,
    )(qa, mB, out_pad, mSFA_pad, mSFB, cu)
    torch.cuda.synchronize()

    err = (out_nat.float() - ref).abs().max().item()
    assert err < 5e-3, f"natural-SFA seqlens_m={seqlens_m} max_err={err}"
    assert torch.equal(out_nat, out_pad), f"natural != dQaccum for seqlens_m={seqlens_m}"


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
