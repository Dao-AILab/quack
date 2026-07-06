import pytest
import torch

import cutlass

from quack.blockscaled.utils import (
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
from quack.blockscaled.quantize import to_blocked


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
        # tile_n=192 with N=448: the last N-tile's SF window (atoms 3,4)
        # straddles past the last allocated SF atom (ceil(448/128)=4).
        # Regression: the old overlapped-window TMA remap presented atoms in
        # groups of 4 and zero-filled past the presented extent, silently
        # zeroing the last 64 output columns; the chunk-granular SFB load
        # bounds-checks the atom-n dim and zero-fills only the truly
        # out-of-range atom.
        (
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (128, 192),
            (1, 1),
            256,
            448,
            256,
            1,
        ),
        # tile_n=192 with multiple N-tiles: exercises mid-atom SF window starts
        # for e4m3 scale factors (regression: the old remap was gated on e8m0
        # only, loading wrong SFB atoms for N-tile index >= 1)
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.Float32,
            (128, 192),
            (1, 1),
            256,
            384,
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
            576,
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
    # rearrangement logic: pad + (l, rm, 128, rk, 4) -> (l, rm, rk, 32, 4, 4)
    rm = (mn + 127) // 128
    rk = (sf_k + 3) // 4
    mn_pad = rm * 128
    sf_k_pad = rk * 4
    padded = torch.zeros(l, mn_pad, sf_k_pad, device="cuda", dtype=torch.uint8)
    padded[:, :mn, :sf_k] = scale_2d
    blocks = padded.view(l, rm, 128, rk, 4).permute(0, 1, 3, 2, 4)
    blocks = blocks.reshape(l, rm, rk, 4, 32, 4).transpose(3, 4).contiguous()
    scale_contig = blocks  # (l, rm, rk, 32, 4, 4)

    # kernel view indexing: byte offset within tile = (m%32)*16 + ((m//32)%4)*4 + (k%4)
    kv = (
        scale_view_for_kernel(scale_contig.view(torch.float8_e8m0fnu), mn, sf_k, l)
        .view(torch.uint8)
        .flatten(-3)
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
        # All supported blockscaled tile_n values (64, 128, 192, 256).
        ((128, 64), (1, 1), 256, 64, 512),
        ((128, 128), (1, 1), 256, 256, 256),
        ((128, 128), (1, 1), 512, 512, 512),
        ((128, 192), (1, 1), 256, 192, 256),
        ((128, 256), (1, 1), 256, 256, 256),
        ((256, 128), (2, 1), 512, 256, 512),
        ((256, 192), (2, 1), 256, 192, 256),
        ((256, 192), (2, 1), 256, 384, 256),
        ((256, 192), (2, 1), 512, 192, 512),
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


@pytest.mark.parametrize("a_major", ["k", "m"])
@pytest.mark.parametrize("b_major", ["k", "n"])
def test_blockscaled_mxfp8_major_modes(a_major, b_major):
    """MXFP8 with A in {k,m}-major × B in {k,n}-major. The SF tensor layout
    stays K-major (hardware convention); only A/B operand strides differ."""
    _skip_if_not_sm100()
    from quack.blockscaled.quantize import to_mx

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
    from quack.blockscaled.utils import pack_scale_2d_to_blocked_contig

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


VARLEN_FMT = {
    # format: (ab_dtype, sf_dtype, sf_vec_size)
    "mxfp8": (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
    "mxfp4": (cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
    "nvfp4": (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
}


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
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
def test_blockscaled_varlen_m_nonaligned(seqlens_m, b_major, fmt):
    """varlen_m with per-expert seqlens not divisible by 128, plus k/n-major B.
    SFA is stored with tile-aligned per-batch padding; kernel reads it via
    offset_batch_SFA."""
    _skip_if_not_sm100()
    if fmt != "mxfp8" and b_major == "n":
        pytest.skip("fp4 operands must be K-major")
    ab_dtype, sf_dtype, sf_vec = VARLEN_FMT[fmt]
    num_experts = len(seqlens_m)
    n, k = 256, 256
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
            ab_dtype,
            sf_dtype,
            seqlens_m=seqlens_m,
            b_major=b_major,
        )
    )
    expected_b_stride0 = 1 if b_major == "n" else mB.shape[1]  # k, or k/2 for packed fp4
    assert mB.stride(0) == expected_b_stride0, (
        f"b_major={b_major} → mB.stride(0) should be {expected_b_stride0}, got {mB.stride()}"
    )
    total_m = int(sum(seqlens_m))
    mSFA = a_sc_contig  # (1, total_padded_rm, rk, 32, 4, 4)
    mSFB = b_sc_contig  # (L, rn, rk, 32, 4, 4)

    mD = torch.empty(total_m, n, dtype=torch.bfloat16, device="cuda")
    runner = compile_blockscaled_gemm_tvm_ffi(
        ab_dtype,
        sf_dtype,
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
    assert err < 5e-3, f"varlen_m non-aligned {fmt} seqlens_m={seqlens_m} max_err={err}"


@pytest.mark.parametrize(
    "seqlens_k",
    [
        [128, 128, 128],  # all aligned to 128
        [128, 256, 128],  # 128-aligned mixed sizes
        [96, 160, 128],  # not 128-aligned (but all sf_vec-aligned)
        [32, 256, 64, 128],  # small + varied
        [100, 220, 65],  # not even sf_vec(32)-aligned: partial last scale block
        [1, 33, 158, 192],  # boundary conditions, non-aligned
    ],
)
def test_blockscaled_mxfp8_varlen_k(seqlens_k):
    """varlen_k blockscaled: per-expert k_i is arbitrary (neither 128- nor
    sf_vec(32)-alignment required; a partial last scale block pairs with the
    ragged TMA's zero-filled value tail). SFA/SFB use tile-aligned per-batch K
    padding and the kernel reads them via offset_batch_SFA/offset_batch_SFB
    padded-K formula."""
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


@pytest.mark.parametrize(
    "seqlens_k",
    [
        [128, 128, 128],  # baseline: all aligned
        [96, 160, 128],  # not 128-aligned
        [100, 220, 65],  # not even sf_vec(32)-aligned
    ],
)
def test_blockscaled_varlen_k_public_api(seqlens_k):
    """varlen_k through the public quack.gemm.gemm API (jit-cached compile path).
    A is (m, total_k) m-major, B is (n, total_k) n-major, and BOTH SFA/SFB are
    tile-aligned K-padded buffers passed as (1, rm/rn, total_padded_rk, 32, 4, 4)."""
    _skip_if_not_sm100()
    from quack.gemm import gemm as gemm_public

    num_experts = len(seqlens_k)
    m, n, sf_vec = 256, 256, 32

    torch.manual_seed(0)
    a_ref_list, b_ref_list, mA, mB, a_sc_contig, b_sc_contig, cu_seqlens_k = (
        create_blockscaled_varlen_k_operands(num_experts, 0, m, n, sf_vec, seqlens_k=seqlens_k)
    )
    mD = torch.empty(num_experts, m, n, dtype=torch.bfloat16, device="cuda")
    gemm_public(
        mA,  # (m, total_k) m-major
        mB,  # (n, total_k) n-major
        mD,  # (L, m, n); gemm() permutes to (m, n, L) internally
        None,
        None,
        tile_M=128,
        tile_N=128,
        cluster_M=1,
        cluster_N=1,
        cu_seqlens_k=cu_seqlens_k,
        SFA=a_sc_contig,
        SFB=b_sc_contig,
    )
    torch.cuda.synchronize()

    for i in range(num_experts):
        ref_i = a_ref_list[i] @ b_ref_list[i].T
        err = (mD[i].float() - ref_i).abs().max().item()
        assert err < 5e-3, f"public API varlen_k seqlens_k={seqlens_k} expert={i} max_err={err}"


@pytest.mark.parametrize(
    "tile_cluster",
    [((128, 128), (1, 1)), ((128, 256), (1, 2)), ((256, 128), (2, 1)), ((256, 256), (2, 2))],
)
@pytest.mark.parametrize("seqlens_k", [[96, 160, 128], [100, 220, 65]])
def test_blockscaled_varlen_k_poisoned_sf_pad(seqlens_k, tile_cluster):
    """For mxfp8 one MMA instruction consumes exactly one SF k-block, so the
    kernel skips the instructions for pad blocks on the ragged last k-tile
    (their A/B values are TMA-zero-filled, contributing nothing) — pad scales
    are never consumed and the gmem SF pad may be arbitrary. Poison it with
    0xFF (e8m0 NaN): any pad byte reaching an MMA would turn whole output rows
    NaN via 0-value x NaN-scale products in the K tail. Instruction issue is a
    leader-only decision, so 2-CTA MMA (tile_M 256) is covered too."""
    _skip_if_not_sm100()
    from quack.gemm import gemm as gemm_public

    (tile_m, tile_n), (cluster_m, cluster_n) = tile_cluster
    num_experts = len(seqlens_k)
    m, n, sf_vec = 256, 256, 32
    torch.manual_seed(0)
    a_ref_list, b_ref_list, mA, mB, SFA, SFB, cu_seqlens_k = create_blockscaled_varlen_k_operands(
        num_experts, 0, m, n, sf_vec, seqlens_k=seqlens_k, sf_pad_byte=0xFF
    )
    mD = torch.empty(num_experts, m, n, dtype=torch.bfloat16, device="cuda")
    gemm_public(
        mA,
        mB,
        mD,
        None,
        None,
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=cluster_m,
        cluster_N=cluster_n,
        cu_seqlens_k=cu_seqlens_k,
        SFA=SFA,
        SFB=SFB,
    )
    torch.cuda.synchronize()
    assert not mD.isnan().any(), "NaN leaked from poisoned SF pad into the output"
    for i in range(num_experts):
        ref_i = a_ref_list[i] @ b_ref_list[i].T
        err = (mD[i].float() - ref_i).abs().max().item()
        assert err < 5e-3, f"poisoned pad seqlens_k={seqlens_k} expert={i} max_err={err}"


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("b_major", ["k", "n"])
@pytest.mark.parametrize(
    "seqlens_m",
    [
        [128, 128, 128],  # baseline: all aligned
        [100, 200, 150],  # none aligned to 128
        [1, 128, 127, 129],  # boundary conditions
    ],
)
def test_blockscaled_varlen_m_public_api(seqlens_m, b_major, fmt):
    """varlen_m through the public quack.gemm.gemm API (jit-cached compile path).
    SFA is the tile-aligned M-padded buffer passed as a
    (1, total_padded_rm, rk, 32, 4, 4) view."""
    _skip_if_not_sm100()
    if fmt != "mxfp8" and b_major == "n":
        pytest.skip("fp4 operands must be K-major")
    from quack.gemm import gemm as gemm_public

    ab_dtype, sf_dtype, sf_vec = VARLEN_FMT[fmt]
    num_experts = len(seqlens_m)
    n, k = 256, 256

    torch.manual_seed(0)
    a_ref_dq, b_ref_dq, mA, mB, a_sc_contig, b_sc_contig, cu_seqlens_m = (
        create_blockscaled_varlen_m_operands(
            num_experts, 0, n, k, sf_vec, ab_dtype, sf_dtype, seqlens_m=seqlens_m, b_major=b_major
        )
    )
    total_m = int(sum(seqlens_m))
    SFA, SFB = a_sc_contig, b_sc_contig  # (1, total_padded_rm, rk, 32, 4, 4), (L, rn, rk, 32, 4, 4)
    mD = torch.empty(total_m, n, dtype=torch.bfloat16, device="cuda")
    gemm_public(
        mA,
        mB.permute(2, 0, 1),  # (n, k, l) -> (l, n, k); gemm() permutes back internally
        mD,
        None,
        None,
        tile_M=128,
        tile_N=128,
        cluster_M=1,
        cluster_N=1,
        cu_seqlens_m=cu_seqlens_m,
        SFA=SFA,
        SFB=SFB,
    )
    torch.cuda.synchronize()

    cu = cu_seqlens_m.tolist()
    ref = torch.cat([a_ref_dq[cu[i] : cu[i + 1]] @ b_ref_dq[i].T for i in range(num_experts)])
    err = (mD.float() - ref).abs().max().item()
    assert err < 5e-3, f"public API varlen_m {fmt} seqlens_m={seqlens_m} max_err={err}"


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
    # stride(1) = (rk + rk_pad) * 512 elements instead of rk * 512.
    a_sc_big = torch.zeros(l, rm, rk + rk_pad, 32, 4, 4, dtype=torch.float8_e8m0fnu, device="cuda")
    b_sc_big = torch.zeros(l, rn, rk + rk_pad, 32, 4, 4, dtype=torch.float8_e8m0fnu, device="cuda")
    a_sc_big[:, :, :rk] = a_sc
    b_sc_big[:, :, :rk] = b_sc
    mSFA_strided = a_sc_big[:, :, :rk]
    mSFB_strided = b_sc_big[:, :, :rk]
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
