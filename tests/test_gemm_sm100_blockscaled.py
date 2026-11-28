import pytest
import torch

import cutlass

from quack.blockscaled.utils import (
    blockscaled_gemm_reference,
    create_blockscaled_operand_quantized,
    create_blockscaled_operand_tensor,
    create_blockscaled_scale_tensor,
    scale_blocked_for_cublas,
    scale_view_for_kernel,
)
from quack.blockscaled.operand import (
    BlockScaledFormat,
    MXFP4,
    MXFP6_E2M3,
    MXFP6_E2M3_PACKED,
    MXFP8_E4M3,
    NVFP4,
)
from quack.cute_dsl_utils import get_compile_target_capacity, get_device_capacity
from quack.gemm import gemm
from quack.gemm_default_epi import GemmDefaultSm100
from quack.sm100_utils import MmaKind
from quack.blockscaled.quantize import to_blocked


_SM100_FAMILY = (10, 11)
_PHYSICAL_ARCH = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
# SM100-family kernel configs (2-CTA tiles, tile_n 64/192). QUACK_ARCH can select a
# dispatch arch but cannot give a proxy GPU tcgen05 instructions.
requires_sm100 = pytest.mark.skipif(
    _PHYSICAL_ARCH not in _SM100_FAMILY
    or get_compile_target_capacity()[0] not in _SM100_FAMILY
    or get_device_capacity()[0] not in _SM100_FAMILY,
    reason="SM100/SM110 tcgen05 kernel configs",
)


def _run_gemm(mA, mB, mD, mSFA, mSFB, fmt, mma_tiler_mn, cluster_shape_mn):
    """quack.gemm.gemm on batch-last (mn, k, l) operands with an explicit tile and cluster,
    on the CLC (dynamic persistent) scheduler."""
    gemm(
        mA.permute(2, 0, 1),
        mB.permute(2, 0, 1),
        mD.permute(2, 0, 1),
        None,
        None,
        tile_M=mma_tiler_mn[0],
        tile_N=mma_tiler_mn[1],
        cluster_M=cluster_shape_mn[0],
        cluster_N=cluster_shape_mn[1],
        is_dynamic_persistent=True,
        SFA=mSFA,
        SFB=mSFB,
        bs_format_a=fmt,
        bs_format_b=fmt,
    )


def test_blockscaled_validation():
    assert GemmDefaultSm100.can_implement(
        MXFP8_E4M3,
        MXFP8_E4M3,
        cutlass.Float32,
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
    assert GemmDefaultSm100.can_implement(
        MXFP8_E4M3,
        MXFP8_E4M3,
        cutlass.Float32,
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
    assert GemmDefaultSm100.can_implement(
        MXFP8_E4M3,
        MXFP8_E4M3,
        cutlass.Float32,
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
    assert GemmDefaultSm100.can_implement(
        MXFP4,
        MXFP4,
        cutlass.Float32,
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
    assert GemmDefaultSm100.can_implement(
        NVFP4,
        NVFP4,
        cutlass.Float32,
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
    assert not GemmDefaultSm100.can_implement(
        MXFP8_E4M3,
        MXFP8_E4M3,
        cutlass.Float32,
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
    assert not GemmDefaultSm100.can_implement(
        MXFP8_E4M3,
        MXFP8_E4M3,
        cutlass.Float32,
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
    assert not GemmDefaultSm100.can_implement(
        MXFP4,
        MXFP4,
        cutlass.Float32,
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
    assert not GemmDefaultSm100.can_implement(
        MXFP8_E4M3,
        MXFP8_E4M3,
        cutlass.Float32,
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
    # fp4 with e4m3 scales at vec 32 is an illegal scale config that no
    # registry format descriptor can express; check the dtype gate directly.
    assert not GemmDefaultSm100.is_valid_dtypes_and_scale_factor_vec_size(
        cutlass.Float4E2M1FN,
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        32,
        cutlass.Float32,
    )
    assert not GemmDefaultSm100.can_implement(
        MXFP8_E4M3,
        MXFP8_E4M3,
        cutlass.Float32,
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

    # Operand layouts. can_implement is not on the gemm() dispatch path (benchmarks filter
    # configs with it), so the kernel tests can't catch a wrong layout rule here.
    def layout_ok(fmt_a, fmt_b, a_major, b_major):
        return GemmDefaultSm100.can_implement(
            fmt_a,
            fmt_b,
            cutlass.Float32,
            cutlass.BFloat16,
            (128, 128),
            (1, 1),
            256,
            256,
            256,
            1,
            a_major,
            b_major,
            "n",
        )

    # fp8 may be M-major A / N-major B; the scale tensor stays K-major.
    for a_major, b_major in [("k", "k"), ("m", "k"), ("k", "n"), ("m", "n")]:
        assert layout_ok(MXFP8_E4M3, MXFP8_E4M3, a_major, b_major), (a_major, b_major)
    # Packed sub-byte operands pack along K, so they must be K-major. Each pair is
    # accepted K-major first, so a rejection below is the layout rule's.
    for fmt_a, fmt_b in [(MXFP4, MXFP4), (MXFP6_E2M3_PACKED, MXFP8_E4M3), (MXFP8_E4M3, MXFP4)]:
        assert layout_ok(fmt_a, fmt_b, "k", "k"), (fmt_a.name, fmt_b.name)
        if fmt_a is not MXFP8_E4M3:
            assert not layout_ok(fmt_a, fmt_b, "m", "k"), (fmt_a.name, "m-major A")
        if fmt_b is not MXFP8_E4M3:
            assert not layout_ok(fmt_a, fmt_b, "k", "n"), (fmt_b.name, "n-major B")


def test_can_implement_operand_kind_polymorphism():
    """One preflight entry point for both kernels: plain cutlass dtypes select
    the dense checks, BlockScaledFormat descriptors the blockscaled checks, and
    one operand of each kind is rejected."""
    common = ((128, 128), (1, 1), 256, 256, 256, 1, "k", "k", "n")
    assert GemmDefaultSm100.can_implement(
        cutlass.BFloat16, cutlass.BFloat16, cutlass.Float32, cutlass.BFloat16, *common
    )
    assert not GemmDefaultSm100.can_implement(
        MXFP8_E4M3, cutlass.Float8E4M3FN, cutlass.Float32, cutlass.BFloat16, *common
    )
    assert not GemmDefaultSm100.can_implement(
        cutlass.Float8E4M3FN, MXFP8_E4M3, cutlass.Float32, cutlass.BFloat16, *common
    )


@pytest.mark.parametrize("fmt_a", [MXFP4, MXFP6_E2M3_PACKED])
def test_mixed_unpack_explicit_tile_k_validation(fmt_a):
    common = ((1, 1), 256, 256, 512, 1, "k", "k", "n")
    for tile_k in (32, 64, 96, 160):
        assert not GemmDefaultSm100.can_implement(
            fmt_a,
            MXFP8_E4M3,
            cutlass.Float32,
            cutlass.BFloat16,
            (128, 128, tile_k),
            *common,
        )
    for tile_k in (128, 256):
        assert GemmDefaultSm100.can_implement(
            fmt_a,
            MXFP8_E4M3,
            cutlass.Float32,
            cutlass.BFloat16,
            (128, 128, tile_k),
            *common,
        )
    # NVFP4 uses vec-16 scales, so its complete SF chunk is 64 K elements.
    assert GemmDefaultSm100.can_implement(
        NVFP4,
        NVFP4,
        cutlass.Float32,
        cutlass.BFloat16,
        (128, 128, 64),
        *common,
    )
    assert not GemmDefaultSm100.can_implement(
        MXFP6_E2M3,
        MXFP8_E4M3,
        cutlass.Float32,
        cutlass.BFloat16,
        (128, 128, 128),
        *common,
    )


def test_direct_quantized_generator_rejects_packed_fp6():
    """The raw-tensor generator has no separate uint8-storage/FP6-MMA dtype."""
    with pytest.raises(ValueError, match="init=quant does not support"):
        create_blockscaled_operand_quantized(
            1,
            128,
            256,
            False,
            32,
            cutlass.Float6E2M3FN,
            cutlass.Float8E8M0FNU,
        )


@requires_sm100
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
        # batched packed fp4: pins the generator's K-majorness for l > 1
        # (a (mn, k/2, l) contiguous alloc would put stride 1 on L, not K)
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
            2,
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
    fmt = BlockScaledFormat.from_cutlass_dtypes(ab_dtype, sf_dtype, sf_vec_size).name
    a_ref, mA = create_blockscaled_operand_tensor(l, m, k, False, ab_dtype)
    b_ref, mB = create_blockscaled_operand_tensor(l, n, k, False, ab_dtype)
    _, mD = create_blockscaled_operand_tensor(l, m, n, False, d_dtype, init="empty")
    sfa_ref, mSFA = create_blockscaled_scale_tensor(l, m, k, sf_vec_size, sf_dtype)
    sfb_ref, mSFB = create_blockscaled_scale_tensor(l, n, k, sf_vec_size, sf_dtype)
    _run_gemm(mA, mB, mD, mSFA, mSFB, fmt, mma_tiler_mn, cluster_shape_mn)

    ref = blockscaled_gemm_reference(a_ref, b_ref, sfa_ref, sfb_ref)
    err = (mD.float() - ref).abs().max().item()
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
@requires_sm100
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
    l, sf_vec = 1, 32

    torch.manual_seed(0)
    a_ref, mA, a_sc = create_blockscaled_operand_quantized(l, m, k, False, sf_vec)
    b_ref, mB, b_sc = create_blockscaled_operand_quantized(l, n, k, False, sf_vec)
    _, mD = create_blockscaled_operand_tensor(l, m, n, False, cutlass.BFloat16, init="empty")

    mSFA = scale_view_for_kernel(a_sc, m, k // sf_vec, l)
    mSFB = scale_view_for_kernel(b_sc, n, k // sf_vec, l)

    _run_gemm(mA, mB, mD, mSFA, mSFB, "mxfp8_e4m3", mma_tiler_mn, cluster_shape_mn)

    # Reference: dequant matmul (a_ref/b_ref are already dequantized)
    d_ref = torch.einsum("mkl,nkl->mnl", a_ref, b_ref)
    err = (mD.float() - d_ref).abs().max().item()
    assert err < 5e-3, f"quack vs dequant max_err={err}"

    # Compare cuBLAS on identical operand bits and scale bytes, at BF16 precision.
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
    torch.testing.assert_close(
        mD.squeeze(-1), out_cublas, rtol=torch.finfo(mD.dtype).eps, atol=1e-7
    )


def test_mma_kind_mirrors_kernel_kind():
    """The tcgen05 kind rules are encoded at two layers that key on different
    things: quack.blockscaled.operand.mma_kind_for_pair (format names, torch
    layer) and MmaKind.for_operands (storage cutlass dtypes, kernel layer). Pin
    the mirror so they cannot drift apart. Host-only, no GPU needed."""
    from quack.blockscaled.operand import BLOCKSCALED_FORMAT_REGISTRY, mma_kind_for_pair
    from quack.cute_dsl_utils import torch2cute_dtype_map

    fmts = list(BLOCKSCALED_FORMAT_REGISTRY.values())
    for fmt_a in fmts:
        for fmt_b in fmts:
            try:
                kind = mma_kind_for_pair(fmt_a, fmt_b)
            except ValueError:
                continue  # not hardware-representable; no instruction to agree on
            kernel_kind = MmaKind.for_operands(
                torch2cute_dtype_map[fmt_a.qdata_dtype],
                torch2cute_dtype_map[fmt_b.qdata_dtype],
                blockscaled=True,
            )
            # PTX spells the vec-32 instantiation of kind::mxf4nvf4 as kind::mxf4.
            assert kernel_kind.value == {"mxf4": "mxf4nvf4"}.get(kind, kind), (
                f"{fmt_a.name} x {fmt_b.name}: kind {kind}, kernel derives {kernel_kind}"
            )


def test_sm100_dtype_gate():
    """Kernel dtype support is a per-architecture gate
    (GemmSm100.is_valid_dtypes_and_scale_factor_vec_size), not a registry
    property. New world: fp6 MMA dtypes are accepted with a Uint8 copy dtype
    (packed fp6 crosses the FFI boundary as raw bytes and is TMA-unpacked
    in-kernel) and mixed fp4/fp8 pairs are accepted with packed-fp4 storage;
    byte-container (one code per uint8) sub-byte storage has no kernel path."""
    from quack.gemm_sm100 import GemmSm100

    ok = GemmSm100.is_valid_dtypes_and_scale_factor_vec_size
    e8m0, e4m3, bf16 = cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN, cutlass.BFloat16
    f4, f6a, f6b, u8 = (
        cutlass.Float4E2M1FN,
        cutlass.Float6E2M3FN,
        cutlass.Float6E3M2FN,
        cutlass.Uint8,
    )
    # fp6 MMA dtypes with Uint8 copy dtype (packed 6-bit gmem, raw bytes at FFI)
    assert ok(f6a, e4m3, e8m0, 32, bf16, a_copy_dtype=u8)
    assert ok(e4m3, f6b, e8m0, 32, bf16, b_copy_dtype=u8)
    assert ok(f6a, f6b, e8m0, 32, bf16, a_copy_dtype=u8, b_copy_dtype=u8)
    # mixed packed fp4 x fp8 (TMA unpack under kind::mxf8f6f4) and pure packed fp4
    assert ok(f4, e4m3, e8m0, 32, bf16)
    assert ok(e4m3, f4, e8m0, 32, bf16)
    assert ok(f4, f4, e8m0, 32, bf16)
    assert ok(f4, f4, e4m3, 16, bf16)  # nvfp4
    # copy dtype must be the storage the kernel implements: byte-container fp4
    # (one code per uint8) and non-Uint8 fp6 copy dtypes have no kernel path
    assert not ok(f4, e4m3, e8m0, 32, bf16, a_copy_dtype=u8)
    assert not ok(f6a, e4m3, e8m0, 32, bf16, a_copy_dtype=e4m3)
    # vec-16 (nvfp4 scale config) requires fp4 on BOTH operands
    assert not ok(f4, e4m3, e4m3, 16, bf16)
    assert not ok(f6a, f6a, e4m3, 16, bf16, a_copy_dtype=u8, b_copy_dtype=u8)
