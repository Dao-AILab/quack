# Copyright (c) 2026, QuACK team.
"""Rubin (sm_107) wide-K MMA forms: fp8 K=64 and fp4 K=128 instructions.

GemmSm100 switches to these forms on sm_107 for M128 1-CTA and M256 2-CTA tiles;
other tiles keep the SM100-K forms. These cases cover both tile families, a tile_n
that is not a multiple of 128, and a K that ends in the middle of a wide instruction.
"""

import math

import pytest
import torch

import cutlass

from quack.blockscaled.utils import (
    blockscaled_gemm_reference,
    compile_blockscaled_gemm_tvm_ffi,
    create_blockscaled_operand_tensor,
    create_blockscaled_scale_tensor,
)
from quack.gemm import gemm
from quack.gemm_default_epi import GemmDefaultSm100
from quack.gemm_sm100 import GemmSm100

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin (sm_107) wide-K MMA forms",
)

TILES = [((128, 128), (1, 1)), ((256, 256), (2, 1)), ((256, 192), (2, 1)), ((128, 64), (1, 1))]
FORMATS = {
    "mxfp8": (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
    "mxfp4": (cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
    "nvfp4": (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
}


def _wide_k_selected(ab_dtype, tile, cluster, sf_vec_size=None, tile_k=0, varlen_k=False):
    gemm_op = GemmDefaultSm100(
        cutlass.Float32, ab_dtype, (*tile, tile_k), (*cluster, 1), sf_vec_size=sf_vec_size
    )
    gemm_op.a_dtype = gemm_op.b_dtype = gemm_op.a_mma_dtype = gemm_op.b_mma_dtype = ab_dtype
    gemm_op.varlen_k = varlen_k
    if sf_vec_size is None:
        sm100_inst_k = 256 // ab_dtype.width
    else:
        sm100_inst_k = GemmSm100._blockscaled_mma_inst_k(ab_dtype, ab_dtype)
    return gemm_op._use_rubin_wide_k(sm100_inst_k)


@pytest.mark.parametrize(
    "ab_dtype,sf_vec_size,tile,cluster,tile_k,varlen_k,expected",
    [
        # M128 1-CTA and M256 2-CTA tiles take the wide forms ...
        (cutlass.Float8E4M3FN, None, (128, 256), (1, 1), 0, False, True),
        (cutlass.Float8E4M3FN, None, (256, 256), (2, 1), 0, False, True),
        (cutlass.Float8E4M3FN, 32, (256, 192), (2, 1), 0, False, True),
        (cutlass.Float4E2M1FN, 16, (128, 64), (1, 1), 0, False, True),
        (cutlass.Float8E4M3FN, None, (256, 256), (2, 1), 256, False, True),
        # ... M64 1-CTA and M128 2-CTA have no wide fp8/fp4 form on Rubin.
        (cutlass.Float8E4M3FN, None, (64, 256), (1, 1), 0, False, False),
        (cutlass.Float8E4M3FN, None, (128, 256), (2, 1), 0, False, False),
        # An explicit tile K must be a multiple of the wide instruction K.
        (cutlass.Float8E4M3FN, None, (256, 256), (2, 1), 96, False, False),
        # varlen_k mxfp8 keeps one instruction per scale block; fp4 never runs varlen_k.
        (cutlass.Float8E4M3FN, 32, (256, 256), (2, 1), 0, True, False),
        # bf16 is unchanged (same per-clock rate as SM100).
        (cutlass.BFloat16, None, (256, 256), (2, 1), 0, False, False),
    ],
)
def test_wide_k_selection(ab_dtype, sf_vec_size, tile, cluster, tile_k, varlen_k, expected):
    selected = _wide_k_selected(ab_dtype, tile, cluster, sf_vec_size, tile_k, varlen_k)
    assert selected is expected


@pytest.mark.parametrize("tile,cluster", TILES)
@pytest.mark.parametrize(
    "fmt,k",
    [
        ("mxfp8", 512),
        ("mxfp8", 288),  # ends mid wide instruction
        ("mxfp4", 512),
        ("mxfp4", 384),  # last k-tile holds one of two wide instructions
        ("nvfp4", 512),
        ("nvfp4", 640),
    ],
)
def test_blockscaled_wide_k(fmt, k, tile, cluster):
    ab_dtype, sf_dtype, sf_vec_size = FORMATS[fmt]
    assert _wide_k_selected(ab_dtype, tile, cluster, sf_vec_size)
    m, n, l = 512, 384, 1
    torch.manual_seed(0)
    a_ref, mA = create_blockscaled_operand_tensor(l, m, k, False, ab_dtype)
    b_ref, mB = create_blockscaled_operand_tensor(l, n, k, False, ab_dtype)
    _, mD = create_blockscaled_operand_tensor(l, m, n, False, cutlass.BFloat16, init="empty")
    sfa_ref, mSFA = create_blockscaled_scale_tensor(l, m, k, sf_vec_size, sf_dtype)
    sfb_ref, mSFB = create_blockscaled_scale_tensor(l, n, k, sf_vec_size, sf_dtype)
    compiled = compile_blockscaled_gemm_tvm_ffi(
        ab_dtype, sf_dtype, sf_vec_size, cutlass.BFloat16, tile, cluster, mA, mB, mD, mSFA, mSFB
    )
    compiled(mA, mB, mD, mSFA, mSFB)
    torch.cuda.synchronize()
    ref = blockscaled_gemm_reference(a_ref, b_ref, sfa_ref, sfb_ref)
    torch.testing.assert_close(mD.float(), ref, atol=ref.abs().max().item() * 2**-7, rtol=1e-2)


@pytest.mark.parametrize("tile,cluster", TILES)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_dense_wide_k(dtype, tile, cluster):
    assert _wide_k_selected(cutlass.Float8E4M3FN, tile, cluster)
    m, n, k = 512, 384, 1024
    torch.manual_seed(0)
    A = (torch.randn(m, k, device="cuda") / math.sqrt(k)).to(dtype)
    B = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(dtype)
    D = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    gemm(A, B, D, None, None, tile[0], tile[1], cluster[0], cluster[1])
    torch.cuda.synchronize()
    ref = A.float() @ B.float().mT
    torch.testing.assert_close(D.float(), ref, atol=ref.abs().max().item() * 2**-7, rtol=1e-2)
