# Copyright (c) 2026, QuACK team.
"""Rubin wide-K MMA selection and ragged-K correctness."""

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
from quack.sm100_utils import MmaKind

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
    gemm_op.varlen_k = varlen_k
    gemm_op.a_unpack = gemm_op.b_unpack = False
    kind = MmaKind.for_operands(ab_dtype, ab_dtype, sf_vec_size is not None)
    return gemm_op._mma_inst_bits_k(kind) == 512


@pytest.mark.parametrize(
    "ab_dtype,sf_vec_size,tile,cluster,tile_k,varlen_k,expected",
    [
        (cutlass.Float8E4M3FN, None, (128, 256), (1, 1), 0, False, True),
        (cutlass.Float8E4M3FN, None, (256, 256), (2, 1), 0, False, True),
        (cutlass.Float8E4M3FN, 32, (256, 192), (2, 1), 0, False, True),
        (cutlass.Float4E2M1FN, 16, (128, 64), (1, 1), 0, False, True),
        (cutlass.Float8E4M3FN, None, (256, 256), (2, 1), 256, False, True),
        # No wide fp8 form at 64 rows per CTA.
        (cutlass.Float8E4M3FN, None, (64, 256), (1, 1), 0, False, False),
        (cutlass.Float8E4M3FN, None, (128, 256), (2, 1), 0, False, False),
        # tile_k=96 is only narrow-instruction aligned.
        (cutlass.Float8E4M3FN, None, (256, 256), (2, 1), 96, False, False),
        # Arbitrary pad scales require narrow mxfp8.
        (cutlass.Float8E4M3FN, 32, (256, 256), (2, 1), 0, True, False),
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
        ("mxfp4", 384),  # One all-pad instruction in the last k-tile.
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
