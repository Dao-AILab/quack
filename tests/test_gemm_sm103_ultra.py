# SM103 (B300) K=96 "ultra" FP4 mainloop (GemmSm100.mma_sm103_ultra): K=96 MMA groups
# straddling k-tiles, K=64 leftover k-tiles, 3-slot SF tmem, misaligned SFB tiles, split-K.
import pytest
import torch

import quack.cache as cache_state
import quack.gemm
from quack.blockscaled.operand import BlockScaledFormat, BlockScaledOperand
from quack.gemm import _compile_gemm, gemm
from quack.gemm_sm100 import GemmSm100

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="SM103 (B300) only",
)


def _check(fmt, tile_mn, cluster_mn, k, m=512, n=448, split_k=1):
    fmt = BlockScaledFormat.from_name(fmt)
    ops = [
        BlockScaledOperand.quantize(
            torch.randn(1, mn, k, device="cuda", dtype=torch.bfloat16) * k**-0.5, fmt
        )
        for mn in (m, n)
    ]
    D = torch.empty(1, m, n, dtype=torch.bfloat16, device="cuda")
    gemm(
        ops[0].qdata,
        ops[1].qdata,
        D,
        None,
        None,
        *tile_mn,
        *cluster_mn,
        SFA=ops[0].scale,
        SFB=ops[1].scale,
        bs_format_a=fmt.name,
        bs_format_b=fmt.name,
        split_k=split_k,
    )
    a_ref, b_ref = (op.dequantize(torch.float32) for op in ops)
    ref = torch.einsum("lmk,lnk->lmn", a_ref, b_ref)
    torch.testing.assert_close(D.float(), ref, atol=5e-3, rtol=1e-3)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize(
    "tile_mn,cluster_mn",
    [
        ((128, 64), (1, 1)),
        ((128, 192), (1, 1)),
        ((128, 256), (1, 1)),
        ((256, 64), (2, 1)),
        ((256, 128), (2, 1)),
        ((256, 192), (2, 1)),
        ((256, 256), (2, 1)),
    ],
)
# k-tiles of 256: 3 = one K=96 group, 4 = group + 1 leftover, 5 = group + 2, 9 = 3 groups
# n = 448: the last tile_n=192/64 tile starts mid SF atom
@pytest.mark.parametrize("k", [768, 1024, 1280, 2304])
def test_sm103_ultra(fmt, tile_mn, cluster_mn, k):
    _check(fmt, tile_mn, cluster_mn, k)


@pytest.mark.parametrize("split_k", [2, 3])
def test_sm103_ultra_split_k(split_k):
    # 13 k-tiles: splits get group + leftover mixes
    _check("nvfp4", (256, 256), (2, 1), 3328, n=512, split_k=split_k)


def test_sm103_ultra_traced(monkeypatch):
    # The K=64 and K=96 paths are bit-identical, so assert the K=96 mainloop was traced.
    orig, calls = GemmSm100.mma_sm103_ultra, []
    monkeypatch.setattr(GemmSm100, "mma_sm103_ultra", lambda *a: calls.append(1) or orig(*a))
    monkeypatch.setattr(cache_state, "CACHE_ENABLED", False)
    monkeypatch.setattr(quack.gemm, "_gemm_plan_cache", {})
    _compile_gemm.cache_clear()
    _check("nvfp4", (256, 256), (2, 1), 768)
    assert calls
