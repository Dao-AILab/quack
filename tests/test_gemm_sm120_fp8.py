# Copyright (c) 2026, Tri Dao.
"""SM120 fp8 GEMM (warp-level MmaFP8Op m16n8k32): numerics vs an fp32
reference on dequantized operands.

No slow-accum path exists on SM120 by design: the fp8 mma.sync f32
accumulator keeps ~21-22 mantissa bits (measured on RTX 5090 silicon —
+1 onto 2^n survives through n=20, one bit short of the bf16 datapath's
n=21; the add truncates rather than rounds). Error drift is therefore
~(K/32) * 2^-21 relative — negligible at any practical K — unlike Hopper's
~13-bit QGMMA accumulator that motivates GemmSm90.fp8_slow_accum.
"""

import math

import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm

# QUACK_ARCH-aware, like test_gemm_transform: the fp8 warp-MMA path is
# sm_89+ mma.sync, so it compiles and runs correctly on the H100 CI proxy
# leg (QUACK_ARCH=120). The accumulator numbers in the docstring are RTX
# 5090 measurements; H100's mma.sync fp8 accumulate is full fp32 RNE
# (+1 onto 2^n survives through n=23, +3 rounds to +4 at ulp 2 — measured
# with the same probe), so the drift bound holds with margin there.
_ARCH = get_device_capacity(torch.device("cuda"))[0] if torch.cuda.is_available() else 0
requires_sm120 = pytest.mark.skipif(_ARCH != 12, reason="SM120 fp8 warp-MMA path")
# The blockscaled MMA kinds need real sm_120/121 silicon, not the QUACK_ARCH=120 proxy.
_PHYSICAL_ARCH = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
requires_sm120_silicon = pytest.mark.skipif(
    _PHYSICAL_ARCH != 12, reason="kind::mxf8f6f4 needs sm_120/121 silicon"
)

_DTYPES = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def _run(m, n, k, dtype, tile_mn, tile_k=None, pingpong=False, split_k=1):
    torch.manual_seed(0)
    A = (torch.randn(m, k, device="cuda") / math.sqrt(k)).to(dtype)
    B = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(dtype)
    D = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    # split_k requires batched operands (the raw 2D-unbatched path asserts)
    batch = lambda t: t.unsqueeze(0) if split_k != 1 else t
    gemm(
        batch(A),
        batch(B),
        batch(D),
        None,
        None,
        tile_mn[0],
        tile_mn[1],
        1,
        1,
        tile_K=tile_k,
        pingpong=pingpong,
        split_k=split_k,
    )
    torch.cuda.synchronize()
    ref = A.float() @ B.float().mT
    # fp8 operands are exact; the only error sources are the truncating
    # tensor-core accumulate (~2^-21/step) and the bf16 store rounding.
    atol = ref.abs().max().item() * 2**-7 + 1e-6
    torch.testing.assert_close(D.float(), ref, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("dtype", ["e4m3", "e5m2"])
@requires_sm120
def test_fp8_basic(dtype):
    _run(512, 1024, 1024, _DTYPES[dtype], (128, 128))


@requires_sm120
def test_fp8_pingpong():
    _run(768, 1536, 2048, torch.float8_e4m3fn, (128, 128), pingpong=True)


@pytest.mark.parametrize("tile_mn,tile_k", [((128, 256), 64), ((64, 128), 128)])
@requires_sm120
def test_fp8_tiles(tile_mn, tile_k):
    # ragged M/N exercises the predicated epilogue; k tail the TMA zero-fill
    _run(384, 896, 1088, torch.float8_e4m3fn, tile_mn, tile_k=tile_k)


@requires_sm120
def test_fp8_split_k():
    _run(256, 512, 8192, torch.float8_e4m3fn, (128, 128), split_k=4)


@requires_sm120
def test_fp8_long_k_accum_drift():
    """The fast-accum guardrail: at K = 16384 the truncating accumulate must
    stay within the ~(K/32)*2^-21 drift bound — if a future change silently
    routes fp8 through a lower-precision accumulate (e.g. f16 acc without
    promotion), the all-positive operands here bias the drift far past it."""
    m, n, k = 128, 256, 16384
    torch.manual_seed(1)
    A = torch.rand(m, k, device="cuda").mul(0.5).to(torch.float8_e4m3fn)
    B = torch.rand(n, k, device="cuda").mul(0.5).to(torch.float8_e4m3fn)
    D = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    gemm(A, B, D, None, None, 128, 128, 1, 1)
    torch.cuda.synchronize()
    ref = A.float() @ B.float().mT
    rel = ((D.float() - ref) / ref).abs().max().item()
    # bf16 store alone is 2^-8; the accumulate adds ~(K/32)*2^-21 (~2.4e-4)
    assert rel < 6e-3, f"fp8 accumulate drift {rel:.2e} exceeds the fp32-acc bound"


@requires_sm120
@pytest.mark.parametrize("tile_m", [pytest.param(128, marks=requires_sm120_silicon), 64])
def test_fp8_mixed_e4m3_e5m2(tile_m):
    """Mixed e4m3 x e5m2 (not blockscaled). tile_m=128 rides kind::mxf8f6f4 with
    constant unit scales (full Blackwell rate); tile_m=64 forces the Ada-instruction
    fallback (MmaFP8MixedOp - the sm_89 opcode takes independent .e4m3/.e5m2
    qualifiers), the same path the H100 QUACK_ARCH=120 CI leg takes."""
    torch.manual_seed(0)
    m, n, k = 256, 256, 512
    A = (torch.randn(m, k, device="cuda") / math.sqrt(k)).to(torch.float8_e4m3fn)
    B = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.float8_e5m2)
    D = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    gemm(A, B, D, None, None, tile_m, 128, 1, 1)
    torch.cuda.synchronize()
    ref = A.float() @ B.float().mT
    atol = ref.abs().max().item() * 2**-7 + 1e-6
    torch.testing.assert_close(D.float(), ref, atol=atol, rtol=1e-2)


@requires_sm120
@pytest.mark.parametrize("cluster_mn", [(1, 2), (2, 1)])
def test_fp8_cluster_rejected(cluster_mn):
    """SM120 has no TMA multicast, so the GEMM rejects clusters other than 1x1 instead of
    compiling multicast loads to driver syscalls."""
    A = torch.randn(256, 512, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(256, 512, device="cuda").to(torch.float8_e4m3fn)
    D = torch.empty(256, 256, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match=r"cluster shape \(1, 1, 1\)"):
        gemm(A, B, D, None, None, 128, 128, *cluster_mn)
