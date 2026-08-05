# Probe: fp6 TMA-unpack numerics vs row pitch.
# _reinterpret_packed_fp6 scales non-K byte strides by *4//3 (floor). Exactness
# in the element domain requires pitch % 3 == 0; this probe checks whether
# 32B-aligned-but-not-%3 pitches still produce correct results (hypothesis: the
# DSL rounds sub-byte strides back up to TMA's 16B granularity, recovering the
# exact pitch for any 32B-aligned value).
import torch

from quack.gemm_interface import gemm, gemm_blockscaled_ref
from quack.blockscaled.operand import BlockScaledOperand

torch.manual_seed(0)
m = n = 128
k = 512  # storage rows: 3*512/4 = 384 bytes


def relayout(op, offset_bytes, row_pad_bytes):
    rows, storage_k = op.qdata.shape
    row_stride = storage_k + row_pad_bytes
    raw = torch.full(
        (offset_bytes + rows * row_stride,),
        0xA5,  # poison the padding: floor-stride bugs would read into it
        dtype=torch.uint8,
        device=op.qdata.device,
    )
    qdata = raw[offset_bytes:].as_strided((rows, storage_k), (row_stride, 1))
    qdata.view(torch.uint8).copy_(op.qdata.view(torch.uint8))
    return BlockScaledOperand.from_parts(qdata, op.scale, op.format, orig_dtype=op.orig_dtype)


x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
A = BlockScaledOperand.quantize(x, "mxfp8_e4m3")
W0 = BlockScaledOperand.quantize(w, "mxfp6_e2m3_packed")

print(f"{'pad':>5} {'pitch':>6} {'%3':>3} {'%32':>4}  rel_err")
for pad in [0, 32, 64, 96, 160, 3744]:
    pitch = 384 + pad
    W = relayout(W0, 32, pad)
    try:
        out = gemm(A, W.mT, tuned=False)
        ref = gemm_blockscaled_ref(A, W.mT)
        rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
        status = f"{rel:.2e}" + ("  <-- WRONG" if rel > 5e-3 else "")
    except Exception as e:
        status = f"raised: {type(e).__name__}: {str(e)[:90]}"
    print(f"{pad:>5} {pitch:>6} {pitch % 3:>3} {pitch % 32:>4}  {status}")
torch.cuda.synchronize()
