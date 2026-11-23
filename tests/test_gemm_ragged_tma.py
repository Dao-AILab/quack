"""Regression tests for issue #243: TMA loads through the 2-extra-dim ragged descriptor
(copy_utils.create_ragged_tensor_for_tma, ptr_shift=False) must not touch memory past the
operand. Each case runs in a subprocess, since an illegal address is a sticky CUDA error
that would poison the pytest process. The operand is placed flush against the end of a
physical mapping inside a larger reserved VA range, so any read past it faults.
"""

import pytest

_MAPPING_END_PRELUDE = r"""
import math
import sys
import torch
from cuda.bindings import driver as cu

dev, dt = torch.device("cuda"), torch.bfloat16
torch.cuda.init()
torch.empty(1, device=dev)  # bind the primary context
torch.manual_seed(0)

def check(r):
    err, *rest = r if isinstance(r, tuple) else (r,)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    return rest[0] if len(rest) == 1 else rest

cu_dev = check(cu.cuDeviceGet(torch.cuda.current_device()))
if not check(cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu_dev)):
    sys.exit(77)  # reported as a skip by the parent

prop = cu.CUmemAllocationProp()
prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
prop.location.id = torch.cuda.current_device()
gran = int(check(cu.cuMemGetAllocationGranularity(
    prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)))
size = gran * -(-(2 << 20) // gran)
va = check(cu.cuMemAddressReserve(4 * size, gran, 0, 0))  # only the first `size` bytes get mapped
handle = check(cu.cuMemCreate(size, prop, 0))
check(cu.cuMemMap(va, size, 0, handle, 0))
access = cu.CUmemAccessDesc()
access.location = prop.location
access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
check(cu.cuMemSetAccess(va, size, [access], 1))

class _Buf:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = dict(
            shape=(nbytes,), typestr="|u1", data=(ptr, False), version=3, strides=None)

def at_mapping_end(rows, cols):
    nbytes = rows * cols * 2
    assert 0 < nbytes <= size
    raw = torch.as_tensor(_Buf(int(va) + size - nbytes, nbytes), device=dev)
    return raw.view(dt).view(rows, cols)

which = sys.argv[1]
lens = [int(x) for x in sys.argv[3:]]
total = sum(lens)
cu_seqlens = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), device=dev,
                          dtype=torch.int32)
"""

_MAPPING_END_VARLEN_K = r"""
from quack.gemm_interface import gemm, gemm_ref

gather_A = sys.argv[2] == "1"
M, N = 384, 256
A_idx = None
if gather_A:
    A = torch.randn(2 * total, M, device=dev, dtype=dt).T  # m-major
    A_idx = torch.randperm(2 * total, device=dev, dtype=torch.int32)[:total]
elif which == "A":
    A = at_mapping_end(total, M).T  # (M, total_k), m-major, flush against the mapping end
    A.copy_(torch.randn(M, total, device=dev, dtype=dt))
else:
    A = torch.randn(total, M, device=dev, dtype=dt).T
if which == "B":
    B = at_mapping_end(total, N)
    B.copy_(torch.randn(total, N, device=dev, dtype=dt))
else:
    B = torch.randn(total, N, device=dev, dtype=dt)
out = gemm(A, B, cu_seqlens_k=cu_seqlens, A_idx=A_idx, tuned=False)
torch.cuda.synchronize()
ref = gemm_ref(A.float(), B.float(), cu_seqlens_k=cu_seqlens, A_idx=A_idx)
ref_bf16 = gemm_ref(A, B, cu_seqlens_k=cu_seqlens, A_idx=A_idx)
err = (out.float() - ref).abs().max().item()
err_bf16 = (ref_bf16.float() - ref).abs().max().item()
assert err <= 2 * err_bf16 + 1e-5, (err, err_bf16)
print("OK")
"""

# For varlen_m, an active M-fold (rowvec) sink is what makes the A and C loads go through
# the ragged descriptor (zero-filled rows past each sequence end).
_MAPPING_END_VARLEN_M_SINK = r"""
from quack.cute_dsl_utils import get_device_capacity
from quack.epilogue.library import dact_dbias_mod
from quack.gemm_config import cta_tile_shape_m

K, N, tile_M, tile_N = 64, 256, 128, 256
num_seqs = len(lens)
def operand(name, rows, cols, scale):
    T = at_mapping_end(rows, cols) if which == name else torch.empty(rows, cols, device=dev, dtype=dt)
    return T.copy_(torch.randn(rows, cols, device=dev, dtype=dt) * scale)
A = operand("A", total, K, 4 / math.sqrt(K))
z = operand("C", total, N, 1.0)  # saved preact, loaded as C
B = torch.randn(num_seqs, N, K, device=dev, dtype=dt) / math.sqrt(K) * 4
D = torch.empty(total, N, device=dev, dtype=dt)
mod = dact_dbias_mod("gelu_tanh_approx")
op = mod.sinks["dwb"]
cta_m = cta_tile_shape_m(tile_M, 1, get_device_capacity(dev)[0])
dwb = torch.empty(op.sink_alloc_shape((total,), N, cta_m, tile_N, num_seqs=num_seqs),
                  device=dev, dtype=torch.float32)
mod.gemm(A, B, D, z, epi_args=dict(dwb=dwb), cu_seqlens_m=cu_seqlens,
         tile_M=tile_M, tile_N=tile_N, cluster_M=1, cluster_N=1)
torch.cuda.synchronize()
dwb_seq = op.host_finalize_varlen(dwb, cu_seqlens, cta_m)
bounds = cu_seqlens.tolist()
zf = z.float().requires_grad_()
da = torch.cat([A[s:e].float() @ B[i].float().mT for i, (s, e) in enumerate(zip(bounds, bounds[1:]))])
torch.nn.functional.gelu(zf, approximate="tanh").backward(da)
dz = zf.grad
def rel(a, b):
    return ((a.float() - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()
assert rel(D, dz) < 3e-2, rel(D, dz)
for i, (s, e) in enumerate(zip(bounds, bounds[1:])):
    assert rel(dwb_seq[i], dz[s:e].sum(0)) < 2e-3, i
print("OK")
"""


def _run_mapping_end(body, *args):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _MAPPING_END_PRELUDE + body, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode == 77:
        pytest.skip("CUDA virtual memory management not supported on this device")
    assert result.returncode == 0 and "OK" in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("lens", [[4], [4, 64], [64, 4], [3, 5, 7]])
@pytest.mark.parametrize("which,gather_A", [("B", False), ("B", True), ("A", False)])
def test_gemm_varlen_k_operand_at_mapping_end(which, gather_A, lens):
    """Partial last K tiles must not read past an operand that ends at a mapping boundary.

    Regression test for issue #243: with MAX_INT extents on the ragged TMA descriptor's
    extra dims, the load touched ~128 KiB from the operand's base address.
    """
    _run_mapping_end(_MAPPING_END_VARLEN_K, which, int(gather_A), *lens)


@pytest.mark.parametrize("lens", [[4], [20, 17], [128, 5]])
@pytest.mark.parametrize("which", ["A", "C"])
def test_gemm_varlen_m_sink_operand_at_mapping_end(which, lens):
    """Same as above for the varlen_m zero-fill loads of A and C (issue #243)."""
    _run_mapping_end(_MAPPING_END_VARLEN_M_SINK, which, 0, *lens)
