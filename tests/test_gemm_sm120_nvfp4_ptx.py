import pytest
import torch

import cutlass

from quack.blockscaled_gemm_utils import compile_blockscaled_gemm_tvm_ffi
from quack.sm120_blockscaled_utils import create_sm120_nvfp4_ab_tensor
from quack.sm120_blockscaled_utils import create_sm120_nvfp4_scale_tensor


def _skip_if_not_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    if torch.cuda.get_device_capability(0)[0] != 12:
        pytest.skip("SM120 required")


def test_sm120_nvfp4_ptx_contains_compact_tma_mainloop(tmp_path, monkeypatch):
    _skip_if_not_sm120()
    monkeypatch.chdir(tmp_path)
    m = n = k = 128
    l = 1
    a = create_sm120_nvfp4_ab_tensor(l, m, k, fill_byte=0x22)
    b = create_sm120_nvfp4_ab_tensor(l, n, k, fill_byte=0x22)
    d = torch.empty((l, m, n), device="cuda", dtype=torch.bfloat16).permute(1, 2, 0)
    _, sfa = create_sm120_nvfp4_scale_tensor(l, m, k)
    _, sfb = create_sm120_nvfp4_scale_tensor(l, n, k)

    runner = compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        cutlass.BFloat16,
        (128, 128),
        (1, 1),
        a,
        b,
        d,
        sfa,
        sfb,
        keep_ptx=True,
    )
    ptx = runner.compiled.__ptx__
    assert ptx is not None

    assert ptx.count("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier") == 2
    assert ptx.count("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier") == 2
    assert ptx.count("cp.async.bulk.tensor.3d.global.shared::cta.tile") == 0
    assert ptx.count("st.global.b") == 128
    assert ptx.count("ldmatrix.sync.aligned.m8n8.x4.shared.b16") == 16
    assert (
        ptx.count("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X")
        == 64
    )
    assert "tcgen05" not in ptx
    assert "shared::cluster" not in ptx
    assert ".multicast" not in ptx
    assert "fence.proxy.tensormap::generic.acquire.gpu" not in ptx
