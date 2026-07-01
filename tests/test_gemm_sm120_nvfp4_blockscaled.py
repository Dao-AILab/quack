import pytest
import torch

import cutlass

from quack.blockscaled_gemm_utils import blockscaled_gemm_reference
from quack.blockscaled_gemm_utils import compile_blockscaled_gemm_tvm_ffi
from quack.sm120_blockscaled_utils import (
    create_sm120_nvfp4_ab_tensor,
    create_sm120_nvfp4_scale_tensor,
    create_sm120_nvfp4_tensorfill_like_ab_tensor,
    create_sm120_nvfp4_tensorfill_like_scale_tensor,
    validate_sm120_nvfp4_scale_storage,
)


def _skip_if_not_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    if torch.cuda.get_device_capability(0)[0] != 12:
        pytest.skip("SM120 required")


def _make_d(m: int, n: int, l: int, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.empty((l, m, n), device="cuda", dtype=dtype).permute(1, 2, 0)


def _make_problem(m: int, n: int, k: int, l: int):
    a_ref, a = create_sm120_nvfp4_tensorfill_like_ab_tensor(l, m, k)
    b_ref, b = create_sm120_nvfp4_tensorfill_like_ab_tensor(l, n, k)
    sfa_ref, sfa = create_sm120_nvfp4_tensorfill_like_scale_tensor(l, m, k)
    sfb_ref, sfb = create_sm120_nvfp4_tensorfill_like_scale_tensor(l, n, k)
    d = _make_d(m, n, l)
    return a_ref, b_ref, sfa_ref, sfb_ref, a, b, d, sfa, sfb


def _compile_runner(
    a,
    b,
    d,
    sfa,
    sfb,
    *,
    keep_ptx: bool = False,
    sm120_nvfp4_path: str = "validated",
    ab_dtype=cutlass.Float4E2M1FN,
    sf_dtype=cutlass.Float8E4M3FN,
    sf_vec_size: int = 16,
    d_dtype=cutlass.BFloat16,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    varlen_m: bool = False,
    varlen_k: bool = False,
):
    return compile_blockscaled_gemm_tvm_ffi(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        d_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        a,
        b,
        d,
        sfa,
        sfb,
        keep_ptx=keep_ptx,
        sm120_nvfp4_path=sm120_nvfp4_path,
        varlen_m=varlen_m,
        varlen_k=varlen_k,
    )


@pytest.mark.parametrize("m,n,k,l", [(128, 128, 128, 1), (256, 256, 256, 1)])
def test_sm120_nvfp4_validated_matches_tensorfill_like_reference(m, n, k, l):
    _skip_if_not_sm120()
    torch.manual_seed(20260525 + m + n + k)
    a_ref, b_ref, sfa_ref, sfb_ref, a, b, d, sfa, sfb = _make_problem(m, n, k, l)

    assert torch.all(a_ref != 0)
    assert torch.all(b_ref != 0)
    assert torch.all(sfa_ref != 0)
    assert torch.all(sfb_ref != 0)

    runner = _compile_runner(a, b, d, sfa, sfb)
    d.zero_()
    runner(a, b, d, sfa, sfb)
    torch.cuda.synchronize()

    ref = blockscaled_gemm_reference(a_ref, b_ref, sfa_ref, sfb_ref)
    torch.testing.assert_close(d.float(), ref, atol=0.25, rtol=2e-2)


def test_sm120_nvfp4_fast_small_shape_runs_and_matches_reference():
    _skip_if_not_sm120()
    torch.manual_seed(20260526)
    m = n = k = 128
    l = 1
    a_ref, b_ref, sfa_ref, sfb_ref, a, b, d, sfa, sfb = _make_problem(m, n, k, l)

    runner = _compile_runner(a, b, d, sfa, sfb, sm120_nvfp4_path="fast")
    d.zero_()
    runner(a, b, d, sfa, sfb)
    torch.cuda.synchronize()

    ref = blockscaled_gemm_reference(a_ref, b_ref, sfa_ref, sfb_ref)
    torch.testing.assert_close(d.float(), ref, atol=0.25, rtol=2e-2)


def test_sm120_nvfp4_validated_and_fast_ptx_paths(tmp_path, monkeypatch):
    _skip_if_not_sm120()
    monkeypatch.chdir(tmp_path)
    m = n = k = 128
    l = 1
    a = create_sm120_nvfp4_ab_tensor(l, m, k, fill_byte=0x22)
    b = create_sm120_nvfp4_ab_tensor(l, n, k, fill_byte=0x22)
    d = _make_d(m, n, l)
    _, sfa = create_sm120_nvfp4_scale_tensor(l, m, k)
    _, sfb = create_sm120_nvfp4_scale_tensor(l, n, k)

    validated = _compile_runner(a, b, d, sfa, sfb, keep_ptx=True)
    fast = _compile_runner(a, b, d, sfa, sfb, keep_ptx=True, sm120_nvfp4_path="fast")
    validated_ptx = validated.compiled.__ptx__
    fast_ptx = fast.compiled.__ptx__
    assert validated_ptx is not None
    assert fast_ptx is not None

    mma = "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
    assert validated_ptx.count(mma) == 64
    assert fast_ptx.count(mma) == 64
    assert validated_ptx.count("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier") == 2
    assert validated_ptx.count("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier") == 2
    assert fast_ptx.count("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier") == 2
    assert fast_ptx.count("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier") == 2

    assert validated_ptx.count("st.global.b") == 128
    assert fast_ptx.count("st.global.b") == 0
    assert "cp.async.bulk.tensor.3d.global.shared::cta" in fast_ptx


def test_sm120_nvfp4_rejects_unsupported_user_contracts():
    _skip_if_not_sm120()
    m = n = k = 128
    l = 1
    a = create_sm120_nvfp4_ab_tensor(l, m, k, fill_byte=0x22)
    b = create_sm120_nvfp4_ab_tensor(l, n, k, fill_byte=0x22)
    d = _make_d(m, n, l)
    _, sfa = create_sm120_nvfp4_scale_tensor(l, m, k)
    _, sfb = create_sm120_nvfp4_scale_tensor(l, n, k)

    unsupported = "SM120 blockscaled GEMM currently supports only NVFP4|SM120 NVFP4"
    with pytest.raises((RuntimeError, ValueError), match=unsupported):
        _compile_runner(a, b, d, sfa, sfb, mma_tiler_mn=(64, 128))
    with pytest.raises((RuntimeError, ValueError), match=unsupported):
        _compile_runner(a, b, d, sfa, sfb, cluster_shape_mn=(2, 1))
    with pytest.raises((RuntimeError, ValueError), match=unsupported):
        _compile_runner(a, b, d, sfa, sfb, ab_dtype=cutlass.Float8E4M3FN)
    with pytest.raises((RuntimeError, ValueError), match=unsupported):
        _compile_runner(a, b, _make_d(m, n, l, torch.float16), sfa, sfb, d_dtype=cutlass.Float16)
    with pytest.raises((RuntimeError, ValueError), match=unsupported):
        _compile_runner(a, b, d, sfa, sfb, varlen_m=True)

    _logical_cols, _physical_cols, pages = validate_sm120_nvfp4_scale_storage(
        sfa, logical_k=k, major_extent=m, batch_extent=l
    )
    legacy_rank4_sfa = torch.empty((m, 16, pages, l), device="cuda", dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="compact 1D interleaved FP8"):
        _compile_runner(a, b, d, legacy_rank4_sfa, sfb)
