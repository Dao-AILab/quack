from pathlib import Path

import pytest
import torch

import cutlass

from quack.gemm_sm120 import GemmSm120
from quack.sm120_blockscaled_utils import (
    create_sm120_nvfp4_ab_tensor,
    create_sm120_nvfp4_scale_tensor,
    validate_sm120_nvfp4_ab_storage,
    validate_sm120_nvfp4_d_storage,
    validate_sm120_nvfp4_scale_storage,
)


def _skip_if_not_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    if torch.cuda.get_device_capability(0)[0] != 12:
        pytest.skip("SM120 required")


def test_sm120_nvfp4_facade_and_config_validation():
    import quack.sm120_utils as sm120_utils

    assert sm120_utils.get_ab_tma_tx_bytes() == 8192
    assert sm120_utils.get_scale_tma_tx_bytes() == 1024
    assert sm120_utils.get_full_tma_tx_bytes() == 18432

    valid = (
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        cutlass.BFloat16,
        (128, 128, 128),
        (1, 1),
        128,
        128,
        128,
        1,
        "k",
        "k",
        "n",
    )
    assert GemmSm120.can_implement_blockscaled(*valid)
    assert not GemmSm120.can_implement_blockscaled(*valid[:4], (64, 64, 128), *valid[5:])
    assert not GemmSm120.can_implement_blockscaled(*valid[:4], (128, 64, 128), *valid[5:])
    assert not GemmSm120.can_implement_blockscaled(*valid[:5], (2, 1), *valid[6:])
    assert not GemmSm120.can_implement_blockscaled(*valid[:10], "m", *valid[11:])
    assert not GemmSm120.can_implement_blockscaled(*valid[:11], "n", *valid[12:])
    assert not GemmSm120.can_implement_blockscaled(*valid[:12], "m")

    with pytest.raises(ValueError, match="\\(128,128,128\\)"):
        GemmSm120(
            cutlass.Float32,
            cutlass.Float4E2M1FN,
            (64, 64, 128),
            (1, 1, 1),
            pingpong=False,
            sf_vec_size=16,
            sf_dtype=cutlass.Float8E4M3FN,
        )


def test_sm120_dense_pingpong_constructor_still_works():
    GemmSm120(
        cutlass.Float32,
        cutlass.BFloat16,
        (128, 128, 64),
        (1, 1, 1),
        pingpong=True,
    )


def test_sm120_nvfp4_path_policy_selects_scheduler_and_epilogue():
    validated = GemmSm120(
        cutlass.Float32,
        cutlass.Float4E2M1FN,
        (128, 128, 128),
        (1, 1, 1),
        pingpong=True,
        sf_vec_size=16,
        sf_dtype=cutlass.Float8E4M3FN,
    )
    assert validated.direct_global_store
    assert validated.direct_cute_static_scheduler

    fast = GemmSm120(
        cutlass.Float32,
        cutlass.Float4E2M1FN,
        (128, 128, 128),
        (1, 1, 1),
        pingpong=True,
        sf_vec_size=16,
        sf_dtype=cutlass.Float8E4M3FN,
        sm120_nvfp4_path="fast",
    )
    assert not fast.direct_global_store
    assert not fast.direct_cute_static_scheduler

    with pytest.raises(ValueError, match="validated.*fast"):
        GemmSm120(
            cutlass.Float32,
            cutlass.Float4E2M1FN,
            (128, 128, 128),
            (1, 1, 1),
            pingpong=True,
            sf_vec_size=16,
            sf_dtype=cutlass.Float8E4M3FN,
            sm120_nvfp4_path="unknown",
        )


def test_sm120_nvfp4_source_has_no_experimental_env_matrix():
    source = (Path(__file__).parents[1] / "quack/gemm_sm120.py").read_text()

    assert "os.environ" not in source
    assert "QUACK_SM120_NVFP4" not in source
    assert "blockscaled_kernel_legacy" not in source
    assert "get_native_tma_desc_addr" not in source


def test_sm120_nvfp4_storage_validation():
    _skip_if_not_sm120()
    m = n = k = 128
    l = 2
    a = create_sm120_nvfp4_ab_tensor(l, m, k, fill_byte=0x22)
    d = torch.empty((l, m, n), device="cuda", dtype=torch.bfloat16).permute(1, 2, 0)
    _, sfa = create_sm120_nvfp4_scale_tensor(l, m, k)

    validate_sm120_nvfp4_ab_storage(a, logical_k=k, major_extent=m, batch_extent=l)
    validate_sm120_nvfp4_d_storage(d, m=m, n=n, l=l)
    _logical_cols, _physical_cols, pages = validate_sm120_nvfp4_scale_storage(
        sfa, logical_k=k, major_extent=m, batch_extent=l
    )

    with pytest.raises(ValueError, match="shape"):
        validate_sm120_nvfp4_ab_storage(
            a.transpose(0, 1), logical_k=k, major_extent=m, batch_extent=l
        )
    legacy_physical_scale = torch.empty((m, 16, pages, l), device="cuda", dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="compact 1D interleaved FP8"):
        validate_sm120_nvfp4_scale_storage(
            legacy_physical_scale, logical_k=k, major_extent=m, batch_extent=l
        )
