import pytest
import torch

import cutlass

from quack.blockscaled_gemm_utils import blockscaled_gemm_reference
from quack.blockscaled_gemm_utils import compile_blockscaled_gemm_tvm_ffi
from quack.sm120_blockscaled_utils import (
    copy_sm120_nvfp4_scale_blocks_to_storage,
    create_sm120_nvfp4_ab_tensor,
    create_sm120_nvfp4_scale_tensor,
    create_sm120_nvfp4_tensorfill_like_ab_tensor,
    create_sm120_nvfp4_tensorfill_like_scale_tensor,
)


def _skip_if_not_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    if torch.cuda.get_device_capability(0)[0] != 12:
        pytest.skip("SM120 required")


def _make_d(m: int, n: int, l: int) -> torch.Tensor:
    return torch.empty((l, m, n), device="cuda", dtype=torch.bfloat16).permute(1, 2, 0)


def _compile_runner(a, b, d, sfa, sfb):
    return compile_blockscaled_gemm_tvm_ffi(
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
    )


def _store_scale_blocks(storage, blocks, k):
    copy_sm120_nvfp4_scale_blocks_to_storage(storage, blocks, logical_k=k)


def _expand_scale_blocks(blocks, k):
    major, logical_cols, l = blocks.shape
    return (
        blocks.permute(0, 2, 1)
        .unsqueeze(-1)
        .expand(major, l, logical_cols, 16)
        .reshape(major, l, logical_cols * 16)
        .permute(0, 2, 1)
    )[:, :k, :]


def test_sm120_nvfp4_single_cta_uniform_and_k64_scale_split():
    _skip_if_not_sm120()
    m = n = k = 128
    l = 1
    a = create_sm120_nvfp4_ab_tensor(l, m, k, fill_byte=0x22)
    b = create_sm120_nvfp4_ab_tensor(l, n, k, fill_byte=0x22)
    d = _make_d(m, n, l)
    _, sfa = create_sm120_nvfp4_scale_tensor(l, m, k)
    _, sfb = create_sm120_nvfp4_scale_tensor(l, n, k)
    runner = _compile_runner(a, b, d, sfa, sfb)

    d.zero_()
    _store_scale_blocks(sfa, torch.ones((m, k // 16, l), device="cuda"), k)
    _store_scale_blocks(sfb, torch.ones((n, k // 16, l), device="cuda"), k)
    runner(a, b, d, sfa, sfb)
    torch.cuda.synchronize()
    torch.testing.assert_close(d.float(), torch.full_like(d.float(), 128.0))

    d.zero_()
    sfa_blocks = torch.ones((m, k // 16, l), device="cuda")
    sfa_blocks[:, 4:8, :].fill_(2.0)
    _store_scale_blocks(sfa, sfa_blocks, k)
    _store_scale_blocks(sfb, torch.ones((n, k // 16, l), device="cuda"), k)
    runner(a, b, d, sfa, sfb)
    torch.cuda.synchronize()
    torch.testing.assert_close(d.float(), torch.full_like(d.float(), 192.0))
    assert not hasattr(runner, "descriptor_cache")


def test_sm120_nvfp4_k384_scale_page_crossing():
    _skip_if_not_sm120()
    m = n = 128
    k = 384
    l = 2
    a = create_sm120_nvfp4_ab_tensor(l, m, k, fill_byte=0x22)
    b = create_sm120_nvfp4_ab_tensor(l, n, k, fill_byte=0x22)
    d = _make_d(m, n, l)
    _, sfa = create_sm120_nvfp4_scale_tensor(l, m, k)
    _, sfb = create_sm120_nvfp4_scale_tensor(l, n, k)
    runner = _compile_runner(a, b, d, sfa, sfb)

    d.zero_()
    sfa_blocks = torch.ones((m, k // 16, l), device="cuda")
    sfa_blocks[:, 8:16, 1].fill_(2.0)
    _store_scale_blocks(sfa, sfa_blocks, k)
    _store_scale_blocks(sfb, torch.ones((n, k // 16, l), device="cuda"), k)
    runner(a, b, d, sfa, sfb)
    torch.cuda.synchronize()

    expected = torch.full_like(d.float(), 384.0)
    expected[:, :, 1].fill_(512.0)
    torch.testing.assert_close(d.float(), expected)


def test_sm120_nvfp4_nonzero_multi_tile_scale_layout_matches_reference():
    _skip_if_not_sm120()
    torch.manual_seed(120)
    m = n = 256
    k = 256
    l = 1
    a = create_sm120_nvfp4_ab_tensor(l, m, k, fill_byte=0x22)
    b = create_sm120_nvfp4_ab_tensor(l, n, k, fill_byte=0x22)
    a_ref = torch.ones((m, k, l), device="cuda")
    b_ref = torch.ones((n, k, l), device="cuda")
    _, sfa = create_sm120_nvfp4_scale_tensor(l, m, k)
    _, sfb = create_sm120_nvfp4_scale_tensor(l, n, k)
    sfa_blocks = torch.ones((m, k // 16, l), device="cuda")
    sfb_blocks = torch.ones((n, k // 16, l), device="cuda")
    sfa_blocks[128:, 0:8, :].fill_(2.0)
    sfa_blocks[:, 8:16, :].fill_(3.0)
    sfb_blocks[128:, 0:8, :].fill_(2.0)
    sfb_blocks[:, 8:16, :].fill_(4.0)
    _store_scale_blocks(sfa, sfa_blocks, k)
    _store_scale_blocks(sfb, sfb_blocks, k)
    sfa_ref = _expand_scale_blocks(sfa_blocks, k)
    sfb_ref = _expand_scale_blocks(sfb_blocks, k)
    d = _make_d(m, n, l)

    assert torch.all(a_ref != 0)
    assert torch.all(b_ref != 0)
    assert torch.all(sfa_ref != 0)
    assert torch.all(sfb_ref != 0)

    runner = _compile_runner(a, b, d, sfa, sfb)
    d.zero_()
    runner(a, b, d, sfa, sfb)
    torch.cuda.synchronize()

    ref = blockscaled_gemm_reference(a_ref, b_ref, sfa_ref, sfb_ref)
    torch.testing.assert_close(d.float(), ref.to(torch.bfloat16).float())


def test_sm120_nvfp4_row_random_nonzero_multi_tile_matches_reference():
    _skip_if_not_sm120()
    torch.manual_seed(20260522)
    m = n = 256
    k = 256
    l = 1
    a = create_sm120_nvfp4_ab_tensor(l, m, k, fill_byte=0x22)
    b = create_sm120_nvfp4_ab_tensor(l, n, k, fill_byte=0x22)
    a_ref = torch.ones((m, k, l), device="cuda")
    b_ref = torch.ones((n, k, l), device="cuda")
    _, sfa = create_sm120_nvfp4_scale_tensor(l, m, k)
    _, sfb = create_sm120_nvfp4_scale_tensor(l, n, k)
    sfa_blocks = torch.randint(1, 4, (m, k // 16, l), device="cuda").float()
    sfb_blocks = torch.randint(1, 4, (n, k // 16, l), device="cuda").float()
    _store_scale_blocks(sfa, sfa_blocks, k)
    _store_scale_blocks(sfb, sfb_blocks, k)
    sfa_ref = _expand_scale_blocks(sfa_blocks, k)
    sfb_ref = _expand_scale_blocks(sfb_blocks, k)
    d = _make_d(m, n, l)

    assert torch.all(a_ref != 0)
    assert torch.all(b_ref != 0)
    assert torch.all(sfa_ref != 0)
    assert torch.all(sfb_ref != 0)

    runner = _compile_runner(a, b, d, sfa, sfb)
    d.zero_()
    runner(a, b, d, sfa, sfb)
    torch.cuda.synchronize()

    ref = blockscaled_gemm_reference(a_ref, b_ref, sfa_ref, sfb_ref)
    torch.testing.assert_close(d.float(), ref.to(torch.bfloat16).float())


def test_sm120_nvfp4_tensorfill_like_6x6_tiles_matches_reference():
    _skip_if_not_sm120()
    torch.manual_seed(20260524)
    m = n = k = 768
    l = 1
    a_ref, a = create_sm120_nvfp4_tensorfill_like_ab_tensor(l, m, k)
    b_ref, b = create_sm120_nvfp4_tensorfill_like_ab_tensor(l, n, k)
    sfa_ref, sfa = create_sm120_nvfp4_tensorfill_like_scale_tensor(l, m, k)
    sfb_ref, sfb = create_sm120_nvfp4_tensorfill_like_scale_tensor(l, n, k)
    d = _make_d(m, n, l)

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
