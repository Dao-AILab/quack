import os

import pytest
import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline.sm90 import PipelineTmaAsync, make_pipeline_state

from quack.sm120_tma_utils import (
    assert_sm120_direct_tma_2d,
    get_sm120_direct_tma_desc_addr,
    has_sm120_direct_tma_2d,
    make_sm120_direct_tma_load_2d_atom,
    make_sm120_direct_tma_smem_layout_2d,
    make_sm120_tma_basis_tensor_2d,
    sm120_direct_tma_load_2d,
)


def _require_sm120_direct_tma():
    if not has_sm120_direct_tma_2d():
        pytest.skip("local CuTe DSL SM120 direct TMA helpers are unavailable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 direct TMA tests require compute capability 12.x")
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")


@cute.kernel
def _sm120_direct_tma_copy_kernel(
    tma_atom: cute.CopyAtom,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
    use_swizzle: cutlass.Constexpr[bool],
):
    smem = cutlass.utils.SmemAllocator()
    smem_layout = make_sm120_direct_tma_smem_layout_2d(d_tile, seq_tile)
    if cutlass.const_expr(use_swizzle):
        s_tile = smem.allocate_tensor(
            dtype,
            smem_layout,
            byte_alignment=128,
            swizzle=cute.make_swizzle(3, 4, 3),
        )
    else:
        s_tile = smem.allocate_tensor(dtype, smem_layout, byte_alignment=128)
    s_mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(2), byte_alignment=8)

    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // 32
    lane = tidx % 32
    desc_ptr = get_sm120_direct_tma_desc_addr(tma_atom)
    pipe = PipelineTmaAsync.create(
        barrier_storage=s_mbar.iterator,
        num_stages=1,
        producer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1),
        consumer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1),
        tx_count=d_tile * seq_tile * dtype.width // 8,
        defer_sync=False,
    )

    if warp == 0:
        producer_state = make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
        pipe.producer_acquire(producer_state)
        sm120_direct_tma_load_2d(
            s_tile.iterator,
            desc_ptr,
            pipe.producer_get_barrier(producer_state),
            coord0,
            coord1,
        )
        producer_state.advance()
        pipe.producer_tail(producer_state)

    if warp == 1:
        consumer_state = make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        pipe.consumer_wait(consumer_state)
        cute.arch.fence_view_async_shared()
        for idx in cutlass.range(lane, d_tile * seq_tile, 32):
            d = idx % d_tile
            seq = idx // d_tile
            dst[seq, d] = s_tile[d, seq].to(cutlass.Float32)
        pipe.consumer_release(consumer_state)


@cute.jit
def _launch_sm120_direct_tma_copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_total: cutlass.Constexpr[int],
    seq_total: cutlass.Constexpr[int],
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
    use_swizzle: cutlass.Constexpr[bool],
):
    gmem_tma = make_sm120_tma_basis_tensor_2d(src, d_total, seq_total)
    smem_layout = make_sm120_direct_tma_smem_layout_2d(
        d_tile,
        seq_tile,
        swizzle=use_swizzle,
    )
    tma_atom, _, _ = make_sm120_direct_tma_load_2d_atom(
        gmem_tma,
        smem_layout,
        (d_tile, seq_tile),
    )
    smem_bytes = d_tile * seq_tile * dtype.width // 8 + 16
    _sm120_direct_tma_copy_kernel(
        tma_atom,
        dst,
        coord0,
        coord1,
        dtype,
        d_tile,
        seq_tile,
        use_swizzle,
    ).launch(grid=[1, 1, 1], block=[64, 1, 1], smem=smem_bytes)


def _make_source(seq_total: int, d_total: int, dtype: torch.dtype) -> torch.Tensor:
    seq = torch.arange(seq_total, device="cuda", dtype=torch.float32)[:, None]
    d = torch.arange(d_total, device="cuda", dtype=torch.float32)[None, :]
    return (seq * 100.0 + d).to(dtype)


def _run_copy_case(torch_dtype, cute_dtype, d_tile, seq_tile, coord0, coord1, use_swizzle=False):
    _require_sm120_direct_tma()
    d_total = 160
    seq_total = 224
    src = _make_source(seq_total, d_total, torch_dtype)
    dst = torch.empty((seq_tile, d_tile), device="cuda", dtype=torch.float32)

    runtime_args = (
        from_dlpack(src),
        from_dlpack(dst),
        cutlass.Int32(coord0),
        cutlass.Int32(coord1),
    )
    compile_args = (
        *runtime_args,
        cute_dtype,
        d_total,
        seq_total,
        d_tile,
        seq_tile,
        use_swizzle,
    )
    cute.compile(_launch_sm120_direct_tma_copy, *compile_args)(*runtime_args)
    torch.cuda.synchronize()

    expected = src[coord1 : coord1 + seq_tile, coord0 : coord0 + d_tile].to(torch.float32)
    torch.testing.assert_close(dst, expected, atol=0, rtol=0)


def test_sm120_direct_tma_feature_probe():
    if has_sm120_direct_tma_2d():
        assert_sm120_direct_tma_2d(require_device=False)


@pytest.mark.parametrize("torch_dtype,cute_dtype", [(torch.float16, cutlass.Float16)])
@pytest.mark.parametrize("d_tile,seq_tile", [(64, 64), (96, 64), (128, 128)])
@pytest.mark.parametrize("coord0,coord1", [(0, 0), (16, 32)])
def test_sm120_direct_tma_2d_copy(torch_dtype, cute_dtype, d_tile, seq_tile, coord0, coord1):
    _run_copy_case(torch_dtype, cute_dtype, d_tile, seq_tile, coord0, coord1)


@pytest.mark.parametrize("d_tile,seq_tile", [(64, 64), (128, 128)])
@pytest.mark.parametrize("coord0,coord1", [(0, 0), (16, 32)])
def test_sm120_direct_tma_2d_copy_bf16(d_tile, seq_tile, coord0, coord1):
    _run_copy_case(torch.bfloat16, cutlass.BFloat16, d_tile, seq_tile, coord0, coord1)


@pytest.mark.parametrize(
    "torch_dtype,cute_dtype",
    [(torch.float16, cutlass.Float16), (torch.bfloat16, cutlass.BFloat16)],
)
def test_sm120_direct_tma_2d_copy_sw128(torch_dtype, cute_dtype):
    _run_copy_case(
        torch_dtype,
        cute_dtype,
        d_tile=64,
        seq_tile=64,
        coord0=16,
        coord1=32,
        use_swizzle=True,
    )
