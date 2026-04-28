#!/usr/bin/env python3
"""Benchmark SM120 direct rank-2 TMA against async and cooperative copies.

The direct path uses the CuTe DSL workaround from CUTLASS PR #3189:
"CuTe DSL: work around SM120 rank-2 TMA cute.copy hang".

The default overlap-fa scenario is intentionally sized above the common TMA
usefulness threshold: TMA usually needs large tiles, roughly >16KB, to amortize
setup/synchronization overhead versus async or cooperative copy paths.
The point of the benchmark is to show whether a producer TMA warp can overlap
large K/V-like tile movement with consumer work while avoiding manual copy loops
on many threads.

Examples:
    python benchmarks/benchmark_sm120_direct_tma.py
    python benchmarks/benchmark_sm120_direct_tma.py --scenario copy
    python benchmarks/benchmark_sm120_direct_tma.py --mode async-copy --d-tile 128 --seq-tile 128
    ncu --profile-from-start off python benchmarks/benchmark_sm120_direct_tma.py --profile
"""

import argparse
import copy
import os
import time

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync, warp as cute_warp
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline.sm90 import PipelineTmaAsync, make_pipeline_state

from quack.sm120_tma_utils import (
    assert_sm120_direct_tma_2d,
    get_sm120_direct_tma_desc_addr,
    make_sm120_direct_tma_load_2d_atom,
    make_sm120_tma_basis_tensor_2d,
    sm120_direct_tma_load_2d,
)
from quack.utils import domain_offset_aligned


COPY_MODES = ("direct-tma", "async-copy", "cooperative-copy")
CONSUMERS = ("mma", "scalar")
PRODUCER_RESOURCES = {
    "direct-tma": "1 elected issuer",
    "async-copy": "1 warp cp.async",
    "cooperative-copy": "1 warp blocking",
}

DTYPES = {
    "float16": (torch.float16, cutlass.Float16),
    "bfloat16": (torch.bfloat16, cutlass.BFloat16),
    "float32": (torch.float32, cutlass.Float32),
}


@cute.jit
def _make_async_tiled_copy(
    dtype: cutlass.Constexpr,
    d_tile: cutlass.Constexpr[int],
    num_copy_threads: cutlass.Constexpr[int],
):
    # Keep this baseline deliberately simple and robust: the minimum cp.async
    # transaction width.  It is still an async G2S baseline, but not a tuned
    # vectorized cp.async pipeline.
    copy_elems = const_expr(max(1, 32 // dtype.width))
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(),
        dtype,
        num_bits_per_copy=copy_elems * dtype.width,
    )
    # The benchmark's logical tile is (d, seq); mode 0 is contiguous.
    # Give each thread a vector along d and let CuTe tile the rest.
    return cute.make_tiled_copy_tv(
        copy_atom,
        cute.make_layout(num_copy_threads),
        cute.make_layout((copy_elems, 1)),
    )


@cute.kernel
def _direct_tma_kernel(
    tma_atom: cute.CopyAtom,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
):
    smem = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((d_tile, seq_tile), stride=(1, d_tile))
    s_tile = smem.allocate_tensor(dtype, smem_layout, byte_alignment=128)
    s_mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(2), byte_alignment=8)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
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
            dst[bidx, seq, d] = s_tile[d, seq]
        pipe.consumer_release(consumer_state)


@cute.jit
def _launch_direct_tma(
    src: cute.Tensor,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_total: cutlass.Constexpr[int],
    seq_total: cutlass.Constexpr[int],
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
):
    gmem_tma = make_sm120_tma_basis_tensor_2d(src, d_total, seq_total)
    smem_layout = cute.make_layout((d_tile, seq_tile), stride=(1, d_tile))
    tma_atom, _, _ = make_sm120_direct_tma_load_2d_atom(
        gmem_tma,
        smem_layout,
        (d_tile, seq_tile),
    )
    smem_bytes = d_tile * seq_tile * dtype.width // 8 + 16
    _direct_tma_kernel(
        tma_atom,
        dst,
        coord0,
        coord1,
        dtype,
        d_tile,
        seq_tile,
    ).launch(grid=[num_ctas, 1, 1], block=[64, 1, 1], smem=smem_bytes)


@cute.kernel
def _async_copy_kernel(
    src_tma: cute.Tensor,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
):
    smem = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((d_tile, seq_tile), stride=(1, d_tile))
    s_tile = smem.allocate_tensor(dtype, smem_layout, byte_alignment=128)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32

    g_tile = cute.local_tile(
        domain_offset_aligned((coord0, coord1), src_tma),
        (d_tile, seq_tile),
        (0, 0),
    )
    tiled_copy = _make_async_tiled_copy(dtype, d_tile, 64)
    thr_copy = tiled_copy.get_slice(tidx)
    tG = thr_copy.partition_S(g_tile)
    tS = thr_copy.partition_D(s_tile)
    cute.copy(thr_copy, tG, tS)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    if warp == 1:
        for idx in cutlass.range(lane, d_tile * seq_tile, 32):
            d = idx % d_tile
            seq = idx // d_tile
            dst[bidx, seq, d] = s_tile[d, seq]


@cute.jit
def _launch_async_copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_total: cutlass.Constexpr[int],
    seq_total: cutlass.Constexpr[int],
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
):
    gmem_tma = make_sm120_tma_basis_tensor_2d(src, d_total, seq_total)
    smem_bytes = d_tile * seq_tile * dtype.width // 8
    _async_copy_kernel(
        gmem_tma,
        dst,
        coord0,
        coord1,
        dtype,
        d_tile,
        seq_tile,
    ).launch(grid=[num_ctas, 1, 1], block=[64, 1, 1], smem=smem_bytes)


@cute.kernel
def _cooperative_copy_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
):
    smem = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((d_tile, seq_tile), stride=(1, d_tile))
    s_tile = smem.allocate_tensor(dtype, smem_layout, byte_alignment=128)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32

    for idx in cutlass.range(tidx, d_tile * seq_tile, 64):
        d = idx % d_tile
        seq = idx // d_tile
        s_tile[d, seq] = src[coord1 + seq, coord0 + d]

    cute.arch.barrier()

    if warp == 1:
        for idx in cutlass.range(lane, d_tile * seq_tile, 32):
            d = idx % d_tile
            seq = idx // d_tile
            dst[bidx, seq, d] = s_tile[d, seq]


@cute.jit
def _launch_cooperative_copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
):
    smem_bytes = d_tile * seq_tile * dtype.width // 8
    _cooperative_copy_kernel(
        src,
        dst,
        coord0,
        coord1,
        dtype,
        d_tile,
        seq_tile,
    ).launch(grid=[num_ctas, 1, 1], block=[64, 1, 1], smem=smem_bytes)


@cute.kernel
def _direct_tma_overlap_fa_kernel(
    tma_atom_k: cute.CopyAtom,
    tma_atom_v: cute.CopyAtom,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
    num_kv_tiles: cutlass.Constexpr[int],
    num_stages: cutlass.Constexpr[int],
    load_v: cutlass.Constexpr[bool],
    compute_iters: cutlass.Constexpr[int],
    consumer_mma: cutlass.Constexpr[bool],
):
    smem = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout(
        (d_tile, seq_tile, num_stages),
        stride=(1, d_tile, d_tile * seq_tile),
    )
    s_k = smem.allocate_tensor(dtype, smem_layout, byte_alignment=128)
    if const_expr(load_v):
        s_v = smem.allocate_tensor(dtype, smem_layout, byte_alignment=128)
    if const_expr(consumer_mma):
        s_q = smem.allocate_tensor(
            dtype,
            cute.make_layout((16, 16), stride=(16, 1)),
            byte_alignment=128,
        )
    s_mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(num_stages * 2), byte_alignment=8)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    desc_k = get_sm120_direct_tma_desc_addr(tma_atom_k)
    if const_expr(load_v):
        desc_v = get_sm120_direct_tma_desc_addr(tma_atom_v)
    bytes_per_tile = d_tile * seq_tile * dtype.width // 8
    tx_count = bytes_per_tile * (2 if const_expr(load_v) else 1)
    pipe = PipelineTmaAsync.create(
        barrier_storage=s_mbar.iterator,
        num_stages=num_stages,
        producer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1),
        consumer_group=cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1),
        tx_count=tx_count,
        defer_sync=False,
    )

    if warp == 0:
        producer_state = make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, num_stages)
        for tile in cutlass.range_constexpr(num_kv_tiles):
            stage = producer_state.index
            seq_origin = coord1 + tile * seq_tile
            pipe.producer_acquire(producer_state)
            sm120_direct_tma_load_2d(
                s_k[None, None, stage].iterator,
                desc_k,
                pipe.producer_get_barrier(producer_state),
                coord0,
                seq_origin,
            )
            if const_expr(load_v):
                sm120_direct_tma_load_2d(
                    s_v[None, None, stage].iterator,
                    desc_v,
                    pipe.producer_get_barrier(producer_state),
                    coord0,
                    seq_origin,
                )
            producer_state.advance()
        pipe.producer_tail(producer_state)

    if const_expr(consumer_mma):
        if warp == 1:
            for idx in cutlass.range(lane, 16 * 16, 32):
                m = idx // 16
                k = idx % 16
                s_q[m, k] = dtype(1.0)
            cute.arch.sync_warp()

            mma_op = cute_warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)))
            thr_mma = tiled_mma.get_slice(lane)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(s_q))
            tSrB = cute.make_rmem_tensor(thr_mma.partition_shape_B((8, 16)), dtype)
            acc_mma = cute.make_rmem_tensor(thr_mma.partition_shape_C((16, 8)), cutlass.Float32)
            smem_copy_atom_A = cute.make_copy_atom(
                cute_warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                dtype,
            )
            smem_copy_atom_B = cute.make_copy_atom(
                cute_warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
            smem_thr_copy_A = smem_tiled_copy_A.get_slice(lane)
            smem_thr_copy_B = smem_tiled_copy_B.get_slice(lane)
            tSsQ = smem_thr_copy_A.partition_S(s_q)
            tSrQ_copy_view = smem_thr_copy_A.retile(tSrQ)
            cute.copy(smem_tiled_copy_A, tSsQ, tSrQ_copy_view)

            consumer_state = make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, num_stages
            )
            acc = cutlass.Float32(0.0)
            for tile in cutlass.range_constexpr(num_kv_tiles):
                pipe.consumer_wait(consumer_state)
                cute.arch.fence_view_async_shared()
                stage = consumer_state.index
                s_k_stage = cute.make_tensor(
                    s_k[None, None, stage].iterator,
                    cute.make_layout((seq_tile, d_tile), stride=(d_tile, 1)),
                )
                if const_expr(load_v):
                    s_v_stage = cute.make_tensor(
                        s_v[None, None, stage].iterator,
                        cute.make_layout((seq_tile, d_tile), stride=(d_tile, 1)),
                    )
                for _ in cutlass.range_constexpr(compute_iters):
                    for n_tile in cutlass.range_constexpr(seq_tile // 8):
                        for k_tile in cutlass.range_constexpr(d_tile // 16):
                            acc_mma.fill(0.0)
                            s_b = cute.local_tile(s_k_stage, (8, 16), (n_tile, k_tile))
                            tSsB = smem_thr_copy_B.partition_S(s_b)
                            tSrB_copy_view = smem_thr_copy_B.retile(tSrB)
                            cute.copy(smem_tiled_copy_B, tSsB, tSrB_copy_view)
                            cute.gemm(tiled_mma, acc_mma, tSrQ, tSrB, acc_mma)
                            acc += acc_mma[0]
                            if const_expr(load_v):
                                acc_mma.fill(0.0)
                                s_b_v = cute.local_tile(s_v_stage, (8, 16), (n_tile, k_tile))
                                tSsB_v = smem_thr_copy_B.partition_S(s_b_v)
                                cute.copy(smem_tiled_copy_B, tSsB_v, tSrB_copy_view)
                                cute.gemm(tiled_mma, acc_mma, tSrQ, tSrB, acc_mma)
                                acc += acc_mma[0]
                pipe.consumer_release(consumer_state)
                consumer_state.advance()
            dst[bidx, lane] = acc
    elif warp == 1:
        consumer_state = make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, num_stages)
        acc = cutlass.Float32(0.0)
        for tile in cutlass.range_constexpr(num_kv_tiles):
            pipe.consumer_wait(consumer_state)
            cute.arch.fence_view_async_shared()
            stage = consumer_state.index
            for _ in cutlass.range_constexpr(compute_iters):
                for idx in cutlass.range(lane, d_tile * seq_tile, 32):
                    d = idx % d_tile
                    seq = idx // d_tile
                    kval = s_k[d, seq, stage].to(cutlass.Float32)
                    acc += kval * cutlass.Float32(0.5)
                    if const_expr(load_v):
                        vval = s_v[d, seq, stage].to(cutlass.Float32)
                        acc += vval * cutlass.Float32(0.25)
            pipe.consumer_release(consumer_state)
            consumer_state.advance()
        dst[bidx, lane] = acc


@cute.jit
def _launch_direct_tma_overlap_fa(
    src_k: cute.Tensor,
    src_v: cute.Tensor,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_total: cutlass.Constexpr[int],
    seq_total: cutlass.Constexpr[int],
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    num_kv_tiles: cutlass.Constexpr[int],
    num_stages: cutlass.Constexpr[int],
    load_v: cutlass.Constexpr[bool],
    compute_iters: cutlass.Constexpr[int],
    consumer_mma: cutlass.Constexpr[bool],
):
    gmem_k_tma = make_sm120_tma_basis_tensor_2d(src_k, d_total, seq_total)
    smem_layout = cute.make_layout((d_tile, seq_tile), stride=(1, d_tile))
    tma_atom_k, _, _ = make_sm120_direct_tma_load_2d_atom(
        gmem_k_tma,
        smem_layout,
        (d_tile, seq_tile),
    )
    if const_expr(load_v):
        gmem_v_tma = make_sm120_tma_basis_tensor_2d(src_v, d_total, seq_total)
        tma_atom_v, _, _ = make_sm120_direct_tma_load_2d_atom(
            gmem_v_tma,
            smem_layout,
            (d_tile, seq_tile),
        )
    else:
        tma_atom_v = tma_atom_k
    operands = 2 if const_expr(load_v) else 1
    mma_smem_bytes = 16 * 16 * dtype.width // 8 if const_expr(consumer_mma) else 0
    smem_bytes = operands * num_stages * d_tile * seq_tile * dtype.width // 8 + mma_smem_bytes + 64
    _direct_tma_overlap_fa_kernel(
        tma_atom_k,
        tma_atom_v,
        dst,
        coord0,
        coord1,
        dtype,
        d_tile,
        seq_tile,
        num_kv_tiles,
        num_stages,
        load_v,
        compute_iters,
        consumer_mma,
    ).launch(grid=[num_ctas, 1, 1], block=[64, 1, 1], smem=smem_bytes)


@cute.kernel
def _async_copy_overlap_fa_kernel(
    src_k_tma: cute.Tensor,
    src_v_tma: cute.Tensor,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
    num_kv_tiles: cutlass.Constexpr[int],
    load_v: cutlass.Constexpr[bool],
    compute_iters: cutlass.Constexpr[int],
    consumer_mma: cutlass.Constexpr[bool],
):
    smem = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((d_tile, seq_tile), stride=(1, d_tile))
    s_k = smem.allocate_tensor(dtype, smem_layout, byte_alignment=128)
    if const_expr(load_v):
        s_v = smem.allocate_tensor(dtype, smem_layout, byte_alignment=128)
    if const_expr(consumer_mma):
        s_q = smem.allocate_tensor(
            dtype,
            cute.make_layout((16, 16), stride=(16, 1)),
            byte_alignment=128,
        )

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    acc = cutlass.Float32(0.0)

    if const_expr(consumer_mma):
        if warp == 1:
            for idx in cutlass.range(lane, 16 * 16, 32):
                m = idx // 16
                k = idx % 16
                s_q[m, k] = dtype(1.0)
            cute.arch.sync_warp()

        mma_op = cute_warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)))
        thr_mma = tiled_mma.get_slice(lane)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(s_q))
        tSrB = cute.make_rmem_tensor(thr_mma.partition_shape_B((8, 16)), dtype)
        acc_mma = cute.make_rmem_tensor(thr_mma.partition_shape_C((16, 8)), cutlass.Float32)
        smem_copy_atom_A = cute.make_copy_atom(
            cute_warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            dtype,
        )
        smem_copy_atom_B = cute.make_copy_atom(
            cute_warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            dtype,
        )
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(lane)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(lane)
        tSsQ = smem_thr_copy_A.partition_S(s_q)
        tSrQ_copy_view = smem_thr_copy_A.retile(tSrQ)
        if warp == 1:
            cute.copy(smem_tiled_copy_A, tSsQ, tSrQ_copy_view)

    tiled_copy = _make_async_tiled_copy(dtype, d_tile, 32)
    thr_copy = tiled_copy.get_slice(lane)
    tSk = thr_copy.partition_D(s_k)
    if const_expr(load_v):
        tSv = thr_copy.partition_D(s_v)

    for tile in cutlass.range_constexpr(num_kv_tiles):
        seq_origin = coord1 + tile * seq_tile
        if warp == 0:
            g_k = cute.local_tile(
                domain_offset_aligned((coord0, seq_origin), src_k_tma),
                (d_tile, seq_tile),
                (0, 0),
            )
            tGk = thr_copy.partition_S(g_k)
            cute.copy(thr_copy, tGk, tSk)
            if const_expr(load_v):
                g_v = cute.local_tile(
                    domain_offset_aligned((coord0, seq_origin), src_v_tma),
                    (d_tile, seq_tile),
                    (0, 0),
                )
                tGv = thr_copy.partition_S(g_v)
                cute.copy(thr_copy, tGv, tSv)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        if const_expr(consumer_mma):
            if warp == 1:
                s_k_stage = cute.make_tensor(
                    s_k.iterator,
                    cute.make_layout((seq_tile, d_tile), stride=(d_tile, 1)),
                )
                if const_expr(load_v):
                    s_v_stage = cute.make_tensor(
                        s_v.iterator,
                        cute.make_layout((seq_tile, d_tile), stride=(d_tile, 1)),
                    )
                for _ in cutlass.range_constexpr(compute_iters):
                    for n_tile in cutlass.range_constexpr(seq_tile // 8):
                        for k_tile in cutlass.range_constexpr(d_tile // 16):
                            acc_mma.fill(0.0)
                            s_b = cute.local_tile(s_k_stage, (8, 16), (n_tile, k_tile))
                            tSsB = smem_thr_copy_B.partition_S(s_b)
                            tSrB_copy_view = smem_thr_copy_B.retile(tSrB)
                            cute.copy(smem_tiled_copy_B, tSsB, tSrB_copy_view)
                            cute.gemm(tiled_mma, acc_mma, tSrQ, tSrB, acc_mma)
                            acc += acc_mma[0]
                            if const_expr(load_v):
                                acc_mma.fill(0.0)
                                s_b_v = cute.local_tile(s_v_stage, (8, 16), (n_tile, k_tile))
                                tSsB_v = smem_thr_copy_B.partition_S(s_b_v)
                                cute.copy(smem_tiled_copy_B, tSsB_v, tSrB_copy_view)
                                cute.gemm(tiled_mma, acc_mma, tSrQ, tSrB, acc_mma)
                                acc += acc_mma[0]
        elif warp == 1:
            for _ in cutlass.range_constexpr(compute_iters):
                for idx in cutlass.range(lane, d_tile * seq_tile, 32):
                    d = idx % d_tile
                    seq = idx // d_tile
                    kval = s_k[d, seq].to(cutlass.Float32)
                    acc += kval * cutlass.Float32(0.5)
                    if const_expr(load_v):
                        vval = s_v[d, seq].to(cutlass.Float32)
                        acc += vval * cutlass.Float32(0.25)
        cute.arch.barrier()

    if warp == 1:
        dst[bidx, lane] = acc


@cute.jit
def _launch_async_copy_overlap_fa(
    src_k: cute.Tensor,
    src_v: cute.Tensor,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_total: cutlass.Constexpr[int],
    seq_total: cutlass.Constexpr[int],
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    num_kv_tiles: cutlass.Constexpr[int],
    load_v: cutlass.Constexpr[bool],
    compute_iters: cutlass.Constexpr[int],
    consumer_mma: cutlass.Constexpr[bool],
):
    gmem_k_tma = make_sm120_tma_basis_tensor_2d(src_k, d_total, seq_total)
    if const_expr(load_v):
        gmem_v_tma = make_sm120_tma_basis_tensor_2d(src_v, d_total, seq_total)
    else:
        gmem_v_tma = gmem_k_tma
    operands = 2 if const_expr(load_v) else 1
    mma_smem_bytes = 16 * 16 * dtype.width // 8 if const_expr(consumer_mma) else 0
    smem_bytes = operands * d_tile * seq_tile * dtype.width // 8 + mma_smem_bytes
    _async_copy_overlap_fa_kernel(
        gmem_k_tma,
        gmem_v_tma,
        dst,
        coord0,
        coord1,
        dtype,
        d_tile,
        seq_tile,
        num_kv_tiles,
        load_v,
        compute_iters,
        consumer_mma,
    ).launch(grid=[num_ctas, 1, 1], block=[64, 1, 1], smem=smem_bytes)


@cute.kernel
def _cooperative_copy_overlap_fa_kernel(
    src_k: cute.Tensor,
    src_v: cute.Tensor,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
    num_kv_tiles: cutlass.Constexpr[int],
    load_v: cutlass.Constexpr[bool],
    compute_iters: cutlass.Constexpr[int],
    consumer_mma: cutlass.Constexpr[bool],
):
    smem = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((d_tile, seq_tile), stride=(1, d_tile))
    s_k = smem.allocate_tensor(dtype, smem_layout, byte_alignment=128)
    if const_expr(load_v):
        s_v = smem.allocate_tensor(dtype, smem_layout, byte_alignment=128)
    if const_expr(consumer_mma):
        s_q = smem.allocate_tensor(
            dtype,
            cute.make_layout((16, 16), stride=(16, 1)),
            byte_alignment=128,
        )

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    acc = cutlass.Float32(0.0)

    if const_expr(consumer_mma):
        if warp == 1:
            for idx in cutlass.range(lane, 16 * 16, 32):
                m = idx // 16
                k = idx % 16
                s_q[m, k] = dtype(1.0)
            cute.arch.sync_warp()

        mma_op = cute_warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)))
        thr_mma = tiled_mma.get_slice(lane)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(s_q))
        tSrB = cute.make_rmem_tensor(thr_mma.partition_shape_B((8, 16)), dtype)
        acc_mma = cute.make_rmem_tensor(thr_mma.partition_shape_C((16, 8)), cutlass.Float32)
        smem_copy_atom_A = cute.make_copy_atom(
            cute_warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            dtype,
        )
        smem_copy_atom_B = cute.make_copy_atom(
            cute_warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            dtype,
        )
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(lane)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(lane)
        tSsQ = smem_thr_copy_A.partition_S(s_q)
        tSrQ_copy_view = smem_thr_copy_A.retile(tSrQ)
        if warp == 1:
            cute.copy(smem_tiled_copy_A, tSsQ, tSrQ_copy_view)

    for tile in cutlass.range_constexpr(num_kv_tiles):
        seq_origin = coord1 + tile * seq_tile
        if warp == 0:
            for idx in cutlass.range(lane, d_tile * seq_tile, 32):
                d = idx % d_tile
                seq = idx // d_tile
                s_k[d, seq] = src_k[seq_origin + seq, coord0 + d]
                if const_expr(load_v):
                    s_v[d, seq] = src_v[seq_origin + seq, coord0 + d]
        cute.arch.barrier()

        if const_expr(consumer_mma):
            if warp == 1:
                s_k_stage = cute.make_tensor(
                    s_k.iterator,
                    cute.make_layout((seq_tile, d_tile), stride=(d_tile, 1)),
                )
                if const_expr(load_v):
                    s_v_stage = cute.make_tensor(
                        s_v.iterator,
                        cute.make_layout((seq_tile, d_tile), stride=(d_tile, 1)),
                    )
                for _ in cutlass.range_constexpr(compute_iters):
                    for n_tile in cutlass.range_constexpr(seq_tile // 8):
                        for k_tile in cutlass.range_constexpr(d_tile // 16):
                            acc_mma.fill(0.0)
                            s_b = cute.local_tile(s_k_stage, (8, 16), (n_tile, k_tile))
                            tSsB = smem_thr_copy_B.partition_S(s_b)
                            tSrB_copy_view = smem_thr_copy_B.retile(tSrB)
                            cute.copy(smem_tiled_copy_B, tSsB, tSrB_copy_view)
                            cute.gemm(tiled_mma, acc_mma, tSrQ, tSrB, acc_mma)
                            acc += acc_mma[0]
                            if const_expr(load_v):
                                acc_mma.fill(0.0)
                                s_b_v = cute.local_tile(s_v_stage, (8, 16), (n_tile, k_tile))
                                tSsB_v = smem_thr_copy_B.partition_S(s_b_v)
                                cute.copy(smem_tiled_copy_B, tSsB_v, tSrB_copy_view)
                                cute.gemm(tiled_mma, acc_mma, tSrQ, tSrB, acc_mma)
                                acc += acc_mma[0]
        elif warp == 1:
            for _ in cutlass.range_constexpr(compute_iters):
                for idx in cutlass.range(lane, d_tile * seq_tile, 32):
                    d = idx % d_tile
                    seq = idx // d_tile
                    kval = s_k[d, seq].to(cutlass.Float32)
                    acc += kval * cutlass.Float32(0.5)
                    if const_expr(load_v):
                        vval = s_v[d, seq].to(cutlass.Float32)
                        acc += vval * cutlass.Float32(0.25)
        cute.arch.barrier()

    if warp == 1:
        dst[bidx, lane] = acc


@cute.jit
def _launch_cooperative_copy_overlap_fa(
    src_k: cute.Tensor,
    src_v: cute.Tensor,
    dst: cute.Tensor,
    coord0: cutlass.Int32,
    coord1: cutlass.Int32,
    dtype: cutlass.Constexpr,
    d_tile: cutlass.Constexpr[int],
    seq_tile: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    num_kv_tiles: cutlass.Constexpr[int],
    load_v: cutlass.Constexpr[bool],
    compute_iters: cutlass.Constexpr[int],
    consumer_mma: cutlass.Constexpr[bool],
):
    operands = 2 if const_expr(load_v) else 1
    mma_smem_bytes = 16 * 16 * dtype.width // 8 if const_expr(consumer_mma) else 0
    smem_bytes = operands * d_tile * seq_tile * dtype.width // 8 + mma_smem_bytes
    _cooperative_copy_overlap_fa_kernel(
        src_k,
        src_v,
        dst,
        coord0,
        coord1,
        dtype,
        d_tile,
        seq_tile,
        num_kv_tiles,
        load_v,
        compute_iters,
        consumer_mma,
    ).launch(grid=[num_ctas, 1, 1], block=[64, 1, 1], smem=smem_bytes)


def benchmark(fn, repeats: int, warmup: int, stat: str) -> tuple[float, list[float]]:
    torch.cuda.synchronize()
    time.sleep(0.2)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    ordered = sorted(samples)
    if stat == "min":
        return ordered[0], samples
    if stat == "second-min":
        return ordered[1] if len(ordered) > 1 else ordered[0], samples
    if stat == "median":
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
        return median, samples
    raise ValueError(f"Unsupported stat: {stat}")


def profile_once(fn, warmup_launches: int) -> None:
    for _ in range(warmup_launches):
        fn()
    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        fn()
        torch.cuda.synchronize()
    finally:
        cudart.cudaProfilerStop()


def make_source(seq_total: int, d_total: int, dtype: torch.dtype) -> torch.Tensor:
    seq = torch.arange(seq_total, device="cuda", dtype=torch.float32)[:, None]
    d = torch.arange(d_total, device="cuda", dtype=torch.float32)[None, :]
    return (seq * 100.0 + d).to(dtype)


def compile_copy_runner(args, mode: str, src: torch.Tensor, dst: torch.Tensor, cute_dtype):
    runtime_args = (
        from_dlpack(src, assumed_align=16),
        from_dlpack(dst),
        cutlass.Int32(args.coord0),
        cutlass.Int32(args.coord1),
    )
    if mode == "direct-tma":
        compile_args = (
            *runtime_args,
            cute_dtype,
            args.d_total,
            args.seq_total,
            args.d_tile,
            args.seq_tile,
            args.num_ctas,
        )
        compiled = cute.compile(_launch_direct_tma, *compile_args)
    elif mode == "async-copy":
        compile_args = (
            *runtime_args,
            cute_dtype,
            args.d_total,
            args.seq_total,
            args.d_tile,
            args.seq_tile,
            args.num_ctas,
        )
        compiled = cute.compile(_launch_async_copy, *compile_args)
    elif mode == "cooperative-copy":
        compile_args = (
            *runtime_args,
            cute_dtype,
            args.d_tile,
            args.seq_tile,
            args.num_ctas,
        )
        compiled = cute.compile(_launch_cooperative_copy, *compile_args)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return lambda: compiled(*runtime_args)


def compile_overlap_runner(
    args,
    mode: str,
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    dst: torch.Tensor,
    cute_dtype,
):
    consumer_mma = args.consumer == "mma"
    runtime_args = (
        from_dlpack(src_k, assumed_align=16),
        from_dlpack(src_v, assumed_align=16),
        from_dlpack(dst),
        cutlass.Int32(args.coord0),
        cutlass.Int32(args.coord1),
    )
    if mode == "direct-tma":
        compile_args = (
            *runtime_args,
            cute_dtype,
            args.d_total,
            args.seq_total,
            args.d_tile,
            args.seq_tile,
            args.num_ctas,
            args.num_kv_tiles,
            args.num_stages,
            args.load_kv,
            args.compute_iters,
            consumer_mma,
        )
        compiled = cute.compile(_launch_direct_tma_overlap_fa, *compile_args)
    elif mode == "async-copy":
        compile_args = (
            *runtime_args,
            cute_dtype,
            args.d_total,
            args.seq_total,
            args.d_tile,
            args.seq_tile,
            args.num_ctas,
            args.num_kv_tiles,
            args.load_kv,
            args.compute_iters,
            consumer_mma,
        )
        compiled = cute.compile(_launch_async_copy_overlap_fa, *compile_args)
    elif mode == "cooperative-copy":
        compile_args = (
            *runtime_args,
            cute_dtype,
            args.d_tile,
            args.seq_tile,
            args.num_ctas,
            args.num_kv_tiles,
            args.load_kv,
            args.compute_iters,
            consumer_mma,
        )
        compiled = cute.compile(_launch_cooperative_copy_overlap_fa, *compile_args)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return lambda: compiled(*runtime_args)


def validate_copy_output(src: torch.Tensor, dst: torch.Tensor, args) -> None:
    expected = src[
        args.coord1 : args.coord1 + args.seq_tile,
        args.coord0 : args.coord0 + args.d_tile,
    ]
    for cta in (0, args.num_ctas - 1):
        torch.testing.assert_close(dst[cta], expected, atol=0, rtol=0)


def run_copy_mode(args, mode: str, src: torch.Tensor, cute_dtype):
    dst = torch.empty(
        (args.num_ctas, args.seq_tile, args.d_tile),
        device="cuda",
        dtype=src.dtype,
    )
    fn = compile_copy_runner(args, mode, src, dst, cute_dtype)
    fn()
    torch.cuda.synchronize()
    if args.check:
        validate_copy_output(src, dst, args)
    if args.profile:
        profile_once(fn, args.profile_warmup)
        return None
    ms, samples = benchmark(fn, args.repeats, args.warmup, args.stat)
    bytes_per_launch = 2 * args.num_ctas * args.d_tile * args.seq_tile * src.element_size()
    gbps = bytes_per_launch / (ms / 1000.0) / 1e9
    return ms, gbps, samples


def run_overlap_mode(args, mode: str, src_k: torch.Tensor, src_v: torch.Tensor, cute_dtype):
    dst = torch.empty((args.num_ctas, 32), device="cuda", dtype=torch.float32)
    fn = compile_overlap_runner(args, mode, src_k, src_v, dst, cute_dtype)
    fn()
    torch.cuda.synchronize()
    if args.profile:
        profile_once(fn, args.profile_warmup)
        return None
    ms, samples = benchmark(fn, args.repeats, args.warmup, args.stat)
    operands = 2 if args.load_kv else 1
    bytes_per_launch = (
        operands
        * args.num_ctas
        * args.num_kv_tiles
        * args.d_tile
        * args.seq_tile
        * src_k.element_size()
    )
    gbps = bytes_per_launch / (ms / 1000.0) / 1e9
    return ms, gbps, samples, dst


def _stage_bytes(args, element_size: int) -> int:
    operands = 2 if args.scenario == "overlap-fa" and args.load_kv else 1
    return operands * args.d_tile * args.seq_tile * element_size


def _extra_smem_bytes(args, element_size: int) -> int:
    return 16 * 16 * element_size if args.scenario == "overlap-fa" and args.consumer == "mma" else 0


def _validate_args(args, element_size: int) -> None:
    if args.coord0 + args.d_tile > args.d_total:
        raise ValueError("coord0 + d_tile must fit within d_total")
    required_seq = args.coord1 + args.seq_tile
    if args.scenario == "overlap-fa":
        required_seq = args.coord1 + args.num_kv_tiles * args.seq_tile
    if required_seq > args.seq_total:
        raise ValueError("requested sequence tiles must fit within seq_total")
    if args.scenario == "overlap-fa":
        if args.num_stages < 1:
            raise ValueError("num_stages must be at least 1")
        if args.consumer == "mma":
            if args.dtype == "float32":
                raise ValueError("--consumer mma supports float16 and bfloat16 only")
            if args.d_tile % 16 != 0:
                raise ValueError("--consumer mma requires d_tile to be a multiple of 16")
            if args.seq_tile % 8 != 0:
                raise ValueError("--consumer mma requires seq_tile to be a multiple of 8")
        smem_bytes = (
            args.num_stages * _stage_bytes(args, element_size)
            + _extra_smem_bytes(args, element_size)
            + 64
        )
        if smem_bytes > 99 * 1024:
            raise ValueError(
                "overlap-fa shared memory exceeds the SM120 99KB limit: "
                f"{smem_bytes} bytes. Reduce --num-stages, --seq-tile, or --head-dim."
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SM120 direct TMA 2D tile loads")
    parser.add_argument("--scenario", choices=["copy", "overlap-fa"], default="overlap-fa")
    parser.add_argument("--mode", choices=[*COPY_MODES, "all"], default="all")
    parser.add_argument("--consumer", choices=CONSUMERS, default="mma")
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="bfloat16")
    parser.add_argument("--d-total", type=int, default=160)
    parser.add_argument("--seq-total", type=int, default=1024)
    parser.add_argument("--d-tile", "--head-dim", dest="d_tile", type=int, default=128)
    parser.add_argument("--seq-tile", type=int, default=64)
    parser.add_argument("--coord0", type=int, default=16)
    parser.add_argument("--coord1", type=int, default=32)
    parser.add_argument("--num-ctas", type=int, default=256)
    parser.add_argument("--num-kv-tiles", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--load-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute-iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--stat", choices=["min", "second-min", "median"], default="second-min")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup", type=int, default=5)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run FA-like rows that vary tile bytes and compute intensity",
    )
    return parser.parse_args()


def _run_case(args, torch_dtype, cute_dtype, name: str | None = None):
    _validate_args(args, torch.empty((), dtype=torch_dtype).element_size())
    src_k = make_source(args.seq_total, args.d_total, torch_dtype)
    src_v = (make_source(args.seq_total, args.d_total, torch_dtype) + 7).contiguous()
    modes = list(COPY_MODES) if args.mode == "all" else [args.mode]
    stage_bytes = _stage_bytes(args, src_k.element_size())
    smem_bytes = (
        stage_bytes
        if args.scenario == "copy"
        else args.num_stages * stage_bytes + _extra_smem_bytes(args, src_k.element_size()) + 64
    )

    if name is not None:
        print(f"\n## {name}")
    print("SM120 direct TMA staging benchmark")
    print(
        f"scenario={args.scenario} dtype={args.dtype} tile=({args.d_tile}, {args.seq_tile}) "
        f"origin=({args.coord0}, {args.coord1}) num_ctas={args.num_ctas}"
    )
    if args.scenario == "overlap-fa":
        print(f"consumer={args.consumer}")
    print(f"load_kv={args.load_kv} stage_bytes={stage_bytes} dynamic_smem_bytes~={smem_bytes}")
    if stage_bytes < 16 * 1024:
        print("note: stage_bytes is below the usual ~16KB TMA usefulness threshold")
    if args.scenario == "overlap-fa":
        print(
            f"num_kv_tiles={args.num_kv_tiles} num_stages={args.num_stages} "
            f"compute_iters={args.compute_iters}"
        )
    if args.profile:
        for mode in modes:
            print(f"Profiling {mode}")
            if args.scenario == "copy":
                run_copy_mode(args, mode, src_k, cute_dtype)
            else:
                run_overlap_mode(args, mode, src_k, src_v, cute_dtype)
        return

    rows = []
    overlap_outputs = {}
    for mode in modes:
        if args.scenario == "copy":
            ms, gbps, samples = run_copy_mode(args, mode, src_k, cute_dtype)
            rows.append((mode, ms, gbps, samples))
        else:
            ms, gbps, samples, dst = run_overlap_mode(args, mode, src_k, src_v, cute_dtype)
            rows.append((mode, ms, gbps, samples))
            overlap_outputs[mode] = dst.clone()

    if args.check and args.scenario == "overlap-fa" and len(overlap_outputs) > 1:
        expected = overlap_outputs[modes[0]]
        for mode in modes[1:]:
            torch.testing.assert_close(overlap_outputs[mode], expected, atol=0, rtol=0)

    print("\n| mode | producer resource | ms | effective GB/s |")
    print("|---|---|---:|---:|")
    for mode, ms, gbps, _ in rows:
        print(f"| {mode} | {PRODUCER_RESOURCES[mode]} | {ms:.4f} | {gbps:.1f} |")
    direct = next((row for row in rows if row[0] == "direct-tma"), None)
    if direct is not None:
        for mode, ms, _, _ in rows:
            if mode != "direct-tma":
                print(f"direct-tma speedup vs {mode}: {ms / direct[1]:.3f}x")
    return rows, stage_bytes


def _sweep_cases(args):
    cases = [
        (
            "below-threshold copy control",
            dict(scenario="copy", d_tile=128, seq_tile=32),
        ),
        (
            "threshold copy control",
            dict(scenario="copy", d_tile=128, seq_tile=64),
        ),
        (
            "scalar FA K/V one-stage light compute",
            dict(
                scenario="overlap-fa",
                consumer="scalar",
                d_tile=128,
                seq_tile=64,
                num_kv_tiles=8,
                num_stages=1,
                compute_iters=1,
            ),
        ),
        (
            "scalar FA K/V two-stage light compute",
            dict(
                scenario="overlap-fa",
                consumer="scalar",
                d_tile=128,
                seq_tile=64,
                num_kv_tiles=8,
                num_stages=2,
                compute_iters=1,
            ),
        ),
        (
            "MMA FA K/V two-stage Tensor Core",
            dict(
                scenario="overlap-fa",
                consumer="mma",
                d_tile=128,
                seq_tile=64,
                num_kv_tiles=8,
                num_stages=2,
                compute_iters=1,
            ),
        ),
        (
            "large MMA FA K/V Tensor Core",
            dict(
                scenario="overlap-fa",
                consumer="mma",
                d_tile=128,
                seq_tile=128,
                num_kv_tiles=4,
                num_stages=1,
                compute_iters=4,
            ),
        ),
    ]
    for name, overrides in cases:
        case = copy.copy(args)
        for key, value in overrides.items():
            setattr(case, key, value)
        yield name, case


def main():
    args = parse_args()
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
    assert_sm120_direct_tma_2d()

    torch_dtype, cute_dtype = DTYPES[args.dtype]
    if args.sweep:
        if args.profile:
            raise ValueError("--sweep and --profile are intentionally separate")
        summary = []
        for name, case in _sweep_cases(args):
            rows, stage_bytes = _run_case(case, torch_dtype, cute_dtype, name=name)
            direct = next(row for row in rows if row[0] == "direct-tma")
            async_copy = next(row for row in rows if row[0] == "async-copy")
            cooperative = next(row for row in rows if row[0] == "cooperative-copy")
            summary.append(
                (
                    name,
                    case.consumer if case.scenario == "overlap-fa" else "copy",
                    stage_bytes / 1024,
                    case.compute_iters if case.scenario == "overlap-fa" else 0,
                    direct[1],
                    async_copy[1],
                    cooperative[1],
                    async_copy[1] / direct[1],
                    cooperative[1] / direct[1],
                )
            )
        print(
            "\n| case | consumer | stage KiB | compute iters | direct ms | async ms | "
            "coop ms | direct/async | direct/coop |"
        )
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in summary:
            (
                name,
                consumer,
                kib,
                compute_iters,
                direct_ms,
                async_ms,
                coop_ms,
                speed_async,
                speed_coop,
            ) = row
            print(
                f"| {name} | {consumer} | {kib:.1f} | {compute_iters} | {direct_ms:.4f} | "
                f"{async_ms:.4f} | {coop_ms:.4f} | {speed_async:.3f}x | {speed_coop:.3f}x |"
            )
        return

    _run_case(args, torch_dtype, cute_dtype)


if __name__ == "__main__":
    main()
