"""Distributed correctness tests for fused GEMM + reduce epilogues (RS / AR).

Marked `dist`: deselected by default, run with `pytest --dist-only` (never
under pytest-xdist — one pytest process launches multiple torchrun ranks).
"""

import argparse
import os
import subprocess
import sys

import pytest
import torch


WORLD_SIZES = [
    4
]  # 4 is good default, 2 ranks can miss numerical/ordering bugs, 8 occupies full node
# (m, n, k, l, ab_dtype, d_dtype); dtypes per quack gemm test convention (bf16
# baseline, fp16 inputs, fp32 out), each non-bf16 pair on its own hard shape
CASES = [
    (4096, 4096, 4096, 1, "bfloat16", "bfloat16"),  # baseline
    (488, 1024, 1024, 2, "bfloat16", "bfloat16"),  # m_per_rank < cta_m: tiny grid, batched
    (528, 4104, 736, 3, "bfloat16", "bfloat16"),  # partial M/N tiles + K residue, batched
    (1032, 2056, 928, 2, "float16", "float16"),  # partial M/N tiles + K residue, batched
    (520, 1028, 672, 3, "bfloat16", "float32"),  # fp32 vec=4 path
]
# Split-K x epi_reduce composition: the finalizing split folds the f32 workspace,
# commits the skip-epi-ops partial to symmetric D, and signals. SERIAL is bitwise
# deterministic; PARALLEL commits in arrival order (ref check only).
SPLIT_K_CASES = [
    # (m, n, k, l, ab_dtype, d_dtype, split_k, split_k_mode)
    (4096, 4096, 4096, 1, "bfloat16", "bfloat16", 2, "serial"),
    (528, 4104, 736, 3, "bfloat16", "bfloat16", 2, "serial"),  # partial M/N tiles + K residue
    (4096, 4096, 4096, 1, "bfloat16", "bfloat16", 2, "parallel"),
]
pytestmark = pytest.mark.dist


def _run_gemm_epi_reduce(
    m,
    n,
    k,
    l=1,
    ab_dtype="bfloat16",
    d_dtype="bfloat16",
    epi_reduce_mode="reduce_scatter",
    split_k=1,
    split_k_mode="serial",
    ws_dtype=None,
):
    import torch.distributed as dist

    import cutlass
    import cutlass.torch as cutlass_torch

    from quack.cute_dsl_utils import get_device_capacity
    from quack.dist_utils import init_distributed, clean_distributed
    from quack.distributed.gemm_epi_reduce import make_epi_reduce_args
    from quack.gemm import gemm
    from quack.gemm_config import SplitKMode

    init_distributed()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size > 1, "launch with torchrun --nproc_per_node > 1"
    sm_major = get_device_capacity(torch.device("cuda"))[0]
    assert sm_major in (10, 11), f"GEMM+RS requires SM100/SM110; got SM{sm_major}x"
    assert m % world_size == 0, f"m ({m}) must be divisible by world_size ({world_size})"
    assert k % world_size == 0, f"k ({k}) must be divisible by world_size ({world_size})"

    dtype_map = {
        "bfloat16": cutlass.BFloat16,
        "float16": cutlass.Float16,
        "float32": cutlass.Float32,
    }
    blockscaled = ab_dtype == "mxfp8"
    d_dtype = dtype_map[d_dtype]
    tile_m, tile_n = 256, 256
    cluster_m, cluster_n = 2, 1
    vec = 128 // d_dtype.width
    assert n % vec == 0, f"n ({n}) must be divisible by {vec} (16B multimem vectors)"

    k_local = k // world_size
    m_per_rank = m // world_size
    torch_d = cutlass_torch.dtype(d_dtype)

    torch.manual_seed(1111 + rank)
    bs_kwargs = {}
    if blockscaled:
        from quack.blockscaled.utils import (
            create_blockscaled_operand_quantized,
            scale_view_for_kernel,
        )

        sf_vec = 32
        assert k_local % (4 * sf_vec) == 0, (
            f"k_local ({k_local}) must be divisible by {4 * sf_vec} (SF tile granularity)"
        )
        # Each rank quantizes its own K-shard; a_ref/b_ref are the f32 dequants.
        a_ref_mkl, aq, a_sc = create_blockscaled_operand_quantized(l, m, k_local, False, sf_vec)
        b_ref_mkl, bq, b_sc = create_blockscaled_operand_quantized(l, n, k_local, False, sf_vec)
        a_gpu, b_gpu = aq.permute(2, 0, 1), bq.permute(2, 0, 1)
        bs_kwargs = dict(
            SFA=scale_view_for_kernel(a_sc, m, k_local // sf_vec, l),
            SFB=scale_view_for_kernel(b_sc, n, k_local // sf_vec, l),
            bs_format_a="mxfp8_e4m3",
            bs_format_b="mxfp8_e4m3",
        )
    else:
        ab_dtype = dtype_map[ab_dtype]
        ab_vec = 128 // ab_dtype.width
        assert k_local % ab_vec == 0, (
            f"k_local ({k_local}) must be divisible by {ab_vec} (16B TMA alignment on A/B)"
        )
        torch_ab = cutlass_torch.dtype(ab_dtype)
        a_torch_cpu = (
            torch.empty(l, m, k_local, dtype=torch.float32)
            .normal_()
            .mul_(1.0 / (k**0.5))
            .to(torch_ab)
        )
        b_torch_cpu = (
            torch.empty(l, n, k_local, dtype=torch.float32)
            .normal_()
            .mul_(1.0 / (k**0.5))
            .to(torch_ab)
        )
        a_gpu = a_torch_cpu.cuda()
        b_gpu = b_torch_cpu.cuda()
    # D output + padded partials workspace + semaphores; torch handles inside the
    # args are the sole refs keeping the symmetric allocs alive.
    torch_ws = cutlass_torch.dtype(dtype_map[ws_dtype]) if ws_dtype is not None else None
    d_arg, epi_reduce_args = make_epi_reduce_args(
        epi_reduce_mode, torch_d, m, n, l, tile_m, tile_n, cluster_m, world_size, ws_dtype=torch_ws
    )
    tf_torch = epi_reduce_args.tile_flags

    sk_mode = SplitKMode.SERIAL if split_k_mode == "serial" else SplitKMode.PARALLEL
    launch = lambda: gemm(
        a_gpu,
        b_gpu,
        d_arg,
        None,  # C
        None,  # tile_count_semaphore
        tile_m,
        tile_n,
        cluster_m,
        cluster_n,
        split_k=split_k,
        split_k_mode=sk_mode,
        epi_reduce_mode=epi_reduce_mode,
        epi_reduce_args=epi_reduce_args,
        **bs_kwargs,
    )
    # d_arg is the output itself: RS a plain (l, m/world, n) slab; AR the full
    # reduced D on every rank after the multicast broadcast.
    out = d_arg

    # PARALLEL split-K commits partials in arrival order (f32 adds are order-
    # sensitive), so relaunches are not bit-identical: launches still run, but the
    # bit-exact asserts below only hold for split_k == 1 and SERIAL.
    bitwise_repeatable = split_k == 1 or sk_mode == SplitKMode.SERIAL

    # r2r: relaunch reuses tile_flags/sync_barrier in place (stale-flag bugs
    # are invisible to a single launch); identical inputs must be bit-identical.
    runs = []
    for _ in range(2):
        launch()
        torch.cuda.synchronize()
        dist.barrier()
        runs.append(out.clone())
    if bitwise_repeatable:
        assert torch.equal(runs[0], runs[1]), "r2r: relaunch not bit-identical"

    # Mutation loop: negate A each iter (bit-exact in every dtype) so stale/raced
    # values differ from expected; rotate a forced-skew straggler; no barrier between iters.
    # For fp8 operands negation is a sign-bit flip (e8m0 scales are unsigned).
    neg_a = (lambda: aq.view(torch.uint8).bitwise_xor_(0x80)) if blockscaled else a_gpu.neg_
    expected = runs[1].clone()
    for it in range(4 * world_size):
        neg_a()
        expected.neg_()
        if it % world_size == rank:
            torch.cuda._sleep(50_000_000)  # ~25 ms straggler: dwarfs one launch
        launch()
        torch.cuda.synchronize()
        if bitwise_repeatable:
            assert torch.equal(out, expected), f"mutation loop iter {it}: stale or raced value"
    dist.barrier()

    # Flags self-reset in-kernel: the relaunch loops above are the reuse test;
    # no int32 wrap is reachable.
    dist.barrier()

    # quack convention: fp32 ref is ground truth; kernel error < 2x same-dtype ref error.
    # Blockscaled ground truth is the f32 dequant of the quantized operands.
    if blockscaled:
        a_ref = a_ref_mkl.permute(2, 0, 1)
        b_ref = b_ref_mkl.permute(2, 0, 1)
    else:
        a_ref = a_torch_cpu.contiguous().cuda()
        b_ref = b_torch_cpu.contiguous().cuda()

    def epilogue_ref(dtype):
        d_full = torch.bmm(a_ref.to(dtype), b_ref.to(dtype).mT)
        if epi_reduce_mode == "reduce_scatter":
            d_red = torch.empty(l, m_per_rank, n, dtype=dtype, device="cuda")
            for i in range(l):
                dist.reduce_scatter_tensor(d_red[i], d_full[i])
            return d_red.float()
        dist.all_reduce(d_full)
        return d_full.float()

    d_ref = epilogue_ref(torch.float32)
    d_pt = epilogue_ref(torch_d)
    torch.cuda.synchronize()

    d_err = (out.float() - d_ref).abs().max()
    d_base = (d_pt.to(torch_d).float() - d_ref).abs().max()
    if rank == 0:
        print(f"D err {d_err:.3e} base {d_base:.3e}")
    assert d_err < 2 * d_base + 1e-5, f"D err {d_err}, baseline {d_base}"
    if rank == 0:
        print("Ref check PASSED")

    dist.barrier()
    clean_distributed()


def _launch_test(world_size, m, n, k, l, ab_dtype, d_dtype, mode, split_k=1, split_k_mode="serial"):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(world_size),
        __file__,
        *["--m", str(m), "--n", str(n), "--k", str(k), "--l", str(l)],
        *["--ab_dtype", ab_dtype, "--d_dtype", d_dtype, "--mode", mode],
        *["--split_k", str(split_k), "--split_k_mode", split_k_mode],
    ]
    result = subprocess.run(
        cmd,
        env=env,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout
    assert "Ref check PASSED" in result.stdout, result.stdout


@pytest.mark.parametrize("world_size", WORLD_SIZES, ids=lambda w: f"world{w}")
@pytest.mark.parametrize("mode", ["reduce_scatter", "all_reduce"], ids=["rs", "ar"])
@pytest.mark.parametrize("m,n,k,l,ab_dtype,d_dtype", CASES)
def test_gemm_epi_reduce(m, n, k, l, ab_dtype, d_dtype, mode, world_size):
    _launch_test(world_size, m, n, k, l, ab_dtype, d_dtype, mode)


@pytest.mark.parametrize("world_size", WORLD_SIZES, ids=lambda w: f"world{w}")
@pytest.mark.parametrize("mode", ["reduce_scatter", "all_reduce"], ids=["rs", "ar"])
@pytest.mark.parametrize("m,n,k,l,ab_dtype,d_dtype,split_k,split_k_mode", SPLIT_K_CASES)
def test_gemm_epi_reduce_split_k(
    m, n, k, l, ab_dtype, d_dtype, split_k, split_k_mode, mode, world_size
):
    _launch_test(world_size, m, n, k, l, ab_dtype, d_dtype, mode, split_k, split_k_mode)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--l", type=int, default=1)
    dtypes = ["bfloat16", "float16", "float32"]
    parser.add_argument("--ab_dtype", choices=dtypes + ["mxfp8"], default="bfloat16")
    parser.add_argument("--d_dtype", choices=dtypes, default="bfloat16")
    parser.add_argument(
        "--mode", choices=["reduce_scatter", "all_reduce"], default="reduce_scatter"
    )
    parser.add_argument("--split_k", type=int, default=1)
    parser.add_argument("--split_k_mode", choices=["serial", "parallel"], default="serial")
    parser.add_argument("--ws_dtype", choices=dtypes, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _run_gemm_epi_reduce(
        args.m,
        args.n,
        args.k,
        args.l,
        args.ab_dtype,
        args.d_dtype,
        args.mode,
        args.split_k,
        args.split_k_mode,
        ws_dtype=args.ws_dtype,
    )
