"""Distributed tests for GEMM + reduce epilogues composed with epilogue-mod
classes (quack.epilogues.linear_act_mod / sq_reduce_mod), driven through the
host plan path (EpiMod.gemm with epi_reduce_mode / epi_reduce_args).

The epi-reduce warps keep D linear (act: mAuxOut = act_fn(D); sq: D = D_raw * w
with sq partials from pre-w D_raw), all slab-local. Marked `dist`: deselected by
default, run with `pytest --dist-only` (never under pytest-xdist — one pytest
process launches multiple torchrun ranks).
"""

import argparse
import os
import subprocess
import sys

import pytest
import torch


WORLD_SIZES = [4]
# (m, n, k, l, act); bf16 only — dtype coverage lives in test_gemm_epi_reduce.py
CASES = [
    (4096, 4096, 4096, 1, "relu"),  # baseline
    (528, 4104, 736, 3, "gelu_tanh_approx"),  # partial M/N tiles + K residue, batched
]
SQ_CASES = [
    (4096, 4096, 4096, 1),  # baseline
    (528, 4104, 736, 3),  # partial M/N tiles + K residue, batched; n_tiles=17 colvec stride
]
TORCH_ACT = {
    "relu": torch.nn.functional.relu,
    "gelu_tanh_approx": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
}

pytestmark = pytest.mark.dist


def _dist_setup(m, k, world_size):
    from quack.cute_dsl_utils import get_device_capacity

    sm_major = get_device_capacity(torch.device("cuda"))[0]
    assert sm_major in (10, 11), f"GEMM epi-reduce requires SM100/SM110; got SM{sm_major}x"
    assert m % world_size == 0, f"m ({m}) must be divisible by world_size ({world_size})"
    assert k % world_size == 0, f"k ({k}) must be divisible by world_size ({world_size})"


def _make_comm(m, n, l, m_per_rank, tile_m, tile_n, cta_m):
    """Symmetric D work buffer + flags/barriers/counters; torch handles only."""
    import cutlass

    from quack.dist_utils import create_multicast_tensor, make_barrier_flags
    from quack.epi_reduce import EpiReduceArguments

    d_cpu = torch.empty(l, m, n, dtype=torch.bfloat16).permute(1, 2, 0)
    _, _, d_torch_gpu, d_torch_gpu_mc, d_peer_torch, _ = create_multicast_tensor(
        d_cpu, cutlass.BFloat16, leading_dim=1
    )
    n_tiles = (n + tile_n - 1) // tile_n
    num_tiles = ((m + cta_m - 1) // cta_m) * n_tiles * l
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tf_torch, tf_torch_mc, _, _ = make_barrier_flags(num_tiles)
    sb_torch, sb_torch_mc, _, _ = make_barrier_flags(num_sms)
    slab_tiles_m = (m_per_rank + cta_m - 1) // cta_m
    counters_torch = torch.zeros(slab_tiles_m * n_tiles * l, dtype=torch.int32, device="cuda")
    epi_reduce_args = EpiReduceArguments(
        mD_mc=d_torch_gpu_mc,
        mD_peers=tuple(d_peer_torch),
        tile_flags=tf_torch,
        tile_flags_mc=tf_torch_mc,
        sync_barrier=sb_torch,
        sync_barrier_mc=sb_torch_mc,
        consumer_counters=counters_torch,
    )
    return d_torch_gpu, epi_reduce_args, tf_torch, counters_torch


def _make_ab(m, n, k_local, l, k, rank):
    torch.manual_seed(1111 + rank)
    a_gpu = (
        torch.empty(l, m, k_local, dtype=torch.float32)
        .normal_()
        .mul_(1.0 / (k**0.5))
        .to(torch.bfloat16)
        .cuda()
    )
    b_gpu = (
        torch.empty(l, n, k_local, dtype=torch.float32)
        .normal_()
        .mul_(1.0 / (k**0.5))
        .to(torch.bfloat16)
        .cuda()
    )
    return a_gpu, b_gpu


def _run_gemm_act_reduce(
    m,
    n,
    k,
    l=1,
    act="relu",
    epi_reduce_mode="reduce_scatter",
    has_bias=False,
    has_c=False,
    has_colvec=False,
):
    import torch.distributed as dist

    from quack.dist_utils import torchrun_init_nvshmem, torchrun_finalize_nvshmem
    from quack.epilogues import linear_act_mod

    torchrun_init_nvshmem()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size > 1, "launch with torchrun --nproc_per_node > 1"
    _dist_setup(m, k, world_size)

    tile_m, tile_n = 256, 256
    cluster_m, cluster_n = 2, 1
    cta_m = tile_m // 2
    assert n % 8 == 0, f"n ({n}) must be divisible by 8 (16B multimem vectors)"
    k_local = k // world_size
    m_per_rank = m // world_size

    a_gpu, b_gpu = _make_ab(m, n, k_local, l, k, rank)
    d_torch_gpu, epi_reduce_args, tf_torch, counters_torch = _make_comm(
        m, n, l, m_per_rank, tile_m, tile_n, cta_m
    )
    d_arg = d_torch_gpu.permute(2, 0, 1)  # caller-order (l, m, n) view
    # Aux is slab-local under epi_reduce (epilogue coords are m/TP-shaped).
    aux_gpu = torch.empty(l, m_per_rank, n, dtype=torch.bfloat16, device="cuda")
    # Bias/C add post-reduce on every rank, so they must agree across ranks: common
    # seed, and each rank passes its slab slice of one logical full C.
    torch.manual_seed(2222)
    bias_gpu = torch.randn(l, n, dtype=torch.float32).cuda() if has_bias else None
    c_gpu, c_full_cpu = None, None
    if has_c:
        c_full_cpu = torch.randn(l, m, n, dtype=torch.float32).to(torch.bfloat16)
        c_gpu = c_full_cpu[:, rank * m_per_rank : (rank + 1) * m_per_rank].contiguous().cuda()
    colvec_gpu, colvec_full = None, None
    if has_colvec:
        # ColVecLoad is m-shaped, so it's slab-local under epi_reduce like C.
        assert m_per_rank % 4 == 0, f"colvec rows ({m_per_rank}) need 16B alignment (4 x fp32)"
        colvec_full = torch.randn(l, m, dtype=torch.float32).cuda()
        colvec_gpu = colvec_full[:, rank * m_per_rank : (rank + 1) * m_per_rank].contiguous()

    mod = linear_act_mod(act, gated=False, has_c=has_c, has_rowvec=has_bias, has_colvec=has_colvec)
    epi_args = {"mAuxOut": aux_gpu}
    if has_bias:
        epi_args["mRowVecBroadcast"] = bias_gpu
    if has_colvec:
        epi_args["mColVecBroadcast"] = colvec_gpu
    launch = lambda: mod.gemm(
        a_gpu,
        b_gpu,
        d_arg,
        c_gpu,
        epi_args=epi_args,
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=cluster_m,
        cluster_N=cluster_n,
        is_dynamic_persistent=True,
        epi_reduce_mode=epi_reduce_mode,
        epi_reduce_args=epi_reduce_args,
    )

    # D view: RS owns its m-slab of the (m, n, l) D; AR holds the full reduced D.
    # Aux is slab-local in both modes.
    if epi_reduce_mode == "reduce_scatter":
        out = d_torch_gpu[rank * m_per_rank : (rank + 1) * m_per_rank].permute(2, 0, 1)
    else:
        out = d_torch_gpu.permute(2, 0, 1)
    aux_out = aux_gpu

    # r2r: relaunch reuses flags/counters in place; D and aux must be bit-identical.
    runs, aux_runs = [], []
    for _ in range(2):
        launch()
        torch.cuda.synchronize()
        dist.barrier()
        runs.append(out.clone())
        aux_runs.append(aux_out.clone())
    assert torch.equal(runs[0], runs[1]), "r2r: relaunch D not bit-identical"
    assert torch.equal(aux_runs[0], aux_runs[1]), "r2r: relaunch aux not bit-identical"

    # The mutation loop checks D only: D has a bit-exact sign-flip invariant when
    # all additive inputs are negated. Aux is act(D), so it has no simple bit-exact
    # negation invariant; aux freshness/correctness is covered by r2r, flag-wrap,
    # and the final activation reference check.
    expected = runs[1].clone()
    for it in range(4 * world_size):
        # Negate every input so the whole affine epilogue stays odd end-to-end.
        a_gpu.neg_()
        if has_bias:
            bias_gpu.neg_()
        if has_c:
            c_gpu.neg_()
        if has_colvec:
            colvec_gpu.neg_()
        expected.neg_()
        if it % world_size == rank:
            torch.cuda._sleep(50_000_000)  # ~25 ms straggler: dwarfs one launch
        launch()
        torch.cuda.synchronize()
        assert torch.equal(out, expected), f"mutation loop iter {it}: stale or raced value"
    dist.barrier()

    # Flag wrap (see test_gemm_epi_reduce.py): relaunch across the int32 boundary.
    wrap_seed = torch.iinfo(torch.int32).max - world_size
    tf_torch.fill_(wrap_seed)
    counters_torch.fill_(wrap_seed)
    dist.barrier()
    for it in range(3):
        launch()
        torch.cuda.synchronize()
        assert torch.equal(out, runs[1]), f"flag-wrap launch {it}: D mismatch"
        assert torch.equal(aux_out, aux_runs[1]), f"flag-wrap launch {it}: aux mismatch"
    dist.barrier()

    # quack convention: fp32 ref is ground truth; kernel error < 2x same-dtype ref error.
    def epilogue_ref(dtype):
        d_full = torch.bmm(a_gpu.to(dtype), b_gpu.to(dtype).mT)
        if epi_reduce_mode == "reduce_scatter":
            d_red = torch.empty(l, m_per_rank, n, dtype=dtype, device="cuda")
            for i in range(l):
                dist.reduce_scatter_tensor(d_red[i], d_full[i])
            post = d_red.float()
            if has_c:
                post += c_gpu.float()
        else:
            dist.all_reduce(d_full)
            post = d_full.float()
            if has_c:
                post += c_full_cpu.cuda().float()
        if has_bias:
            post += bias_gpu.unsqueeze(1)
        if has_colvec:
            cv = colvec_gpu if epi_reduce_mode == "reduce_scatter" else colvec_full
            post += cv.unsqueeze(-1)
        slab = (
            post
            if epi_reduce_mode == "reduce_scatter"
            else (post[:, rank * m_per_rank : (rank + 1) * m_per_rank])
        )
        return post, slab

    post_ref, slab_ref = epilogue_ref(torch.float32)
    post_pt, slab_pt = epilogue_ref(torch.bfloat16)
    aux_ref = TORCH_ACT[act](slab_ref)
    aux_pt = TORCH_ACT[act](slab_pt).to(torch.bfloat16)
    torch.cuda.synchronize()

    d_err = (out.float() - post_ref).abs().max()
    d_base = (post_pt.bfloat16().float() - post_ref).abs().max()
    assert d_err < 2 * d_base + 1e-5, f"D err {d_err} vs 2x baseline {d_base}"
    aux_err = (aux_out.float() - aux_ref).abs().max()
    aux_base = (aux_pt.float() - aux_ref).abs().max()
    assert aux_err < 2 * aux_base + 1e-5, f"aux err {aux_err} vs 2x baseline {aux_base}"
    if rank == 0:
        print("Ref check PASSED")

    dist.barrier()
    torchrun_finalize_nvshmem()


def _run_gemm_sq_reduce(m, n, k, l=1, epi_reduce_mode="reduce_scatter", has_c=False):
    import torch.distributed as dist

    from quack.dist_utils import torchrun_init_nvshmem, torchrun_finalize_nvshmem
    from quack.epilogues import sq_reduce_mod

    torchrun_init_nvshmem()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size > 1, "launch with torchrun --nproc_per_node > 1"
    _dist_setup(m, k, world_size)

    tile_m, tile_n = 256, 256
    cluster_m, cluster_n = 2, 1
    cta_m = tile_m // 2
    assert n % 8 == 0, f"n ({n}) must be divisible by 8 (16B multimem vectors)"
    k_local = k // world_size
    m_per_rank = m // world_size
    n_tiles = (n + tile_n - 1) // tile_n

    a_gpu, b_gpu = _make_ab(m, n, k_local, l, k, rank)
    d_torch_gpu, epi_reduce_args, tf_torch, counters_torch = _make_comm(
        m, n, l, m_per_rank, tile_m, tile_n, cta_m
    )
    d_arg = d_torch_gpu.permute(2, 0, 1)
    # Per-N-tile sq partials, slab-local under epi_reduce; norm_weight is common
    # across ranks (multiplies the reduced D on every rank).
    colvec_gpu = torch.zeros(l, m_per_rank, n_tiles, dtype=torch.float32, device="cuda")
    torch.manual_seed(2222)
    w_gpu = torch.randn(l, n, dtype=torch.float32).cuda()
    # Residual C adds pre-sq (D_raw = reduce + C, stats of the post-residual stream);
    # common seed, each rank passes its slab slice of one logical full C.
    c_gpu = None
    if has_c:
        c_full_cpu = torch.randn(l, m, n, dtype=torch.float32).to(torch.bfloat16)
        c_gpu = c_full_cpu[:, rank * m_per_rank : (rank + 1) * m_per_rank].contiguous().cuda()

    mod = sq_reduce_mod(has_c=has_c, has_rowvec=True, has_aux=False)
    epi_args = {"mRowVecBroadcast": w_gpu, "mColVecReduce": colvec_gpu}
    launch = lambda: mod.gemm(
        a_gpu,
        b_gpu,
        d_arg,
        c_gpu,
        epi_args=epi_args,
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=cluster_m,
        cluster_N=cluster_n,
        is_dynamic_persistent=True,
        epi_reduce_mode=epi_reduce_mode,
        epi_reduce_args=epi_reduce_args,
    )

    if epi_reduce_mode == "reduce_scatter":
        out = d_torch_gpu[rank * m_per_rank : (rank + 1) * m_per_rank].permute(2, 0, 1)
    else:
        out = d_torch_gpu.permute(2, 0, 1)

    # r2r: D and the sq partials must both be bit-identical across relaunches.
    runs, sq_runs = [], []
    for _ in range(2):
        launch()
        torch.cuda.synchronize()
        dist.barrier()
        runs.append(out.clone())
        sq_runs.append(colvec_gpu.clone())
    assert torch.equal(runs[0], runs[1]), "r2r: relaunch D not bit-identical"
    assert torch.equal(sq_runs[0], sq_runs[1]), "r2r: relaunch sq partials not bit-identical"

    # Mutation loop: negate additive inputs (A, C) but keep norm_weight (multiplicative:
    # negating it would cancel and D would not flip sign). D_out = (-D_raw)*w flips
    # bit-exactly; the sq partials are negation-invariant so they must equal run 1.
    expected = runs[1].clone()
    for it in range(4 * world_size):
        a_gpu.neg_()
        if has_c:
            c_gpu.neg_()
        expected.neg_()
        if it % world_size == rank:
            torch.cuda._sleep(50_000_000)  # ~25 ms straggler: dwarfs one launch
        launch()
        torch.cuda.synchronize()
        assert torch.equal(out, expected), f"mutation loop iter {it}: stale or raced value"
        assert torch.equal(colvec_gpu, sq_runs[1]), f"mutation loop iter {it}: sq partials"
    dist.barrier()

    # Flag wrap (see test_gemm_epi_reduce.py): relaunch across the int32 boundary.
    wrap_seed = torch.iinfo(torch.int32).max - world_size
    tf_torch.fill_(wrap_seed)
    counters_torch.fill_(wrap_seed)
    dist.barrier()
    for it in range(3):
        launch()
        torch.cuda.synchronize()
        assert torch.equal(out, runs[1]), f"flag-wrap launch {it}: D mismatch"
        assert torch.equal(colvec_gpu, sq_runs[1]), f"flag-wrap launch {it}: sq mismatch"
    dist.barrier()

    # quack convention: fp32 reference = ground truth; kernel error < 2x bf16-impl error.
    def epilogue_ref(dtype):
        d_full = torch.bmm(a_gpu.to(dtype), b_gpu.to(dtype).mT)
        if epi_reduce_mode == "reduce_scatter":
            d_red = torch.empty(l, m_per_rank, n, dtype=dtype, device="cuda")
            for i in range(l):
                dist.reduce_scatter_tensor(d_red[i], d_full[i])
            post = d_red.float()
        else:
            dist.all_reduce(d_full)
            post = d_full.float()
        if has_c:
            if epi_reduce_mode == "reduce_scatter":
                post += c_gpu.float()
            else:
                post += c_full_cpu.cuda().float()
        slab = (
            post
            if epi_reduce_mode == "reduce_scatter"
            else (post[:, rank * m_per_rank : (rank + 1) * m_per_rank])
        )
        # Per-tile sq partials (sq is pre-norm_weight), matching the kernel's slots.
        pad = n_tiles * tile_n - n
        s = torch.nn.functional.pad(slab, (0, pad)) if pad else slab
        sq_tiles = s.view(l, m_per_rank, n_tiles, tile_n).square().sum(-1)
        return post * w_gpu.unsqueeze(1), sq_tiles

    d_ref, sqt_ref = epilogue_ref(torch.float32)
    d_pt, sqt_pt = epilogue_ref(torch.bfloat16)
    torch.cuda.synchronize()

    d_err = (out.float() - d_ref).abs().max()
    d_base = (d_pt.bfloat16().float() - d_ref).abs().max()
    # Assert per-tile (the kernel's actual output): localized defects aren't diluted
    # by a row fold, and fold-accumulated rounding bias isn't asserted (printed only).
    sq_err = (colvec_gpu - sqt_ref).abs().max()
    sq_base = (sqt_pt - sqt_ref).abs().max()
    if rank == 0:
        print(
            f"D err {d_err:.3e} base {d_base:.3e} | sq tile err {sq_err:.3e} "
            f"base {sq_base:.3e} bias {(colvec_gpu - sqt_ref).mean():.2e}"
        )
    assert d_err < 2 * d_base + 1e-5, f"D err {d_err}, baseline {d_base}"
    assert sq_err < 2 * sq_base + 1e-5, f"sq tile err {sq_err}, baseline {sq_base}"
    if rank == 0:
        print("Ref check PASSED")

    dist.barrier()
    torchrun_finalize_nvshmem()


@pytest.mark.parametrize("world_size", WORLD_SIZES, ids=lambda w: f"world{w}")
@pytest.mark.parametrize("extras", ["plain", "bias", "c", "colvec", "all"])
@pytest.mark.parametrize("mode", ["reduce_scatter", "all_reduce"], ids=["rs", "ar"])
@pytest.mark.parametrize("m,n,k,l,act", CASES)
def test_gemm_act_reduce(m, n, k, l, act, mode, extras, world_size):
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
        *["--act", act, "--mode", mode],
        *{
            "plain": [],
            "bias": ["--bias"],
            "c": ["--c_res"],
            "colvec": ["--colvec"],
            "all": ["--bias", "--c_res", "--colvec"],
        }[extras],
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
@pytest.mark.parametrize("extras", ["plain", "c"])
@pytest.mark.parametrize("mode", ["reduce_scatter", "all_reduce"], ids=["rs", "ar"])
@pytest.mark.parametrize("m,n,k,l", SQ_CASES)
def test_gemm_sq_reduce(m, n, k, l, mode, extras, world_size):
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
        *["--variant", "sq", "--mode", mode],
        *({"plain": [], "c": ["--c_res"]}[extras]),
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


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["act", "sq"], default="act")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--act", choices=sorted(TORCH_ACT), default="relu")
    parser.add_argument(
        "--mode", choices=["reduce_scatter", "all_reduce"], default="reduce_scatter"
    )
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--c_res", action="store_true")
    parser.add_argument("--colvec", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.variant == "act":
        _run_gemm_act_reduce(
            args.m,
            args.n,
            args.k,
            args.l,
            args.act,
            args.mode,
            args.bias,
            args.c_res,
            args.colvec,
        )
    else:
        _run_gemm_sq_reduce(args.m, args.n, args.k, args.l, args.mode, args.c_res)
