import argparse
import math
import time

import torch
from triton.testing import do_bench

from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm as quack_gemm

"""
Split-K GEMM benchmark (SM100 / B200-B300 only).

Split-K parallelizes the K reduction of a single output tile across `split_k`
work units (turnstile fp32 fixup). It pays off when there are FEW output tiles
(small M, N) but a LARGE K: a plain GEMM launches ~one CTA per output tile and
cannot fill the GPU, while split-K spreads the K reduction over more CTAs. On
shapes with many output tiles the GPU is already saturated and split-K only adds
reduction overhead -- the large square shape in the default sweep shows that
contrast.

Usage:
    python benchmarks/benchmark_gemm_splitk.py
    python benchmarks/benchmark_gemm_splitk.py --shapes "128,128,16384,1;256,256,65536,1"
    python benchmarks/benchmark_gemm_splitk.py --split_k 1,2,3,4,8 --dtype bf16
    python benchmarks/benchmark_gemm_splitk.py --tile_shape_mn 128,128 --skip_ref_check
"""

_TORCH_DTYPE_MAP = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "BFloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16,
    "Float16": torch.float16, "half": torch.float16,
}
# bf16 accumulates more error over large K than fp16's reference comparison.
_TOL = {torch.bfloat16: 3e-2, torch.float16: 1e-2}

# K-heavy small-M/N shapes (split-K should help) + one large square (it should not).
_DEFAULT_SHAPES = [
    (128, 128, 16384, 1),
    (128, 128, 65536, 1),
    (256, 256, 32768, 1),
    (512, 512, 16384, 1),
    (4096, 4096, 4096, 1),
]


def _bench(fn, warmup: int, rep: int) -> float:
    time.sleep(0.3)
    return do_bench(fn, warmup=warmup, rep=rep)


def _parse_shapes(s: str):
    shapes = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = tuple(int(x.strip()) for x in chunk.split(","))
        if len(vals) != 4:
            raise argparse.ArgumentTypeError(f"shape '{chunk}' must have 4 ints (m,n,k,l)")
        shapes.append(vals)
    return shapes


def _parse_ints(s: str):
    return tuple(int(x.strip()) for x in s.split(","))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split-K GEMM benchmark using quack.gemm.gemm()")
    p.add_argument("--shapes", type=_parse_shapes, default=_DEFAULT_SHAPES,
                   help="';'-separated m,n,k,l tuples. Default: a K-heavy sweep + a large square.")
    p.add_argument("--split_k", type=_parse_ints, default=(1, 2, 4, 8, 16, 32),
                   help="Comma-separated split factors to sweep (must include 1 for the baseline).")
    p.add_argument("--split_k_mode", type=str, default="parallel", choices=["parallel", "serial"],
                   help="parallel: per-split workspace slices + separate reduce kernel "
                        "(cuBLAS-style); serial: fused in-kernel turnstile reduction.")
    p.add_argument("--tile_shape_mn", type=_parse_ints, default=(128, 128),
                   help="CTA tile (tile_M,tile_N). split_k>1 requires tile_M != 256.")
    p.add_argument("--cluster_shape_mn", type=_parse_ints, default=(1, 1), help="Cluster (M,N).")
    p.add_argument("--dtype", type=str, default="bf16", help="A/B/D dtype: bf16 or fp16.")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--skip_ref_check", action="store_true")
    args = p.parse_args()
    if args.dtype not in _TORCH_DTYPE_MAP:
        p.error(f"--dtype must be one of {sorted(_TORCH_DTYPE_MAP)}")
    if 1 not in args.split_k:
        p.error("--split_k must include 1 (the no-split baseline)")
    return args


def run_shape(m, n, k, l, *, split_ks, split_k_mode, tile_M, tile_N, cluster_M, cluster_N, dtype,
              warmup, rep, skip_ref_check) -> None:
    device = "cuda"
    torch.manual_seed(0)
    A = torch.randn(l, m, k, dtype=dtype, device=device) / math.sqrt(k)
    B = torch.randn(l, n, k, dtype=dtype, device=device) / math.sqrt(k)
    D = torch.empty(l, m, n, dtype=dtype, device=device)

    flops = 2 * m * n * k * l
    n_tiles = math.ceil(m / tile_M) * math.ceil(n / tile_N) * l
    print(
        f"\n=== m={m} n={n} k={k} l={l}  "
        f"({n_tiles} output tile(s) @ {tile_M}x{tile_N}, {dtype}) ==="
    )

    ref = torch.bmm(A.float(), B.float().mT).to(dtype) if not skip_ref_check else None

    def report(name, t, *, base=None):
        tflops = flops / (t * 1e9)
        rel = f"   ({base / t:5.2f}x vs split_k=1)" if base is not None else ""
        print(f"  {name:<22}: {t:8.4f} ms, {tflops:8.1f} TFLOP/s{rel}")

    t_cublas = _bench(lambda: torch.bmm(A, B.mT), warmup, rep)
    report("cuBLAS (torch.bmm)", t_cublas)

    base_t = None
    best = (None, float("inf"))
    for sk in split_ks:
        def fn(sk=sk):
            quack_gemm(
                A, B, D, C=None, tile_count_semaphore=None,
                tile_M=tile_M, tile_N=tile_N, cluster_M=cluster_M, cluster_N=cluster_N,
                persistent=True, split_k=sk, split_k_mode=split_k_mode,
            )

        if not skip_ref_check:
            fn()
            torch.cuda.synchronize()
            torch.testing.assert_close(D, ref, atol=_TOL[dtype], rtol=1e-3)

        t = _bench(fn, warmup, rep)
        if sk == 1:
            base_t = t
        report(f"quack split_k={sk}", t, base=base_t)
        if t < best[1]:
            best = (sk, t)

    if base_t is not None and best[0] is not None:
        print(
            f"  -> best: split_k={best[0]}  "
            f"{base_t / best[1]:.2f}x vs split_k=1,  {t_cublas / best[1]:.2f}x vs cuBLAS"
        )


def main() -> None:
    args = parse_args()
    cap = get_device_capacity(torch.device("cuda"))
    if cap[0] not in (10, 11):
        raise SystemExit(f"split-K GEMM is SM100 only; got SM{cap[0]}{cap[1]}.")
    if args.split_k != (1,) and args.tile_shape_mn[0] == 256:
        raise SystemExit("split_k>1 does not support 2-CTA tiles (tile_M=256).")

    dtype = _TORCH_DTYPE_MAP[args.dtype]
    tile_M, tile_N = args.tile_shape_mn
    cluster_M, cluster_N = args.cluster_shape_mn
    print(
        f"Split-K GEMM benchmark | tile={tile_M}x{tile_N} cluster={cluster_M}x{cluster_N} "
        f"| split_k={list(args.split_k)} mode={args.split_k_mode} "
        f"| warmup={args.warmup} iters={args.iterations} "
        f"| ref_check={not args.skip_ref_check}"
    )
    for (m, n, k, l) in args.shapes:
        run_shape(
            m, n, k, l, split_ks=args.split_k, split_k_mode=args.split_k_mode,
            tile_M=tile_M, tile_N=tile_N,
            cluster_M=cluster_M, cluster_N=cluster_N, dtype=dtype,
            warmup=args.warmup, rep=args.iterations, skip_ref_check=args.skip_ref_check,
        )
    print("\nPASS")


if __name__ == "__main__":
    main()
