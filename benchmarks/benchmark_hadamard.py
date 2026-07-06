import argparse
import math
import os

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch
from triton.testing import Benchmark, do_bench, perf_report

from quack.bench.bench_utils import run_and_print
from quack.transform.hadamard import hadamard_transform, hadamard_transform_ref

try:
    from fast_hadamard_transform import hadamard_transform as fast_hadamard_transform
except ImportError:
    fast_hadamard_transform = None


# Hadamard kernel currently supports N <= 32768
MN_PAIRS = [
    (8192, 256),
    (8192, 512),
    (8192, 1024),
    (8192, 2048),
    (8192, 4096),
    (8192, 8192),
    (8192, 16384),
    (8192, 32768),
]

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _parse_int_token(text: str) -> int:
    token = text.strip().lower()
    multiplier = 1024 if token.endswith("k") else 1
    if multiplier != 1:
        token = token[:-1]
    try:
        return int(token) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers, optionally using k suffix"
        ) from exc


def _parse_int_csv(text: str) -> list[int]:
    return [_parse_int_token(item) for item in text.split(",") if item.strip()]


def _result(num_bytes: int, ms: float) -> dict:
    gbps = num_bytes / (ms / 1000) / 1e9
    return {"ms": round(ms, 4), "GB/s": round(gbps)}


def _providers(include_torch_ref: bool):
    providers = [("quack", "quack")]
    if fast_hadamard_transform is not None:
        providers.append(("fast_hadamard", "fast-hadamard"))
    providers.append(("torch_clone", "torch.clone (lower bound)"))
    if include_torch_ref:
        providers.append(("torch_ref", "torch FWHT ref"))
    return providers


def make_benchmark(dtype_name: str, include_torch_ref: bool, x_vals=None) -> Benchmark:
    line_vals, line_names = zip(*_providers(include_torch_ref))
    return Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals if x_vals is not None else MN_PAIRS,
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name=f"hadamard-{dtype_name}",
        args={"dtype_name": dtype_name},
        xlabel="(M, N)",
        ylabel="GB/s",
    )


def hadamard_runner(M, N, provider, dtype_name):
    dtype = DTYPE_MAP[dtype_name]
    scale = 1.0 / math.sqrt(1 << (N - 1).bit_length())

    x = torch.randn(M, N, device="cuda", dtype=dtype)

    if provider == "quack":
        fn = lambda: hadamard_transform(x, scale=scale)
    elif provider == "fast_hadamard":
        fn = lambda: fast_hadamard_transform(x, scale)
    elif provider == "torch_clone":
        fn = lambda: torch.clone(x)
    elif provider == "torch_ref":
        fn = lambda: hadamard_transform_ref(x, scale=scale)
    else:
        raise ValueError(provider)

    ms = do_bench(fn, warmup=10, rep=100)
    # I/O: read x + write y
    nbytes = 2 * x.numel() * x.dtype.itemsize
    return _result(nbytes, ms)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hadamard transform")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    parser.add_argument(
        "--M",
        type=_parse_int_csv,
        default=None,
        help="M value(s), comma-separated. Defaults to 8192 when --N is provided.",
    )
    parser.add_argument(
        "--N",
        type=_parse_int_csv,
        default=None,
        help="N value(s), comma-separated, e.g. --N 1024,2048,4096.",
    )
    parser.add_argument(
        "--include_torch_ref",
        action="store_true",
        help="Also bench the slow torch FWHT reference",
    )
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    if args.M is not None and args.N is None:
        parser.error("--M requires --N")
    if args.N is None:
        x_vals = None
    else:
        m_vals = args.M if args.M is not None else [8192]
        x_vals = [(M, N) for M in m_vals for N in args.N]

    torch.manual_seed(0)

    bench = perf_report(make_benchmark(args.dtype, args.include_torch_ref, x_vals))(hadamard_runner)
    run_and_print(bench, save_path=args.save_path)


if __name__ == "__main__":
    main()
