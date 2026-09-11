import argparse
import math
import os

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch
from triton.testing import Benchmark, do_bench, perf_report

from quack.bench.bench_utils import run_and_print
from quack.rotary import apply_rotary_emb


# Keep the flattened row count (B * S * H)
# fixed while increasing D, then shrink rows for large D to keep the
# benchmark tensor size bounded.  B/S/H stay explicit in the table so
# sequence/head-layout effects remain visible.
SHAPE_CASES = {
    "rows1m-d32-full": (8, 4096, 32, 32, 32),
    "rows1m-d64-full": (8, 4096, 32, 64, 64),
    "rows1m-d96-full": (8, 4096, 32, 96, 96),
    "rows1m-d128-full": (8, 4096, 32, 128, 128),
    "rows1m-d128-half": (8, 4096, 32, 128, 64),
    "rows512k-d256-full": (4, 4096, 32, 256, 256),
    "rows512k-d256-half": (4, 4096, 32, 256, 128),
    "rows256k-d512-full": (2, 4096, 32, 512, 512),
    "rows256k-d512-half": (2, 4096, 32, 512, 256),
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def generate_cos_sin(seqlen: int, rotary_dim: int, device: str, dtype: torch.dtype):
    assert rotary_dim % 2 == 0
    angle = torch.rand(seqlen, rotary_dim // 2, device=device) * 2 * math.pi
    cos = torch.cos(angle).to(dtype=dtype)
    sin = torch.sin(angle).to(dtype=dtype)
    return cos, sin


def generate_offsets(use_offsets: bool, batch: int, seqlen: int, device: str):
    if not use_offsets:
        return None
    return torch.randint(0, seqlen + 1, (batch,), dtype=torch.int32, device=device)


def rotate_half(x: torch.Tensor, interleaved: bool):
    if not interleaved:
        x0, x1 = x.chunk(2, dim=-1)
        return torch.cat((-x1, x0), dim=-1)
    x0, x1 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x1, x0), dim=-1).reshape_as(x)


def rotary_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: torch.Tensor | None,
    interleaved: bool,
):
    seqlen = x.shape[1]
    rotary_dim = cos.shape[-1] * 2
    if seqlen_offsets is None:
        cos = cos[:seqlen]
        sin = sin[:seqlen]
    else:
        arange = torch.arange(seqlen, device=x.device).view(1, seqlen)
        idx = seqlen_offsets.view(-1, 1) + arange
        cos = cos[idx]
        sin = sin[idx]

    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    if not interleaved:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    else:
        cos = cos.unsqueeze(-1).expand(*cos.shape, 2).reshape(*cos.shape[:-1], rotary_dim)
        sin = sin.unsqueeze(-1).expand(*sin.shape, 2).reshape(*sin.shape[:-1], rotary_dim)
    if cos.dim() == 3:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    x_ro = x[..., :rotary_dim]
    out_ro = x_ro * cos + rotate_half(x_ro, interleaved) * sin
    return torch.cat((out_ro, x[..., rotary_dim:]), dim=-1)


def rotary_ref_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: torch.Tensor | None,
    interleaved: bool,
):
    rotary_dim = cos.shape[-1] * 2
    x[..., :rotary_dim].copy_(
        rotary_ref(x[..., :rotary_dim], cos, sin, seqlen_offsets, interleaved)
    )
    return x


def _providers(include_torch_eager: bool):
    providers = [("quack", "quack"), ("torch_compile", "torch.compile")]
    if include_torch_eager:
        providers.append(("torch_eager", "torch eager"))
    return providers


def _result(num_bytes: int, ms: float) -> dict:
    gbps = num_bytes / (ms / 1000) / 1e9
    return {"ms": round(ms, 4), "GB/s": round(gbps)}


def _logical_bytes(
    batch: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    rotary_dim: int,
    dtype: torch.dtype,
    cossin_dtype: torch.dtype,
    inplace: bool,
    use_offsets: bool,
) -> int:
    # HBM-style lower-bound bytes: x read + output write, plus the unique
    # cos/sin table footprint once.  Counting cos/sin per head is an effective
    # reuse metric and can exceed physical HBM bandwidth when the table is hot
    # in cache.
    x_elems = batch * seqlen * nheads * (rotary_dim if inplace else headdim)
    cossin_elems = (2 * seqlen if use_offsets else seqlen) * rotary_dim
    return 2 * x_elems * dtype.itemsize + cossin_elems * cossin_dtype.itemsize


def make_benchmark(
    dtype_name: str,
    cossin_dtype_name: str,
    interleaved: bool,
    use_offsets: bool,
    backward: bool,
    inplace: bool,
    include_torch_eager: bool,
    warmup: int,
    rep: int,
    x_vals=None,
) -> Benchmark:
    line_vals, line_names = zip(*_providers(include_torch_eager))
    direction = "bwd" if backward else "fwd"
    suffix = [
        direction,
        dtype_name,
        f"cossin-{cossin_dtype_name}",
        "interleaved" if interleaved else "contiguous-pair",
        "offsets" if use_offsets else "no-offsets",
    ]
    if inplace:
        suffix.append("inplace")
    return Benchmark(
        x_names=["B", "S", "H", "D", "rotary_dim"],
        x_vals=x_vals if x_vals is not None else list(SHAPE_CASES.values()),
        line_arg="provider",
        line_vals=list(line_vals),
        line_names=list(line_names),
        plot_name="rotary-" + "-".join(suffix),
        args={
            "dtype_name": dtype_name,
            "cossin_dtype_name": cossin_dtype_name,
            "interleaved": interleaved,
            "use_offsets": use_offsets,
            "backward": backward,
            "inplace": inplace,
            "warmup": warmup,
            "rep": rep,
        },
        xlabel="(B, S, H, D, rotary_dim)",
        ylabel="GB/s",
    )


def _make_inputs(B, S, H, D, rotary_dim, dtype_name, cossin_dtype_name, use_offsets):
    dtype = DTYPE_MAP[dtype_name]
    cossin_dtype = DTYPE_MAP[cossin_dtype_name]
    x = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    cos_len = 2 * S if use_offsets else S
    cos, sin = generate_cos_sin(cos_len, rotary_dim, "cuda", cossin_dtype)
    seqlen_offsets = generate_offsets(use_offsets, B, S, "cuda")
    return x, cos, sin, seqlen_offsets


def _check_correctness(
    B,
    S,
    H,
    D,
    rotary_dim,
    dtype_name,
    cossin_dtype_name,
    interleaved,
    use_offsets,
    backward,
    inplace,
):
    x, cos, sin, seqlen_offsets = _make_inputs(
        B, S, H, D, rotary_dim, dtype_name, cossin_dtype_name, use_offsets
    )
    x_ref = x.detach().clone()
    if backward:
        x = x.requires_grad_()
        x_ref = x_ref.requires_grad_()
        out = apply_rotary_emb(x, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved)
        out_ref = rotary_ref(
            x_ref.float(), cos.float(), sin.float(), seqlen_offsets, interleaved
        ).to(dtype=x.dtype)
        grad = torch.randn_like(out)
        (dx,) = torch.autograd.grad(out, x, grad)
        (dx_ref,) = torch.autograd.grad(out_ref, x_ref, grad)
        torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-3)
        torch.testing.assert_close(dx, dx_ref, atol=1e-2, rtol=1e-3)
        return

    out = apply_rotary_emb(x, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved)
    out_ref = rotary_ref(x_ref.float(), cos.float(), sin.float(), seqlen_offsets, interleaved).to(
        dtype=x.dtype
    )
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-3)
    if inplace:
        x_inplace = x.detach().clone()
        out_inplace = apply_rotary_emb(
            x_inplace,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=interleaved,
            inplace=True,
        )
        assert out_inplace.data_ptr() == x_inplace.data_ptr()
        torch.testing.assert_close(out_inplace, out_ref, atol=1e-2, rtol=1e-3)


def rotary_runner(
    B,
    S,
    H,
    D,
    rotary_dim,
    provider,
    dtype_name,
    cossin_dtype_name,
    interleaved,
    use_offsets,
    backward,
    inplace,
    warmup,
    rep,
):
    x, cos, sin, seqlen_offsets = _make_inputs(
        B, S, H, D, rotary_dim, dtype_name, cossin_dtype_name, use_offsets
    )
    bytes_moved = _logical_bytes(
        B, S, H, D, rotary_dim, x.dtype, cos.dtype, inplace and not backward, use_offsets
    )

    if provider == "quack":
        if backward:
            x = x.requires_grad_()
            y = apply_rotary_emb(
                x,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=interleaved,
                inplace=False,
            )
            dy = torch.randn_like(y)
            fn = lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)
            grad_to_none = (x,)
        else:
            fn = lambda: apply_rotary_emb(
                x,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=interleaved,
                inplace=inplace,
            )
            grad_to_none = None
    elif provider == "torch_compile":
        ref_fn = rotary_ref_inplace if inplace else rotary_ref
        compiled = torch.compile(
            lambda x, cos, sin, offsets: ref_fn(x, cos, sin, offsets, interleaved)
        )
        if backward:
            x = x.requires_grad_()
            y = compiled(x, cos, sin, seqlen_offsets)
            dy = torch.randn_like(y)
            fn = torch.compile(
                lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)
            )
            grad_to_none = (x,)
        else:
            fn = lambda: compiled(x, cos, sin, seqlen_offsets)
            grad_to_none = None
    elif provider == "torch_eager":
        if backward:
            x = x.requires_grad_()
            y = rotary_ref(x, cos, sin, seqlen_offsets, interleaved)
            dy = torch.randn_like(y)
            fn = lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)
            grad_to_none = (x,)
        else:
            ref_fn = rotary_ref_inplace if inplace else rotary_ref
            fn = lambda: ref_fn(x, cos, sin, seqlen_offsets, interleaved)
            grad_to_none = None
    else:
        raise ValueError(provider)

    ms = do_bench(fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none)
    return _result(bytes_moved, ms)


def _parse_shape_args(args):
    shape_values = [args.B, args.S, args.H, args.D, args.rotary_dim]
    if any(v is not None for v in shape_values):
        if any(v is None for v in shape_values):
            raise SystemExit("--B, --S, --H, --D, and --rotary_dim must be provided together")
        return [tuple(shape_values)]
    if args.case:
        return [SHAPE_CASES[name] for name in args.case]
    return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark rotary embedding fwd / bwd")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    parser.add_argument("--cossin_dtype", default=None, choices=list(DTYPE_MAP))
    parser.add_argument("--case", action="append", choices=sorted(SHAPE_CASES))
    parser.add_argument("--B", type=int, default=None, help="Batch for a single custom shape")
    parser.add_argument(
        "--S", type=int, default=None, help="Sequence length for a single custom shape"
    )
    parser.add_argument(
        "--H", type=int, default=None, help="Number of heads for a single custom shape"
    )
    parser.add_argument(
        "--D", type=int, default=None, help="Head dimension for a single custom shape"
    )
    parser.add_argument(
        "--rotary_dim", type=int, default=None, help="Rotary dimension for a custom shape"
    )
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--offsets", action="store_true", help="Benchmark tensor seqlen_offsets")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--inplace", action="store_true", help="Only supported for forward benchmarks"
    )
    parser.add_argument("--include_torch_eager", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="Run a correctness check before timing"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    if args.backward and args.inplace:
        parser.error("--inplace is only supported for forward benchmarks")
    cossin_dtype = args.cossin_dtype or args.dtype
    x_vals = _parse_shape_args(args)
    torch.manual_seed(0)
    if args.check:
        for shape in x_vals if x_vals is not None else SHAPE_CASES.values():
            _check_correctness(
                *shape,
                args.dtype,
                cossin_dtype,
                args.interleaved,
                args.offsets,
                args.backward,
                args.inplace,
            )

    torch.manual_seed(0)
    bench = perf_report(
        make_benchmark(
            args.dtype,
            cossin_dtype,
            args.interleaved,
            args.offsets,
            args.backward,
            args.inplace,
            args.include_torch_eager,
            args.warmup,
            args.rep,
            x_vals,
        )
    )(rotary_runner)
    run_and_print(bench, save_path=args.save_path)


if __name__ == "__main__":
    main()
