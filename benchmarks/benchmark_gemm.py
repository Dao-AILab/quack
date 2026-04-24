import argparse
import time

import torch
from triton.testing import do_bench

from quack.gemm import gemm as quack_gemm

"""
GEMM benchmark using quack.gemm.gemm() (dense path) or the SM100 blockscaled
path (MXFP8 / MXFP4 / NVFP4) via --blockscaled.

Usage (dense):
    python benchmarks/benchmark_gemm.py --mnkl 512,7168,2048,256 \
        --tile_shape_mn 256,256 --cluster_shape_mn 2,1 --persistent \
        --varlen_m --gather_A --use_tma_gather --skip_ref_check

Usage (blockscaled MXFP8, with cuBLAS comparison):
    python benchmarks/benchmark_gemm.py --mnkl 4096,4096,4096,1 \
        --blockscaled --ab_dtype Float8E4M3FN --sf_dtype Float8E8M0FNU \
        --sf_vec_size 32 --init quant --compare_cublas

Usage (blockscaled MXFP4):
    python benchmarks/benchmark_gemm.py --mnkl 4096,4096,4096,1 \
        --blockscaled --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU \
        --sf_vec_size 32 --d_dtype Float32

Usage (blockscaled NVFP4):
    python benchmarks/benchmark_gemm.py --mnkl 4096,4096,4096,1 \
        --blockscaled --ab_dtype Float4E2M1FN --sf_dtype Float8E4M3FN \
        --sf_vec_size 16 --d_dtype Float32
"""


def _bench_and_report(name: str, fn, flops: int, warmup: int, rep: int, gbps_bytes: int = 0) -> float:
    """Run do_bench and print a standardized timing + TFLOPS (+ GB/s) line.
    Returns the timing in ms."""
    time.sleep(0.5)
    t = do_bench(fn, warmup=warmup, rep=rep)
    tflops = flops / (t * 1e9)
    if gbps_bytes:
        gbps = gbps_bytes / (t * 1e6)
        print(f"{name}: {t:.3f} ms,  {tflops:7.1f} TFLOP/s,  {gbps:.0f} GB/s")
    else:
        print(f"{name}: {t:.3f} ms,  {tflops:7.1f} TFLOP/s")
    return t


_TORCH_DTYPE_MAP = {
    "BFloat16": torch.bfloat16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "Float16": torch.float16, "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "Float32": torch.float32, "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
}


def _torch_dtype(name: str) -> torch.dtype:
    if name not in _TORCH_DTYPE_MAP:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {name}. Choose from {sorted(_TORCH_DTYPE_MAP.keys())}")
    return _TORCH_DTYPE_MAP[name]


def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GEMM benchmark using quack.gemm.gemm()")

    parser.add_argument(
        "--mnkl", type=parse_comma_separated_ints, default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mn", type=parse_comma_separated_ints, default=(128, 256),
        help="Cta tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn", type=parse_comma_separated_ints,
        choices=[(1, 1), (2, 1), (1, 2), (2, 2)], default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument("--tolerance", type=float, default=3e-02, help="Tolerance for validation")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=30, help="Benchmark iterations")
    parser.add_argument("--persistent", action="store_true", help="Persistent kernel")
    parser.add_argument("--dynamic_persistent", action="store_true", help="Dynamic persistent")
    parser.add_argument("--pingpong", action="store_true", help="Pingpong kernel")
    parser.add_argument("--varlen_m", action="store_true", help="Variable length M dimension")
    parser.add_argument("--varlen_k", action="store_true", help="Variable length K dimension")
    parser.add_argument("--gather_A", action="store_true", help="Gather A")
    parser.add_argument("--use_tma_gather", action="store_true", help="Use TMA gather4 for A")
    parser.add_argument("--max_swizzle_size", type=int, default=8, help="Max swizzle size")
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")
    # Dtype flags. Blockscaled path is selected automatically when --sf_dtype is passed.
    parser.add_argument("--ab_dtype", type=str, default=None,
                        help="A/B input dtype. Default: BFloat16 for dense, auto-detected for "
                             "blockscaled (MXFP8 if sf=E8M0/vec=32, NVFP4 if sf=E4M3FN/vec=16). "
                             "Dense: BFloat16/Float16/Float32. "
                             "Blockscaled: Float8E4M3FN/Float8E5M2/Float4E2M1FN.")
    parser.add_argument("--sf_dtype", type=str, default=None,
                        help="Scale-factor dtype. Setting this or --sf_vec_size enables blockscaled: "
                             "Float8E8M0FNU (MX) or Float8E4M3FN (NVFP4). "
                             "Auto-inferred from --sf_vec_size if omitted.")
    parser.add_argument("--sf_vec_size", type=int, default=None,
                        help="Blockscaled scale vector size (32 for MX, 16 for NVFP4). "
                             "Setting this enables the blockscaled path.")
    parser.add_argument("--d_dtype", type=str, default="BFloat16",
                        help="Output dtype: BFloat16/Float16/Float32 (applies to both dense and blockscaled).")
    parser.add_argument("--c_dtype", type=str, default=None,
                        help="Optional C-tensor dtype (for alpha*A@B + beta*C). Default: no C tensor.")
    parser.add_argument("--a_major", type=str, default=None, choices=["k", "m"],
                        help="A operand major mode. Blockscaled: MXFP8 supports k/m, "
                             "MXFP4/NVFP4 must be k. Dense: varlen_k forces m, others default "
                             "to k if omitted.")
    parser.add_argument("--b_major", type=str, default=None, choices=["k", "n"],
                        help="B operand major mode. Blockscaled: MXFP8 supports k/n, "
                             "MXFP4/NVFP4 must be k. Dense: varlen_k forces n, others default "
                             "to k if omitted.")

    args = parser.parse_args()
    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    if len(args.tile_shape_mn) != 2:
        parser.error("--tile_shape_mn must contain exactly 2 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")
    return args


def _run_blockscaled(args):
    """Blockscaled (MXFP8 / MXFP4 / NVFP4) path using compile_blockscaled_gemm_tvm_ffi."""
    import cutlass
    from quack.blockscaled_gemm_utils import (
        blockscaled_gemm_reference,
        compile_blockscaled_gemm_tvm_ffi,
        create_blockscaled_operand_quantized,
        create_blockscaled_operand_tensor,
        create_blockscaled_varlen_m_operands,
        scale_blocked_for_cublas,
        torch_dtype_for_cutlass,
    )
    from quack.cute_dsl_utils import get_device_capacity
    from quack.gemm_default_epi import GemmDefaultSm100

    sm_major = get_device_capacity(torch.device("cuda"))[0]
    assert sm_major in (10, 11), (
        f"Blockscaled GEMM requires SM100 (B200/B300) or SM110; got SM{sm_major}x. "
        "MXFP8/MXFP4/NVFP4 use tcgen05 UMMA which is SM100+."
    )

    if args.varlen_k or args.gather_A or args.pingpong:
        raise NotImplementedError(
            "blockscaled + varlen_k/gather/pingpong is not wired up yet. "
            "Only --varlen_m is currently supported for blockscaled (MXFP8 only)."
        )

    m, n, k, l = args.mnkl
    mma_tiler_mn = args.tile_shape_mn
    cluster_shape_mn = args.cluster_shape_mn
    # Default sf_vec_size: 32 (MX). Auto-pick sf_dtype / ab_dtype from (sf_vec_size, ab_dtype).
    sf_vec_size = args.sf_vec_size if args.sf_vec_size is not None else 32
    if args.sf_dtype is None:
        if sf_vec_size == 32:
            sf_dtype = cutlass.Float8E8M0FNU             # MXFP8 / MXFP4
        elif sf_vec_size == 16:
            sf_dtype = cutlass.Float8E4M3FN              # NVFP4
        else:
            raise ValueError(f"Cannot auto-pick sf_dtype for sf_vec_size={sf_vec_size}. Pass --sf_dtype.")
    else:
        sf_dtype = cutlass.dtype(args.sf_dtype)
    d_dtype = cutlass.dtype(args.d_dtype)
    # Auto-pick ab_dtype if user didn't set it.
    if args.ab_dtype is None:
        if sf_dtype == cutlass.Float8E8M0FNU and sf_vec_size == 32:
            ab_dtype = cutlass.Float8E4M3FN        # MXFP8 default (user can override -> MXFP4)
        elif sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 16:
            ab_dtype = cutlass.Float4E2M1FN        # NVFP4
        else:
            raise ValueError(
                f"Cannot auto-detect --ab_dtype for sf_dtype={sf_dtype}, sf_vec_size={sf_vec_size}. "
                f"Pass --ab_dtype explicitly."
            )
    else:
        ab_dtype = cutlass.dtype(args.ab_dtype)

    # MXFP4/NVFP4 require K-major for both operands. Only MXFP8 supports m/n-major.
    a_major = args.a_major if args.a_major is not None else "k"
    b_major = args.b_major if args.b_major is not None else "k"
    is_fp4 = ab_dtype == cutlass.Float4E2M1FN
    if is_fp4 and (a_major != "k" or b_major != "k"):
        raise ValueError(
            f"MXFP4/NVFP4 require K-major for both A and B; got a_major={a_major}, b_major={b_major}"
        )
    if not GemmDefaultSm100.can_implement_blockscaled(
        ab_dtype, sf_dtype, sf_vec_size, d_dtype, mma_tiler_mn, cluster_shape_mn,
        m, n, k, l, a_major, b_major, "n",
    ):
        raise TypeError(
            f"Unsupported blockscaled config: ab={ab_dtype}, sf={sf_dtype}, vec={sf_vec_size}, "
            f"d={d_dtype}, tiler={mma_tiler_mn}, cluster={cluster_shape_mn}, "
            f"a_major={a_major}, b_major={b_major}"
        )

    assert k % sf_vec_size == 0, f"k ({k}) must be divisible by sf_vec_size ({sf_vec_size})"
    if args.varlen_m:
        # varlen_m: l is num_experts, m is per-expert m, total_m = m * l.
        # Requires MXFP8 currently.
        assert ab_dtype == cutlass.Float8E4M3FN and sf_dtype == cutlass.Float8E8M0FNU and sf_vec_size == 32, (
            "blockscaled varlen_m currently only supports MXFP8"
        )
        # A must stay k-major in varlen_m (the per-expert padded SF offset
        # targets the M axis); B can be k- or n-major (MXFP8 only).
        assert a_major == "k", (
            f"varlen_m currently requires a_major=k; got a={a_major}"
        )
        total_m = m * l
        a_ref_dq, b_ref_dq, mA, mB, a_sc_contig, b_sc_contig, cu_seqlens_m = (
            create_blockscaled_varlen_m_operands(
                l, m, n, k, sf_vec_size, b_major=b_major,
            )
        )
        # (l, rm, rk, 512) contig scale — consumed directly by the kernel.
        mSFA, mSFB = a_sc_contig, b_sc_contig
        mD = torch.empty(total_m, n, dtype=torch_dtype_for_cutlass(d_dtype), device="cuda")
        runner = compile_blockscaled_gemm_tvm_ffi(
            ab_dtype, sf_dtype, sf_vec_size, d_dtype, mma_tiler_mn, cluster_shape_mn,
            mA, mB, mD, mSFA, mSFB, varlen_m=True,
        )
        def fn():
            runner(mA, mB, mD, mSFA, mSFB, cu_seqlens_m)
    else:
        a_ref, mA, a_sc_contig = create_blockscaled_operand_quantized(
            l, m, k, a_major == "m", sf_vec_size, ab_dtype, sf_dtype,
        )
        b_ref, mB, b_sc_contig = create_blockscaled_operand_quantized(
            l, n, k, b_major == "n", sf_vec_size, ab_dtype, sf_dtype,
        )
        # (l, rm, rk, 512) contig scale — consumed directly by the kernel.
        mSFA, mSFB = a_sc_contig, b_sc_contig
        sfa_ref = torch.ones_like(a_ref)
        sfb_ref = torch.ones_like(b_ref)
        _, mD = create_blockscaled_operand_tensor(l, m, n, False, d_dtype, init="empty")
        runner = compile_blockscaled_gemm_tvm_ffi(
            ab_dtype, sf_dtype, sf_vec_size, d_dtype, mma_tiler_mn, cluster_shape_mn,
            mA, mB, mD, mSFA, mSFB,
        )
        def fn():
            runner(mA, mB, mD, mSFA, mSFB)

    if not args.skip_ref_check:
        fn()
        torch.cuda.synchronize()
        tol = 5e-3 if d_dtype != cutlass.Float32 else 5e-4
        if args.varlen_m:
            # Per-expert matmul reference using dequantized operands
            ref = torch.cat([
                a_ref_dq[cu_seqlens_m[i]:cu_seqlens_m[i + 1]] @ b_ref_dq[i].T
                for i in range(l)
            ])
            torch.testing.assert_close(mD.float(), ref, atol=tol, rtol=1e-3)
        else:
            ref = blockscaled_gemm_reference(a_ref, b_ref, sfa_ref, sfb_ref)
            torch.testing.assert_close(mD.float(), ref, atol=tol, rtol=1e-3)
        print("Ref check PASSED")

    print("Running SM100 Blockscaled GEMM with:")
    print(f"mnkl: {args.mnkl}")
    print(f"tile_shape_mn: {mma_tiler_mn}, cluster_shape_mn: {cluster_shape_mn}")
    print(f"ab_dtype: {ab_dtype}, sf_dtype: {sf_dtype}, sf_vec_size: {sf_vec_size}, d_dtype: {args.d_dtype}")
    print(f"a_major: {a_major}, b_major: {b_major}")

    flops = 2 * m * n * k * l
    timing = _bench_and_report("quack ", fn, flops, args.warmup_iterations, args.iterations)

    if args.varlen_m:
        print("(skipping cuBLAS: varlen_m not supported)")
        return
    if l != 1:
        # F.scaled_mm is 2D-only and torch._scaled_grouped_mm needs a specific layout
        # with per-group swizzled scales we don't build here. Looping F.scaled_mm per
        # batch would be an unfair comparison (hides batching potential), so skip.
        print("(skipping cuBLAS: batched blockscaled mm not supported via a single call)")
        return
    if a_major != "k" or b_major != "k":
        # F.scaled_mm requires A (M,K) row-major and B (K,N) col-major —
        # i.e. both operands K-contiguous. Skip for m/n-major to avoid an
        # apples-vs-oranges copy+transpose.
        print(f"(skipping cuBLAS: F.scaled_mm needs a_major=k, b_major=k; got a={a_major}, b={b_major})")
        return
    from torch.nn.functional import scaled_mm, ScalingType, SwizzleType
    scaling_recipe_map = {32: ScalingType.BlockWise1x32, 16: ScalingType.BlockWise1x16}
    if sf_vec_size not in scaling_recipe_map:
        print(f"(skipping cuBLAS: unsupported sf_vec_size={sf_vec_size})")
        return
    recipe = scaling_recipe_map[sf_vec_size]
    a_cub = mA[:, :, 0].contiguous()
    b_cub = mB[:, :, 0].contiguous()
    a_sc_cub = scale_blocked_for_cublas(a_sc_contig, m, k // sf_vec_size, 0)
    b_sc_cub = scale_blocked_for_cublas(b_sc_contig, n, k // sf_vec_size, 0)
    out_dtype_t = _torch_dtype(args.d_dtype) if args.d_dtype != "Float32" else torch.bfloat16

    def fn_cublas():
        return scaled_mm(
            a_cub, b_cub.t(),
            a_sc_cub, recipe, b_sc_cub, recipe,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4, swizzle_b=SwizzleType.SWIZZLE_32_4_4,
            output_dtype=out_dtype_t,
        )

    if not args.skip_ref_check:
        out_cublas = fn_cublas()
        torch.cuda.synchronize()
        err = (mD.squeeze(-1).float() - out_cublas.float()).abs().max().item()
        same_dtype = (mD.dtype == out_cublas.dtype)
        exact = same_dtype and torch.equal(mD.squeeze(-1), out_cublas)
        print(f"quack vs cuBLAS: max_abs_err={err:.3e}  bit_exact={exact}")

    t_cublas = _bench_and_report("cuBLAS", fn_cublas, flops, args.warmup_iterations, args.iterations)
    print(f"  (quack speedup vs cuBLAS: {t_cublas/timing:.2f}x)")


def run(args):
    if args.sf_dtype is not None or args.sf_vec_size is not None:
        return _run_blockscaled(args)
    m, n, k, l = args.mnkl
    tile_M, tile_N = args.tile_shape_mn
    cluster_M, cluster_N = args.cluster_shape_mn
    persistent = args.persistent or args.dynamic_persistent
    varlen_m, varlen_k, gather_A = args.varlen_m, args.varlen_k, args.gather_A
    warmup, repeats = args.warmup_iterations, args.iterations
    tolerance = args.tolerance
    ab_dtype = _torch_dtype(args.ab_dtype) if args.ab_dtype is not None else torch.bfloat16
    d_dtype = _torch_dtype(args.d_dtype)

    from quack.cute_dsl_utils import get_device_capacity
    device_capacity = get_device_capacity(torch.device("cuda"))
    if device_capacity[0] in [10, 11]:
        persistent = True

    # a_major / b_major control the memory order. Defaults: varlen_k -> m/n
    # (kernel requirement), everything else -> k.
    if varlen_k:
        a_major = args.a_major if args.a_major is not None else "m"
        b_major = args.b_major if args.b_major is not None else "n"
        assert a_major == "m" and b_major == "n", (
            f"dense varlen_k requires a_major=m, b_major=n; got a={a_major}, b={b_major}"
        )
    else:
        a_major = args.a_major if args.a_major is not None else "k"
        b_major = args.b_major if args.b_major is not None else "k"
        if varlen_m:
            assert a_major == "k", (
                f"dense varlen_m requires a_major=k; got a={a_major}"
            )

    print("Running Dense GEMM with:")
    print(f"mnkl: {args.mnkl}")
    print(f"Tile Shape: {args.tile_shape_mn}, Cluster Shape: {args.cluster_shape_mn}")
    print(f"a_major: {a_major}, b_major: {b_major}")
    print(f"Use TMA gather: {args.use_tma_gather}")
    print(f"Warmup iterations: {warmup}")
    print(f"Iterations: {repeats}")
    print(f"Skip reference checking: {args.skip_ref_check}")

    torch.manual_seed(1111)
    device = "cuda"

    # ── Tensor creation ───────────────────────────────────────────────────────
    # quack.gemm.gemm() conventions:
    #   A: (l, m, k) or (total_m, k) if varlen_m
    #   B: (l, n, k)
    #   D: (l, m, n) or (total_m, n) if varlen_m — n-major
    cu_seqlens_m, cu_seqlens_k, A_idx = None, None, None
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=device) if args.dynamic_persistent else None
    )

    def _make_a_non_varlen(l_, m_, k_, major):
        """(l, m, k) with requested major. k-major: contig; m-major: transposed."""
        if major == "k":
            return torch.randn(l_, m_, k_, dtype=ab_dtype, device=device) / (k_**0.5)
        else:  # m-major: stride (m*k, 1, m)
            return (
                torch.randn(l_, k_, m_, dtype=ab_dtype, device=device).transpose(1, 2)
                / (k_**0.5)
            )

    def _make_b_non_varlen(l_, n_, k_, major):
        """(l, n, k) with requested major. k-major: contig; n-major: transposed."""
        if major == "k":
            return torch.randn(l_, n_, k_, dtype=ab_dtype, device=device) / (k_**0.5)
        else:  # n-major: stride (n*k, 1, n)
            return (
                torch.randn(l_, k_, n_, dtype=ab_dtype, device=device).transpose(1, 2)
                / (k_**0.5)
            )

    if varlen_m:
        total_m = m * l
        cu_seqlens_m = torch.arange(0, l + 1, dtype=torch.int32, device=device) * m
        A = torch.randn(total_m, k, dtype=ab_dtype, device=device) / (k**0.5)
        if gather_A:
            A_idx = torch.randperm(total_m, dtype=torch.int32, device=device)
        B = _make_b_non_varlen(l, n, k, b_major)
        D = torch.empty(total_m, n, dtype=d_dtype, device=device)
    elif varlen_k:
        total_k = k * l
        cu_seqlens_k = torch.arange(0, l + 1, dtype=torch.int32, device=device) * k
        # m-major A, n-major B for varlen_k (enforced above).
        if gather_A:
            larger_k = total_k * 2
            A = torch.randn(larger_k, m, dtype=ab_dtype, device=device).T
            A_idx = torch.randperm(larger_k, dtype=torch.int32, device=device)[:total_k]
        else:
            A = torch.randn(total_k, m, dtype=ab_dtype, device=device).T
        B = torch.randn(total_k, n, dtype=ab_dtype, device=device).T
        D = torch.empty(l, m, n, dtype=d_dtype, device=device)
    else:
        A = _make_a_non_varlen(l, m, k, a_major)
        B = _make_b_non_varlen(l, n, k, b_major)
        D = torch.empty(l, m, n, dtype=d_dtype, device=device)

    C = None
    if args.c_dtype is not None:
        c_dtype_torch = _torch_dtype(args.c_dtype)
        c_shape = D.shape
        C = torch.randn(c_shape, dtype=c_dtype_torch, device=device) / (k**0.5)

    # ── Run / ref check ───────────────────────────────────────────────────────
    def fn():
        quack_gemm(
            A, B, D, C=C,
            tile_count_semaphore=tile_count_semaphore,
            tile_M=tile_M, tile_N=tile_N,
            cluster_M=cluster_M, cluster_N=cluster_N,
            pingpong=args.pingpong,
            persistent=persistent,
            is_dynamic_persistent=args.dynamic_persistent,
            max_swizzle_size=args.max_swizzle_size,
            cu_seqlens_m=cu_seqlens_m,
            cu_seqlens_k=cu_seqlens_k,
            A_idx=A_idx,
            use_tma_gather=args.use_tma_gather,
        )
        if tile_count_semaphore is not None and varlen_m:
            tile_count_semaphore.zero_()

    if not args.skip_ref_check:
        fn()
        torch.cuda.synchronize()
        if varlen_m:
            ref = torch.cat([
                (A[A_idx[cu_seqlens_m[i]:cu_seqlens_m[i+1]]] if gather_A
                 else A[cu_seqlens_m[i]:cu_seqlens_m[i+1]]) @ B[i].T
                for i in range(l)
            ])
        elif varlen_k:
            ref = torch.stack([
                (A[:, A_idx[cu_seqlens_k[i]:cu_seqlens_k[i+1]]] if gather_A
                 else A[:, cu_seqlens_k[i]:cu_seqlens_k[i+1]]) @
                B[:, cu_seqlens_k[i]:cu_seqlens_k[i+1]].T
                for i in range(l)
            ])
        else:
            ref = torch.bmm(A, B.mT)
        if C is not None:
            ref = ref + C.float()
        torch.testing.assert_close(D, ref.to(d_dtype), atol=tolerance, rtol=1e-3)
        print("Ref check PASSED")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    flops = 2 * m * n * k * l
    bytes_A = m * k * l * ab_dtype.itemsize
    bytes_B = n * k * l * ab_dtype.itemsize
    bytes_D = m * n * l * d_dtype.itemsize
    bytes_C = (m * n * l * C.dtype.itemsize) if C is not None else 0
    total_bytes = bytes_A + bytes_B + bytes_D + bytes_C

    fn_cublas = None
    if not (varlen_m or varlen_k) and not gather_A:
        fn_cublas = lambda: torch.bmm(A, B.mT)
        _bench_and_report("cuBLAS", fn_cublas, flops, warmup, repeats)

    timing = _bench_and_report("quack ", fn, flops, warmup, repeats, gbps_bytes=total_bytes)
    fn()

    if fn_cublas is not None:
        _bench_and_report("cuBLAS", fn_cublas, flops, warmup, repeats)


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
    print("PASS")
