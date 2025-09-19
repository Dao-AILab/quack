import argparse
import enum
from typing import Tuple, Type, Callable, Optional, Union
from functools import partial
import math

import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass import Int32, Boolean

from quack.dense_gemm_sm90 import GemmSm90, TileSchedulerOptions
from quack.varlen_utils import VarlenArguments

"""
To run this example:

.. code-block:: bash

    python examples/hopper/dense_gemm.py                                   \
      --mnkl 8192,8192,8192,1 --tile_shape_mn 128,256                      \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --d_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --d_major n

The above example command compute batched gemm with M=8192, N=8192, K=8192,
batch_count=1. The Hopper WGMMA tile shape is 128x256x64 and the cluster shape
is (1,1). The input, mma accumulator and output data type are set as fp16, fp32
and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/hopper/dense_gemm.py                               \
      --mnkl 8192,8192,8192,1 --tile_shape_mn 128,256,64                  \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --d_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --d_major n

Constraints:
* Supported input data types: fp16, fp8 (e4m3fn, e5m2)
* For fp16 types, A and B must have the same data type
* For fp8 types, A and B can have different types (e4m3fn or e5m2) but both must be 8-bit
* Fp8 types only support k-major layout
* Only fp32 accumulation is supported in this example
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* CTA tile shape K must be 64
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
"""


# /////////////////////////////////////////////////////////////////////////////
#  Helpers to parse args
# /////////////////////////////////////////////////////////////////////////////
def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of MxNxKxL GEMM on Hopper.")

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mn",
        type=parse_comma_separated_ints,
        default=(128, 256),
        help="Cta tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        choices=[(1, 1), (2, 1), (1, 2), (2, 2)],
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument(
        "--a_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--b_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--d_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        default=None,
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--d_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--tolerance", type=float, default=3e-02, help="Tolerance for validation")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument("--persistent", action="store_true", help="Persistent kernel")
    parser.add_argument(
        "--dynamic_persistent", action="store_true", help="Dynamic persistent kernel"
    )
    parser.add_argument("--pingpong", action="store_true", help="Pingpong kernel")
    parser.add_argument("--varlen_m", action="store_true", help="Variable length M dimension")
    parser.add_argument("--varlen_k", action="store_true", help="Variable length K dimension")
    parser.add_argument("--permute_batch", action="store_true", help="Permute batch in varlen_k")
    parser.add_argument("--gather_A", action="store_true", help="Gather A")
    parser.add_argument("--fp8_fast_accum", action="store_true", help="FP8 fast accum")
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")
    parser.add_argument("--use_cpu_varlen_m", action="store_true", help="Use CPU varlen M")
    parser.add_argument("--use_cpu_varlen_k", action="store_true", help="Use CPU varlen K")

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    if len(args.tile_shape_mn) != 2:
        parser.error("--tile_shape_mn must contain exactly 2 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")
    if args.use_cpu_varlen_m and not args.varlen_m:
        parser.error("--use_cpu_varlen_m requires --varlen_m")
    if args.use_cpu_varlen_k and not args.varlen_k:
        parser.error("--use_cpu_varlen_k requires --varlen_k")

    return args


def run(
    mnkl: Tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
    c_dtype: Optional[Type[cutlass.Numeric]],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    d_major: str,
    c_major: str,
    tile_shape_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    persistent: bool,
    dynamic_persistent: bool,
    pingpong: bool,
    varlen_m: bool,
    varlen_k: bool,
    permute_batch: bool,
    gather_A: bool,
    fp8_fast_accum: bool,
    use_cpu_varlen_m: bool,
    use_cpu_varlen_k: bool,
    **kwargs,
):
    """
    Prepare A/B/D/C tensors, launch GPU kernel, and reference checking.

    :param mnkl: Problem size (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param a_dtype: Data type for input tensor A
    :type a_dtype: Type[cutlass.Numeric]
    :param b_dtype: Data type for input tensor B
    :type b_dtype: Type[cutlass.Numeric]
    :param d_dtype: Data type for output tensor C
    :type d_dtype: Type[cutlass.Numeric]
    :param acc_dtype: Data type for accumulation during matrix multiplication
    :type acc_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/d_major: Memory layout of tensor A/B/C
    :type a_major/b_major/d_major: str
    :param tile_shape_mn: CTA tile shape (M, N)
    :type tile_shape_mn: Tuple[int, int, int]
    :param cluster_shape_mn: Cluster shape (M, N)
    :type cluster_shape_mn: Tuple[int, int]
    :param tolerance: Tolerance value for reference validation comparison
    :type tolerance: float
    :param warmup_iterations: Number of warmup iterations before benchmarking, defaults to 0
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations to run, defaults to 1
    :type iterations: int, optional
    :param skip_ref_check: Whether to skip reference result validation, defaults to False
    :type skip_ref_check: bool, optional
    """

    if dynamic_persistent:
        persistent = True

    print("Running Hopper Dense GEMM with:")
    print(f"mnkl: {mnkl}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, D dtype: {d_dtype}, C_dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, D: {d_major}")
    print(f"Tile Shape: {tile_shape_mn}, Cluster Shape: {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")

    # Unpack parameters
    m, n, k, l = mnkl
    cluster_shape_mnk = (*cluster_shape_mn, 1)

    # Skip unsupported types
    if not GemmSm90.is_valid_dtypes(
        a_dtype, b_dtype, acc_dtype, d_dtype, a_major, b_major
    ):
        raise TypeError(
            f"Skipping due to unsupported combination of types and majors: {a_dtype}, {b_dtype}, {acc_dtype}, {d_dtype}, {a_major=}, {b_major=}"
        )

    # Prepare pytorch tensors: A, B (random from 0 to 2) and C (all zero)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True):
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # else : (l, mode0, mode1) -> (mode0, mode1, l)
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        is_unsigned = dtype in {cutlass.Uint8}
        # Temporarily use uint8 as torch does not support fp8 type
        torch_dtype = cutlass_torch.dtype(dtype)
        gen_dtype = (
            torch_dtype
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch.bfloat16
        )

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass.torch.create_and_permute_torch_tensor(
            shape,
            gen_dtype,
            permute_order=permute_order,
            # init_type=cutlass.torch.TensorInitType.RANDOM,
            # init_config=cutlass.torch.RandomInitConfig(
            #     min_val=0 if is_unsigned else -2, max_val=4 if is_unsigned else 2
            # ),
            init_type=cutlass.torch.TensorInitType.GAUSSIAN,
            init_config=cutlass.torch.GaussianInitConfig(std=k ** (-0.5), scale=1),
        ).to(torch_dtype)
        # Create dtype torch tensor (gpu)
        torch_tensor = torch_tensor_cpu.cuda()

        # Create f32 torch tensor (cpu)
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # Create dtype cute tensor (gpu)
        torch_tensor_view = (
            torch_tensor
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch_tensor.view(torch.uint8)
        )
        cute_tensor = from_dlpack(torch_tensor_view, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=(0 if is_mode0_major else 1))
            cute_tensor = cute_tensor.mark_compact_shape_dynamic(
                mode=(1 if not is_mode0_major else 0),
                stride_order=(2, 0, 1) if not is_mode0_major else (2, 1, 0),
                divisibility=(128 // dtype.width),
            )
        cute_tensor = cutlass.torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor

    a, mA, a_torch = create_and_permute_tensor(l, m, k, a_major == "m", a_dtype)
    if gather_A:
        assert a_major == "k"
        a_idx = torch.randperm(l * m, dtype=torch.int32, device="cuda")
        from einops import rearrange

        a = rearrange(rearrange(a, "m k l -> (m l) k")[a_idx.cpu()], "(m l) k -> m k l", m=m)
        a_torch = rearrange(a_torch, "m k l -> (m l) k")
        mA = from_dlpack(a_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        a_idx_reshaped = rearrange(a_idx, "(m l) -> l m", m=m).contiguous().transpose(0, 1)
        mAIdx = from_dlpack(a_idx_reshaped, assumed_align=4).mark_layout_dynamic(leading_dim=0)
    else:
        mAIdx = None
    b, mB, b_torch = create_and_permute_tensor(l, n, k, b_major == "n", b_dtype)
    _, mD, d_torch = create_and_permute_tensor(l, m, n, d_major == "m", d_dtype)
    if c_dtype is not None:
        c, mC, c_torch = create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)
    else:
        c, mC, c_torch = None, None, None
    if varlen_m:
        assert a_major == "k"
        assert d_major == "n"
        from einops import rearrange

        a, d_torch = [rearrange(t, "m x l -> (l m) x") for t in (a, d_torch)]
        if not gather_A:
            (a_torch,) = [rearrange(t, "m x l -> (l m) x") for t in (a_torch,)]
        if c_dtype is not None:
            c, c_torch = [rearrange(t, "m x l -> (l m) x") for t in (c, c_torch)]
            mC = from_dlpack(c_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mA = from_dlpack(a_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mD = from_dlpack(d_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        # TODO: generate random cu_seqlens_m
        cu_seqlens_m = torch.arange(0, l + 1, dtype=torch.int32, device="cuda") * m
        mCuSeqlensM = from_dlpack(cu_seqlens_m, assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if use_cpu_varlen_m:
            mCuSeqlensM = list(range(0, (l + 1) * m, m))
        if gather_A:
            a_idx_reshaped = rearrange(a_idx_reshaped, "m l -> (l m)")
            mAIdx = from_dlpack(a_idx_reshaped, assumed_align=4).mark_layout_dynamic(leading_dim=0)
    else:
        cu_seqlens_m, mCuSeqlensM = None, None

    if varlen_k:
        from einops import rearrange
        assert not gather_A

        a, a_torch = [rearrange(t, "m k l -> m (l k)") for t in (a, a_torch)]
        b, b_torch = [rearrange(t, "n k l -> n (l k)") for t in (b, b_torch)]
        mA = from_dlpack(a_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1 if a_major == "k" else 0)
        mB = from_dlpack(b_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1 if b_major == "k" else 0)
        # TODO: generate random cu_seqlens_k
        cu_seqlens_k = torch.arange(0, l + 1, dtype=torch.int32, device="cuda") * k
        mCuSeqlensK = from_dlpack(cu_seqlens_k, assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if use_cpu_varlen_k:
            mCuSeqlensK = list(range(0, (l + 1) * k, k))
    else:
        cu_seqlens_k, mCuSeqlensK = None, None


    if varlen_m or varlen_k:  # Need to allocate space in gmem to store tensormaps
        if not persistent:
            total_m = m * l
            block_size_m = tile_shape_mn[0] * cluster_shape_mnk[0]
            block_size_n = tile_shape_mn[1] * cluster_shape_mnk[1]
            total_clusters_m_max = (total_m + l * (block_size_m - 1)) // block_size_m
            total_clusters_max = total_clusters_m_max * ((n + block_size_n - 1) // block_size_n)
            total_ctas = total_clusters_max * cluster_shape_mnk[0] * cluster_shape_mnk[1]
        else:
            total_ctas = cutlass.utils.HardwareInfo().get_device_multiprocessor_count()
        num_tensormaps = 2 if varlen_k else (1 if not pingpong else 2)
        # 128 bytes per tensormap
        tensormaps_torch = torch.empty(total_ctas, num_tensormaps, 128 // 8, dtype=torch.int64, device="cuda")
        tensormaps_tensor = from_dlpack(
            tensormaps_torch, assumed_align=128
        ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
    else:
        tensormaps_tensor = None

    gemm = GemmSm90(
        acc_dtype,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong=pingpong,
        is_persistent=persistent,
        fp8_fast_accum=fp8_fast_accum,
        gather_A=gather_A,
    )

    # Compute max active clusters on current device
    if persistent:
        max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )
        if dynamic_persistent:
            tile_count_semaphore = torch.zeros(1, dtype=torch.int32, device="cuda")
        else:
            tile_count_semaphore = None
        # max_active_clusters = 1
    else:
        max_active_clusters = 0
        tile_count_semaphore = None
    if permute_batch:
        batch_idx_permute = torch.randperm(l, dtype=torch.int32, device="cuda")
        batch_idx_permute_tensor = from_dlpack(batch_idx_permute, assumed_align=4).mark_layout_dynamic(leading_dim=0)
    else:
        batch_idx_permute_tensor = None
    scheduler_args = TileSchedulerOptions(
        Int32(max_active_clusters),
        tile_count_semaphore=make_ptr(
            Int32, tile_count_semaphore.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        ) if tile_count_semaphore is not None else None,
        batch_idx_permute=batch_idx_permute_tensor
    )

    epi_args = gemm.EpilogueArguments()
    varlen_args = VarlenArguments(mCuSeqlensM, mCuSeqlensK, tensormaps_tensor)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # compile gemm kernel
    compiled_gemm = cute.compile(
        gemm,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        mAIdx,
        current_stream,
    )

    if not skip_ref_check:
        # execution
        compiled_gemm(mA, mB, mD, mC, epi_args, scheduler_args, varlen_args, mAIdx, current_stream)
        if tile_count_semaphore is not None and varlen_m:
            tile_count_semaphore.zero_()

        torch.cuda.synchronize()

        # Ref check
        if not varlen_m and not varlen_k:
            ref = torch.einsum("mkl,nkl->mnl", a, b)
        elif varlen_m:
            ref = torch.cat(
                [
                    torch.einsum("mk,nk->mn", a[cu_seqlens_m[i] : cu_seqlens_m[i + 1]], b[:, :, i])
                    for i in range(l)
                ],
                dim=0,
            )
        else:
            assert varlen_k
            ref = torch.stack(
                [
                    torch.einsum("mk,nk->mn", a[:, cu_seqlens_k[i] : cu_seqlens_k[i + 1]], b[:, cu_seqlens_k[i] : cu_seqlens_k[i + 1]])
                    for i in range(l)
                ],
                dim=-1,
            )
        if c is not None:
            ref = ref + c
        ref = ref.cpu()

        if d_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
            # m major: (l, n, m) -> (m, n, l)
            # n major: (l, m, n) -> (m, n, l)
            permute_order = (1, 2, 0) if d_major == "n" else (2, 1, 0)
            shape = (l, m, n) if d_major == "n" else (l, n, m)
            f8_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
                shape,
                torch.uint8,
                permute_order=permute_order,
                init_type=cutlass_torch.TensorInitType.SKIP,
            ).cuda()
            # Create dtype cute tensor (gpu)
            ref_d_tensor = from_dlpack(f8_torch_tensor, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if d_major == "n" else 0)
            )
            ref_d_tensor.element_type = d_dtype
            ref_d_tensor = cutlass_torch.convert_cute_tensor(
                ref,
                ref_d_tensor,
                d_dtype,
                is_dynamic_layout=True,
            )
            ref_d = f8_torch_tensor.cpu()
        else:
            ref_d = ref.to(cutlass_torch.dtype(d_dtype))

        out = d_torch.cpu().squeeze()
        out_ref = ref_d.squeeze()
        # breakpoint()
        torch.testing.assert_close(d_torch.cpu(), ref_d, atol=tolerance, rtol=1e-03)

    # return

    from triton.testing import do_bench

    flops = 2 * m * n * k * l
    # Calculate memory bandwidth
    bytes_A = m * k * l * (a_dtype.width // 8)  # A tensor: (m, k, l)
    bytes_B = n * k * l * (b_dtype.width // 8)  # B tensor: (n, k, l)
    bytes_D = m * n * l * (d_dtype.width // 8)  # D tensor: (m, n, l)
    bytes_C = m * n * l * (c_dtype.width // 8) if c_dtype is not None else 0  # C tensor: (m, n, l)
    total_bytes = bytes_A + bytes_B + bytes_D + bytes_C  # Read A, B, C; Write D

    repeats = iterations
    warmup = warmup_iterations

    import time

    if not (varlen_m or varlen_k) and not gather_A:
        time.sleep(0.5)
        if a_dtype.width == 8:
            assert l == 1
            scale_ab = torch.ones((1,), dtype=torch.float32, device="cuda")
            fn_cublas = lambda: torch._scaled_mm(
                a_torch[:, :, 0],
                b_torch[:, :, 0].mT,
                scale_a=scale_ab,
                scale_b=scale_ab,
                out_dtype=torch.bfloat16,
                use_fast_accum=fp8_fast_accum,
            )
        else:
            if c_torch is None:
                fn_cublas = lambda: torch.matmul(
                    a_torch.permute(2, 0, 1), b_torch.permute(2, 0, 1).mT
                )
            else:
                c_torch_convert = c_torch.to(a_torch.dtype)  # In case C is in FP32
                fn_cublas = lambda: torch.baddbmm(
                    c_torch_convert.permute(2, 0, 1),
                    a_torch.permute(2, 0, 1),
                    b_torch.permute(2, 0, 1).mT,
                )
        timing_cublas = do_bench(fn_cublas, warmup=warmup, rep=repeats)
        tflops_cublas = flops / (timing_cublas * 1e9)  # Convert to TFlops
        print(f"CuBLAS Average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")

    time.sleep(0.5)

    def fn():
        compiled_gemm(mA, mB, mD, mC, epi_args, scheduler_args, varlen_args, mAIdx, current_stream)
        if tile_count_semaphore is not None and varlen_m:
            tile_count_semaphore.zero_()

    timing = do_bench(fn, warmup=warmup, rep=repeats)
    # Idk why but for some cases the 1st run is much slower
    time.sleep(0.5)
    timing = do_bench(fn, warmup=warmup, rep=repeats)
    tflops = flops / (timing * 1e9)  # Convert to TFlops
    gbps = total_bytes / (timing * 1e6)  # Convert to GB/s (1e9 for ms->s, 1e9 for B->GB)
    print(f"Cute-DSL Average time: {timing:.3f} ms, TFLOPS: {tflops:.1f}, GB/s: {gbps:.0f}")
    fn()

    if not (varlen_m or varlen_k) and not gather_A:
        time.sleep(0.5)
        timing_cublas = do_bench(fn_cublas, warmup=warmup, rep=repeats)
        tflops_cublas = flops / (timing_cublas * 1e9)  # Convert to TFlops
        print(f"CuBLAS Average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")

        from flash_attn.utils.benchmark import pytorch_profiler

        pytorch_profiler(fn_cublas)
        pytorch_profiler(fn)
        # pytorch_profiler(torch.sort, d_torch.squeeze(), dim=-1)
        # pytorch_profiler(torch.compile(torch.sort), d_torch.squeeze(), dim=-1)
        # pytorch_profiler(torch.topk, d_torch.squeeze(), dim=-1, k=1)
        # pytorch_profiler(torch.compile(torch.topk), d_torch.squeeze(), dim=-1, k=1)
        # pytorch_profiler(torch.square, d_torch.squeeze())


if __name__ == "__main__":
    args = parse_arguments()
    run(
        args.mnkl,
        args.a_dtype,
        args.b_dtype,
        args.d_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.d_major,
        args.c_major,
        args.tile_shape_mn,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.persistent,
        args.dynamic_persistent,
        args.pingpong,
        args.varlen_m,
        args.varlen_k,
        args.permute_batch,
        args.gather_A,
        args.fp8_fast_accum,
        args.use_cpu_varlen_m,
        args.use_cpu_varlen_k,
    )
    print("PASS")
