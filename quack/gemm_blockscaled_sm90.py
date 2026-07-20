# Copyright (c) 2026, Tri Dao.

"""PyTorch-friendly interface for the MXFP8 blockscaled GEMM (SM90 / Hopper).

SM90-only. The SM100 (Blackwell) blockscaled path lives in the unified
`quack.gemm_interface` (`gemm`, `gemm_act`, ...) + `quack.blockscaled.utils`; that
interface rejects SM90 blockscaled, so the SM90 mxfp8 logic (128-element K-blocks,
software-applied f32 scales) is kept here.

Layout overview:
  A:       (M, K)     or (L, M, K)       dtype float8_e4m3fn, K-contiguous (row-major)
  B:       (K, N)     or (L, K, N)       dtype float8_e4m3fn, K-contiguous (col-major)
  A_scale: (M, K/128) or (L, M, K/128)   dtype float32, M-innermost
  B_scale: (K/128, N) or (L, K/128, N)   dtype float32
  out:     (M, N)     or (L, M, N)       dtype bfloat16/float16, contiguous

"K-contiguous" means stride 1 on the K axis. This matches how torchao/cuBLAS
use `torch._scaled_mm(a, b.t(), ...)`: weight stored as `(N, K)` row-major,
pass `W.mT` (zero-copy view of shape `(K, N)` with K-contig) as B. The
interface applies `.mT` internally to reach the `(N, K) K-major` layout the
kernels consume; no data is copied.
"""

from functools import partial
from typing import Optional, Tuple

import torch
from torch import Tensor
import triton
import triton.language as tl

import cutlass
import cutlass.cute as cute

from quack.activation import act_fn_map, dgate_fn_map, gate_fn_map
from quack.cache import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.gemm_act import (
    GemmActMixin,
    GemmActSm90,
    GemmGatedSm90,
)
from quack.autotuner import autotune, AutotuneConfig
from quack.gemm_config import GemmConfig, _get_sm90_blockscaled_configs
from quack.gemm_dact import GemmDGatedMixin, GemmDGatedSm90
from quack.gemm_interface import (
    Activation,
    GatedActivation,
    _concat_interleave_bias,
    _empty_k_matmul_into,
    gated_to_pytorch_fn_map,
    prune_invalid_gemm_configs,
)
from quack.gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    div_for_dtype,
    get_major,
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
    perm3d_single,
)
from quack.quant import blockwise_quant

_SF_VEC_SIZE_SM90 = 128  # SM90 K-block size (activations and weights)
_WEIGHT_BLOCK_N_SM90 = 128  # SM90 N-block size for weight scales


def default_config(device):
    cap = get_device_capacity(device)[0]
    if cap == 9:
        return GemmConfig(
            tile_m=128,
            tile_n=256,
            cluster_m=1,
            cluster_n=2,
            pingpong=False,
            is_dynamic_persistent=False,
        )
    else:
        raise NotImplementedError("Currently only Hopper is supported")


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def quantize_act(x: Tensor) -> Tuple[Tensor, Tensor]:
    """SM90 activation quantization: `blockwise_quant` (128-element K-blocks, M-innermost f32
    scales) — the fast CuTe quantizer the GEMM's SFA expects. For varlen/grouped-M, quantize
    here (unpadded) then dQaccum-pad with `grouped_scale_to_dqaccum` / `permute_scale_to_dqaccum`.
    """
    return blockwise_quant(x, block_size=_SF_VEC_SIZE_SM90, scale_transpose=True)


# ---------------------------------------------------------------------------
# SM90 GEMM (Hopper)
# ---------------------------------------------------------------------------


@jit_cache
def _compile_mxfp8_gemm_act_sm90(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    postact_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    postact_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    activation,
    rowvec_dtype,
    colvec_dtype,
    colvec_ndim,
    varlen_m,
    gather_A,
    concat_layout,
    device_capacity,
    sr_seed_mode=0,
    use_tma_gather=False,
    epi_tile_n=None,
):
    is_gated = activation in gate_fn_map
    GemmCls = GemmGatedSm90 if is_gated else GemmActSm90

    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        varlen_m=varlen_m,
        gather_A=gather_A,
    )

    pa_leading = 1 if postact_major == "n" else 0
    pa_n = cute.sym_int() if is_gated else n
    div_pa = div_for_dtype(postact_dtype)
    pa_leading_dim = 1 if is_gated else pa_leading
    pa_shape = (m, pa_n) if varlen_m else (m, pa_n, l)
    mAuxOut = fake_tensor(postact_dtype, pa_shape, leading_dim=pa_leading_dim, divisibility=div_pa)

    mRowVec = (
        fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4) if rowvec_dtype else None
    )
    if colvec_ndim == 2:
        mColVec = (
            fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
            if colvec_dtype
            else None
        )
    elif colvec_ndim == 1:
        mColVec = (
            fake_tensor(colvec_dtype, (m,), leading_dim=0, divisibility=4) if colvec_dtype else None
        )
    else:
        mColVec = None

    from cutlass import Int32
    from cutlass.cute.runtime import make_ptr

    act_fn = gate_fn_map[activation] if is_gated else act_fn_map[activation]

    def fake_scalar(mode):
        if mode == 0:
            return None
        elif mode == 1:
            return Int32(0)
        else:
            return make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=4)

    epi_args = GemmCls.EpilogueArguments(
        mAuxOut,
        act_fn,
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        rounding_mode=0,  # RoundingMode.RN, Constexpr baked at compile time
        sr_seed=fake_scalar(sr_seed_mode),
    )
    scheduler_args = make_fake_scheduler_args(is_dynamic_persistent, False, l)
    varlen_args = make_fake_varlen_args(varlen_m, False, gather_A, m if varlen_m else None)

    # SM90 blockscaled: float32 scales.
    # A scale: dispatch produces (m, sf_k, l) (or (m, sf_k) varlen) with M innermost
    # so the TMA atom can do a single contiguous (BLOCK_M, 1) burst per K-stage.
    # B scale: (l, n_blocks, sf_k) — read directly from gmem in math warps (no TMA).
    sf_k_sym = cute.sym_int()
    n_blocks_sym = cute.sym_int()
    if varlen_m:
        # SFA's M extent is dQaccum-padded (total_padded_m), independent of A's
        # total_m — give it its own symbol so the kernel does NOT bake in
        # mSFA.shape[0] == mA.shape[0] (which no longer holds after padding).
        padded_m_sym = cute.sym_int()
        fake_sfa = fake_tensor(
            cutlass.Float32, (padded_m_sym, sf_k_sym), leading_dim=0, divisibility=1
        )
    else:
        fake_sfa = fake_tensor(cutlass.Float32, (m, sf_k_sym, l), leading_dim=0, divisibility=1)
    fake_sfb = fake_tensor(
        cutlass.Float32, (l, n_blocks_sym, sf_k_sym), leading_dim=2, divisibility=1
    )
    return compile_gemm_kernel(
        partial(
            GemmCls,
            sf_vec_size=_SF_VEC_SIZE_SM90,
            weight_n_block=_WEIGHT_BLOCK_N_SM90,
            epi_tile_n=epi_tile_n,
        ),
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        gather_A,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        mSFA=fake_sfa,
        mSFB=fake_sfb,
        use_tma_gather=use_tma_gather,
        concat_layout=concat_layout or None,
    )


def mxfp8_gemm_act_dispatch_sm90(
    A: Tensor,  # (l, m, k) K-contig
    B: Tensor,  # (l, n, k) K-contig
    A_scale: Tensor,  # (l, m, k/32) K-contig
    B_scale: Tensor,  # (l, n, k/32) K-contig
    D: Optional[Tensor],  # (l, m, n) or None (preact_out)
    C: Optional[Tensor],  # (l, m, n) or None
    PostAct: Tensor,  # (l, m, n//2) for gated
    tile_count_semaphore: Optional[Tensor],
    activation: str,
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    tile_K: int | None = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,
    colvec_bias: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    use_tma_gather: bool = False,
    concat_layout: tuple | None = None,
    epi_tile_n: int | None = None,
) -> None:
    varlen_m = cu_seqlens_m is not None
    gather_A = A_idx is not None

    A_p = perm3d_single(A, varlen_m)
    B_p = perm3d_single(B)
    D_p = perm3d_single(D, varlen_m) if D is not None else None
    C_p = perm3d_single(C, varlen_m) if C is not None else None
    PostAct_p = perm3d_single(PostAct, varlen_m)

    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n") if D_p is not None else None
    c_major = get_major(C_p, "m", "n") if C_p is not None else None
    postact_major = get_major(PostAct_p, "m", "n")

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype] if D is not None else None
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    colvec_ndim = colvec_bias.ndim if colvec_bias is not None else 0

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] == 9, "mxfp8_gemm_act_dispatch_sm90 requires SM90 (Hopper)"

    if not GemmActSm90.is_valid_dtypes(
        a_dtype, b_dtype, cutlass.Float32, d_dtype, a_major, b_major
    ):
        raise ValueError(
            f"unsupported SM90 mxfp8 config: a_dtype={a_dtype}, b_dtype={b_dtype}, "
            f"d_dtype={d_dtype}, a_major={a_major}, b_major={b_major} "
            f"(SM90 fp8 requires K-major A and B)"
        )

    concat_layout_key = tuple(sorted(concat_layout)) if concat_layout else ()
    compiled_fn = _compile_mxfp8_gemm_act_sm90(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        postact_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        postact_major,
        (tile_M, tile_N, tile_K) if tile_K is not None else (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        activation,
        torch2cute_dtype_map[rowvec_bias.dtype] if rowvec_bias is not None else None,
        torch2cute_dtype_map[colvec_bias.dtype] if colvec_bias is not None else None,
        colvec_ndim,
        varlen_m,
        gather_A,
        concat_layout_key,
        device_capacity,
        use_tma_gather=use_tma_gather,
        epi_tile_n=epi_tile_n,
    )

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    epi_args = GemmActMixin.EpilogueArguments(
        PostAct_p,
        None,  # act_fn is Constexpr, baked at compile time
        mRowVecBroadcast=rowvec_bias,
        mColVecBroadcast=colvec_bias,
        rounding_mode=None,  # Constexpr, baked at compile time
        sr_seed=None,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters, max_swizzle_size, tile_count_semaphore
    )
    varlen_args = make_varlen_args(cu_seqlens_m, None, A_idx)

    # Scales are float32.
    # SFA: kernel sees logical shape (m, sf_k, l) (or (m, sf_k) for varlen) with
    # *M as the innermost contiguous dim* — matches the DeepGEMM a.cu reference
    # convention. TMA loads (BLOCK_M, 1) per K-stage as a single 512B burst from
    # M-contiguous memory; a K-major (sf_k stride 1) layout would force TMA to
    # do a strided gather and read wrong values.
    if varlen_m:
        sfa_sm90 = A_scale
    else:
        sfa_sm90 = A_scale.permute(1, 2, 0)  # view: (m, sf_k, l), M innermost
    sfb_sm90 = B_scale.contiguous()  # (l, n_blocks, sf_k); read directly from gmem
    compiled_fn(
        A_p,
        B_p,
        D_p,
        C_p,
        epi_args,
        scheduler_args,
        varlen_args,
        sfa_sm90,
        sfb_sm90,
        None,
    )


@autotune(
    configs=[AutotuneConfig(config=c) for c in _get_sm90_blockscaled_configs("gated")],
    key=["activation", "dynamic_scheduler"],
    prune_configs_by={"early_config_prune": prune_invalid_gemm_configs},
)
def mxfp8_gemm_gated_tuned_sm90(
    # (M, K) or or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N)
    A_scale: Tensor,
    B_scale: Tensor,
    # (M, N) or (L, M, N) or (total_M, N) if varlen_m - None if not storing preact
    preact_out: Optional[Tensor],
    postact_out: Tensor,  # (M, N//2) or (L, M, N//2) or (total_M, N//2) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: GatedActivation = "swiglu",
    cu_seqlens_m: Optional[Tensor] = None,  # (L+1), int32
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
) -> None:
    if config is None:
        config = default_config(A.device)
    varlen_m = cu_seqlens_m is not None
    if varlen_m:
        assert not config.swap_ab, "Variable-length sequences not supported with swap_ab"
    if A.ndim == 2 and not varlen_m:
        A = A.unsqueeze(0)  # (1, M, K)
        A_scale = A_scale.unsqueeze(0)
    B, B_scale = B.mT, B_scale.mT  # (N, K) or (L, N, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)  # (1, N, K)
        B_scale = B_scale.unsqueeze(0)
    if C is not None and C.ndim == 2 and not varlen_m:
        C = C.unsqueeze(0)  # (1, M, N)
    if preact_out is not None and preact_out.ndim == 2 and not varlen_m:
        D = preact_out.unsqueeze(0)
    else:
        D = preact_out
    if postact_out.ndim == 2 and not varlen_m:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)  # (L, N)
    if concat_layout and "bias" in concat_layout:
        if bias is not None and bias.dtype.itemsize >= 4:
            bias_key = "mColVecBroadcast" if config.swap_ab else "mRowVecBroadcast"
            concat_layout = tuple(bias_key if k == "bias" else k for k in concat_layout)
        else:
            concat_layout = tuple(k for k in concat_layout if k != "bias")
            if bias is not None:
                bias = _concat_interleave_bias(bias)
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    mxfp8_gemm_act_dispatch_sm90(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        A_scale if not config.swap_ab else B_scale,
        B_scale if not config.swap_ab else A_scale,
        (D if not config.swap_ab else D.mT) if D is not None else None,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        rowvec_bias=bias if not config.swap_ab else None,
        colvec_bias=bias if config.swap_ab else None,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        use_tma_gather=config.use_tma_gather,
        concat_layout=concat_layout,
        epi_tile_n=config.epi_tile_n,
    )


@autotune(
    configs=[AutotuneConfig(config=c) for c in _get_sm90_blockscaled_configs()],
    key=["activation", "dynamic_scheduler"],
    prune_configs_by={"early_config_prune": prune_invalid_gemm_configs},
)
def mxfp8_gemm_act_tuned_sm90(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    A_scale: Tensor,
    B_scale: Tensor,
    preact_out: Optional[Tensor],  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Activation = None,
    cu_seqlens_m: Optional[Tensor] = None,  # (L+1), int32
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
) -> None:
    if config is None:
        config = default_config(A.device)
    varlen_m = cu_seqlens_m is not None
    if varlen_m:
        assert not config.swap_ab, "Variable-length sequences not supported with swap_ab"
    if A.ndim == 2 and not varlen_m:
        A = A.unsqueeze(0)  # (1, M, K)
        A_scale = A_scale.unsqueeze(0)
    B, B_scale = B.mT, B_scale.mT  # (N, K) or (L, N, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)  # (1, N, K)
        B_scale = B_scale.unsqueeze(0)
    if C is not None and C.ndim == 2 and not varlen_m:
        C = C.unsqueeze(0)  # (1, M, N)
    if preact_out is not None and preact_out.ndim == 2 and not varlen_m:
        D = preact_out.unsqueeze(0)
    else:
        D = preact_out
    if postact_out.ndim == 2 and not varlen_m:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)  # (L, N)
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    mxfp8_gemm_act_dispatch_sm90(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        A_scale if not config.swap_ab else B_scale,
        B_scale if not config.swap_ab else A_scale,
        (D if not config.swap_ab else D.mT) if D is not None else None,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        rowvec_bias=bias if not config.swap_ab else None,
        colvec_bias=bias if config.swap_ab else None,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        use_tma_gather=config.use_tma_gather,
        epi_tile_n=config.epi_tile_n,
    )


def mxfp8_gemm_gated_out_sm90(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    A_scale: Tensor,
    B_scale: Tensor,
    preact_out: Optional[Tensor],  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Tensor,  # (M, N//2) or (L, M, N//2) or (total_M, N//2) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: GatedActivation = "swiglu",
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: Optional[str] = None,
) -> None:
    """GEMM with gated activation and pre-allocated output tensors."""
    fn = (
        mxfp8_gemm_gated_tuned_sm90
        if tuned
        else partial(mxfp8_gemm_gated_tuned_sm90.fn, config=None)
    )
    fn(
        A,
        B,
        A_scale,
        B_scale,
        preact_out,
        postact_out,
        C,
        bias,
        activation,
        cu_seqlens_m,
        A_idx,
        dynamic_scheduler,
        concat_layout=tuple(concat_layout.split(",")) if concat_layout else None,
    )


def mxfp8_gemm_act_out_sm90(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    A_scale: Tensor,
    B_scale: Tensor,
    preact_out: Optional[Tensor],  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Activation = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    tuned: bool = True,
) -> None:
    """GEMM with activation and pre-allocated output tensors."""
    fn = mxfp8_gemm_act_tuned_sm90 if tuned else partial(mxfp8_gemm_act_tuned_sm90.fn, config=None)
    fn(
        A,
        B,
        A_scale,
        B_scale,
        preact_out,
        postact_out,
        C,
        bias,
        activation,
        cu_seqlens_m,
        A_idx,
        dynamic_scheduler,
    )


def mxfp8_gemm_act_sm90(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    A_scale: Tensor,
    B_scale: Tensor,
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Activation = None,
    preact_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    store_preact: bool = True,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
) -> Tuple[Optional[Tensor], Tensor]:
    """GEMM with activation (or gated activation) and optional output tensors."""
    is_gated = activation in gated_to_pytorch_fn_map
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
    varlen_m = cu_seqlens_m is not None
    if not varlen_m:
        # SM90 constraints (non-varlen): M % 8, and the "1d2d" block-scaled quant scheme:
        #   A_scale: (..., M,        K // 128)   from quantize_act_sm90    (1 × 128)
        #   B_scale: (..., K // 128, N // 128)   from quantize_weight_sm90 (128 × 128), passed as .mT
        m_dim = A.shape[-2]  # works for both 2D (M, K) and 3D (L, M, K)
        k_dim = A.shape[-1]
        n_dim = B.shape[-1]
        assert m_dim % 8 == 0, f"SM90 mxfp8 GEMM requires M % 8 == 0; got M={m_dim}"
        assert A_scale.shape[-2:] == (m_dim, k_dim // 128), (
            f"SM90 expects A_scale from quantize_act_sm90 (1x128): "
            f"shape (..., M={m_dim}, K/128={k_dim // 128}); got {tuple(A_scale.shape)}"
        )
        assert B_scale.shape[-2:] == (k_dim // 128, n_dim // 128), (
            f"SM90 expects B_scale from quantize_weight_sm90 (128x128, passed as .mT): "
            f"shape (..., K/128={k_dim // 128}, N/128={n_dim // 128}); got {tuple(B_scale.shape)}"
        )
    # Determine output shape based on gather_A
    if varlen_m:
        total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
        out_shape = (total_m, B.shape[-1])
    elif A.ndim == 2:
        out_shape = (A.shape[0], B.shape[-1])
    else:
        out_shape = (A.shape[0], A.shape[-2], B.shape[-1])
    postact_shape = (*out_shape[:-1], out_shape[-1] // 2) if is_gated else out_shape
    if preact_out is None and store_preact:
        preact_out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    if postact_out is None:
        postact_out = torch.empty(postact_shape, dtype=postact_dtype, device=A.device)
    # Empty-input fast path. For M=0 or N=0 the outputs are empty; for K=0
    # (A@B == 0) the no-bias / no-C surface yields preact=0 and act(0)=0 for
    # every supported activation, so both outputs are zero.
    if postact_out.numel() == 0 or A.numel() == 0:
        if preact_out is not None:
            _empty_k_matmul_into(preact_out)
        _empty_k_matmul_into(postact_out)
        return preact_out, postact_out
    concat_str = ",".join(concat_layout) if concat_layout else None
    if is_gated:
        mxfp8_gemm_gated_out_sm90(
            A,
            B,
            A_scale,
            B_scale,
            preact_out,
            postact_out,
            C,
            bias,
            activation,
            cu_seqlens_m,
            A_idx,
            dynamic_scheduler,
            tuned,
            concat_layout=concat_str,
        )
    else:
        mxfp8_gemm_act_out_sm90(
            A,
            B,
            A_scale,
            B_scale,
            preact_out,
            postact_out,
            C,
            bias,
            activation,
            cu_seqlens_m,
            A_idx,
            dynamic_scheduler,
            tuned,
        )
    return preact_out, postact_out


# ---------------------------------------------------------------------------
# SM90 GEMM + gated-activation backward (MXFP8 blockscaled A/B, bf16 packed C/D)
# ---------------------------------------------------------------------------
#
# Fuses `dOut @ W -> raw_dA`, colvec (topk-score) scaling, and the SwiGLU/GeGLU
# backward into one epilogue, mirroring the bf16 `GemmDGatedSm90` kernel
# (quack.gemm_dact) but with an MXFP8 blockscaled mainloop. `GemmDGatedSm90`
# already subclasses the same `GemmSm90` mainloop that `GemmGatedSm90` uses for
# the forward blockscaled gated GEMM, which is itself generic over blockscaled
# vs. bf16 (toggled by `sf_vec_size`/`mSFA`/`mSFB`) — so no new mainloop math is
# needed, only the compile/dispatch plumbing to hand it MXFP8 scale factors.
#
# PreAct (h, read as "C") and D (dh, written as "D") are physically bf16 but
# viewed as float32 (2 packed bf16 values per element) — same 16-bit-packed-
# into-32-bit trick `GemmDGatedMixin` uses in the bf16 kernel.


@jit_cache
def _compile_mxfp8_gemm_dgated_sm90(
    a_dtype,
    b_dtype,
    d_dtype,  # dtype of D as viewed as f32 (packed dh)
    c_dtype,  # dtype of C as viewed as f32 (packed h)
    postact_dtype,
    implicit_dtype,  # actual 16-bit dtype packed into C/D
    a_major,
    b_major,
    d_major,
    c_major,
    postact_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    activation,
    colvec_scale_dtype,
    colvec_scale_ndim,
    colvec_reduce_dtype,
    colvec_reduce_ndim,
    varlen_m,
    gather_A,
    device_capacity,
    use_tma_gather=False,
):
    GemmCls = GemmDGatedSm90
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        varlen_m=varlen_m,
        gather_A=gather_A,
    )
    div_pa = div_for_dtype(postact_dtype)
    pa_leading = 1 if postact_major == "n" else 0
    pa_shape = (m, n) if varlen_m else (m, n, l)
    mAuxOut = fake_tensor(postact_dtype, pa_shape, leading_dim=pa_leading, divisibility=div_pa)

    act_fn = dgate_fn_map[activation]

    mColVec = None
    if colvec_scale_ndim == 2:
        mColVec = fake_tensor(colvec_scale_dtype, (l, m), leading_dim=1, divisibility=4)
    elif colvec_scale_ndim == 1:
        mColVec = fake_tensor(colvec_scale_dtype, (m,), leading_dim=0, divisibility=4)
    mColVecReduce = None
    n_tiles = cute.sym_int()
    if colvec_reduce_ndim == 3:
        mColVecReduce = fake_tensor(
            colvec_reduce_dtype, (l, m, n_tiles), leading_dim=2, divisibility=1
        )
    elif colvec_reduce_ndim == 2:
        mColVecReduce = fake_tensor(
            colvec_reduce_dtype, (m, n_tiles), leading_dim=1, divisibility=1
        )

    epi_args = GemmCls.EpilogueArguments(
        mAuxOut,
        act_fn,
        mColVecBroadcast=mColVec,
        mColVecReduce=mColVecReduce,
    )

    def _set_implicit_dtype(gemm_obj):
        gemm_obj.implicit_dtype = implicit_dtype

    scheduler_args = make_fake_scheduler_args(is_dynamic_persistent, False, l)
    varlen_args = make_fake_varlen_args(varlen_m, False, gather_A, m if varlen_m else None)

    # Scales: same convention as `_compile_mxfp8_gemm_act_sm90` (M-innermost SFA,
    # gmem-resident SFB); A here is dOut, B is the backward-oriented weight.
    sf_k_sym = cute.sym_int()
    n_blocks_sym = cute.sym_int()
    if varlen_m:
        padded_m_sym = cute.sym_int()
        fake_sfa = fake_tensor(
            cutlass.Float32, (padded_m_sym, sf_k_sym), leading_dim=0, divisibility=1
        )
    else:
        fake_sfa = fake_tensor(cutlass.Float32, (m, sf_k_sym, l), leading_dim=0, divisibility=1)
    fake_sfb = fake_tensor(
        cutlass.Float32, (l, n_blocks_sym, sf_k_sym), leading_dim=2, divisibility=1
    )
    return compile_gemm_kernel(
        partial(GemmCls, sf_vec_size=_SF_VEC_SIZE_SM90, weight_n_block=_WEIGHT_BLOCK_N_SM90),
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        gather_A,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        post_init=_set_implicit_dtype,
        mSFA=fake_sfa,
        mSFB=fake_sfb,
        use_tma_gather=use_tma_gather,
    )


def mxfp8_gemm_dgated_dispatch_sm90(
    A: Tensor,  # (l, m, k) dOut, K-contig fp8
    B: Tensor,  # (l, n, k) backward-oriented weight, K-contig fp8
    A_scale: Tensor,
    B_scale: Tensor,
    D: Tensor,  # (l, m, 2n) or (total_m, 2n) if varlen_m — dh, bf16 (viewed as f32)
    C: Tensor,  # same shape as D — h (PreAct), bf16 (viewed as f32)
    PostAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m — a_prime = colvec_scale * a
    tile_count_semaphore: Optional[Tensor],
    activation: str,
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    tile_K: int | None = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    colvec_scale: Optional[Tensor] = None,
    colvec_reduce: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    use_tma_gather: bool = False,
) -> None:
    varlen_m = cu_seqlens_m is not None
    gather_A = A_idx is not None

    implicit_dtype = torch2cute_dtype_map[D.dtype]
    assert D.element_size() == 2, "D (dh) dtype must be fp16 or bf16"
    assert C.element_size() == 2, "C (PreAct/h) dtype must be fp16 or bf16"
    D = D.view(torch.float32)
    C = C.view(torch.float32)

    A_p = perm3d_single(A, varlen_m)
    B_p = perm3d_single(B)
    D_p = perm3d_single(D, varlen_m)
    C_p = perm3d_single(C, varlen_m)
    PostAct_p = perm3d_single(PostAct, varlen_m)

    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n")
    c_major = get_major(C_p, "m", "n")
    postact_major = get_major(PostAct_p, "m", "n")

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype]
    c_dtype = torch2cute_dtype_map[C.dtype]
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    colvec_scale_ndim = colvec_scale.ndim if colvec_scale is not None else 0
    colvec_reduce_ndim = colvec_reduce.ndim if colvec_reduce is not None else 0

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] == 9, "mxfp8_gemm_dgated_dispatch_sm90 requires SM90 (Hopper)"

    if not GemmActSm90.is_valid_dtypes(
        a_dtype, b_dtype, cutlass.Float32, d_dtype, a_major, b_major
    ):
        raise ValueError(
            f"unsupported SM90 mxfp8 config: a_dtype={a_dtype}, b_dtype={b_dtype}, "
            f"d_dtype={d_dtype}, a_major={a_major}, b_major={b_major} "
            f"(SM90 fp8 requires K-major A and B)"
        )

    compiled_fn = _compile_mxfp8_gemm_dgated_sm90(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        postact_dtype,
        implicit_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        postact_major,
        (tile_M, tile_N, tile_K) if tile_K is not None else (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        activation,
        torch2cute_dtype_map[colvec_scale.dtype] if colvec_scale is not None else None,
        colvec_scale_ndim,
        torch2cute_dtype_map[colvec_reduce.dtype] if colvec_reduce is not None else None,
        colvec_reduce_ndim,
        varlen_m,
        gather_A,
        device_capacity,
        use_tma_gather=use_tma_gather,
    )

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    epi_args = GemmDGatedMixin.EpilogueArguments(
        PostAct_p,
        None,  # act_bwd_fn is Constexpr, baked at compile time
        mColVecBroadcast=colvec_scale,
        mColVecReduce=colvec_reduce,
        rounding_mode=None,
        sr_seed=None,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters, max_swizzle_size, tile_count_semaphore
    )
    varlen_args = make_varlen_args(cu_seqlens_m, None, A_idx)

    if varlen_m:
        sfa_sm90 = A_scale
    else:
        sfa_sm90 = A_scale.permute(1, 2, 0)  # (m, sf_k, l), M innermost
    sfb_sm90 = B_scale.contiguous()  # (l, n_blocks, sf_k)
    compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, sfa_sm90, sfb_sm90, None)


def mxfp8_gemm_dgated_tuned_sm90(
    A: Tensor,  # dOut: (M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A
    B: Tensor,  # backward-oriented weight: (K, N) or (L, K, N)
    A_scale: Tensor,
    B_scale: Tensor,
    PreAct: Tensor,  # h: (M, 2N) or (total_M, 2N) if varlen_m
    dx_out: Tensor,  # dh: same shape as PreAct
    postact_out: Tensor,  # a_prime = colvec_scale * a: (M, N) or (total_M, N) if varlen_m
    colvec_scale: Optional[Tensor] = None,  # (M,) or (total_M,) if varlen_m
    activation: GatedActivation = "swiglu",
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    config: Optional[GemmConfig] = None,
) -> Optional[Tensor]:
    if config is None:
        config = default_config(A.device)
    varlen_m = cu_seqlens_m is not None
    assert not config.swap_ab, "mxfp8_gemm_dgated_tuned_sm90 does not support swap_ab"
    og_ndim_2 = A.ndim == 2 and not varlen_m
    if A.ndim == 2 and not varlen_m:
        A = A.unsqueeze(0)
        A_scale = A_scale.unsqueeze(0)
    B, B_scale = B.mT, B_scale.mT  # (N, K) or (L, N, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)
        B_scale = B_scale.unsqueeze(0)
    if PreAct.ndim == 2 and not varlen_m:
        PreAct = PreAct.unsqueeze(0)
    D = dx_out.unsqueeze(0) if (dx_out.ndim == 2 and not varlen_m) else dx_out
    PostAct = postact_out.unsqueeze(0) if (postact_out.ndim == 2 and not varlen_m) else postact_out
    if colvec_scale is not None and colvec_scale.ndim == 1 and not varlen_m:
        colvec_scale = colvec_scale.unsqueeze(0)

    if colvec_reduce:
        tile_n = config.tile_n
        shape_n = (B.shape[-2] + tile_n - 1) // tile_n
        if varlen_m:
            total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
            colvec_shape = (total_m, shape_n)
        else:
            colvec_shape = (A.shape[0], A.shape[-2], shape_n)
        colvec_reduce_partial = torch.empty(colvec_shape, dtype=torch.float32, device=A.device)
    else:
        colvec_reduce_partial = None

    dynamic_scheduler = config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    mxfp8_gemm_dgated_dispatch_sm90(
        A,
        B,
        A_scale,
        B_scale,
        D,
        PreAct,
        PostAct,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        colvec_scale=colvec_scale,
        colvec_reduce=colvec_reduce_partial,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        use_tma_gather=config.use_tma_gather,
    )
    if colvec_reduce:
        colvec_reduce_final = colvec_reduce_partial.sum(dim=-1)
        if og_ndim_2:
            colvec_reduce_final = colvec_reduce_final.squeeze(0)
        return colvec_reduce_final
    return None


def mxfp8_gemm_dgated_sm90(
    A: Tensor,  # dOut, fp8
    B: Tensor,  # backward-oriented weight, fp8, (K, N) or (L, K, N)
    A_scale: Tensor,
    B_scale: Tensor,
    PreAct: Tensor,  # h, bf16, (M, 2N) or (total_M, 2N) if varlen_m
    dx_out: Optional[Tensor] = None,  # dh, bf16; allocated if None
    postact_out: Optional[Tensor] = None,  # a_prime, bf16; allocated if None
    colvec_scale: Optional[Tensor] = None,
    activation: GatedActivation = "swiglu",
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    config: Optional[GemmConfig] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """SwiGLU/GeGLU-backward GEMM, MXFP8 blockscaled A/B (SM90). Computes, in one
    fused kernel: `raw_da = A @ B`, `da = colvec_scale * raw_da`,
    `(dh, a_prime) = dgate_bwd(PreAct, da)` (`a_prime` additionally scaled by
    `colvec_scale`), and optionally `colvec_reduce[m] = sum_n PreAct_act(m,n) * raw_da(m,n)`
    (the `dout . y` term needed for the router-score gradient).
    """
    out_dtype = PreAct.dtype if out_dtype is None else out_dtype
    postact_dtype = PreAct.dtype if postact_dtype is None else postact_dtype
    if config is None:
        config = default_config(A.device)
    varlen_m = cu_seqlens_m is not None
    if varlen_m:
        total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
        out_shape = (total_m, B.shape[-1] * 2)
    elif A.ndim == 2:
        out_shape = (A.shape[0], B.shape[-1] * 2)
    else:
        out_shape = (A.shape[0], A.shape[-2], B.shape[-1] * 2)
    postact_shape = (*out_shape[:-1], out_shape[-1] // 2)
    if dx_out is None:
        dx_out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    if postact_out is None:
        postact_out = torch.empty(postact_shape, dtype=postact_dtype, device=A.device)
    colvec_reduce_final = mxfp8_gemm_dgated_tuned_sm90(
        A,
        B,
        A_scale,
        B_scale,
        PreAct,
        dx_out,
        postact_out,
        colvec_scale=colvec_scale,
        activation=activation,
        colvec_reduce=colvec_reduce,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        config=config,
    )
    return dx_out, postact_out, colvec_reduce_final


# ---------------------------------------------------------------------------
# Public entry point + weight quantization
# ---------------------------------------------------------------------------


def mxfp8_gemm_act(*args, **kwargs) -> Tuple[Optional[Tensor], Tensor]:
    """GEMM + (optionally gated) activation (SM90 only). See `mxfp8_gemm_act_sm90`."""
    cap = torch.cuda.get_device_capability(torch.cuda.current_device())[0]
    if cap == 9:
        return mxfp8_gemm_act_sm90(*args, **kwargs)
    raise NotImplementedError(f"mxfp8_gemm_act: sm_{cap}0 not supported (SM90 only)")


@triton.jit
def _quantize_weight_sm90_kernel(
    w_ptr,  # (E, N, K) bf16/f32, row-major
    q_ptr,  # (E, N, K), or (E, K, N) if TRANSPOSE, float8_e4m3fn
    sc_ptr,  # (E, N // 128, K // 128), or (E, K // 128, N // 128) if TRANSPOSE, f32
    N,
    K,
    nbn,  # N // 128
    nbk,  # K // 128
    BLOCK: tl.constexpr,  # 128 (block_rows == block_cols == sf_vec == weight_n_block)
    TRANSPOSE: tl.constexpr,
):
    # One 128x128 MXFP8 block per program. Bit-exact with quack.mx_utils.to_mx_2d
    # (FLOOR-mode e8m0 scale): amax over the tile -> power-of-2 dequant scale -> quantize.
    #
    # All offset arithmetic below is int64: `e * N * K` (and the analogous scale
    # offset) overflows int32 once E*N*K exceeds ~2.1B — e.g. E=512, N=2048, K=4096
    # already does — silently wrapping into an out-of-bounds pointer and faulting.
    e = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    pid_k = tl.program_id(2).to(tl.int64)
    N = N.to(tl.int64)
    K = K.to(tl.int64)
    nbn = nbn.to(tl.int64)
    nbk = nbk.to(tl.int64)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    offs_k = pid_k * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    w_off = e * N * K + offs_n[:, None] * K + offs_k[None, :]
    tile = tl.load(w_ptr + w_off).to(tl.float32)
    amax = tl.max(tl.abs(tile))
    # Per-block power-of-2 quant scale: FP8_MAX/amax truncated to a power of two by zeroing
    # the mantissa (& 0xFF800000). Quantize with that same pow2 scale and store its reciprocal
    # as the dequant scale, so q * stored_scale == tile up to fp8 rounding (the SM90 GEMM applies
    # the scale as a plain f32 multiply). amax==0 -> scale 1 (all-zero tile stays zero).
    scale = tl.where(amax == 0, 1.0, 448.0 / amax)
    scale = (scale.to(tl.uint32, bitcast=True) & 0xFF800000).to(tl.float32, bitcast=True)
    q = (tile * scale).to(q_ptr.dtype.element_ty)
    if TRANSPOSE:
        # Transpose the 128x128 tile in registers and write it straight to its
        # transposed position — avoids materializing a separate (E, K, N)
        # `.contiguous()` copy of `w` before quantizing it.
        q_off = e * K * N + offs_k[:, None] * N + offs_n[None, :]
        tl.store(q_ptr + q_off, tl.trans(q))
        tl.store(sc_ptr + e * nbk * nbn + pid_k * nbn + pid_n, 1.0 / scale)
    else:
        tl.store(q_ptr + w_off, q)
        tl.store(sc_ptr + e * nbn * nbk + pid_n * nbk + pid_k, 1.0 / scale)


def quantize_weight_sm90(
    w: torch.Tensor, transpose: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """128x128 blockwise FP8 weight quantization in a single Triton kernel.

    Each 128x128 block gets one power-of-2 scale (FP8_MAX/amax, mantissa-truncated).
    `scale` is the *dequant* scale (reciprocal) the SM90 GEMM multiplies by:
    `w ≈ qdata * scale`. N and K must be divisible by 128.

    If `transpose=False` (default): returns (qdata (..., N, K), scale (..., N//128, K//128)).
    If `transpose=True`: quantizes `w.mT` (last two dims swapped) directly from `w`'s
    original memory layout, in the same kernel launch — avoiding a separate materializing
    `.mT.contiguous()` copy before quantizing. Returns (qdata (..., K, N), scale
    (..., K//128, N//128)), i.e. exactly `quantize_weight_sm90(w.mT)` but without the copy.
    """
    *batch, N, K = w.shape
    assert N % 128 == 0 and K % 128 == 0, f"N ({N}) and K ({K}) must be divisible by 128"
    w3d = w.reshape(-1, N, K).contiguous()
    E = w3d.shape[0]
    nbn, nbk = N // 128, K // 128
    if transpose:
        q = torch.empty(E, K, N, dtype=torch.float8_e4m3fn, device=w.device)
        sc = torch.empty(E, nbk, nbn, dtype=torch.float32, device=w.device)
    else:
        q = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=w.device)
        sc = torch.empty(E, nbn, nbk, dtype=torch.float32, device=w.device)
    _quantize_weight_sm90_kernel[(E, nbn, nbk)](
        w3d, q, sc, N, K, nbn, nbk, BLOCK=128, TRANSPOSE=transpose
    )
    if transpose:
        return q.reshape(*batch, K, N), sc.reshape(*batch, K // 128, N // 128)
    return q.reshape(*batch, N, K), sc.reshape(*batch, N // 128, K // 128)
