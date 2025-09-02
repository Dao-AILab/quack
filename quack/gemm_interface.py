# Copyright (c) 2025, Tri Dao
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from quack.gemm_config import GemmConfig, get_all_configs

from quack.autotuner import autotune, AutotuneConfig


def gemm_swiglu_out_ref(
    A: Tensor, B: Tensor, out: Optional[Tensor], store_preact: bool
) -> (Tensor, Tensor):
    preact = torch.mm(A, B)
    out_ = F.silu(preact[..., ::2]) * preact[..., 1::2]
    if out is not None:
        out.copy_(out_)
    else:
        out = out_
    if not store_preact:
        preact = None
    return out, preact


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs(epilogue=None)], key=["sm_carveout"]
)
def gemm_tuned(
    A: Tensor,
    B: Tensor,
    sm_carveout: int = 0,
    config: Optional[GemmConfig] = None,
) -> (Tensor, Optional[Tensor]):
    if config is None:
        config = GemmConfig(
            tile_m=256,
            tile_n=192,
            cluster_m=2,
            cluster_n=1,
            pingpong=False,
            raster_order=2,
            max_swizzle_size=1,
        )
    out = torch.ops.quack.gemm_impl.default(
        A if not config.swap_ab else B.T,
        B if not config.swap_ab else A.T,
        sm_carveout,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        not config.swap_ab,  # C_rowmajor
        config.pingpong,
        config.raster_order,
        config.max_swizzle_size,
    )
    return out if not config.swap_ab else out.T


@torch.library.custom_op("quack::gemm", mutates_args=(), device_types="cuda")
def gemm(A: Tensor, B: Tensor, sm_carveout: int = 0) -> Tensor:
    return gemm_tuned(A, B, sm_carveout)


@torch.library.register_fake("quack::gemm")
def gemm_ref(A: Tensor, B: Tensor, sm_carveout: int = 0) -> Tensor:
    return torch.mm(A, B)


@autotune(configs=[AutotuneConfig(config=c) for c in get_all_configs("add")])
def gemm_add_tuned(
    A: Tensor,
    B: Tensor,
    C: Tensor,
    config: Optional[GemmConfig] = None,
) -> (Tensor, Optional[Tensor]):
    if config is None:
        config = GemmConfig(
            tile_m=256,
            tile_n=192,
            cluster_m=2,
            cluster_n=1,
            pingpong=False,
            raster_order=2,
            max_swizzle_size=1,
        )
    out = torch.ops.quack.gemm_add_impl.default(
        A if not config.swap_ab else B.T,
        B if not config.swap_ab else A.T,
        C if not config.swap_ab else C.T,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.pingpong,
        config.raster_order,
        config.max_swizzle_size,
    )
    return out if not config.swap_ab else out.T


@torch.library.custom_op("quack::gemm_add", mutates_args=(), device_types="cuda")
def gemm_add(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return gemm_add_tuned(A, B, C)


@torch.library.register_fake("quack::gemm_add")
def gemm_add_ref(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return C + torch.mm(A, B)


@torch.library.custom_op("quack::gemm_add_t", mutates_args=(), device_types="cuda")
def gemm_t_add(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return gemm_add_tuned(A, B.T, C)


@torch.library.register_fake("quack::gemm_add_t")
def gemm_t_add_ref(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return gemm_add_ref(A, B.T, C)


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs("swiglu")], key=["store_preact"]
)
def gemm_swiglu_tuned(
    A: Tensor,
    B: Tensor,
    store_preact: bool = True,
    config: Optional[GemmConfig] = None,
) -> (Tensor, Optional[Tensor]):
    if config is None:
        config = GemmConfig(
            tile_m=256,
            tile_n=192,
            cluster_m=2,
            cluster_n=1,
            pingpong=False,
            raster_order=2,
            max_swizzle_size=1,
        )
    # out, preact
    return torch.ops.quack.gemm_swiglu_impl.default(
        A,
        B,
        store_preact,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.pingpong,
        config.raster_order,
        config.max_swizzle_size,
    )


# Specifying the schema manually here since torch.library._infer_schema doesn't work when return
# type is a tuple of Tensor
@torch.library.custom_op(
    "quack::gemm_swiglu",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor A, Tensor B, bool store_preact) -> (Tensor, Tensor)",
)
def gemm_swiglu(A: Tensor, B: Tensor, store_preact: bool = True) -> (Tensor, Tensor):
    return gemm_swiglu_tuned(A, B, store_preact=store_preact)


@torch.library.register_fake("quack::gemm_swiglu")
def gemm_swiglu_ref(A: Tensor, B: Tensor, store_preact: bool) -> (Tensor, Tensor):
    return gemm_swiglu_out_ref(A, B, None, store_preact)


# @torch.library.custom_op("quack::gemm_swiglu_t", mutates_args=(), device_types="cuda",
#                          schema="(Tensor A, Tensor B, bool store_preact) -> (Tensor, Tensor)")
# def gemm_swiglu_t(A: Tensor, B: Tensor, store_preact: bool = True) -> (Tensor, Tensor):
#     return gemm_swiglu_tuned(A, B.T, store_preact=store_preact)


# @torch.library.register_fake("quack::gemm_swiglu_t")
# def gemm_swiglu_t_ref(A: Tensor, B: Tensor, store_preact: bool) -> (Tensor, Tensor):
#     return gemm_swiglu_ref(A, B.T, store_preact)


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs("dswiglu")], key=["sm_carveout"]
)
def gemm_dswiglu_tuned(
    A: Tensor,
    B: Tensor,
    preact: Tensor,
    sm_carveout: int = 0,
    config: Optional[GemmConfig] = None,
) -> (Tensor, Tensor):
    if config is None:
        config = GemmConfig(
            tile_m=128,
            tile_n=192,
            cluster_m=2,
            cluster_n=1,
            pingpong=True,
            raster_order=2,
            max_swizzle_size=1,
        )
    out, postact = torch.ops.quack.gemm_dswiglu_impl.default(
        A if not config.swap_ab else B.T,
        B if not config.swap_ab else A.T,
        preact if not config.swap_ab else preact.T,
        sm_carveout,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        not config.swap_ab,  # C_rowmajor
        config.pingpong,
        config.raster_order,
        config.max_swizzle_size,
    )
    return (out, postact) if not config.swap_ab else (out.T, postact.T)


# Specifying the schema manually here since torch.library._infer_schema doesn't work when return
# type is a tuple of Tensor
@torch.library.custom_op(
    "quack::gemm_dswiglu",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor preact, int sm_carveout=0) -> (Tensor, Tensor)",
)
def gemm_dswiglu(A: Tensor, B: Tensor, preact: Tensor, sm_carveout: int = 0) -> (Tensor, Tensor):
    return gemm_dswiglu_tuned(A, B, preact, sm_carveout)


@torch.library.register_fake("quack::gemm_dswiglu")
def gemm_dswiglu_ref(
    A: Tensor, B: Tensor, preact: Tensor, sm_carveout: int = 0
) -> (Tensor, Tensor):
    # A: (M, K), B: (K, N), preact: (M, 2 * N)
    dout = torch.mm(A, B)
    p0, p1 = preact[..., ::2], preact[..., 1::2]
    sigmoid = torch.sigmoid(p0)
    silu = F.silu(p0)
    postact = silu * p1
    d0 = sigmoid * (1 + p0 * (1 - sigmoid)) * p1 * dout
    d1 = F.silu(p0) * dout
    out = torch.stack([d0, d1], dim=-1).reshape(d0.shape[:-1] + (2 * d0.shape[-1],))
    return out, postact


@autotune(configs=[AutotuneConfig(config=c) for c in get_all_configs("lse")])
def gemm_lse_tuned(
    A: Tensor,
    B: Tensor,
    softcap: float = 0.0,
    config: Optional[GemmConfig] = None,
) -> (Tensor, Tensor):
    if config is None:
        config = GemmConfig(
            tile_m=256,
            tile_n=192,
            cluster_m=2,
            cluster_n=1,
            pingpong=False,
            raster_order=2,
            max_swizzle_size=1,
        )
    out, lse_partial = torch.ops.quack.gemm_lse_impl.default(
        A,
        B,
        None,  # bias
        softcap,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.pingpong,
        config.raster_order,
        config.max_swizzle_size,
    )
    lse = torch.logsumexp(lse_partial, dim=-1)
    return out, lse


@torch.library.custom_op(
    "quack::gemm_lse",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor A, Tensor B, float softcap=0.0) -> (Tensor, Tensor)",
)
def gemm_lse(A: Tensor, B: Tensor, softcap: float = 0.0) -> (Tensor, Tensor):
    return gemm_lse_tuned(A, B, softcap)


@torch.library.register_fake("quack::gemm_lse")
def gemm_lse_ref(A: Tensor, B: Tensor, softcap: float = 0.0) -> (Tensor, Tensor):
    # A: (M, K), B: (K, N)
    out = torch.mm(A, B)
    if softcap > 0:
        out_fp32 = torch.tanh(out.to(torch.float32) / softcap) * softcap
        out = out_fp32.to(out.dtype)
    else:
        out_fp32 = out.to(torch.float32)
    lse = torch.logsumexp(out_fp32, dim=-1)
    return out, lse


@torch.library.custom_op(
    "quack::gemm_lse_t",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor A, Tensor B, float softcap=0.0) -> (Tensor, Tensor)",
)
def gemm_lse_t(A: Tensor, B: Tensor, softcap: float = 0.0) -> (Tensor, Tensor):
    return gemm_lse_tuned(A, B.T, softcap)


@torch.library.register_fake("quack::gemm_lse_t")
def gemm_lse_t_ref(A: Tensor, B: Tensor, softcap: float = 0.0) -> (Tensor, Tensor):
    return gemm_lse_ref(A, B.T, softcap)
