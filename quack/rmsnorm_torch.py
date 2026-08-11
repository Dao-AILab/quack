# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

"""Pure-PyTorch references for the RMSNorm/LayerNorm family, shared by every backend.

These live outside ``quack.rmsnorm`` so ROCm can reach them: that module pulls
in the CuTe stack and fails to import without ``cuda.bindings``, while the
FlyDSL backend and the shared benchmark need the same reference to compare
against. Keeping a single copy is what stops the two backends from drifting
onto subtly different baselines. ``quack.rmsnorm`` re-exports every name here,
so existing importers are unaffected.
"""

import torch
from torch import Tensor

__all__ = [
    "layernorm_mean_ref",
    "layernorm_ref",
    "layernorm_rstd_ref",
    "rmsnorm_bwd_ref",
    "rmsnorm_ref",
]


def rmsnorm_ref(x, w=None, bias=None, residual=None, eps=1e-6, weight_offset=0.0):
    # rsqrt, not a divide by sqrt: equivalent to within an ulp, but rstd is
    # what the kernels actually compute and return, and what rmsnorm_bwd_ref
    # consumes -- naming it here keeps the reference's dataflow aligned with
    # the kernel's.
    x_f32 = x.float()
    if residual is not None:
        residual_f32 = residual.float()
        x_f32 = x_f32 + residual_f32
    rstd = torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + eps)
    x_norm = x_f32 * rstd
    out = x_norm * (w.float() + weight_offset) if w is not None else x_norm
    if bias is not None:
        out = out + bias.float()
    if residual is None:
        return out.to(x.dtype)
    else:
        return out.to(x.dtype), x_f32.to(residual.dtype)


def rmsnorm_bwd_ref(x, w, dout, rstd, eps=1e-6, weight_offset=0.0):
    """Reference implementation for RMSNorm backward pass."""
    x_f32 = x.float()
    x_hat = x_f32 * rstd.unsqueeze(1)
    if w is not None:
        wdy = dout * (w.float() + weight_offset)
    else:
        wdy = dout
    c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)
    dx = (wdy - x_hat * c1) * rstd.unsqueeze(1)

    # dL/dW
    if w is not None:
        dw = (dout * x_hat).sum(dim=0)
        return dx.to(x.dtype), dw.to(w.dtype)
    else:
        return dx.to(x.dtype), None


def layernorm_ref(x: Tensor, w: Tensor, eps: float = 1e-6) -> Tensor:
    """Reference implementation for LayerNorm."""
    x_f32 = x.float()
    return torch.nn.functional.layer_norm(x_f32, w.shape, w, None, eps).to(x.dtype)


def layernorm_rstd_ref(x: torch.Tensor, eps: float = 1e-6):
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = ((x_f32 - mean) ** 2).mean(dim=-1)
    return 1.0 / torch.sqrt(var + eps)


def layernorm_mean_ref(x: torch.Tensor) -> torch.Tensor:
    return x.float().mean(dim=-1)
