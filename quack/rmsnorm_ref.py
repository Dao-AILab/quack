# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

"""Pure-PyTorch RMSNorm references, shared by every backend.

These live outside ``quack.rmsnorm`` so ROCm can reach them: that module pulls
in the CuTe stack and fails to import without ``cuda.bindings``, while the
FlyDSL backend and the shared benchmark need the same reference to compare
against. ``quack.rmsnorm`` re-exports both names, so existing importers are
unaffected.
"""

import torch

__all__ = ["rmsnorm_bwd_ref", "rmsnorm_ref"]


def rmsnorm_ref(x, w=None, bias=None, residual=None, eps=1e-6, weight_offset=0.0):
    x_f32 = x.float()
    if residual is not None:
        residual_f32 = residual.float()
        x_f32 = x_f32 + residual_f32
    x_norm = x_f32 / (torch.sqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + eps))
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
