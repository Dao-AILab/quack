# Copyright (c) 2025, Tri Dao
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from quack.gemm_interface import gemm, gemm_add_inplace, gemm_act, gemm_dact
from quack.gemm_interface import gemm_gated, gemm_dgated
from quack.gemm_interface import _apply_gated_activation, act_to_pytorch_fn_map


def _ensure_contiguous(t):
    """Ensure last-dim stride is 1. Under torch.compile use unconditional .contiguous()
    (dynamo can't inspect strides on fake tensors); otherwise check first to avoid copies.
    """
    if torch.compiler.is_compiling():
        return t.contiguous()
    return t if t.stride(-1) == 1 else t.contiguous()


def linear_fwd_convert_type(*tensors):
    autocast_dtype = torch.get_autocast_dtype("cuda")
    if torch.is_autocast_enabled():
        tensors = tuple(t.to(dtype=autocast_dtype) for t in tensors)
    return tensors


def linear_fwd_postprocess(ctx, x, weight, weight_og, needs_x_w_grad):
    needs_input_grad, needs_weight_grad = needs_x_w_grad
    if not needs_input_grad:
        weight, weight_og = None, None
    if not needs_weight_grad:
        x = None
    ctx.save_for_backward(x, weight, weight_og if ctx.fuse_grad_accum else None)


def linear_bwd_compute_input_grad(ctx, dout, weight, matmul_fn):
    if ctx.needs_input_grad[0]:
        assert weight is not None
        return matmul_fn(dout, weight)
    else:
        return None


def linear_bwd_compute_weight_grad(ctx, dout, x, weight_og, matmul_fn, matmul_inplace_fn):
    if ctx.needs_input_grad[1]:
        assert x is not None
        x = x.flatten(0, -2)
        # fuse_grad_accum is not compatible with torch.compile
        if not ctx.fuse_grad_accum or weight_og.grad is None or torch.compiler.is_compiling():
            dweight = matmul_fn(dout.T, x, out_dtype=ctx.weight_dtype)
        else:
            # print("Using fuse grad accum in Linear", dout.shape, x.shape, weight_og.grad.shape)
            matmul_inplace_fn(dout.T, x, weight_og.grad)
            dweight = weight_og.grad
            weight_og.grad = None  # So that pytorch doesn't add dweight to weight_og.grad again
    else:
        dweight = None
    return dweight


def _recompute_act_postact(preact, activation, activation_limit=None):
    """Recompute postact from preact using the activation function (no GEMM)."""
    return act_to_pytorch_fn_map[activation](preact)


def _recompute_gated_postact(preact, activation, activation_limit=None):
    """Recompute gated postact from interleaved preact (no GEMM)."""
    return _apply_gated_activation(
        activation, preact[..., ::2], preact[..., 1::2], activation_limit
    )


# --- Ops bundles: matmul function configurations ---
# Each ops class is a namespace holding the matmul functions for a specific variant
# (tuned/untuned, act/gated, etc.). Passed as a non-tensor arg to apply() and stored on ctx.


class _LinearOps:
    matmul_fwd_fn = gemm
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True)
    matmul_bwd_dw = partial(gemm, dynamic_scheduler=True)
    matmul_bwd_dw_inplace = partial(gemm_add_inplace, dynamic_scheduler=True)


class _LinearUntunedOps(_LinearOps):
    matmul_fwd_fn = partial(gemm, tuned=False)
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True, tuned=False)
    matmul_bwd_dw = partial(gemm, dynamic_scheduler=True, tuned=False)
    matmul_bwd_dw_inplace = partial(gemm_add_inplace, dynamic_scheduler=True, tuned=False)


class _LinearActOps(_LinearOps):
    matmul_fwd_fn = gemm_act


class _LinearActUntunedOps(_LinearUntunedOps):
    matmul_fwd_fn = partial(gemm_act, tuned=False)


class _LinearGatedOps(_LinearOps):
    matmul_fwd_fn = gemm_gated


class _LinearGatedUntunedOps:
    matmul_fwd_fn = partial(gemm_gated, tuned=False)
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True, tuned=False)
    matmul_bwd_dw = partial(gemm, dynamic_scheduler=True, tuned=False)
    matmul_bwd_dw_inplace = partial(gemm_add_inplace, dynamic_scheduler=True, tuned=False)


class _LinearGatedConcatOps(_LinearGatedOps):
    matmul_fwd_fn = partial(gemm_gated, concat_layout=("B", "bias"))
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True, concat_layout=("B",))
    matmul_bwd_dw = partial(gemm, dynamic_scheduler=True, concat_layout=("out",))
    matmul_bwd_dw_inplace = partial(
        gemm_add_inplace, dynamic_scheduler=True, concat_layout=("C", "out")
    )


class _LinearGatedConcatUntunedOps(_LinearGatedUntunedOps):
    matmul_fwd_fn = partial(gemm_gated, tuned=False, concat_layout=("B", "bias"))
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True, tuned=False, concat_layout=("B",))
    matmul_bwd_dw = partial(gemm, dynamic_scheduler=True, tuned=False, concat_layout=("out",))
    matmul_bwd_dw_inplace = partial(
        gemm_add_inplace, dynamic_scheduler=True, tuned=False, concat_layout=("C", "out")
    )


class _DActLinearOps(_LinearOps):
    matmul_bwd_dx = partial(gemm_dact, dynamic_scheduler=True)
    recompute_postact = staticmethod(_recompute_act_postact)


class _DActLinearUntunedOps(_LinearUntunedOps):
    matmul_bwd_dx = partial(gemm_dact, dynamic_scheduler=True, tuned=False)
    recompute_postact = staticmethod(_recompute_act_postact)


class _DGatedLinearOps(_LinearOps):
    matmul_bwd_dx = partial(gemm_dgated, dynamic_scheduler=True)
    recompute_postact = staticmethod(_recompute_gated_postact)


class _DGatedLinearUntunedOps(_LinearUntunedOps):
    matmul_bwd_dx = partial(gemm_dgated, dynamic_scheduler=True, tuned=False)
    recompute_postact = staticmethod(_recompute_gated_postact)


# --- Autograd Functions (all @staticmethod, torch.compile-compatible) ---


class LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, fuse_grad_accum, ops):
        """
        x: (..., in_features)
        weight: (out_features, in_features)
        bias: (out_features,) or None
        out: (..., out_features)
        """
        # Convert types while autocast is still enabled, then disable it for the body.
        x, weight = linear_fwd_convert_type(x, weight)
        with torch.amp.autocast("cuda", enabled=False):
            ctx.weight_dtype = weight.dtype
            ctx.fuse_grad_accum = fuse_grad_accum
            ctx.ops = ops
            weight_og = weight
            batch_shape = x.shape[:-1]
            x = x.flatten(0, -2)
            out = ops.matmul_fwd_fn(x, weight.T, bias=bias)
            linear_fwd_postprocess(
                ctx, x, weight, weight_og, needs_x_w_grad=ctx.needs_input_grad[:2]
            )
            ctx.bias_dtype = bias.dtype if bias is not None else None
            ctx.compute_dbias = bias is not None and ctx.needs_input_grad[2]
            return out.reshape(*batch_shape, out.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        """
        dout: (..., out_features)
        """
        with torch.amp.autocast("cuda", enabled=False):
            ops = ctx.ops
            x, weight, weight_og = ctx.saved_tensors  # weight_og is None if not ctx.fuse_grad_accum
            batch_shape = dout.shape[:-1]
            dout = _ensure_contiguous(dout.flatten(0, -2))
            dbias = dout.sum(0, dtype=ctx.bias_dtype) if ctx.compute_dbias else None
            dx = linear_bwd_compute_input_grad(ctx, dout, weight, ops.matmul_bwd_dx)
            dx = dx.reshape(*batch_shape, dx.shape[-1]) if dx is not None else None
            dweight = linear_bwd_compute_weight_grad(
                ctx, dout, x, weight_og, ops.matmul_bwd_dw, ops.matmul_bwd_dw_inplace
            )
            return dx, dweight, dbias, None, None


def linear_func(x, weight, bias=None, fuse_grad_accum=False, tuned=True):
    ops = _LinearOps if tuned else _LinearUntunedOps
    return LinearFunc.apply(x, weight, bias, fuse_grad_accum, ops)


class LinearActFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, weight, activation, bias, store_preact, fuse_grad_accum, ops, activation_limit
    ):
        """
        x: (..., in_features)
        weight: (out_features, in_features)
        bias: (out_features,) or None
        out: (..., out_features)
        Return both out and post-activation, but only out is differentiable.
        """
        x, weight = linear_fwd_convert_type(x, weight)
        with torch.amp.autocast("cuda", enabled=False):
            ctx.weight_dtype = weight.dtype
            ctx.fuse_grad_accum = fuse_grad_accum
            ctx.ops = ops
            weight_og = weight
            batch_shape = x.shape[:-1]
            x = x.flatten(0, -2)
            out, postact = ops.matmul_fwd_fn(
                x,
                weight.T,
                bias=bias,
                activation=activation,
                store_preact=store_preact,
                activation_limit=activation_limit,
            )
            linear_fwd_postprocess(
                ctx, x, weight, weight_og, needs_x_w_grad=ctx.needs_input_grad[:2]
            )
            if out is not None:
                out = out.reshape(*batch_shape, out.shape[-1])
            ctx.bias_dtype = bias.dtype if bias is not None else None
            ctx.compute_dbias = bias is not None and ctx.needs_input_grad[3]
            ctx.mark_non_differentiable(postact)
            ctx.set_materialize_grads(False)  # We don't want to materialize grads for postact
            return out, postact.reshape(*batch_shape, postact.shape[-1])

    @staticmethod
    def backward(ctx, dout, *args):
        with torch.amp.autocast("cuda", enabled=False):
            ops = ctx.ops
            x, weight, weight_og = ctx.saved_tensors
            batch_shape = dout.shape[:-1]
            dout = _ensure_contiguous(dout.flatten(0, -2))
            dbias = dout.sum(0, dtype=ctx.bias_dtype) if ctx.compute_dbias else None
            dx = linear_bwd_compute_input_grad(ctx, dout, weight, ops.matmul_bwd_dx)
            dx = dx.reshape(*batch_shape, dx.shape[-1]) if dx is not None else None
            dweight = linear_bwd_compute_weight_grad(
                ctx, dout, x, weight_og, ops.matmul_bwd_dw, ops.matmul_bwd_dw_inplace
            )
            return dx, dweight, None, dbias, None, None, None, None


def linear_act_func(
    x, weight, activation, bias=None, store_preact=True, fuse_grad_accum=False, tuned=True
):
    ops = _LinearActOps if tuned else _LinearActUntunedOps
    return LinearActFunc.apply(
        x, weight, activation, bias, store_preact, fuse_grad_accum, ops, None
    )


def linear_gated_func(
    x,
    weight,
    activation,
    bias=None,
    store_preact=True,
    fuse_grad_accum=False,
    tuned=True,
    concat_layout=False,
    activation_limit=None,
):
    if concat_layout:
        ops = _LinearGatedConcatOps if tuned else _LinearGatedConcatUntunedOps
    else:
        ops = _LinearGatedOps if tuned else _LinearGatedUntunedOps
    return LinearActFunc.apply(
        x,
        weight,
        activation,
        bias,
        store_preact,
        fuse_grad_accum,
        ops,
        activation_limit,
    )


class DActLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, preact, weight, x, activation, bias, fuse_grad_accum, ops, activation_limit):
        """
        x: (..., in_features)
        weight: (out_features, in_features)
        bias: (out_features,) or None
        out: (..., out_features)
        Takes in an extra preact argument which is the pre-activation, to be used in the backward pass.
        """
        x, weight = linear_fwd_convert_type(x, weight)
        with torch.amp.autocast("cuda", enabled=False):
            ctx.weight_dtype = weight.dtype
            ctx.fuse_grad_accum = fuse_grad_accum
            ctx.ops = ops
            weight_og = weight
            batch_shape = x.shape[:-1]
            x = x.flatten(0, -2)
            out = ops.matmul_fwd_fn(x, weight.T, bias=bias)
            # Store preact instead of x, we will recompute x (postact) in backward.
            # dpreact needs gemm_dact(dout, weight, preact) → needs both weight and preact.
            # dweight needs postact: if dpreact is also needed, postact comes from gemm_dact;
            # otherwise we can recompute postact = act(preact) cheaply without weight.
            need_preact = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
            need_weight = ctx.needs_input_grad[0]  # only gemm_dact needs weight
            linear_fwd_postprocess(
                ctx, preact, weight, weight_og, needs_x_w_grad=(need_weight, need_preact)
            )
            ctx.activation = activation
            ctx.activation_limit = activation_limit
            ctx.bias_dtype = bias.dtype if bias is not None else None
            ctx.compute_dbias = bias is not None and ctx.needs_input_grad[4]
            return out.reshape(*batch_shape, out.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        """
        dout: (..., out_features)
        """
        with torch.amp.autocast("cuda", enabled=False):
            ops = ctx.ops
            # weight_og is None if not ctx.fuse_grad_accum
            preact, weight, weight_og = ctx.saved_tensors
            batch_shape = dout.shape[:-1]
            dout = _ensure_contiguous(dout.flatten(0, -2))
            dbias = dout.sum(0, dtype=ctx.bias_dtype) if ctx.compute_dbias else None
            if ctx.needs_input_grad[0]:
                # Need dpreact: gemm_dact(dout, weight, preact) → (dpreact, postact)
                preact = preact.flatten(0, -2)
                assert weight is not None
                dpreact, x = ops.matmul_bwd_dx(
                    dout,
                    weight,
                    preact,
                    activation=ctx.activation,
                    activation_limit=ctx.activation_limit,
                )
            elif ctx.needs_input_grad[1]:
                # Only need dweight: recompute postact from preact cheaply (no GEMM needed)
                preact = preact.flatten(0, -2)
                x = ops.recompute_postact(preact, ctx.activation, ctx.activation_limit)
                dpreact = None
            else:
                dpreact, x = None, None
            dpreact = (
                dpreact.reshape(*batch_shape, dpreact.shape[-1]) if dpreact is not None else None
            )
            dweight = linear_bwd_compute_weight_grad(
                ctx, dout, x, weight_og, ops.matmul_bwd_dw, ops.matmul_bwd_dw_inplace
            )
            return dpreact, dweight, None, None, dbias, None, None, None


def act_linear_func(preact, weight, x, activation, bias=None, fuse_grad_accum=False, tuned=True):
    ops = _DActLinearOps if tuned else _DActLinearUntunedOps
    return DActLinearFunc.apply(preact, weight, x, activation, bias, fuse_grad_accum, ops, None)


def gated_linear_func(
    preact,
    weight,
    x,
    activation,
    bias=None,
    fuse_grad_accum=False,
    tuned=True,
    activation_limit=None,
):
    ops = _DGatedLinearOps if tuned else _DGatedLinearUntunedOps
    return DActLinearFunc.apply(
        preact, weight, x, activation, bias, fuse_grad_accum, ops, activation_limit
    )


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        fuse_grad_accum: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.fuse_grad_accum = fuse_grad_accum

    def forward(self, input: Tensor) -> Tensor:
        if input.is_cuda and self.in_features % 8 == 0 and self.out_features % 8 == 0:
            return linear_func(input, self.weight, self.bias, fuse_grad_accum=self.fuse_grad_accum)
        else:
            return F.linear(input, self.weight, self.bias)
