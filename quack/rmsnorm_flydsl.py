"""Experimental FlyDSL RMSNorm entry point.

This module intentionally does not import FlyDSL at import time. The compiler
and Quack-owned kernels are loaded only inside opaque custom-op bodies.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from quack._torch_library_op import torch_library_op
from quack.backends.flydsl import FLYDSL_INSTALL_HINT
from quack.backends.flydsl.rmsnorm_config import (
    get_bwd_workspace_rows,
    select_bwd_config,
    select_fwd_config,
)
from quack.backends.registry import REGISTRY
from quack.backends.rmsnorm_spec import normalize_rmsnorm_spec
from quack.backends.target import Target, target_from_tensor, vendor_from_torch_build

_BUILD_VENDOR = vendor_from_torch_build()
_PROVIDER = REGISTRY.load_provider("rmsnorm", "amd")
_FLYDSL_AVAILABLE = _PROVIDER.is_available()


def _raise_if_flydsl_unavailable() -> None:
    if not _FLYDSL_AVAILABLE:
        raise ModuleNotFoundError(FLYDSL_INSTALL_HINT)


def _raise_if_unsupported(spec, target: Target) -> None:
    support = _PROVIDER.supports(spec, target)
    if not support:
        raise NotImplementedError(support.reason)


def _runtime_spec(x: Tensor, weight: Tensor, eps: float):
    return normalize_rmsnorm_spec(x, weight=weight, eps=eps)


def _validate_eps(eps: float) -> None:
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError(f"eps must be finite and nonnegative, got {eps}")


def _flydsl_compile_target() -> tuple[str, str]:
    try:
        from flydsl.compiler.backends import get_backend
    except ModuleNotFoundError as exc:
        if exc.name == "flydsl":
            raise ModuleNotFoundError(FLYDSL_INSTALL_HINT) from None
        raise
    target = get_backend().target
    return target.backend, target.arch.split(":", 1)[0]


def _raise_if_compile_target_mismatch(invocation_target: Target) -> None:
    backend, compile_arch = _flydsl_compile_target()
    if backend != "rocm":
        raise RuntimeError(
            "quack.rmsnorm_flydsl requires FlyDSL's ROCm compile backend; "
            f"the active backend is {backend!r}"
        )
    if compile_arch != invocation_target.arch:
        raise RuntimeError(
            "FlyDSL compile target does not match the invocation tensor: "
            f"FlyDSL will compile for {compile_arch!r}, but the tensor is on "
            f"{invocation_target.arch!r}. Set ARCH or FLYDSL_GPU_ARCH to the tensor "
            "architecture. Mixed-architecture execution is disabled until FlyDSL #878 lands."
        )


def _import_fwd_backend():
    try:
        from quack.backends.flydsl.rmsnorm_kernel import run_rmsnorm_fwd
    except ModuleNotFoundError as exc:
        if exc.name == "flydsl":
            raise ModuleNotFoundError(FLYDSL_INSTALL_HINT) from None
        raise
    return run_rmsnorm_fwd


def _import_bwd_backend():
    try:
        from quack.backends.flydsl.rmsnorm_bwd_kernel import run_rmsnorm_bwd
    except ModuleNotFoundError as exc:
        if exc.name == "flydsl":
            raise ModuleNotFoundError(FLYDSL_INSTALL_HINT) from None
        raise
    return run_rmsnorm_bwd


@torch_library_op(
    "quack::_rmsnorm_flydsl_fwd",
    mutates_args={"out", "rstd"},
    device_types="cuda",
    schema=("(Tensor x, Tensor weight, Tensor(a2!) out, Tensor(a3!) rstd, float eps) -> ()"),
)
def _rmsnorm_flydsl_fwd(
    x: Tensor,
    weight: Tensor,
    out: Tensor,
    rstd: Tensor,
    eps: float,
) -> None:
    """Real custom-op body; device discovery and FlyDSL compilation stay here."""

    _validate_eps(eps)
    target = target_from_tensor(x)
    _raise_if_unsupported(_runtime_spec(x, weight, eps), target)
    _raise_if_compile_target_mismatch(target)
    if x.numel() == 0:
        return
    props = torch.cuda.get_device_properties(target.device_index)
    config = select_fwd_config(
        x.shape[0],
        x.shape[1],
        x.dtype,
        weight.dtype,
        target,
        props.multi_processor_count,
    )
    _import_fwd_backend()(x, weight, out, rstd, eps, target, config)


@torch_library_op(
    "quack::_rmsnorm_flydsl_bwd",
    mutates_args={"dx", "dweight", "workspace"},
    device_types="cuda",
    schema=(
        "(Tensor x, Tensor weight, Tensor doutput, Tensor rstd, "
        "Tensor(a4!) dx, Tensor(a5!) dweight, Tensor(a6!) workspace) -> ()"
    ),
)
def _rmsnorm_flydsl_bwd(
    x: Tensor,
    weight: Tensor,
    doutput: Tensor,
    rstd: Tensor,
    dx: Tensor,
    dweight: Tensor,
    workspace: Tensor,
) -> None:
    """Real backward body with runtime atomic/two-stage path selection."""

    target = target_from_tensor(x)
    _raise_if_unsupported(_runtime_spec(x, weight, 0.0), target)
    _raise_if_compile_target_mismatch(target)
    if x.numel() == 0:
        return
    props = torch.cuda.get_device_properties(target.device_index)
    config = select_bwd_config(
        x.shape[0],
        x.shape[1],
        x.dtype,
        weight.dtype,
        target,
        props.multi_processor_count,
    )
    _import_bwd_backend()(
        x,
        weight,
        doutput,
        rstd,
        dx,
        dweight,
        workspace,
        target,
        config,
    )


class RMSNormFunction(torch.autograd.Function):
    """Autograd boundary for already-flattened contiguous tensors."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float):
        out = torch.empty_like(x)
        rstd = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        _rmsnorm_flydsl_fwd(x, weight, out, rstd, eps)
        ctx.save_for_backward(x, weight, rstd)
        return out

    @staticmethod
    def backward(ctx, doutput: Tensor):
        x, weight, rstd = ctx.saved_tensors
        doutput = doutput.contiguous()
        dx = torch.empty_like(x)
        dweight = torch.zeros_like(weight, dtype=torch.float32)
        m, n = x.shape
        workspace_rows = get_bwd_workspace_rows(m, n, x.dtype)
        workspace = torch.empty(
            workspace_rows * n,
            device=x.device,
            dtype=torch.float32,
        )
        _rmsnorm_flydsl_bwd(
            x,
            weight,
            doutput,
            rstd,
            dx,
            dweight,
            workspace,
        )
        return dx, dweight, None


def _contiguous_for_backend(tensor: Tensor, name: str) -> Tensor:
    if torch.compiler.is_compiling():
        return tensor.contiguous()
    if not tensor.is_contiguous():
        raise ValueError(f"FlyDSL RMSNorm requires contiguous {name}")
    return tensor


def rmsnorm(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    prenorm: bool = False,
    weight_offset: float = 0.0,
) -> Tensor:
    """Plain weighted RMSNorm on gfx942/gfx950 through FlyDSL.

    This explicit entry remains available alongside the public
    ``quack.rmsnorm`` backend dispatch.
    """

    if _BUILD_VENDOR != "amd":
        raise RuntimeError(
            "quack.rmsnorm_flydsl requires a ROCm PyTorch build; "
            f"detected build vendor {_BUILD_VENDOR!r}"
        )
    _raise_if_flydsl_unavailable()
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)!r}")
    if x.device.type != "cuda":
        raise ValueError(f"FlyDSL RMSNorm requires a HIP tensor, got {x.device}")
    if x.dim() == 0:
        raise ValueError("FlyDSL RMSNorm input must have at least one dimension")
    if x.shape[-1] == 0:
        raise ValueError("FlyDSL RMSNorm normalized dimension must be nonzero")

    spec = normalize_rmsnorm_spec(
        x,
        weight=weight,
        bias=bias,
        residual=residual,
        out_dtype=out_dtype,
        residual_dtype=residual_dtype,
        eps=eps,
        prenorm=prenorm,
        weight_offset=weight_offset,
    )
    _raise_if_unsupported(
        spec,
        Target(vendor=_BUILD_VENDOR, arch="unknown", device_index=-1),
    )
    if weight.dim() != 1:
        raise ValueError(f"FlyDSL RMSNorm weight must be 1D, got shape {tuple(weight.shape)}")
    if weight.shape[0] != x.shape[-1]:
        raise ValueError(
            f"x last dimension ({x.shape[-1]}) must match weight length ({weight.shape[0]})"
        )
    if weight.device != x.device:
        raise ValueError(f"x and weight must share a device, got {x.device} and {weight.device}")
    # Dynamo can represent a non-default scalar argument as SymFloat, but
    # ``math.isfinite(SymFloat)`` returns a Python bool that cannot enter an FX
    # graph. The eager check still provides the public diagnostic; compiled
    # execution passes the runtime epsilon through the opaque custom op.
    if not torch.compiler.is_compiling():
        _validate_eps(spec.eps)

    original_shape = x.shape
    n = original_shape[-1]
    x = _contiguous_for_backend(x, "input")
    weight = _contiguous_for_backend(weight, "weight")
    x_flat = x.reshape(-1, n)
    out_flat = RMSNormFunction.apply(x_flat, weight, spec.eps)
    return out_flat.reshape(original_shape)


__all__ = ["RMSNormFunction", "rmsnorm"]
