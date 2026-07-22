"""Provider metadata for Quack's experimental FlyDSL RMSNorm."""

from importlib.util import find_spec

import torch

from quack.backends.protocol import Support

_SUPPORTED_ARCHES = frozenset({"gfx942", "gfx950"})
_SUPPORTED_INPUT_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})


def is_available() -> bool:
    return find_spec("flydsl") is not None


def supports(spec, target) -> Support:
    if target.vendor != "amd":
        return Support.no(f"FlyDSL RMSNorm requires an AMD target, got {target.vendor!r}")
    if target.arch != "unknown" and target.arch not in _SUPPORTED_ARCHES:
        return Support.no(
            "FlyDSL RMSNorm currently supports gfx942 and gfx950; "
            f"the input tensor targets {target.arch!r}"
        )
    if not spec.has_weight:
        return Support.no("FlyDSL RMSNorm requires an explicit weight")
    if spec.has_bias:
        return Support.no("FlyDSL RMSNorm does not yet support bias")
    if spec.has_residual:
        return Support.no("FlyDSL RMSNorm does not yet support residual fusion")
    if spec.prenorm:
        return Support.no("FlyDSL RMSNorm does not yet support prenorm outputs")
    if spec.per_head:
        return Support.no("FlyDSL RMSNorm does not yet support per-head weights")
    if spec.out_dtype is not None:
        return Support.no("FlyDSL RMSNorm does not yet support out_dtype")
    if spec.residual_dtype is not None:
        return Support.no("FlyDSL RMSNorm does not yet support residual_dtype")
    if spec.weight_offset != 0.0:
        return Support.no("FlyDSL RMSNorm does not yet support weight_offset")
    if spec.input_dtype not in _SUPPORTED_INPUT_DTYPES:
        return Support.no(
            "FlyDSL RMSNorm input dtype must be float16, bfloat16, or float32; "
            f"got {spec.input_dtype}"
        )
    if spec.weight_dtype != torch.float32:
        return Support.no(
            "FlyDSL RMSNorm requires an FP32 weight for FP16/BF16/FP32 inputs; "
            f"got {spec.weight_dtype}"
        )
    return Support.yes()


def load_op():
    from quack.rmsnorm_flydsl import rmsnorm

    return rmsnorm


__all__ = ["is_available", "load_op", "supports"]
