# Copyright (c) 2026, Tri Dao.
"""Per-operand transforms for GEMM mainloops (RS A-operand produces today;
TransformB smem rewrites later). Ported from the transformA branch onto the
interleaved RS mainloop's copy_block seam.

Transforms are a first-class axis of the generic GEMM host layer: pass any
handle (a transform mod: a DecodeFormat, an ``@a_transform`` mod,
``dropout_a(p)``; or a W4 format name) as ``transform_a=`` to :func:`quack.gemm_runtime.host.build_gemm_epi_plan` — or
compose with fn epilogues via ``EpiMod.__call__/plan/gemm(transform_a=...)``
— and it
rides the same jit/disk cache, async compile, and EpiOp argument machinery as
every epilogue variant (see quack.operand_transform.host for the W4
blob/strip geometry). ``quack.gemm_w4.gemm_w4a16`` is
thin sugar over that path.

Layering note: the fn frontend (a_transform) and the decode formats sit
ABOVE the kernel (they import quack.gemm_runtime.identity for the semantic
keyer and the mod base), while transform.py and kinds.py are
KERNEL-SIDE code imported BY GemmSm90. The frontend re-exports below are
lazy (PEP 562) so the kernel can import this package without a cycle.
"""

from quack.operand_transform.transform import (  # noqa: F401
    AuxKTileStrip,
    AuxOperandA,
    TransformA,
    TransformADropout,
    TransformAOperand,
    TransformAValue,
    TransformAW4,
)

_FRONTEND = ("ATransformMod", "FnFormat", "PackedInput", "a_transform", "dropout_a")
_HOST = ("transform_a_operand",)

__all__ = [
    "AuxKTileStrip",
    "AuxOperandA",
    "TransformA",
    "TransformADropout",
    "TransformAOperand",
    "TransformAValue",
    "TransformAW4",
    *_FRONTEND,
    *_HOST,
]


def __getattr__(name):
    if name in _FRONTEND:
        from quack.operand_transform import frontend

        return getattr(frontend, name)
    if name in _HOST:
        from quack.operand_transform import host

        return getattr(host, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
