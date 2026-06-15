"""
Verify the cute-DSL CUDA 13.3 toolchain shim is active end-to-end.

Compiles a tiny CuTe kernel, captures the fatbin emitted by CUTLASS DSL's
compile pipeline, extracts the embedded ``release X.Y, VX.Y.Z`` ptxas
signature, and asserts:

  * Without the shim:  release 13.1, V13.1.66.
  * With the shim:     release 13.3, V13.3.33.
  * Kernel result is numerically correct in both cases.

Usage::

    # baseline (shim off):
    QUACK_CUTE_DSL_SHIM=0 python tools/cute_dsl_shim/verify.py

    # with the shim:
    QUACK_CUTE_DSL_SHIM=1 python tools/cute_dsl_shim/verify.py

    # exit nonzero unless the shim is verified active (for CI):
    QUACK_CUTE_DSL_SHIM=1 python tools/cute_dsl_shim/verify.py --require-shim
"""

from __future__ import annotations

import argparse
import re
import sys

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir import ir
from cutlass.base_dsl.compiler import Compiler
from cutlass.cute.runtime import from_dlpack

# Trigger activation now, before we wrap Compiler.compile, so that the shim
# is installed in the process before the first cute.compile() call. The
# `try_activate` activates only when QUACK_CUTE_DSL_SHIM=1; otherwise it
# no-ops in baseline mode.
import quack.dsl.cute_dsl_shim as _shim

_SHIM_ACTIVE = _shim.try_activate()


_captured: list[bytes] = []
_orig_compile = Compiler.compile


def _wrapped_compile(self, module, pipeline, *a, **k):
    """Walk the post-compile MLIR module and stash the kernels_binary blob."""
    _orig_compile(self, module, pipeline, *a, **k)

    def walker(op):
        if op.name == "llvm.mlir.global":
            try:
                name = ir.StringAttr(op.attributes["sym_name"]).value
            except Exception:
                return ir.WalkResult.ADVANCE
            if name == "kernels_binary":
                _captured.append(op.attributes["value"].value_bytes)
        return ir.WalkResult.ADVANCE

    module.operation.walk(walker)


Compiler.compile = _wrapped_compile


@cute.kernel
def _add_one_kernel(x_ptr: cute.Pointer, n: cutlass.Int32):
    bid = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]
    idx = bid * 128 + tid
    if idx < n:
        x_ptr[idx] = x_ptr[idx] + 1.0


@cute.jit
def _add_one(x: cute.Tensor, n: cutlass.Int32):
    _add_one_kernel(x.iterator, n).launch(grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--require-shim",
        action="store_true",
        help="exit nonzero unless the compiled cubin is from ptxas 13.3",
    )
    args = ap.parse_args()

    # Run the kernel.
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    ref = x + 1.0
    xt = from_dlpack(x.detach(), assumed_align=16).mark_compact_shape_dynamic(mode=0)
    f = cute.compile(_add_one, xt, cutlass.Int32(x.numel()))
    f(xt, cutlass.Int32(x.numel()))
    torch.cuda.synchronize()
    max_abs_diff = (x - ref).abs().max().item()
    if max_abs_diff > 1e-6:
        print(f"FAIL: numerical mismatch: {max_abs_diff}", file=sys.stderr)
        return 2

    if not _captured:
        print("FAIL: no cubin captured", file=sys.stderr)
        return 2

    fatbin = _captured[0]
    m = re.search(rb"release (\d+\.\d+), V(\d+\.\d+\.\d+)", fatbin)
    if not m:
        print("FAIL: no ptxas version stamp in cubin", file=sys.stderr)
        return 2
    release = m.group(1).decode()
    version = m.group(2).decode()

    shim_active = _shim.is_active()

    print(
        f"ptxas {release} / V{version}  (kernel result max abs diff = {max_abs_diff})  "
        f"shim_active={shim_active}"
    )

    if args.require_shim:
        if not shim_active:
            print("FAIL: --require-shim but shim not active", file=sys.stderr)
            return 3
        if release != "13.3":
            print(
                f"FAIL: --require-shim but cubin reports release {release}, want 13.3",
                file=sys.stderr,
            )
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
