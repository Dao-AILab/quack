# Copyright (c) 2025, Tri Dao.

"""Tests for NamedTuple-based arguments with TVM-FFI compilation."""

from enum import IntEnum
from typing import NamedTuple

import pytest
import torch

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

import quack.cache

# Ensure the Constexpr converter patch is loaded.
import quack.cute_dsl_utils  # noqa: F401
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple
from quack.varlen_utils import VarlenArguments


@cute.kernel
def _copy_if_present(mOut: cute.Tensor, mSrc: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < mOut.shape[0]:
        mOut[tidx] = mSrc[tidx]


@cute.jit
def copy_from_varlen(mOut: cute.Tensor, args: VarlenArguments):
    if const_expr(args.mCuSeqlensM is not None):
        _copy_if_present(mOut, args.mCuSeqlensM).launch(grid=(1, 1, 1), block=(128, 1, 1))


def _compile_copy_from_varlen():
    n = cute.sym_int()
    out_fake = fake_tensor(cute.Int32, (n,), divisibility=1)
    cu_seqlens_fake = fake_tensor(cute.Int32, (n,), divisibility=1)
    varlen_args = VarlenArguments(mCuSeqlensM=cu_seqlens_fake)
    return cute.compile(copy_from_varlen, out_fake, varlen_args, options="--enable-tvm-ffi")


@pytest.mark.parametrize("N", [8, 32, 64])
def test_varlen_namedtuple_tvm_ffi(N):
    """Compile a kernel taking VarlenArguments (NamedTuple) via TVM-FFI and run it."""
    compiled_fn = _compile_copy_from_varlen()
    cu_seqlens = torch.arange(N, dtype=torch.int32, device="cuda")
    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    compiled_fn(out, VarlenArguments(mCuSeqlensM=cu_seqlens))
    torch.testing.assert_close(out, cu_seqlens)


def test_varlen_construction():
    """Smoke test that VarlenArguments NamedTuple has the right interface."""
    # Default construction (all None)
    args = VarlenArguments()
    assert args.mCuSeqlensM is None
    assert args.mCuSeqlensK is None
    assert args.mAIdx is None
    assert hasattr(args, "_fields")
    assert args._fields == ("mCuSeqlensM", "mCuSeqlensK", "mAIdx")

    # Keyword construction
    t = torch.zeros(4, dtype=torch.int32, device="cuda")
    args2 = VarlenArguments(mCuSeqlensM=t, mCuSeqlensK=None, mAIdx=t)
    assert args2.mCuSeqlensM is t
    assert args2.mCuSeqlensK is None
    assert args2.mAIdx is t


class MyEnum(IntEnum):
    ADD = 0
    MUL = 1


@mlir_namedtuple
class MyArgs(NamedTuple):
    mOut: cute.Tensor
    mA: cute.Tensor
    op: cutlass.Constexpr[int] = 0


@cute.kernel
def _apply_op_add(mOut: cute.Tensor, mA: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < mOut.shape[0]:
        mOut[tidx] = mA[tidx] + 1


@cute.kernel
def _apply_op_mul(mOut: cute.Tensor, mA: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < mOut.shape[0]:
        mOut[tidx] = mA[tidx] * 2


@cute.jit
def apply_op(args: MyArgs):
    if const_expr(args.op == MyEnum.ADD):
        _apply_op_add(args.mOut, args.mA).launch(grid=(1, 1, 1), block=(128, 1, 1))
    else:
        _apply_op_mul(args.mOut, args.mA).launch(grid=(1, 1, 1), block=(128, 1, 1))


@pytest.mark.parametrize("op", [MyEnum.ADD, MyEnum.MUL])
def test_constexpr_in_namedtuple(op):
    """Constexpr fields are baked in at compile time, passed as None at call time."""
    n = cute.sym_int()
    out_fake = fake_tensor(cute.Float32, (n,), divisibility=1)
    a_fake = fake_tensor(cute.Float32, (n,), divisibility=1)

    compiled = cute.compile(
        apply_op,
        MyArgs(mOut=out_fake, mA=a_fake, op=int(op)),
        options="--enable-tvm-ffi",
    )

    a = torch.ones(32, dtype=torch.float32, device="cuda")
    out = torch.zeros(32, dtype=torch.float32, device="cuda")
    # At call time, constexpr fields must be None (baked into compiled fn)
    compiled(MyArgs(mOut=out, mA=a, op=None))

    expected = (a + 1) if op == MyEnum.ADD else (a * 2)
    torch.testing.assert_close(out, expected)
