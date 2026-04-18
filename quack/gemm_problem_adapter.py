# Copyright (c) 2025, Tri Dao.

from typing import NamedTuple, Optional
from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Int32, const_expr

from quack.cute_dsl_utils import ParamsBase, mlir_namedtuple


@mlir_namedtuple
class ProblemArguments(NamedTuple):
    pass


@dataclass
class ProblemParams(ParamsBase):
    pass


@mlir_namedtuple
class GroupedProblemArguments(NamedTuple):
    mProblemIndex: Optional[cute.Tensor] = None
    mProblemK: Optional[cute.Tensor] = None


@dataclass
class GroupedProblemParams(ParamsBase):
    problem_index: Optional[cute.Tensor] = None
    problem_k: Optional[cute.Tensor] = None

    @staticmethod
    def create(args: GroupedProblemArguments, *, loc=None, ip=None) -> "GroupedProblemParams":
        return GroupedProblemParams(problem_index=args.mProblemIndex, problem_k=args.mProblemK)


def default_problem_to_underlying_arguments(
    args: ProblemArguments | None, *, loc=None, ip=None
) -> ProblemParams:
    return ProblemParams()


def grouped_problem_to_underlying_arguments(
    args: GroupedProblemArguments | None, *, loc=None, ip=None
) -> GroupedProblemParams:
    if args is None:
        args = GroupedProblemArguments()
    return GroupedProblemParams.create(args, loc=loc, ip=ip)


class GroupedProblemAdapterMixin:
    ProblemArguments = GroupedProblemArguments
    ProblemParams = GroupedProblemParams

    def problem_to_underlying_arguments(
        self, args: GroupedProblemArguments | None = None, *, loc=None, ip=None
    ) -> GroupedProblemParams:
        return grouped_problem_to_underlying_arguments(args, loc=loc, ip=ip)

    @cute.jit
    def problem_get_problem_idx(self, params: GroupedProblemParams, work):
        return grouped_problem_idx(params, work)

    @cute.jit
    def problem_get_len_k(self, params: GroupedProblemParams, varlen_manager, work):
        return grouped_problem_len_k(params, work, varlen_manager)

    @cute.jit
    def problem_get_batch_A(self, params: GroupedProblemParams, mA_mkl, varlen_manager, work):
        return grouped_problem_batch_a(params, mA_mkl, varlen_manager, work)

    @cute.jit
    def problem_get_batch_B(self, params: GroupedProblemParams, mB_nkl, varlen_manager, work):
        return grouped_problem_batch_b(params, mB_nkl, varlen_manager, work)

    @cute.jit
    def problem_get_batch_epi(self, params: GroupedProblemParams, mX_mnl, varlen_manager, work):
        return grouped_problem_batch_epi(params, mX_mnl, varlen_manager, work)


@cute.jit
def default_problem_idx(params: ProblemParams, work, *, loc=None, ip=None) -> Int32:
    del params, loc, ip
    return work.problem_idx


@cute.jit
def grouped_problem_idx(params: GroupedProblemParams, work, *, loc=None, ip=None) -> Int32:
    del loc, ip
    if const_expr(params.problem_index is None):
        return work.problem_idx
    return params.problem_index[work.problem_idx]


@cute.jit
def default_problem_len_k(params: ProblemParams, work, varlen_manager, *, loc=None, ip=None) -> Int32:
    del params, loc, ip
    return varlen_manager.len_k(work.problem_idx)


@cute.jit
def grouped_problem_len_k(
    params: GroupedProblemParams, work, varlen_manager, *, loc=None, ip=None
) -> Int32:
    del loc, ip
    if const_expr(params.problem_k is not None):
        problem_idx = grouped_problem_idx(params, work)
        return params.problem_k[problem_idx]
    return varlen_manager.len_k(grouped_problem_idx(params, work))


@cute.jit
def default_problem_batch_a(params: ProblemParams, mA_mkl, varlen_manager, work, *, loc=None, ip=None):
    del params, loc, ip
    return varlen_manager.offset_batch_A(mA_mkl, work.problem_idx)


@cute.jit
def grouped_problem_batch_a(
    params: GroupedProblemParams, mA_mkl, varlen_manager, work, *, loc=None, ip=None
):
    del loc, ip
    return varlen_manager.offset_batch_A(mA_mkl, grouped_problem_idx(params, work))


@cute.jit
def default_problem_batch_b(params: ProblemParams, mB_nkl, varlen_manager, work, *, loc=None, ip=None):
    del params, loc, ip
    return varlen_manager.offset_batch_B(mB_nkl, work.problem_idx)


@cute.jit
def grouped_problem_batch_b(
    params: GroupedProblemParams, mB_nkl, varlen_manager, work, *, loc=None, ip=None
):
    del loc, ip
    return varlen_manager.offset_batch_B(mB_nkl, grouped_problem_idx(params, work))


@cute.jit
def default_problem_batch_epi(
    params: ProblemParams, mX_mnl, varlen_manager, work, *, loc=None, ip=None
):
    del params, loc, ip
    return None if const_expr(mX_mnl is None) else varlen_manager.offset_batch_epi(mX_mnl, work.problem_idx)


@cute.jit
def grouped_problem_batch_epi(
    params: GroupedProblemParams, mX_mnl, varlen_manager, work, *, loc=None, ip=None
):
    del loc, ip
    return (
        None
        if const_expr(mX_mnl is None)
        else varlen_manager.offset_batch_epi(mX_mnl, grouped_problem_idx(params, work))
    )
