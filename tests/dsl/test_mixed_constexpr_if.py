# Copyright (c) 2026, Tri Dao.
"""Tests for quack.dsl.mixed_constexpr_if.

Contract under test: ``if const_expr(S) and D:`` / ``if const_expr(S) or D:``
fold the static prefix at trace time and only materialize a dynamic if-region
in the branch where the dynamic remainder matters.  Three layers:

1. AST-shape tests on the rewriter itself (matching rules, and/or nesting,
   prefix folding, non-matches left alone).
2. Preprocessor-output tests: run the patched DSL preprocessor on real jit
   functions and assert where if-regions appear — this is the codegen
   guarantee ("the ungated path contains no dynamic if").
3. End-to-end kernels on GPU checking numeric behavior for every
   (static, dynamic) combination, including elif positions.
"""

import ast

import pytest
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.base_dsl.ast_preprocessor import DSLPreprocessor
from cutlass.cute.runtime import from_dlpack

from quack.dsl.mixed_constexpr_if import _rewrite_elif_chain, rewrite_mixed_constexpr_if


# ---------------------------------------------------------------------------
# 1. AST-shape tests on the rewriter
# ---------------------------------------------------------------------------


def parse_if(src: str) -> ast.If:
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.If)
    return node


def is_constexpr_call(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and ast.unparse(node.func).endswith("const_expr")


def is_constexpr_if(node: ast.stmt) -> bool:
    return isinstance(node, ast.If) and is_constexpr_call(node.test)


def test_rewrite_and_shape():
    node = parse_if("if const_expr(s) and d:\n    x = 1\nelse:\n    x = 2")
    out = rewrite_mixed_constexpr_if(node)
    assert out is not None and is_constexpr_if(out)
    assert ast.unparse(out.test) == "const_expr(s)"
    # then-branch: dynamic if with the original body and a copy of orelse
    (inner,) = out.body
    assert isinstance(inner, ast.If)
    assert ast.unparse(inner.test) == "d"
    assert ast.unparse(inner.body[0]) == "x = 1"
    assert ast.unparse(inner.orelse[0]) == "x = 2"
    # else-branch: the original orelse
    assert ast.unparse(out.orelse[0]) == "x = 2"


def test_rewrite_or_shape():
    node = parse_if("if const_expr(s) or d:\n    x = 1\nelse:\n    x = 2")
    out = rewrite_mixed_constexpr_if(node)
    assert out is not None and is_constexpr_if(out)
    # then-branch: the raw body, no dynamic if anywhere
    assert ast.unparse(out.body[0]) == "x = 1"
    (inner,) = out.orelse
    assert isinstance(inner, ast.If)
    assert ast.unparse(inner.test) == "d"
    assert ast.unparse(inner.body[0]) == "x = 1"
    assert ast.unparse(inner.orelse[0]) == "x = 2"


def test_rewrite_multiple_dynamic_operands_stay_joined():
    node = parse_if("if const_expr(s) and d1 and d2:\n    x = 1")
    out = rewrite_mixed_constexpr_if(node)
    (inner,) = out.body
    assert ast.unparse(inner.test) == "d1 and d2"


def test_rewrite_constexpr_prefix_folds():
    node = parse_if("if const_expr(s1) and cutlass.const_expr(s2) and d:\n    x = 1")
    out = rewrite_mixed_constexpr_if(node)
    assert ast.unparse(out.test) == "const_expr(s1 and s2)"
    (inner,) = out.body
    assert ast.unparse(inner.test) == "d"


def test_rewrite_all_static_folds_to_single_constexpr_if():
    node = parse_if("if const_expr(s1) or const_expr(s2):\n    x = 1\nelse:\n    x = 2")
    out = rewrite_mixed_constexpr_if(node)
    assert is_constexpr_if(out)
    assert ast.unparse(out.test) == "const_expr(s1 or s2)"
    assert ast.unparse(out.body[0]) == "x = 1"
    assert ast.unparse(out.orelse[0]) == "x = 2"


def test_rewrite_elif_chain_normalizes_copied_tail():
    node = parse_if(
        """\
if d0:
    x = 0
elif const_expr(s1) and d1:
    x = 1
elif const_expr(s2) and d2:
    x = 2
else:
    x = 3
"""
    )
    out = _rewrite_elif_chain(node)
    mixed_tests = [
        child
        for child in ast.walk(out)
        if isinstance(child, ast.BoolOp) and any(is_constexpr_call(value) for value in child.values)
    ]
    assert not mixed_tests


@pytest.mark.parametrize(
    "src",
    [
        "if d and const_expr(s):\n    x = 1",  # const_expr not leading
        "if const_expr(s):\n    x = 1",  # not a BoolOp: stock handling
        "if d1 and d2:\n    x = 1",  # no const_expr at all
        "if const_expr(s, t) and d:\n    x = 1",  # not a 1-arg const_expr call
        "if not const_expr(s) and d:\n    x = 1",  # negation not recognized
    ],
)
def test_rewrite_leaves_non_matches_alone(src):
    assert rewrite_mixed_constexpr_if(parse_if(src)) is None


# ---------------------------------------------------------------------------
# 2. Preprocessor-output tests (the codegen guarantee)
# ---------------------------------------------------------------------------


@cute.jit
def _preproc_or(out: cute.Tensor, d: Int32, flag: cutlass.Constexpr[bool]):
    val = Int32(2)
    if const_expr(flag) or d == 1:
        val = Int32(1)
    out[0] = val


@cute.jit
def _preproc_and(out: cute.Tensor, d: Int32, flag: cutlass.Constexpr[bool]):
    val = Int32(2)
    if const_expr(flag) and d == 1:
        val = Int32(1)
    else:
        val = Int32(3)
    out[0] = val


@cute.jit
def _preproc_elif(out: cute.Tensor, d: Int32, flag: cutlass.Constexpr[bool]):
    val = Int32(0)
    if d == 0:
        val = Int32(1)
    elif const_expr(flag) and d == 1:
        val = Int32(2)
    else:
        val = Int32(3)
    out[0] = val


def preprocess(fn) -> str:
    pre = DSLPreprocessor(["cutlass"])
    with pre.get_session() as session:
        tree = session.transform(fn, dict(fn.__globals__))
    return ast.unparse(tree)


def constexpr_branches(src: str):
    """Return (then_src, else_src) of the single ``if const_expr(flag):`` in src."""
    tree = ast.parse(src)
    hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and ast.unparse(node.test) == "const_expr(flag)"
    ]
    assert len(hits) == 1, src
    node = hits[0]
    return (
        "\n".join(ast.unparse(stmt) for stmt in node.body),
        "\n".join(ast.unparse(stmt) for stmt in node.orelse),
    )


def test_preprocessed_or_static_branch_has_no_if_region():
    src = preprocess(_preproc_or)
    then_src, else_src = constexpr_branches(src)
    # Static-true path is traced with zero dynamic ifs — codegen untouched.
    assert "if_region" not in then_src and "if_selector" not in then_src
    assert then_src == "val = Int32(1)"
    # The dynamic gate only exists on the static-false path.
    assert "if_selector" in else_src


def test_preprocessed_and_static_false_branch_has_no_if_region():
    src = preprocess(_preproc_and)
    then_src, else_src = constexpr_branches(src)
    assert "if_selector" in then_src
    assert else_src == "val = Int32(3)"


def test_preprocessed_elif_is_rewritten_despite_create_if_function_recursion():
    # elif recursion in the DSL bypasses visit_If; the patch normalizes the
    # chain up front.  The constexpr if must survive into the elif's region.
    src = preprocess(_preproc_elif)
    then_src, else_src = constexpr_branches(src)
    assert "if_selector" in then_src
    assert else_src == "val = Int32(3)"


# ---------------------------------------------------------------------------
# 3. End-to-end kernels
# ---------------------------------------------------------------------------


@cute.kernel
def _kernel_and(out: cute.Tensor, d: Int32, flag: cutlass.Constexpr[bool]):
    val = Int32(2)
    if const_expr(flag) and d == 1:
        val = Int32(1)
    else:
        val = Int32(3)
    out[0] = val


@cute.kernel
def _kernel_or(out: cute.Tensor, d: Int32, flag: cutlass.Constexpr[bool]):
    val = Int32(2)
    if const_expr(flag) or d == 1:
        val = Int32(1)
    else:
        val = Int32(3)
    out[0] = val


@cute.kernel
def _kernel_elif(out: cute.Tensor, d: Int32, flag: cutlass.Constexpr[bool]):
    val = Int32(0)
    if d == 0:
        val = Int32(1)
    elif const_expr(flag) and d == 1:
        val = Int32(2)
    else:
        val = Int32(3)
    out[0] = val


@cute.kernel
def _kernel_all_static(
    out: cute.Tensor, flag1: cutlass.Constexpr[bool], flag2: cutlass.Constexpr[bool]
):
    val = Int32(2)
    if const_expr(flag1) or const_expr(flag2):
        val = Int32(1)
    out[0] = val


@cute.kernel
def _kernel_trailing_constexpr(out: cute.Tensor, d: Int32, flag: cutlass.Constexpr[bool]):
    # const_expr not leading: falls back to a fully dynamic if (stock DSL),
    # which must still be numerically correct.
    val = Int32(2)
    if d == 1 or const_expr(flag):
        val = Int32(1)
    out[0] = val


@cute.jit
def _launch_scalar(
    kernel: cutlass.Constexpr, out: cute.Tensor, d: Int32, flag: cutlass.Constexpr[bool]
):
    kernel(out, d, flag).launch(grid=(1, 1, 1), block=(1, 1, 1))


@cute.jit
def _launch_two_flags(
    out: cute.Tensor, flag1: cutlass.Constexpr[bool], flag2: cutlass.Constexpr[bool]
):
    _kernel_all_static(out, flag1, flag2).launch(grid=(1, 1, 1), block=(1, 1, 1))


def run_kernel(kernel, d: int, flag: bool) -> int:
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    fn = cute.compile(_launch_scalar, kernel, from_dlpack(out), Int32(d), flag)
    fn(from_dlpack(out), Int32(d))
    torch.cuda.synchronize()
    return out.item()


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@requires_cuda
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("d", [0, 1])
def test_kernel_and(flag, d):
    expected = 1 if (flag and d == 1) else 3
    assert run_kernel(_kernel_and, d, flag) == expected


@requires_cuda
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("d", [0, 1])
def test_kernel_or(flag, d):
    expected = 1 if (flag or d == 1) else 3
    assert run_kernel(_kernel_or, d, flag) == expected


@requires_cuda
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("d", [0, 1, 2])
def test_kernel_elif(flag, d):
    if d == 0:
        expected = 1
    elif flag and d == 1:
        expected = 2
    else:
        expected = 3
    assert run_kernel(_kernel_elif, d, flag) == expected


@requires_cuda
@pytest.mark.parametrize("flag1", [False, True])
@pytest.mark.parametrize("flag2", [False, True])
def test_kernel_all_static(flag1, flag2):
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    fn = cute.compile(_launch_two_flags, from_dlpack(out), flag1, flag2)
    fn(from_dlpack(out))
    torch.cuda.synchronize()
    assert out.item() == (1 if (flag1 or flag2) else 2)


@requires_cuda
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("d", [0, 1])
def test_kernel_trailing_constexpr_fallback(flag, d):
    expected = 1 if (d == 1 or flag) else 2
    assert run_kernel(_kernel_trailing_constexpr, d, flag) == expected
