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
import importlib.util
import inspect
import sys

import pytest
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.base_dsl.ast_preprocessor import DSLPreprocessor
from cutlass.cute.runtime import from_dlpack

from quack.dsl.mixed_constexpr_if import (
    OPT_IN_FLAG,
    _rewrite_elif_chain,
    rewrite_mixed_constexpr_if,
)

# This module is not part of the quack package, so opt in to the rewrite (#217).
__quack_mixed_constexpr_if__ = True


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
    # @cute.jit / @cute.kernel return a functools.wraps wrapper defined inside
    # CUTLASS, whose __globals__ are CUTLASS's.  The DSL preprocesses the
    # wrapped function with its own module globals, so do the same here.
    raw = inspect.unwrap(fn)
    pre = DSLPreprocessor(["cutlass"])
    with pre.get_session() as session:
        tree = session.transform(raw, dict(raw.__globals__))
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
# 2b. Scope (#217): importing quack must not change how other libraries'
#     kernels are lowered.  The rewrite applies to quack's own modules and to
#     modules that opt in with ``__quack_mixed_constexpr_if__ = True``.
# ---------------------------------------------------------------------------

_SCOPE_SRC = """
import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr


@cute.jit
def mixed_and(out: cute.Tensor, d: Int32, flag: cutlass.Constexpr[bool]):
    val = Int32(2)
    if const_expr(flag) and d == 1:
        val = Int32(1)
    else:
        val = Int32(3)
    out[0] = val


@cute.kernel
def kv_loader(
    out: cute.Tensor, kv_left: Int32, kv_right: Int32, MAY_BE_EMPTY: cutlass.Constexpr[bool]
):
    # Shape of cuDNN frontend's SM100 FP8 attention loader from #217.  Stock DSL
    # keeps kv_row_base branch-local; the rewrite made the loop carry it, None on
    # one path -> TYPE_UNSTABLE_JOIN.
    if const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
        pass
    else:
        kv_row_base = kv_left * 4
        out[0] = kv_row_base
    if const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
        pass
    else:
        for kv_loop in cutlass.range(kv_left, kv_right, 1):
            kv_row_base = kv_loop * 4
            out[1] = kv_row_base


@cute.jit
def launch_kv_loader(
    out: cute.Tensor, kv_left: Int32, kv_right: Int32, MAY_BE_EMPTY: cutlass.Constexpr[bool]
):
    kv_loader(out, kv_left, kv_right, MAY_BE_EMPTY).launch(grid=(1, 1, 1), block=(1, 1, 1))
"""


def load_scope_module(tmp_path, monkeypatch, module_name: str, extra_src: str = ""):
    """Import _SCOPE_SRC from a real file under ``module_name`` (source must be inspectable)."""
    path = tmp_path / f"{module_name.replace('.', '_')}.py"
    path.write_text(_SCOPE_SRC + extra_src)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def num_constexpr_flag_ifs(src: str) -> int:
    return sum(
        isinstance(node, ast.If) and ast.unparse(node.test) == "const_expr(flag)"
        for node in ast.walk(ast.parse(src))
    )


def loop_carried_names(src: str) -> list[str]:
    """Names every lowered ``cutlass.range`` loop carries (its ``write_args_names``)."""
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("loop_selector"):
            for kw in node.keywords:
                if kw.arg == "write_args_names":
                    names += ast.literal_eval(kw.value)
    return names


def test_issue_217_loop_does_not_carry_branch_local_name(tmp_path, monkeypatch):
    # The reported failure mode, checked at the preprocessor level (no GPU):
    # outside quack the loop must not carry kv_row_base.  On main it did.
    module = load_scope_module(tmp_path, monkeypatch, "third_party_kernels")
    carried = loop_carried_names(preprocess(module.kv_loader))
    assert carried, "expected a lowered cutlass.range loop"
    assert "kv_row_base" not in carried


def test_rewrite_skips_modules_outside_quack(tmp_path, monkeypatch):
    module = load_scope_module(tmp_path, monkeypatch, "third_party_kernels")
    src = preprocess(module.mixed_and)
    # Stock DSL lowering: the whole BoolOp test is one dynamic if-region, so no
    # `if const_expr(flag):` survives and the else branch is not duplicated.
    assert num_constexpr_flag_ifs(src) == 0
    assert "if_selector" in src


def test_rewrite_applies_to_opted_in_module(tmp_path, monkeypatch):
    module = load_scope_module(
        tmp_path, monkeypatch, "user_opt_in_kernels", f"\n{OPT_IN_FLAG} = True\n"
    )
    then_src, else_src = constexpr_branches(preprocess(module.mixed_and))
    assert "if_selector" in then_src
    assert else_src == "val = Int32(3)"


def test_rewrite_applies_to_quack_modules(tmp_path, monkeypatch):
    module = load_scope_module(tmp_path, monkeypatch, "quack._scope_test_kernels")
    then_src, else_src = constexpr_branches(preprocess(module.mixed_and))
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


@requires_cuda
@pytest.mark.parametrize(
    "kv_left,kv_right,expected", [(1, 3, [4, 8]), (3, 3, [0, 0]), (2, 4, [8, 12])]
)
def test_issue_217_third_party_kernel_compiles_after_import_quack(
    tmp_path, monkeypatch, kv_left, kv_right, expected
):
    # quack is imported by this test module.  On main this compile failed with
    # TYPE_UNSTABLE_JOIN for kv_row_base; outside quack it must lower as stock.
    module = load_scope_module(tmp_path, monkeypatch, "third_party_kernels")
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    fn = cute.compile(
        module.launch_kv_loader, from_dlpack(out), Int32(kv_left), Int32(kv_right), True
    )
    fn(from_dlpack(out), Int32(kv_left), Int32(kv_right))
    torch.cuda.synchronize()
    assert out.tolist() == expected
