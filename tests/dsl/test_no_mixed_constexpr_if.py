"""Guard against `if const_expr(S) and/or D:` tests in quack kernels.

The CuTe DSL only folds an `if` whose whole test is `const_expr(...)`; a mixed
test lowers to a dynamic if that traces every branch. Nest the static and
dynamic tests explicitly instead: `if const_expr(S): ... elif D: ...`.
"""

import ast
from pathlib import Path

import quack


def _is_const_expr(node: ast.expr) -> bool:
    func = getattr(node, "func", None)
    return isinstance(node, ast.Call) and (
        getattr(func, "attr", None) == "const_expr" or getattr(func, "id", None) == "const_expr"
    )


def test_no_mixed_constexpr_if_tests():
    root = Path(quack.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{node.lineno}"
        for path in sorted(root.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and _is_const_expr(node.test.values[0])
    ]
    assert not offenders, f"mixed const_expr if tests: {offenders}"
