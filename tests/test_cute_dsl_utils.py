from quack import cute_dsl_utils


def test_patched_converter_forwards_constexpr_keyword(monkeypatch):
    calls = []
    expected = object()

    def original(arg, arg_name, arg_type, ctx, *, is_constexpr=False):
        calls.append((arg, arg_name, arg_type, ctx, is_constexpr))
        return expected

    monkeypatch.setattr(cute_dsl_utils, "_original_convert_single_arg", original)
    arg, arg_type, ctx = object(), object(), object()

    actual = cute_dsl_utils._patched_convert_single_arg(
        arg,
        "field",
        arg_type,
        ctx,
        is_constexpr=True,
    )

    assert actual is expected
    assert calls == [(arg, "field", arg_type, ctx, True)]
