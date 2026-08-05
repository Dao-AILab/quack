# Copyright (c) 2026, Tri Dao.

import pytest

from cutlass import Float16, Float32
import cutlass.cute as cute
import cutlass.cute.tensor as cutlass_cute_tensor

from quack.dsl import cute_tensor


def test_rmem_tensor_to_materializes_converted_fragment(monkeypatch) -> None:
    calls = []

    class Loaded:
        def to(self, dtype, *, loc=None, ip=None):
            calls.append(("loaded.to", dtype, loc, ip))
            return "converted"

    class Src:
        memspace = cute.AddressSpace.rmem

        def load(self, *, loc=None, ip=None):
            calls.append(("load", loc, ip))
            return Loaded()

    class Dst:
        def store(self, data, *, loc=None, ip=None):
            calls.append(("store", data, loc, ip))

    dst = Dst()

    def make_rmem_tensor_like(src, dtype, *, loc=None, ip=None):
        calls.append(("make_rmem_tensor_like", src, dtype, loc, ip))
        return dst

    monkeypatch.setattr(cute, "make_rmem_tensor_like", make_rmem_tensor_like)

    src = Src()
    result = cutlass_cute_tensor._Tensor.to(src, Float32)

    assert result is dst
    assert calls == [
        ("make_rmem_tensor_like", src, Float32, None, None),
        ("load", None, None),
        ("loaded.to", Float32, None, None),
        ("store", "converted", None, None),
    ]


def test_tensor_contiguous_materializes_compact_rmem_fragment(monkeypatch) -> None:
    calls = []

    class Src:
        shape = (2, 3)
        element_type = Float16

    class Dst:
        pass

    dst = Dst()

    def make_rmem_tensor(shape, dtype, *, loc=None, ip=None):
        calls.append(("make_rmem_tensor", shape, dtype, loc, ip))
        return dst

    def autovec_copy(src, dst_arg, *, loc=None, ip=None):
        calls.append(("autovec_copy", src, dst_arg, loc, ip))

    monkeypatch.setattr(cute, "make_rmem_tensor", make_rmem_tensor)
    monkeypatch.setattr(cute, "autovec_copy", autovec_copy)

    src = Src()
    result = cutlass_cute_tensor._Tensor.contiguous(src)

    assert result is dst
    assert calls == [
        ("make_rmem_tensor", (2, 3), Float16, None, None),
        ("autovec_copy", src, dst, None, None),
    ]


def test_tensor_clone_materializes_matching_rmem_fragment(monkeypatch) -> None:
    calls = []

    class Src:
        pass

    class Dst:
        pass

    dst = Dst()

    def make_rmem_tensor_like(src, *, loc=None, ip=None):
        calls.append(("make_rmem_tensor_like", src, loc, ip))
        return dst

    def autovec_copy(src, dst_arg, *, loc=None, ip=None):
        calls.append(("autovec_copy", src, dst_arg, loc, ip))

    monkeypatch.setattr(cute, "make_rmem_tensor_like", make_rmem_tensor_like)
    monkeypatch.setattr(cute, "autovec_copy", autovec_copy)

    src = Src()
    result = cutlass_cute_tensor._Tensor.clone(src)

    assert result is dst
    assert calls == [
        ("make_rmem_tensor_like", src, None, None),
        ("autovec_copy", src, dst, None, None),
    ]


def test_tensor_has_contiguous_and_clone_methods() -> None:
    assert hasattr(cutlass_cute_tensor._Tensor, "contiguous")
    assert hasattr(cutlass_cute_tensor._Tensor, "clone")


def test_rmem_tensor_to_rejects_non_rmem_tensor() -> None:
    class Src:
        memspace = cute.AddressSpace.gmem

    with pytest.raises(ValueError, match="only supported for rmem"):
        cutlass_cute_tensor._Tensor.to(Src(), Float32)


def test_patch_is_idempotent() -> None:
    patched_to = cutlass_cute_tensor._Tensor.to
    patched_clone = cutlass_cute_tensor._Tensor.clone
    patched_contiguous = cutlass_cute_tensor._Tensor.contiguous
    cute_tensor.patch_cute_tensor()
    assert cutlass_cute_tensor._Tensor.to is patched_to
    assert cutlass_cute_tensor._Tensor.clone is patched_clone
    assert cutlass_cute_tensor._Tensor.contiguous is patched_contiguous
