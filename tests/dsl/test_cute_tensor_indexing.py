# Copyright (c) 2025, Tri Dao.

import pytest

import cutlass.cute as cute
from cutlass._mlir import ir

import cutlass.cute.tensor as cute_tensor

from quack.dsl import cute_tensor_indexing
from quack.dsl.cute_tensor_indexing import _canonicalize_cute_tensor_index

# These tests only exercise pure-Python canonicalization, monkey-patch installation,
# and MLIR-builder construction (no kernel compile, no GPU). Under --compile-only
# they would contribute nothing to the .o cache, so skip them entirely.
pytestmark = pytest.mark.compile_only_skip(
    "pure-Python / MLIR-builder tests; no kernel compile to cache"
)


def test_canonicalize_colon_and_ellipsis() -> None:
    assert _canonicalize_cute_tensor_index((1, slice(None), 2)) == (1, None, 2)
    assert _canonicalize_cute_tensor_index((Ellipsis, 2), (4, 5, 6)) == (None, None, 2)
    assert _canonicalize_cute_tensor_index((1, Ellipsis), (4, 5, 6)) == (1, None, None)
    assert _canonicalize_cute_tensor_index((1, Ellipsis, 2), (4, 5, 6, 7)) == (
        1,
        None,
        None,
        2,
    )


def test_canonicalize_hierarchical_colon_and_ellipsis() -> None:
    assert _canonicalize_cute_tensor_index(((1, slice(None)), 2)) == ((1, None), 2)
    assert _canonicalize_cute_tensor_index(((Ellipsis, 1), 2), ((4, 5), 6)) == (
        (None, 1),
        2,
    )
    assert _canonicalize_cute_tensor_index((Ellipsis, (1, slice(None), 2)), (4, (5, 6, 7))) == (
        None,
        (1, None, 2),
    )
    assert _canonicalize_cute_tensor_index(
        (((1, slice(None)), slice(None)), Ellipsis), (((2, 3), 4), 5, 6)
    ) == (((1, None), None), None, None)


def test_canonicalize_single_full_slice() -> None:
    assert _canonicalize_cute_tensor_index(slice(None)) is None
    assert _canonicalize_cute_tensor_index(Ellipsis, (4, 5, 6)) == (None, None, None)


def test_patched_getitem_reaches_real_cute_tensor() -> None:
    with ir.Context(), ir.Location.unknown():
        tensor = cute.make_identity_tensor((2, 3, 4))
        assert str(tensor[:, 1, 2]) == str(tensor[None, 1, 2])
        assert str(tensor[0, ..., 2]) == str(tensor[0, None, 2])

        hierarchical = cute.make_identity_tensor(((2, 3), 4))
        assert str(hierarchical[(..., 1), 2]) == str(hierarchical[(None, 1), 2])


def test_patch_is_idempotent() -> None:
    patched_getitem = cute_tensor._Tensor.__getitem__
    cute_tensor_indexing.patch_cute_tensor_indexing()
    assert cute_tensor._Tensor.__getitem__ is patched_getitem


def test_patched_setitem_forwards_canonicalized_index(monkeypatch) -> None:
    """End-to-end check that the patched __setitem__ canonicalizes `:`/`...`
    and forwards to CuTe's original. Guards against signature regressions in
    cute._Tensor.__setitem__ that would otherwise only surface in kernel code.
    """
    captured: list[tuple[object, object]] = []

    def spy(self, idx, data, *, loc=None, ip=None):
        captured.append((idx, data))

    monkeypatch.setattr(cute_tensor._Tensor, "_quack_original_setitem", spy, raising=False)
    monkeypatch.setattr(cute_tensor._Tensor, "__setitem__", cute_tensor_indexing._make_setitem(spy))

    with ir.Context(), ir.Location.unknown():
        tensor = cute.make_identity_tensor((2, 3, 4))
        tensor[:, 1, 2] = "sentinel"
        tensor[0, ..., 2] = "sentinel2"

    assert captured == [((None, 1, 2), "sentinel"), ((0, None, 2), "sentinel2")]


def test_rejects_unsupported_python_slices() -> None:
    with pytest.raises(ValueError, match="only supports full slices"):
        _canonicalize_cute_tensor_index((1, slice(0, 2)))
    with pytest.raises(ValueError, match="only supports full slices"):
        _canonicalize_cute_tensor_index(((1, slice(0, 2)), 3))


@pytest.mark.parametrize("idx", [(Ellipsis, Ellipsis), (0, Ellipsis, Ellipsis)])
def test_rejects_multiple_ellipses(idx) -> None:
    with pytest.raises(ValueError, match="at most one ellipsis"):
        _canonicalize_cute_tensor_index(idx, (2, 3, 4))


def test_ellipsis_error_message_includes_offending_index() -> None:
    with pytest.raises(ValueError, match=r"in \(Ellipsis, 2\)"):
        _canonicalize_cute_tensor_index((Ellipsis, 2))
    with pytest.raises(ValueError, match="expand ellipsis"):
        _canonicalize_cute_tensor_index(Ellipsis)
