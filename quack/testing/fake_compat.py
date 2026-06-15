# Copyright (c) 2026, Tri Dao.
"""FakeTensor-compatibility shims for the compile-only test pass.

Under ``pytest --compile-only`` the plugin installs a session-wide
``CompileOnlyFakeTensorMode`` and pushes the ``_COMPILE_ONLY_DEPTH``
ContextVar. Several PyTorch operations and test idioms don't compose with
FakeTensor:

* :func:`torch.nonzero` has a data-dependent output shape; under
  ``FakeTensorMode`` it raises ``DynamicOutputShapeException``.
* :func:`torch.Tensor.data_ptr` on a FakeTensor returns ``0`` and emits a
  ``UserWarning: Accessing the data pointer of FakeTensor is deprecated`` that
  will become an error in a future PyTorch release.
* ``torch.profiler`` kernel counts assume real CUDA launches, of which there
  are none under compile-only.

The shims here let tests express the *intent* ("give me the indices of the
unpadded positions", "assert these two tensors alias") without sprinkling
``if is_compile_only(): ... else: ...`` branches across the suite. Each shim
takes the eager-mode path during phase 2 and a fake-friendly substitute
during phase 1, returning whatever the kernel-dispatch downstream needs to
hit the right compile-key signature.

If you find yourself reaching for ``is_compile_only()`` in a test body to gate
a single line, that's a sign the pattern belongs here.
"""

from __future__ import annotations

import torch

from quack.cache import is_compile_only


__all__ = [
    "fake_safe_nonzero",
    "assert_aliased",
]


def fake_safe_nonzero(mask: torch.Tensor) -> torch.Tensor:
    """Flatten-then-nonzero with a FakeTensor-safe fallback.

    Under compile-only mode ``torch.nonzero`` raises
    ``DynamicOutputShapeException``. The kernel signatures downstream don't
    care about the *values* of the indices; they only need a 1-D int64 tensor
    of the right length so the test setup can proceed to the kernel dispatch.
    We return ``torch.arange(mask.numel())`` in that case -- same dtype, same
    device, same rank, fully-padded semantically (every position is an index).

    Eager mode delegates to :func:`torch.nonzero` so phase 2 sees the real,
    padding-mask-respecting indices and the numerical assertions pass.

    Args:
        mask: a 1-D or higher-rank int/bool tensor; treated as the original
            ``mask.reshape(-1)`` input to ``nonzero``.

    Returns:
        Flattened ``int64`` indices tensor.
    """
    if is_compile_only():
        return torch.arange(mask.numel(), dtype=torch.int64, device=mask.device)
    return torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()


def assert_aliased(a: torch.Tensor, b: torch.Tensor) -> None:
    """Assert ``a`` and ``b`` share storage. No-op under compile-only mode.

    Aliasing checks via ``data_ptr() ==`` are inherently phase-2 invariants:
    they require real storage, and FakeTensor's ``.data_ptr()`` returns ``0``
    plus emits a deprecation warning. Wrapping the check in this helper makes
    the phase boundary syntactic at every call site::

        # Before
        if pre_allocate_out and not is_compile_only():
            assert out.data_ptr() == out_buf.data_ptr()

        # After
        if pre_allocate_out:
            assert_aliased(out, out_buf)

    Eager mode does the real comparison; compile-only returns silently so the
    rest of the test body can continue executing (and dispatching kernels) up
    to the next phase-2-only assertion.
    """
    if is_compile_only():
        return
    assert a.data_ptr() == b.data_ptr(), (
        f"expected tensors to alias, but data_ptr() differs: {a.data_ptr()!r} != {b.data_ptr()!r}"
    )
