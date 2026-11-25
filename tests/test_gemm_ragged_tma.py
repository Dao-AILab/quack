"""Tests for the 2-extra-dim ragged TMA view (copy_utils.create_ragged_tensor_for_tma,
ptr_shift=False), which varlen GEMM loads go through.

Issue #243: with MAX_INT extents on the two extra dims, a TMA load whose box is partially
out of bounds in the ragged dim also touched ~128 KiB from the operand's base address on
Blackwell, faulting when the operand sat near the end of a mapping. The extents must stay
at what is actually indexed, and the wraparound must still resolve each sequence to its
own rows.
"""

import pytest
import torch

import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from quack import copy_utils


def _compile_probe(T, out, ragged_dim):
    @cute.kernel
    def probe_kernel(
        mT: cute.Tensor, view: cute.Tensor, mOut: cute.Tensor, offset: Int32, length: Int32
    ):
        rank = cute.rank(view)
        mOut[0] = cute.size(view, mode=[ragged_dim])
        mOut[1] = cute.size(view, mode=[rank - 2])
        mOut[2] = cute.size(view, mode=[rank - 1])
        seq = copy_utils.offset_ragged_tensor(view, offset, length, ragged_dim=ragged_dim)
        # Byte distance from T to the first row of the sequence, modulo 2^64.
        mOut[3] = seq.iterator.toint() - mT.iterator.toint()

    @cute.jit
    def probe(mT: cute.Tensor, mOut: cute.Tensor, offset: Int32, length: Int32):
        view = copy_utils.create_ragged_tensor_for_tma(mT, ragged_dim=ragged_dim)
        probe_kernel(mT, view, mOut, offset, length).launch(grid=(1, 1, 1), block=(1, 1, 1))

    return cute.compile(probe, T, out, Int32(0), Int32(0))


@pytest.mark.parametrize(
    "shape,ragged_dim",
    [((24, 16), 0), ((16, 24), 1), ((24, 8, 4), 0), ((4, 24, 8), 1), ((4, 8, 24), 2)],
)
def test_ragged_tma_view_extents_and_wraparound(shape, ragged_dim):
    total = shape[ragged_dim]
    # As in the GEMM operands, the ragged dim is the outermost one in memory.
    rest = shape[:ragged_dim] + shape[ragged_dim + 1 :]
    x = torch.empty((total, *rest), device="cuda", dtype=torch.bfloat16).movedim(0, ragged_dim)
    out = torch.zeros(4, device="cuda", dtype=torch.int64)
    T = from_dlpack(x, assumed_align=16).mark_layout_dynamic()
    mOut = from_dlpack(out)
    probe = _compile_probe(T, mOut, ragged_dim)
    row_bytes = x.stride(ragged_dim) * x.element_size()
    # (offset, length): first, middle, last (ends exactly at the ragged extent), empty
    for offset, length in [(0, 5), (5, 12), (17, 7), (0, total), (9, 0), (total, 0)]:
        probe(T, mOut, Int32(offset), Int32(length))
        ragged_extent, extra0, extra1, delta = out.tolist()
        assert ragged_extent == copy_utils.BIG_INT
        # Tight extents (#243): extra dim 0 is indexed only at BIG_INT, extra dim 1 at
        # offset + length <= total.
        assert (extra0, extra1) == (copy_utils.BIG_INT + 1, total + 1)
        assert offset + length < extra1
        # Ragged coordinate BIG_INT - length is the sequence's first row, so its rows end
        # exactly where the ragged dim goes out of bounds.
        assert delta % (1 << 64) == offset * row_bytes, (offset, length)
