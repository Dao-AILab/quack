# Copyright (c) 2026, Tri Dao.

"""FlyDSL sibling of :mod:`quack.reduction_base`.

Same job as the CuTe base -- pick a thread geometry for one row and describe the
tile it covers -- but host-side only. The CuTe base also hands the kernel a
``TiledCopy`` and the shared buffer it reduces through; here the kernel builds
both, from the numbers computed here, for two reasons:

* a FlyDSL ``TiledCopy`` is a bundle of MLIR values, so it cannot be built
  outside the trace and passed in;
* FlyDSL keys a compiled artifact on the kernel's source plus the *scalar*
  closure values it can see, and it cannot see through an object reference, so
  anything the kernel body specializes on has to reach it as a plain value
  rather than as ``self``. Two feature sets would otherwise share a kernel.

The cluster tier is dropped throughout: CDNA has no distributed shared memory to
split a row across CTAs.
"""

from quack.flydsl_constants import WAVE_SIZE

__all__ = ["ReductionBase"]


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class ReductionBase:
    def __init__(self, dtype: type, N: int):
        self.dtype = dtype
        self.N = N

    def _threads_per_row(self) -> int:
        raise NotImplementedError()

    def _num_threads(self) -> int:
        return 128 if self.N <= 16384 else 256

    def _blocks_per_tile(self, vecsize: int) -> int:
        """How many ``threads_per_row * vecsize`` blocks one tile covers.

        The whole row by default, which is what CuTe does: the fragment then
        holds the row and the second pass reads it back out of registers. A
        subclass returns fewer when a row is too wide to keep resident, trading
        a reload for register pressure.
        """
        return self._num_blocks_n(vecsize)

    def _num_blocks_n(self, vecsize: int) -> int:
        return _ceil_div(self.N // vecsize, self._threads_per_row())

    def _num_tiles_n(self, vecsize: int) -> int:
        """Tiles the kernel walks to cover one row; 1 when the row is resident."""
        return _ceil_div(self._num_blocks_n(vecsize), self._blocks_per_tile(vecsize))

    def _get_tiler(self, vecsize: int = 1):
        """The tile one CTA covers, as the CuTe base's ``_get_tiled_copy`` sizes it."""
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        num_threads, threads_per_row = self._num_threads(), self._threads_per_row()
        assert num_threads % WAVE_SIZE == 0
        blocks_per_tile = self._blocks_per_tile(vecsize)
        return (num_threads // threads_per_row, vecsize * blocks_per_tile * threads_per_row)

    def _waves_per_row(self) -> int:
        return max(self._threads_per_row() // WAVE_SIZE, 1)

    def _reduction_buffer_shape(self):
        """``(rows per block, waves per row)``.

        CuTe reads these two counts off the TV layout; FlyDSL's ``make_layout_tv``
        flattens the thread mode, so they come from the thread geometry that
        produced it instead. There is no third mode either: CuTe carries one so
        LayerNorm can stage its mean and variance through the same buffer, and
        this port has only RMSNorm's single reduction.
        """
        waves_per_row = self._waves_per_row()
        return (self._num_threads() // WAVE_SIZE // waves_per_row, waves_per_row)
