# Copyright (c) 2025-2026, Tri Dao.

"""TMEM storage helpers for CuTe DSL kernels."""

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Literal, Optional, TYPE_CHECKING

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils

if TYPE_CHECKING:
    from quack.spec.tensor_spec import BoundMMASm100


# The TMEM analogue of the SMEM SharedStorage @cute.struct, but
# column-addressed: TMEM is 128 lanes x 512 columns of 32-bit cells, a field's
# footprint is its column count (via `tcgen05.find_tmem_tensor_col_offset`),
# every field spans all 128 lanes, and offsets are added to the (32-bit-typed)
# TMEM base pointer. Field layouts come from a tiled_mma, not from
# (dtype, size), so fields are declared in MatmulSpec vocabulary, and dtype
# recasting (e.g. an f32 base pointer re-viewed as bf16 for a TMEM operand)
# happens inside the field.


def _tmem_dp_stride(dtype: type[cutlass.Numeric]) -> int:
    assert dtype.width <= 32 and 32 % dtype.width == 0, (
        f"TMEM layout expects a sub-32b or 32b dtype, got width={dtype.width}"
    )
    return (1 << 16) * (32 // dtype.width)


def m64_half_partition_offset(
    dtype: type[cutlass.Numeric], partition: Literal["lower", "upper"]
) -> int:
    """Base-pointer offset for the alternate M=64 half-subpartition."""
    assert partition in ("lower", "upper"), f"expected lower or upper, got {partition}"
    return 0 if partition == "lower" else 16 * _tmem_dp_stride(dtype)


def make_tmem_layout(
    dtype: type[cutlass.Numeric],
    shape: tuple[int, int],
    stage: int,
    *,
    interleaved: bool = False,
):
    """Role-free logical TMEM layout for dense tcgen05 storage.

    SM100 TMEM addresses are `(dp << 16) | col` in 32-bit words. `tmem_ptr<T>`
    applies the sub-word scaling for `T`, so the DP-lane stride in element units
    is `(1 << 16) * (32 / bits(T))`. Columns stay contiguous in element units.

    Only physical M=64 and M=128 are represented here. M=128 uses all DP lanes
    linearly. M=64 uses half-subpartitions: rows are grouped as `(16, 4)` and
    mapped to DPs `[0:16], [32:48], [64:80], [96:112]`.

    `interleaved=False` is the tcgen05 NonInterleaved layout used by TS-A and
    TS C-fragments: every stage advances by the column footprint. For M=64,
    `interleaved=True` packs stages in lower/upper half-subpartitions before
    advancing columns, matching the 1SM SS accumulator C-fragment. Stage 3 is
    rejected for interleaved layouts because the exact 3-stage pattern is not a
    rectangular affine layout.
    """
    assert len(shape) == 2, f"TMEM layout expects a 2D tile shape, got {shape}"
    assert stage in (1, 2, 3, 4), f"TMEM layout supports stage 1, 2, 3, or 4, got {stage}"
    rows, cols = shape
    assert rows in (64, 128), f"tcgen05 A-source TMEM layout expects M=64 or 128, got {rows}"
    elems_per_col = 32 // dtype.width
    dp_stride = _tmem_dp_stride(dtype)
    stage_stride = ((cols + elems_per_col - 1) // elems_per_col) * elems_per_col
    if rows == 64:
        if interleaved:
            assert stage != 3, "interleaved M=64 TMEM layout does not support stage=3"
            half_partition_stride = m64_half_partition_offset(dtype, "upper")
            if stage == 1:
                stage_shape = 1
                stage_stride = 0
            elif stage == 2:
                stage_shape = 2
                stage_stride = half_partition_stride
            else:
                stage_shape = (2, 2)
                stage_stride = (half_partition_stride, stage_stride)
            return cute.make_layout(
                ((16, 4), cols, stage_shape),
                stride=((dp_stride, 32 * dp_stride), 1, stage_stride),
            )
        return cute.make_layout(
            ((16, 4), cols, stage),
            stride=((dp_stride, 32 * dp_stride), 1, 0 if stage == 1 else stage_stride),
        )
    return cute.make_layout(
        cute.append(shape, stage),
        stride=(dp_stride, 1, 0 if stage == 1 else stage_stride),
    )


@dataclass
class _TmemFieldBase:
    """MLIR marshaling shared by TMEM field kinds.

    The field-owned MMA/copy objects are cute values; stage counts are static.
    Threading a field, or the whole TmemStruct, across DSL region boundaries
    re-binds those cute values.
    """

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.tiled_mma)

    def __new_from_mlir_values__(self, values):
        return replace(self, tiled_mma=cutlass.new_from_mlir_values(self.tiled_mma, values))


@dataclass
class TmemAcc(_TmemFieldBase):
    """Accumulator region: staged (MMA, MMA_M, MMA_N[, STAGE]) TMEM tensor."""

    mma: "BoundMMASm100"
    stages: Optional[int] = None

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.mma)

    def __new_from_mlir_values__(self, values):
        return replace(self, mma=cutlass.new_from_mlir_values(self.mma, values))

    def _make_frag(self) -> cute.Tensor:
        return self.mma._make_acc_frag(stages=self.stages)

    def num_cols(self) -> int:
        return tcgen05.find_tmem_tensor_col_offset(self._make_frag())

    def view(self, base_ptr, col_offset: int) -> cute.Tensor:
        return cute.make_tensor(base_ptr + col_offset, self._make_frag().layout)


@dataclass
class TmemOperandA(_TmemFieldBase):
    """TMEM-resident storage region later viewed as an MMA A operand."""

    mma: "BoundMMASm100"
    stage: Optional[int] = None

    def _physical_A(self):
        # BoundMMASm100 keeps logical A/B plus swap_AB. TmemOperandA is about
        # hardware operand A, so swapped MMAs source logical B.T from TMEM.
        if self.mma.swap_AB:
            assert self.mma.B is not None, "TmemOperandA requires BoundMMASm100.B when swapped"
            return self.mma.B.T
        assert self.mma.A is not None, "TmemOperandA requires BoundMMASm100.A"
        return self.mma.A

    def _physical_mnk(self):
        return (
            (self.mma.N, self.mma.M, self.mma.K)
            if self.mma.swap_AB
            else (
                self.mma.M,
                self.mma.N,
                self.mma.K,
            )
        )

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.mma)

    def __new_from_mlir_values__(self, values):
        return replace(self, mma=cutlass.new_from_mlir_values(self.mma, values))

    def num_cols(self) -> int:
        A = self._physical_A()
        assert A is not None, "TmemOperandA requires BoundMMASm100 physical A"
        stage = self.stage if self.stage is not None else A.stage
        layout = sm100_utils.make_smem_layout_a(
            self.mma.tiled_mma,
            self._physical_mnk(),
            A.dtype,
            stage,
        )
        shape = layout.outer.shape if hasattr(layout, "outer") else layout.shape
        return tcgen05.find_tmem_tensor_col_offset(self.mma.tiled_mma.make_fragment_A(shape))

    def view(self, base_ptr, col_offset: int) -> cute.Tensor:
        A = self._physical_A()
        assert A is not None, "TmemOperandA requires BoundMMASm100 physical A"
        cta_group = cute.size(self.mma.tiled_mma.thr_id.shape)
        ptr = cute.recast_ptr(base_ptr + col_offset, dtype=A.dtype)
        stage = self.stage if self.stage is not None else A.stage
        rows, cols = A.shape
        if cta_group == 2:
            assert rows in (64, 128), (
                f"2CTA TS-A TMEM layout expects per-CTA M=64 or 128, got {rows}"
            )
            rows = 128
        # Do not call TensorSpec.tmem_layout() here: that uses storage_shape so
        # `.T` remains a view of the same backing storage. TmemOperandA allocates
        # a fresh physical tcgen05 A tile, so `S.T` must materialize as shape
        # `(D, N)`, not reuse `S`'s backing `(N, D)` storage shape.
        return cute.make_tensor(ptr, make_tmem_layout(A.dtype, (rows, cols), stage))


def alias_acc_as_tmem(
    acc: cute.Tensor,
    dtype: type[cutlass.Numeric],
    shape: tuple[int, int],
    *,
    acc_cols: int,
    stage: int,
) -> cute.Tensor:
    """View the leading dtype tile of a TMEM accumulator allocation.

    This is for intentional subrange aliasing, e.g. linear attention's bf16 P
    tile over the f32 QK accumulator. Recast the full accumulator stage to the
    destination dtype first, then compose out the logical tile. This preserves
    dtype-scaled DP strides and the accumulator's physical stage stride, so if
    QK uses physical columns `[0, 128)` and `[128, 256)`, bf16 P uses logical
    columns `[0, 128)` backed by physical columns `[0, 64)` and `[128, 192)`.

    This currently covers the 1CTA physical-M=128 use case. If we reuse it for
    1CTA M=64 or 2CTA TS-A aliasing, it should grow the same half-subpartition /
    duplicated-local-view handling as `TensorSpec.tmem_layout(cta_group=...)`.
    """
    rows, cols = shape
    assert rows in (64, 128), f"TMEM alias expects M=64 or 128, got {rows}"
    assert dtype.width <= 32, f"TMEM alias expects <=32-bit dtype, got width={dtype.width}"
    elems_per_col = 32 // dtype.width
    assert cols <= acc_cols * elems_per_col, (
        f"TMEM alias shape {shape} does not fit in {acc_cols} physical accumulator cols"
    )
    dp_stride = _tmem_dp_stride(dtype)
    stage_stride = 0 if stage == 1 else acc_cols * elems_per_col
    if rows == 64:
        layout = cute.make_layout(
            ((16, 4), cols, stage),
            stride=((dp_stride, 32 * dp_stride), 1, stage_stride),
        )
    else:
        layout = cute.make_layout(cute.append(shape, stage), stride=(dp_stride, 1, stage_stride))
    return cute.make_tensor(cute.recast_ptr(acc.iterator, dtype=dtype), layout)


class TmemStruct:
    """Named TMEM regions for a kernel, packed back-to-back in declaration order."""

    def __init__(self, **fields):
        self._fields = fields
        self._offsets = {}
        field_cols = {
            name: 0 if field is None else field.num_cols() for name, field in fields.items()
        }
        offset = 0
        for name, num_cols in field_cols.items():
            self._offsets[name] = offset
            offset += num_cols
        num_cols = 32
        while num_cols < offset:
            num_cols *= 2
        max_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        assert num_cols <= max_cols, (
            f"TMEM plan needs {offset} cols ({field_cols}); "
            f"power-of-2 allocation {num_cols} exceeds the {max_cols}-col capacity"
        )
        self.num_cols = num_cols

    def col_offset(self, name: str) -> int:
        return self._offsets[name]

    def bind(self, base_ptr) -> SimpleNamespace:
        """Materialize all field views at the retrieved TMEM base pointer."""
        return SimpleNamespace(
            **{
                name: None if f is None else f.view(base_ptr, self._offsets[name])
                for name, f in self._fields.items()
            }
        )

    def __extract_mlir_values__(self):
        values = []
        self._field_lengths = []
        for f in self._fields.values():
            v = [] if f is None else cutlass.extract_mlir_values(f)
            values += v
            self._field_lengths.append(len(v))
        return values

    def __new_from_mlir_values__(self, values):
        new = object.__new__(TmemStruct)
        new_fields = {}
        offset = 0
        for (name, f), n in zip(self._fields.items(), self._field_lengths):
            new_fields[name] = (
                None if f is None else cutlass.new_from_mlir_values(f, values[offset : offset + n])
            )
            offset += n
        new._fields = new_fields
        new._offsets = dict(self._offsets)
        new.num_cols = self.num_cols
        return new
