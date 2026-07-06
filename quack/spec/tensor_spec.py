# Copyright (c) 2025-2026, Tri Dao.

"""Spec abstractions for declarative kernel operands.
Note: this is a prototype and the API could change rapidly.

`TensorSpec` is a declarative description of a staged tile (dtype, shape, SMEM
stage, layout) that drives SMEM layout creation, TMA atom construction, and TMA
pipelines. The spec is **storage-only and MMA/epilogue-role agnostic**: the
physical SMEM layout (swizzle + addressing) is keyed by storage facts (dtype,
tile shape, major-axis pattern, stages). For SM100, TensorSpec's storage/TMA
layout is flat; tcgen05's nested operand layout is derived later by MatmulSpec.

`MatmulSpec` (returned by `A @ B` on two TensorSpecs) owns everything
role-dependent: operand major modes deduced from `(layout, transposed, is_A)`,
the tiled_mma, and the role-nested SMEM views. TMA atoms are storage-layout
driven: single-CTA paths use the same flat CTA-value map as CuTe's generic tile
TMA helper, while SM100 2-CTA loads use a tcgen05 panel map because each peer
CTA owns instruction panels rather than a contiguous half tile. For SM90/SM120,
`bind_mma(thr)` returns a `BoundMMA` with per-warpgroup partitioned A/B
fragments. For SM100 (tcgen05), use `tiled_mma()` + `smem_view_A/B()` or
`bind_mma(tiled_mma=...)` for fragment views, and `with_tma_load(gmem, ...)` /
`with_tma(op, gmem, ...)` for TMA bindings.

Shapes are FULL logical tiles (what the MMA computes on). Peer-CTA
distribution is a TensorSpec storage property: `cta_group=2` splits the
storage-leading (MN) mode across the peer pair, so each CTA allocates/loads
`storage_shape = (mn/2, k)`. The split rule is role-free — a 2-CTA MMA splits
A along M and B along N, but both are mode 0 of the `(MN, K)` storage tile —
so SMEM layouts and TMA atoms never need to know operand roles; MMA
construction reads `cta_group` back off the operand specs and validates it.

Designed to be marshaled across the `@cute.kernel` boundary: `tma_atom` and
`gmem` cross via `__extract_mlir_values__` / `__new_from_mlir_values__`;
`smem` is populated inside the kernel via `with_smem(storage_field)` and
preserved from the host-side template (it lives in JIT-local scope, so cute
doesn't need to marshal it).
"""

from dataclasses import dataclass, replace
from typing import Literal, Optional, Tuple, Type
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils import LayoutEnum
import cutlass.utils.blackwell_helpers as sm100_utils


from quack import copy_utils, layout_utils, pipeline
from quack import sm90_utils
from quack.spec import mma as spec_mma
from quack.spec import smem as spec_smem
from quack.spec import tmem as spec_tmem
from quack.spec import tma as spec_tma


@dataclass
class TensorSpec:
    """Declarative spec for an operand tile. Owns dtype/shape/layout/stage so
    SMEM layouts, TMA atoms, and MMA configs can be derived from it.

    Shape rank:
      - 2D `(rows, cols)` — matmul operand tiles. Drives `MatmulSpec`/`bind_mma`,
        SM90/SM100 swizzled SMEM layouts, and 2-mode TMA atoms.
      - 1D `(vec,)` — vector-with-stage aux operands (Scale, Bias, Gamma, ...).
        `smem_layout()` returns a trivial `cute.make_layout((vec, stage))` (no
        swizzle); `tma_copy_bytes()` and `_make_tma_atom()` use `mode=[0]`. Cannot
        be used as a matmul operand (`__matmul__` / `bind_mma` are 2D-only).

    `shape` is the FULL logical tile. With `cta_group=2` each peer CTA
    stores/loads only its shard: `storage_shape` splits the storage-leading
    (MN) mode in half, and SMEM layouts / TMA atoms / TMEM layouts are derived
    from that per-CTA shard.

    `stage=None` means the tile lives in registers (no SMEM layout, no TMA).
    `transposed=True` is a logical .T view of the same storage (2D only).

    After `with_tma(...)`, the returned spec also carries the call-dynamic
    `tma_atom` and `gmem` (TMA tensor). The bound spec crosses the `@cute.kernel`
    boundary as a single arg — only the cute-object fields are MLIR-marshaled;
    static fields are preserved from the host-side template."""

    dtype: Type[cutlass.Numeric]
    shape: Tuple[
        int, ...
    ]  # 2D for matmul operands; 1D supported for vector-with-stage aux operands
    stage: Optional[int] = None
    # `layout` is the physical storage order of the backing tile. `transposed`
    # is only a logical view flag from `.T`: the same storage can be used as a
    # transposed matmul operand without changing its backing layout/bytes.
    layout: LayoutEnum = LayoutEnum.ROW_MAJOR
    # Optional override for the swizzled major-mode extent used to select the
    # SMEM atom. Most kernels can derive this from `shape`, but multi-warpgroup
    # SM90 kernels sometimes intentionally split the major mode per warpgroup
    # while keeping the same logical tile shape.
    major_mode_size: Optional[int] = None
    transposed: bool = False
    # Peer-CTA distribution for tcgen05 2-CTA MMAs. `shape` stays the full
    # logical tile; with cta_group=2 the storage-leading mode (the MN dim of
    # the (MN, K) storage convention) is split across the peer pair, so each
    # CTA allocates/loads `(mn / 2, k)`. This is a storage fact, not an MMA
    # role: A splits along M and B along N, but both are mode 0 of the storage
    # tile, so SMEM layouts and TMA atoms stay role-free. MMAs consuming the
    # spec must be built with a matching cta_group (validated in MatmulSpec).
    cta_group: int = 1
    # TMA binding: one bundled `cpasync.TmaInfo` (atom + TMA coordinate tensor
    # + construction smem layout) — what _make_tma_atom returns. `tma_atom`/`gmem`
    # are exposed as properties.
    tma: Optional[cpasync.TmaInfo] = None
    # Plain (non-TMA) gmem tensor, for operands copied with non-TMA helpers.
    gmem_raw: Optional[cute.Tensor] = None
    smem: Optional[cute.Tensor] = None  # populated inside the kernel via with_smem()
    tmem: Optional[cute.Tensor] = None  # populated inside the kernel via with_tmem()

    def __extract_mlir_values__(self):
        # Marshal the TMA binding + plain gmem across the kernel boundary
        # (`TmaInfo` implements the marshaling protocol itself). `smem` is
        # created inside the kernel via with_smem() and lives in JIT-local
        # scope; cute can pass it by reference without marshaling.
        values = []
        self._n_tma = 0
        self._n_gmem = 0
        if self.tma is not None:
            v = cutlass.extract_mlir_values(self.tma)
            values += v
            self._n_tma = len(v)
        if self.gmem_raw is not None:
            v = cutlass.extract_mlir_values(self.gmem_raw)
            values += v
            self._n_gmem = len(v)
        return values

    def __new_from_mlir_values__(self, values):
        offset = 0
        new_tma = None
        if self.tma is not None:
            new_tma = cutlass.new_from_mlir_values(self.tma, values[offset : offset + self._n_tma])
            offset += self._n_tma
        new_gmem = None
        if self.gmem_raw is not None:
            new_gmem = cutlass.new_from_mlir_values(
                self.gmem_raw, values[offset : offset + self._n_gmem]
            )
            offset += self._n_gmem
        return replace(self, tma=new_tma, gmem_raw=new_gmem)

    @property
    def tma_atom(self) -> Optional[cute.CopyAtom]:
        return self.tma.atom if self.tma is not None else None

    @property
    def gmem(self) -> Optional[cute.Tensor]:
        """The gmem tensor: the TMA coordinate tensor when TMA-bound, else the
        plain tensor attached via with_gmem()."""
        if self.tma is not None:
            return self.tma.tma_tensor
        return self.gmem_raw

    def with_tma(
        self,
        op,
        gmem_tensor: cute.Tensor,
        *,
        num_multicast: int = 1,
        internal_type: Optional[Type[cutlass.Numeric]] = None,
        cta_v_map=None,
        gmem_raw: Optional[cute.Tensor] = None,
    ) -> "TensorSpec":
        """Return a new spec carrying a generic tile TMA binding.

        `gmem_tensor` is the TMA coordinate tensor. Use `gmem_raw` when the
        kernel also needs ordinary GMEM indexing, e.g. when `gmem_tensor` is a
        ragged/role-specific view used only for TMA.
        """
        tma = self._make_tma_atom(
            op,
            gmem_tensor,
            num_multicast=num_multicast,
            internal_type=internal_type,
            cta_v_map=cta_v_map,
        )
        return replace(
            self,
            tma=tma,
            gmem_raw=gmem_raw if gmem_raw is not None else self.gmem_raw,
        )

    def with_tma_load(
        self,
        gmem_tensor: cute.Tensor,
        *,
        num_multicast: int = 1,
        internal_type: Optional[Type[cutlass.Numeric]] = None,
        gmem_raw: Optional[cute.Tensor] = None,
    ) -> "TensorSpec":
        """`with_tma` for G2S loads with the op derived from the spec.

        Role-free: the op's cta_group comes from the spec's storage
        distribution and multicast from `num_multicast` — no A/B distinction
        (replaces role-named op selectors like `cluster_shape_to_tma_atom_B`
        at the kernel level).
        """
        cg = tcgen05.CtaGroup.TWO if self.cta_group == 2 else tcgen05.CtaGroup.ONE
        op = (
            cpasync.CopyBulkTensorTileG2SMulticastOp(cg)
            if num_multicast > 1
            else cpasync.CopyBulkTensorTileG2SOp(cg)
        )
        return self.with_tma(
            op,
            gmem_tensor,
            num_multicast=num_multicast,
            internal_type=internal_type,
            gmem_raw=gmem_raw,
        )

    def with_tma_info(
        self,
        tma_atom: cute.CopyAtom,
        tma_tensor,
        smem_layout=None,
        *,
        gmem_raw: Optional[cute.Tensor] = None,
    ) -> "TensorSpec":
        """Return a new spec carrying an externally-built TMA binding.

        Use this for kernels that need CUTLASS' role-aware TMA construction but
        still want to pass one TensorSpec across the kernel boundary.
        """
        return replace(
            self,
            tma=cpasync.TmaInfo(tma_atom, tma_tensor, smem_layout),
            gmem_raw=gmem_raw if gmem_raw is not None else self.gmem_raw,
        )

    def with_gmem(self, gmem) -> "TensorSpec":
        """Return a new spec with only a plain GMEM tensor attached.

        Use for operands that should cross the kernel boundary as TensorSpecs but
        are copied with non-TMA helpers.
        """
        return replace(self, gmem_raw=gmem)

    def with_smem(
        self, storage_or_tensor, *, layout=None, single_stage: bool = False
    ) -> "TensorSpec":
        """Return a new spec with the SMEM tensor attached (call inside the kernel).
        Semantics:
        - storage field, no layout: derive this spec's `smem_layout()` and recast
          the backing pointer to this spec's dtype if the storage field differs.
        - storage field, layout: use the supplied layout and recast to this spec's dtype.
        - cute.Pointer, no layout: derive this spec's `smem_layout()` and reinterpret the
          pointer as this spec's dtype/layout.
        - cute.Pointer, layout: reinterpret the pointer as this spec's dtype with the layout.
        - cute.Tensor, no layout: bind the tensor exactly as-is; dtype must already match.
        - cute.Tensor, layout: unsupported; pass `tensor.iterator` to request reinterpretation."""
        is_storage_field = const_expr(hasattr(storage_or_tensor, "get_tensor"))
        is_tensor = const_expr(hasattr(storage_or_tensor, "iterator"))
        is_pointer = const_expr(isinstance(storage_or_tensor, cute.Pointer))

        if layout is not None:
            assert not is_tensor, (
                "with_smem(tensor, layout=...) is unsupported; pass tensor.iterator"
            )
            if const_expr(is_storage_field):
                smem = self.get_smem_tensor(storage_or_tensor, layout)
            else:
                assert is_pointer, "with_smem(..., layout=...) expects a storage field or pointer"
                smem = self._make_smem_tensor_from_ptr(storage_or_tensor, layout)
        elif is_storage_field:
            smem = self.get_smem_tensor(storage_or_tensor)
        elif is_pointer:
            smem = self._make_smem_tensor_from_ptr(storage_or_tensor, self.smem_layout())
        else:
            assert is_tensor, "with_smem expects a storage field or cute.Tensor"
            smem = storage_or_tensor
            assert const_expr(smem.element_type == self.dtype), "SMEM tensor dtype mismatch"
        if const_expr(single_stage):
            assert self.stage == 1, "single_stage=True requires TensorSpec stage=1"
            assert const_expr(cute.rank(smem) == self.rank + 1), (
                "single_stage=True expects a staged SMEM tensor"
            )
            smem = smem[..., 0]
        return replace(self, smem=smem)

    def _make_smem_tensor_from_ptr(self, ptr: cute.Pointer, layout) -> cute.Tensor:
        """Reinterpret an SMEM pointer with `layout` and this spec's dtype."""
        if hasattr(layout, "outer"):
            return cute.make_tensor(
                cute.recast_ptr(ptr, layout.inner, dtype=self.dtype), layout.outer
            )
        return cute.make_tensor(cute.recast_ptr(ptr, dtype=self.dtype), layout)

    def tmem_layout(self):
        """Role-free flat TMEM storage layout for this spec.

        TMEM MMAs may need nested role layouts (e.g. tcgen05 A operand), but the
        TensorSpec-owned storage identity stays `(rows, cols[, stage])`, matching
        the SMEM/TMA side of the spec for the all-DP physical M=128 case. The
        strides are TMEM-specific, not compact: rows stride by DP lane, columns
        are contiguous, and stages advance by the logical column footprint. A
        physical M=64 is represented with a nested row mode because tcgen05 uses
        half-subpartition DP lanes for that MMA shape. For 2CTA TS-A, the full
        MMA M/N live at the tiled_mma level; this local TMEM A view is still per
        CTA. Per-CTA M=64 is duplicated into all 128 local DP lanes, and per-CTA
        M=128 already occupies all 128 local DP lanes.
        """
        assert not self.in_rmem, "register tensor has no TMEM layout"
        assert self.rank == 2, "TMEM TensorSpec storage is currently only defined for 2D tiles"
        rows, cols = self.storage_shape
        if self.cta_group == 2:
            assert rows in (64, 128), (
                f"2CTA TS-A TMEM layout expects per-CTA M=64 or 128, got {rows}"
            )
            local_tmem_rows = 128
        else:
            local_tmem_rows = rows
        return spec_tmem.make_tmem_layout(self.dtype, (local_tmem_rows, cols), self.stage)

    def with_tmem(
        self,
        storage_or_tensor,
        *,
        m64_partition: Literal["lower", "upper"] = "lower",
    ) -> "TensorSpec":
        """Return a new spec with a TMEM tensor attached.

        Existing TMEM tensors are attached as-is. This keeps the public API
        role-free: `TmemOperandA` fields already carry the spec storage layout,
        while `TmemAcc` fields carry the MMA C-fragment layout.

        Raw pointers are interpreted as this spec's dense storage layout. For
        1CTA M=64 storage, `m64_partition` can select the alternate 16-DP
        half-subpartition by shifting the base pointer.
        """
        if hasattr(storage_or_tensor, "iterator"):
            assert m64_partition == "lower", "m64_partition only applies to TS-A TMEM"
            return replace(self, tmem=storage_or_tensor)

        ptr = storage_or_tensor
        rows, _ = self.storage_shape
        assert m64_partition == "lower" or (self.cta_group == 1 and rows == 64), (
            "m64_partition='upper' is only meaningful for 1CTA M=64 TS-A TMEM"
        )
        ptr = cute.recast_ptr(ptr, dtype=self.dtype)
        if m64_partition == "upper":
            ptr = ptr + spec_tmem.m64_half_partition_offset(self.dtype, m64_partition)
        tmem = cute.make_tensor(ptr, self.tmem_layout())
        return replace(self, tmem=tmem)

    @property
    def T(self) -> "TensorSpec":
        # Logical .T view of the same storage — carries over tma_atom/gmem/smem
        # so the matmul-spec can default operand smems from `spec.smem` regardless of T.
        return replace(self, shape=(self.shape[1], self.shape[0]), transposed=not self.transposed)

    @property
    def in_rmem(self) -> bool:
        return self.stage is None

    @property
    def rank(self) -> int:
        """Shape rank: 2 for matmul operand tiles, 1 for vector-with-stage aux operands.
        Drives the rank-1 branches in `smem_layout`/`tma_copy_bytes`/`make_tma_atom`/
        `storage_shape` (skip swizzle/matmul-side logic)."""
        return len(self.shape)

    def __post_init__(self):
        assert self.cta_group in (1, 2), f"cta_group must be 1 or 2, got {self.cta_group}"
        if self.cta_group == 2:
            assert len(self.shape) == 2, "cta_group=2 requires a 2D matmul-operand spec"

    @property
    def full_storage_shape(self) -> Tuple[int, ...]:
        """Full-tile shape in storage order — the GMEM tile extent for
        `cute.local_tile` at TMA load sites, independent of cta_group."""
        # 1D has nothing to transpose.
        if self.rank == 1:
            return self.shape
        return (self.shape[1], self.shape[0]) if self.transposed else self.shape

    @property
    def storage_shape(self) -> Tuple[int, ...]:
        """Per-CTA storage shape: the full tile with the leading (MN) storage
        mode split across the peer pair when cta_group=2."""
        full = self.full_storage_shape
        if self.rank == 1 or self.cta_group == 1:
            return full
        assert full[0] % self.cta_group == 0, (
            f"leading storage dim {full[0]} not divisible by cta_group {self.cta_group}"
        )
        return (full[0] // self.cta_group, full[1])

    def smem_layout(self):
        """Derive the SMEM layout for this operand.

        1D specs (vector-with-stage aux operands) use a trivial
        `cute.make_layout`, with no swizzling.

        2D specs use the arch-specific SMEM atom selected by
        `spec_smem.make_smem_layout`, then expose a coalesced role-free
        storage/allocation view. The swizzled axis is the storage-contiguous
        axis named by `layout`: ROW_MAJOR means K-contiguous, COL_MAJOR means
        MN-contiguous. MMA-specific nested operand views are constructed
        separately in MatmulSpec.

        The layout is built FRESH on every call (not cached). This matters because
        SMEM layout values are MLIR-region-local: a layout built on the host doesn't
        survive crossing into a `@cute.kernel`. The 2D path reconstructs the
        layout from primitive (marshaled) inputs on each call — compile-time
        only, no runtime cost."""
        assert not self.in_rmem, "register tensor has no SMEM layout"
        if self.rank == 1:
            # 1D vector-with-stage: no swizzling needed (small + naturally aligned).
            return cute.make_layout((self.shape[0], self.stage))
        return spec_smem.make_smem_layout(
            self.dtype,
            self.layout,
            self.storage_shape,
            self.stage,
            self.major_mode_size,
        )

    def tma_copy_bytes(self, *, full_tile: bool = False) -> int:
        """Bytes of one CTA's TMA transfer for a single stage.

        `full_tile=True` scales to the whole logical tile across the peer pair
        (× cta_group) — the mbarrier tx count for 2-CTA loads, where both
        peers' transactions arrive at the same barrier."""
        # 1D specs have a single non-stage mode; 2D specs have two.
        modes = [0] if self.rank == 1 else [0, 1]
        per_cta = cute.size_in_bytes(self.dtype, cute.select(self.smem_layout(), mode=modes))
        return per_cta * self.cta_group if full_tile else per_cta

    def _make_tma_atom(
        self,
        op,
        gmem_tensor,
        num_multicast: int = 1,
        internal_type: Optional[Type[cutlass.Numeric]] = None,
        cta_v_map=None,
    ):
        if cta_v_map is not None:
            modes = [0] if self.rank == 1 else [0, 1]
            tma_smem_layout = cute.select(self.smem_layout(), mode=modes)
        elif self.rank == 2 and self.cta_group == 2:
            assert (
                isinstance(
                    op,
                    (cpasync.CopyBulkTensorTileG2SOp, cpasync.CopyBulkTensorTileG2SMulticastOp),
                )
                and op.cta_group == tcgen05.CtaGroup.TWO
            ), f"cta_group=2 spec requires a 2-CTA G2S load op, got {op}"
            # The SMEM descriptor can use the normal flat per-CTA storage view.
            # Only the GMEM coordinate map is non-contiguous: for a full 512-wide
            # leading tile, one CTA owns panels 0..127 and 256..383.
            tma_smem_layout = cute.select(self.smem_layout(), mode=[0, 1])
            cta_v_map = spec_tma._sm100_dense_tma_flat_cta_v_map(self.storage_shape, cta_group=2)
        else:
            assert getattr(op, "cta_group", None) != tcgen05.CtaGroup.TWO, (
                "2-CTA load op on a cta_group=1 spec — declare the spec with cta_group=2"
            )
            modes = [0] if self.rank == 1 else [0, 1]
            tma_smem_layout = cute.select(self.smem_layout(), mode=modes)
            cta_v_map = cute.composition(
                cute.make_identity_layout(gmem_tensor.shape),
                self.storage_shape,
            )
        return spec_tma._make_tiled_tma_atom_from_cta_v_map(
            op,
            gmem_tensor,
            tma_smem_layout,
            cta_v_map,
            num_multicast,
            internal_type=internal_type,
        )

    def smem_struct(self, align: int):
        """The aligned SMEM byte allocation for this spec, for inclusion in a SharedStorage class."""
        return cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.smem_layout())], align
        ]

    def get_smem_tensor(self, storage_field, layout=None):
        """Materialize the SMEM tensor backed by `storage_field` with this spec's
        layout (or the supplied `layout`), recast to this spec's dtype."""
        if layout is None:
            layout = self.smem_layout()
        if hasattr(layout, "outer"):
            smem = storage_field.get_tensor(layout.outer, swizzle=layout.inner, dtype=self.dtype)
        else:
            smem = storage_field.get_tensor(layout, dtype=self.dtype)
        if const_expr(smem.element_type != self.dtype):
            smem = cute.make_tensor(cute.recast_ptr(smem.iterator, dtype=self.dtype), smem.layout)
        return smem

    @property
    def smem_T(self) -> cute.Tensor:
        """`transpose_view` of `self.smem` — the layout-transposed view used as
        partition_B input when the operand's matmul B-side is MN-major. Hot path
        in mamba/linear-attn kernels (sBt = transpose_view(B.smem))."""
        assert self.smem is not None, "smem not bound — call with_smem(...) first"
        return layout_utils.transpose_view(self.smem)

    def tma_load_fn(self, g_tile, cta_coord=0, cta_layout=None, *, peer_coord=0, **kwargs):
        """Build a TMA load copy fn (gmem → smem) bound to this spec's tma_atom + smem.

        `g_tile` is the FULL-tile gmem slice — typically
        `cute.local_tile(m, spec.full_storage_shape, coord)`. For cta_group=2
        this CTA's leading-mode shard is selected internally from `peer_coord`
        (the MMA peer rank, `mma_tile_coord_v`), keeping the slice convention
        paired with the TMA atom's flat CTA-value map.
        Defaults `cta_coord=0`, `cta_layout=cute.make_layout(1)` (no multicast).
        Returns the same `(copy_fn, ...)` tuple as `copy_utils.tma_get_copy_fn`."""
        if self.cta_group != 1:
            g_tile = spec_tma.slice_tma_tile_by_mma_cta(
                g_tile, self.storage_shape[0], peer_coord, self.cta_group
            )
        if cta_layout is None:
            cta_layout = cute.make_layout(1)
        return copy_utils.tma_get_copy_fn(
            self.tma_atom, cta_coord, cta_layout, g_tile, self.smem, **kwargs
        )

    def tma_store_fn(self, g_tile, cta_coord=0, cta_layout=None, **kwargs):
        """Build a TMA store copy fn (smem → gmem) bound to this spec's tma_atom + smem.
        Defaults `cta_coord=0`, `cta_layout=cute.make_layout(1)` (no multicast)."""
        if cta_layout is None:
            cta_layout = cute.make_layout(1)
        return copy_utils.tma_get_copy_fn(
            self.tma_atom, cta_coord, cta_layout, self.smem, g_tile, **kwargs
        )

    def tma_pipeline_umma(
        self,
        producer_group,
        consumer_group,
        *,
        barrier_storage=None,
        full_tile: bool = False,
        extra_bytes: int = 0,
        **kwargs,
    ):
        """A PipelineTmaUmma sized by this spec — num_stages and tx_count come
        from the spec (stage count / per-stage TMA bytes), so the pipeline
        cannot drift from the storage ring it guards and kernels don't repeat
        the stage/byte bookkeeping per operand.

        Omit `barrier_storage` to let the pipeline allocate reserved smem for
        its mbarriers.

        `extra_bytes` is added to tx_count, for one pipeline guarding several
        operands loaded per stage (the gemm A+B pattern):
        `A.tma_pipeline_umma(..., extra_bytes=B.tma_copy_bytes())`."""
        return pipeline.PipelineTmaUmma.create(
            num_stages=self.stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_bytes(full_tile=full_tile) + extra_bytes,
            barrier_storage=barrier_storage,
            defer_sync=True,
            **kwargs,
        )

    def tma_pipeline_async(
        self,
        producer_group,
        consumer_group,
        *,
        barrier_storage=None,
        extra_bytes: int = 0,
        **kwargs,
    ):
        """SM90 counterpart of `tma_pipeline_umma`: a PipelineTmaAsync (TMA
        producer -> async thread consumers, e.g. WGMMA warpgroups) sized by
        this spec's stage count and per-stage TMA bytes. Multicast cluster
        shape goes through `cta_layout_vmnk=`. No `full_tile` here — peer-CTA
        storage splitting (cta_group=2) is tcgen05-only.

        Omit `barrier_storage` to let the pipeline allocate reserved smem for
        its mbarriers.

        `extra_bytes` is added to tx_count, for one pipeline guarding several
        operands loaded per stage (the gemm A+B pattern):
        `A.tma_pipeline_async(..., extra_bytes=B.tma_copy_bytes())`."""
        return pipeline.PipelineTmaAsync.create(
            num_stages=self.stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_bytes() + extra_bytes,
            barrier_storage=barrier_storage,
            defer_sync=True,
            **kwargs,
        )

    def __matmul__(self, other: "TensorSpec") -> "MatmulSpec":
        return MatmulSpec(self, other)


@dataclass
class BoundMMA:
    """A tiled_mma plus its partitioned operand fragments and matmul shape (M, N, K).
    Bundles the per-MMA boilerplate that follows `(A @ B).bind_mma(...)`.

    `frag_A`/`frag_B` semantics differ per arch:
    - WGMMA (Hopper): multi-stage descriptors used with `A_idx`/`B_idx` in gemm.
    - Warp-level (SM120): single-stage rmem fragments — the kernel must do its
      own ldmatrix SMEM->RMEM step into them before each MMA.

    `tiled_copy_s2r_A`/`B` are the SMEM->RMEM (ldmatrix) `cute.TiledCopy`s —
    used by SM120 for the explicit SMEM->RMEM step before each MMA, and useful
    on SM90 for non-WGMMA paths that load into register frags.
    `tiled_copy_r2s_A`/`B` are the RMEM->SMEM (stmatrix) counterparts — used by
    kernels that stage an A-operand transform back through SMEM (e.g. for a
    follow-on MMA in a different tiling).
    (The per-stage SMEM *partition view* is a kernel mainloop concern and is
    not on this object — derive via
    `tiled_copy_s2r_A.get_slice(thr).partition_S(sA)` at the use site.)"""

    tiled_mma: cute.TiledMma
    frag_A: Optional[cute.Tensor]
    frag_B: Optional[cute.Tensor]
    M: int  # logical M (the user's matmul A side); when swap_AB, physical wgmma sees N here
    N: int  # logical N (the user's matmul B side)
    K: int
    tiled_copy_s2r_A: Optional[cute.TiledCopy] = None
    tiled_copy_s2r_B: Optional[cute.TiledCopy] = None
    tiled_copy_r2s_A: Optional[cute.TiledCopy] = None
    tiled_copy_r2s_B: Optional[cute.TiledCopy] = None
    # When True, the underlying wgmma was constructed with operand roles swapped
    # (logical A → physical B and vice versa) — typically as a wgmma-instruction
    # reduction trick when the logical A's M is too large but B's N would fit a
    # single instance. The user keeps thinking in logical (A, B) terms; .acc(),
    # .fn(), and .r2s_C() handle the physical swap internally.
    swap_AB: bool = False

    # MLIR marshaling — without these the cute jit boundary auto-flattens this
    # dataclass to its first cute-typed field (`tiled_mma`), losing the rest.
    # Static fields (M/N/K) are preserved from the host-side template.
    def __extract_mlir_values__(self):
        values = []
        self._lengths = {}
        for name in (
            "tiled_mma",
            "frag_A",
            "frag_B",
            "tiled_copy_s2r_A",
            "tiled_copy_s2r_B",
            "tiled_copy_r2s_A",
            "tiled_copy_r2s_B",
        ):
            obj = getattr(self, name)
            if obj is not None:
                v = cutlass.extract_mlir_values(obj)
                values += v
                self._lengths[name] = len(v)
            else:
                self._lengths[name] = 0
        return values

    def __new_from_mlir_values__(self, values):
        new_fields = {}
        offset = 0
        for name in (
            "tiled_mma",
            "frag_A",
            "frag_B",
            "tiled_copy_s2r_A",
            "tiled_copy_s2r_B",
            "tiled_copy_r2s_A",
            "tiled_copy_r2s_B",
        ):
            n = self._lengths[name]
            if n > 0:
                obj = getattr(self, name)
                new_fields[name] = cutlass.new_from_mlir_values(obj, values[offset : offset + n])
                offset += n
            else:
                new_fields[name] = None
        return replace(self, **new_fields)

    def acc(self, shape=None, dtype=Float32) -> cute.Tensor:
        """Allocate an accumulator rmem tensor. `shape` defaults to logical (M, N).
        Extra modes after (M, N) are appended to the partitioned C layout, e.g.
        `(M, N, stage)` becomes `(MMA, MMA_M, MMA_N, stage)`. When swap_AB, the
        physical wgmma C-side is (N, M) — we feed that to partition_shape_C; the
        resulting rmem holds the transposed accumulator, but the user can treat
        it as opaque (fill / pass to .fn / .r2s_C)."""
        if shape is None:
            shape = (self.M, self.N)
        if self.swap_AB:
            shape = (shape[1], shape[0], *shape[2:])  # physical (N, M, ...)
        acc_shape = self.tiled_mma.partition_shape_C(shape[:2])
        for extra_mode in shape[2:]:
            acc_shape = cute.append(acc_shape, extra_mode)
        return cute.make_rmem_tensor(acc_shape, dtype)

    def clone_frag_A(self) -> cute.Tensor:
        """Allocate another rmem tensor matching this MMA's frag_A — used for
        multi-stage RS patterns where each stage needs its own A operand."""
        assert self.frag_A is not None, "no frag_A to clone (call bind_mma with source='RS')"
        return cute.make_rmem_tensor(self.frag_A.layout, self.frag_A.element_type)

    def fn(self, acc, zero_init=False, frag_A=None, frag_B=None):
        """Return a callable that captures `acc`/default frags — call per-iteration in a loop.
        `frag_A`/`frag_B` override the bound fragments either here or at call time
        (multi-stage RS pattern where the A fragment is produced in the loop).
        When swap_AB, A_idx/B_idx are user-logical and get swapped internally."""
        default_A = frag_A if frag_A is not None else self.frag_A
        default_B = frag_B if frag_B is not None else self.frag_B

        def _fn(A_idx=None, B_idx=None, wg_wait=-1, zero_init=zero_init, frag_A=None, frag_B=None):
            fA = frag_A if frag_A is not None else default_A
            fB = frag_B if frag_B is not None else default_B
            if self.swap_AB:
                return sm90_utils.gemm_w_idx(
                    self.tiled_mma, acc, fA, fB, zero_init, B_idx, A_idx, wg_wait
                )
            return sm90_utils.gemm_w_idx(
                self.tiled_mma, acc, fA, fB, zero_init, A_idx, B_idx, wg_wait
            )

        return _fn

    def fn_zero_init(self, shape=None, frag_A=None, frag_B=None):
        """Return a partial for the zero-init gemm variant (allocates its own acc).
        `shape` defaults to logical (M, N) — swapped to physical (N, M) when swap_AB.
        A_idx/B_idx are user-logical and get swapped internally when swap_AB."""
        if shape is None:
            shape = (self.M, self.N)
        if self.swap_AB:
            shape = (shape[1], shape[0])
        fA = frag_A if frag_A is not None else self.frag_A
        fB = frag_B if frag_B is not None else self.frag_B
        inner = partial(sm90_utils.gemm_zero_init, self.tiled_mma, shape, fA, fB)
        if self.swap_AB:

            def _fn(A_idx=None, B_idx=None, **kw):
                return inner(A_idx=B_idx, B_idx=A_idx, **kw)

            return _fn
        return inner

    # SMEM<->RMEM helpers for the A and C operand positions of this MMA.
    # Thin wrappers over `quack.copy_utils.get_smem_(load|store)_(A|C)` that
    # bind `self.tiled_mma` so call sites read as `mma.r2s_C(sC, tidx)` instead
    # of `copy_utils.get_smem_store_C(tiled_mma_pv, sC, tidx)`. Each returns the
    # same `(copy_fn, thr_copy, partitioned_tensor)` tuple as the underlying
    # helper. No B variants — WGMMA loads B via descriptor with no register
    # staging path, mirroring `copy_utils`'s lack of `get_smem_(load|store)_B`.
    def s2r_A(self, sA, thr, **kwargs):
        return copy_utils.get_smem_load_A(self.tiled_mma, sA, thr, **kwargs)

    def r2s_A(self, sA, thr, **kwargs):
        return copy_utils.get_smem_store_A(self.tiled_mma, sA, thr, **kwargs)

    def s2r_C(self, sC, thr, **kwargs):
        # Mirror r2s_C's swap_AB handling so epilogue inputs staged in the
        # logical (M, N) layout read back correctly from a swapped MMA.
        if self.swap_AB:
            kwargs["transpose"] = not kwargs.get("transpose", False)
            sC = layout_utils.transpose_view(sC)
        return copy_utils.get_smem_load_C(self.tiled_mma, sC, thr, **kwargs)

    def r2s_C(self, sC, thr, **kwargs):
        # When swap_AB, the rmem accumulator is physically (N, M) but the user's
        # `sC` is in logical (M, N) layout. Auto-fix: feed transpose_view(sC) so
        # make_tiled_copy_C sees matching shape, and toggle the stmatrix transpose
        # bit so the data lands in sC's underlying storage in logical orientation.
        if self.swap_AB:
            kwargs["transpose"] = not kwargs.get("transpose", False)
            sC = layout_utils.transpose_view(sC)
        return copy_utils.get_smem_store_C(self.tiled_mma, sC, thr, **kwargs)


@dataclass
class BoundMMASm100(BoundMMA):
    """A tcgen05 tiled_mma plus its SMEM-descriptor operand fragments and full
    MMA tile shape (M, N, K). The SM100 counterpart of `BoundMMA`, shaped by
    the tcgen05 execution model:

    - `frag_A`/`frag_B` are multi-stage SMEM descriptor fragments
      ((MMA, MMA_M/N, MMA_K, STAGE)); there is no register staging path.
    - The accumulator lives in TMEM, not RMEM: `acc(tmem_ptr, stages=)`
      builds the staged accumulator tensor at a retrieved TMEM pointer.
      (Use `MatmulSpec.acc_layout_sm100` from warps that only read the
      accumulator and never bind fragments, e.g. epilogue warps.)
    - The MMA is issued from the MMA warp of the **leader CTA** only — for
      cta_group=2 the peer CTA contributes operand SMEM but does not issue.
      Gate `gemm()` with an `is_leader_cta` check; the single-thread election
      within the warp is handled by `cute.gemm` itself.
    - `gemm()` / `fn()` issue through a fresh `cute.make_mma_atom(tiled_mma.op)`
      so ACCUMULATE/SFA/SFB mutations stay local to the helper and do not force
      callers to loop-carry `tiled_mma` through dynamic regions.

    `M/N/K` are the user's logical full tile dims (the spec shapes). With
    `swap_AB=True`, the physical tcgen05 C tile is `(N, M)`; accumulator
    helpers derive that physical layout internally. The per-CTA storage shards
    stay on `TensorSpec.storage_shape`, where TMA and SMEM views need them."""

    A: Optional[TensorSpec] = None
    B: Optional[TensorSpec] = None

    def _physical_acc_mn(self) -> Tuple[int, int]:
        return (self.N, self.M) if self.swap_AB else (self.M, self.N)

    def _make_acc_frag(self, *, stages: Optional[int] = None) -> cute.Tensor:
        shape = self.tiled_mma.partition_shape_C(self._physical_acc_mn())
        if stages is not None:
            shape = cute.append(shape, stages)
        return self.tiled_mma.make_fragment_C(shape)

    @property
    def num_k_blocks(self) -> int:
        return cute.size(self.frag_A, mode=[2])

    def acc_layout(self, *, stages: Optional[int] = None):
        """Staged TMEM accumulator layout: (MMA, MMA_M, MMA_N[, STAGE])."""
        return self._make_acc_frag(stages=stages).layout

    def acc(self, tmem_ptr, *, stages: Optional[int] = None) -> cute.Tensor:
        """The staged accumulator tensor at a retrieved TMEM pointer."""
        return cute.make_tensor(tmem_ptr, self.acc_layout(stages=stages))

    def t2r_C(
        self,
        acc: cute.Tensor,
        tidx: Int32,
        dst_dtype: Type[cutlass.Numeric],
        *,
        epi_tile: Optional[cute.Tile] = None,
        num_cols: Optional[int] = None,
        transpose: bool = False,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Build the tcgen05 accumulator TMEM->RMEM copy.

        Returns `(tiled_t2r, tTR_tAcc, tTR_rAcc)`, where `tTR_tAcc` is the
        partitioned TMEM source and `tTR_rAcc` is the per-thread register
        fragment to reuse for each epilogue subtile. Pass `epi_tile=None` for
        non-epilogue spills that should partition the raw staged accumulator.

        With `swap_AB`, the accumulator is physically `(N, M)`. For epilogue
        copies, `transpose` also presents the logical `(M, N)` accumulator view.
        For raw `epi_tile=None` spills, `transpose` only selects the row/col
        copy policy and the raw accumulator view is kept. The t2r atom is
        selected from that source view shape, `transpose`-derived copy layout,
        and destination dtype.
        At least one of `epi_tile` or `num_cols` is required. `num_cols` uses
        the same Blackwell atom-selection heuristic but substitutes a synthetic
        `(tile_m, num_cols)` epilogue tile, useful for non-epilogue spills that
        copy a narrower major-mode slice.
        """
        assert epi_tile is not None or num_cols is not None, "pass epi_tile and/or num_cols"
        tAcc = acc[(None, None), 0, 0, None]
        copy_layout = LayoutEnum.COL_MAJOR if const_expr(transpose) else LayoutEnum.ROW_MAJOR
        tile_m, tile_n = cute.size(tAcc.shape, mode=[0]), cute.size(tAcc.shape, mode=[1])
        is_2cta = cute.size(self.tiled_mma.thr_id.shape) == 2
        atom_tile = epi_tile if const_expr(epi_tile is not None) else (tile_m, num_cols)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (tile_m, tile_n, self.K),
            copy_layout,
            dst_dtype,
            acc.element_type,
            atom_tile,
            is_2cta,
        )
        if const_expr(epi_tile is None):
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc[None, None, 0])
            tTR_tAcc = tiled_copy_t2r.get_slice(tidx).partition_S(tAcc)
            tTR_rAcc = copy_utils.tmem_reg_frag(tiled_copy_t2r, tTR_tAcc[..., 0])
        else:
            tAcc_epi = cute.flat_divide(tAcc, epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[None, None, 0, 0, 0])
            tTR_tAcc = tiled_copy_t2r.get_slice(tidx).partition_S(tAcc_epi)
            tTR_rAcc = copy_utils.tmem_reg_frag(tiled_copy_t2r, tTR_tAcc[..., 0, 0, 0])
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def r2s_C(self, tiled_t2r: cute.TiledCopy, sC, thr, **kwargs):
        # SM100 epilogues chain SMEM stores from the TMEM-load tiled copy so
        # register ownership matches the preceding t2r_C load.
        if self.swap_AB:
            kwargs["transpose"] = not kwargs.get("transpose", False)
            sC = layout_utils.transpose_view(sC)
        return copy_utils.get_smem_store_C(tiled_t2r, sC, thr, **kwargs)

    def s2r_C(self, tiled_t2r: cute.TiledCopy, sC, thr, r_layout, **kwargs):
        """SMEM-load counterpart of `r2s_C`, for epilogue INPUTS that were
        TMA-staged into the output buffer (gemm's C, ssd's z): the register
        fragment is allocated at `r_layout` (pass the t2r fragment's layout)
        so its linear element order matches the t2r/r2s fragments, and
        swap_AB handling mirrors `r2s_C` (transpose_view + transposed copy).

        Unless a `copy_atom` is passed, this defaults to vectorized universal
        loads rather than ldmatrix: the flat TensorSpec SMEM layouts'
        per-thread partitions cannot statically prove ldmatrix's 16B source
        alignment (the store side goes through CUTLASS's SM100 store-op
        selector, which handles that; there is no load-side equivalent).
        Scalar copies under a transposed view, where per-thread elements are
        not contiguous.

        Returns `(tiled_copy, tRS_r, tSR_r, tSR_s)`; load via
        `cute.copy(tiled_copy, tSR_s[..., idx], tSR_r)` then read `tRS_r`."""
        if self.swap_AB:
            kwargs["transpose"] = not kwargs.get("transpose", False)
            sC = layout_utils.transpose_view(sC)
        if "copy_atom" not in kwargs:
            dtype = sC.element_type
            transpose = kwargs.get("transpose", False)
            kwargs["copy_atom"] = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                dtype,
                num_bits_per_copy=dtype.width * (1 if transpose else 2),
            )
        return copy_utils.s2r_partition_from_t2r(tiled_t2r, sC, thr, r_layout, **kwargs)

    def fn(self, acc, zero_init=False, pre_kblock_fn=None):
        """Return a per-iteration tcgen05 GEMM callable.

        This mirrors `BoundMMA.fn` for SM90 call sites. It is intentionally
        stateless: callers only pass logical stage indices and do not thread
        `tiled_mma` through the mainloop:

          `fn(A_idx=..., B_idx=..., acc_idx=...)`

        `acc` can be either a single TMEM accumulator tensor
        `(MMA, MMA_M, MMA_N)` or a staged tensor
        `(MMA, MMA_M, MMA_N, STAGE)`. When `acc_idx` is provided, that stage is
        selected before issuing the MMA.

        For call sequences that accumulate across multiple dynamic calls, pass a
        dynamic `zero_init` flag (true for the first call, false afterward) or
        call the lower-level `gemm(...)` helper directly.
        """

        def _fn(A_idx, B_idx=None, acc_idx=None, zero_init=zero_init, pre_kblock_fn=pre_kblock_fn):
            acc_cur = acc if acc_idx is None else acc[None, None, None, acc_idx]
            self.gemm(
                acc_cur,
                A_idx,
                stage_B=B_idx,
                zero_init=zero_init,
                pre_kblock_fn=pre_kblock_fn,
            )

        return _fn

    def gemm(
        self, acc, stage, *, stage_B=None, tiled_mma=None, zero_init=False, pre_kblock_fn=None
    ):
        """Issue the unrolled k-block MMAs of one stage into `acc`.

        Call from the MMA warp of the leader CTA, between the pipeline
        consumer_wait and consumer_release for the operand stages — the
        pipeline choreography stays with the kernel.

        - `stage` indexes frag_A's stage mode; `stage_B` defaults to `stage`
          and exists for kernels whose A and B operands come from different
          pipelines (e.g. linear attention, where Q and K have different
          stage counts).
        - `zero_init=True` clears ACCUMULATE before the first k-block. In a
          dynamic K-tile loop, pass `zero_init=(k_tile == 0)` so only the first
          call starts a fresh accumulation.
        - `pre_kblock_fn(mma_atom, k_blk)` runs before each k-block's gemm —
          e.g. blockscaled SFA/SFB tmem pointer updates via `mma_atom.set`.
        """
        if const_expr(self.swap_AB):
            logical_A_stage = stage
            logical_B_stage = stage if const_expr(stage_B is None) else stage_B
            stage = logical_B_stage
            stage_B = logical_A_stage
        tm = tiled_mma if tiled_mma is not None else self.tiled_mma
        spec_mma.gemm_sm100(
            tm,
            acc,
            self.frag_A,
            self.frag_B,
            stage,
            stage_B=stage_B,
            zero_init=zero_init,
            pre_kblock_fn=pre_kblock_fn,
        )


class MatmulSpec:
    """Result of A @ B on two TensorSpecs. Deduces operand major modes from
    storage layout + transposed flag; derives tiler_n from B's N dim.

    `M/N/K` are the FULL logical matmul dims (spec shapes are full tiles);
    `cta_group` is read off the operand specs and must agree between A and B."""

    def __init__(self, A: TensorSpec, B: TensorSpec):
        assert A.shape[1] == B.shape[0], f"matmul shape mismatch: {A.shape} @ {B.shape}"
        assert A.cta_group == B.cta_group, (
            f"matmul operand cta_group mismatch: {A.cta_group} vs {B.cta_group}"
        )
        # A.dtype and B.dtype may differ for mixed-precision MMAs (e.g. fp8 ops
        # with different a/b widths supported by the underlying tiled_mma).
        self.A, self.B = A, B
        self.M, self.K = A.shape
        self.N = B.shape[1]
        self.cta_group = A.cta_group

    def __extract_mlir_values__(self):
        # MatmulSpec is a compile-time role view over two TensorSpecs, but the
        # underlying specs may carry dynamic TMA/gmem values. Forward marshaling
        # to A/B so MatmulSpec locals can remain live across DSL dynamic control
        # flow (`if warp_idx == ...`, dynamic loops, etc.).
        values = []
        a_values = cutlass.extract_mlir_values(self.A)
        b_values = cutlass.extract_mlir_values(self.B)
        values += a_values
        values += b_values
        self._n_A = len(a_values)
        self._n_B = len(b_values)
        return values

    def __new_from_mlir_values__(self, values):
        offset = 0
        A = self.A
        if self._n_A > 0:
            A = cutlass.new_from_mlir_values(self.A, values[offset : offset + self._n_A])
            offset += self._n_A
        B = self.B
        if self._n_B > 0:
            B = cutlass.new_from_mlir_values(self.B, values[offset : offset + self._n_B])
            offset += self._n_B
        return MatmulSpec(A, B)

    def tiled_mma(
        self,
        source: Literal["SS", "RS", "TS"] = "SS",
        *,
        acc_dtype: Type[cutlass.Numeric] = Float32,
        atom_layout_mnk: Tuple[int, int, int] = (1, 1, 1),
        permutation_mnk: Optional[Tuple[int, int, int]] = None,
        arch=None,
    ) -> cute.TiledMma:
        # cta_group is a storage property of the operand specs, not an MMA
        # parameter — make_tiled_mma_for_arch reads it off `self.cta_group`.
        return spec_mma.make_tiled_mma_for_arch(
            self,
            source=source,
            atom_layout_mnk=atom_layout_mnk,
            acc_dtype=acc_dtype,
            permutation_mnk=permutation_mnk,
            arch=arch,
        )

    def bind_mma(
        self,
        thr=None,
        *,
        sA: Optional[cute.Tensor] = None,
        sB: Optional[cute.Tensor] = None,
        source: Literal["SS", "RS", "TS"] = "SS",
        tmem_A_ptr=None,
        atom_layout_mnk: Tuple[int, int, int] = (1, 1, 1),
        acc_dtype: Type[cutlass.Numeric] = Float32,
        permutation_mnk: Optional[Tuple[int, int, int]] = None,
        tiled_mma: Optional[cute.TiledMma] = None,
        swap_AB: bool = False,
        bind_operands: bool = True,
        arch=None,
    ) -> BoundMMA:
        """Build an arch-appropriate BoundMMA.

        SM90/SM120 use the register-fragment path and accept `source="SS"` or
        `"RS"`. SM100 uses the tcgen05/TMEM path and accepts `source="SS"` or
        `"TS"`; passing `tmem_A_ptr` implies `"TS"`.

        - `thr` is the SM90/SM120 index (or layout) passed to
          `tiled_mma.get_slice(...)`.
          Pass `tidx` for single-thread frags or a wg-thread layout for warp-
          group-partitioned frags. When omitted (`thr=None`), frag construction
          is skipped (`frag_A`/`frag_B` are None) — useful when the caller only
          needs the s2r/r2s tiled_copies and builds its own frags differently.
        - `sA`/`sB` default to `spec.smem`, **auto-transposed when the operand's
          major mode is "MN"**.
        - For `source="RS"`, `frag_A` is auto-allocated as an rmem tensor with
          shape `tiled_mma.partition_shape_A((M, K))`; on SM90 the physical A
          operand is treated as K-major even if the logical TensorSpec also has
          an MN-major SMEM backing.
        - `tiled_mma`: pass a pre-built TiledMma to bypass the arch-dispatched
          default (e.g. for warp-level MMA with custom permutation_mnk on SM120,
          or any non-default MMA op selection).
        - `bind_operands=False`: return a layout-only BoundMMA/BoundMMASm100
          with `frag_A`/`frag_B` left as None. This is useful for sizing a TMEM
          accumulator before operand storage has been bound.
        - `swap_AB=True`: physically compute `(B.T @ A.T)` instead of `(A @ B)` —
          useful when logical M is too large but logical N would fit a single
          wgmma instance (e.g. M=128, N=64 → 2 wgmma; swap to 64×128 → 1 wgmma).
          The user keeps thinking in logical (A, B) terms; the returned BoundMMA's
          `.acc()`/`.fn()`/`.r2s_C()` handle the physical swap automatically.
          On SM100, `source="TS"` with `swap_AB=True` means logical B.T is the
          physical TMEM A operand; `.fn(...)(A_idx=, B_idx=)` still accepts
          logical indices and routes to physical stages.
        """
        if arch is None:
            arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        if arch.major in [10, 11]:
            assert thr is None, "SM100 bind_mma does not use a thread slice; pass thr=None"
            phys = MatmulSpec(self.B.T, self.A.T) if swap_AB else self
            if phys.A.tmem is not None:
                assert tmem_A_ptr is None, "pass either A.with_tmem(...) or tmem_A_ptr, not both"
                source = "TS"
            source = "TS" if tmem_A_ptr is not None else source
            if tiled_mma is None:
                tiled_mma = phys.tiled_mma(
                    acc_dtype=acc_dtype,
                    source=source,
                    permutation_mnk=permutation_mnk,
                    arch=arch,
                )
            full_m, full_n, full_k = self.mma_tiler_mnk(tiled_mma)
            if not bind_operands:
                frag_A, frag_B = None, None
            elif phys.A.tmem is not None:
                assert sA is None, "TMEM A source cannot also pass sA"
                frag_A = phys.tmem_view_A(tiled_mma)
            elif tmem_A_ptr is not None:
                assert sA is None, "pass either sA (SMEM A) or tmem_A_ptr (TMEM A), not both"
                frag_A = phys.frag_A_tmem(tiled_mma, tmem_A_ptr)
            else:
                assert source != "TS", "SM100 TS MMA requires a bound TMEM physical A operand"
                frag_A = tiled_mma.make_fragment_A(phys.smem_view_A(tiled_mma, sA))
            if bind_operands:
                sB = phys.smem_view_B(tiled_mma, sB)
            return BoundMMASm100(
                tiled_mma=tiled_mma,
                frag_A=frag_A,
                frag_B=None if not bind_operands else tiled_mma.make_fragment_B(sB),
                M=full_m,
                N=full_n,
                K=full_k,
                A=self.A,
                B=self.B,
                swap_AB=swap_AB,
            )

        assert self.cta_group == 1, (
            f"{arch.name} bind_mma requires cta_group=1 operands, got {self.cta_group}"
        )
        assert source in ("SS", "RS"), f"{arch.name} bind_mma source must be SS or RS, got {source}"
        # `phys` is the *physical* MatmulSpec — what the wgmma is actually built
        # against. When swap_AB, we flip operand roles: logical A becomes the
        # physical B operand and vice versa (i.e., compute (B.T @ A.T) on hardware).
        # Downstream code uses `phys.{A,B,M,N,K}` to construct tiled_mma + frags +
        # smem partitions; the returned BoundMMA records the user's *logical*
        # (M, N, K) and the swap_AB flag, so .acc/.fn/.r2s_C reconcile internally.
        phys = MatmulSpec(self.B.T, self.A.T) if swap_AB else self
        if tiled_mma is None:
            tiled_mma = phys.tiled_mma(
                source=source,
                atom_layout_mnk=atom_layout_mnk,
                acc_dtype=acc_dtype,
                permutation_mnk=permutation_mnk,
                arch=arch,
            )
        if thr is not None:
            thr_mma = tiled_mma.get_slice(thr)
            # Lazily resolve sA/sB so register-only specs (e.g. source="RS" with
            # an rmem-only A like dP) don't try to dereference a missing .smem.
            sB_eff = sB if sB is not None else phys._smem_for(phys.B, is_A=False)
            # Arch-dispatched frag construction. WGMMA (Hopper) frags are multi-stage
            # descriptors that index into SMEM via A_idx/B_idx. Warp-level (SM120)
            # frags are single-stage rmem tensors — the ldmatrix SMEM->RMEM step is
            # the kernel's mainloop concern.
            if arch.major in [8, 12]:  # Warp-level — single-stage rmem frags
                if source == "RS":
                    frag_A = cute.make_rmem_tensor(
                        tiled_mma.partition_shape_A((phys.M, phys.K)), phys.A.dtype
                    )
                else:
                    sA_eff = sA if sA is not None else phys._smem_for(phys.A, is_A=True)
                    frag_A = tiled_mma.make_fragment_A(
                        thr_mma.partition_A(sA_eff)[None, None, None, 0]
                    )
                frag_B = tiled_mma.make_fragment_B(thr_mma.partition_B(sB_eff)[None, None, None, 0])
            else:  # WGMMA (Hopper) — multi-stage descriptors
                if source == "RS":
                    frag_A = cute.make_rmem_tensor(
                        tiled_mma.partition_shape_A((phys.M, phys.K)), phys.A.dtype
                    )
                else:
                    sA_eff = sA if sA is not None else phys._smem_for(phys.A, is_A=True)
                    frag_A = tiled_mma.make_fragment_A(thr_mma.partition_A(sA_eff))
                frag_B = tiled_mma.make_fragment_B(thr_mma.partition_B(sB_eff))
        else:
            frag_A, frag_B = None, None
        # Ldmatrix/stmatrix copy atoms — generic (works on any arch). Caller derives
        # the per-stage SMEM partition view at the use site:
        #   smem_view = mma.tiled_copy_s2r_A.get_slice(thr).partition_S(sA)
        a_transpose = phys.A.layout.is_m_major_a()
        b_transpose = phys.B.layout.is_n_major_b()
        tiled_copy_s2r_A = cute.make_tiled_copy_A(
            copy_utils.get_smem_load_atom(phys.A.dtype, transpose=a_transpose), tiled_mma
        )
        tiled_copy_s2r_B = cute.make_tiled_copy_B(
            copy_utils.get_smem_load_atom(phys.B.dtype, transpose=b_transpose), tiled_mma
        )
        tiled_copy_r2s_A = cute.make_tiled_copy_A(
            copy_utils.get_smem_store_atom(phys.A.dtype, transpose=a_transpose), tiled_mma
        )
        tiled_copy_r2s_B = cute.make_tiled_copy_B(
            copy_utils.get_smem_store_atom(phys.B.dtype, transpose=b_transpose), tiled_mma
        )
        # M/N/K are LOGICAL (the user's matmul). swap_AB is the only flag the
        # downstream BoundMMA needs to reconcile logical ↔ physical.
        return BoundMMA(
            tiled_mma=tiled_mma,
            frag_A=frag_A,
            frag_B=frag_B,
            M=self.M,
            N=self.N,
            K=self.K,
            tiled_copy_s2r_A=tiled_copy_s2r_A,
            tiled_copy_s2r_B=tiled_copy_s2r_B,
            tiled_copy_r2s_A=tiled_copy_r2s_A,
            tiled_copy_r2s_B=tiled_copy_r2s_B,
            swap_AB=swap_AB,
        )

    def _smem_for(self, t: TensorSpec, is_A: bool) -> cute.Tensor:
        """Return the SMEM view to feed partition_{A,B}.

        partition_A wants storage in (M, K) order; partition_B wants (N, K).
        The spec's logical `shape` respects `.T` — physical storage_shape is
        `(shape[1], shape[0]) if transposed else shape`. So whether we need a
        transpose-view is fully determined by `transposed`:
          - A: transposed → storage is (K, M), need flip to (M, K).
          - B: not transposed → storage is (K, N), need flip to (N, K).
        The `layout` (ROW_MAJOR/COL_MAJOR) only affects the storage major mode
        (which dim is contiguous), which is orthogonal to shape order and is
        already handled in `_operand_major` for tiled_mma construction."""
        needs_transpose = t.transposed if is_A else not t.transposed
        return layout_utils.transpose_view(t.smem) if needs_transpose else t.smem

    # ---- SM100 (tcgen05) ----------------------------------------------------
    # Role is an MMA concern, so it lives here rather than on the TensorSpec:
    # the same spec (and the same SMEM bytes) can be the A operand of one MMA
    # and the B operand of another. Spec shapes are full logical tiles; peer
    # distribution (cta_group) is a storage property of the specs — the 2-CTA
    # MMA splits A along M and B along N, each peer CTA holding half of both,
    # and each spec's `storage_shape` is its per-CTA shard.
    # The role-nested layouts below are byte-identical to the specs' flat
    # `smem_layout()` (same swizzle, same addressing); they only differ in the
    # mode nesting that `partition_A/B` / `make_fragment_A/B` expect.

    def mma_tiler_mnk(self, tiled_mma: cute.TiledMma) -> Tuple[int, int, int]:
        """Full MMA tile (M, N, K). Spec shapes are full logical tiles, so this
        is just (M, N, K); the tiled_mma's cta_group (read off `thr_id`) is
        validated against the operands' storage distribution."""
        mma_cta_group = cute.size(tiled_mma.thr_id.shape)
        assert mma_cta_group == self.cta_group, (
            f"tiled_mma cta_group {mma_cta_group} != operand spec cta_group {self.cta_group}"
        )
        return (self.M, self.N, self.K)

    def smem_layout_A(self, tiled_mma: cute.TiledMma, *, stage: Optional[int] = None):
        """Role-nested staged SMEM layout for the A operand — the
        ((atom), rest_m, rest_k, stage) view that `partition_A` /
        `make_fragment_A` expect."""
        stage = stage if stage is not None else self.A.stage
        return sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler_mnk(tiled_mma), self.A.dtype, stage
        )

    def smem_layout_B(self, tiled_mma: cute.TiledMma, *, stage: Optional[int] = None):
        """Role-nested staged SMEM layout for the B operand — the
        ((atom), rest_n, rest_k, stage) view that `partition_B` /
        `make_fragment_B` expect."""
        stage = stage if stage is not None else self.B.stage
        return sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler_mnk(tiled_mma), self.B.dtype, stage
        )

    @staticmethod
    def _view_smem_as(smem: cute.Tensor, layout) -> cute.Tensor:
        return cute.make_tensor(smem.iterator, layout.outer if hasattr(layout, "outer") else layout)

    def smem_view_A(
        self,
        tiled_mma: cute.TiledMma,
        smem: Optional[cute.Tensor] = None,
        *,
        stage: Optional[int] = None,
    ) -> cute.Tensor:
        """tcgen05 A-operand view over flat TensorSpec storage."""
        smem = smem if smem is not None else self.A.smem
        assert smem is not None, "A smem not bound — call with_smem(...) or pass smem"
        return self._view_smem_as(smem, self.smem_layout_A(tiled_mma, stage=stage))

    def smem_view_B(
        self,
        tiled_mma: cute.TiledMma,
        smem: Optional[cute.Tensor] = None,
        *,
        stage: Optional[int] = None,
    ) -> cute.Tensor:
        """tcgen05 B-operand view over flat TensorSpec storage."""
        smem = smem if smem is not None else self.B.smem
        assert smem is not None, "B smem not bound — call with_smem(...) or pass smem"
        return self._view_smem_as(smem, self.smem_layout_B(tiled_mma, stage=stage))

    def tmem_view_A(
        self,
        tiled_mma: cute.TiledMma,
        tmem: Optional[cute.Tensor] = None,
        *,
        stage: Optional[int] = None,
    ) -> cute.Tensor:
        """tcgen05 A-operand TMEM view over flat TensorSpec TMEM storage.

        The multi-stage stride comes from the bound TMEM tensor: aliased
        storage (e.g. a bf16 P ring over a wider f32 accumulator ring, see
        `alias_acc_as_tmem`) strides its stages by the ALIASED region's
        footprint, not by this operand's own column footprint."""
        tmem = tmem if tmem is not None else self.A.tmem
        assert tmem is not None, "A tmem not bound — call with_tmem(...) or pass tmem"
        stage_stride = None
        if cute.rank(tmem.layout) >= 3 and cute.size(tmem.layout, mode=[2]) > 1:
            stage_stride = tmem.layout.stride[2]
        return self.frag_A_tmem(tiled_mma, tmem.iterator, stage=stage, stage_stride=stage_stride)

    def acc_layout_sm100(self, tiled_mma: cute.TiledMma, *, stages: Optional[int] = None):
        """Staged TMEM accumulator layout ((MMA, MMA_M, MMA_N[, STAGE])) for
        this matmul. Standalone so warps that never bind operand fragments
        (e.g. epilogue warps reading the accumulator) can build the tensor
        with just the tiled_mma: `cute.make_tensor(tmem_ptr, layout)`."""
        shape = tiled_mma.partition_shape_C(self.mma_tiler_mnk(tiled_mma)[:2])
        if stages is not None:
            shape = cute.append(shape, stages)
        return tiled_mma.make_fragment_C(shape).layout

    def frag_A_tmem(
        self,
        tiled_mma: cute.TiledMma,
        tmem_ptr,
        *,
        stage: Optional[int] = None,
        stage_stride: Optional[int] = None,
    ) -> cute.Tensor:
        """TMEM-resident A-operand fragment ((MMA, MMA_M, MMA_K, STAGE)) at
        `tmem_ptr`, for MMAs built with `source="TS"` (the A tile is produced
        into TMEM by a previous stage, e.g. linear attention's masked Q@K^T fed
        to P@V). Also usable standalone by the warp that *writes* the operand
        into TMEM (tcgen05 store partitioning).

        `stage_stride` (elements of the operand dtype) overrides the trailing
        stage-mode stride, for storage whose stages are NOT packed at this
        operand's own footprint (e.g. a bf16 ring aliased over a wider f32
        accumulator ring)."""
        layout = self.smem_layout_A(tiled_mma, stage=stage)
        shape = layout.outer.shape if hasattr(layout, "outer") else layout.shape
        fake = tiled_mma.make_fragment_A(shape)
        frag_layout = fake.layout
        if stage_stride is not None:
            rank = cute.rank(frag_layout)
            new_stride = tuple(frag_layout.stride[i] for i in range(rank - 1)) + (stage_stride,)
            frag_layout = cute.make_layout(frag_layout.shape, stride=new_stride)
        return cute.make_tensor(cute.recast_ptr(tmem_ptr, dtype=fake.element_type), frag_layout)

    @staticmethod
    def _operand_major(t: TensorSpec, is_A: bool) -> Literal["K", "MN"]:
        # Which logical matmul dim (K vs MN) is the fast dim in storage?
        # Register-only A operands (in_rmem, no SMEM layout) follow CuTe's K-major
        # fragment convention regardless of the spec's `transposed` flag. This is
        # a WGMMA (SM90/SM120) concern only — the SM100 path uses
        # `_storage_major` directly, both because tcgen05's A never lives in
        # rmem and because `stage=None` there can simply mean "not yet known".
        if is_A and t.in_rmem:
            return "K"
        return MatmulSpec._storage_major(t, is_A)

    @staticmethod
    def _storage_major(t: TensorSpec, is_A: bool) -> Literal["K", "MN"]:
        # ROW_MAJOR storage: storage[1] is fast; COL_MAJOR: storage[0] is fast.
        # For A: matmul (M, K) maps to storage (0, 1) untransposed, (1, 0) transposed.
        # For B: matmul (K, N) maps to storage (0, 1) untransposed, (1, 0) transposed.
        is_row_major = t.layout == LayoutEnum.ROW_MAJOR
        if is_A:
            base = "K" if is_row_major else "MN"
        else:
            base = "MN" if is_row_major else "K"
        return base if not t.transposed else ("MN" if base == "K" else "K")
