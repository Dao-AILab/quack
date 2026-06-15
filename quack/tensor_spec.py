# Copyright (c) 2025-2026, Tri Dao.

"""Spec abstractions for declarative SM90 kernel operands.
Note: this is a prototype and the API could change rapidly. While this seems to be working
well for Sm90, the design doesn't work well with Sm100 and will need a revision soon.

`TensorSpec` is a declarative description of a 2D operand tile (dtype, shape,
SMEM stage, layout) that drives SMEM layout creation, TMA atom construction,
TMA pipelines, MMA configuration, and per-warpgroup operand partitioning.

`MatmulSpec` (returned by `A @ B` on two TensorSpecs) deduces operand major
modes from `(layout, transposed, is_A)` and `tiler_n` from B's N dim. Calling
`bind_mma(thr)` returns a `BoundMMA` with the per-warpgroup partitioned A/B
fragments and shape (M, N, K) ready for `acc()` / `gemm()` / `fn()`.

Designed to be marshaled across the `@cute.kernel` boundary: `tma_atom` and
`gmem` cross via `__extract_mlir_values__` / `__new_from_mlir_values__`;
`smem` is populated inside the kernel via `with_smem(storage_field)` and
preserved from the host-side template (it lives in JIT-local scope, so cute
doesn't need to marshal it).
"""

from dataclasses import dataclass, replace
from typing import Any, Literal, Optional, Tuple, Type
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.nvgpu import cpasync, warp, tcgen05
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import OperandMajorMode
import cutlass.utils.blackwell_helpers as sm100_utils


from quack import copy_utils, layout_utils
from quack import sm90_utils
from quack import pipeline as pipeline_custom


def _make_warp_tiled_mma(
    a_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    atom_layout_mnk: Tuple[int, int, int],
    mma_inst_mnk: Tuple[int, int, int] = (16, 8, 16),
) -> cute.TiledMma:
    """Build a warp-level TiledMma for SM120 (consumer / RTX-Blackwell).
    Mirrors `gemm_sm120._setup_tiled_mma`'s permutation_mnk derivation."""
    if a_dtype.width == 16:  # fp16 / bf16
        op = warp.MmaF16BF16Op(a_dtype, acc_dtype, mma_inst_mnk)
    else:
        raise NotImplementedError(
            f"warp-level MMA backend doesn't yet support a_dtype={a_dtype} (width={a_dtype.width})"
        )
    tC = cute.make_layout(atom_layout_mnk)
    atom_m, atom_n, atom_k = atom_layout_mnk
    # The N dim is multiplied by 2 to leverage ldmatrix.x4 (matches the reference
    # blackwell_geforce/dense_gemm.py). A nested-layout permutation_n adds extra
    # modes to partition_A/B output, which breaks the standard mainloop slicing.
    permutation_mnk = (
        atom_m * mma_inst_mnk[0],
        atom_n * mma_inst_mnk[1] * 2,
        atom_k * mma_inst_mnk[2],
    )
    return cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)


def make_tiled_mma_for_arch(
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    a_major: Literal["K", "MN"],
    b_major: Literal["K", "MN"],
    tiler_n: int,
    source: Literal["SS", "RS"] = "SS",
    atom_layout_mnk: Tuple[int, int, int] = (1, 1, 1),
    acc_dtype: Type[cutlass.Numeric] = Float32,
) -> cute.TiledMma:
    """Arch-dispatched TiledMma builder. Hopper → WGMMA via sm90_utils.make_tiled_mma;
    SM120 consumer → warp-level via _make_warp_tiled_mma. Free function so kernel
    classes can call this directly when their A/B TensorSpecs aren't yet set up
    (e.g. inside `_setup_tiled_mma` where K is still being resolved)."""
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    if arch.major == 9:  # Hopper — WGMMA
        return sm90_utils.make_tiled_mma(
            a_dtype,
            a_major,
            b_major,
            tiler_n,
            source=source,
            atom_layout_mnk=atom_layout_mnk,
            b_dtype=b_dtype,
            acc_dtype=acc_dtype,
        )
    elif arch.major == 12:  # SM120 consumer — warp-level MMA
        return _make_warp_tiled_mma(a_dtype, acc_dtype, atom_layout_mnk)
    raise NotImplementedError(
        f"make_tiled_mma_for_arch has no backend for {arch.name} (major={arch.major})."
    )


# Axis-pattern SMEM layout helpers (Mojo-style: keyed on K-major / MN-major,
# not on operand role). Each builds a dummy SM100 tiled_mma internally and
# dispatches to `sm100_utils.make_smem_layout_a/_b`. The dummy's other-operand
# fields are filled with safe defaults — undocumented but stable across cute
# SDK versions in our use; if you hit silent layout mismatches after a cute
# upgrade, drop down to the SDK helpers directly with an explicit tiled_mma.


def make_smem_layout_kmajor(
    dtype: Type[cutlass.Numeric],
    shape: Tuple[int, int],
    stages: int,
    *,
    cta_group: int = 1,
):
    """SMEM layout where the operand's K axis is the **fast (cols) storage dim**.

    Use for operands that read K-major: A direct (cols=K), or B via .T
    (cols=K after transpose). For row-major K-fast storage this is the
    natural layout — the most common case.

    `shape = (rows, cols)`. For cta_group=2 A-side, `rows` is the per-CTA M
    (full M = rows * cta_group, split across the two CTAs).
    """
    cta_group_enum = tcgen05.CtaGroup.TWO if cta_group == 2 else tcgen05.CtaGroup.ONE
    M, K = shape[0] * cta_group, shape[1]
    dummy = sm100_utils.make_trivial_tiled_mma(
        dtype,
        dtype,
        OperandMajorMode.K,
        OperandMajorMode.K,
        Float32,
        cta_group_enum,
        (M, 64),  # 64 = small valid dummy N
    )
    tiler = (M, 0, K)
    return sm100_utils.make_smem_layout_a(dummy, tiler, dtype, stages)


def make_smem_layout_mnmajor(
    dtype: Type[cutlass.Numeric],
    shape: Tuple[int, int],
    stages: int,
    *,
    cta_group: int = 1,
):
    """SMEM layout where the operand's K axis is the **slow (rows) storage dim**.

    Use for operands that read MN-major: A via .T (rows=K after transpose),
    or B direct (rows=K). The cols dim is N (or M); it remains fast in storage.

    `shape = (rows, cols)`. The SDK reads `(N=rows, K=cols)` from the mma_tiler
    via `dice (None, 1, 1)` — so this matches the existing TensorSpec convention
    where shape[0] is treated as the operand's leading dim.
    """
    cta_group_enum = tcgen05.CtaGroup.TWO if cta_group == 2 else tcgen05.CtaGroup.ONE
    N, K = shape[0], shape[1]
    dummy = sm100_utils.make_trivial_tiled_mma(
        dtype,
        dtype,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        Float32,
        cta_group_enum,
        (64, N),  # 64 = small valid dummy M
    )
    tiler = (0, N, K)
    return sm100_utils.make_smem_layout_b(dummy, tiler, dtype, stages)


@dataclass
class TensorSpec:
    """Declarative spec for an operand tile. Owns dtype/shape/layout/stage so
    SMEM layouts, TMA atoms, and MMA configs can be derived from it.

    Shape rank:
      - 2D `(rows, cols)` — matmul operand tiles. Drives `MatmulSpec`/`bind_mma`,
        SM90/SM100 swizzled SMEM layouts, and 2-mode TMA atoms.
      - 1D `(vec,)` — vector-with-stage aux operands (Scale, Bias, Gamma, ...).
        `smem_layout()` returns a trivial `cute.make_layout((vec, stage))` (no
        swizzle); `tma_copy_bytes()` and `make_tma_atom()` use `mode=[0]`. Cannot
        be used as a matmul operand (`__matmul__` / `bind_mma` are 2D-only).

    `stage=None` means the tile lives in registers (no SMEM layout, no TMA).
    `transposed=True` is a logical .T view of the same storage (2D only).

    After `with_tma(op, gmem)`, the returned spec also carries the call-dynamic
    `tma_atom` and `gmem` (TMA tensor). The bound spec crosses the `@cute.kernel`
    boundary as a single arg — only the cute-object fields are MLIR-marshaled;
    static fields are preserved from the host-side template."""

    dtype: Type[cutlass.Numeric]
    shape: Tuple[
        int, ...
    ]  # 2D for matmul operands; 1D supported for vector-with-stage aux operands
    stage: Optional[int] = None
    layout: LayoutEnum = LayoutEnum.ROW_MAJOR
    transposed: bool = False
    tma_atom: Optional[cute.CopyAtom] = None
    gmem: Optional[cute.Tensor] = None
    smem: Optional[cute.Tensor] = None  # populated inside the kernel via with_smem()
    # Override for SMEM layout. When set, `smem_layout()` returns this instead of
    # deriving. `with_tiled_mma()` populates this; `with_smem_layout()` exposes it.
    smem_layout_override: Optional[Any] = None
    # SM100 fields. cta_group=1 (default) means single-CTA MMA — role is irrelevant.
    # cta_group=2 means the MMA spans 2 CTAs and SPLITS A's M dim (B is shared);
    # the layout differs per role, so `mma_role` ("A" or "B") must be set, and
    # the spec is essentially per-MMA (a 2-CTA tensor that needs to be both A and
    # B in different MMAs requires two specs with different SMEM allocations).
    # `mma_tiler_mnk` is derived from `(shape, cta_group, role)`:
    #   - A: (shape[0] * cta_group, _, shape[1])  — M scales, N irrelevant for A
    #   - B: (_, shape[0],          shape[1])     — N is per-CTA, M irrelevant for B
    cta_group: int = 1
    mma_role: Optional[Literal["A", "B"]] = None
    # When True, `smem_layout()` dispatches to `sm100_utils.make_smem_layout_epi`
    # — `shape` is interpreted as the epi tile. Use for SM100 epilogue staging
    # buffers (output operand, etc.) where the layout isn't an MMA-side layout.
    is_epi: bool = False
    # Bound tiled_mma for SM100 layout derivation. Marshaled across the
    # @cute.kernel boundary (so the in-region tiled_mma is used to build the
    # layout inside the kernel — caching the layout from host doesn't work
    # because the layout value is region-local).
    _tiled_mma: Optional[cute.TiledMma] = None
    # Axis-pattern hint (Mojo-style). When set, `smem_layout()` builds the layout
    # lazily via `make_smem_layout_kmajor` / `_mnmajor`. Mma-role-free —
    # determines order/swizzle from the operand's storage axis pattern alone.
    # Bypasses the role/tiled_mma path entirely; preferred over `mma_role` for
    # cta_group=1.
    smem_axis_pattern: Optional[Literal["K", "MN"]] = None

    def __post_init__(self):
        if self.cta_group == 2:
            assert self.mma_role in ("A", "B"), (
                "cta_group=2 requires mma_role='A' or 'B' — the 2-CTA MMA splits "
                "A's M dim, so the SMEM layout differs per role"
            )
        if self.is_epi:
            assert self.mma_role is None, "is_epi and mma_role are mutually exclusive"

    def _mma_tiler_for_layout(self) -> Tuple[int, int, int]:
        """Reconstruct the mma_tiler tuple needed by sm100_utils.make_smem_layout_*.
        Only (M, K) for A and (N, K) for B are read by the helpers (via cute.dice),
        so the remaining slot is filled with `0` as a don't-care.

        For cta_group=1 with `mma_role=None`, defaults to the A-style tiler
        (shape[0], 0, shape[1]); the resulting layout is equivalent to the B-style
        one because the swizzle depends only on (dtype, major_mode) and both helpers
        slice out the same (shape[0], shape[1]) tile."""
        if self.mma_role == "B":
            return (0, self.shape[0], self.shape[1])
        return (self.shape[0] * self.cta_group, 0, self.shape[1])

    def _build_dummy_tiled_mma(self) -> "cute.TiledMma":
        """Construct a minimal SM100 tiled_mma sufficient for `partition_shape_A/B`
        on this spec's role. Uses dummy values for the OTHER operand's side
        (b_dtype/b_major/N for A specs; a_dtype/a_major/M for B specs). The cute
        SDK doesn't currently consult those in the role's partition function — but
        this is undocumented behavior. If you hit silent layout mismatches after a
        cute upgrade, pass an explicit `tiled_mma` via `with_tiled_mma()` to bypass
        this path. For cta_group=1 with no explicit role we fall through the A
        branch (the resulting layout matches B's exactly — see `_mma_tiler_for_layout`)."""

        cta_group_enum = tcgen05.CtaGroup.TWO if self.cta_group == 2 else tcgen05.CtaGroup.ONE
        own_major = self.layout.mma_major_mode()
        other_major = OperandMajorMode.K  # dummy — unused for our partition
        if self.mma_role == "B":
            full_N = self.shape[0]
            return sm100_utils.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                other_major,
                own_major,
                Float32,
                cta_group_enum,
                (64, full_N),  # 64 = small valid dummy M
            )
        full_M = self.shape[0] * self.cta_group
        return sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            own_major,
            other_major,
            Float32,
            cta_group_enum,
            (full_M, 64),  # 64 = small valid dummy N
        )

    def __extract_mlir_values__(self):
        # Marshal tma_atom + gmem + _tiled_mma across the kernel boundary. `smem`
        # is created inside the kernel via with_smem() and lives in JIT-local
        # scope; cute can pass it by reference without marshaling.
        values = []
        self._n_atom = 0
        self._n_gmem = 0
        self._n_tmma = 0
        if self.tma_atom is not None:
            v = cutlass.extract_mlir_values(self.tma_atom)
            values += v
            self._n_atom = len(v)
        if self.gmem is not None:
            v = cutlass.extract_mlir_values(self.gmem)
            values += v
            self._n_gmem = len(v)
        if self._tiled_mma is not None:
            v = cutlass.extract_mlir_values(self._tiled_mma)
            values += v
            self._n_tmma = len(v)
        return values

    def __new_from_mlir_values__(self, values):
        offset = 0
        new_atom = None
        if self.tma_atom is not None:
            new_atom = cutlass.new_from_mlir_values(
                self.tma_atom, values[offset : offset + self._n_atom]
            )
            offset += self._n_atom
        new_gmem = None
        if self.gmem is not None:
            new_gmem = cutlass.new_from_mlir_values(
                self.gmem, values[offset : offset + self._n_gmem]
            )
            offset += self._n_gmem
        new_tmma = None
        if self._tiled_mma is not None:
            new_tmma = cutlass.new_from_mlir_values(
                self._tiled_mma, values[offset : offset + self._n_tmma]
            )
            offset += self._n_tmma
        return replace(self, tma_atom=new_atom, gmem=new_gmem, _tiled_mma=new_tmma)

    def with_tma(self, op, gmem_tensor, num_multicast: int = 1) -> "TensorSpec":
        """Return a new spec with this call's tma_atom and gmem TMA tensor attached.
        Pass `num_multicast > 1` for cluster-multicast TMA loads (caller is responsible
        for choosing a multicast-capable `op` like `CopyBulkTensorTileG2SMulticastOp`)."""
        atom, tma_tensor = self.make_tma_atom(op, gmem_tensor, num_multicast=num_multicast)
        return replace(self, tma_atom=atom, gmem=tma_tensor)

    def with_tma_atom(self, tma_atom, gmem) -> "TensorSpec":
        """Return a new spec with a pre-built tma_atom and gmem attached. Use when
        the caller already constructed the atom (e.g. kernel taking raw atom args
        rather than TensorSpec args)."""
        return replace(self, tma_atom=tma_atom, gmem=gmem)

    def with_gmem(self, gmem) -> "TensorSpec":
        """Return a new spec with only a GMEM tensor attached.

        Use for operands that should cross the kernel boundary as TensorSpecs but
        are copied with non-TMA helpers.
        """
        return replace(self, gmem=gmem)

    def with_smem(self, storage_or_tensor) -> "TensorSpec":
        """Return a new spec with the SMEM tensor attached (call inside the kernel).
        Accepts either a SmemAllocator storage field (derives the tensor with this
        spec's layout) or an already-built cute.Tensor (uses it directly — handy
        when the kernel allocated SMEM via an external layout)."""
        if hasattr(storage_or_tensor, "get_tensor"):
            smem = self.get_smem_tensor(storage_or_tensor)
        else:
            smem = storage_or_tensor
        return replace(self, smem=smem)

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

    @property
    def storage_shape(self) -> Tuple[int, ...]:
        # 1D has nothing to transpose.
        if self.rank == 1:
            return self.shape
        return (self.shape[1], self.shape[0]) if self.transposed else self.shape

    def smem_layout(self, tiled_mma=None):
        """Derive the SMEM layout for this operand.

        Resolution order:
        0. 1D shape (vector-with-stage aux operand) → trivial `cute.make_layout`,
           no swizzling. Bypasses all matmul-side layout logic.
        1. `smem_layout_override` set → return it as-is.
        2. `smem_axis_pattern` set → axis-pattern helper (Mojo-style, role-free).
        3. `is_epi` → SM100 epi helper.
        4. SM100 path (`cta_group == 2` or `mma_role` set):
           - `tiled_mma` arg → use it
           - bound `_tiled_mma` → use it
           - else → build a minimal dummy tiled_mma from this spec's own info
             (other-operand side filled with dummies; safe under current cute internals)
           `mma_role` is required because the SM100 helpers consult
           `tiled_mma.op.{a,b}_major_mode` and `partition_shape_{A,B}`, which can
           differ between A and B for the same TiledMma — even at cta_group=1.
        5. Otherwise → SM90 helper (default).

        The layout is built FRESH on every call (not cached). This matters because
        SMEM layout values are MLIR-region-local: a layout built on the host doesn't
        survive crossing into a `@cute.kernel`. Paths 2/4 reconstruct the layout
        from primitive (marshaled) inputs on each call — compile-time only,
        no runtime cost."""
        assert not self.in_rmem, "register tensor has no SMEM layout"
        if self.smem_layout_override is not None:
            return self.smem_layout_override
        if self.rank == 1:
            # 1D vector-with-stage: no swizzling needed (small + naturally aligned).
            return cute.make_layout((self.shape[0], self.stage))
        if self.smem_axis_pattern is not None:
            if self.smem_axis_pattern == "K":
                return make_smem_layout_kmajor(
                    self.dtype, self.shape, self.stage, cta_group=self.cta_group
                )
            return make_smem_layout_mnmajor(
                self.dtype, self.shape, self.stage, cta_group=self.cta_group
            )
        if self.is_epi:
            return sm100_utils.make_smem_layout_epi(self.dtype, self.layout, self.shape, self.stage)
        if self.cta_group == 2 or self.mma_role is not None:
            tm = tiled_mma if tiled_mma is not None else self._tiled_mma
            if tm is None:
                tm = self._build_dummy_tiled_mma()
            mma_tiler = self._mma_tiler_for_layout()
            if self.mma_role == "B":
                return sm100_utils.make_smem_layout_b(tm, mma_tiler, self.dtype, self.stage)
            return sm100_utils.make_smem_layout_a(tm, mma_tiler, self.dtype, self.stage)
        return sm90_utils.make_smem_layout(self.dtype, self.layout, self.storage_shape, self.stage)

    def with_smem_layout(self, layout) -> "TensorSpec":
        """Return a new spec with an externally-built SMEM layout cached on it.
        Use only when you need a layout the spec can't derive (e.g. a custom
        non-canonical layout). For the common SM100 case, prefer `with_tiled_mma`."""
        return replace(self, smem_layout_override=layout)

    def with_tiled_mma(self, tiled_mma) -> "TensorSpec":
        """Bind a tiled_mma reference to the spec (for SM100). Marshals across
        the @cute.kernel boundary; downstream `smem_layout()` rebuilds the layout
        in-region using the bound tiled_mma. No layout caching — see `smem_layout`
        docstring for why."""
        return replace(self, _tiled_mma=tiled_mma)

    def with_axis_pattern(self, pattern: Literal["K", "MN"]) -> "TensorSpec":
        """Set the axis-pattern hint (Mojo-style). `smem_layout()` then dispatches
        to `make_smem_layout_kmajor` / `_mnmajor` — role-free, mma-agnostic.
        Layout is rebuilt fresh inside the kernel region so MLIR isolation holds."""
        return replace(self, smem_axis_pattern=pattern)

    def tma_copy_bytes(self) -> int:
        # 1D specs have a single non-stage mode; 2D specs have two.
        modes = [0] if self.rank == 1 else [0, 1]
        return cute.size_in_bytes(self.dtype, cute.select(self.smem_layout(), mode=modes))

    def make_tma_atom(self, op, gmem_tensor, num_multicast: int = 1):
        modes = [0] if self.rank == 1 else [0, 1]
        return cpasync.make_tiled_tma_atom(
            op,
            gmem_tensor,
            cute.select(self.smem_layout(), mode=modes),
            self.storage_shape,
            num_multicast=num_multicast,
        )

    def smem_struct(self, align: int):
        """The aligned SMEM byte allocation for this spec, for inclusion in a SharedStorage class."""
        return cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.smem_layout())], align
        ]

    def make_tma_pipeline(self, barrier_storage, producer, consumer, **kwargs):
        """Build a TMA pipeline whose stage count and tx_count come from this spec."""
        return pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=barrier_storage,
            num_stages=self.stage,
            producer_group=producer,
            consumer_group=consumer,
            tx_count=self.tma_copy_bytes(),
            **kwargs,
        )

    def get_smem_tensor(self, storage_field):
        """Materialize the SMEM tensor backed by `storage_field` with this spec's layout."""
        layout = self.smem_layout()
        if hasattr(layout, "outer"):
            return storage_field.get_tensor(layout.outer, swizzle=layout.inner)
        return storage_field.get_tensor(layout)

    @property
    def smem_T(self) -> cute.Tensor:
        """`transpose_view` of `self.smem` — the layout-transposed view used as
        partition_B input when the operand's matmul B-side is MN-major. Hot path
        in mamba/linear-attn kernels (sBt = transpose_view(B.smem))."""
        assert self.smem is not None, "smem not bound — call with_smem(...) first"
        return layout_utils.transpose_view(self.smem)

    def tma_load_fn(self, g_tile, cta_coord=0, cta_layout=None, **kwargs):
        """Build a TMA load copy fn (gmem → smem) bound to this spec's tma_atom + smem.
        Defaults `cta_coord=0`, `cta_layout=cute.make_layout(1)` (no multicast).
        Returns the same `(copy_fn, ...)` tuple as `copy_utils.tma_get_copy_fn`."""
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
        """Return a partial that captures `acc`/frags — call per-iteration in a loop.
        `frag_A`/`frag_B` override the bound fragments (multi-stage RS pattern).
        When swap_AB, A_idx/B_idx are user-logical and get swapped internally."""
        fA = frag_A if frag_A is not None else self.frag_A
        fB = frag_B if frag_B is not None else self.frag_B
        inner = partial(sm90_utils.gemm_w_idx, self.tiled_mma, acc, fA, fB, zero_init=zero_init)
        if self.swap_AB:
            # Wrap so logical (A_idx, B_idx) map to physical (B_idx, A_idx); all other
            # kwargs (including caller-overridden zero_init/wg_wait) pass through `inner`.
            def _fn(A_idx=None, B_idx=None, **kw):
                return inner(A_idx=B_idx, B_idx=A_idx, **kw)

            return _fn
        return inner

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


class MatmulSpec:
    """Result of A @ B on two TensorSpecs. Deduces operand major modes from
    storage layout + transposed flag; derives tiler_n from B's N dim."""

    def __init__(self, A: TensorSpec, B: TensorSpec):
        assert A.shape[1] == B.shape[0], f"matmul shape mismatch: {A.shape} @ {B.shape}"
        # A.dtype and B.dtype may differ for mixed-precision MMAs (e.g. fp8 ops
        # with different a/b widths supported by the underlying tiled_mma).
        self.A, self.B = A, B
        self.M, self.K = A.shape
        self.N = B.shape[1]

    def tiled_mma(
        self,
        source: Literal["SS", "RS"] = "SS",
        atom_layout_mnk: Tuple[int, int, int] = (1, 1, 1),
        acc_dtype: Type[cutlass.Numeric] = Float32,
    ) -> cute.TiledMma:
        # Operand major modes always derive from the spec's storage layout.
        # `source` is independent: a kernel can synthesize a register tensor
        # (e.g. dt-modulated B) and still want the MMA configured with the
        # original SMEM operand's major mode.
        a_major = self._operand_major(self.A, is_A=True)
        b_major = self._operand_major(self.B, is_A=False)
        return make_tiled_mma_for_arch(
            self.A.dtype,
            self.B.dtype,
            a_major,
            b_major,
            self.N,
            source=source,
            atom_layout_mnk=atom_layout_mnk,
            acc_dtype=acc_dtype,
        )

    def bind_mma(
        self,
        thr=None,
        sA: Optional[cute.Tensor] = None,
        sB: Optional[cute.Tensor] = None,
        source: Literal["SS", "RS"] = "SS",
        atom_layout_mnk: Tuple[int, int, int] = (1, 1, 1),
        acc_dtype: Type[cutlass.Numeric] = Float32,
        tiled_mma: Optional[cute.TiledMma] = None,
        swap_AB: bool = False,
    ) -> BoundMMA:
        """Build the tiled_mma + per-thread/warp-group partitioned A/B fragments.

        - `thr` is the index (or layout) passed to `tiled_mma.get_slice(...)`.
          Pass `tidx` for single-thread frags or a wg-thread layout for warp-
          group-partitioned frags. When omitted (`thr=None`), frag construction
          is skipped (`frag_A`/`frag_B` are None) — useful when the caller only
          needs the s2r/r2s tiled_copies and builds its own frags differently.
        - `sA`/`sB` default to `spec.smem`, **auto-transposed when the operand's
          major mode is "MN"**.
        - For `source="RS"`, `frag_A` is auto-allocated as an rmem tensor with
          shape `tiled_mma.partition_shape_A((M, K))`.
        - `tiled_mma`: pass a pre-built TiledMma to bypass the arch-dispatched
          default (e.g. for warp-level MMA with custom permutation_mnk on SM120,
          or any non-default MMA op selection).
        - `swap_AB=True`: physically compute `(B.T @ A.T)` instead of `(A @ B)` —
          useful when logical M is too large but logical N would fit a single
          wgmma instance (e.g. M=128, N=64 → 2 wgmma; swap to 64×128 → 1 wgmma).
          The user keeps thinking in logical (A, B) terms; the returned BoundMMA's
          `.acc()`/`.fn()`/`.r2s_C()` handle the physical swap automatically:
          `.acc()` returns a transposed-shape rmem; `.fn(...)(A_idx=, B_idx=)`
          accepts logical indices and routes to physical; `.r2s_C(sC, ...)` auto-
          transposes the smem view + flips the stmatrix transpose bit.
        """
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
            arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
            if arch.major == 12:  # SM120 warp-level — single-stage rmem frags
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

    @staticmethod
    def _operand_major(t: TensorSpec, is_A: bool) -> Literal["K", "MN"]:
        # Which logical matmul dim (K vs MN) is the fast dim in storage?
        # ROW_MAJOR storage: storage[1] is fast; COL_MAJOR: storage[0] is fast.
        # For A: matmul (M, K) maps to storage (0, 1) untransposed, (1, 0) transposed.
        # For B: matmul (K, N) maps to storage (0, 1) untransposed, (1, 0) transposed.
        # Register-only A operands (in_rmem, no SMEM layout) follow CuTe's K-major
        # fragment convention regardless of the spec's `transposed` flag.
        if is_A and t.in_rmem:
            return "K"
        is_row_major = t.layout == LayoutEnum.ROW_MAJOR
        if is_A:
            base = "K" if is_row_major else "MN"
        else:
            base = "MN" if is_row_major else "K"
        return base if not t.transposed else ("MN" if base == "K" else "K")
