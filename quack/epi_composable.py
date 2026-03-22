# Copyright (c) 2025, Tri Dao.
"""ComposableEpiMixin: composes EpiOps into epilogue hook methods.

Subclasses declare _epi_ops as a tuple of EpiOp instances. The mixin auto-generates
epi_smem_bytes_per_stage, epi_get_smem_struct, epi_get_smem_tensors, epi_begin,
epi_begin_loop, epi_end, and EpilogueParams by querying each op.

epi_begin and epi_begin_loop return dicts keyed by op name, so epi_visit_subtile
can access values by name (e.g. epi_loop_tensors["alpha"]).

EpilogueParams is auto-generated from _epi_ops (via param_fields()) plus any
_extra_param_fields declared on the subclass. Subclasses still define
EpilogueArguments and epi_to_underlying_arguments manually.
"""

from dataclasses import make_dataclass, MISSING

import cutlass.cute as cute
from cutlass import const_expr

from quack.epi_ops import EpiContext, Scalar


def _compute_smem_map(ops):
    """Pre-compute name → smem tensor index for each non-Scalar op."""
    smem_map = {}
    idx = 0
    for op in ops:
        if not isinstance(op, Scalar):
            smem_map[op.name] = idx
            idx += 1
    return smem_map


def _make_epi_params(epi_ops, extra_fields, bases):
    """Build EpilogueParams dataclass from epi_ops + extra fields.

    Required fields (default=MISSING) are placed first, then optional fields.
    """
    required, optional = [], []
    for op in epi_ops:
        for name, typ, default in op.param_fields():
            (required if default is MISSING else optional).append((name, typ, default))
    for name, typ, default in extra_fields:
        (required if default is MISSING else optional).append((name, typ, default))
    fields = [(n, t) for n, t, _ in required] + [(n, t, d) for n, t, d in optional]
    return make_dataclass("EpilogueParams", fields, bases=bases)


class ComposableEpiMixin:
    """Base mixin that composes EpiOps into the standard epilogue hooks."""

    _epi_ops = ()
    _extra_param_fields = ()  # [(name, type, default), ...] for non-op params (e.g. act_fn)
    _epi_param_bases = ()  # Base classes for EpilogueParams (e.g. (ParamsBase,))
    _epi_smem_map = {}
    _epi_has_async_ops = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._epi_ops:
            cls._epi_smem_map = _compute_smem_map(cls._epi_ops)
            cls._epi_has_async_ops = any(op.needs_async_fence() for op in cls._epi_ops)
            # Auto-generate EpilogueParams if not explicitly defined on this class
            if "EpilogueParams" not in cls.__dict__:
                cls.EpilogueParams = _make_epi_params(
                    cls._epi_ops, cls._extra_param_fields, cls._epi_param_bases
                )

    # --- Host-side: args → params ---

    def _epi_ops_to_params_dict(self, args):
        """Merge each op's to_params into a single dict. Subclasses call this,
        add custom fields, then construct self.EpilogueParams(**d)."""
        d = {}
        for op in self._epi_ops:
            d.update(op.to_params(self, args))
        return d

    # --- Host-side: smem allocation (queried from ops) ---

    @classmethod
    def epi_smem_bytes_per_stage(cls, args, cta_tile_shape_mnk, epi_tile):
        return sum(
            op.smem_bytes(getattr(args, op.name, None), cta_tile_shape_mnk, epi_tile)
            for op in cls._epi_ops
        )

    def epi_get_smem_struct(self, params):
        fields = {}
        for op in self._epi_ops:
            result = op.smem_struct_field(self, params)
            if result is not None:
                name, ftype = result
                fields[name] = ftype
        EpiSharedStorage = type("EpiSharedStorage", (), {"__annotations__": fields})
        return cute.struct(EpiSharedStorage)

    def epi_get_smem_tensors(self, params, storage):
        return tuple(
            op.get_smem_tensor(self, params, storage.epi)
            for op in self._epi_ops
            if not isinstance(op, Scalar)
        )

    def epi_get_tma_atoms(self, params, *, loc=None, ip=None):
        atoms = []
        for op in self._epi_ops:
            atoms.extend(op.tma_atoms(self, params))
        return atoms

    # --- Device-side: kernel execution (delegates to ops) ---

    @cute.jit
    def epi_begin(
        self,
        params,
        epi_smem_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        epilogue_barrier,
        tidx,
    ):
        ctx = EpiContext(
            self,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            epilogue_barrier,
            tidx,
        )
        smem_map = self._epi_smem_map
        results = {
            op.name: op.begin(
                self,
                getattr(params, op.name, None),
                epi_smem_tensors[smem_map[op.name]] if op.name in smem_map else None,
                ctx,
            )
            for op in self._epi_ops
        }
        if const_expr(self._epi_has_async_ops):
            has_async_data = any(
                getattr(params, op.name, None) is not None
                for op in self._epi_ops
                if op.needs_async_fence()
            )
            if const_expr(has_async_data):
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                epilogue_barrier.arrive_and_wait()
        return results

    def epi_begin_loop(self, params, epi_tensors, epi_coord):
        return {
            op.name: op.begin_loop(self, epi_tensors[op.name], epi_coord) for op in self._epi_ops
        }

    @cute.jit
    def epi_end(
        self,
        params,
        epi_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        for op in self._epi_ops:
            op.end(
                self,
                getattr(params, op.name, None),
                epi_tensors[op.name],
                epi_tile,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tile_coord_mnkl,
                varlen_manager,
                tidx,
            )
