# Copyright (c) 2025, Wentao Guo, Tri Dao.
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from quack.cute_dsl_utils import mlir_namedtuple
from quack.epi_composable import ComposableEpiMixin
from quack.epi_ops import Scalar, RowVecLoad, ColVecLoad
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.rounding import RoundingMode
import quack.utils as utils


class GemmDefaultEpiMixin(ComposableEpiMixin):
    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        RowVecLoad("mRowVecBroadcast"),
        ColVecLoad("mColVecBroadcast"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        d = self._epi_ops_to_params_dict(args)
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        alpha = epi_loop_tensors["alpha"]
        beta = epi_loop_tensors["beta"]
        tDrRowVec = epi_loop_tensors["mRowVecBroadcast"]
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        rD = tRS_rD.load()
        # Apply alpha scaling to accumulator if alpha is provided (not None)
        if const_expr(hasattr(params, "alpha") and params.alpha is not None):
            alpha = utils.load_scalar_or_pointer(params.alpha)
            rD *= alpha
        # Apply C with beta scaling
        if const_expr(tRS_rC is not None):
            if const_expr(not hasattr(params, "beta") or params.beta is None):
                # beta is None, default behavior: add C (beta=1.0)
                rD += tRS_rC.load().to(tRS_rD.element_type)
            else:
                beta = utils.load_scalar_or_pointer(params.beta)
                rD += beta * tRS_rC.load().to(tRS_rD.element_type)
        tRS_rD.store(rD)
        if const_expr(tDrRowVec is not None):
            for i in cutlass.range(cute.size(tDrRowVec), unroll_full=True):
                tRS_rD[i] += tDrRowVec[i]
        if const_expr(tDrColVec is not None):
            for i in cutlass.range(cute.size(tDrColVec), unroll_full=True):
                tRS_rD[i] += tDrColVec[i]
        return None

    def epi_setup_postact(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Returns None — default epilogue has no postact output."""
        return None

    @cute.jit
    def epi_convert_postact(
        self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
    ):
        """Convert postact from acc_dtype to output dtype. Override for custom postprocessing."""
        return tRS_rPostAct


class GemmDefaultSm90(GemmDefaultEpiMixin, GemmSm90):
    pass


class GemmDefaultSm100(GemmDefaultEpiMixin, GemmSm100):
    pass
