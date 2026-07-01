# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""SM120 CTA-local TMA pipeline helpers."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import Boolean, Int32, dsl_user_op, if_generate
from cutlass.cute.typing import Pointer
from cutlass.pipeline import CooperativeGroup, PipelineState
from cutlass.pipeline.sm90 import PipelineTmaAsync


@dataclass(frozen=True)
class PipelineTmaWarpMma(PipelineTmaAsync):
    """SM120 CTA-local TMA pipeline for warp-level MMA consumers."""

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: Optional[Pointer] = None,
        cta_layout_vmnk=None,
        tidx: Optional[Int32] = None,
        mcast_mode_mn: Tuple[int, int] = (1, 1),
        defer_sync: bool = False,
    ) -> "PipelineTmaWarpMma":
        base = PipelineTmaAsync.create(
            num_stages=num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            barrier_storage=barrier_storage,
            cta_layout_vmnk=cta_layout_vmnk,
            tidx=tidx,
            mcast_mode_mn=mcast_mode_mn,
            defer_sync=defer_sync,
        )
        return PipelineTmaWarpMma(
            base.sync_object_full,
            base.sync_object_empty,
            base.num_stages,
            base.producer_mask,
            base.consumer_mask,
            base.is_signalling_thread,
        )

    @dsl_user_op
    def producer_acquire_already_elected(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Acquire a TMA load stage from inside an existing ``elect_one`` block."""
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        cute.arch.mbarrier_arrive_and_expect_tx(
            self.producer_get_barrier(state, loc=loc, ip=ip),
            self.sync_object_full.tx_count,
            loc=loc,
            ip=ip,
        )


__all__ = [
    "PipelineTmaWarpMma",
]
