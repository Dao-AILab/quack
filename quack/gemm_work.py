# Copyright (c) 2025, Tri Dao.

from typing import NamedTuple, Optional

import cutlass.cute as cute
from cutlass import Int32, Boolean

from quack.cute_dsl_utils import mlir_namedtuple


@mlir_namedtuple
class WorkDesc(NamedTuple):
    tile_coord_mnkl: cute.Coord
    problem_idx: Int32
    k_tile_begin: Int32 = Int32(0)
    k_tile_count: Optional[Int32] = None
    split_k_idx: Int32 = Int32(0)
    split_k_parts: Int32 = Int32(1)
    is_final_split: Boolean = Boolean(True)
    is_valid_tile: Boolean = Boolean(False)

    @property
    def tile_idx(self):
        return self.tile_coord_mnkl

    @property
    def batch_idx(self):
        return self.tile_coord_mnkl[3]


def make_work_desc(tile_coord_mnkl: cute.Coord, is_valid_tile: Boolean) -> WorkDesc:
    return WorkDesc(
        tile_coord_mnkl=tile_coord_mnkl,
        problem_idx=tile_coord_mnkl[3],
        is_valid_tile=is_valid_tile,
    )
