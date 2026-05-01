import cutlass
from cutlass._mlir import ir

from quack.epi_ops import ColVecReduce, EpiSmemBytes, RowVecLoad, TileLoad, TileStore


class _ArgTensor:
    element_type = cutlass.Float32


def test_colvec_reduce_smem_bytes_default_to_direct_write_path():
    op = ColVecReduce("mColVecReduce")

    with ir.Context():
        assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == EpiSmemBytes()


def test_colvec_reduce_smem_bytes_use_atom_layout_n_warp_staging():
    op = ColVecReduce("mColVecReduce")

    with ir.Context():
        assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32), (4, 2, 1)) == EpiSmemBytes(
            unstaged=64 * 1 * 4
        )
        assert op.smem_bytes(None, (64, 128, 64), (64, 32), (4, 2, 1)) == EpiSmemBytes()


def test_vecload_smem_bytes_are_unstaged():
    op = RowVecLoad("mRowVecBroadcast")

    with ir.Context():
        assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == EpiSmemBytes(
            unstaged=128 * 4
        )


def test_tile_store_smem_bytes_are_d_staged():
    op = TileStore("mAuxOut")

    with ir.Context():
        assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == EpiSmemBytes(
            d_stage=64 * 32 * 4
        )


def test_tile_load_smem_accounting_is_c_stage():
    op = TileLoad("mTile")

    with ir.Context():
        assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == EpiSmemBytes(
            c_stage=64 * 32 * 4
        )
        assert op.smem_bytes(None, (64, 128, 64), (64, 32)) == EpiSmemBytes()
