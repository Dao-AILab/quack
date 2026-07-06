import cutlass

from quack.epi_ops import ColVecReduce, EpiSmemBytes, RowVecLoad, TileLoad, TileStore
from quack.testing.trace import run_traced

# smem_bytes needs a live MLIR context for cute layout algebra. run_traced,
# not `with ir.Context()`: raw contexts corrupt the process (see
# quack.testing.trace).


class _ArgTensor:
    element_type = cutlass.Float32


def test_colvec_reduce_smem_bytes_default_to_direct_write_path():
    op = ColVecReduce("mColVecReduce")

    def check():
        assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == EpiSmemBytes()

    run_traced(check)


def test_colvec_reduce_smem_bytes_use_atom_layout_n_warp_staging():
    op = ColVecReduce("mColVecReduce")

    # op.smem_bytes is only called for active (non-None) args by the framework;
    # ComposableEpiMixin.epi_smem_bytes filters from `args` before calling.
    def check():
        assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32), (4, 2, 1)) == EpiSmemBytes(
            unstaged=64 * 1 * 4
        )

    run_traced(check)


def test_vecload_smem_bytes_are_unstaged():
    op = RowVecLoad("mRowVecBroadcast")

    def check():
        assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == EpiSmemBytes(
            unstaged=128 * 4
        )

    run_traced(check)


def test_tile_store_smem_bytes_are_d_staged():
    op = TileStore("mAuxOut")

    def check():
        assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == EpiSmemBytes(
            d_stage=64 * 32 * 4
        )

    run_traced(check)


def test_tile_load_smem_accounting_is_c_stage():
    op = TileLoad("mTile")

    def check():
        assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == EpiSmemBytes(
            c_stage=64 * 32 * 4
        )

    run_traced(check)
