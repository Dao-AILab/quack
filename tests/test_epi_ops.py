import cutlass

from quack.epi_ops import ColVecReduce


class _ArgTensor:
    element_type = cutlass.Float32


def test_colvec_reduce_smem_bytes_default_to_direct_write_path():
    op = ColVecReduce("mColVecReduce")

    assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == 0


def test_colvec_reduce_smem_bytes_use_configured_n_warp_staging():
    op = ColVecReduce("mColVecReduce", max_warps_in_n=2)

    assert op.smem_bytes(_ArgTensor(), (64, 128, 64), (64, 32)) == 64 * 2 * 4
    assert op.smem_bytes(None, (64, 128, 64), (64, 32)) == 0
