# Copyright (c) 2026, Tri Dao.

import pytest

from quack.rmsnorm_flydsl_config import RmsNormFwdConfig, rows_per_block


@pytest.mark.parametrize(
    "n,dtype_width,expected",
    [
        (8, 16, (8, 1, 1, 1, 256, False, False)),
        (192, 16, (8, 32, 1, 24, 8, True, False)),
        (760, 16, (8, 64, 2, 95, 1, True, False)),
        (4096, 16, (8, 128, 4, 512, 1, False, False)),
        (262144, 32, (4, 1024, 64, 65536, 1, False, True)),
    ],
)
def test_forward_config_geometry(n, dtype_width, expected):
    config = RmsNormFwdConfig.for_forward(n, dtype_width)
    actual = (
        config.vecsize,
        config.num_threads,
        config.num_tiles,
        config.num_vecs,
        rows_per_block(config),
        config.needs_predicate,
        config.reload_from_gmem,
    )
    assert actual == expected


def test_forward_config_rejects_unsupported_width():
    with pytest.raises(ValueError, match="unsupported element width"):
        RmsNormFwdConfig.for_forward(256, 8)
