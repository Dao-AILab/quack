import torch
from quack.dist_utils import make_barrier_flags
from quack.epi_reduce import EpiReduceArguments


def make_epi_reduce_args(mD_mc, mD_peers, m, n, l, tile_M, tile_N, cluster_M, num_ranks):
    """Allocate the epi_reduce_mode semaphores and assemble EpiReduceArguments.

    Call once at setup (make_barrier_flags is a collective symmetric alloc) and
    reuse across launches: flags are monotonic — never reset — and
    consumer_counters hold each consumer tile's epoch base, so never re-zero one
    without the other. Sizing is per (shape, tile config); a mismatched launch is
    rejected by validate_epi_reduce_args, whose cta_m derivation mirrors this one.

    Returns the torch-tensor flavor the TVM-FFI surfaces consume (EpiMod.gemm /
    quack.gemm.gemm); direct cute.compile callers build the cute flavor themselves.
    """
    assert m % num_ranks == 0, "epi_reduce_mode slab math needs M % num_ranks == 0"
    use_2cta = cluster_M % 2 == 0 and tile_M in (128, 256)
    cta_m = tile_M // (2 if use_2cta else 1)
    n_tiles = (n + tile_N - 1) // tile_N
    num_tiles = ((m + cta_m - 1) // cta_m) * n_tiles * l
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tile_flags, tile_flags_mc, _, _ = make_barrier_flags(num_tiles)
    sync_barrier, sync_barrier_mc, _, _ = make_barrier_flags(num_sms)
    slab_tiles_m = (m // num_ranks + cta_m - 1) // cta_m
    counters = torch.zeros(slab_tiles_m * n_tiles * l, dtype=torch.int32, device="cuda")
    return EpiReduceArguments(
        mD_mc=mD_mc,
        mD_peers=tuple(mD_peers),
        tile_flags=tile_flags,
        tile_flags_mc=tile_flags_mc,
        sync_barrier=sync_barrier,
        sync_barrier_mc=sync_barrier_mc,
        consumer_counters=counters,
    )
