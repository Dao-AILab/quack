import torch
from quack.dist_utils import make_barrier_flags
from quack.epi_reduce import EpiReduceArguments


def make_epi_reduce_args(mD_mc, mD_peers, m, n, l, cta_m, cta_n, num_ranks):
    """Allocate the epi_reduce_mode semaphores and assemble EpiReduceArguments.

    Call once at setup (make_barrier_flags is a collective symmetric alloc) and
    reuse across launches: flags are monotonic — never reset — and
    consumer_counters hold each consumer tile's epoch base, so never re-zero one
    without the other.

    Returns the torch-tensor flavor the TVM-FFI surfaces consume (EpiMod.gemm /
    quack.gemm.gemm); direct cute.compile callers build the cute flavor themselves.
    """
    assert m % num_ranks == 0, "epi_reduce_mode slab math needs M % num_ranks == 0"
    n_tiles = (n + cta_n - 1) // cta_n
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
