import torch
from quack.dist_utils import make_barrier_flags, make_symmetric_tensor
from quack.epi_reduce import EpiReduceArguments, epi_reduce_workspace_shape


def make_epi_reduce_args(
    mode, d_dtype, m, n, l, tile_M, tile_N, cluster_M, num_ranks, ws_dtype=None
):
    """Allocate the epi_reduce_mode buffers and return (D, EpiReduceArguments).

    D is the caller-order output the launch consumes: reduce_scatter — a plain
    local (l, m/world, n) tensor; all_reduce — this rank's view of a full
    symmetric (l, m, n) tensor (the commit broadcasts through its mc view, kept
    as mD_mc). Partials live in the padded symmetric workspace, never in D;
    ws_dtype overrides their dtype (e.g. float32 for exact partials at 2x the
    bytes), reduce_scatter only.

    Call once at setup (symmetric allocs are collective) and reuse across
    launches: flags are monotonic — never reset — and consumer_counters hold
    each consumer tile's epoch base, so never re-zero one without the other.
    Sizing is per (shape, tile config); a mismatched launch is rejected by
    validate_epi_reduce_args, whose cta_m derivation mirrors this one.

    Returns the torch-tensor flavor the TVM-FFI surfaces consume (EpiMod.gemm /
    quack.gemm.gemm); direct cute.compile callers build the cute flavor themselves.
    """
    assert mode in ("reduce_scatter", "all_reduce"), f"unknown epi_reduce_mode {mode}"
    assert m % num_ranks == 0, "epi_reduce_mode slab math needs M % num_ranks == 0"
    ws_dtype = d_dtype if ws_dtype is None else ws_dtype
    assert mode == "reduce_scatter" or ws_dtype == d_dtype, (
        "all_reduce requires workspace dtype == D dtype"
    )
    use_2cta = cluster_M % 2 == 0 and tile_M in (128, 256)
    cta_m = tile_M // (2 if use_2cta else 1)
    m_pad, n_pad = epi_reduce_workspace_shape(m, n, cta_m, tile_N)
    workspace, workspace_mc, _ = make_symmetric_tensor((l, m_pad, n_pad), ws_dtype, (1, 2, 0))
    mD_mc = None
    if mode == "all_reduce":
        d_knl, mD_mc, _ = make_symmetric_tensor((l, m, n), d_dtype, (1, 2, 0))
        d = d_knl.permute(2, 0, 1)  # caller-order (l, m, n) view
    else:
        d = torch.empty(l, m // num_ranks, n, dtype=d_dtype, device="cuda")
    n_tiles = (n + tile_N - 1) // tile_N
    num_tiles = ((m + cta_m - 1) // cta_m) * n_tiles * l
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tile_flags, tile_flags_mc, _, _ = make_barrier_flags(num_tiles)
    sync_barrier, sync_barrier_mc, _, _ = make_barrier_flags(num_sms)
    slab_tiles_m = (m // num_ranks + cta_m - 1) // cta_m
    counters = torch.zeros(slab_tiles_m * n_tiles * l, dtype=torch.int32, device="cuda")
    return d, EpiReduceArguments(
        mD_mc=mD_mc,
        workspace=workspace,
        workspace_mc=workspace_mc,
        tile_flags=tile_flags,
        tile_flags_mc=tile_flags_mc,
        sync_barrier=sync_barrier,
        sync_barrier_mc=sync_barrier_mc,
        consumer_counters=counters,
    )
