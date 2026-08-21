import torch
from quack.dist_utils import make_symm_mem_flags, make_symm_mem_tensor
from quack.epi_reduce import EpiReduceArguments, epi_reduce_workspace_shape


def make_epi_reduce_args(
    mode,
    d_dtype,
    m,
    n,
    l,
    tile_M,
    tile_N,
    cluster_M,
    num_ranks,
    ws_dtype=None,
    d_major="n",
):
    """Allocate the epi_reduce_mode buffers and return (D, EpiReduceArguments).

    D is the caller-order output the launch consumes: reduce_scatter — a plain
    local (l, m/world, n) tensor, or (l, m, n/world) under slab_dim="n" (the
    swap_ab relabeling: the caller's sharded dim is kernel-N); all_reduce — this
    rank's view of a full symmetric (l, m, n) tensor (the commit broadcasts
    through its mc view, kept as mD_mc). Partials live in the padded symmetric
    workspace, never in D; ws_dtype overrides their dtype (e.g. float32 for
    exact partials at 2x the bytes), reduce_scatter only.

    Call once at setup (symmetric allocs are collective) and reuse across
    launches: flags self-reset in-kernel, so the one-time zero-fill here is the
    only initialization. Sizing is per (shape, tile config); a mismatched launch is rejected by
    validate_epi_reduce_args, whose cta_m derivation mirrors this one.

    Returns the torch-tensor flavor the TVM-FFI surfaces consume (EpiMod.gemm /
    quack.gemm.gemm); direct cute.compile callers build the cute flavor themselves.
    """
    assert mode in ("reduce_scatter", "all_reduce"), f"unknown epi_reduce_mode {mode}"
    assert d_major in ("m", "n"), f"unknown d_major {d_major}"
    # The slab axis is D's strided axis, so each rank's slice of D is contiguous.
    slab_dim = "m" if d_major == "n" else "n"
    slab_len = m if slab_dim == "m" else n
    assert slab_len % num_ranks == 0, (
        f"epi_reduce_mode slab math needs {slab_dim} % num_ranks == 0"
    )
    ws_dtype = d_dtype if ws_dtype is None else ws_dtype
    assert mode == "reduce_scatter" or ws_dtype == d_dtype, (
        "all_reduce requires workspace dtype == D dtype"
    )
    use_2cta = cluster_M % 2 == 0 and tile_M in (128, 256)
    cta_m = tile_M // (2 if use_2cta else 1)
    m_pad, n_pad = epi_reduce_workspace_shape(m, n, cta_m, tile_N, slab_dim)
    # Workspace majorness matches D's: the multimem atoms (producer store,
    # reducer ld_reduce, commit st) all run along D's contiguous axis.
    if d_major == "n":
        workspace, workspace_mc = make_symm_mem_tensor((l, m_pad, n_pad), ws_dtype, (1, 2, 0))
    else:
        workspace, workspace_mc = make_symm_mem_tensor((l, n_pad, m_pad), ws_dtype, (2, 1, 0))
    mD_mc = None
    if mode == "all_reduce":
        if d_major == "n":
            d_knl, mD_mc = make_symm_mem_tensor((l, m, n), d_dtype, (1, 2, 0))
            d = d_knl.permute(2, 0, 1)  # caller-order (l, m, n) view
        else:
            d_knl, mD_mc = make_symm_mem_tensor((l, n, m), d_dtype, (2, 1, 0))
            d = d_knl.permute(2, 0, 1)
    elif slab_dim == "m":
        d = torch.empty(l, m // num_ranks, n, dtype=d_dtype, device="cuda")
    elif d_major == "n":
        d = torch.empty(l, m, n // num_ranks, dtype=d_dtype, device="cuda")
    else:
        d = torch.empty(l, n // num_ranks, m, dtype=d_dtype, device="cuda").transpose(-1, -2)
    n_tiles = (n + tile_N - 1) // tile_N
    num_tiles = ((m + cta_m - 1) // cta_m) * n_tiles * l
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    # exit-barrier slots live in the flag tail (hot allocation, one per resident CTA)
    tile_flags, tile_flags_mc = make_symm_mem_flags(num_tiles + num_sms)
    return d, EpiReduceArguments(
        mD_mc=mD_mc,
        workspace=workspace,
        workspace_mc=workspace_mc,
        tile_flags=tile_flags,
        tile_flags_mc=tile_flags_mc,
    )
