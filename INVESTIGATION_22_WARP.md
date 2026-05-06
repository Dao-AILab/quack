# Investigation: SM100 gated `(2, 2)` epilogue warp-shape bug

## Setup

Branch: `explore-22-warp` (forked from `fix-gated-dgated` HEAD).

The `_valid_2cta_m` overrides on `GemmGatedMixin` and `GemmDGatedMixin`
have been **reverted** so the bug fires. Plus `print()` instrumentation in
`quack/gemm_base.py`, `quack/epi_ops.py`, and `quack/gemm_act.py`.

Repro: `instr_run.py`. Run with fresh `QUACK_CACHE_DIR` and
`QUACK_CACHE_ENABLED=0` to force re-compile each run.

## Trigger

`tile_m=128, cluster_m=2, is_dynamic_persistent=True, use_tma_gather=True`
on the gated forward path. With 2-CTA, `cta_tile_m=64`, which forces
`compute_epilogue_tile_shape` to a `(2, 2)` M-warps × N-warps layout. The
non-gated D path with the same warp shape works correctly — the bug is
specific to the gated half-N postact aux-out chain.

## Localization (the smoking gun)

`tiled_copy_aux_out_r2s` is built via:

    cute.make_tiled_copy_S(aux_atom, tiled_copy_r2s)

`make_tiled_copy_S` keeps the source-side threading from `tiled_copy_r2s`
(D's r2s copy) and only swaps the per-atom store op. The Tiler MN is
inherited verbatim — full-N D dimensions, NOT half-N aux dimensions.

Side-by-side for `tile_m=128, cm=2, swiglu fp16`, `cta_tile_shape=(64,256)`:

| object                                | shape / layout                                                          |
|---------------------------------------|-------------------------------------------------------------------------|
| D's r2s `tiled_copy_r2s` Tiler MN     | `((2,32):(32,1), (2,32):(32,1))` = 64M × 64N                            |
| Aux's r2s `tiled_copy_aux_out_r2s`    | `((2,32):(32,1), (2,32):(32,1))` = 64M × 64N (**same as D**)            |
| Aux's r2s TV layout                   | `((32,2,2),(1,32)):((2,1,64),(0,128))` -- 32 values per thread          |
| Aux's smem `sAuxOut.layout`           | `((8,8),(16,2),(1,2)):((16,128),(1,1024),(0,2048))` = 64M × 32N         |
| D's smem `sD.layout`                  | `((8,8),(32,2),(1,2)):((32,256),(1,2048),(0,4096))` = 64M × 64N         |

**Mismatch:** the aux r2s copy has a 64×64 tiler producing 32 values per
thread × 128 threads = 4096 elements, but aux smem per stage holds only
64×32 = 2048 elements. Each aux smem position is written by **two
threads** -- warp 0's threads and warp 1's threads collide on the same
smem range. Whichever thread arrives last "wins"; warp 1's data is lost.

The TMA descriptor for aux *is* correct (it scatters smem regions to gmem
at warp-stride 64). It's just that smem holds duplicated data when the TMA
reads it -- both the (smem) "warp 0 region" and the (smem) "warp 1 region"
hold warp 0's values after the r2s race. TMA then dutifully writes warp 0's
values to gmem `[0..15]` and warp 0's values again to gmem `[64..79]`,
producing the observed:

    postact[0,  0..15] = warp 0's values   (correct)
    postact[0, 64..79] = warp 0's values   (DUPLICATE -- should be warp 1)

## Why the (4, 1) warp shape works

For `cluster_m=1` (= `(4, 1)` warp shape), `epi_tile_n` is just `int 32`
(no Layout). After `_gated_epi_tile_fn` halves to `int 16`, aux smem is
flat with only 1 N-warp. The Tiler MN match between D and aux remains
"D's full-N tile = aux's full-N tile" because there's no warp-N split in
either; the per-thread value count of 16 lands cleanly in aux smem with no
collision.

Per-thread `tRS_rD.layout` is `((1,32),1,1):((0,1),0,0)` for **both** warp
shapes. The bug is purely in the destination-side (smem) partitioning of
`tiled_copy_aux_out_r2s`, not in registers or in `act_fn` indexing.

## Why D's full-N path is unaffected

D's smem layout has 64 N elements (twice aux's), with warp 1 at smem
stride 2048. D's r2s tiler `((2,32),(2,32))` produces 32 values per
thread × 128 threads = 4096 elements -- matches D smem per stage exactly.
No collision.

## Fix direction

The aux r2s tiled copy must be re-tiled to match aux's tile dimensions
(half N) before being used to partition `sAuxOut`. Two plausible builders:

1. Build from scratch via `make_tiled_copy_D(aux_atom, sAuxOut.layout)` so
   the destination shape comes from aux smem rather than D's r2s.
2. Re-tile `tiled_copy_r2s` to halve its N extent before passing through
   `make_tiled_copy_S`.

Either approach requires careful handling of the per-thread register slice
(`tRS_rAuxOut` has 16 fp32 elements per thread, derived via
`recast_layout(2, 1, tRS_rD.layout)`). The atom returned by
`sm100_utils.get_smem_store_op(aux_layout, aux_dtype, acc_dtype, tiled_copy_t2r)`
is selected based on `tiled_copy_t2r` (D's full-N pattern) -- it likely
needs to be rebuilt from a t2r-equivalent for aux's half-N slice as well.

This is real cuTeDSL design work. The current `_valid_2cta_m` override on
`GemmGatedMixin` / `GemmDGatedMixin` is the practical workaround; this
investigation explains exactly why the override is needed and what would
need to change to remove it.

## Reproduction commands

```bash
git checkout explore-22-warp
CACHE=$(mktemp -d /tmp/quack_explore_XXXX)
CUDA_VISIBLE_DEVICES=0 QUACK_CACHE_DIR=$CACHE QUACK_CACHE_ENABLED=0 \
    python instr_run.py
# CLUSTER_M=1 to compare against the working (4, 1) warp shape.
```

The instrumentation prints D-path and aux-path layouts side by side; the
mismatch in Tiler MN vs sAuxOut shape is the smoking gun.
