# Investigation: SM100 gated `(2, 2)` epilogue warp-shape bug — RESOLVED

## TL;DR

**Bug fixed at the source level.** The `_valid_2cta_m` overrides on
`GemmGatedMixin` and `GemmDGatedMixin` (commit `b10ffed`) are no longer
needed; this branch removes them and replaces the workaround with a real
fix in `quack/gemm_act.py` — a 1-method override on `GemmGatedMixin` that
constructs the aux-out r2s tiled copy with explicit thread + value layouts
so the tiler MN matches aux smem.

## Verification

With `b10ffed`'s overrides reverted on this branch (so the bug *would* fire
without the new fix):

| test                                                  | result                                                 |
|-------------------------------------------------------|--------------------------------------------------------|
| `solo_ab_min.py`                                      | phase A rel=0.0000 PASS, phase B rel=0.0000 PASS       |
| `instr_run.py` original buggy shape (M=32768, E=8)    | preact rel=0, postact rel=8.21e-4 PASS                 |
| 6 (M, H, I, E) × 2 cluster_m configs                  | All 12 PASS, identical errors between cm=1 and cm=2    |
| `test_untuned_buggy_tiles.py --shapes small`          | 208/208 PASS, 0 timeouts (4 shapes × 52 forced configs)|
| `sweep_gated_dgated.py` (216 autotuned shape grid)    | 216/216 PASS                                           |

Phase A's monkey-patch in `solo_ab_min.py` is now a no-op (the override
method doesn't exist on the class); even so, with 2-CTA forced on, output
is correct.

## Root cause (recap)

The gated postact tile has **half** the N elements of D's tile (via
`_gated_epi_tile_fn`'s `recast_layout(2, 1, ...)`). The original
construction at `gemm_act.py:104`:

    cute.make_tiled_copy_S(aux_atom, tiled_copy_r2s)

inherits **D's full-N tiler MN** (e.g. 64×64) and applies it to aux smem
which is half-N (e.g. 64×32). Per epi-iter, 128 threads × 32 vals/thread =
4096 elements get emitted into a 2048-element smem region — a 2× overlap.

For the (4, 1) epi-warp shape this is harmless: the over-emission has
stride 0 in the smem layout's phantom N-warp dim (since there's only 1
N-warp), so it's a no-op self-overwrite. For the (2, 2) shape, the smem
N-warp dim has stride 1024 — the over-emitted elements land at warp 1's
smem region, clobbering warp 1's data with a duplicate of warp 0's. TMA
then dutifully scatters the duplicated smem to two distinct gmem
positions, producing the observed corruption pattern at gmem[0..15] ==
gmem[64..79].

The non-gated D path is unaffected because aux smem and D's smem have
the same dimensions there (no half-N recast).

The dgated bwd path is unaffected because `GemmDGatedMixin._epi_ops` uses
`TileStore("mAuxOut")` with no `epi_tile_fn` (no half-N recast). The
preventive override that `b10ffed` added on `GemmDGatedMixin` was
empirically unneeded; the sweeps with that override removed all pass.

## The fix

`quack/gemm_act.py` adds an override on `GemmGatedMixin` only:

```python
def epi_make_aux_out_tiled_copy_r2s(self, params, tiled_copy_r2s, tiled_copy_t2r):
    if self.arch != 100:
        return super().epi_make_aux_out_tiled_copy_r2s(
            params, tiled_copy_r2s, tiled_copy_t2r
        )
    copy_atom_aux_out_r2s = self.epi_make_aux_out_copy_atom_r2s(params, tiled_copy_t2r)
    cta_tile_aux_m, _ = self.cta_tile_shape_aux_out_mn
    _, num_n_warps, _ = self.epi_smem_warp_shape_mnk()
    epi_tile_aux_n = cute.size(params.epi_tile_mAuxOut[1])
    vals_per_thread_n = epi_tile_aux_n // num_n_warps
    thr_layout = cute.make_layout(
        (cta_tile_aux_m, num_n_warps), stride=(1, cta_tile_aux_m)
    )
    val_layout = cute.make_layout((1, vals_per_thread_n))
    return cute.make_tiled_copy_tv(copy_atom_aux_out_r2s, thr_layout, val_layout)
```

Threading is `(cta_tile_aux_m, num_n_warps)` with stride `(1, cta_tile_aux_m)`
— 128 threads laid out as 1 thread per (M-row, N-warp) cell. Each thread
holds `vals_per_thread_n = size(epi_tile_aux_n) / num_n_warps` values
along N. Total = 128 × `vals_per_thread_n` = aux smem per stage exactly,
no overlap. SM90/SM120 fall back to the original construction (the
Layout-typed `epi_tile_n` is SM100-specific via `compute_epilogue_tile_shape`).

## What was removed in this branch

- `GemmGatedMixin._valid_2cta_m -> (256,)` (workaround, no longer needed).
- `GemmDGatedMixin._valid_2cta_m -> (256,)` (preventive workaround,
  empirically unneeded — dgated has no half-N recast and no (2, 2) bug).
- `GemmSm100._valid_2cta_m()` method indirection (introduced by `b10ffed`
  to support the workaround).

## Reproduction

```bash
git checkout explore-22-warp
CACHE=$(mktemp -d /tmp/quack_explore_XXXX)
CUDA_VISIBLE_DEVICES=0 QUACK_CACHE_DIR=$CACHE QUACK_CACHE_ENABLED=0 \
    python instr_run.py
# CLUSTER_M=2 (default) reproduces the previously-buggy cocktail; both
# phase A and phase B of solo_ab_min.py now PASS with rel=0.
```
