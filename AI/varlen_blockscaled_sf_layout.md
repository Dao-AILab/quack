# Variable-length blockscaled SF layout (dQaccum-style padding)

How the quack SM100 blockscaled GEMM stores scale factors (SFA / SFB) when the
varying dimension (M for `varlen_m`, K for `varlen_k`) has per-expert lengths
that are **not necessarily multiples of 128**. Without padding, SF tiles
(which cover 128 source rows/cols) would straddle expert boundaries; with
padding, each expert's SF region starts on a 128-aligned tile boundary and the
kernel reads scales from a single unified buffer via a per-batch offset.

This mirrors flash-attention's `dQaccum` layout (`flash_attn/cute/interface.py`
and `seqlen_info.py`'s `offset_padded`) but simplified.

Scope: MXFP8 only. MXFP4 / NVFP4 require K-major operands and aren't wired up
for varlen.

## Notation

- `L` — number of experts / batches.
- `m_b` — expert `b`'s length along the varying dim (M or K).
- `c_b = cu_seqlens[b] = Σ_{j<b} m_j`. Exclusive prefix sum; `c_0 = 0`,
  `c_L = total_m` (or `total_k`).
- `rm_b = ⌈m_b / 128⌉` — tiles expert `b`'s SF occupies along the varying dim.
  (128 = `sf_vec_size * 4` for MXFP8; one 512-byte SF tile covers 128 source
  rows × 4 scale cols.)

## Format

SF is a single unified buffer whose varying dim is padded so every expert
starts at a tile boundary. Following flash-attention's `offset_padded` trick,
the allocation is sized for `off[L]` tiles — i.e. the "hypothetical next
batch's" start position — using the same formula the kernel uses to index
into it.

### Per-expert tile offset

```
off[b] = c_b // 128 + b          # in tile units
```

Equivalently `(c_b + b·128) // 128 * 128 / 128`, matching
`SeqlenInfo.offset_padded / 128`, but without the `// * *` back-and-forth.

Expert `b` occupies tiles `[off[b], off[b] + rm_b)`.

### Allocation size (tile units)

```
total_padded_rm = ⌈total_m / 128⌉ + (L − 1)
```

Proven sufficient (see Proof 2 below — "tighter alternative"). This form is
tight (zero waste) when `total_m` is an exact multiple of 128 and matches
`total_m//128 + L` otherwise. The byte size is `total_padded_rm × rk × 512`
bytes.

### Torch storage shape

```
SFA: (1, total_padded_rm, rk_const, 512)
SFB: (1, rn,              total_padded_rk, 512)   # for varlen_k
```

Where `rk_const` / `rn` are the non-varying tile counts. The 512-byte inner
dim is the hardware-fixed swizzled tile (see `varlen_sf_layout_proof.md`
companion / `BlockScaledBasicChunk`).

## Correctness — two proofs

### Proof 1: no overlap between consecutive experts

Need `off[b] + rm_b ≤ off[b+1]`. Expanding with `c_b = 128q + r`
(`0 ≤ r < 128`), `m_b = 128p + s` (`0 ≤ s < 128`):

```
(q + b) + ⌈m_b/128⌉  ≤  ((c_b + m_b) // 128) + (b+1)
⇔  ⌈m_b/128⌉  ≤  (r + m_b) // 128 + 1
```

| case | `s` | `r + s` | `⌈m_b/128⌉` | `(r+m_b)//128 + 1` | verdict |
|---|---|---|---|---|---|
| A1 | `= 0` | — (`r==0`) | `p`   | `p + 1`   | slack 1 |
| A2 | `= 0` | — (`r>0`)  | `p`   | `p + 1`   | slack 1 |
| B1 | `> 0` | `< 128`    | `p+1` | `p + 1`   | **tight** |
| B2 | `> 0` | `≥ 128`    | `p+1` | `p + 2`   | slack 1 |

In every case LHS ≤ RHS. **No overlap. ∎**

### Proof 2: allocation is sufficient (`⌈total_m/128⌉ + (L−1)`)

Need `off[L−1] + rm_{L−1} ≤ ⌈total_m/128⌉ + (L−1)`. Cancel `L−1` and let
`c = c_{L−1} = 128q + r`, `m = m_{L−1} = 128p + s`, `total_m = c + m`; need
`q + ⌈m/128⌉ ≤ ⌈total_m/128⌉`:

| case | `s` | `r + s` | `⌈m/128⌉` | `⌈total_m/128⌉` | verdict |
|---|---|---|---|---|---|
| A1 | `= 0` | `r==0` | `p`   | `q+p`   | tight |
| A2 | `= 0` | `r>0`  | `p`   | `q+p+1` | slack 1 |
| B1 | `> 0` | `< 128`| `p+1` | `q+p+1` | tight |
| B2 | `> 0` | `==128`| `p+1` | `q+p+1` | tight |
| B3 | `> 0` | `>128` | `p+1` | `q+p+2` | slack 1 |

In every case LHS ≤ RHS, so the allocation is sufficient. ∎

### Simpler alternative (not used)

`total_m // 128 + L` (which equals `off[L]`, the hypothetical next-batch
start) is also a valid upper bound — by Proof 1 applied iteratively,
`off[L−1] + rm_{L−1} ≤ off[L]`. It is `== ⌈total_m/128⌉ + (L−1)` when
`total_m % 128 > 0` and **1 tile larger** when `total_m` is a multiple of
128. We prefer the tighter form because LLM workloads frequently have
128-aligned `total_m` (prefill, batched training).

## Kernel indexing

After `tile_atom_to_shape_SF_strided` builds the SF layout, the outer tile
dim (`rm` for `varlen_m`, `rk` for `varlen_k`) is exposed as the second
element of a compound mode `((32, 4), rm_or_rk)`. We offset just that outer
element via `cute.domain_offset` with a compound coord:

```python
# varlen_m (M padded)
offset_tile = cu_seqlens_m[batch_idx] // 128 + batch_idx
mSFA_batch = cute.domain_offset(((0, offset_tile), None), mSFA_mkl)

# varlen_k (K padded)
offset_tile = cu_seqlens_k[batch_idx] // 128 + batch_idx
mSFA_batch = cute.domain_offset((None, (0, offset_tile)), mSFA_mkl)
mSFB_batch = cute.domain_offset((None, (0, offset_tile)), mSFB_nkl)
```

No `* 128` anywhere on the hot path — tile alignment is syntactic, and the
compiler sees the outer rm/rk stride (`s_rm` / `s_rk` in bytes) applied to
a tile-unit offset integer.

## Implementation pointers

- `quack/varlen_utils.py`
  - `VarlenManager.offset_batch_SFA` — padded M or K offset via compound coord.
  - `VarlenManager.offset_batch_SFB` — padded K offset for `varlen_k`.
- `quack/gemm_sm100.py` — layout setup distinguishes `varlen_m` (pad M from
  `mSFA.shape[1] * 128`) vs `varlen_k` (pad K from `mSFA.shape[2] * 128`).
- `quack/blockscaled_gemm_utils.py`
  - `create_blockscaled_varlen_m_operands(seqlens_m=...)`
  - `create_blockscaled_varlen_k_operands(seqlens_k=...)`
  - Both fill a source-unit torch buffer at offset
    `(c_b // 128 + b) * 128`, then pass through
    `pack_scale_2d_to_blocked_contig` to the `(1, rmn, rk, 512)` layout.
- `quack/layout_utils.py`
  - `tile_atom_to_shape_SF_strided(shape, sf_vec_size, sf_strides)` — builds
    the CuTe layout using mSFA's own strides and shape, so the padded total
    (not the unpadded `mA.shape`) drives the outer rm/rk count.

## Tests

`tests/test_gemm_blockscaled.py`:
- `test_blockscaled_mxfp8_varlen_m_nonaligned` — 4 seqlen patterns × 2 B-majors = 8 cases.
  Patterns include `[128, 128, 128]`, `[100, 200, 150]`, `[30, 300, 64, 200]`,
  `[1, 128, 127, 129]`.
- `test_blockscaled_mxfp8_varlen_k` — 4 patterns including non-128-aligned
  `[96, 160, 128]` and `[32, 256, 64, 128]`.

All per-expert reference checks use `a_ref_list[i] @ b_ref_list[i].T` (or
equivalent cat along the non-varying dim) against the kernel's single-pass
output, verifying each expert's region is correctly populated without
overlap or underflow.
