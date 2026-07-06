# Variable-length blockscaled SF layout (tile-aligned per-batch padding)

How the quack SM100 blockscaled GEMM stores scale factors (SFA / SFB) when the
varying dimension (M for `varlen_m`, K for `varlen_k`) has per-expert lengths
that are **not necessarily multiples of 128**. Without padding, SF tiles
(which cover 128 source rows/cols) would straddle expert boundaries; with
padding, each expert's SF region starts on a 128-aligned tile boundary and the
kernel reads scales from a single unified buffer via a per-batch offset.

We call this **tile-aligned per-batch padding**.

Scope: `varlen_m` supports MXFP8 / MXFP4 / NVFP4 — fp4's K-major operand
requirement coincides with `varlen_m`'s (A must be K-major), and per-batch
offsets slice whole M rows, which stay byte-aligned (`k/2` bytes/row) and
TMA-aligned (`k % 32`). `varlen_k` is MXFP8-only: it needs M/N-major operands
(conflicting with fp4's K-major), and per-expert K offsets would hit sub-byte
packing. For `varlen_k`, per-expert `k_i` is arbitrary — not even
`sf_vec_size` (32) alignment is required: a non-aligned `k_i` leaves the
expert's last scale block covering a partial chunk, and the ragged value TMA
zero-fills beyond `cu_seqlens_k[i+1]`, so the tail contributes exactly 0.

**SF pad and ragged K:** K is the *reduction* dim, so the pad inside each
expert's last SF atom column is loaded by the kernel (SF TMA loads whole
512 B atoms; unlike A/B it is not ragged-bounded) and would pair with the
zero-filled value tail — `0xFF` there (e8m0 NaN) would poison the accumulator
via `0 × NaN`. The kernel avoids this by **skipping the MMA instructions for
pad blocks**: for mxfp8, one tcgen05 blockscaled MMA instruction consumes
exactly one SF k-block (`inst_k = 256 bits / 8 = 32 = sf_vec_size`), so on a
ragged batch's last k-tile the mma loop issues only the
`ceil((len_k % 128) / 32)` valid instructions (see `GemmSm100.mma`,
`sf_valid_insts`). The skipped instructions' A/B values are TMA-zero-filled
and contribute nothing, so this changes no numerics — it just never
references the pad scales in tmem. Instruction issue is a leader-only
decision, so 2-CTA MMA (tile_M 256) is covered with no extra
synchronization. Measured cost: ~0 (it strictly *removes* MMA work on ragged
tails). So gmem SF pad may be arbitrary bytes — `torch.empty` buffers are
fine.

Notes: (1) an earlier prototype instead zeroed the pad bytes in smem from
the MMA warp (32-bit RMW per scale word + `fence.proxy.async.shared` before
the UTCCP) — also ~free, but it could not cover 2-CTA, where the UTCCP
sources SF from both CTAs' smem and only the leader waits on the stage;
instruction skipping made it obsolete. (2) Ragged SF TMA is not an
alternative: the ragged-extent trick cuts at whole-atom (128 source-K)
granularity, and the dangerous bytes are *inside* the last atom. (3) The
skip requires `inst_k == sf_vec_size`; mxfp4/nvfp4 (inst_k spans 2/4 SF
blocks) keep the zero-init-pad contract — moot for `varlen_k`, which is
MXFP8-only. (4) `varlen_m` has no such issue at all: its M-pad garbage lands
in output rows the epilogue never stores.

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
starts at a tile boundary. The allocation is sized for `off[L]` tiles — i.e.
the "hypothetical next batch's" start position — using the same formula the
kernel uses to index into it.

### Per-expert tile offset

```
off[b] = c_b // 128 + b          # in tile units
```

Equivalently `((c_b + b·128) // 128 * 128) / 128` — the "round the shifted
prefix down to a tile boundary" form — but without the `// * *` back-and-forth.

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
SFA: (1, total_padded_rm, rk_const, 32, 4, 4)
SFB: (1, rn,              total_padded_rk, 32, 4, 4)   # for varlen_k
```

Where `rk_const` / `rn` are the non-varying tile counts. The inner
`(32, 4, 4)` atom (512 bytes, strides `(16, 4, 1)`) is the hardware-fixed
swizzled tile (see `BlockScaledBasicChunk`).

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

## Minimality — can we pad less?

Three levels, from the current scheme down to the information-theoretic floor.

### 1. Given `off[b] = c_b // 128 + b`, the allocation is exactly tight

`⌈total_m/128⌉ + (L−1)` is achieved, e.g. by `m = [1, 1, …, 1, rest]` (every
boundary is Proof 1's tight case B1). And since `cu_seqlens` lives on device,
host-side allocation/validation can only use `(total_m, L)`; the worst case
over all splits is the best a shape-only bound can do. Accordingly
`validate_blockscaled_sf` requires `SFA.shape[1] ≥ ⌈total_m/128⌉ + (L−1)` —
a per-pattern-exact smaller allocation would be rejected, since verifying it
needs the seqlen values on host (sync).

### 2. Among offsets computable from `(c_b, b)` alone, within ⌈(L−1)/128⌉ tiles of optimal

The optimal such scheme is `off'[b] = (c_b − b) // 128 + b`.

- **Valid** (no overlap): with `c_b − b = 128q + r`, `m_b = 128p + s`, the
  non-overlap condition reduces to
  `q + p + [s>0] ≤ q + p + ⌊(r + s − 1)/128⌋ + 1`, which holds: for `s > 0`,
  `r + s − 1 ≥ 0`; for `s = 0` it holds with equality when `r = 0`
  (`⌊−1/128⌋ = −1`).
- **Optimal** (lower bound): any valid `f` satisfies
  `f(c_{j+1}, j+1) ≥ f(c_j, j) + rm_j` on every pattern, so
  `f(c, b) ≥ Σ_{j<b} rm_j` for every split of `c` into `b` parts. The
  adversarial split uses parts `≡ 1 (mod 128)`, giving
  `f(c, b) ≥ ⌊(c − b)/128⌋ + b` — exactly `off'`.
- **Gap to ours**: `c//128 − ⌊(c−b)/128⌋ ≤ ⌈b/128⌉`, i.e. ≤ 1 tile per 128
  experts (allocation `⌊(total_m − L)/128⌋ + L` saves ≤ 1 tile for `L ≤ 128`,
  and only when `0 < total_m % 128 < L`).
- **Why we don't switch**: besides the negligible saving, `off'` needs
  **floor** division on `c_b − b`, which goes *negative* when leading experts
  are empty (`m_b = 0`, so `c_b < b` — routine in MoE); Int32 `//` in the
  kernel truncates toward zero, rounding the wrong way there. The current
  formula only divides non-negative values. It's also a user-facing storage
  contract — producers of padded SFA would all have to change in lockstep.

### 3. The packed minimum needs an extra kernel input

The true floor is `Σ_b ⌈m_b/128⌉` tiles (each expert needs whole tiles to
itself — regions must be tile-aligned and one 512-B atom can't span experts).
Its offset `Σ_{j<b} ⌈m_j/128⌉` is **not** a function of `(c_b, b)`: two
prefixes with the same `(c, b)` can need different tile counts. Reaching it
requires a second prefix array (`cu_padded_rm`), computed on host (sync) or in
a device prologue. What that buys: ≤ `L−1` tiles = `(L−1) · rk · 512` bytes
≈ `4k(L−1)` bytes (~28 KB/expert at k=7168). Relative worst case is ~2× SF
memory (all `m_b = 128`), but SFA is `1/sf_vec_size` = 1/32 of A's bytes, so
even that is ~3% of the A operand. Not worth the extra input.

All three claims are brute-force-verified over every pattern with `L ≤ 5`,
`m_b ≤ 3·tile` at a scaled tile size (the algebra is tile-size-agnostic).

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
- `quack/blockscaled/utils.py`
  - `create_blockscaled_varlen_m_operands(seqlens_m=...)`
  - `create_blockscaled_varlen_k_operands(seqlens_k=...)`
  - Both fill a source-unit torch buffer at offset
    `(c_b // 128 + b) * 128`, then pass through
    `pack_scale_2d_to_blocked_contig` to the `(1, rmn, rk, 32, 4, 4)` layout.
- Public API (both `varlen_m` and `varlen_k`):
  - `quack/gemm.py::gemm(..., cu_seqlens_m=..., SFA=..., SFB=...)` — SFA is the
    padded buffer viewed as `(1, total_padded_rm, rk, 32, 4, 4)`, SFB per-batch
    `(L, rn, rk, 32, 4, 4)`. In `_compile_gemm` the fake SFA gets its own batch
    sym (its batch dim is 1, not `l`).
  - `quack/gemm.py::gemm(..., cu_seqlens_k=..., SFA=..., SFB=...)` — A is
    `(m, total_k)` m-major, B `(n, total_k)` n-major; both SFA and SFB are
    K-padded `(1, rm/rn, total_padded_rk, 32, 4, 4)` buffers (both fake SF
    tensors get their own batch syms).
  - `quack/gemm_tvm_ffi_utils.py::validate_blockscaled_sf(num_batches=..., varlen_k=...)` —
    checks `SFA.shape[1] >= ceil(total_m/128) + (L-1)` (varlen_m) or
    `SF*.shape[2] >= ceil(total_k/128) + (L-1)` (varlen_k, both buffers).
  - `quack/gemm_interface.py::gemm((A, SFA), (B, SFB), cu_seqlens_m=...)` and
    `gemm((A, SFA), (B, SFB), cu_seqlens_k=...)` (B passed as `(total_k, n)`).
- `quack/layout_utils.py`
  - `tile_atom_to_shape_SF_strided(shape, sf_vec_size, sf_strides)` — builds
    the CuTe layout using mSFA's own strides and shape, so the padded total
    (not the unpadded `mA.shape`) drives the outer rm/rk count.

## Tests

`tests/test_gemm_sm100_blockscaled.py`:
- `test_blockscaled_varlen_m_public_api` — same setup through
  `quack.gemm.gemm` (3 patterns × 2 B-majors × 3 formats); interface-level coverage in
  `tests/test_gemm_blockscaled_interface.py::test_blockscaled_gemm_varlen_m`.
- `test_blockscaled_varlen_m_nonaligned` — 4 seqlen patterns × 2 B-majors × 3 formats.
  Patterns include `[128, 128, 128]`, `[100, 200, 150]`, `[30, 300, 64, 200]`,
  `[1, 128, 127, 129]`.
- `test_blockscaled_mxfp8_varlen_k` — 6 patterns including non-128-aligned
  `[96, 160, 128]`, `[32, 256, 64, 128]` and non-32-aligned `[100, 220, 65]`,
  `[1, 33, 158, 192]` (partial last scale block per expert).
- `test_blockscaled_varlen_k_public_api` — same through `quack.gemm.gemm`
  (3 patterns incl. non-32-aligned); interface-level coverage in
  `tests/test_gemm_blockscaled_interface.py::test_blockscaled_gemm_varlen_k`.

All per-expert reference checks use `a_ref_list[i] @ b_ref_list[i].T` (or
equivalent cat along the non-varying dim) against the kernel's single-pass
output, verifying each expert's region is correctly populated without
overlap or underflow.
