# Copyright (c) 2026, Tri Dao.
"""Tests for the software e2m1 (fp4) stochastic-rounding emulation.

Contract under test (PTX cvt.rs semantics on the e2m1 grid): with D_top the
top 8 discarded mantissa bits of |x| relative to its grid cell's ulp, the
value rounds away from zero iff D_top + r8 >= 256 — bits of x below the top 8
can never bridge the carry (integer-plus-residue argument in quack/rounding.py).
Specials follow satfinite: NaN -> 0x7 (+6.0, sign dropped), |x| > 6 incl Inf
-> sign|0x7, +-0 keeps its sign. The reference model below encodes exactly
this rule; the emulation must match it BITWISE, and — independently of the
model — an exhaustive 256-value rand sweep per grid cell must show round-up
count == D_top on every quad slot (the unbiasedness probe used against the
B300 hardware instruction).

The sw rand mapping is the straight byte order (v_i <- rand[8i+7:8i], the
sm_100a f8x4 convention), NOT sm_103a's permuted/bit-reversed order. The
emulation is explicit cvt.rz/fma/integer PTX, so calling it DIRECTLY compiles
and behaves identically on every target, hw-cvt ones included — those tests
run everywhere. Only the fragment-level test skips on hw-cvt targets: it goes
through convert_f32_frag_sr, which dispatches to the hw instruction there.
"""

import numpy as np
import pytest
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32, Float32
from cutlass.cute.runtime import from_dlpack

from quack.cute_dsl_utils import get_compile_target_capacity
from quack.rounding import (
    PHILOX_KEY_A,
    PHILOX_KEY_B,
    PHILOX_N_ROUNDS_DEFAULT,
    PHILOX_ROUND_A,
    PHILOX_ROUND_B,
    convert_f32_frag_sr,
    cvt_f32x4_e2m1x4_rs_direct,
    cvt_f32x4_e2m1x4_rs_sw,
)

if not torch.cuda.is_available():
    pytest.skip(reason="needs CUDA", allow_module_level=True)

# The sw emulation itself compiles on sm_100a/103a too, but convert_f32_frag_sr
# would pick the hw instruction there (different rand byte mapping), and the
# bitwise reference below intentionally encodes the sw mapping.
IS_HW_CVT_TARGET = get_compile_target_capacity()[0] == 10

E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
E2M1_VALUES = np.concatenate([E2M1_GRID, -E2M1_GRID])  # decode table, code -> value


def ref_e2m1_rs(x: np.ndarray, r8: np.ndarray) -> np.ndarray:
    """Exact reference: nibble = sign | (cell + [D_top + r8 >= 256]), satfinite.

    All arithmetic is exact: |x| and the grid points are exact in float64,
    cell ulps are powers of two, and D_top = floor(frac * 256) needs at most
    32 significand bits.
    """
    assert x.dtype == np.float32
    xb = x.view(np.uint32)
    sign = (xb >> 31).astype(np.int64)
    with np.errstate(invalid="ignore"):  # signaling-NaN cast warning
        a = np.abs(x.astype(np.float64))
    cell = np.clip(np.searchsorted(E2M1_GRID, a, side="right") - 1, 0, 7)
    g = E2M1_GRID[cell]
    ulp = E2M1_GRID[np.minimum(cell + 1, 7)] - g
    interior = cell < 7  # cell 7 (a >= 6) has no upward neighbor: clamp region
    frac = np.where(interior, (a - g) / np.where(interior, ulp, 1.0), 0.0)
    dtop = np.floor(frac * 256.0).astype(np.int64)
    assert (dtop >= 0).all() and (dtop <= 255).all()
    carry = (dtop + r8.astype(np.int64)) >= 256
    code = cell + np.where(interior, carry, 0)
    code = np.where(a > 6.0, 7, code)  # satfinite clamp, incl +-Inf
    nib = (sign << 3) | code
    return np.where(np.isnan(x), 7, nib).astype(np.uint8)


def decode_e2m1(nib: np.ndarray) -> np.ndarray:
    return E2M1_VALUES[nib.astype(np.int64)]


_QUAD_CVTS = {"sw": cvt_f32x4_e2m1x4_rs_sw, "direct": cvt_f32x4_e2m1x4_rs_direct}
# The direct-closer variant needs the sm_100+ e2m1 cvt; bit-identical to "sw"
# by construction, which these tests prove against the same reference model.
QUAD_VARIANTS = ["sw"] + (["direct"] if get_compile_target_capacity()[0] >= 10 else [])


def _make_quad_launch(cvt_fn):
    @cute.kernel
    def _quad_kernel(mX: cute.Tensor, mR: cute.Tensor, mOut: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 256 + tidx
        if i < cute.size(mR.shape):
            packed = cvt_fn(mX[4 * i], mX[4 * i + 1], mX[4 * i + 2], mX[4 * i + 3], Uint32(mR[i]))
            mOut[i] = Uint32(packed).to(Int32)

    @cute.jit
    def _quad_launch(mX: cute.Tensor, mR: cute.Tensor, mOut: cute.Tensor):
        _quad_kernel(mX, mR, mOut).launch(
            grid=(cute.ceil_div(cute.size(mR.shape), 256), 1, 1), block=(256, 1, 1)
        )

    return _quad_launch


_quad_fn_cache = {}


def run_quads(x: torch.Tensor, rand: torch.Tensor, variant: str = "sw") -> np.ndarray:
    """x: (4*n,) float32 cuda; rand: (n,) int32 cuda. Returns (n,) uint16 packed."""
    n = rand.numel()
    assert x.numel() == 4 * n
    out = torch.empty(n, dtype=torch.int32, device="cuda")
    # cute.compile specializes on the tensor shapes; key the cache on them
    if (n, variant) not in _quad_fn_cache:
        _quad_fn_cache[(n, variant)] = cute.compile(
            _make_quad_launch(_QUAD_CVTS[variant]),
            from_dlpack(x),
            from_dlpack(rand),
            from_dlpack(out),
        )
    _quad_fn_cache[(n, variant)](from_dlpack(x), from_dlpack(rand), from_dlpack(out))
    torch.cuda.synchronize()
    return out.cpu().numpy().astype(np.uint32).astype(np.uint16)


def check_bitwise(x: torch.Tensor, rand: torch.Tensor, variant: str = "sw"):
    packed = run_quads(x, rand, variant)
    x_np = x.cpu().numpy()
    r_np = rand.cpu().numpy().view(np.uint32)
    nibs_ref = np.zeros((r_np.size, 4), dtype=np.uint8)
    for j in range(4):
        r8 = ((r_np >> (8 * j)) & 0xFF).astype(np.uint8)
        nibs_ref[:, j] = ref_e2m1_rs(np.ascontiguousarray(x_np.reshape(-1, 4)[:, j]), r8)
    packed_ref = (
        nibs_ref[:, 0]
        | (nibs_ref[:, 1].astype(np.uint16) << 4)
        | (nibs_ref[:, 2].astype(np.uint16) << 8)
        | (nibs_ref[:, 3].astype(np.uint16) << 12)
    ).astype(np.uint16)
    mismatch = packed != packed_ref
    if mismatch.any():
        idx = np.flatnonzero(mismatch)[:10]
        xs = x_np.reshape(-1, 4)[idx]
        detail = "\n".join(
            f"  quad {i}: x={xs[k]} bits={[hex(b) for b in xs[k].view(np.uint32)]} "
            f"rand={r_np[i]:#010x} got={packed[i]:#06x} want={packed_ref[i]:#06x}"
            for k, i in enumerate(idx)
        )
        pytest.fail(f"{mismatch.sum()}/{packed.size} quads mismatch:\n{detail}")


@pytest.mark.parametrize("variant", QUAD_VARIANTS)
@pytest.mark.parametrize("seed", [0, 1])
def test_e2m1_rs_sw_random_bits(seed, variant):
    """4M pure-bit-random f32 (covers NaN payloads, Infs, denormals, the whole
    exponent range) x random rand words, bitwise vs the reference model."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n_vals = 1 << 22
    x = torch.randint(
        -(2**31), 2**31 - 1, (n_vals,), generator=gen, dtype=torch.int32, device="cuda"
    ).view(torch.float32)
    rand = torch.randint(
        -(2**31), 2**31 - 1, (n_vals // 4,), generator=gen, dtype=torch.int32, device="cuda"
    )
    check_bitwise(x, rand, variant)


@pytest.mark.parametrize("variant", QUAD_VARIANTS)
@pytest.mark.parametrize("seed", [0])
def test_e2m1_rs_sw_in_range(seed, variant):
    """4M values concentrated where the grid actually lives (uniform [-8, 8]
    and scaled normals), so every cell and both knees get dense coverage."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n_vals = 1 << 22
    u = torch.rand(n_vals // 2, generator=gen, device="cuda") * 16 - 8
    exps = torch.randint(-20, 5, (n_vals // 2,), generator=gen, device="cuda", dtype=torch.int32)
    m = torch.rand(n_vals // 2, generator=gen, device="cuda") * 2 - 1
    x = torch.cat([u, m * torch.pow(2.0, exps.float())]).float()
    rand = torch.randint(
        -(2**31), 2**31 - 1, (n_vals // 4,), generator=gen, dtype=torch.int32, device="cuda"
    )
    check_bitwise(x, rand, variant)


def _special_values() -> np.ndarray:
    grid = E2M1_GRID[1:]
    vals = [0.0, 6.5, 7.0, 8.0, 100.0, 3.4028235e38, np.inf]
    for g in grid:
        vals += [
            g,
            np.nextafter(np.float32(g), np.float32(-np.inf)),
            np.nextafter(np.float32(g), np.float32(np.inf)),
            g + 0.25 * (0.5 if g < 2 else 1.0),
        ]
    bits = [
        0x00000001,  # min subnormal
        0x007FFFFF,  # max subnormal
        0x00800000,  # min normal
        0x33800000,  # 2^-24
        0x38000000,  # 2^-15
        0x3F7FFFFF,  # just under 1.0
        0x7F800001,  # signaling NaN, min payload
        0x7FC00000,  # canonical quiet NaN
        0x7FFFFFFF,  # max-payload NaN
    ]
    x = np.array(vals, dtype=np.float32)
    bits_np = np.array(bits, dtype=np.uint32)
    return np.concatenate(
        [x, -x, bits_np.view(np.float32), (bits_np | 0x80000000).view(np.float32)]
    )


@pytest.mark.parametrize("variant", QUAD_VARIANTS)
def test_e2m1_rs_sw_specials(variant):
    """Every special (NaNs both signs and payload extremes, +-Inf, +-0,
    subnormals, grid points and their f32 neighbors, clamp region) x rand in
    {all-zeros, all-ones, random}, in every quad slot (values are tiled so
    each special visits each slot)."""
    sp = _special_values()
    n = ((sp.size + 3) // 4) * 4
    reps = []
    for shift in range(4):
        reps.append(np.roll(np.resize(sp, n), shift))
    x_np = np.concatenate(reps)
    rng = np.random.default_rng(0)
    for rand_np in [
        np.zeros(x_np.size // 4, dtype=np.uint32),
        np.full(x_np.size // 4, 0xFFFFFFFF, dtype=np.uint32),
        rng.integers(0, 1 << 32, x_np.size // 4, dtype=np.uint32),
    ]:
        x = torch.from_numpy(x_np).cuda()
        rand = torch.from_numpy(rand_np.view(np.int32)).cuda()
        check_bitwise(x, rand, variant)


@pytest.mark.parametrize("variant", QUAD_VARIANTS)
def test_e2m1_rs_sw_unbiased_exhaustive(variant):
    """Model-independent probe (the same one run against the B300 hw): for
    fixed x, sweep all 256 rand bytes in one quad slot; the round-up count
    must equal D_top exactly — SR is unbiased at 8-bit granularity and the
    byte routing per slot is correct. Runs a handful of x per grid cell,
    including exact grid points (count 0) and cell-edge values (count 255)."""
    rng = np.random.default_rng(0)
    xs = []
    for c in range(7):
        lo, hi = E2M1_GRID[c], E2M1_GRID[c + 1]
        xs += [lo, np.nextafter(np.float32(hi), np.float32(0.0))]
        xs += list(rng.uniform(lo, hi, 6))
    xs = np.array(xs, dtype=np.float32)
    xs = np.concatenate([xs, -xs])
    for slot in range(4):
        x_np = np.zeros((xs.size * 256, 4), dtype=np.float32)
        x_np[:, slot] = np.repeat(xs, 256)
        r_np = np.tile(np.arange(256, dtype=np.uint32), xs.size) << (8 * slot)
        packed = run_quads(
            torch.from_numpy(x_np.reshape(-1)).cuda(),
            torch.from_numpy(r_np.view(np.int32)).cuda(),
            variant,
        )
        nib = (packed >> (4 * slot)) & 0xF
        vals = decode_e2m1(nib).reshape(xs.size, 256)
        base = decode_e2m1(ref_e2m1_rs(xs, np.zeros(xs.size, dtype=np.uint8)))
        up_count = (vals != base[:, None]).sum(axis=1)
        # D_top computed directly (not via the model's carry path)
        a = np.abs(xs.astype(np.float64))
        cell = np.clip(np.searchsorted(E2M1_GRID, a, side="right") - 1, 0, 7)
        g = E2M1_GRID[cell]
        ulp = E2M1_GRID[np.minimum(cell + 1, 7)] - g
        dtop = np.floor((a - g) / ulp * 256.0).astype(np.int64)
        np.testing.assert_array_equal(
            up_count, dtop, err_msg=f"slot {slot}: round-up count != D_top"
        )


# ---------------------------------------------------------------------------
# Fragment-level: convert_f32_frag_sr through the Philox plumbing
# ---------------------------------------------------------------------------

N_PER_THREAD = 32
N_THREADS = 128


@cute.kernel
def _frag_kernel(mX: cute.Tensor, mOut: cute.Tensor, seed: Int32):
    tidx, _, _ = cute.arch.thread_idx()
    frag = cute.make_rmem_tensor(cute.make_layout(N_PER_THREAD), Float32)
    for j in cutlass.range_constexpr(N_PER_THREAD):
        frag[j] = mX[tidx * N_PER_THREAD + j]
    out = convert_f32_frag_sr(frag, cutlass.Float4E2M1FN, seed, tidx)
    out_i32 = cute.recast_tensor(out, Int32)
    for j in cutlass.range_constexpr(N_PER_THREAD // 8):
        mOut[tidx * (N_PER_THREAD // 8) + j] = out_i32[j]


@cute.jit
def _frag_launch(mX: cute.Tensor, mOut: cute.Tensor, seed: Int32):
    _frag_kernel(mX, mOut, seed).launch(grid=(1, 1, 1), block=(N_THREADS, 1, 1))


def philox4_py(counter: int, key: int, n_rounds: int = PHILOX_N_ROUNDS_DEFAULT):
    M = 0xFFFFFFFF
    c0, c1, c2, c3 = counter & M, 0, 0, 0
    k0, k1 = key & M, 0
    for _ in range(n_rounds):
        prod_b = PHILOX_ROUND_B * c2
        prod_a = PHILOX_ROUND_A * c0
        c0 = ((prod_b >> 32) ^ c1 ^ k0) & M
        c2 = ((prod_a >> 32) ^ c3 ^ k1) & M
        c1 = prod_b & M
        c3 = prod_a & M
        k0 = (k0 + PHILOX_KEY_A) & M
        k1 = (k1 + PHILOX_KEY_B) & M
    return c0, c1, c2, c3


@pytest.mark.skipif(
    IS_HW_CVT_TARGET, reason="convert_f32_frag_sr dispatches to the hw cvt.rs on this target"
)
@pytest.mark.parametrize("seed", [0, 12345])
def test_e2m1_frag_sr_philox_wiring(seed):
    """convert_f32_frag_sr(Float4E2M1FN) end to end: Philox counter layout
    (one word per quad, (group<<16)|tid counter, 4 quads per batch), straight
    byte mapping, and the packed-nibble vector bitcast, vs a host model."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n_vals = N_THREADS * N_PER_THREAD
    x = torch.rand(n_vals, generator=gen, device="cuda") * 16 - 8
    x = x.float()
    out = torch.empty(n_vals // 8, dtype=torch.int32, device="cuda")
    fn = cute.compile(_frag_launch, from_dlpack(x), from_dlpack(out), Int32(seed))
    fn(from_dlpack(x), from_dlpack(out), Int32(seed))
    torch.cuda.synchronize()
    got = out.cpu().numpy().view(np.uint8)  # nibble pairs, little-endian per byte

    x_np = x.cpu().numpy().reshape(N_THREADS, N_PER_THREAD)
    nib_ref = np.zeros((N_THREADS, N_PER_THREAD), dtype=np.uint8)
    for t in range(N_THREADS):
        for q in range(N_PER_THREAD // 4):
            group, intra = q // 4, q % 4
            words = philox4_py((group << 16) | t, seed & 0xFFFFFFFF)
            word = words[intra]
            for j in range(4):
                r8 = np.array([(word >> (8 * j)) & 0xFF], dtype=np.uint8)
                nib_ref[t, 4 * q + j] = ref_e2m1_rs(x_np[t, 4 * q + j : 4 * q + j + 1], r8)[0]
    ref_bytes = (nib_ref[:, 0::2] | (nib_ref[:, 1::2] << 4)).reshape(-1)
    np.testing.assert_array_equal(got, ref_bytes)
