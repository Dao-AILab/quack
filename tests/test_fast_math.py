# Copyright (c) 2026, Tri Dao.
"""Tests for quack.fast_math.FastDivmod (magic-number divmod).

Contract under test: exact quotient/remainder for any non-negative Int32 dividend
(n < 2^31) and any divisor 1 <= d < 2^31, with the divisor-1 magic-wrap sentinel.
The failure mode of magic division is q off by one, which first appears at
dividends of the form k*d - 1 (maximal remainder) with large n, and at divisors
that maximize the magic (2^(c-1)+1) or the rounding error (2^c - 1) — so the
tests probe those structured points, not just random values.
"""

import pytest
import torch

import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from quack.fast_math import FastDivmod

LIM = 1 << 31
N = 4096


def edge_divisors():
    divs = set()
    for c in range(1, 32):
        for d in [(1 << c), (1 << c) - 1, (1 << c) + 1, (1 << max(c - 1, 0)) + 1]:
            if 1 <= d < LIM:
                divs.add(d)
    # d=1 (sentinel), classic hard divisors, max divisor, magic-sensitive values
    divs |= {1, 3, 5, 7, 641, 6700417, LIM - 1, LIM - 2, 715827883, 1431655765}
    return sorted(divs)


def reference_magic(d):
    """Host model of FastDivmod's magic selection (mirrors both init paths)."""
    c = (d - 1).bit_length()
    s = max(c - 1, 0)
    m = ((1 << (32 + s)) + d - 1) // d
    return m & 0xFFFFFFFF, s, m


def adversarial_dividends(d, gen):
    vals = [0, 1, 2, d - 1, min(d, LIM - 1), min(d + 1, LIM - 1), LIM - 1, LIM - 2]
    kmax = (LIM - 1) // d
    # max-remainder points k*d - 1 walking down from the top of the range
    for k in range(kmax, max(kmax - 64, 0), -1):
        vals.append(k * d - 1)
        vals.append(min(k * d, LIM - 1))
    vals = [v for v in vals if 0 <= v < LIM]
    pad = torch.randint(0, LIM - 1, (N - len(vals),), generator=gen, dtype=torch.int64)
    return torch.cat([torch.tensor(vals, dtype=torch.int64), pad]).to(torch.int32)


def test_magic_fits_32_bits():
    """The round-up magic must fit in u32 for every divisor (2^32 only for d=1,
    which wraps to the 0 sentinel). Worst case is d = 2^(c-1) + 1."""
    for d in edge_divisors():
        _, _, m_full = reference_magic(d)
        if d == 1:
            assert m_full == 1 << 32  # wraps to sentinel 0
        else:
            assert m_full < 1 << 32, f"magic overflow for d={d}"


def test_host_model_exact_below_2_31():
    for d in edge_divisors():
        m, s, _ = reference_magic(d)
        kmax = (LIM - 1) // d
        ns = {0, 1, 2, d - 1, d, LIM - 1, LIM - 2, kmax * d - 1, kmax * d}
        for n in ns:
            if not (0 <= n < LIM):
                continue
            q = n if m == 0 else ((n * m) >> 32) >> s
            assert q == n // d, f"d={d} n={n}"


def test_bound_is_tight_above_2_31():
    """The 2^31 contract edge is real: for non-power-of-2 divisors the first
    structured failure exists in [2^31, 2^32) and never below."""
    shown = 0
    for d in edge_divisors():
        if d < 3 or (d & (d - 1)) == 0:
            continue  # powers of two are exact everywhere
        m, s, m_full = reference_magic(d)
        r0 = m_full * d - (1 << (32 + s))
        if r0 == 0:
            continue
        k = ((1 << (32 + s)) // r0) // d + 1
        for kk in range(max(k - 2, 1), k + 200):
            n = kk * d - 1
            if n >= (1 << 32):
                break
            q = ((n * m) >> 32) >> s
            if q != n // d:
                assert n >= LIM, f"failure INSIDE contract: d={d} n={n}"
                shown += 1
                break
    assert shown > 0  # the demonstration found real out-of-contract failures


@cute.kernel
def _divmod_kernel(mN: cute.Tensor, mQ: cute.Tensor, mR: cute.Tensor, divisor: Int32):
    fdd = FastDivmod(divisor)  # dynamic-divisor path (ctlz + 64-bit magic math)
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 256 + tidx
    if i < cute.size(mN.shape):
        q, r = divmod(mN[i], fdd)
        mQ[i] = q
        mR[i] = r


@cute.jit
def _launch(mN: cute.Tensor, mQ: cute.Tensor, mR: cute.Tensor, divisor: Int32):
    _divmod_kernel(mN, mQ, mR, divisor).launch(
        grid=(cute.ceil_div(cute.size(mN.shape), 256), 1, 1), block=(256, 1, 1)
    )


@pytest.mark.parametrize("seed", [0])
def test_device_divmod_adversarial(seed):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    gen = torch.Generator().manual_seed(seed)
    n_dev = torch.empty(N, dtype=torch.int32, device="cuda")
    q_dev = torch.empty_like(n_dev)
    r_dev = torch.empty_like(n_dev)
    fn = cute.compile(_launch, from_dlpack(n_dev), from_dlpack(q_dev), from_dlpack(r_dev), Int32(1))
    for d in edge_divisors():
        n_host = adversarial_dividends(d, gen)
        n_dev.copy_(n_host)
        fn(from_dlpack(n_dev), from_dlpack(q_dev), from_dlpack(r_dev), Int32(d))
        torch.cuda.synchronize()
        n64 = n_host.to(torch.int64)
        q_ref = n64 // d
        torch.testing.assert_close(q_dev.cpu().to(torch.int64), q_ref, rtol=0, atol=0)
        torch.testing.assert_close(r_dev.cpu().to(torch.int64), n64 - q_ref * d, rtol=0, atol=0)
