# Copyright (c) 2026, Tri Dao.
"""MXFP8 grouped GEMM dispatcher for quack.

Routes a grouped (per-expert) MXFP8 GEMM over cumulative ``offs`` onto quack's
two existing SM100 blockscaled engines, mirroring ``torch._scaled_grouped_mm``:

  * uniform 128-aligned groups -> batched-L dense (:func:`mxfp8_gemm_out`)
  * ragged 128-aligned groups  -> varlen-M (:func:`compile_blockscaled_gemm_tvm_ffi`)
  * ragged non-128 groups      -> varlen-M, dQaccum-padded SFA

Contract (2D-A / 3D-B + offs, like ``torch._scaled_grouped_mm``)::

    a:    (total_m, K)     float8_e4m3fn, K-contiguous
    b:    (E, K, N)        float8_e4m3fn, K-contiguous (b.transpose(-2,-1).is_contiguous())
    offs: (E,)             int32, CUMULATIVE per-group end rows; offs[-1] == total_m
    sfa:  (total_m, K//32) float8_e8m0fnu, K-contiguous (raw per-block scales)
    sfb:  (E, N, K//32)    float8_e8m0fnu, K-contiguous (raw per-block scales)
    out:  optional (total_m, N) bfloat16, contiguous

Returns bf16 ``(total_m, N)``.
"""

from functools import lru_cache
from typing import List, Optional, Tuple

import cutlass
import torch
from torch import Tensor

from quack.blockscaled_gemm_utils import (
    compile_blockscaled_gemm_tvm_ffi,
    pack_scale_2d_to_blocked_contig,
    scale_view_for_kernel,
)
from quack.gemm_blockscaled_interface import (
    _compile_cached,
    _default_tiler_cluster,
    mxfp8_gemm_out,
)

SF_VEC = 32


def _uniform_group_size(offsets_host: Tuple[int, ...]) -> Optional[int]:
    prev, g = 0, None
    for cur in offsets_host:
        sz = cur - prev
        if g is None:
            g = sz
        elif sz != g:
            return None
        prev = cur
    return g


def _group_sizes(offsets_host: Tuple[int, ...]) -> List[int]:
    prev, out = 0, []
    for cur in offsets_host:
        out.append(cur - prev)
        prev = cur
    return out


def _boundaries_128_aligned(offsets_host: Tuple[int, ...]) -> bool:
    prev = 0
    for cur in offsets_host:
        if prev % 128 != 0 or cur % 128 != 0:
            return False
        prev = cur
    return True


def _choose_route(offsets_host: Tuple[int, ...]) -> Tuple[str, Optional[int]]:
    """Pick the kernel route from cumulative host offsets. Returns (route, g) where g is
    the uniform group size for the 'uniform' route, else None. Routes:
      'uniform'        -> batched-L dense (equal, 128-aligned groups)
      'varlen'         -> varlen-M, natural SFA (ragged, all boundaries 128-aligned)
      'varlen_dqaccum' -> varlen-M, dQaccum-padded SFA (arbitrary / non-128 boundaries)
    """
    g = _uniform_group_size(offsets_host)
    if g is not None and g % 128 == 0:
        return "uniform", g
    if _boundaries_128_aligned(offsets_host):
        return "varlen", None
    return "varlen_dqaccum", None


def _dqaccum_padded_sfa(sfa: Tensor, group_sizes: List[int], sf_k: int, e: int) -> Tensor:
    """Pack SFA for the varlen-M kernel in dQaccum-padded layout: expert i's
    scale rows live at padded tile offset (cu_seqlens[i] // 128 + i) * 128,
    matching VarlenManager.offset_batch_SFA. Returns packed (1, rm, rk, 512)."""
    tile = 128
    total_m = sum(group_sizes)
    total_padded_m = ((total_m + tile - 1) // tile + (e - 1)) * tile
    sa_padded = sfa.new_zeros(total_padded_m, sf_k)
    off = 0
    for i, m_i in enumerate(group_sizes):
        off_padded = (off // tile + i) * tile
        sa_padded[off_padded : off_padded + m_i] = sfa[off : off + m_i]
        off += m_i
    return pack_scale_2d_to_blocked_contig(sa_padded.view(1, total_padded_m, sf_k))


@lru_cache(maxsize=32)
def _varlen_runner(n: int, k: int, e: int, tiler, cluster, sfa_tile_aligned: bool = True):
    # Compile once; the kernel uses cute.sym_int for total_m so one compile serves
    # all group configurations with the same (n, k, e, tiler, cluster). With
    # sfa_tile_aligned=True the kernel reads SFA in natural (unpadded) packed layout
    # (cu // 128 offset); =False expects the dQaccum-padded layout (cu // 128 + b).
    dev = torch.device("cuda")
    sf_k = k // SF_VEC
    fake_mA = torch.empty(128 * e, k, dtype=torch.float8_e4m3fn, device=dev)
    fake_mB = torch.empty(e, n, k, dtype=torch.float8_e4m3fn, device=dev).permute(1, 2, 0)
    fake_mD = torch.empty(128 * e, n, dtype=torch.bfloat16, device=dev)
    z = torch.zeros(128 * e, sf_k, dtype=torch.float8_e8m0fnu, device=dev)
    if sfa_tile_aligned:
        fake_sfa = pack_scale_2d_to_blocked_contig(z.view(1, 128 * e, sf_k))
    else:
        fake_sfa = _dqaccum_padded_sfa(z, [128] * e, sf_k, e)
    fake_sfb = pack_scale_2d_to_blocked_contig(
        torch.zeros(e, n, sf_k, dtype=torch.float8_e8m0fnu, device=dev)
    )
    return compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        SF_VEC,
        cutlass.BFloat16,
        tiler,
        cluster,
        fake_mA,
        fake_mB,
        fake_mD,
        fake_sfa,
        fake_sfb,
        varlen_m=True,
        sfa_tile_aligned=sfa_tile_aligned,
    )


def _run_varlen_natural(a, offs, sfa, mB, mSFB, out, e, k, n, sf_k, tiler, cluster):
    """Run the varlen-M kernel with NATURAL (unpadded) SFA -- packs SFA like torch's
    to_blocked (no dQaccum scatter) and uses the sfa_tile_aligned kernel path. Requires
    128-aligned per-expert boundaries (caller-checked). ``mB`` / ``mSFB`` are the (fixed)
    B operand + packed B-scale; ``cu_seqlens`` is built on-device from ``offs``."""
    total_m = a.shape[0]
    mSFA = pack_scale_2d_to_blocked_contig(sfa.view(1, total_m, sf_k))
    # torch.zeros (not empty + ``cu[0] = 0``) so building cu_seqlens is a pure device op:
    # a host-scalar ``cu[0] = 0`` write is a CPU->CUDA copy that breaks CUDA-graph capture.
    cu = torch.zeros(e + 1, dtype=torch.int32, device=offs.device)
    cu[1:].copy_(offs.to(torch.int32))
    runner = _varlen_runner(n, k, e, tiler, cluster, sfa_tile_aligned=True)
    runner(a, mB, out, mSFA, mSFB, cu)
    return out


def _default_tiler_cluster_for(total_m: int, e: int, n: int):
    """Single sync-free tiler choice for the whole grouped problem, keyed on the MEAN
    tokens-per-expert (total_m // e). Using the mean (not max(group_sizes)) keeps the
    choice sync-free -- derived from tensor shapes, no device->host copy."""
    return _default_tiler_cluster(max(1, total_m // e), n)


def _run_varlen_dqaccum(a, offs, sfa, mB, mSFB, out, e, k, n, sf_k, tiler, cluster):
    """Run the varlen-M kernel for ARBITRARY (incl. non-128) ragged boundaries: pack SFA in
    dQaccum-padded layout and use the sfa_tile_aligned=False kernel path. A and the output are
    passed NATURALLY (no full-A pad/copy, no gather-back -- only the small SFA is padded).
    cu_seqlens is built on device from offs; group sizes (host) are needed only for the SFA pack."""
    total_m = a.shape[0]
    group_sizes = _group_sizes(tuple(int(v) for v in offs.detach().cpu().tolist()))
    mSFA = _dqaccum_padded_sfa(sfa, group_sizes, sf_k, e)
    cu = torch.zeros(e + 1, dtype=torch.int32, device=offs.device)
    cu[1:].copy_(offs.to(torch.int32))
    runner = _varlen_runner(n, k, e, tiler, cluster, sfa_tile_aligned=False)
    runner(a, mB, out, mSFA, mSFB, cu)
    return out


def mxfp8_grouped_gemm(
    a: Tensor,
    b: Tensor,
    offs: Tensor,
    sfa: Tensor,
    sfb: Tensor,
    out: Optional[Tensor] = None,
    uniform: bool = False,
    varlen: bool = False,
) -> Tensor:
    """Grouped MXFP8 GEMM. See module docstring for shape/layout contract.

    If ``uniform=True`` the caller asserts every group has the same size
    ``total_m // E`` (e.g. fixed-capacity MoE). The route is then determined from
    shapes alone, so ``offs`` is never read on the host and the call runs the dense
    batched-L path directly.

    If ``varlen=True`` the caller instead asserts ragged groups with 128-aligned
    boundaries (e.g. capacity-padded MoE). ``offs`` is consumed only on device
    (cu_seqlens) and the varlen-M kernel uses its natural (sfa_tile_aligned) SFA path.

    Both flag paths read ``offs`` only on device, so they are CUDA-graph capturable.
    ``uniform`` and ``varlen`` are mutually exclusive.
    """
    assert a.dim() == 2 and b.dim() == 3 and sfa.dim() == 2 and sfb.dim() == 3
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert sfa.dtype == torch.float8_e8m0fnu and sfb.dtype == torch.float8_e8m0fnu
    assert not (uniform and varlen), "uniform and varlen are mutually exclusive"
    total_m, k = a.shape
    e, kb, n = b.shape
    assert kb == k and k % SF_VEC == 0, f"K mismatch a={k} b={kb} (or not %{SF_VEC})"
    sf_k = k // SF_VEC
    assert tuple(sfa.shape) == (total_m, sf_k), f"sfa {tuple(sfa.shape)} != {(total_m, sf_k)}"
    assert tuple(sfb.shape) == (e, n, sf_k), f"sfb {tuple(sfb.shape)} != {(e, n, sf_k)}"
    if out is None:
        out = torch.empty((total_m, n), dtype=torch.bfloat16, device=a.device)
    if total_m == 0:
        return out

    # Shape-determined route (g = total_m // E); offs is never read on the host.
    # The caller asserts groups are equal & 128-aligned (e.g. fixed-capacity MoE).
    if uniform:
        assert total_m % e == 0, f"uniform=True needs total_m {total_m} % E {e} == 0"
        g = total_m // e
        assert g % 128 == 0, f"uniform=True needs (total_m // E) = {g} to be % 128 == 0"
        mxfp8_gemm_out(
            a.view(e, g, k), b, sfa.view(e, g, sf_k), sfb.transpose(1, 2), out.view(e, g, n)
        )
        return out

    # Ragged groups with 128-aligned boundaries (caller-asserted). offs is consumed
    # only on device; natural SFA via the sfa_tile_aligned kernel path (no dQaccum scatter).
    if varlen:
        assert total_m % 128 == 0, f"varlen=True needs total_m {total_m} % 128 == 0"
        tiler, cluster = _default_tiler_cluster(total_m // e, n)
        return _run_varlen_natural(
            a,
            offs,
            sfa,
            b.permute(2, 1, 0),
            pack_scale_2d_to_blocked_contig(sfb),
            out,
            e,
            k,
            n,
            sf_k,
            tiler,
            cluster,
        )

    offsets_host = tuple(int(v) for v in offs.detach().cpu().tolist())
    assert len(offsets_host) == e, f"offs len {len(offsets_host)} != E {e}"
    assert offsets_host[-1] == total_m, f"offs[-1] {offsets_host[-1]} != total_m {total_m}"
    route, g = _choose_route(offsets_host)

    if route == "uniform":
        mxfp8_gemm_out(
            a.view(e, g, k), b, sfa.view(e, g, sf_k), sfb.transpose(1, 2), out.view(e, g, n)
        )
        return out

    tiler, cluster = _default_tiler_cluster_for(total_m, e, n)  # mean (total_m // e), sync-free
    mB = b.permute(2, 1, 0)
    mSFB = pack_scale_2d_to_blocked_contig(sfb)
    if route == "varlen":
        return _run_varlen_natural(a, offs, sfa, mB, mSFB, out, e, k, n, sf_k, tiler, cluster)
    return _run_varlen_dqaccum(a, offs, sfa, mB, mSFB, out, e, k, n, sf_k, tiler, cluster)


class MXFP8GroupedGemm:
    """Prepared grouped MXFP8 GEMM with the fixed B operand baked in.

    Pre-packs the B-scale and reuses the cached compiled kernel, so each call does
    only per-step work (A-scale pack + the kernel). Call it like
    ``torch._scaled_grouped_mm``::

        gemm = MXFP8GroupedGemm(b, sfb)        # b/sfb are the fixed expert weights
        out = gemm(a, offs, sfa)               # per step (offs may change per call)
        out = gemm(a, offs, sfa, out=buf)      # reuse an output buffer

    With ``uniform=True`` (at construction or per call) the route is derived from
    shapes (g = total_m // E), so ``offs`` is never read on the host.

    Per-call work is on fresh tensors; for CUDA-graph replay, call once on the
    capture tensors then update ``a`` / ``sfa`` / ``out`` in place between replays.
    """

    def __init__(self, b: Tensor, sfb: Tensor, uniform: bool = False):
        assert b.dim() == 3 and sfb.dim() == 3
        assert b.dtype == torch.float8_e4m3fn and sfb.dtype == torch.float8_e8m0fnu
        e, k, n = b.shape
        assert k % SF_VEC == 0, f"K={k} not a multiple of {SF_VEC}"
        sf_k = k // SF_VEC
        assert tuple(sfb.shape) == (e, n, sf_k), f"sfb {tuple(sfb.shape)} != {(e, n, sf_k)}"
        self.e, self.k, self.n, self.sf_k = e, k, n, sf_k
        self.uniform = uniform
        # Pre-pack the fixed B-scale once; shared across calls and routes.
        self._sfb_packed = pack_scale_2d_to_blocked_contig(sfb)  # varlen passes this directly
        self._sfb_view_dense = scale_view_for_kernel(self._sfb_packed, n, sf_k, e)  # dense view
        self._mB_dense = b.mT.permute(1, 2, 0)  # (n,k,e) K-major view (dense/uniform)
        self._mB_varlen = b.permute(2, 1, 0)  # (n,k,e) K-major view (varlen)

    def _dense(self, a: Tensor, g: int, sfa: Tensor, out: Tensor) -> Tensor:
        e, k, n, sf_k = self.e, self.k, self.n, self.sf_k
        mA = a.view(e, g, k).permute(1, 2, 0)
        mD = out.view(e, g, n).permute(1, 2, 0)
        tiler, cluster = _default_tiler_cluster(g, n)
        runner = _compile_cached(
            g, n, k, e, tiler, cluster, torch.bfloat16, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
        )
        sfa_view = scale_view_for_kernel(
            pack_scale_2d_to_blocked_contig(sfa.view(e, g, sf_k)), g, sf_k, e
        )
        runner(mA, self._mB_dense, mD, sfa_view, self._sfb_view_dense)
        return out

    def __call__(
        self,
        a: Tensor,
        offs: Tensor,
        sfa: Tensor,
        out: Optional[Tensor] = None,
        uniform: Optional[bool] = None,
        varlen: bool = False,
    ) -> Tensor:
        uniform = self.uniform if uniform is None else uniform
        assert not (uniform and varlen), "uniform and varlen are mutually exclusive"
        e, k, n, sf_k = self.e, self.k, self.n, self.sf_k
        total_m, ka = a.shape
        assert ka == k, f"K mismatch a={ka} vs b={k}"
        assert a.dtype == torch.float8_e4m3fn and sfa.dtype == torch.float8_e8m0fnu
        assert tuple(sfa.shape) == (total_m, sf_k), f"sfa {tuple(sfa.shape)} != {(total_m, sf_k)}"
        if out is None:
            out = torch.empty((total_m, n), dtype=torch.bfloat16, device=a.device)
        if total_m == 0:
            return out

        # Sync-free uniform path: shape-derived route + pre-packed B-scale.
        if uniform:
            assert total_m % e == 0, f"uniform=True needs total_m {total_m} % E {e} == 0"
            g = total_m // e
            assert g % 128 == 0, f"uniform=True needs (total_m // E) = {g} % 128 == 0"
            return self._dense(a, g, sfa, out)

        # Sync-free varlen path: ragged 128-aligned groups; natural SFA, cached B-scale.
        if varlen:
            assert total_m % 128 == 0, f"varlen=True needs total_m {total_m} % 128 == 0"
            tiler, cluster = _default_tiler_cluster(total_m // e, n)
            return _run_varlen_natural(
                a, offs, sfa, self._mB_varlen, self._sfb_packed, out, e, k, n, sf_k, tiler, cluster
            )

        # General route via offs (one host sync; B-scale already packed).
        offsets_host = tuple(int(v) for v in offs.detach().cpu().tolist())
        assert len(offsets_host) == e, f"offs len {len(offsets_host)} != E {e}"
        assert offsets_host[-1] == total_m, f"offs[-1] {offsets_host[-1]} != total_m {total_m}"
        route, g = _choose_route(offsets_host)
        if route == "uniform":
            return self._dense(a, g, sfa, out)
        tiler, cluster = _default_tiler_cluster_for(total_m, e, n)
        if route == "varlen":
            return _run_varlen_natural(
                a, offs, sfa, self._mB_varlen, self._sfb_packed, out, e, k, n, sf_k, tiler, cluster
            )
        return _run_varlen_dqaccum(
            a, offs, sfa, self._mB_varlen, self._sfb_packed, out, e, k, n, sf_k, tiler, cluster
        )
