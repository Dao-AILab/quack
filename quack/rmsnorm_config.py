# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

"""Launch configuration for the RMSNorm backward kernel.

Mirrors :mod:`quack.gemm_config`: a frozen dataclass that captures the launch
knobs, plus arch-specific factories that own the heuristics. The forward
kernel's launch logic is much simpler and stays inline in :class:`RMSNorm`.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class RmsNormBwdConfig:
    num_threads: int
    threads_per_row: int
    cluster_n: int
    # None = recompute from registers; "smem" = reload from shared memory.
    reload_wdy: Optional[str]
    reload_x: Optional[str]
    use_tma: bool
    # Number of smem stages used by the prefetch pipeline. Drives both the
    # cp.async (use_tma=False) and TMA (use_tma=True) paths, which lead by
    # ``smem_stages - 1`` batches. Larger depths hide more latency at the cost
    # of smem footprint.
    smem_stages: int = 2

    @classmethod
    def from_analytical_heuristic(
        cls,
        N: int,
        dtype_width: int,
        dout_width: int,
        arch_major: Optional[int] = None,
        T_hint: int = 0,
    ) -> "RmsNormBwdConfig":
        """Pick a launch config from the hand-tuned analytical heuristic.

        ``arch_major`` defaults to the current device's capability.
        ``arch_major >= 10`` selects the Blackwell heuristic; anything else
        uses the legacy/default path tuned on Hopper. For autotuning, use
        :func:`get_all_bwd_configs`.
        """
        if arch_major is None:
            arch_major = _detect_arch_major()
        if arch_major >= 10:
            return _for_blackwell_bwd(N, dtype_width, dout_width, T_hint)
        return _for_hopper_bwd(N, dtype_width, arch_major)


def _for_hopper_bwd(N: int, dtype_width: int, arch_major: int) -> RmsNormBwdConfig:
    num_threads = 128 if N <= 4096 else 256
    for limit, threads in [(64, 8), (128, 16), (256, 32), (512, 64), (4096, 128)]:
        if N <= limit:
            threads_per_row = threads
            break
    else:
        threads_per_row = 256

    if arch_major < 9:
        cluster_n = 1
    else:
        max_cluster = 8 if arch_major == 12 else 16
        if arch_major == 12 and dtype_width >= 32:
            thresholds = [(1024, 1), (8 * 1024, 2), (16 * 1024, 4), (32 * 1024, 8)]
        else:
            thresholds = [(8 * 1024, 1), (16 * 1024, 2), (32 * 1024, 4), (64 * 1024, 8)]
        cluster_n = max_cluster
        for limit, cluster in thresholds:
            if N <= limit:
                cluster_n = cluster
                break

    return RmsNormBwdConfig(
        num_threads=num_threads,
        threads_per_row=threads_per_row,
        cluster_n=cluster_n,
        reload_wdy=None if N <= 16 * 1024 else "smem",
        reload_x=None,
        use_tma=False,
    )


def _for_blackwell_bwd(
    N: int, dtype_width: int, dout_width: int, T_hint: int = 0
) -> RmsNormBwdConfig:
    """Pick a launch config for RMSNorm bwd on Blackwell.

    All thresholds are expressed in ``row_bytes = N * max(x, dout)`` so a
    single ladder handles bf16, fp32, and mixed-dtype combinations. ``x_bytes``
    governs the X tile (loads, smem footprint); ``max_bytes`` is the wider
    side which sets register pressure for the per-thread fragments.
    """
    # Safety floor for very narrow rows: keep tpr below 128 so we don't over-
    # parallelise tiny problems.
    if N <= 64:
        threads_per_row = 8
    elif N <= 128:
        threads_per_row = 16
    elif N <= 256:
        threads_per_row = 32
    elif N <= 512:
        threads_per_row = 64
    else:
        threads_per_row = None

    if threads_per_row is not None:
        return RmsNormBwdConfig(
            num_threads=128,
            threads_per_row=threads_per_row,
            cluster_n=1,
            reload_wdy=None,
            reload_x=None,
            use_tma=False,
        )

    max_bytes = max(dtype_width, dout_width) // 8
    row_bytes = N * max_bytes

    if row_bytes >= 48 * 1024:
        # Spread the row across a CTA cluster. Step back to cluster_n=4 only
        # when T is tiny AND the row isn't extreme — otherwise cn=8 keeps each
        # CTA's tile small enough to fit comfortably in registers.
        cluster_n = 4 if 0 < T_hint <= 1024 and row_bytes <= 64 * 1024 else 8
        num_threads, threads_per_row = 128, 128
        # Override if this cluster_n would overflow the device's smem budget.
        cluster_n = _bump_cluster_n_for_smem(
            cluster_n,
            N,
            smem_stages=2,
            sum_bytes=(dtype_width + dout_width) // 8,
            max_cluster=_max_cluster_for(10),  # Blackwell
        )
    elif row_bytes > 16 * 1024:
        # Wider than 128 threads can comfortably handle at cluster_n=1; bump
        # threads/row to keep per-thread fragments small.
        cluster_n = 1
        num_threads, threads_per_row = 256, 256
    else:
        cluster_n = 1
        num_threads, threads_per_row = 128, 128

    bytes_per_thread_frag = (N // cluster_n) // threads_per_row * max_bytes

    # TMA pays off when the cluster needs prefetch (cn>=4), and also for
    # fp32-class single-CTA wide rows where TMA's wider descriptors amortise
    # setup. Pure bf16 single-CTA cases mostly don't benefit and can lose ~5%.
    use_tma = cluster_n >= 4 or (max_bytes >= 4 and row_bytes >= 16 * 1024)
    # reload_x: wide end of the cluster ladder, plus fp32-class single-CTA
    # cases where the wider X fragments crowd registers across the row
    # reduction barrier.
    reload_x = (
        "smem"
        if (cluster_n >= 8 and N >= 32 * 1024)
        or (cluster_n == 1 and max_bytes >= 4 and bytes_per_thread_frag >= 64)
        else None
    )
    # reload_wdy: cluster cases get it for free, plus single-CTA cases where
    # each thread holds ≥64 bytes of fragment (the wdy register count is then
    # large enough to spill).
    reload_wdy = "smem" if cluster_n >= 4 or bytes_per_thread_frag >= 64 else None

    return RmsNormBwdConfig(
        num_threads=num_threads,
        threads_per_row=threads_per_row,
        cluster_n=cluster_n,
        reload_wdy=reload_wdy,
        reload_x=reload_x,
        use_tma=use_tma,
    )


def _get_sm_count_hopper(N: int, sm_count: int) -> int:
    # This should be tuned on how many CTAs can be launched on each SM.
    sm_count_multiple = (
        16 if N <= 256 else (8 if N <= 1024 else (4 if N <= 2048 else (2 if N <= 4096 else 1)))
    )
    # By right, if we're using cluster, this should be cluster_count not sm_count.
    # But for cluster >= 4, due to quantization we would need to query active max cluster.
    # Instead we just do sm_count * 2, which is reasonably larger than active_cluster_count to
    # avoid wave quantization.
    return (
        sm_count * sm_count_multiple if N <= 8192 else sm_count // 2 if N <= 16384 else sm_count * 2
    )


def _get_sm_count_blackwell(N: int, sm_count: int) -> int:
    if N <= 256:
        return sm_count * 16
    if N <= 1024:
        return sm_count * 8
    if N <= 2048:
        return sm_count * 4
    return sm_count * 2


def get_sm_count(N: int, device: torch.device) -> int:
    props = torch.cuda.get_device_properties(device)
    if props.major >= 10:
        return _get_sm_count_blackwell(N, props.multi_processor_count)
    return _get_sm_count_hopper(N, props.multi_processor_count)


_CTA_THREAD_SIZE = (128, 256)
_THREADS_PER_REDUCTION_DIM = (8, 16, 32, 64, 128, 256)


def _max_dynamic_smem_bytes() -> int:
    """Per-CTA opt-in dynamic smem capacity for the current device.

    Returns 0 when CUDA is unavailable (callers should treat this as "no
    smem-budget guard"). Falls back to ``shared_memory_per_block`` on older
    PyTorch builds that lack the ``_optin`` field.
    """
    if not torch.cuda.is_available():
        return 0
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)


def _bump_cluster_n_for_smem(
    cluster_n: int,
    N: int,
    smem_stages: int,
    sum_bytes: int,
    max_cluster: int,
) -> int:
    """Raise ``cluster_n`` to the smallest power of 2 such that the bwd's
    sX+sdO data buffers fit under the device's opt-in dynamic smem.

    Footprint per CTA is ``(N / cluster_n) * smem_stages * sum_bytes`` where
    ``sum_bytes = x_bytes + dout_bytes``. Snaps up to the next power of 2,
    floors at the input ``cluster_n`` (never lowers the tuning's choice), and
    caps at ``max_cluster``. If the required cluster_n exceeds ``max_cluster``
    the runtime guard in ``RMSNormBackward.__init__`` raises a precise overflow
    error. Returns the input unchanged if CUDA is unavailable.
    """
    # Reserved for row-reduction buffer, mbars, and smem alignment overhead.
    _BWD_SMEM_RESERVED_BYTES = 4 * 1024
    smem_max = _max_dynamic_smem_bytes()
    if smem_max <= 0:
        return cluster_n
    budget = max(smem_max - _BWD_SMEM_RESERVED_BYTES, 1)
    needed = (N * smem_stages * sum_bytes + budget - 1) // budget  # ceil-div
    pow2 = 1
    while pow2 < needed:
        pow2 *= 2
    return min(max(pow2, cluster_n), max_cluster)


def _detect_arch_major() -> int:
    """Return the major device capability of the current CUDA device.

    Falls back to 0 (no-cluster, no-TMA) when CUDA is unavailable so the
    autotune search space stays well-defined for CPU-only imports.
    """
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_capability(torch.cuda.current_device())[0]


def _max_cluster_for(arch_major: int) -> int:
    """Maximum cluster_n supported on this arch."""
    if arch_major < 9:
        return 1
    # SM12x (RTX 50) supports up to 8; Hopper/Blackwell up to 16.
    return 8 if arch_major == 12 else 16
