# Copyright (c) 2026, QuACK team.
"""Blockscaled AllGather+GEMM: gather the quantized values AND the blocked
scale factors under ONE flag set, fully overlapped with the GEMM.

A blockscaled GEMM consumes two distributed tensors per shard — the qdata
bytes and their per-block scale factors — but the AG gate contract is
per-shard, not per-tensor: the AB-load warp's gate (ag_wait_m_tile)
precedes ALL of a tile's TMA loads, SFA included. So the ONLY missing piece
is transport-side: get the shard's SFA bytes into a resident symmetric
buffer before that shard's flag lands. Without this, callers pre-gather SFA
with an exposed NCCL all-gather on the compute stream.

Quantization is the CALLER's job: the runner transports already-quantized
operands only. Scale blocks are K major, so per-shard
quantization of an M-sharded tensor is bit-identical to quantizing the
gathered tensor.

The interface is the parent's, made type-symmetric: ``gather(X)`` yields
``(X_gathered, ag_args)``. A plain ``torch.Tensor`` X runs the parent
verbatim.
:class:`~quack.blockscaled.operand.BlockScaledOperand` X engages the SFA
lane and yields a gathered ``BlockScaledOperand``. The operand is quack's
canonical (qdata, scale, format) container. Its constructor enforces the
qdata-dtype-vs-format and blocked-scale-atom invariants, so a runner/GEMM
format mixup still fails loudly at construction or at the GEMM's
`validate_blockscaled_sf` cross-check.

Two facts make the SFA lane nearly free:

1. LAYOUT: the hardware-fixed blocked SF layout (rm=M/128, rk, 32, 4, 4) is
   contiguous with the 128-row M-block dim OUTERMOST, so for
   shard_m % 128 == 0 each rank's packed shard SFA is EXACTLY a contiguous
   byte slice of the full packed tensor at offset rank * sfa_shard_bytes.

2. ORDERING: SFA peer sends are enqueued on the parent's push_stream DURING
   the gather() body, i.e. BEFORE the parent's __exit__ enqueues the
   [A send, flag write] pairs. Same-stream CE order therefore places every
   peer's SFA memcpy before that peer's flag write: one flag publishes both
   payloads, by the same flag-implies-data argument the parent makes for A
   alone. (Per peer the CE queue runs [SFA, A, flag] — the tiny SFA op
   first, so it never delays the shard's arrival.) Local SFA staging runs
   on the ambient stream after the parent already wrote the LOCAL flags,
   which is sound: local flags only matter to MY GEMM, whose launch is
   ambient-ordered after the staging; peers' gates read THEIR flag
   replicas, written by the push stream.

FORMATS: the runner is parameterized by the A-side format name (the same
descriptors the dense GEMM resolves from ``bs_format_a``), which supplies
the qdata byte geometry (``storage_k``: fp8 K bytes/row, fp4x2 K/2, packed
fp6 3K/4), the scale dtype (e8m0; e4m3 for nvfp4), and the SF block size
(``sf_vec_size``: 32; 16 for nvfp4). B and SFB never touch the runner —
they are replicated operands — so mixed A x B format pairs work exactly as
in the dense GEMM: pass the matching ``bs_format_a``/``bs_format_b`` to the
GEMM inside the body. Validated on 2xGB200: the full
{mxfp8_e4m3, mxfp8_e5m2, mxfp4}^2 A x B matrix, bitwise-equal to the dense
GEMM on NCCL-gathered operands, eager and under CUDA-graph replay.

Everything else is inherited unchanged from `AllGatherRunner` (its module
docstring is the base design record): the SFA buffer rotates on the same
2-buffer parity, and reuse safety needs NO new sync. `ev_reuse` certifies
"all ranks' GEMM on this parity finished", and the GEMM reads A and SFA
together, so the same event covers both buffers. Capture safety likewise:
`ev_sfa_staged` is recorded and waited within the same call, the SFA sends
ride the parent-joined `push_stream`, and `_wait_reuse`'s capture-skip logic is
reused verbatim (the body-time wait sees exactly the state the parent's
exit-time wait would).

`arrival_chunks` is fixed at 1 (chunk boundaries would also have to land on
128-row SF blocks; chunking small shards regresses anyway).
"""

from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cuda.bindings import runtime
from quack.blockscaled.operand import BlockScaledFormat, BlockScaledOperand
from quack.distributed.all_gather_gemm import _check, AllGatherRunner

__all__ = ["BlockScaledAllGatherRunner"]


class BlockScaledAllGatherRunner(AllGatherRunner):
    """AllGatherRunner with a scale-factor lane for blockscaled GEMMs.

    Same ``gather()`` interface as the parent, over
    :class:`BlockScaledOperand` shards::

        runner = BlockScaledAllGatherRunner(shard_m, k, "mxfp8_e4m3")
        shard = BlockScaledOperand(a_q, sfa_blocked, "mxfp8_e4m3")
        with runner.gather(shard) as (a_op, ag_args):
            gemm(a_op.qdata, b_q, d, None, None, tm, tn, cm, cn,
                 SFA=a_op.scale, SFB=sfb, bs_format_a="mxfp8_e4m3",
                 bs_format_b="mxfp8_e5m2",  # B format independent, as in dense
                 ag_args=ag_args)

    A plain-tensor shard falls through to the parent's byte-only gather.
    Any number of blockscaled GEMMs may share the gather, exactly as with
    the parent.
    """

    def __init__(
        self,
        shard_m: int,
        k: int,
        a_format: str = "mxfp8_e4m3",
        group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        capture_lockstep: bool = False,
    ):
        fmt = BlockScaledFormat.from_name(a_format)
        # storage extent of one row, in container elements (all registered
        # containers are 1-byte, so this is also the row byte count)
        row_storage = fmt.storage_k(k)
        row_bytes = row_storage * fmt.qdata_dtype.itemsize
        # The parent transports the shard as raw (shard_m, row_bytes) uint8.
        super().__init__(
            shard_m,
            row_bytes,
            torch.uint8,
            group=group,
            device=device,
            arrival_chunks=1,
            capture_lockstep=capture_lockstep,
        )
        self.a_fmt = fmt
        self.logical_k = k
        self.row_storage = row_storage
        # No-padding geometry: shard boundaries on whole 128-row SF atoms,
        # sf_k on whole 4-wide SF blocks (pack_scale_2d_to_blocked_contig
        # would zero-pad otherwise, and padded shards no longer concatenate).
        assert shard_m % 128 == 0, shard_m
        assert k % (fmt.sf_vec_size * 4) == 0, (k, fmt.sf_vec_size)
        self.sf_k = k // fmt.sf_vec_size
        self.sfa_rm = shard_m // 128
        self.sfa_rk = self.sf_k // 4
        self.sfa_shard_bytes = self.sfa_rm * self.sfa_rk * 512
        self.sfa_buffer_bytes = self.world_size * self.sfa_shard_bytes
        # Same 2-buffer parity rotation as the parent's A buffers.
        self.sfa_bufs = symm_mem.empty(
            (2, self.world_size * self.sfa_rm, self.sfa_rk, 32, 4, 4),
            dtype=torch.uint8,
            device=self.device,
        )
        self.sfa_handle = symm_mem.rendezvous(self.sfa_bufs, self.group.group_name)
        self.ev_sfa_staged = torch.cuda.Event()

    def _current_parity(self) -> int:
        return (
            self._capture_parity
            if torch.cuda.is_current_stream_capturing()
            else self.last_parity
        )

    def _sfa_shard_src(self, shard: BlockScaledOperand) -> torch.Tensor:
        """The shard's scale bytes shaped like its slot (rm, rk, 32, 4, 4).
        The operand constructor already validated the blocked atom; here we
        pin the shard-level facts: our exact (rm, rk) extents, a leading
        batch dim of 1 if present, and contiguity (the slot copy and the CE
        push address raw bytes)."""
        u8 = shard.scale.view(torch.uint8)
        if u8.ndim == 6:
            assert u8.shape[0] == 1, (
                f"leading SFA batch dim must be 1, got {tuple(u8.shape)}"
            )
            u8 = u8.squeeze(0)
        assert u8.shape[:2] == (self.sfa_rm, self.sfa_rk), (
            f"blocked SFA shard must be ({self.sfa_rm}, {self.sfa_rk}, 32, 4, 4), "
            f"got {tuple(u8.shape)}"
        )
        assert u8.is_contiguous()
        return u8

    @contextmanager
    def gather(self, shard):
        """Overlapped all-gather around any GEMM(s): ``gather(X)`` yields
        ``(X_gathered, ag_args)``.

        - ``torch.Tensor`` X: the parent's byte-only gather, verbatim.
        - :class:`BlockScaledOperand` X (qdata (shard_m, storage_k) K-major
          in the runner's format, blocked scale): the SFA lane runs under
          the same arrival flags, and X_gathered is a BlockScaledOperand
          over the two rotating symmetric buffers.

        Gathered views alias rotating buffers — valid until the next call
        on this runner; regather in backward (see the parent's docstring).
        """
        if isinstance(shard, torch.Tensor):
            with super().gather(shard) as result:
                yield result
            return
        assert isinstance(shard, BlockScaledOperand), (
            f"gather takes a torch.Tensor or BlockScaledOperand, got {type(shard)}"
        )
        assert shard.format.name == self.a_fmt.name, (
            f"operand format {shard.format.name} != runner format {self.a_fmt.name}"
        )
        assert shard.quant_dim == -1, (
            "AG shards must be K-quantized/K-major (quant_dim=-1): the gathered "
            "dim is M and scale blocks must not straddle it"
        )
        assert shard.per_tensor_scale is None, (
            "per-tensor scale is per-rank state; gathering it needs a "
            "rank-uniform contract that is not defined yet"
        )
        assert shard.qdata.shape == (self.shard_m, self.row_storage), (
            f"qdata shard must be ({self.shard_m}, {self.row_storage}) "
            f"[{self.a_fmt.name} storage extent of logical K={self.logical_k}], "
            f"got {tuple(shard.qdata.shape)}"
        )
        sfa_src = self._sfa_shard_src(shard)
        with super().gather(shard.qdata.view(torch.uint8)) as (a_full_u8, ag_args):
            parity = self._current_parity()
            sfa_buf = self.sfa_bufs[parity]
            gathered = BlockScaledOperand(
                a_full_u8.view(self.a_fmt.qdata_dtype),
                sfa_buf[None],  # (1, ws*rm, rk, 32, 4, 4); ctor views to scale dtype
                self.a_fmt,
                orig_dtype=shard.orig_dtype,
            )
            if self.world_size == 1:
                sfa_buf.copy_(sfa_src)
                yield gathered, ag_args
                return
            # Local staging on the ambient stream.
            local = sfa_buf[self.rank * self.sfa_rm : (self.rank + 1) * self.sfa_rm]
            local.copy_(sfa_src)
            self.ev_sfa_staged.record()
            # SFA peer sends on the parent's push_stream, enqueued BEFORE
            # the parent's exit enqueues [A send, flag write] per peer:
            # stream order puts every SFA memcpy before that peer's flag.
            self.push_stream.wait_event(self.ev_sfa_staged)
            self._wait_reuse(self.push_stream, parity)  # peers' SFA replicas EMPTY
            src = sfa_buf.data_ptr() + self.rank * self.sfa_shard_bytes
            for step in range(1, self.world_size):
                dst_rank = (self.rank - step) % self.world_size
                _check(
                    runtime.cudaMemcpyAsync(
                        self.sfa_handle.buffer_ptrs[dst_rank]
                        + parity * self.sfa_buffer_bytes
                        + self.rank * self.sfa_shard_bytes,
                        src,
                        self.sfa_shard_bytes,
                        runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        self.push_stream.cuda_stream,
                    )
                )
            yield gathered, ag_args
