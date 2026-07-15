"""Host-side distributed runtime helpers for the fused reduce-scatter / all-reduce
GEMM path (use_epi_reduce): nvshmem + torch.distributed setup, symmetric / multicast
tensor allocation, and the barrier flags the kernel's comm layer reads. These are the
runtime contract for calling GemmSm100(use_epi_reduce=...), not benchmarking helpers.
"""

import os

import numpy as np
import torch
import torch.distributed as dist

import nvshmem.core
from cuda.core.experimental import Device

import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack


def torchrun_init_nvshmem(dist_initialized=False):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = Device(local_rank)
    dev.set_current()
    if not dist_initialized:
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    num_ranks = dist.get_world_size()
    uid = nvshmem.core.get_unique_id(empty=(local_rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)
    nvshmem.core.init(
        device=dev, uid=uid, rank=local_rank, nranks=num_ranks, initializer_method="uid"
    )


# Cleanup callbacks run (in reverse) by torchrun_finalize_nvshmem, so factories that allocate
# symmetric memory internally (barrier flags, multicast views) don't leak into finalize errors.
_finalizers = []


def on_finalize(fn):
    _finalizers.append(fn)


def torchrun_finalize_nvshmem():
    for fn in reversed(_finalizers):
        try:
            fn()
        except Exception:
            pass  # tolerate already-freed tensors
    _finalizers.clear()
    nvshmem.core.finalize()
    dist.destroy_process_group()


def create_multicast_tensor(torch_tensor_cpu, dtype, leading_dim, is_dynamic_layout=True):
    """Copy a CPU tensor into fresh symmetric memory; return cute + torch handles.

    Allocates in the source's natural (descending-stride) order and permutes back, so
    the symmetric tensor matches the source's strides for any rank / major-ness. Returns
    (cute, cute_mc, torch_gpu, torch_gpu_mc, peer_torch, cute_peers); the symmetric
    allocation self-registers its free via on_finalize."""
    ndim = torch_tensor_cpu.dim()
    base_order = sorted(range(ndim), key=lambda d: -torch_tensor_cpu.stride(d))
    inv = [base_order.index(d) for d in range(ndim)]
    base = nvshmem.core.tensor(
        tuple(torch_tensor_cpu.shape[d] for d in base_order), dtype=torch_tensor_cpu.dtype
    )
    torch_gpu = base.permute(inv)
    torch_gpu.copy_(torch_tensor_cpu)
    base_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, base)
    torch_gpu_mc = base_mc.permute(inv)
    peer_torch = [
        nvshmem.core.get_peer_tensor(base, r).permute(inv) for r in range(dist.get_world_size())
    ]
    cute_peers = [from_dlpack(t) for t in peer_torch]
    tensor_mc = from_dlpack(torch_gpu_mc, assumed_align=16)
    tensor = from_dlpack(torch_gpu, assumed_align=16)
    tensor.element_type = dtype
    if is_dynamic_layout:
        tensor_mc = tensor_mc.mark_layout_dynamic(leading_dim=leading_dim)
        tensor = tensor.mark_layout_dynamic(leading_dim=leading_dim)
    tensor = cutlass_torch.convert_cute_tensor(
        torch_gpu, tensor, dtype, is_dynamic_layout=is_dynamic_layout
    )
    on_finalize(lambda: (nvshmem.core.free_tensor(base_mc), nvshmem.core.free_tensor(base)))
    return tensor, tensor_mc, torch_gpu, torch_gpu_mc, peer_torch, cute_peers


def make_barrier_flags(num_flags):
    bf_torch = nvshmem.core.tensor((num_flags,), dtype=torch.int32)
    bf_torch.fill_(0)
    bf_torch_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, bf_torch)
    bf = from_dlpack(bf_torch).mark_layout_dynamic()
    bf_mc = from_dlpack(bf_torch_mc).mark_layout_dynamic()
    on_finalize(lambda: (nvshmem.core.free_tensor(bf_torch_mc), nvshmem.core.free_tensor(bf_torch)))
    return bf_torch, bf_torch_mc, bf, bf_mc
