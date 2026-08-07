"""Distributed helpers for the fused reduce-scatter / all-reduce GEMM path
(epi_reduce_mode): torch.distributed setup, symmetric / multicast tensor
allocation via torch symm_mem, the barrier flags the kernel's comm layer reads,
and the multimem intrinsic dispatch its reduce uses. These are the runtime
contract for calling GemmSm100(epi_reduce_mode=...), not benchmarking helpers.
"""

import math
import os

import torch
import torch.distributed as dist

import cutlass
import cutlass.utils as utils


def multimem_ld_reduce_128b(dtype):
    """128-bit multimem.ld_reduce(add) variant for dtype; all return 4x b32 (x, y, z, w)."""
    if dtype == cutlass.Float16:
        return utils.distributed.multimem_ld_reduce_8xf16
    if dtype == cutlass.Float32:
        return utils.distributed.multimem_ld_reduce_4xf32
    if dtype == cutlass.BFloat16:
        return utils.distributed.multimem_ld_reduce_8xbf16
    if dtype == cutlass.Float8E4M3FN:
        return utils.distributed.multimem_ld_reduce_16xe4m3
    if dtype == cutlass.Float8E5M2:
        return utils.distributed.multimem_ld_reduce_16xe5m2
    raise NotImplementedError(f"multimem_ld_reduce_128b: unsupported dtype {dtype}")


def init_distributed():
    """torchrun NCCL setup (idempotent), matching the cutlass distributed
    examples' init_distributed. Returns (global_rank, world_size, device)."""
    if not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl", world_size=world_size, rank=global_rank, device_id=device
        )
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", global_rank))
    device = torch.device("cuda", local_rank)
    return global_rank, world_size, device


def clean_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _wrap_device_ptr(ptr, shape, torch_dtype):
    """CUDA-typed non-owning view of a raw device pointer (e.g. a symm_mem
    multicast VA) on the current device; the symm_mem handle owns the memory."""
    dev = torch.device("cuda", torch.cuda.current_device())
    nbytes = math.prod(shape) * torch_dtype.itemsize
    storage = torch._C._construct_storage_from_data_pointer(ptr, dev, nbytes)
    return torch.empty(0, dtype=torch_dtype, device=dev).set_(storage).view(shape)


def make_symm_mem_tensor(shape, torch_dtype, permute):
    """Uninitialized symmetric tensor plus its multicast view via torch symm_mem.

    No host staging — an output or workspace buffer needs no initial contents.
    shape is in allocation (row-major) order and permute applies to both returns,
    so shape=(l, m, n) with permute=(1, 2, 0) gives kernel-order (m, n, l)
    n-major views. torch owns the allocation — no explicit free."""
    import torch.distributed._symmetric_memory as symm_mem

    base = symm_mem.empty(tuple(shape), dtype=torch_dtype, device="cuda")
    hdl = symm_mem.rendezvous(base, group=dist.group.WORLD)
    mc = _wrap_device_ptr(hdl.multicast_ptr, tuple(base.shape), base.dtype)
    return base.permute(permute), mc.permute(permute)


def make_symm_mem_flags(num_flags):
    """Zero-filled symmetric int32 flag array + multicast view via torch symm_mem.
    Each rank zero-fills its own copy before the (collective) rendezvous."""
    import torch.distributed._symmetric_memory as symm_mem

    flags = symm_mem.empty((num_flags,), dtype=torch.int32, device="cuda")
    flags.fill_(0)
    hdl = symm_mem.rendezvous(flags, group=dist.group.WORLD)
    flags_mc = _wrap_device_ptr(hdl.multicast_ptr, (num_flags,), flags.dtype)
    return flags, flags_mc
