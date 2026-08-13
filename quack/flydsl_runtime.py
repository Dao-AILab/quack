# Copyright (c) 2026, Tri Dao.

"""Generic host plumbing shared by every FlyDSL kernel in quack.

Nothing here knows about a particular kernel: architecture validation, launch
dispatch, stream and dtype translation, and the row layout rules the ROCm
buffer descriptors depend on.

Caching is split, and a kernel module needs to know which half it owns.
``run_compiled`` holds one compiled artifact per launcher and does no keying of
its own, so **the caller must key its launchers by device** as well as by
specialization -- rmsnorm passes the device ordinal to its ``functools.cache``d
builder for exactly that reason. FlyDSL caches artifacts on the JitFunction by
argument signature alone, so a launcher shared across GPUs hands the second one
a module loaded into the first one's context and the launch fails with
hipErrorInvalidDevice. Neither layer is serialized against concurrent builders,
and FlyDSL's own on-disk cache under ``~/.flydsl/cache`` sits below both.

This is a sibling of :mod:`quack.cache.jit` rather than a reuse of it: that
module's disk half is CuTe-specific (``.o`` export plus a tvm_ffi
``load_module``) and it imports cutlass, so it will not load on ROCm.
"""

import torch

from quack._platform import IS_ROCM_BUILD
from quack.flydsl_constants import MAX_ACCESS_BITS

if not IS_ROCM_BUILD:
    raise ImportError("quack.flydsl_runtime requires a ROCm PyTorch build")

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch

__all__ = [
    "MAX_ACCESS_BITS",
    "SUPPORTED_DTYPES",
    "current_raw_stream",
    "dtype_spec",
    "empty_placeholder",
    "packed_rows",
    "run_compiled",
    "validate_arch",
]

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_DTYPE_SPECS = {
    torch.float16: (fx.Float16, 16),
    torch.bfloat16: (fx.BFloat16, 16),
    torch.float32: (fx.Float32, 32),
}

_EMPTY_CACHE: dict[tuple[int, torch.dtype], torch.Tensor] = {}


def dtype_spec(dtype: torch.dtype):
    """Translate a public torch dtype to its FlyDSL scalar type and bit width."""
    try:
        return _DTYPE_SPECS[dtype]
    except KeyError:
        raise TypeError(f"unsupported dtype: {dtype}") from None


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def validate_arch(device: torch.device, supported: frozenset, kernel: str) -> str:
    """Reject a device, a compile target or a runtime that disagree on the arch.

    Only gcnArchName is normalized. ARCH is a raw passthrough into FlyDSL's
    compile target and reaches the ROCDL pipeline as ``chip``, so a value the
    device never reports has to fail here rather than be massaged into one.

    Nothing is memoized across calls. ``run_compiled`` only reaches here on a
    launcher's cold path, and torch already caches the properties object, so a
    per-device arch cache would buy one string split -- at the price of every
    caller after the first inheriting the first one's ``supported`` set.
    """
    index = _device_index(device)
    # ROCm appends feature flags: "gfx950:sramecc+:xnack-".
    actual = torch.cuda.get_device_properties(index).gcnArchName.split(":", 1)[0]
    if actual not in supported:
        names = ", ".join(sorted(supported))
        raise ValueError(f"FlyDSL {kernel} supports {names}; cuda:{index} is {actual}")

    target = flyc.get_backend().target
    if target.backend != "rocm":
        raise RuntimeError(
            f"FlyDSL {kernel} requires FlyDSL's ROCm backend, but it targets {target.backend!r}"
        )
    if target.arch != actual:
        raise ValueError(
            f"cuda:{index} is {actual}, but FlyDSL compiles for {target.arch}; "
            "set ARCH and FLYDSL_GPU_ARCH to the device architecture"
        )
    runtime_arch = get_rocm_arch()
    if runtime_arch != actual:
        raise ValueError(
            f"cuda:{index} is {actual}, but FlyDSL runtime helpers use {runtime_arch}; "
            "set FLYDSL_GPU_ARCH or HSA_OVERRIDE_GFX_VERSION to the device architecture"
        )
    return actual


# torch.cuda.current_stream() allocates a Stream wrapper per call, ~1.6us of a
# ~21us host path. Inductor's generated code binds the same raw accessor.
_get_raw_stream = torch._C._cuda_getCurrentRawStream


def current_raw_stream(device: torch.device) -> int:
    return _get_raw_stream(_device_index(device))


def empty_placeholder(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A shared zero-element tensor for arguments a specialization ignores.

    Keyed by ordinal, not by the torch.device object: callers pass whatever
    ``x.device`` gave them, and "cuda" and "cuda:0" hash apart while naming the
    same GPU.
    """
    key = (_device_index(device), dtype)
    tensor = _EMPTY_CACHE.get(key)
    if tensor is None:
        tensor = torch.empty(0, device=device, dtype=dtype)
        _EMPTY_CACHE[key] = tensor
    return tensor


def _rows_are_disjoint_and_packed(tensor: torch.Tensor) -> bool:
    if tensor.stride(-1) != 1:
        return False
    access_bytes = MAX_ACCESS_BITS // 8
    if tensor.data_ptr() % access_bytes:
        return False
    span = tensor.shape[-1]
    for stride, size in sorted(
        zip(tensor.stride()[:-1], tensor.shape[:-1]), key=lambda axis: axis[0]
    ):
        if size > 1 and (stride * tensor.element_size()) % access_bytes:
            return False
        if stride < span:
            return False
        span = stride * (size - 1) + span
    return True


def packed_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Preserve safe row padding and copy only incompatible layouts."""
    if _rows_are_disjoint_and_packed(tensor):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def run_compiled(exe, device: torch.device, args: tuple, *, supported: frozenset, kernel: str):
    """Dispatch a launcher, compiling it on this process's first use.

    Follows aiter's ``_run_compiled``: the ``CompiledFunction`` is stashed on
    the launcher, and ``flyc.compile`` both compiles and performs the first
    launch, so the miss branch must not call the result again.

    One artifact per launcher is only correct because the caller keys its
    launchers by device as well as by specialization; see this module's
    docstring for what goes wrong when it does not.

    One consequence of the shape is deliberate: nothing serializes a cold key,
    so concurrent threads can each build it. FlyDSL's per-key file lock still
    makes only one of them run the MLIR pipeline, and either artifact is valid.
    A mid-process change to ARCH or FLYDSL_GPU_ARCH is likewise not detected
    once an artifact exists; validate_arch runs per launcher on the miss.
    """
    with torch.cuda.device(device):
        cf = getattr(exe, "_cf", None)
        if cf is not None:
            cf(*args)
            return
        validate_arch(device, supported, kernel)
        exe._cf = flyc.compile(exe, *args)
