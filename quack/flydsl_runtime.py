# Copyright (c) 2026, Tri Dao.

"""Generic host plumbing shared by every FlyDSL kernel in quack.

Nothing here knows about a particular kernel: architecture validation, the
specialization cache and its lock, stream and dtype translation, and the row
layout rules the ROCm buffer descriptors depend on. A kernel module supplies
its own cache key type and its own builder.

This is the FlyDSL sibling of :mod:`quack.cache.jit`, which cannot be reused:
that module's disk half is CuTe-specific (``.o`` export plus a tvm_ffi
``load_module``) and it imports cutlass, so it will not load on ROCm. FlyDSL
already persists compiled artifacts itself under ``~/.flydsl/cache``, so the
only cache this module owns is the in-process one that turns a ~33 ms disk-cache
hydration into a ~20 us call.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch

__all__ = [
    "ACCESS_BITS",
    "SUPPORTED_DTYPES",
    "current_raw_stream",
    "dtype_spec",
    "empty_placeholder",
    "packed_rows",
    "rows_are_disjoint_and_packed",
    "run_compiled",
    "unambiguous_layout",
    "validate_arch",
]

# Widest MUBUF transaction, and therefore the alignment every vectorized row
# access is built around.
ACCESS_BITS = 128

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_DTYPE_SPECS = {
    torch.float16: (fx.Float16, 16),
    torch.bfloat16: (fx.BFloat16, 16),
    torch.float32: (fx.Float32, 32),
}

_DEVICE_ARCH_CACHE: dict[int, str] = {}
_EMPTY_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def dtype_spec(dtype: torch.dtype):
    """Translate a public torch dtype to its FlyDSL scalar type and bit width."""
    try:
        return _DTYPE_SPECS[dtype]
    except KeyError:
        raise TypeError(f"unsupported dtype: {dtype}") from None


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _normalize_arch(arch: str) -> str:
    arch = str(arch).split(":", 1)[0]
    if arch.startswith("gfx"):
        return arch
    parts = arch.split(".")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        # HSA_OVERRIDE_GFX_VERSION spells the stepping as one hex digit, so
        # gfx90a arrives as 9.0.10 and a decimal join would yield gfx9010.
        return f"gfx{parts[0]}{parts[1]}{int(parts[2]):x}"
    return arch


def validate_arch(device: torch.device, supported: frozenset, kernel: str) -> str:
    """Reject a device, a compile target or a runtime that disagree on the arch."""
    index = _device_index(device)
    actual = _DEVICE_ARCH_CACHE.get(index)
    if actual is None:
        actual = _normalize_arch(torch.cuda.get_device_properties(index).gcnArchName)
        if actual not in supported:
            names = ", ".join(sorted(supported))
            raise ValueError(f"FlyDSL {kernel} supports {names}; cuda:{index} is {actual}")
        _DEVICE_ARCH_CACHE[index] = actual

    target = flyc.get_backend().target
    compile_arch = _normalize_arch(target.arch)
    if target.backend != "rocm":
        raise RuntimeError(
            f"FlyDSL {kernel} requires FlyDSL's ROCm backend, but it targets {target.backend!r}"
        )
    if compile_arch != actual:
        raise ValueError(
            f"cuda:{index} is {actual}, but FlyDSL compiles for {compile_arch}; "
            "set ARCH and FLYDSL_GPU_ARCH to the device architecture"
        )
    runtime_arch = _normalize_arch(get_rocm_arch())
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
    """A shared zero-element tensor for arguments a specialization ignores."""
    key = (device, dtype)
    tensor = _EMPTY_CACHE.get(key)
    if tensor is None:
        tensor = torch.empty(0, device=device, dtype=dtype)
        _EMPTY_CACHE[key] = tensor
    return tensor


def unambiguous_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure FlyDSL identifies the last dimension as the unit-stride row."""
    row = tensor.dim() - 1
    offenders = [
        axis for axis in range(row) if tensor.shape[axis] == 1 and tensor.stride(axis) == 1
    ]
    if not offenders:
        return tensor
    for axis in reversed(offenders):
        tensor = tensor.squeeze(axis)
    for axis in offenders:
        tensor = tensor.unsqueeze(axis)
    return tensor


def rows_are_disjoint_and_packed(tensor: torch.Tensor) -> bool:
    if tensor.stride(-1) != 1:
        return False
    access_bytes = ACCESS_BITS // 8
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
    tensor = unambiguous_layout(tensor)
    if rows_are_disjoint_and_packed(tensor):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def run_compiled(exe, device: torch.device, args: tuple, *, supported: frozenset, kernel: str):
    """Dispatch a launcher, compiling it on this process's first use.

    Follows aiter's ``_run_compiled``: the ``CompiledFunction`` is stashed on
    the launcher object, which a ``@functools.cache`` over the builder has
    already keyed by specialization, so there is no second dictionary here.
    ``flyc.compile`` both compiles and performs the first launch, so the miss
    branch must not call the result again.

    Two consequences of that shape are deliberate. Nothing serializes a cold
    key, so concurrent threads can each build it -- FlyDSL's per-key file lock
    still makes only one of them run the MLIR pipeline, and either artifact is
    valid. And the compiled kernel is keyed by shape and dtype alone, so it is
    reused across devices and across a mid-process change to ARCH or
    FLYDSL_GPU_ARCH; validate_arch runs on the miss and would reject the first
    such build, not a later reuse.
    """
    with torch.cuda.device(device):
        cf = getattr(exe, "_cf", None)
        if cf is not None:
            cf(*args)
            return
        validate_arch(device, supported, kernel)
        exe._cf = flyc.compile(exe, *args)
