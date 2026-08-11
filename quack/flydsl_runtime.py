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

import os
import threading

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch

__all__ = [
    "ACCESS_BITS",
    "SUPPORTED_DTYPES",
    "SpecializationCache",
    "compile_context",
    "current_raw_stream",
    "dtype_spec",
    "empty_placeholder",
    "packed_rows",
    "rows_are_disjoint_and_packed",
    "unambiguous_layout",
    "validate_arch",
]

# Widest MUBUF transaction, and therefore the alignment every vectorized row
# access is built around.
ACCESS_BITS = 128

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Anything that can change what FlyDSL compiles for must invalidate the
# specialization cache, since the jit cache key cannot see these.
COMPILE_ENV_VARS = (
    "FLYDSL_COMPILE_BACKEND",
    "ARCH",
    "FLYDSL_GPU_ARCH",
    "HSA_OVERRIDE_GFX_VERSION",
)

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
        return f"gfx{parts[0]}{parts[1]}{parts[2]}"
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


def compile_context(device: torch.device) -> tuple:
    """The part of a cache key that is about the target, not the arguments."""
    return (
        _device_index(device),
        *(os.environ.get(name, "") for name in COMPILE_ENV_VARS),
    )


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


class SpecializationCache:
    """One compiled FlyDSL launcher per specialization key, for this process.

    Reads take no lock; a miss serializes on the build lock and re-checks, so
    concurrent callers on a new key compile once rather than racing. FlyDSL's
    own on-disk cache sits underneath and survives the process.
    """

    __slots__ = ("_entries", "_kernel", "_lock", "_supported")

    def __init__(self, kernel: str, supported: frozenset):
        self._entries: dict = {}
        self._lock = threading.RLock()
        self._kernel = kernel
        self._supported = supported

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key) -> bool:
        return key in self._entries

    def clear(self) -> None:
        self._entries.clear()

    def launch(self, key, device: torch.device, build, args: tuple) -> None:
        """Run this specialization, compiling it first if this process has not.

        ``build(key)`` is called only on a miss and must return the ``@flyc.jit``
        launcher for ``key``; it is taken as a plain function rather than a
        closure so a warm call allocates nothing. Architecture validation is
        deferred to the same miss, so a hit pays only the dict lookup.
        """
        with torch.cuda.device(device):
            compiled = self._entries.get(key)
            if compiled is not None:
                compiled(*args)
                return
            with self._lock:
                compiled = self._entries.get(key)
                if compiled is None:
                    validate_arch(device, self._supported, self._kernel)
                    # flyc.compile returns a CompiledFunction and performs its
                    # first launch, so a fresh compile must not be called again.
                    self._entries[key] = flyc.compile(build(key), *args)
                    return
            compiled(*args)
