# Copyright (c) 2026, Tri Dao.

"""Generic host plumbing shared by every FlyDSL kernel: arch validation, launch
dispatch, stream and dtype translation, ROCm buffer-descriptor row rules.

A ``Launcher`` holds one artifact and keys nothing itself, so **the caller must
key its launchers by device** as well as by specialization. FlyDSL keys
artifacts on the JitFunction by argument signature alone, so a launcher shared
across GPUs hands the second one a module loaded into the first one's context:
hipErrorInvalidDevice. FlyDSL's own on-disk cache under ``~/.flydsl/cache``
sits below both.

A sibling of :mod:`quack.cache.jit`, not a reuse: that module's disk half is
CuTe-specific (``.o`` export plus tvm_ffi ``load_module``) and imports cutlass,
so it will not load on ROCm.
"""

import os

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
    "Launcher",
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

    Only gcnArchName is normalized: ARCH passes through to FlyDSL's compile
    target and reaches the ROCDL pipeline as ``chip``, so a value the device
    never reports has to fail here rather than be massaged into one.
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

    Keyed by ordinal, not torch.device: "cuda" and "cuda:0" hash apart.
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


class Launcher:
    """A FlyDSL entry point plus the ``CompiledFunction`` its first launch builds.

    The artifact is held here rather than attached to the ``JitFunction``: that
    object's underscore namespace belongs to FlyDSL (api_stability.md §2), it has
    no ``__slots__`` to catch a collision, and FlyDSL's own ``_mem_cache`` already
    keys artifacts by argument signature. ``__slots__`` also makes the hot-path
    load cheaper than the ``getattr`` default it replaces.
    """

    __slots__ = ("cf", "jit_fn")

    def __init__(self, jit_fn):
        self.jit_fn = jit_fn
        self.cf = None


def run_compiled(
    launcher: Launcher, device: torch.device, args: tuple, *, supported: frozenset, kernel: str
):
    """Dispatch a launcher, compiling it on this process's first use.

    Follows aiter's ``_run_compiled``: the ``CompiledFunction`` is stashed on the
    launcher, and ``flyc.compile`` both compiles and performs the first launch,
    so the miss branch must not call the result again.

    Nothing serializes a cold key; FlyDSL's per-key file lock still runs the MLIR
    pipeline once and either artifact is valid. validate_arch runs on the miss
    only, so a mid-process ARCH change goes undetected once an artifact exists.
    """
    with torch.cuda.device(device):
        if launcher.cf is not None:
            launcher.cf(*args)
            return
        # Quack has no compile-only API. FlyDSL checks this generic environment
        # variable only on its JitFunction path, where it would suppress the
        # mandatory first launch; compiled-callable hits do not consult it.
        if os.environ.get("COMPILE_ONLY", "").lower() in ("1", "true", "yes", "on"):
            raise RuntimeError(f"COMPILE_ONLY is unsupported by eager FlyDSL {kernel}")
        validate_arch(device, supported, kernel)
        launcher.cf = flyc.compile(launcher.jit_fn, *args)
