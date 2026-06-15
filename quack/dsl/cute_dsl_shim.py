"""
Python loader for the cute-DSL CUDA 13.3 toolchain shim.

Activates the C shim built from tools/cute_dsl_shim/cute_dsl_shim.c, which
redirects CUTLASS DSL's embedded CUDA 13.1 ptxas + libnvvm to the system
CUDA 13.3 toolchain. See tools/cute_dsl_shim/README.md and DESIGN.md.

Activation order, from quack/__init__.py::
    from quack.dsl.cute_dsl_shim import try_activate
    if try_activate():
        # shim handled it; skip the Python ptxas hook
        pass

Environment variables (all optional)::

    QUACK_CUTE_DSL_SHIM
        ``0``/``off`` (default) or ``1``/``on``.
        ``1``: activate; raise if anything is missing.
        unset/``0``: do not activate (allows the legacy Python ptxas hook).

    QUACK_CUTE_DSL_SHIM_LIB
        Absolute path to libcute_dsl_shim.so. If unset, the loader searches
        <repo>/tools/cute_dsl_shim/libcute_dsl_shim.so.

    QUACK_CUTE_DSL_SHIM_PTXAS    (default /usr/local/cuda/bin/ptxas)
    QUACK_CUTE_DSL_SHIM_LIBNVVM  (default: /usr/local/cuda-13.3/nvvm/lib64,
                                  then /usr/local/cuda/nvvm/lib64)
    QUACK_CUTE_DSL_SHIM_NO_NVVM=1   only patch ptxas, keep embedded libnvvm 13.1
    QUACK_CUTE_DSL_SHIM_NO_PTXAS=1  only patch libnvvm, keep embedded ptxas 13.1
    QUACK_CUTE_DSL_SHIM_DEBUG=1     verbose stderr output from the C side
    QUACK_CUTE_DSL_SHIM_FORCE=1     skip the SHA256 wheel-validation check
        (use only if you've manually verified the wheel matches a known one)
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

__all__ = ["try_activate", "is_active", "WheelOffsets", "WHEEL_OFFSETS"]


# ---------------------------------------------------------------------------
# Per-wheel offset records. Derived statically from the
# `_cutlass_ir.cpython-*-x86_64-linux-gnu.so` shipped with
# `nvidia-cutlass-dsl-libs-cu13 == 4.5.2`.
#
# To regenerate for a new wheel, run the scripts documented in
# tools/cute_dsl_shim/recon/README.md on the extracted `_cutlass_ir.so`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WheelOffsets:
    """Virtual-address offsets within the loaded `_cutlass_ir.so`.

    All addresses are byte offsets from the relocated base of the .so. A
    value of 0 means "not yet derived" — the C side skips that intercept.
    """

    python_tag: str  # cp310, cp311, cp312, cp313, cp314, cp314t
    arch: str  # x86_64
    cutlass_dsl_version: str  # e.g. "4.5.2"
    sha256: str  # SHA256 of _cutlass_ir.so
    size: int  # file size in bytes

    # libNVVM dispatch table — patched by overwriting the table pointer
    # and setting the guard byte.
    libnvvm_guard_va: int
    libnvvm_table_va: int

    # nvPTXCompiler public-API entry points — each patched with a 12-byte
    # movabs/jmp trampoline.
    nvptx_create_va: int
    nvptx_compile_va: int
    nvptx_destroy_va: int
    nvptx_get_compiled_program_size_va: int
    nvptx_get_compiled_program_va: int
    nvptx_get_error_log_size_va: int
    nvptx_get_error_log_va: int


# Six known x86_64 wheels for nvidia-cutlass-dsl-libs-cu13 == 4.5.2.
# Each entry is keyed by the SHA256 of the corresponding `_cutlass_ir.so`.
WHEEL_OFFSETS: dict[str, WheelOffsets] = {
    # cp310
    "149b500e06aefe75aebfe2e0bb33093f5ba598a11bd1aa2b0e0cb4d5603e4b7c": WheelOffsets(
        python_tag="cp310",
        arch="x86_64",
        cutlass_dsl_version="4.5.2",
        sha256="149b500e06aefe75aebfe2e0bb33093f5ba598a11bd1aa2b0e0cb4d5603e4b7c",
        size=151_789_864,
        libnvvm_guard_va=0x092D0AA8,
        libnvvm_table_va=0x092D0AB0,
        nvptx_create_va=0x03A93AC0,
        nvptx_compile_va=0x03A93F20,
        nvptx_destroy_va=0x03A93CD0,
        nvptx_get_compiled_program_size_va=0x03A94370,
        nvptx_get_compiled_program_va=0x03A943B0,
        nvptx_get_error_log_size_va=0x03A94420,
        nvptx_get_error_log_va=0x03A94460,
    ),
    # cp311
    "2ebfa4c096eb3acfbaeb57605494d047b46b6e8fd88c610de178de4bcd0ce61b": WheelOffsets(
        python_tag="cp311",
        arch="x86_64",
        cutlass_dsl_version="4.5.2",
        sha256="2ebfa4c096eb3acfbaeb57605494d047b46b6e8fd88c610de178de4bcd0ce61b",
        size=151_789_864,
        libnvvm_guard_va=0x092D0AA8,
        libnvvm_table_va=0x092D0AB0,
        nvptx_create_va=0x03A93E00,
        nvptx_compile_va=0x03A94260,
        nvptx_destroy_va=0x03A94010,
        nvptx_get_compiled_program_size_va=0x03A946B0,
        nvptx_get_compiled_program_va=0x03A946F0,
        nvptx_get_error_log_size_va=0x03A94760,
        nvptx_get_error_log_va=0x03A947A0,
    ),
    # cp312 — the version currently used in this checkout
    "73b760621e35910305e7bdf8f4c2c0d928c10527a243f8f11a76046edba4f6d8": WheelOffsets(
        python_tag="cp312",
        arch="x86_64",
        cutlass_dsl_version="4.5.2",
        sha256="73b760621e35910305e7bdf8f4c2c0d928c10527a243f8f11a76046edba4f6d8",
        size=151_793_928,
        libnvvm_guard_va=0x092D1A88,
        libnvvm_table_va=0x092D1A90,
        nvptx_create_va=0x03A946C0,
        nvptx_compile_va=0x03A94B20,
        nvptx_destroy_va=0x03A948D0,
        nvptx_get_compiled_program_size_va=0x03A94F70,
        nvptx_get_compiled_program_va=0x03A94FB0,
        nvptx_get_error_log_size_va=0x03A95020,
        nvptx_get_error_log_va=0x03A95060,
    ),
    # cp313
    "6f0055b3c3468fb43c29afd5bb428bc8e507f4e604a82b9f49a17953915b64f6": WheelOffsets(
        python_tag="cp313",
        arch="x86_64",
        cutlass_dsl_version="4.5.2",
        sha256="6f0055b3c3468fb43c29afd5bb428bc8e507f4e604a82b9f49a17953915b64f6",
        size=151_793_928,
        libnvvm_guard_va=0x092D1A88,
        libnvvm_table_va=0x092D1A90,
        nvptx_create_va=0x03A94680,
        nvptx_compile_va=0x03A94AE0,
        nvptx_destroy_va=0x03A94890,
        nvptx_get_compiled_program_size_va=0x03A94F30,
        nvptx_get_compiled_program_va=0x03A94F70,
        nvptx_get_error_log_size_va=0x03A94FE0,
        nvptx_get_error_log_va=0x03A95020,
    ),
    # cp314
    "abd429a941a985ab510b27039e2ad48fe63d04fb766b295869b27cbed731e0fc": WheelOffsets(
        python_tag="cp314",
        arch="x86_64",
        cutlass_dsl_version="4.5.2",
        sha256="abd429a941a985ab510b27039e2ad48fe63d04fb766b295869b27cbed731e0fc",
        size=151_789_832,
        libnvvm_guard_va=0x092D0A68,
        libnvvm_table_va=0x092D0A70,
        nvptx_create_va=0x03A939C0,
        nvptx_compile_va=0x03A93E20,
        nvptx_destroy_va=0x03A93BD0,
        nvptx_get_compiled_program_size_va=0x03A94270,
        nvptx_get_compiled_program_va=0x03A942B0,
        nvptx_get_error_log_size_va=0x03A94320,
        nvptx_get_error_log_va=0x03A94360,
    ),
    # cp314t (free-threaded)
    "037e7594e2c19cb746ce6b424a325835be8c28e76ad26a4fbd4fc9d0ea1d8b3a": WheelOffsets(
        python_tag="cp314t",
        arch="x86_64",
        cutlass_dsl_version="4.5.2",
        sha256="037e7594e2c19cb746ce6b424a325835be8c28e76ad26a4fbd4fc9d0ea1d8b3a",
        size=151_798_184,
        libnvvm_guard_va=0x092D2B88,
        libnvvm_table_va=0x092D2B90,
        nvptx_create_va=0x03A95A80,
        nvptx_compile_va=0x03A95EE0,
        nvptx_destroy_va=0x03A95C90,
        nvptx_get_compiled_program_size_va=0x03A96330,
        nvptx_get_compiled_program_va=0x03A96370,
        nvptx_get_error_log_size_va=0x03A963E0,
        nvptx_get_error_log_va=0x03A96420,
    ),
}


# ---------------------------------------------------------------------------
# ctypes ABI mirror of the C structs in cute_dsl_shim.c.
# Keep ABI_VERSION and field order in sync.
# ---------------------------------------------------------------------------

ABI_VERSION = 1

FLAG_DEBUG = 1 << 0
FLAG_SKIP_NVVM = 1 << 1
FLAG_SKIP_PTXAS = 1 << 2


class CShimOffsets(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("libnvvm_guard_va", ctypes.c_size_t),
        ("libnvvm_table_va", ctypes.c_size_t),
        ("nvptx_create_va", ctypes.c_size_t),
        ("nvptx_compile_va", ctypes.c_size_t),
        ("nvptx_destroy_va", ctypes.c_size_t),
        ("nvptx_get_compiled_program_size_va", ctypes.c_size_t),
        ("nvptx_get_compiled_program_va", ctypes.c_size_t),
        ("nvptx_get_error_log_size_va", ctypes.c_size_t),
        ("nvptx_get_error_log_va", ctypes.c_size_t),
    ]


class CShimConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ptxas_path", ctypes.c_char_p),
        ("libnvvm_path", ctypes.c_char_p),
    ]


# ---------------------------------------------------------------------------
# Loader internals
# ---------------------------------------------------------------------------

# Module-level state so quack/__init__.py can query post-hoc.
_activated: bool = False
_active_offsets: WheelOffsets | None = None
_active_lib: ctypes.CDLL | None = None  # kept alive for the process


def is_active() -> bool:
    """Return True iff the C shim has been successfully installed in-process."""
    return _activated


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if not v:
        return default
    return v.strip().lower() in ("1", "on", "yes", "true")


def _mode() -> str:
    """Return 'on' only when QUACK_CUTE_DSL_SHIM explicitly enables it."""
    v = os.environ.get("QUACK_CUTE_DSL_SHIM", "").strip().lower()
    if v in ("1", "on", "yes", "true", "strict"):
        return "on"
    # Unset, empty, 0/off, and unknown values are all conservative no-ops.
    return "off"


def _find_cutlass_ir_path() -> Path | None:
    """Locate the installed `_cutlass_ir.cpython-*.so`. Returns None if not
    importable yet (caller should ensure cutlass is imported first).
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec("cutlass._mlir._mlir_libs._cutlass_ir")
    except Exception:
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin)


def _find_cutlass_ir_base() -> int | None:
    """Find the relocated base address of the mapped `_cutlass_ir.so`.

    Linux-only — reads /proc/self/maps. Returns None if not currently mapped.
    """
    try:
        maps = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return None
    needle = "_cutlass_ir.cpython-"
    for line in maps:
        if needle not in line:
            continue
        parts = line.split()
        # Format: "start-end perm offset dev inode path"
        start_end = parts[0]
        offset = int(parts[2], 16)
        start = int(start_end.split("-")[0], 16)
        # base = start of the first segment = start - file_offset
        return start - offset
    return None


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _candidate_shim_paths() -> list[Path]:
    env = os.environ.get("QUACK_CUTE_DSL_SHIM_LIB")
    if env:
        return [Path(env)]
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    repo_dev = repo_root / "tools" / "cute_dsl_shim" / "libcute_dsl_shim.so"
    return [repo_dev]


def _load_shim_lib() -> tuple[ctypes.CDLL | None, Path | None]:
    for p in _candidate_shim_paths():
        if not p.exists():
            continue
        try:
            lib = ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            warnings.warn(f"cute_dsl_shim: failed to dlopen {p}: {e}", stacklevel=2)
            continue
        # ABI sanity check
        try:
            lib.cute_dsl_shim_abi_version.restype = ctypes.c_uint32
            v = lib.cute_dsl_shim_abi_version()
            if v != ABI_VERSION:
                warnings.warn(
                    f"cute_dsl_shim: ABI mismatch at {p}: lib={v} expected={ABI_VERSION}",
                    stacklevel=2,
                )
                continue
        except AttributeError:
            warnings.warn(
                f"cute_dsl_shim: {p} missing cute_dsl_shim_abi_version; old build?",
                stacklevel=2,
            )
            continue
        return lib, p
    return None, None


def _bind_install(lib: ctypes.CDLL) -> None:
    lib.cute_dsl_shim_install.restype = ctypes.c_int
    lib.cute_dsl_shim_install.argtypes = [
        ctypes.c_size_t,
        ctypes.POINTER(CShimOffsets),
        ctypes.POINTER(CShimConfig),
    ]
    lib.cute_dsl_shim_last_error.restype = ctypes.c_char_p
    lib.cute_dsl_shim_last_error.argtypes = []
    lib.cute_dsl_shim_is_active.restype = ctypes.c_int
    lib.cute_dsl_shim_is_active.argtypes = []


class ShimError(RuntimeError):
    pass


def _build_offsets(wo: WheelOffsets) -> CShimOffsets:
    return CShimOffsets(
        abi_version=ABI_VERSION,
        libnvvm_guard_va=wo.libnvvm_guard_va,
        libnvvm_table_va=wo.libnvvm_table_va,
        nvptx_create_va=wo.nvptx_create_va,
        nvptx_compile_va=wo.nvptx_compile_va,
        nvptx_destroy_va=wo.nvptx_destroy_va,
        nvptx_get_compiled_program_size_va=wo.nvptx_get_compiled_program_size_va,
        nvptx_get_compiled_program_va=wo.nvptx_get_compiled_program_va,
        nvptx_get_error_log_size_va=wo.nvptx_get_error_log_size_va,
        nvptx_get_error_log_va=wo.nvptx_get_error_log_va,
    )


def _build_config() -> tuple[CShimConfig, list[bytes]]:
    """Build the CShimConfig + a list of byte buffers that the caller must
    keep alive for the duration of the install call.

    The libnvvm/ptxas paths are *resolved* (not just whatever the env var
    happened to say): env override > /usr/local/cuda-13.3 > /usr/local/cuda.
    This guarantees the C side actually loads the CUDA 13.3 toolchain even
    when /usr/local/cuda points elsewhere. (Without this resolution the C
    side would silently fall back to its compiled-in default.)
    """
    flags = 0
    if _env_bool("QUACK_CUTE_DSL_SHIM_DEBUG"):
        flags |= FLAG_DEBUG
    if _env_bool("QUACK_CUTE_DSL_SHIM_NO_NVVM"):
        flags |= FLAG_SKIP_NVVM
    if _env_bool("QUACK_CUTE_DSL_SHIM_NO_PTXAS"):
        flags |= FLAG_SKIP_PTXAS

    keepalive: list[bytes] = []
    ptxas_p = _resolve_ptxas_path()
    libnvvm_p = _resolve_libnvvm_path()
    ptxas_b: bytes | None = None
    libnvvm_b: bytes | None = None
    if ptxas_p is not None:
        ptxas_b = str(ptxas_p).encode()
        keepalive.append(ptxas_b)
    if libnvvm_p is not None:
        libnvvm_b = str(libnvvm_p).encode()
        keepalive.append(libnvvm_b)
    cfg = CShimConfig(
        abi_version=ABI_VERSION,
        flags=flags,
        ptxas_path=ptxas_b,
        libnvvm_path=libnvvm_b,
    )
    return cfg, keepalive


def _resolve_libnvvm_path() -> Path | None:
    """Return a usable libnvvm.so.4 path or None.

    Priority: explicit env override > /usr/local/cuda-13.3 > /usr/local/cuda.
    """
    env = os.environ.get("QUACK_CUTE_DSL_SHIM_LIBNVVM")
    if env:
        p = Path(env)
        return p if p.exists() else None
    candidates = [
        Path("/usr/local/cuda-13.3/nvvm/lib64/libnvvm.so.4"),
        Path("/usr/local/cuda/nvvm/lib64/libnvvm.so.4"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _resolve_ptxas_path() -> Path | None:
    env = os.environ.get("QUACK_CUTE_DSL_SHIM_PTXAS")
    if env:
        p = Path(env)
        return p if p.exists() else None
    p = Path("/usr/local/cuda/bin/ptxas")
    return p if p.exists() else None


def _ensure_cutlass_imported() -> bool:
    """Make sure cutlass is imported so `_cutlass_ir.so` is mapped."""
    if "cutlass" in sys.modules:
        return True
    try:
        __import__("cutlass")
        return True
    except Exception as e:
        warnings.warn(f"cute_dsl_shim: failed to import cutlass: {e}", stacklevel=2)
        return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def try_activate() -> bool:
    """Activate the shim. Returns True iff it was successfully installed.

    Behavior summary by `QUACK_CUTE_DSL_SHIM`:
      - unset/off: always return False, no-op.
      - on:        activate, raising ShimError on any failure.
    """
    global _activated, _active_offsets, _active_lib

    if _activated:
        return True

    mode = _mode()
    if mode == "off":
        return False

    strict = mode == "on"

    def _miss(msg: str) -> bool:
        if strict:
            raise ShimError(msg)
        if _env_bool("QUACK_CUTE_DSL_SHIM_DEBUG"):
            sys.stderr.write(f"[cute_dsl_shim] skip: {msg}\n")
        return False

    # 0. Linux + x86_64 only for v1. (aarch64 deferred.)
    if sys.platform != "linux":
        return _miss(f"unsupported platform {sys.platform}")
    import platform as _pf

    machine = _pf.machine()
    if machine not in ("x86_64", "AMD64"):
        return _miss(f"unsupported machine {machine}; v1 is x86_64-only")

    # 1. Cheap up-front check: do we even have the shim .so to load?
    lib, lib_path = _load_shim_lib()
    if lib is None:
        return _miss(
            "libcute_dsl_shim.so not found; build it with "
            "`make -C tools/cute_dsl_shim`, or set QUACK_CUTE_DSL_SHIM_LIB"
        )

    # 2. Make sure _cutlass_ir.so is on disk and mapped.
    if not _ensure_cutlass_imported():
        return _miss("cutlass not importable")
    so_path = _find_cutlass_ir_path()
    if so_path is None or not so_path.exists():
        return _miss("could not locate _cutlass_ir.so")
    base = _find_cutlass_ir_base()
    if base is None:
        return _miss("_cutlass_ir.so not in /proc/self/maps after import")

    # 3. Validate wheel by SHA256 unless explicitly forced.
    sha = _sha256_of(so_path)
    wo = WHEEL_OFFSETS.get(sha)
    if wo is None:
        if _env_bool("QUACK_CUTE_DSL_SHIM_FORCE"):
            warnings.warn(
                f"cute_dsl_shim: unknown wheel sha256={sha}; QUACK_CUTE_DSL_SHIM_FORCE "
                "is set, attempting cp312 offsets — this WILL CORRUPT the process "
                "if the wheel does not match cp312 x86_64 4.5.2",
                stacklevel=2,
            )
            wo = WHEEL_OFFSETS["73b760621e35910305e7bdf8f4c2c0d928c10527a243f8f11a76046edba4f6d8"]
        else:
            return _miss(
                f"unknown _cutlass_ir.so sha256={sha} (path={so_path}); "
                f"known wheels: cp310/11/12/13/14/14t x86_64 of "
                f"nvidia-cutlass-dsl-libs-cu13 4.5.2. Set "
                f"QUACK_CUTE_DSL_SHIM_FORCE=1 to bypass (dangerous)."
            )

    # 4. Verify the system 13.3 toolchain is reachable, unless we're skipping
    #    those pieces.
    no_nvvm = _env_bool("QUACK_CUTE_DSL_SHIM_NO_NVVM")
    no_ptxas = _env_bool("QUACK_CUTE_DSL_SHIM_NO_PTXAS")
    if not no_nvvm and _resolve_libnvvm_path() is None:
        return _miss(
            "libnvvm.so.4 not found at /usr/local/cuda-13.3/nvvm/lib64 or "
            "/usr/local/cuda/nvvm/lib64; set QUACK_CUTE_DSL_SHIM_LIBNVVM or "
            "QUACK_CUTE_DSL_SHIM_NO_NVVM=1"
        )
    if not no_ptxas and _resolve_ptxas_path() is None:
        return _miss(
            "ptxas not found at /usr/local/cuda/bin/ptxas; set "
            "QUACK_CUTE_DSL_SHIM_PTXAS or QUACK_CUTE_DSL_SHIM_NO_PTXAS=1"
        )

    # 5. Bind C entry points.
    _bind_install(lib)

    # 6. Install. The C side does the actual mprotect/patch work.
    off = _build_offsets(wo)
    cfg, _keepalive = _build_config()
    rc = lib.cute_dsl_shim_install(
        ctypes.c_size_t(base),
        ctypes.byref(off),
        ctypes.byref(cfg),
    )
    # _keepalive holds the encoded bytes for ptxas_path/libnvvm_path; the C
    # side only dereferences them during the install call, so once rc is
    # returned the buffers can be GC'd.
    if rc < 0:
        err = (lib.cute_dsl_shim_last_error() or b"").decode("utf-8", errors="replace")
        return _miss(f"shim install failed (rc={rc}): {err}")

    # rc == 0 (OK) or 1 (ALREADY) both count as active.
    _activated = True
    _active_offsets = wo
    _active_lib = lib  # keep alive for the process
    if _env_bool("QUACK_CUTE_DSL_SHIM_DEBUG"):
        sys.stderr.write(
            f"[cute_dsl_shim] activated: wheel={wo.python_tag}/{wo.arch} "
            f"sha={wo.sha256[:16]} lib={lib_path}\n"
        )
    return True
