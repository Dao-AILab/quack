"""
System ptxas replacement for CUTLASS DSL.

Usage::

    CUTE_DSL_KEEP_PTX=1 CUTE_DSL_PTXAS_PATH=/usr/local/cuda/bin/ptxas pytest tests/

Environment variables:
    CUTE_DSL_PTXAS_PATH    - Path to ptxas (e.g., /usr/local/cuda/bin/ptxas)
    CUTE_DSL_KEEP_PTX      - Must be set to 1 before cutlass is imported
    CUTE_DSL_PTXAS_VERBOSE - Set to 1 for verbose output
    CUTE_DSL_DUMP_DIR      - Directory for dumped PTX files (default: cwd)
    CUTE_DSL_KEEP_CUBIN    - Set to 1 to save compiled cubin files
"""

import os
import sys
import re
import ctypes
import subprocess
from pathlib import Path

import cutlass


CUTE_DSL_PTXAS_PATH = os.environ.get("CUTE_DSL_PTXAS_PATH", None)

if CUTE_DSL_PTXAS_PATH:
    os.environ["CUTE_DSL_KEEP_PTX"] = "1"
VERBOSE = os.environ.get("CUTE_DSL_PTXAS_VERBOSE", "0") == "1"

_original_load_cuda_library = None
_original_create_tvm_ffi_function = None
_user_wanted_ptx = False  # True if user originally set CUTE_DSL_KEEP_PTX=1


def _log(msg: str):
    if VERBOSE:
        print(f"[ptxas] {msg}", file=sys.stderr)


def _read_ptx(ptx_path: Path) -> str | None:
    try:
        return ptx_path.read_bytes().decode("utf-8", errors="ignore").rstrip("\x00")
    except OSError as exc:
        _log(f"Failed to read {ptx_path}: {exc}")
        return None


def _read_complete_ptx(ptx_path: Path) -> str | None:
    content = _read_ptx(ptx_path)
    if content is None or not content.rstrip().endswith("}"):
        return None
    return content


def _get_ptx(compiled_func) -> tuple[str, Path] | None:
    """Find dumped PTX for the compiled function."""
    func_name = getattr(compiled_func, "function_name", None)
    if not func_name:
        _log("Compiled function is missing function_name")
        return None

    dump_dir = Path(os.environ.get("CUTE_DSL_DUMP_DIR", Path.cwd()))
    dump_dir.mkdir(parents=True, exist_ok=True)

    ptx_paths = sorted(
        dump_dir.rglob("*.ptx"), key=lambda path: path.stat().st_mtime_ns, reverse=True
    )
    _log(f"Searching dumped PTX for {func_name} in {dump_dir}")
    _log(f"Found {len(ptx_paths)} PTX candidate files in {dump_dir}")

    # Strategy 1: match by filename
    filename_matches = [ptx_path for ptx_path in ptx_paths if func_name in ptx_path.name]
    if filename_matches:
        _log(f"Found {len(filename_matches)} filename matches for {func_name}")
        for ptx_path in filename_matches:
            content = _read_complete_ptx(ptx_path)
            if content is None:
                continue
            _log(f"Using PTX filename match for {func_name}: {ptx_path}")
            return content, ptx_path

    # Strategy 2: match by .entry directive inside PTX
    entry_pattern = re.compile(rf"\.entry\s+{re.escape(func_name)}(?:\s|\()", re.MULTILINE)
    for ptx_path in ptx_paths:
        content = _read_complete_ptx(ptx_path)
        if content is None:
            continue
        if entry_pattern.search(content):
            _log(f"Found PTX for {func_name}: {ptx_path}")
            return content, ptx_path

    # Strategy 3: use sole candidate as fallback
    if len(ptx_paths) == 1:
        content = _read_complete_ptx(ptx_paths[0])
        if content is not None:
            _log(f"Using sole PTX candidate for {func_name}: {ptx_paths[0]}")
            return content, ptx_paths[0]

    _log(f"No PTX found for function {func_name} in {dump_dir}")
    return None


def _compile_ptx(ptx_path: Path, ptx_content: str) -> bytes:
    """Compile PTX to cubin using system ptxas."""
    # Extract arch from PTX
    match = re.search(r"\.target\s+(sm_\d+[a-z]?)", ptx_content)
    arch = match.group(1) if match else "sm_90a"

    # Write stripped content back if needed
    if ptx_path.read_text() != ptx_content:
        ptx_path.write_text(ptx_content)

    # Compile
    cubin_tmp = ptx_path.with_suffix(".cubin.tmp")
    try:
        assert CUTE_DSL_PTXAS_PATH is not None
        result = subprocess.run(
            [CUTE_DSL_PTXAS_PATH, f"-arch={arch}", "-O3", "-o", str(cubin_tmp), str(ptx_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ptxas failed: {result.stderr}")

        cubin_data = cubin_tmp.read_bytes()
        _log(f"Compiled {ptx_path.name} -> {len(cubin_data)} bytes ({arch})")

        # Save cubin if CUTE_DSL_KEEP_CUBIN is set
        if os.environ.get("CUTE_DSL_KEEP_CUBIN", "0") == "1":
            cubin_out = ptx_path.with_suffix(".cubin")
            cubin_out.write_bytes(cubin_data)
            _log(f"Saved: {cubin_out}")

        return cubin_data
    finally:
        cubin_tmp.unlink(missing_ok=True)


def _patched_load_cuda_library(self):
    """Replacement for _load_cuda_library that uses system ptxas."""

    result = _get_ptx(self)
    if not result:
        _log("PTX not found, falling back to embedded ptxas")
        return _original_load_cuda_library(self)

    ptx_content, ptx_path = result

    try:
        cubin = _compile_ptx(ptx_path, ptx_content)
    except Exception as e:
        _log(f"Compilation failed ({e}), falling back to embedded ptxas")
        return _original_load_cuda_library(self)

    # Load cubin
    import cuda.bindings.runtime as cuda_runtime

    err, library = cuda_runtime.cudaLibraryLoadData(cubin, None, None, 0, None, None, 0)
    if err != cuda_runtime.cudaError_t.cudaSuccess:
        _log(f"cudaLibraryLoadData failed ({err}), falling back to embedded ptxas")
        return _original_load_cuda_library(self)

    # Register kernels on all devices (must match cuda_load_to_device's void*** convention)
    _, cuda_load_to_device = self._get_cuda_init_and_load()
    lib_handle = ctypes.c_void_p(int(library))
    ptr_to_lib = ctypes.pointer(lib_handle)
    ptr_to_ptr_to_lib = ctypes.pointer(ptr_to_lib)
    dev_id = ctypes.c_int32(0)
    err_val = ctypes.c_int32(0)
    args = (ctypes.c_void_p * 3)(
        ctypes.cast(ptr_to_ptr_to_lib, ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(dev_id), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(err_val), ctypes.c_void_p),
    )

    for dev in range(self.num_devices):
        dev_id.value = dev
        cuda_load_to_device(args)
        if err_val.value != 0:
            _log("cuda_load_to_device failed, falling back to embedded ptxas")
            return _original_load_cuda_library(self)

    _log(f"Loaded kernel from {ptx_path.name}")

    # Delete PTX if user didn't originally want it kept
    if not _user_wanted_ptx:
        ptx_path.unlink(missing_ok=True)

    return [cuda_runtime.cudaLibrary_t(lib_handle.value)]


def _patched_create_tvm_ffi_function(self):
    # Ensure CUDA library is loaded before TVM FFI creation
    if getattr(self, "_ptxas_cuda_library", None) is None:
        self._ptxas_cuda_library = self._load_cuda_library()
        _log(
            f"Loaded {len(self._ptxas_cuda_library)} CUDA libraries before creating TVM FFI function"
        )
    return _original_create_tvm_ffi_function(self)


def patch():
    """Install system ptxas hook. Call before importing cutlass."""
    global _original_load_cuda_library, _original_create_tvm_ffi_function, _user_wanted_ptx

    assert CUTE_DSL_PTXAS_PATH is not None
    if not os.path.isfile(CUTE_DSL_PTXAS_PATH) or not os.access(CUTE_DSL_PTXAS_PATH, os.X_OK):
        raise RuntimeError(f"ptxas not found: {CUTE_DSL_PTXAS_PATH}")

    _user_wanted_ptx = os.environ.get("CUTE_DSL_KEEP_PTX", "0") == "1"
    assert os.environ.get("CUTE_DSL_KEEP_PTX", "0") == "1", (
        "Require CUTE_DSL_KEEP_PTX=1 to use system's ptxas"
    )

    patched = False
    cuda_jit_function_cls = cutlass.cutlass_dsl.cuda_jit_executor.CudaDialectJitCompiledFunction
    if cuda_jit_function_cls._load_cuda_library is not _patched_load_cuda_library:
        _original_load_cuda_library = cuda_jit_function_cls._load_cuda_library
        cuda_jit_function_cls._load_cuda_library = _patched_load_cuda_library
        patched = True

    from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunctionBase

    if (
        TVMFFIJitCompiledFunctionBase._create_tvm_ffi_function
        is not _patched_create_tvm_ffi_function
    ):
        _original_create_tvm_ffi_function = TVMFFIJitCompiledFunctionBase._create_tvm_ffi_function
        TVMFFIJitCompiledFunctionBase._create_tvm_ffi_function = _patched_create_tvm_ffi_function
        patched = True

    if patched:
        _log(f"Installed system ptxas patch with {CUTE_DSL_PTXAS_PATH}")
    else:
        _log("System ptxas patch already installed")
