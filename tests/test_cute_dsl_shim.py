"""
Regression tests for the cute-DSL CUDA 13.3 toolchain shim.

These tests are skipped unless the shim's prerequisites are present:
  * Built libcute_dsl_shim.so (run `make -C tools/cute_dsl_shim`).
  * System CUDA 13.3 ptxas and libnvvm reachable at default paths.
  * Installed cutlass-dsl matches a known wheel SHA256.

When prerequisites are present, the tests assert end-to-end that:
  * try_activate() returns True.
  * The fatbin produced by a fresh cute.compile() bears `release 13.3`.
  * A small reference kernel still produces numerically correct output.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM_LIB = REPO_ROOT / "tools" / "cute_dsl_shim" / "libcute_dsl_shim.so"
VERIFY_PY = REPO_ROOT / "tools" / "cute_dsl_shim" / "verify.py"


def _shim_prereqs_present() -> bool:
    """Mirror the runtime checks in quack.dsl.cute_dsl_shim.try_activate."""
    if sys.platform != "linux":
        return False
    if not SHIM_LIB.exists():
        return False
    try:
        from quack.dsl.cute_dsl_shim import WHEEL_OFFSETS, _find_cutlass_ir_path, _sha256_of
    except Exception:
        return False
    so = _find_cutlass_ir_path()
    if so is None or not so.exists():
        return False
    if _sha256_of(so) not in WHEEL_OFFSETS:
        return False
    # libnvvm + ptxas
    libnvvm = (
        Path("/usr/local/cuda-13.3/nvvm/lib64/libnvvm.so.4"),
        Path("/usr/local/cuda/nvvm/lib64/libnvvm.so.4"),
    )
    if not any(p.exists() for p in libnvvm):
        return False
    if not Path("/usr/local/cuda/bin/ptxas").exists():
        return False
    return True


_PREREQS = _shim_prereqs_present()


pytestmark = pytest.mark.skipif(
    not _PREREQS,
    reason="cute_dsl_shim prerequisites missing (libcute_dsl_shim.so, "
    "CUDA 13.3 toolchain, or known wheel SHA256)",
)


def _run_verify(env_overrides: dict[str, str | None]) -> tuple[int, str, str]:
    env = os.environ.copy()
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    p = subprocess.run(
        [sys.executable, str(VERIFY_PY), "--require-shim"]
        if env_overrides.get("QUACK_CUTE_DSL_SHIM") == "1"
        else [sys.executable, str(VERIFY_PY)],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return p.returncode, p.stdout, p.stderr


def test_default_off_reports_13_1():
    """Sanity: by default the shim is off and cubins bear ptxas 13.1."""
    rc, out, err = _run_verify({"QUACK_CUTE_DSL_SHIM": None})
    assert rc == 0, f"verify failed:\nstdout:\n{out}\nstderr:\n{err}"
    m = re.search(r"ptxas (\d+\.\d+)", out)
    assert m, f"no ptxas version in stdout: {out!r}"
    assert m.group(1) == "13.1", f"baseline reports unexpected ptxas: {m.group(1)}"
    assert "shim_active=False" in out


def test_shim_reports_13_3():
    """Shim activates and the cubin bears ptxas 13.3."""
    rc, out, err = _run_verify({"QUACK_CUTE_DSL_SHIM": "1"})
    assert rc == 0, f"verify failed:\nstdout:\n{out}\nstderr:\n{err}"
    m = re.search(r"ptxas (\d+\.\d+)", out)
    assert m, f"no ptxas version in stdout: {out!r}"
    assert m.group(1) == "13.3", f"shim active but cubin reports ptxas {m.group(1)} (expected 13.3)"
    assert "shim_active=True" in out


def test_shim_skip_nvvm_still_changes_ptxas():
    """With NVVM disabled the cubin should still be from ptxas 13.3
    (but the embedded NVVM 13.1 will keep emitting PTX .version 9.1)."""
    rc, out, err = _run_verify(
        {
            "QUACK_CUTE_DSL_SHIM": "1",
            "QUACK_CUTE_DSL_SHIM_NO_NVVM": "1",
        }
    )
    assert rc == 0, f"verify failed:\nstdout:\n{out}\nstderr:\n{err}"
    m = re.search(r"ptxas (\d+\.\d+)", out)
    assert m and m.group(1) == "13.3", f"got {out!r}"
    assert "shim_active=True" in out


def test_real_kernel_under_shim():
    """Ensure a real quack kernel still passes its numerical reference
    when run with the shim active. Uses rmsnorm because it's the smallest
    end-to-end kernel the test suite exercises."""
    env = os.environ.copy()
    env["QUACK_CUTE_DSL_SHIM"] = "1"
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-x",
            "--no-header",
            "tests/test_rmsnorm.py::test_rmsnorm_forward_backward"
            "[False-1-192-input_dtype0-weight_dtype0-1e-06]",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        cwd=str(REPO_ROOT),
    )
    assert p.returncode == 0, (
        f"rmsnorm regression under shim failed:\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    )
