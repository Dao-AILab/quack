# Copyright (c) 2026, Tri Dao.

"""Backend selection and import isolation, exercised in fresh interpreters."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_IS_ROCM = torch.version.hip is not None


def _run_isolated(code: str) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        (str(_ROOT), env["PYTHONPATH"]) if env.get("PYTHONPATH") else (str(_ROOT),)
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    _IS_ROCM and importlib.util.find_spec("flydsl") is None,
    reason="ROCm dispatch requires the optional FlyDSL package",
)
def test_rmsnorm_fwd_dispatches_by_torch_build():
    from quack import rmsnorm_fwd

    expected = "quack.rmsnorm_flydsl" if _IS_ROCM else "quack.rmsnorm"
    assert rmsnorm_fwd.__module__ == expected


@pytest.mark.skipif(not _IS_ROCM, reason="ROCm import-isolation contract")
def test_rocm_cute_exports_fail_without_submodule_fallback():
    _run_isolated(
        """
import sys
import quack

try:
    from quack import rmsnorm
except ImportError as exc:
    assert "requires the CUDA/CuTe backend" in str(exc)
else:
    raise AssertionError("ROCm imported the CuTe RMSNorm backend")

assert "quack.rmsnorm" not in sys.modules
assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)
"""
    )


@pytest.mark.skipif(not _IS_ROCM, reason="ROCm import-isolation contract")
def test_rocm_async_compile_is_rejected_before_loading_cute():
    _run_isolated(
        """
import sys
import pytest
from quack.testing import pytest_plugin

class Config:
    @staticmethod
    def getoption(name, default=None):
        return 1 if name == "--async-compile" else default

config = Config()
try:
    pytest_plugin.pytest_configure(config)
except pytest.UsageError as exc:
    assert "unavailable on ROCm" in str(exc)
else:
    raise AssertionError("ROCm accepted --async-compile")

assert not getattr(config, "_quack_async_pool_active", False)
pytest_plugin.pytest_unconfigure(config)

assert "quack.cache.async_compile" not in sys.modules
assert "quack.cache.jit" not in sys.modules
assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)
"""
    )


@pytest.mark.skipif(_IS_ROCM, reason="CUDA-side FlyDSL import guard")
@pytest.mark.parametrize("module", ["quack.flydsl_runtime", "quack.rmsnorm_flydsl"])
def test_cuda_rejects_flydsl_modules_before_importing_flydsl(module):
    _run_isolated(
        f"""
import importlib
import importlib.abc
import sys

class ForbidFlyDSL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "flydsl" or fullname.startswith("flydsl."):
            raise AssertionError(f"FlyDSL import attempted: {{fullname}}")
        return None

sys.meta_path.insert(0, ForbidFlyDSL())
try:
    importlib.import_module({module!r})
except ImportError as exc:
    assert "requires a ROCm PyTorch build" in str(exc)
else:
    raise AssertionError("CUDA imported a ROCm-only FlyDSL module")

assert not any(name == "flydsl" or name.startswith("flydsl.") for name in sys.modules)
"""
    )


def test_flydsl_config_does_not_import_the_dsl():
    _run_isolated(
        """
import sys
import quack.rmsnorm_flydsl_config

assert not any(name == "flydsl" or name.startswith("flydsl.") for name in sys.modules)
"""
    )


@pytest.mark.skipif(not _IS_ROCM, reason="FlyDSL is optional only on the ROCm path")
def test_rocm_benchmark_help_does_not_import_flydsl():
    _run_isolated(
        """
import importlib.abc
import runpy
import sys

class ForbidFlyDSL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "flydsl" or fullname.startswith("flydsl."):
            raise AssertionError(f"FlyDSL import attempted: {fullname}")
        return None

sys.meta_path.insert(0, ForbidFlyDSL())
sys.argv = ["benchmark_rmsnorm.py", "--help"]
try:
    runpy.run_path("benchmarks/benchmark_rmsnorm.py", run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError("--help did not exit")
"""
    )
