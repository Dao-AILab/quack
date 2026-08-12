# Copyright (c) 2026, Tri Dao.

"""Host plumbing the FlyDSL kernels share: arch validation, placeholders, and
the import guards that keep a ROCm-only install from reaching for CuTe."""

import dataclasses
import re
import sys

import pytest
import torch
from flydsl_env import ARCH as _ARCH
from flydsl_env import CAN_RUN as _CAN_RUN
from flydsl_env import DEVICE as _DEVICE
from flydsl_env import SKIP_REASON as _SKIP_REASON

pytestmark = pytest.mark.skipif(not _CAN_RUN, reason=_SKIP_REASON)

if _CAN_RUN:
    import quack.flydsl_runtime as fly_runtime


@pytest.mark.parametrize("arch", ("9.5.0", "gfx942"), ids=("dotted", "wrong-gfx"))
def test_compile_target_that_the_device_never_reports_is_rejected(monkeypatch, arch):
    # ARCH reaches FlyDSL's ROCDL pipeline as `chip` verbatim, so an override
    # the device never reports has to fail here. Normalizing "9.5.0" to gfx950
    # would let it past this check and fail later inside the compiler instead.
    # get_backend() memoizes one instance per (name, arch), so patching the
    # instance is what validate_arch reads back.
    backend = fly_runtime.flyc.get_backend()
    monkeypatch.setattr(backend, "target", dataclasses.replace(backend.target, arch=arch))

    with pytest.raises(ValueError, match=f"FlyDSL compiles for {re.escape(arch)}"):
        fly_runtime.validate_arch(_DEVICE, frozenset({_ARCH}), "RMSNorm")


def test_every_caller_gets_its_own_supported_set_checked():
    # A per-device arch cache that also gated the supported check would let the
    # second kernel inherit the first one's verdict on this GPU.
    assert fly_runtime.validate_arch(_DEVICE, frozenset({_ARCH}), "RMSNorm") == _ARCH

    with pytest.raises(ValueError, match=f"supports gfx900; cuda:\\d+ is {_ARCH}"):
        fly_runtime.validate_arch(_DEVICE, frozenset({"gfx900"}), "Elsewhere")


def test_placeholders_are_shared_across_spellings_of_one_device():
    # Callers hand over whatever x.device gave them, so an unindexed "cuda" and
    # an explicit "cuda:N" have to land on the same cached tensor.
    bare = fly_runtime.empty_placeholder(torch.device("cuda"), torch.float32)
    indexed = fly_runtime.empty_placeholder(_DEVICE, torch.float32)
    assert bare is indexed
    assert bare.numel() == 0 and bare.device.index == _DEVICE.index


def test_rocm_cute_exports_fail_without_submodule_fallback():
    with pytest.raises(ImportError, match="requires the CUDA/CuTe backend"):
        __import__("quack", fromlist=("rmsnorm",))

    assert "quack.rmsnorm" not in sys.modules
    assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)


def test_async_compile_is_rejected_before_loading_cute():
    from quack.testing import pytest_plugin

    class AsyncCompileConfig:
        @staticmethod
        def getoption(*_args, **_kwargs):
            return 1

    with pytest.raises(pytest.UsageError, match="unavailable on ROCm"):
        pytest_plugin.pytest_configure(AsyncCompileConfig())

    assert "quack.cache.jit" not in sys.modules
    assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)
