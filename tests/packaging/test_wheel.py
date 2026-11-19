"""Validate regular wheels without importing CUDA dependencies."""

import importlib.util
import os
from pathlib import Path
import sys
import zipfile

import pytest


@pytest.fixture(scope="module")
def wheel():
    wheels = list((Path(__file__).resolve().parents[2] / "dist").glob("*.whl"))
    if not wheels:
        pytest.skip("Build a regular wheel with python -m build --wheel first")
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as archive:
        yield archive


def test_wheel_contents(wheel):
    assert wheel.testzip() is None
    names = wheel.namelist()
    assert "quack/_filelock.py" in names
    assert not any("__editable__" in name for name in names)
    for name in names:
        if name.endswith(".py"):
            compile(wheel.read(name), name, "exec")


@pytest.fixture
def locks(wheel, tmp_path):
    if sys.platform != "win32":
        pytest.skip("Windows LockFileEx")
    module_path = tmp_path / "_filelock.py"
    module_path.write_bytes(wheel.read("quack/_filelock.py"))
    spec = importlib.util.spec_from_file_location("wheel_filelock", module_path)
    lock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lock)
    first = os.open(tmp_path / "lock", os.O_CREAT | os.O_RDWR)
    second = os.open(tmp_path / "lock", os.O_RDWR)
    try:
        yield lock, first, second
    finally:
        os.close(second)
        os.close(first)


def test_packaged_exclusive_lock(locks):
    lock, first, second = locks
    lock.flock(first, lock.LOCK_EX | lock.LOCK_NB)
    with pytest.raises(BlockingIOError):
        lock.flock(second, lock.LOCK_EX | lock.LOCK_NB)
    lock.flock(first, lock.LOCK_UN)
    lock.flock(second, lock.LOCK_EX)
    lock.flock(second, lock.LOCK_UN)


def test_packaged_shared_lock(locks):
    lock, first, second = locks
    lock.flock(first, lock.LOCK_SH | lock.LOCK_NB)
    lock.flock(second, lock.LOCK_SH | lock.LOCK_NB)
    lock.flock(second, lock.LOCK_UN)
    with pytest.raises(BlockingIOError):
        lock.flock(second, lock.LOCK_EX | lock.LOCK_NB)
    lock.flock(first, lock.LOCK_UN)


def test_packaged_blocking_lock(locks):
    lock, first, _ = locks
    lock.flock(first, lock.LOCK_EX)
    lock.flock(first, lock.LOCK_UN)


@pytest.mark.parametrize("operation", [0, 3, 16])
def test_packaged_invalid_operation(locks, operation):
    lock, first, _ = locks
    with pytest.raises(ValueError):
        lock.flock(first, operation)
