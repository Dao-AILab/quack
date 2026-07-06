# Copyright (c) 2026, Tri Dao.
"""Forkserver preload module for the async compile pool.

Imported once inside the multiprocessing *forkserver* process (see
``multiprocessing.set_forkserver_preload``). Every pool worker is then
``fork()``-ed from that warm process and inherits the imported interpreter
state via copy-on-write: worker startup drops from ~13 s (torch 4 s +
cutlass/cute/tvm_ffi 9 s per spawn) to ~0.1 s per fork.

This is the same architecture as PyTorch Inductor's compile-worker
``SubprocPool``: one sidecar pays the import, workers fork from it.

Fork-safety: nothing here may initialize CUDA (a forked child of a
CUDA-initialized process is undefined behavior). Importing torch and
cutlass does not create a CUDA context; workers additionally run with
``CUDA_VISIBLE_DEVICES=""`` + ``QUACK_ARCH``/``CUTE_DSL_ARCH`` overrides so
the compile path never touches the driver (the same mechanism the CPU-only
compile workflow uses).
"""

import os
import subprocess

# Pin the target arch BEFORE importing quack: import-time code paths (e.g.
# rmsnorm_config._detect_arch_major) consult QUACK_ARCH via
# get_device_capacity and would otherwise initialize CUDA — which both makes
# the forkserver's context leak into children and trips torch's forked-child
# guard. nvidia-smi queries the capability without creating a CUDA context.
if "QUACK_ARCH" not in os.environ:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        cap = out.stdout.strip().splitlines()[0].strip()  # e.g. "9.0"
        major, minor = cap.split(".")
        os.environ["QUACK_ARCH"] = f"{major}{minor}"
        os.environ.setdefault(
            "CUTE_DSL_ARCH", f"sm_{major}{minor}a" if int(major) >= 9 else f"sm_{major}{minor}"
        )
    except Exception:
        pass  # CPU-only box: rely on user-provided env, as before

# Belt and suspenders: even if some import still tries to touch CUDA, make
# it see no devices rather than creating a context in the forkserver.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import quack.cache  # noqa: F401, E402  (pulls torch, cutlass.cute, tvm_ffi)
