#!/usr/bin/env python3
"""Dump PTX and SASS of cute-dsl kernels from a script or Python module.

Disables the QuACK persistent kernel cache, sets CUTE_DSL_KEEP=ptx,cubin, runs
the target, then disassembles all generated .cubin files with nvdisasm.

Usage::

    python tools/dump_sass.py benchmarks/benchmark_gemm.py -- --mnkl 4096,4096,4096,1
    python tools/dump_sass.py benchmarks/benchmark_gemm.py -o /tmp/sass -- --mnkl 4096,4096,4096,1
    python tools/dump_sass.py benchmarks/benchmark_gemm.py --ptx-only -- --mnkl 4096,4096,4096,1
    python tools/dump_sass.py -m pytest -- -q 'tests/test_linear.py::test_gemm_gated_pingpong_configs' -s
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_nvdisasm():
    path = shutil.which("nvdisasm")
    if path:
        return path
    for cuda_dir in sorted(Path("/usr/local").glob("cuda*"), reverse=True):
        candidate = cuda_dir / "bin" / "nvdisasm"
        if candidate.is_file():
            return str(candidate)
    return None


def add_cute_keep_tokens(env, required_tokens):
    keep_tokens = {
        token.strip().lower()
        for token in env.get("CUTE_DSL_KEEP", "").split(",")
        if token.strip()
    }
    if "all" not in keep_tokens:
        keep_tokens.update(required_tokens)
    env["CUTE_DSL_KEEP"] = ",".join(sorted(keep_tokens))


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        idx = argv.index("--")
        our_argv, target_args = argv[:idx], argv[idx + 1 :]
    else:
        our_argv, target_args = argv, []

    parser = argparse.ArgumentParser(
        description="Dump PTX and SASS of cute-dsl kernels.",
        usage=(
            "%(prog)s SCRIPT [-o DIR] [--ptx-only] [-- SCRIPT_ARGS...]\n"
            "       %(prog)s [-o DIR] [--ptx-only] -m MODULE [-- MODULE_ARGS...]"
        ),
    )
    parser.add_argument("script", nargs="?", help="Python script to run")
    parser.add_argument("-m", "--module", help="Python module to run")
    parser.add_argument("-o", "--output-dir", default="dump_sass_out", help="Output directory")
    parser.add_argument("--ptx-only", action="store_true", help="Skip SASS disassembly")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Allow QuACK to use its persistent .o cache instead of forcing recompilation",
    )
    args = parser.parse_args(our_argv)
    if args.module is not None:
        if args.script is not None:
            parser.error("module arguments must come after --")
        cmd = [sys.executable, "-m", args.module, *target_args]
    else:
        if args.script is None:
            parser.error("expected SCRIPT or -m MODULE")
        script = Path(args.script)
        if not script.is_file():
            print(f"Error: {script} not found", file=sys.stderr)
            sys.exit(1)
        cmd = [sys.executable, str(script), *target_args]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("*.ptx", "*.cubin", "*.sass"):
        for f in out_dir.glob(ext):
            f.unlink()

    env = os.environ.copy()
    if not args.use_cache:
        env["QUACK_CACHE_ENABLED"] = "0"
    add_cute_keep_tokens(env, {"ptx", "cubin"})
    env["CUTE_DSL_DUMP_DIR"] = str(out_dir.resolve())

    print(f"Running: {' '.join(cmd)}")
    print(f"Dump dir: {out_dir.resolve()}\n")
    if not args.use_cache:
        print("QuACK cache: disabled via QUACK_CACHE_ENABLED=0\n")
    subprocess.run(cmd, env=env)

    ptx_files = sorted(out_dir.glob("*.ptx"))
    cubin_files = sorted(out_dir.glob("*.cubin"))
    print(f"\nPTX: {len(ptx_files)}, CUBIN: {len(cubin_files)}")
    for f in ptx_files:
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")

    if not args.ptx_only and cubin_files:
        nvdisasm = find_nvdisasm()
        if nvdisasm is None:
            print("nvdisasm not found — skipping SASS disassembly", file=sys.stderr)
        else:
            for cubin in cubin_files:
                sass_path = cubin.with_suffix(".sass")
                result = subprocess.run([nvdisasm, str(cubin)], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  nvdisasm failed: {cubin.name}: {result.stderr.strip()}", file=sys.stderr)
                    continue
                sass_path.write_text(result.stdout)
                print(f"  {sass_path.name}  ({result.stdout.count(chr(10))} lines)")

    print("\nSASS files:")
    for f in sorted(out_dir.glob("*.sass")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
