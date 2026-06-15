# Recon scripts for updating the wheel-offsets table

These scripts derive the runtime offsets needed by `quack/dsl/cute_dsl_shim.py`'s
`WHEEL_OFFSETS` table when a new `nvidia-cutlass-dsl` wheel is released.

## What needs to be regenerated per wheel

For each `_cutlass_ir.cpython-XYZ-x86_64-linux-gnu.so`:

1. SHA256 of the file (the dict key in `WHEEL_OFFSETS`).
2. `libnvvm_guard_va` and `libnvvm_table_va` — the dispatch-table guard
   byte and table-pointer globals.
3. The seven `nvPTXCompiler*` entry-point VAs.

## How

```bash
# Use the currently installed cutlass-dsl
python3 find_ptxas_entries.py | tee /tmp/ptxas.txt

# Or point at a specific wheel's extracted .so
python3 find_ptxas_entries.py \
  --so /tmp/cp313/cutlass/_mlir/_mlir_libs/_cutlass_ir.cpython-313-x86_64-linux-gnu.so

# libnvvm dispatch-table:
python3 find_libnvvm_table.py --so <path>
```

The libnvvm `guard`/`table` globals can also be derived directly from
the PTX-emission function's first two RIP-relative reads, which are
`movzx eax, BYTE PTR [rip+disp32]` (guard) and
`cmp QWORD PTR [rip+disp32], 0` (table-pointer). The checked-in
`WHEEL_OFFSETS` entries already include the cp310..cp314{,t} x86_64
wheel data for the supported CuTe DSL release.

## Adding a new wheel

1. Run both scripts on the new `.so`.
2. Add a new entry to `quack/dsl/cute_dsl_shim.py::WHEEL_OFFSETS`, keyed by
   the SHA256, with the 9 offsets.
3. Run `pytest tests/test_cute_dsl_shim.py` to confirm activation + a
   real-kernel regression passes.

## Cross-architecture

The current shim is x86_64-only (12-byte `movabs rax, imm64; jmp rax`
trampolines, ELF `/proc/self/maps` parsing). Aarch64 is a separate
deferred work item — see `../DESIGN.md` § "Future work".
