"""
Find nvPTXCompiler*, nvFatbin*, and nvvm* public-API entry points inside the
stripped _cutlass_ir.so by:
  1. Locating the assertion/__PRETTY_FUNCTION__ strings (anchors).
  2. Finding the LEA that loads each anchor's address.
  3. Disassembling backwards/forwards from the LEA to find the related CALL.
  4. The CALL's target is the public API function we want to hijack.
"""

import argparse
import importlib.util
import struct

import capstone


def _default_so_path() -> str:
    spec = importlib.util.find_spec("cutlass._mlir._mlir_libs._cutlass_ir")
    if spec is None or spec.origin is None:
        raise SystemExit(
            "could not locate _cutlass_ir.so; pass --so /path/to/_cutlass_ir.so"
        )
    return spec.origin


_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--so", default=None,
                 help="Path to _cutlass_ir.so (default: installed cutlass-dsl)")
ARGS = _ap.parse_args()
SO_PATH = ARGS.so or _default_so_path()
DATA = open(SO_PATH, "rb").read()
print(f"# scanning {SO_PATH} ({len(DATA)} bytes)")

# .text spans these file offsets (== vaddrs for first LOAD, offset=0, vaddr=0).
# Derived for cp312-x86_64 4.5.2; if you're scanning a different wheel, run
# `readelf -S` on the .so and adjust these to match the [.text] section.
TEXT_VA_START = 0x548240
TEXT_VA_END   = 0x79a2a74
# Rodata where the assertion strings live:
RODATA_VA_START = 0x79a2ac0
RODATA_VA_END   = 0x88680 + 0x28

# Helper: file offset == vaddr in first LOAD segment.
def rd(va, n):
    return DATA[va:va+n]

# Anchors: each as `<bytes_in_rodata>` (NUL-terminated literal).
ANCHORS = [
    # nvPTXCompiler — used as assertion / __FUNCTION__ strings
    "nvPTXCompilerCreate(&compiler, ptxCode.size(), ptxCode.c_str())",
    "nvPTXCompilerGetErrorLogSize(compiler, &logSize)",
    "nvPTXCompilerGetErrorLog(compiler, log.data())",
    "nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize)",
    "nvPTXCompilerGetCompiledProgram(compiler, (void *)binary.data())",
    "nvPTXCompilerDestroy(&compiler)",
    # nvFatbin
    "nvFatbinCreate(&handle, cubinOpts, 1)",
    "nvFatbinAddCubin( handle, binary.data(), binary.size(), chip.data(), nullptr)",
    "nvFatbinAddPTX( handle, ptxCode.data(), ptxCode.size(), chip.data(), nullptr, nullptr)",
    "nvFatbinSize(handle, &fatbinSize)",
    "nvFatbinGet(handle, (void *)fatbin.data())",
    "nvFatbinDestroy(&handle)",
]

# Locate each anchor's vaddr (== file offset).
def find_anchor(s):
    needle = s.encode() + b"\x00"
    i = DATA.find(needle)
    if i < 0:
        return None
    return i

anchor_vas = {}
for a in ANCHORS:
    va = find_anchor(a)
    if va is None:
        print(f"!! anchor missing: {a!r}")
    else:
        anchor_vas[a] = va

# Find every LEA reg, [rip+disp32] that targets each anchor.
# Encoding: REX.W (0x48 / 0x4c) + 0x8d + ModR/M(0x05,0x0d,0x15,0x1d,0x25,0x2d,0x35,0x3d) + disp32
def find_lea_loads_of_va(target_va):
    hits = []
    LEA_PREFIXES = []
    for rex in (0x48, 0x4c):
        for mod in (0x05,0x0d,0x15,0x1d,0x25,0x2d,0x35,0x3d):
            LEA_PREFIXES.append(bytes([rex, 0x8d, mod]))
    for pre in LEA_PREFIXES:
        pos = TEXT_VA_START
        while True:
            i = DATA.find(pre, pos, TEXT_VA_END)
            if i < 0: break
            disp = struct.unpack_from("<i", DATA, i+3)[0]
            tgt = (i + 7) + disp
            if tgt == target_va:
                hits.append(i)
            pos = i + 1
    return hits

# For each LEA, scan forward to find the next CALL instruction. That CALL's
# target is most likely the public API entry point. (The wrapper structure
# typically is: CALL api; cmp ret_val; jne err_label; ... err_label loads the
# anchor string and bails.)  Sometimes the LEA is in the err label; we may
# need to look BOTH forward and backward.

cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
cs.detail = True

def disasm_window(start_va, n_bytes=512):
    code = DATA[start_va:start_va+n_bytes]
    return list(cs.disasm(code, start_va))

def find_nearby_call(lea_va, max_window=4096):
    """Search forward AND backward from `lea_va` for direct CALLs."""
    found_forward = []
    found_backward = []
    # Forward scan: disassemble starting at the LEA itself (it's 7 bytes long).
    for ins in disasm_window(lea_va, max_window):
        if ins.mnemonic == "call" and ins.op_str.startswith("0x"):
            try: t = int(ins.op_str, 16); found_forward.append((ins.address, t, ins))
            except ValueError: pass
            if len(found_forward) >= 5: break
    # Backward scan: walk back in ~64-byte chunks, disassemble, take last few CALLs.
    for back in (64, 128, 256, 512, 1024, 2048):
        block_start = lea_va - back
        if block_start < TEXT_VA_START: break
        seen_lea = False
        local_calls = []
        for ins in disasm_window(block_start, back + 16):
            if ins.address == lea_va:
                seen_lea = True
                break
            if ins.mnemonic == "call" and ins.op_str.startswith("0x"):
                try: t = int(ins.op_str, 16); local_calls.append((ins.address, t, ins))
                except ValueError: pass
        if seen_lea and local_calls:
            found_backward = local_calls
            break
    return found_forward, found_backward

print(f"\n{'='*78}")
print(f"Anchor LEAs and nearby CALL targets (candidate API entry points)")
print(f"{'='*78}\n")

call_targets = {}  # target_va -> {anchors that point to it}

for a in ANCHORS:
    va = anchor_vas.get(a)
    if va is None: continue
    leas = find_lea_loads_of_va(va)
    if not leas:
        print(f"-- {a!r}: NO LEA")
        continue
    print(f"-- anchor: {a!r}")
    print(f"   string vaddr: 0x{va:x}")
    for lea_va in leas:
        fwd, bwd = find_nearby_call(lea_va)
        print(f"   LEA at 0x{lea_va:x}:")
        print(f"     forward calls:  {[hex(t) for _,t,_ in fwd[:5]]}")
        print(f"     backward calls: {[hex(t) for _,t,_ in bwd[-5:]]}")
        for _, t, ins in fwd[:2] + bwd[-2:]:
            call_targets.setdefault(t, set()).add(a)

print(f"\n{'='*78}")
print(f"Call target summary  (sorted)")
print(f"{'='*78}\n")
for t, anchors in sorted(call_targets.items()):
    print(f"  0x{t:x}  <-  {len(anchors)} anchor(s):")
    for a in anchors:
        print(f"      {a}")
