"""Find what writes to the dispatch-table global at 0x92d1a90.

If something writes to it (even lazily), we know the wrapper IS reachable.
If nothing writes to it, it's truly dead.
"""
import struct, capstone
SO = "/opt/venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/_mlir/_mlir_libs/_cutlass_ir.cpython-312-x86_64-linux-gnu.so"
DATA = open(SO, "rb").read()

GLOBAL_VA = 0x92d1a90
TEXT_START, TEXT_END = 0x548240, 0x79a2a74

# Patterns that could write to a RIP-relative global at addr X:
# - mov qword ptr [rip+disp32], reg     48 89 XX 25 disp32   (REX.W + opcode 89)
#   But 89 with mod=00, rm=101 is `mov [rip+disp32], reg`, modrm = 00 reg 101
# - mov qword ptr [rip+disp32], imm32   48 c7 05 disp32 imm32
# Let me just scan for any RIP-relative load OR store of GLOBAL_VA and report.

cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
cs.detail = True

# Brute force: scan all RIP-relative effective addresses by trying to disassemble at every byte.
# Too slow. Instead, scan for the byte patterns most likely to encode a write to GLOBAL_VA.

# Compute the disp32 used by an instruction at byte i to reference GLOBAL_VA.
# disp32 = GLOBAL_VA - (i + instr_len)
# We need instr_len to know where to put disp32. For x86-64 with REX prefix:
# mov [rip+disp32], reg  -> 48 89 /r modrm=00 reg 101 -> length 7
# mov qword ptr [rip+disp32], imm32  -> 48 c7 05 disp32 imm32 -> length 11
# mov rax, qword ptr [rip+disp32]    -> 48 8b 05 disp32      -> length 7
# lea rax, [rip+disp32]              -> 48 8d 05 disp32      -> length 7

# For each instruction length, try every position and check the disp32 matches.
def scan_rip_pattern(prefix_bytes, instr_len, op_name):
    """For an instruction with the given prefix and length, find all positions where it targets GLOBAL_VA."""
    hits = []
    pos = TEXT_START
    while True:
        i = DATA.find(prefix_bytes, pos, TEXT_END)
        if i < 0: break
        # The disp32 field starts at i + len(prefix_bytes); but for 'mov [rip+d], imm32' the disp32 starts at i + 3.
        # Let me just have the caller pass the disp offset.
        disp_offset = len(prefix_bytes)
        disp = struct.unpack_from("<i", DATA, i + disp_offset)[0]
        # The RIP value at the moment of effective-address computation is i + instr_len
        # (for store-type with disp32 at the END except imm32 which follows disp32).
        if op_name == "mov_qword_imm32":
            # 48 c7 05 disp32 imm32 — length 11; RIP at end-of-instruction = i+11
            # Actually for RIP-relative addressing: EA = next_RIP + disp32. next_RIP = i + instr_len.
            ea = i + instr_len + disp
        else:
            ea = i + instr_len + disp
        if ea == GLOBAL_VA:
            hits.append(i)
        pos = i + 1
    return hits

# Common 7-byte instructions: 48 8b 05 disp32 (mov rax, [rip+disp]); 48 8d 05 disp32 (lea rax, [rip+disp])
# 64-bit register-to-mem stores: 48 89 05 disp32 (mov [rip+disp], rax), 48 89 0d (rcx), 48 89 15 (rdx), 48 89 1d (rbx), 48 89 25 (rsp—nonsensical), 48 89 2d (rbp), 48 89 35 (rsi), 48 89 3d (rdi).

print(f"Searching for instructions that REFERENCE 0x{GLOBAL_VA:x}:")
print()

# Stores to the global (writes):
for reg_id, reg_name in [(0x05,"rax"),(0x0d,"rcx"),(0x15,"rdx"),(0x1d,"rbx"),(0x2d,"rbp"),(0x35,"rsi"),(0x3d,"rdi")]:
    hits = scan_rip_pattern(bytes([0x48, 0x89, reg_id]), 7, f"mov_qword_{reg_name}")
    if hits:
        print(f"  mov [0x{GLOBAL_VA:x}], {reg_name}  (48 89 {reg_id:02x}): {len(hits)} hit(s)")
        for h in hits[:5]: print(f"     @ 0x{h:x}")

# 32-bit reg stores (REX.W=0):  89 05/0d/... — but the global is 8 bytes so a 32-bit store is unusual
# imm32 store: 48 c7 05 disp32 imm32
hits = scan_rip_pattern(b"\x48\xc7\x05", 11, "mov_qword_imm32")
if hits:
    print(f"  mov qword ptr [0x{GLOBAL_VA:x}], imm32  (48 c7 05): {len(hits)} hit(s)")
    for h in hits[:5]:
        imm = struct.unpack_from("<i", DATA, h+7)[0]
        print(f"     @ 0x{h:x} imm32=0x{imm:x}")

# 32-bit reg stores without REX (smaller globals — unlikely here)
hits = scan_rip_pattern(b"\xc7\x05", 10, "mov_dword_imm32")
if hits:
    print(f"  mov dword ptr [0x{GLOBAL_VA:x}], imm32  (c7 05): {len(hits)} hit(s)")
    for h in hits[:5]: print(f"     @ 0x{h:x}")

# LEA-form reference (taking address-of):
for reg_id, reg_name in [(0x05,"rax"),(0x0d,"rcx"),(0x15,"rdx"),(0x1d,"rbx"),(0x2d,"rbp"),(0x35,"rsi"),(0x3d,"rdi")]:
    hits = scan_rip_pattern(bytes([0x48, 0x8d, reg_id]), 7, f"lea_{reg_name}")
    if hits:
        print(f"  lea {reg_name}, [0x{GLOBAL_VA:x}]  (48 8d {reg_id:02x}): {len(hits)} hit(s)")
        for h in hits[:5]: print(f"     @ 0x{h:x}")

# Loads (for reference)
for reg_id, reg_name in [(0x05,"rax"),(0x0d,"rcx"),(0x15,"rdx"),(0x1d,"rbx"),(0x2d,"rbp"),(0x35,"rsi"),(0x3d,"rdi")]:
    hits = scan_rip_pattern(bytes([0x48, 0x8b, reg_id]), 7, f"load_{reg_name}")
    if hits:
        print(f"  mov {reg_name}, qword ptr [0x{GLOBAL_VA:x}]  (48 8b {reg_id:02x}): {len(hits)} hit(s)")
        for h in hits[:5]: print(f"     @ 0x{h:x}")
