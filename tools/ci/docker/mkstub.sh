#!/bin/bash
# mkstub LIB...: replace each shared library with a stub of a few KB that has the
# same soname and exports the same dynamic symbols (and symbol versions).
#
# libtorch_cuda.so hard-links CUDA libraries the test suite never calls (cuFFT,
# cuSPARSE, NCCL, ...). They cannot simply be deleted, or `import torch` fails to
# load; a stub satisfies the dynamic linker at ~1 GB less. Every stubbed function
# traps, so calling one (torch.fft, sparse ops, torch.distributed, cuDNN conv)
# kills the process with SIGILL instead of raising. CI image only.
#
# Not a general ELF export copier: it handles plain functions and data objects
# with at most one version each, which is what these libraries export. A symbol
# exported under several versions is rejected; IFUNC/TLS exports are not preserved.
set -euo pipefail
for lib in "$@"; do
    [ -f "$lib" ] || { echo "mkstub: no such library: $lib" >&2; exit 1; }
    soname=$(readelf -d "$lib" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')
    [ -n "$soname" ] || { echo "mkstub: $lib has no soname" >&2; exit 1; }
    d=$(mktemp -d)
    dup=$(nm -D --defined-only --with-symbol-versions "$lib" | awk '$2 != "A" { split($3, a, "@"); print a[1] }' | sort | uniq -d | head -3)
    [ -z "$dup" ] || { echo "mkstub: $lib exports symbols under several versions (e.g. $dup); not supported" >&2; exit 1; }
    # nm prints "addr type name[@[@]version]"; functions become trapping bodies,
    # data objects zeroed placeholders, and versions go into a linker version script.
    nm -D --defined-only --with-symbol-versions "$lib" | awk -v d="$d" '
        { type = $2; n = split($3, a, "@"); name = a[1]; ver = (n > 1) ? a[n] : "" }
        type ~ /^[TtWwiI]$/ { printf "void f%d(void) __asm__(\"%s\"); void f%d(void) { __builtin_trap(); }\n", NR, name, NR > d "/stub.c" }
        type ~ /^[BbDdRrVvSsGg]$/ { printf "char v%d[64] __asm__(\"%s\") = {0};\n", NR, name > d "/stub.c" }
        ver != "" && type != "A" { syms[ver] = syms[ver] " \"" name "\";" }
        ver != "" && type == "A" { syms[ver] = syms[ver] "" }
        END { for (v in syms) printf "%s { global:%s };\n", v, syms[v] > d "/versions.map" }'
    touch "$d/stub.c"
    args=(-shared -fPIC -O0 -w -o "$d/stub.so" "$d/stub.c" "-Wl,-soname,$soname")
    if [ -s "$d/versions.map" ]; then args+=("-Wl,--version-script,$d/versions.map"); fi
    gcc "${args[@]}"
    printf 'mkstub: %-24s %4d MB -> %3d KB (%d exports)\n' "$(basename "$lib")" \
        $(( $(stat -c %s "$lib") / 1000000 )) $(( $(stat -c %s "$d/stub.so") / 1000 )) "$(wc -l < "$d/stub.c")"
    # Replace, do not overwrite in place: the file may be hard-linked elsewhere.
    mv -f "$d/stub.so" "$lib"
    chmod 755 "$lib"
    rm -rf "$d"
done
