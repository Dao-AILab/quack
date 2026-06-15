/*
 * cute_dsl_shim.c — runtime patcher that redirects the CUDA 13.1 toolchain
 * embedded inside CUTLASS DSL's `_cutlass_ir.cpython-*.so` to a system
 * CUDA 13.3 toolchain.
 *
 * Two intercepts:
 *   1. libNVVM dispatch-table patch  — overwrite the function-pointer table
 *      that the cute-DSL libNVVM wrapper builds lazily. Each slot is
 *      pointed directly at the corresponding libnvvm.so.4 symbol resolved
 *      from CUDA 13.3 via dlopen+dlsym. The guard byte that gates the lazy
 *      initializer is set to 1 so the wrapper skips rebuilding the table.
 *   2. nvPTXCompiler entry-point trampolines — overwrite the first 12 bytes
 *      of each of the seven public-API functions with `movabs rax, addr;
 *      jmp rax` to a small in-process shim that fork/exec's an external
 *      ptxas binary and returns its cubin via the same opaque-handle ABI.
 *
 * The C side has NO hardcoded VAs. The caller passes the relocated
 * `_cutlass_ir.so` base address and a per-wheel offsets struct via the
 * public install entry point. Wheel discovery (SHA256 → offsets) is done
 * in Python.
 *
 * Build:
 *   gcc -shared -fPIC -O2 -fvisibility=hidden \
 *       cute_dsl_shim.c -o libcute_dsl_shim.so -ldl -lpthread
 *
 * Public C API (see the declarations below).
 *
 * Threading: install must be called exactly once, early, before any
 * concurrent compile activity. The internal patcher mutex makes it safe
 * to call install() from multiple threads (subsequent callers no-op), but
 * the cute-DSL pipeline itself may not tolerate being patched mid-flight.
 *
 * Verified on x86_64 Linux with nvidia-cutlass-dsl-libs-cu13 == 4.5.2,
 * CPython 3.10..3.14{,t}. See ../README.md for the offset table.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/* ============================================================
 *  Public ABI (kept in sync with quack/dsl/cute_dsl_shim.py)
 * ============================================================ */

#define CUTE_DSL_SHIM_ABI_VERSION 1u

/* All VAs are unsigned offsets relative to _cutlass_ir.so's mapped base.
 * A VA of 0 means "skip this entry" — useful for partial installs.
 */
struct cute_dsl_shim_offsets {
    uint32_t abi_version;          /* must be CUTE_DSL_SHIM_ABI_VERSION */
    uintptr_t libnvvm_guard_va;
    uintptr_t libnvvm_table_va;
    uintptr_t nvptx_create_va;
    uintptr_t nvptx_compile_va;
    uintptr_t nvptx_destroy_va;
    uintptr_t nvptx_get_compiled_program_size_va;
    uintptr_t nvptx_get_compiled_program_va;
    uintptr_t nvptx_get_error_log_size_va;
    uintptr_t nvptx_get_error_log_va;
};

#define CUTE_DSL_SHIM_FLAG_DEBUG       (1u << 0)
#define CUTE_DSL_SHIM_FLAG_SKIP_NVVM   (1u << 1)
#define CUTE_DSL_SHIM_FLAG_SKIP_PTXAS  (1u << 2)

struct cute_dsl_shim_config {
    uint32_t abi_version;          /* must be CUTE_DSL_SHIM_ABI_VERSION */
    uint32_t flags;
    const char *ptxas_path;        /* NULL → /usr/local/cuda/bin/ptxas */
    const char *libnvvm_path;      /* NULL → /usr/local/cuda/nvvm/lib64/libnvvm.so.4 */
};

/* Return codes. Positive = "already done" (idempotent re-call); 0 = success;
 * negative = error. On error cute_dsl_shim_last_error() returns a static
 * string with details.
 */
#define CUTE_DSL_SHIM_OK              0
#define CUTE_DSL_SHIM_ALREADY         1
#define CUTE_DSL_SHIM_E_ABI         (-1)
#define CUTE_DSL_SHIM_E_ARGS        (-2)
#define CUTE_DSL_SHIM_E_LIBNVVM     (-3)  /* dlopen / dlsym failed */
#define CUTE_DSL_SHIM_E_MPROTECT    (-4)
#define CUTE_DSL_SHIM_E_OOM         (-5)

__attribute__((visibility("default")))
int cute_dsl_shim_install(uintptr_t cutlass_ir_base,
                          const struct cute_dsl_shim_offsets *off,
                          const struct cute_dsl_shim_config *cfg);

__attribute__((visibility("default")))
const char *cute_dsl_shim_last_error(void);

__attribute__((visibility("default")))
int cute_dsl_shim_is_active(void);

__attribute__((visibility("default")))
uint32_t cute_dsl_shim_abi_version(void);

/* ============================================================
 *  Logging + error reporting
 * ============================================================ */

static int g_debug = 0;
static pthread_mutex_t g_err_mu = PTHREAD_MUTEX_INITIALIZER;
static char g_err_buf[512] = { 0 };

__attribute__((format(printf, 1, 2)))
static void dbg(const char *fmt, ...) {
    if (!g_debug) return;
    va_list ap;
    va_start(ap, fmt);
    fputs("[cute-dsl-shim] ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

__attribute__((format(printf, 1, 2)))
static void set_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    pthread_mutex_lock(&g_err_mu);
    vsnprintf(g_err_buf, sizeof(g_err_buf), fmt, ap);
    pthread_mutex_unlock(&g_err_mu);
    va_end(ap);
    if (g_debug) fprintf(stderr, "[cute-dsl-shim] ERROR: %s\n", g_err_buf);
}

const char *cute_dsl_shim_last_error(void) {
    return g_err_buf;
}

uint32_t cute_dsl_shim_abi_version(void) {
    return CUTE_DSL_SHIM_ABI_VERSION;
}

/* ============================================================
 *  nvPTXCompiler shim — fork/exec external ptxas
 * ============================================================ */

typedef int nvPTXCompileResult;
typedef void *nvPTXCompilerHandle;

#define NVPTXCOMPILE_SUCCESS                              0
#define NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE        1
#define NVPTXCOMPILE_ERROR_INVALID_INPUT                  2
#define NVPTXCOMPILE_ERROR_COMPILATION_FAILURE            3
#define NVPTXCOMPILE_ERROR_INTERNAL                       4
#define NVPTXCOMPILE_ERROR_OUT_OF_MEMORY                  5
#define NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE 6
#define NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION        7

#define SHIM_HANDLE_MAGIC 0x515541434b505458ULL  /* 'QUACKPTX' */

struct shim_state {
    uint64_t magic;
    char *ptx;
    size_t ptx_size;
    int compile_result;
    void *cubin;
    size_t cubin_size;
    char *err_log;          /* NUL-terminated; err_log_size includes NUL */
    size_t err_log_size;
};

/* PATH_MAX on Linux is 4096; some systems don't expose it from <limits.h>. */
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
static char g_ptxas_bin[PATH_MAX] = "/usr/local/cuda/bin/ptxas";

/* Translate libnvptxcompiler-style options to ptxas CLI form where they
 * differ. Most options are already accepted by ptxas; the ones we know
 * need translation are:
 *   --gpu-name <X>      -> -arch=<X>
 *   --gpu-name=<X>      -> -arch=<X>
 *   --opt-level <N>     -> -O<N>
 *   --opt-level=<N>     -> -O<N>
 *   --compile-only      -> dropped (default for cubin output)
 *   --verbose           -> kept (ptxas accepts -v / --verbose)
 *
 * Anything we don't recognize is passed through verbatim — ptxas's CLI
 * accepts long-form options like --gpu-name= as aliases for -arch=
 * since CUDA 11.x, so verbatim pass-through is mostly fine; the explicit
 * translation here is defensive.
 *
 * Returns a NULL-terminated char** of strdup'd args (caller must free
 * each entry and the array). argv[0] is reserved for the ptxas binary
 * path and left NULL — fill it in before exec.
 */
static char **translate_options(int n_opts, const char *const *opts,
                                int *out_argc)
{
    int cap = n_opts + 8;
    char **argv = (char **)calloc((size_t)cap, sizeof(char *));
    if (!argv) return NULL;
    int argc = 1;  /* slot 0 reserved for ptxas binary */
    for (int i = 0; i < n_opts; i++) {
        const char *o = opts[i];
        if (!o) continue;
        char *translated = NULL;
        if (strcmp(o, "--gpu-name") == 0 && i + 1 < n_opts) {
            if (asprintf(&translated, "-arch=%s", opts[++i]) < 0) translated = NULL;
        } else if (strncmp(o, "--gpu-name=", 11) == 0) {
            if (asprintf(&translated, "-arch=%s", o + 11) < 0) translated = NULL;
        } else if (strcmp(o, "--opt-level") == 0 && i + 1 < n_opts) {
            if (asprintf(&translated, "-O%s", opts[++i]) < 0) translated = NULL;
        } else if (strncmp(o, "--opt-level=", 12) == 0) {
            if (asprintf(&translated, "-O%s", o + 12) < 0) translated = NULL;
        } else if (strcmp(o, "--compile-only") == 0) {
            continue;
        } else {
            translated = strdup(o);
        }
        if (!translated) {
            for (int j = 0; j < argc; j++) free(argv[j]);
            free(argv);
            return NULL;
        }
        if (argc + 1 >= cap) {
            cap *= 2;
            char **nb = (char **)realloc(argv, (size_t)cap * sizeof(char *));
            if (!nb) {
                free(translated);
                for (int j = 0; j < argc; j++) free(argv[j]);
                free(argv);
                return NULL;
            }
            argv = nb;
        }
        argv[argc++] = translated;
    }
    argv[argc] = NULL;
    *out_argc = argc;
    return argv;
}

static void argv_free(char **argv, int argc) {
    if (!argv) return;
    for (int i = 0; i < argc; i++) free(argv[i]);
    free(argv);
}

/* Drain a pipe fd into a malloc'd, NUL-terminated buffer. */
static char *drain_pipe(int fd, size_t *out_len) {
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    while (1) {
        if (len + 1024 + 1 > cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) break;
            buf = nb;
        }
        ssize_t n = read(fd, buf + len, cap - len - 1);
        if (n < 0) { if (errno == EINTR) continue; break; }
        if (n == 0) break;
        len += (size_t)n;
    }
    buf[len] = '\0';
    if (out_len) *out_len = len;
    return buf;
}

/* Read a file fully into a malloc'd buffer. */
static int slurp_file(const char *path, void **out, size_t *out_len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) { close(fd); return -1; }
    void *buf = malloc((size_t)st.st_size ? (size_t)st.st_size : 1);
    if (!buf) { close(fd); return -1; }
    size_t got = 0;
    while (got < (size_t)st.st_size) {
        ssize_t n = read(fd, (char *)buf + got, (size_t)st.st_size - got);
        if (n < 0) { if (errno == EINTR) continue; free(buf); close(fd); return -1; }
        if (n == 0) break;
        got += (size_t)n;
    }
    close(fd);
    *out = buf;
    *out_len = got;
    return 0;
}

/* Spawn external ptxas. Returns 0 on success (cubin/err filled), -1 on
 * internal error (err filled), >0 ptxas exit code (err filled). */
static int run_ptxas(struct shim_state *s, int n_opts, const char *const *opts) {
    /* Race-safe temp files via mkstemps. The 4/6 trailing chars are
     * preserved by mkstemps so we keep the .ptx / .cubin extensions. */
    char ptx_tmpl[]   = "/tmp/cute_dsl_shim_XXXXXX.ptx";
    char cubin_tmpl[] = "/tmp/cute_dsl_shim_XXXXXX.cubin";
    int ptx_fd = mkstemps(ptx_tmpl, 4);
    if (ptx_fd < 0) {
        s->err_log = strdup("mkstemps(ptx) failed\n");
        s->err_log_size = s->err_log ? strlen(s->err_log) + 1 : 0;
        return -1;
    }
    int cubin_fd = mkstemps(cubin_tmpl, 6);
    if (cubin_fd < 0) {
        close(ptx_fd); unlink(ptx_tmpl);
        s->err_log = strdup("mkstemps(cubin) failed\n");
        s->err_log_size = s->err_log ? strlen(s->err_log) + 1 : 0;
        return -1;
    }
    /* We just want the cubin path; ptxas will truncate-write it. */
    close(cubin_fd);
    unlink(cubin_tmpl);

    /* Write PTX */
    const char *p = s->ptx;
    size_t left = s->ptx_size;
    while (left > 0) {
        ssize_t n = write(ptx_fd, p, left);
        if (n < 0) {
            if (errno == EINTR) continue;
            close(ptx_fd); unlink(ptx_tmpl);
            s->err_log = strdup("write(ptx) failed\n");
            s->err_log_size = s->err_log ? strlen(s->err_log) + 1 : 0;
            return -1;
        }
        p += n; left -= (size_t)n;
    }
    close(ptx_fd);

    /* Build argv */
    int argc = 0;
    char **argv = translate_options(n_opts, opts, &argc);
    if (!argv) {
        unlink(ptx_tmpl);
        s->err_log = strdup("translate_options OOM\n");
        s->err_log_size = s->err_log ? strlen(s->err_log) + 1 : 0;
        return -1;
    }
    argv[0] = strdup(g_ptxas_bin);
    /* append -o <cubin> <ptx> */
    char *opt_o    = strdup("-o");
    char *opt_path = strdup(cubin_tmpl);
    char *opt_ptx  = strdup(ptx_tmpl);
    if (!argv[0] || !opt_o || !opt_path || !opt_ptx) {
        free(opt_o); free(opt_path); free(opt_ptx);
        argv_free(argv, argc); unlink(ptx_tmpl);
        s->err_log = strdup("argv OOM\n");
        s->err_log_size = s->err_log ? strlen(s->err_log) + 1 : 0;
        return -1;
    }
    /* Reallocate argv to fit 3 trailing entries. translate_options leaves
     * argc fitting in a cap+ space; we just check. */
    char **nb = (char **)realloc(argv, (size_t)(argc + 4) * sizeof(char *));
    if (!nb) {
        free(opt_o); free(opt_path); free(opt_ptx);
        argv_free(argv, argc); unlink(ptx_tmpl);
        s->err_log = strdup("argv realloc OOM\n");
        s->err_log_size = s->err_log ? strlen(s->err_log) + 1 : 0;
        return -1;
    }
    argv = nb;
    argv[argc++] = opt_o;
    argv[argc++] = opt_path;
    argv[argc++] = opt_ptx;
    argv[argc]   = NULL;

    if (g_debug) {
        /* NVVM 13.3 emits a very long --ext-desc-string=... blob of
         * obfuscated telemetry. Truncate any single argument to 80 chars
         * in the debug log to keep it readable. */
        fputs("[cute-dsl-shim] exec:", stderr);
        for (int i = 0; i < argc; i++) {
            const char *a = argv[i];
            size_t la = strlen(a);
            if (la > 80) fprintf(stderr, " %.80s...[%zub]", a, la);
            else         fprintf(stderr, " %s", a);
        }
        fputc('\n', stderr);
    }

    int err_pipe[2];
    if (pipe(err_pipe) != 0) {
        argv_free(argv, argc); unlink(ptx_tmpl);
        s->err_log = strdup("pipe() failed\n");
        s->err_log_size = s->err_log ? strlen(s->err_log) + 1 : 0;
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(err_pipe[0]); close(err_pipe[1]);
        argv_free(argv, argc); unlink(ptx_tmpl);
        s->err_log = strdup("fork() failed\n");
        s->err_log_size = s->err_log ? strlen(s->err_log) + 1 : 0;
        return -1;
    }
    if (pid == 0) {
        close(err_pipe[0]);
        dup2(err_pipe[1], STDERR_FILENO);
        dup2(err_pipe[1], STDOUT_FILENO);  /* ptxas occasionally prints to stdout */
        close(err_pipe[1]);
        execv(g_ptxas_bin, argv);
        /* exec failed */
        const char msg[] = "execv(ptxas) failed\n";
        ssize_t __r = write(STDERR_FILENO, msg, sizeof(msg) - 1); (void)__r;
        _exit(127);
    }
    close(err_pipe[1]);

    size_t err_len = 0;
    char *err_buf = drain_pipe(err_pipe[0], &err_len);
    close(err_pipe[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    int ec = WIFEXITED(status) ? WEXITSTATUS(status)
           : WIFSIGNALED(status) ? (128 + WTERMSIG(status))
           : -1;

    argv_free(argv, argc);

    s->err_log = err_buf;
    s->err_log_size = err_buf ? (err_len + 1) : 0;

    if (ec == 0) {
        void *cb = NULL; size_t cbn = 0;
        if (slurp_file(cubin_tmpl, &cb, &cbn) != 0) {
            ec = -1;
        } else {
            s->cubin = cb;
            s->cubin_size = cbn;
        }
    }

    unlink(ptx_tmpl);
    unlink(cubin_tmpl);
    return ec;
}

/* ---- 7 nvPTXCompiler shim functions ---- */

__attribute__((visibility("default")))
nvPTXCompileResult shim_nvPTXCompilerCreate(nvPTXCompilerHandle *out,
                                            size_t ptx_len, const char *ptx) {
    if (!out || !ptx) return NVPTXCOMPILE_ERROR_INVALID_INPUT;
    struct shim_state *s = (struct shim_state *)calloc(1, sizeof(*s));
    if (!s) return NVPTXCOMPILE_ERROR_OUT_OF_MEMORY;
    s->magic = SHIM_HANDLE_MAGIC;
    s->ptx = (char *)malloc(ptx_len + 1);
    if (!s->ptx) { free(s); return NVPTXCOMPILE_ERROR_OUT_OF_MEMORY; }
    memcpy(s->ptx, ptx, ptx_len);
    s->ptx[ptx_len] = '\0';
    s->ptx_size = ptx_len;
    *out = s;
    dbg("Create handle=%p ptx_len=%zu", s, ptx_len);
    return NVPTXCOMPILE_SUCCESS;
}

__attribute__((visibility("default")))
nvPTXCompileResult shim_nvPTXCompilerDestroy(nvPTXCompilerHandle *h) {
    if (!h || !*h) return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    struct shim_state *s = (struct shim_state *)*h;
    if (s->magic != SHIM_HANDLE_MAGIC) return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    free(s->ptx); free(s->cubin); free(s->err_log);
    s->magic = 0;
    free(s);
    *h = NULL;
    return NVPTXCOMPILE_SUCCESS;
}

__attribute__((visibility("default")))
nvPTXCompileResult shim_nvPTXCompilerCompile(nvPTXCompilerHandle h,
                                             int n_opts,
                                             const char *const *opts) {
    struct shim_state *s = (struct shim_state *)h;
    if (!s || s->magic != SHIM_HANDLE_MAGIC)
        return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    /* Discard any previous result. */
    free(s->cubin);   s->cubin = NULL;   s->cubin_size = 0;
    free(s->err_log); s->err_log = NULL; s->err_log_size = 0;
    s->compile_result = NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE;

    int ec = run_ptxas(s, n_opts, opts);
    if (ec == 0) {
        s->compile_result = NVPTXCOMPILE_SUCCESS;
        dbg("Compile OK handle=%p cubin_size=%zu", s, s->cubin_size);
        return NVPTXCOMPILE_SUCCESS;
    }
    s->compile_result = NVPTXCOMPILE_ERROR_COMPILATION_FAILURE;
    dbg("Compile FAIL handle=%p exit=%d err=\"%.200s\"",
        s, ec, s->err_log ? s->err_log : "");
    return NVPTXCOMPILE_ERROR_COMPILATION_FAILURE;
}

__attribute__((visibility("default")))
nvPTXCompileResult shim_nvPTXCompilerGetCompiledProgramSize(
    nvPTXCompilerHandle h, size_t *out)
{
    struct shim_state *s = (struct shim_state *)h;
    if (!s || s->magic != SHIM_HANDLE_MAGIC || !out)
        return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    if (s->compile_result != NVPTXCOMPILE_SUCCESS)
        return NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE;
    *out = s->cubin_size;
    return NVPTXCOMPILE_SUCCESS;
}

__attribute__((visibility("default")))
nvPTXCompileResult shim_nvPTXCompilerGetCompiledProgram(
    nvPTXCompilerHandle h, void *buf)
{
    struct shim_state *s = (struct shim_state *)h;
    if (!s || s->magic != SHIM_HANDLE_MAGIC || !buf)
        return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    if (s->compile_result != NVPTXCOMPILE_SUCCESS)
        return NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE;
    memcpy(buf, s->cubin, s->cubin_size);
    return NVPTXCOMPILE_SUCCESS;
}

__attribute__((visibility("default")))
nvPTXCompileResult shim_nvPTXCompilerGetErrorLogSize(
    nvPTXCompilerHandle h, size_t *out)
{
    struct shim_state *s = (struct shim_state *)h;
    if (!s || s->magic != SHIM_HANDLE_MAGIC || !out)
        return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    /* libnvptxcompiler contract: size includes NUL. */
    *out = s->err_log_size;
    return NVPTXCOMPILE_SUCCESS;
}

__attribute__((visibility("default")))
nvPTXCompileResult shim_nvPTXCompilerGetErrorLog(
    nvPTXCompilerHandle h, char *buf)
{
    struct shim_state *s = (struct shim_state *)h;
    if (!s || s->magic != SHIM_HANDLE_MAGIC || !buf)
        return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    if (s->err_log && s->err_log_size > 0) {
        memcpy(buf, s->err_log, s->err_log_size);
    } else {
        buf[0] = '\0';
    }
    return NVPTXCOMPILE_SUCCESS;
}

/* ============================================================
 *  libNVVM dispatch table patch
 * ============================================================ */

/* The cute-DSL libNVVM wrapper builds a 16-slot function-pointer table at
 * first compile. We populate that table ourselves with CUDA 13.3 nvvm*
 * pointers, store the table address into the wrapper's `[guard+8]` global,
 * and set the `[guard]` byte to 1 so the wrapper's pthread_once-like
 * initializer skips rebuilding it.
 *
 * Slot order observed in disassembly of cp312-x86_64 wheel (see DESIGN.md).
 * The order has been stable across the 6 cp310..cp314{,t} wheels we tested.
 *
 * Slot order is validated by end-to-end compilation tests. In particular,
 * nvvmVerifyProgram is at +0x38 and nvvmCompileProgram is at +0x40.
 */
static const char *const NVVM_TABLE_SLOTS[] = {
    "nvvmGetErrorString",         /* 0x00 */
    "nvvmVersion",                /* 0x08 */
    "nvvmIRVersion",              /* 0x10 */
    "nvvmCreateProgram",          /* 0x18 */
    "nvvmDestroyProgram",         /* 0x20 */
    "nvvmAddModuleToProgram",     /* 0x28 */
    "nvvmLazyAddModuleToProgram", /* 0x30 */
    "nvvmVerifyProgram",          /* 0x38 */
    "nvvmCompileProgram",         /* 0x40 */
    "nvvmGetCompiledResultSize",  /* 0x48 */
    "nvvmGetCompiledResult",      /* 0x50 */
    "nvvmGetProgramLogSize",      /* 0x58 */
    "nvvmGetProgramLog",          /* 0x60 */
};
#define NVVM_TABLE_SLOT_COUNT ((int)(sizeof(NVVM_TABLE_SLOTS)/sizeof(*NVVM_TABLE_SLOTS)))

/* The wrapper allocates 16 slots (0x80 bytes). Slots beyond NVVM_TABLE_SLOT_COUNT
 * are unused / zero. We mirror that layout. */
#define NVVM_TABLE_PHYS_SLOTS 16

static void *g_nvvm_lib = NULL;
static void *g_nvvm_table[NVVM_TABLE_PHYS_SLOTS];

/* Load CUDA 13.3 libnvvm and fill g_nvvm_table. Returns 0 on success. */
static int load_nvvm_table(const char *libnvvm_path) {
    g_nvvm_lib = dlopen(libnvvm_path, RTLD_NOW | RTLD_LOCAL);
    if (!g_nvvm_lib) {
        set_err("dlopen(%s) failed: %s", libnvvm_path, dlerror());
        return CUTE_DSL_SHIM_E_LIBNVVM;
    }
    for (int i = 0; i < NVVM_TABLE_SLOT_COUNT; i++) {
        void *p = dlsym(g_nvvm_lib, NVVM_TABLE_SLOTS[i]);
        if (!p) {
            set_err("dlsym(%s) failed in %s", NVVM_TABLE_SLOTS[i], libnvvm_path);
            return CUTE_DSL_SHIM_E_LIBNVVM;
        }
        g_nvvm_table[i] = p;
    }
    for (int i = NVVM_TABLE_SLOT_COUNT; i < NVVM_TABLE_PHYS_SLOTS; i++)
        g_nvvm_table[i] = NULL;
    dbg("loaded %s, %d slots populated", libnvvm_path, NVVM_TABLE_SLOT_COUNT);
    return 0;
}

/* Install the table pointer + guard byte. Caller has validated VAs. */
static int patch_libnvvm(uintptr_t base,
                         uintptr_t guard_va, uintptr_t table_va)
{
    long ps_l = sysconf(_SC_PAGESIZE);
    size_t ps = (size_t)(ps_l > 0 ? ps_l : 4096);

    /* Both writes are 1 and 8 bytes respectively; if they're on the same
     * page, do one mprotect. If on different pages, two. */
    uintptr_t guard_abs = base + guard_va;
    uintptr_t table_abs = base + table_va;
    uintptr_t page_g = guard_abs & ~(uintptr_t)(ps - 1);
    uintptr_t page_t = table_abs & ~(uintptr_t)(ps - 1);

    /* Make both pages RW. */
    if (mprotect((void *)page_g, ps, PROT_READ | PROT_WRITE) != 0) {
        set_err("mprotect(libnvvm guard page %p): %s", (void *)page_g, strerror(errno));
        return CUTE_DSL_SHIM_E_MPROTECT;
    }
    if (page_t != page_g) {
        if (mprotect((void *)page_t, ps, PROT_READ | PROT_WRITE) != 0) {
            mprotect((void *)page_g, ps, PROT_READ);
            set_err("mprotect(libnvvm table page %p): %s", (void *)page_t, strerror(errno));
            return CUTE_DSL_SHIM_E_MPROTECT;
        }
    }

    /* Write table pointer first, then guard. Order matters: if a parallel
     * thread reads guard==1 it must already see the new table ptr. */
    void **table_slot = (void **)table_abs;
    *table_slot = g_nvvm_table;
    __sync_synchronize();
    *(volatile uint8_t *)guard_abs = 1;
    __sync_synchronize();

    /* Restore to read-only (these regions live in .bss/.data; PROT_READ
     * is enough — we don't need executable. Also: the original protection
     * was RW for .bss/.data. Leaving them RW is fine too; we choose
     * PROT_READ | PROT_WRITE to match the original .data perm. */
    if (mprotect((void *)page_g, ps, PROT_READ | PROT_WRITE) != 0) {
        set_err("mprotect(libnvvm guard page restore): %s", strerror(errno));
        return CUTE_DSL_SHIM_E_MPROTECT;
    }
    if (page_t != page_g) {
        if (mprotect((void *)page_t, ps, PROT_READ | PROT_WRITE) != 0) {
            set_err("mprotect(libnvvm table page restore): %s", strerror(errno));
            return CUTE_DSL_SHIM_E_MPROTECT;
        }
    }

    dbg("libnvvm patched: guard@%p=1, table@%p=%p", (void *)guard_abs,
        (void *)table_abs, (void *)g_nvvm_table);
    return 0;
}

/* ============================================================
 *  nvPTXCompiler entry-point trampolines (x86_64)
 * ============================================================ */

/* movabs rax, imm64 ; jmp rax  (12 bytes) */
static void make_trampoline_x86_64(void *target, uint8_t out[12]) {
    out[0] = 0x48; out[1] = 0xb8;
    uint64_t a = (uint64_t)target;
    memcpy(&out[2], &a, 8);
    out[10] = 0xff; out[11] = 0xe0;
}

static int patch_one_entry(uintptr_t target_addr, void *thunk, const char *name) {
    long ps_l = sysconf(_SC_PAGESIZE);
    size_t ps = (size_t)(ps_l > 0 ? ps_l : 4096);
    uintptr_t page = target_addr & ~(uintptr_t)(ps - 1);
    size_t span = ((target_addr + 12 - page) + ps - 1) & ~(ps - 1);

    if (mprotect((void *)page, span, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        set_err("mprotect RWX %s @%p: %s", name, (void *)target_addr, strerror(errno));
        return CUTE_DSL_SHIM_E_MPROTECT;
    }
    uint8_t tramp[12];
    make_trampoline_x86_64(thunk, tramp);
    memcpy((void *)target_addr, tramp, 12);
    if (mprotect((void *)page, span, PROT_READ | PROT_EXEC) != 0) {
        set_err("mprotect RX %s @%p: %s", name, (void *)target_addr, strerror(errno));
        return CUTE_DSL_SHIM_E_MPROTECT;
    }
    dbg("patched %s @%p -> %p", name, (void *)target_addr, thunk);
    return 0;
}

struct entry_spec {
    const char *name;
    uintptr_t va_offset_field;  /* offset of uintptr_t in cute_dsl_shim_offsets */
    void *thunk;
};

#define ENTRY(name_, field_, thunk_)                                  \
    { name_,                                                          \
      offsetof(struct cute_dsl_shim_offsets, field_),                 \
      (void *)(thunk_) }

static int patch_ptxas(uintptr_t base, const struct cute_dsl_shim_offsets *off) {
    const struct entry_spec specs[] = {
        ENTRY("nvPTXCompilerCreate",                 nvptx_create_va,
              shim_nvPTXCompilerCreate),
        ENTRY("nvPTXCompilerCompile",                nvptx_compile_va,
              shim_nvPTXCompilerCompile),
        ENTRY("nvPTXCompilerDestroy",                nvptx_destroy_va,
              shim_nvPTXCompilerDestroy),
        ENTRY("nvPTXCompilerGetCompiledProgramSize", nvptx_get_compiled_program_size_va,
              shim_nvPTXCompilerGetCompiledProgramSize),
        ENTRY("nvPTXCompilerGetCompiledProgram",     nvptx_get_compiled_program_va,
              shim_nvPTXCompilerGetCompiledProgram),
        ENTRY("nvPTXCompilerGetErrorLogSize",        nvptx_get_error_log_size_va,
              shim_nvPTXCompilerGetErrorLogSize),
        ENTRY("nvPTXCompilerGetErrorLog",            nvptx_get_error_log_va,
              shim_nvPTXCompilerGetErrorLog),
    };
    int n = (int)(sizeof(specs) / sizeof(specs[0]));
    for (int i = 0; i < n; i++) {
        uintptr_t va = *(const uintptr_t *)((const char *)off + specs[i].va_offset_field);
        if (va == 0) {
            dbg("skip %s (VA=0)", specs[i].name);
            continue;
        }
        int r = patch_one_entry(base + va, specs[i].thunk, specs[i].name);
        if (r != 0) return r;
    }
    return 0;
}

/* ============================================================
 *  Public install entry point
 * ============================================================ */

static pthread_mutex_t g_install_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_installed = 0;

int cute_dsl_shim_is_active(void) {
    return g_installed;
}

int cute_dsl_shim_install(uintptr_t cutlass_ir_base,
                          const struct cute_dsl_shim_offsets *off,
                          const struct cute_dsl_shim_config *cfg)
{
    if (!off || !cfg) {
        set_err("install: NULL offsets or config");
        return CUTE_DSL_SHIM_E_ARGS;
    }
    if (off->abi_version != CUTE_DSL_SHIM_ABI_VERSION ||
        cfg->abi_version != CUTE_DSL_SHIM_ABI_VERSION) {
        set_err("install: ABI mismatch (caller=%u/%u, shim=%u)",
                off->abi_version, cfg->abi_version, CUTE_DSL_SHIM_ABI_VERSION);
        return CUTE_DSL_SHIM_E_ABI;
    }
    if (cutlass_ir_base == 0) {
        set_err("install: cutlass_ir_base is 0");
        return CUTE_DSL_SHIM_E_ARGS;
    }

    pthread_mutex_lock(&g_install_mu);
    if (g_installed) {
        pthread_mutex_unlock(&g_install_mu);
        return CUTE_DSL_SHIM_ALREADY;
    }

    g_debug = (cfg->flags & CUTE_DSL_SHIM_FLAG_DEBUG) ? 1 : 0;
    if (cfg->ptxas_path && cfg->ptxas_path[0]) {
        snprintf(g_ptxas_bin, sizeof(g_ptxas_bin), "%s", cfg->ptxas_path);
    }

    dbg("install: base=%p flags=0x%x ptxas=%s nvvm=%s",
        (void *)cutlass_ir_base, cfg->flags, g_ptxas_bin,
        cfg->libnvvm_path ? cfg->libnvvm_path : "(default)");

    int do_nvvm  = !(cfg->flags & CUTE_DSL_SHIM_FLAG_SKIP_NVVM);
    int do_ptxas = !(cfg->flags & CUTE_DSL_SHIM_FLAG_SKIP_PTXAS);

    if (do_nvvm) {
        if (off->libnvvm_guard_va == 0 || off->libnvvm_table_va == 0) {
            set_err("install: libnvvm guard/table VA is 0");
            pthread_mutex_unlock(&g_install_mu);
            return CUTE_DSL_SHIM_E_ARGS;
        }
        const char *p = (cfg->libnvvm_path && cfg->libnvvm_path[0])
                          ? cfg->libnvvm_path
                          : "/usr/local/cuda/nvvm/lib64/libnvvm.so.4";
        int r = load_nvvm_table(p);
        if (r != 0) { pthread_mutex_unlock(&g_install_mu); return r; }
        r = patch_libnvvm(cutlass_ir_base, off->libnvvm_guard_va, off->libnvvm_table_va);
        if (r != 0) { pthread_mutex_unlock(&g_install_mu); return r; }
    }

    if (do_ptxas) {
        int r = patch_ptxas(cutlass_ir_base, off);
        if (r != 0) { pthread_mutex_unlock(&g_install_mu); return r; }
    }

    g_installed = 1;
    pthread_mutex_unlock(&g_install_mu);
    g_err_buf[0] = '\0';
    dbg("install: success");
    return CUTE_DSL_SHIM_OK;
}
