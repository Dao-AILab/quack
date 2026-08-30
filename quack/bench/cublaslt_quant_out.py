# Copyright (c) 2026, Tri Dao.
"""Minimal ctypes cublasLt harness for block-scaled GEMM with QUANTIZED OUTPUT.

cublasLt (CUDA >= 12.8) generates block scale factors for a narrow-precision D
via CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER / _MODE (VEC32_UE8M0 for mxfp8
out, VEC16_UE4M3 for nvfp4 out) — the same SF-generation epilogue quack's
BlockScaleFactorStore implements (and matches bit-exactly). Neither
torch._scaled_mm / F.scaled_mm nor any common wrapper exposes the D-out scale
attributes, hence this direct harness for benchmarking.

Layout mapping (everything row-major on the torch side):
    D(M, N) row-major == cublas col-major D'(N, M):
    call (m'=N, n'=K-reduced .. n'=M, k'=K) with
      A-slot = our B (N, K) row-major == col-major (K, N), opA = T -> (N, K)
      B-slot = our A (M, K) row-major == col-major (K, M), opB = N -> (K, M)
    which is exactly the TN layout fp8/fp4 tensor-core matmuls require.
    Block scales for A/B/D-out all use the tiled layout that is byte-identical
    to quack's blocked (rm, rk, 32, 4, 4) atom, flattened (see
    scale_blocked_for_cublas) — A-slot scale = SFB, B-slot scale = SFA, and the
    D-out scales come back directly comparable with quack's SFD.
"""

import ctypes

import torch

# Values extracted from cublasLt.h / library_types.h (CUDA 13.3).
_DESC_TRANSA = 3
_DESC_TRANSB = 4
_DESC_A_SCALE_POINTER = 17
_DESC_B_SCALE_POINTER = 18
_DESC_A_SCALE_MODE = 31
_DESC_B_SCALE_MODE = 32
_DESC_D_OUT_SCALE_POINTER = 36
_DESC_D_OUT_SCALE_MODE = 37
_OP_N = 0
_OP_T = 1
_COMPUTE_32F = 68
_R_32F = 0
_R_16BF = 14
_R_8F_E4M3 = 28
_R_4F_E2M1 = 33
_SCALE_VEC16_UE4M3 = 1
_SCALE_VEC32_UE8M0 = 2
_PREF_MAX_WORKSPACE_BYTES = 1
_HEURISTIC_RESULT_BYTES = 96  # {algo[64], workspaceSize, state, wavesCount, reserved}


def _load_lib():
    for name in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError("libcublasLt not found")


_lt = None


def _check(status, what):
    if status != 0:
        raise RuntimeError(f"cublasLt {what} failed with status {status}")


def _set_attr_i32(desc, attr, value):
    v = ctypes.c_int32(value)
    _check(
        _lt.cublasLtMatmulDescSetAttribute(desc, attr, ctypes.byref(v), ctypes.sizeof(v)),
        f"DescSetAttribute({attr})",
    )


def _set_attr_ptr(desc, attr, ptr):
    v = ctypes.c_void_p(ptr)
    _check(
        _lt.cublasLtMatmulDescSetAttribute(desc, attr, ctypes.byref(v), ctypes.sizeof(v)),
        f"DescSetAttribute({attr})",
    )


class CublasLtQuantOutGemm:
    """One (M, N, K, format) problem: descriptors + heuristic resolved once,
    ``run()`` is a single cublasLtMatmul call writing (D, SFD).

    fmt: "mxfp8" (fp8 e4m3 data, e8m0 vec-32 scales) or "nvfp4" (packed fp4
    data, e4m3 vec-16 scales). A is (M, K) row-major, B is (N, K) row-major,
    both quantized with blocked (rm/rn, rk, 32, 4, 4) scales. D is (M, N[/2])
    row-major in the same value dtype; SFD comes back in quack's blocked
    layout (byte-identical to cublas's tiled scale layout).
    """

    def __init__(self, qa, sfa, qb, sfb, m, n, k, fmt, device="cuda"):
        global _lt
        if _lt is None:
            _lt = _load_lib()
        self.m, self.n, self.k = m, n, k
        self.qa, self.sfa, self.qb, self.sfb = qa, sfa, qb, sfb
        if fmt == "mxfp8":
            data_type, self.vec, scale_mode = _R_8F_E4M3, 32, _SCALE_VEC32_UE8M0
            self.torch_dtype, self.sf_dtype = torch.float8_e4m3fn, torch.float8_e8m0fnu
            n_stored = n
        elif fmt == "mxfp4":
            data_type, self.vec, scale_mode = _R_4F_E2M1, 32, _SCALE_VEC32_UE8M0
            self.torch_dtype, self.sf_dtype = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
            n_stored = n // 2
        elif fmt == "nvfp4":
            data_type, self.vec, scale_mode = _R_4F_E2M1, 16, _SCALE_VEC16_UE4M3
            self.torch_dtype, self.sf_dtype = torch.float4_e2m1fn_x2, torch.float8_e4m3fn
            n_stored = n // 2
        else:
            raise ValueError(fmt)
        self.D = torch.empty(m, n_stored, dtype=self.torch_dtype, device=device)
        rm, rk = -(-m // 128), -(-n // (4 * self.vec))
        self.SFD = torch.empty(rm, rk, 32, 4, 4, dtype=self.sf_dtype, device=device)

        self._handle = ctypes.c_void_p()
        _check(_lt.cublasLtCreate(ctypes.byref(self._handle)), "Create")
        self._desc = ctypes.c_void_p()
        _check(
            _lt.cublasLtMatmulDescCreate(ctypes.byref(self._desc), _COMPUTE_32F, _R_32F),
            "MatmulDescCreate",
        )
        _set_attr_i32(self._desc, _DESC_TRANSA, _OP_T)
        _set_attr_i32(self._desc, _DESC_TRANSB, _OP_N)
        # All scale pointers must be set BEFORE the heuristic query (a null
        # pointer with a block-scale mode fails with INVALID_VALUE), and an
        # mx-scaled narrow-precision D REQUIRES the D-out scale attributes.
        _set_attr_i32(self._desc, _DESC_A_SCALE_MODE, scale_mode)
        _set_attr_i32(self._desc, _DESC_B_SCALE_MODE, scale_mode)
        _set_attr_i32(self._desc, _DESC_D_OUT_SCALE_MODE, scale_mode)
        _set_attr_ptr(self._desc, _DESC_A_SCALE_POINTER, sfb.data_ptr())
        _set_attr_ptr(self._desc, _DESC_B_SCALE_POINTER, sfa.data_ptr())
        _set_attr_ptr(self._desc, _DESC_D_OUT_SCALE_POINTER, self.SFD.data_ptr())

        def layout(rows, cols, ld, dtype=data_type):
            lay = ctypes.c_void_p()
            _check(
                _lt.cublasLtMatrixLayoutCreate(
                    ctypes.byref(lay),
                    ctypes.c_int32(dtype),
                    ctypes.c_uint64(rows),
                    ctypes.c_uint64(cols),
                    ctypes.c_int64(ld),
                ),
                "MatrixLayoutCreate",
            )
            return lay

        # A-slot = our B: col-major (K, N); B-slot = our A: col-major (K, M);
        # C/D: col-major (N, M) == our row-major (M, N). lds in logical elements.
        # C must be a 16/32-bit type for narrow-precision GEMMs (it's the
        # beta/bias matrix); beta = 0 so it is never read (we pass D's pointer).
        self._layA = layout(k, n, k)
        self._layB = layout(k, m, k)
        self._layC = layout(n, m, n, dtype=_R_16BF)
        self._layD = layout(n, m, n)

        pref = ctypes.c_void_p()
        _check(_lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "PreferenceCreate")
        ws_bytes = ctypes.c_uint64(64 * 1024 * 1024)
        _check(
            _lt.cublasLtMatmulPreferenceSetAttribute(
                pref, _PREF_MAX_WORKSPACE_BYTES, ctypes.byref(ws_bytes), ctypes.sizeof(ws_bytes)
            ),
            "PreferenceSetAttribute",
        )
        self._ws = torch.empty(ws_bytes.value, dtype=torch.uint8, device=device)
        self._result = (ctypes.c_byte * _HEURISTIC_RESULT_BYTES)()
        count = ctypes.c_int32(0)
        _check(
            _lt.cublasLtMatmulAlgoGetHeuristic(
                self._handle,
                self._desc,
                self._layA,
                self._layB,
                self._layC,
                self._layD,
                pref,
                1,
                ctypes.byref(self._result),
                ctypes.byref(count),
            ),
            "AlgoGetHeuristic",
        )
        if count.value == 0:
            raise RuntimeError(f"cublasLt has no algo for this quant-out problem ({fmt})")
        self._alpha = ctypes.c_float(1.0)
        self._beta = ctypes.c_float(0.0)

    def run(self):
        """D, SFD = quantize(A @ B^T) for the bound (qa, sfa), (qb, sfb)."""
        stream = torch.cuda.current_stream().cuda_stream
        _check(
            _lt.cublasLtMatmul(
                self._handle,
                self._desc,
                ctypes.byref(self._alpha),
                ctypes.c_void_p(self.qb.data_ptr()),
                self._layA,
                ctypes.c_void_p(self.qa.data_ptr()),
                self._layB,
                ctypes.byref(self._beta),
                ctypes.c_void_p(0),  # C unused (beta = 0); aliasing D trips validation
                self._layC,
                ctypes.c_void_p(self.D.data_ptr()),
                self._layD,
                ctypes.byref(self._result),  # algo is the first heuristic-result field
                ctypes.c_void_p(self._ws.data_ptr()),
                ctypes.c_uint64(self._ws.numel()),
                ctypes.c_void_p(stream),
            ),
            "Matmul",
        )
        return self.D, self.SFD
