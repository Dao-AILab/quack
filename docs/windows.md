# Native Windows x64 validation

Windows uses LockFileEx for shared/exclusive cache locks and spawn for async
compile workers. POSIX hosts retain fcntl and forkserver. Windows PyTorch does
not supply Triton, so the Windows package dependency installs triton-windows.

Validated on RTX 5060 Laptop (SM120, 8 GB), Windows 11 x64, driver 610.62,
CUDA toolkit 13.4, Python 3.12.16, CUDA PyTorch 2.14.1+cu130, CuTe-DSL 4.8.1,
apache-tvm-ffi 0.1.15.post2 and triton-windows 3.9.0.post30.

Install CUDA PyTorch from its CUDA wheel index, then install this source checkout
or its regular wheel. Use an x64 Visual Studio developer terminal if native
linking requires compiler tools. Do not force a custom CUTE_DSL_PTXAS_PATH unless
that compiler path has been separately validated.

From the repository directory:

    python -I -m pytest tests/test_windows_filelock.py tests/test_windows_sm120.py tests/test_cache.py -q
    python -I -m pytest tests/test_softmax.py --async-compile=2 -k "test_softmax and False and 192 and 199" -q

For the async command, set QUACK_CACHE_DIR to an empty directory to test cold
worker compilation. Native tests compare FP16/BF16 GEMM, softmax and RMSNorm
outputs and reduction gradients against FP32 references. The async selection
also covers FP32 softmax. Lock tests cover exclusive contention, concurrent
readers, unlock, blocking acquisition and invalid operations.

This is a validated SM120 subset, not certification of all GPU architectures,
quantized formats, distributed operations or every torch.compile configuration.
An isolated regular-wheel installation was also tested with paired FA4 through
two processes: 59 numerical checks each; the second process loads cached x64
COFF objects and rejects new QuACK/FA4 kernel compilation. Tested dependency
files were shared with the development environment; this is not a clean-machine
test of downloading and installing every dependency.
