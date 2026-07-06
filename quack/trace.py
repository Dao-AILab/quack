"""Compatibility tombstone for the removed QuACK trace API."""

raise ImportError(
    "quack.trace has been removed. QuACK now uses NVIDIA IKET directly; import "
    "`cutlass.cute.experimental.iket` and run workloads under "
    "`python -m iket.cli.main ... profile -- ...`. See "
    "`examples/example_iket_trace.py` for a minimal marker workload and "
    "`examples/example_gemm_trace.py` for a real GEMM trace."
)
