import time

import torch
import torch.nn as nn

from quack.rmsnorm import _rmsnorm_fwd, QuackRMSNorm, rmsnorm, rmsnorm_ref, rstd_ref
from tabulate import tabulate


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * torch.rsqrt(torch.tensor(x.shape[-1], dtype=x.dtype))
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed


def benchmark_implementation(
    implementation_name,
    model,
    input_data,
    num_iterations=100,
    warmup_iterations=10,
):
    """Benchmark a specific implementation and return timing results."""
    # Warmup
    for _ in range(warmup_iterations):
        _ = model(input_data)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_iterations):
        _ = model(input_data)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / num_iterations

    return {
        "implementation": implementation_name,
        "avg_time_ms": avg_time_ms,
        "total_time_ms": elapsed_time_ms,
    }


def benchmark_rmsnorm_cuda(
    input_shape,
    normalized_dim,
    num_iterations=100,
    warmup_iterations=10,
    dtype=torch.bfloat16,
):
    """Run benchmarks for different RMSNorm implementations and return results."""
    input_data = torch.randn(input_shape, device="cuda", dtype=dtype)
    results = []

    # Benchmark PyTorch RMSNorm
    rms_norm_layer = torch.nn.RMSNorm(normalized_dim, device="cuda", dtype=dtype)
    result = benchmark_implementation(
        "PyTorch RMSNorm", rms_norm_layer, input_data, num_iterations, warmup_iterations
    )
    results.append(result)

    # Benchmark TorchCompile RMSNorm
    compiled_rms_norm = torch.compile(RMSNorm(dim=normalized_dim)).cuda().to(dtype)
    result = benchmark_implementation(
        "TorchCompile RMSNorm",
        compiled_rms_norm,
        input_data,
        num_iterations,
        warmup_iterations,
    )
    results.append(result)

    # Benchmark QuackRMSNorm
    quack_rms_norm = QuackRMSNorm(dim=normalized_dim).cuda().to(dtype)
    result = benchmark_implementation(
        "Quack RMSNorm", quack_rms_norm, input_data, num_iterations, warmup_iterations
    )
    results.append(result)

    return results


def display_results_table(all_results):
    """Display benchmark results in a tabular format with visual grouping."""
    headers = [
        "Batch Size",
        "Seq Length",
        "Hidden Size",
        "Implementation",
        "Avg Time (ms)",
        "Speedup",
    ]

    # Sort configurations for consistent display order
    configs = sorted(all_results.keys())

    # Group by batch size for better visualization
    current_batch = None

    print("\n" + "=" * 80)
    print("RMSNorm Benchmark Results")
    print("=" * 80)

    for config in configs:
        batch_size, seq_len, hidden_size = config
        results = all_results[config]

        # Add visual separation between different batch sizes
        if current_batch != batch_size and current_batch is not None:
            print("\n" + "-" * 80)

        current_batch = batch_size

        # Calculate baseline (PyTorch RMSNorm) time for speedup calculation
        baseline_time = next(
            r["avg_time_ms"]
            for r in results
            if r["implementation"] == "PyTorch RMSNorm"
        )

        # Prepare data for this configuration
        table_data = []
        for result in results:
            speedup = (
                baseline_time / result["avg_time_ms"]
                if result["avg_time_ms"] > 0
                else float("inf")
            )
            table_data.append(
                [
                    batch_size,
                    seq_len,
                    hidden_size,
                    result["implementation"],
                    f"{result['avg_time_ms']:.4f}",
                    f"{speedup:.2f}x",
                ]
            )

        # Print this configuration group
        print(
            f"\nBatch Size: {batch_size}, Sequence Length: {seq_len}, Hidden Size: {hidden_size}"
        )
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Calculate and print the speedup of Quack vs TorchCompile
        torchcompile_time = next(
            r["avg_time_ms"]
            for r in results
            if r["implementation"] == "TorchCompile RMSNorm"
        )
        quack_time = next(
            r["avg_time_ms"] for r in results if r["implementation"] == "Quack RMSNorm"
        )
        quack_vs_torchcompile = (
            torchcompile_time / quack_time if quack_time > 0 else float("inf")
        )
        print(f"Quack vs TorchCompile Speedup: {quack_vs_torchcompile:.2f}x")


if __name__ == "__main__":
    # Define batch sizes and sequence lengths to benchmark
    batch_sizes = [1, 4, 8]
    sequence_lengths = [2048, 4096, 8192, 16384, 32768]
    hidden_features = 4096  # Fixed hidden dimension
    dtype = torch.bfloat16

    num_benchmark_iterations = 50
    num_warmup_iterations = 20

    print(f"Running RMSNorm benchmarks across different sequence lengths...")
    print(f"Hidden dimension: {hidden_features}, Data type: {dtype}")
    print(f"Iterations: {num_benchmark_iterations}, Warmup: {num_warmup_iterations}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run benchmarks.")
        exit(1)

    # Print GPU information
    device_name = torch.cuda.get_device_name(0)
    print(f"Running on GPU: {device_name}")

    # Store all results
    all_results = {}

    try:
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                # Skip very large configurations that might cause OOM
                if batch_size * sequence_length * hidden_features > 2**31:
                    print(
                        f"Skipping BS={batch_size}, SeqLen={sequence_length} (too large)"
                    )
                    continue

                print(f"\nBenchmarking: BS={batch_size}, SeqLen={sequence_length}...")

                shape = (batch_size, sequence_length, hidden_features)
                norm_dim = hidden_features

                try:
                    results = benchmark_rmsnorm_cuda(
                        input_shape=shape,
                        normalized_dim=norm_dim,
                        num_iterations=num_benchmark_iterations,
                        warmup_iterations=num_warmup_iterations,
                        dtype=dtype,
                    )
                    all_results[(batch_size, sequence_length, hidden_features)] = (
                        results
                    )
                except Exception as e:
                    print(
                        f"Error benchmarking BS={batch_size}, SeqLen={sequence_length}: {e}"
                    )

        # Display results in a table
        print("\n=== RMSNorm Benchmark Results ===")
        display_results_table(all_results)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted. Displaying partial results...")
        if all_results:
            display_results_table(all_results)
