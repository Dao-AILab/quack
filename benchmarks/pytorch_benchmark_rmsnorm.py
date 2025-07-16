import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import triton.testing

from quack.rmsnorm import QuackRMSNorm
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
    """Benchmark a specific implementation and return timing results using Triton's do_bench."""
    # Convert iterations to time in ms (approximate)
    # We'll use warmup_ms and rep_ms instead of iterations to ensure consistent timing
    warmup_ms = 25  # Default warmup time in ms
    rep_ms = 100  # Default repetition time in ms

    # Define the function to benchmark
    def benchmark_fn():
        return model(input_data)

    # Reset CUDA memory stats before benchmarking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Record starting memory
    start_mem = torch.cuda.memory_allocated()

    # Use Triton's do_bench to benchmark the function
    # This ensures L2 cache is cleared between runs
    avg_time_ms = triton.testing.do_bench(
        benchmark_fn, warmup=warmup_ms, rep=rep_ms, return_mode="mean"
    )

    # Record peak memory usage during benchmarking
    peak_mem = torch.cuda.max_memory_allocated()
    peak_mem_mb = (peak_mem - start_mem) / (1024 * 1024)  # Convert to MB

    # Calculate total time (approximate)
    total_time_ms = avg_time_ms * num_iterations

    return {
        "implementation": implementation_name,
        "avg_time_ms": avg_time_ms,
        "total_time_ms": total_time_ms,
        "peak_mem_mb": peak_mem_mb,
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
        "Peak Mem (MB)",
        "Speedup",
    ]

    # Sort configurations for consistent display order
    configs = sorted(all_results.keys())

    # Group by batch size for better visualization
    current_batch = None

    print("\n" + "=" * 80)
    print("RMSNorm Benchmark Results")
    print("=" * 80)

    # Collect speedup data for graphs
    quack_vs_pytorch_speedups = []
    quack_vs_torchcompile_speedups = []
    config_labels = []

    for config in configs:
        batch_size, seq_len, hidden_size = config
        results = all_results[config]

        # Add visual separation between different batch sizes
        if current_batch != batch_size and current_batch is not None:
            print("\n" + "-" * 80)

        current_batch = batch_size

        # Calculate baseline (PyTorch RMSNorm) time for speedup calculation
        baseline_time = next(
            r["avg_time_ms"] for r in results if r["implementation"] == "PyTorch RMSNorm"
        )

        # Prepare data for this configuration
        table_data = []
        for result in results:
            speedup = (
                baseline_time / result["avg_time_ms"] if result["avg_time_ms"] > 0 else float("inf")
            )
            table_data.append(
                [
                    batch_size,
                    seq_len,
                    hidden_size,
                    result["implementation"],
                    f"{result['avg_time_ms']:.4f}",
                    f"{result['peak_mem_mb']:.2f}",
                    f"{speedup:.2f}x",
                ]
            )

        # Print this configuration group
        print(f"\nBatch Size: {batch_size}, Sequence Length: {seq_len}, Hidden Size: {hidden_size}")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Calculate and print the speedup of Quack vs TorchCompile
        pytorch_time = next(
            r["avg_time_ms"] for r in results if r["implementation"] == "PyTorch RMSNorm"
        )
        torchcompile_time = next(
            r["avg_time_ms"] for r in results if r["implementation"] == "TorchCompile RMSNorm"
        )
        quack_time = next(
            r["avg_time_ms"] for r in results if r["implementation"] == "Quack RMSNorm"
        )

        quack_vs_pytorch = pytorch_time / quack_time if quack_time > 0 else float("inf")
        quack_vs_torchcompile = torchcompile_time / quack_time if quack_time > 0 else float("inf")

        print(f"Quack vs PyTorch Speedup: {quack_vs_pytorch:.2f}x")
        print(f"Quack vs TorchCompile Speedup: {quack_vs_torchcompile:.2f}x")

        # Collect data for graphs
        quack_vs_pytorch_speedups.append(quack_vs_pytorch)
        quack_vs_torchcompile_speedups.append(quack_vs_torchcompile)
        config_labels.append(f"BS={batch_size}, Seq={seq_len}")

    # Collect memory usage data
    pytorch_mem = []
    torchcompile_mem = []
    quack_mem = []

    for config in configs:
        results = all_results[config]
        for result in results:
            if result["implementation"] == "PyTorch RMSNorm":
                pytorch_mem.append(result["peak_mem_mb"])
            elif result["implementation"] == "TorchCompile RMSNorm":
                torchcompile_mem.append(result["peak_mem_mb"])
            elif result["implementation"] == "Quack RMSNorm":
                quack_mem.append(result["peak_mem_mb"])

    # Calculate average memory usage
    avg_pytorch_mem = sum(pytorch_mem) / len(pytorch_mem) if pytorch_mem else 0
    avg_torchcompile_mem = sum(torchcompile_mem) / len(torchcompile_mem) if torchcompile_mem else 0
    avg_quack_mem = sum(quack_mem) / len(quack_mem) if quack_mem else 0

    # Calculate memory savings percentages
    mem_savings_vs_pytorch = (
        ((avg_pytorch_mem - avg_quack_mem) / avg_pytorch_mem * 100) if avg_pytorch_mem > 0 else 0
    )
    mem_savings_vs_torchcompile = (
        ((avg_torchcompile_mem - avg_quack_mem) / avg_torchcompile_mem * 100)
        if avg_torchcompile_mem > 0
        else 0
    )

    # Calculate and print average and median speedups
    avg_quack_vs_pytorch = sum(quack_vs_pytorch_speedups) / len(quack_vs_pytorch_speedups)
    avg_quack_vs_torchcompile = sum(quack_vs_torchcompile_speedups) / len(
        quack_vs_torchcompile_speedups
    )

    # Calculate median speedups
    median_quack_vs_pytorch = sorted(quack_vs_pytorch_speedups)[len(quack_vs_pytorch_speedups) // 2]
    median_quack_vs_torchcompile = sorted(quack_vs_torchcompile_speedups)[
        len(quack_vs_torchcompile_speedups) // 2
    ]

    # If there's an even number of samples, take the average of the two middle values
    if len(quack_vs_pytorch_speedups) % 2 == 0:
        middle = len(quack_vs_pytorch_speedups) // 2
        median_quack_vs_pytorch = (
            sorted(quack_vs_pytorch_speedups)[middle - 1]
            + sorted(quack_vs_pytorch_speedups)[middle]
        ) / 2

    if len(quack_vs_torchcompile_speedups) % 2 == 0:
        middle = len(quack_vs_torchcompile_speedups) // 2
        median_quack_vs_torchcompile = (
            sorted(quack_vs_torchcompile_speedups)[middle - 1]
            + sorted(quack_vs_torchcompile_speedups)[middle]
        ) / 2

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("-" * 80)
    print(f"Average Quack vs PyTorch Speedup: {avg_quack_vs_pytorch:.2f}x across all sizes tested")
    print(
        f"Median Quack vs PyTorch Speedup: {median_quack_vs_pytorch:.2f}x across all sizes tested"
    )
    print(
        f"Average Quack vs TorchCompile Speedup: {avg_quack_vs_torchcompile:.2f}x across all sizes tested"
    )
    print(
        f"Median Quack vs TorchCompile Speedup: {median_quack_vs_torchcompile:.2f}x across all sizes tested"
    )

    print("\n" + "-" * 80)
    print("MEMORY USAGE SUMMARY")
    print("-" * 80)
    print(f"Average PyTorch RMSNorm Memory: {avg_pytorch_mem:.2f} MB")
    print(f"Average TorchCompile RMSNorm Memory: {avg_torchcompile_mem:.2f} MB")
    print(f"Average Quack RMSNorm Memory: {avg_quack_mem:.2f} MB")
    print(f"Memory Savings vs PyTorch: {mem_savings_vs_pytorch:.2f}%")
    print(f"Memory Savings vs TorchCompile: {mem_savings_vs_torchcompile:.2f}%")
    print("=" * 80)

    # Generate and save graphs
    generate_speedup_graphs(
        quack_vs_pytorch_speedups, quack_vs_torchcompile_speedups, config_labels
    )


def generate_speedup_graphs(quack_vs_pytorch, quack_vs_torchcompile, config_labels):
    """Generate and save speedup comparison graphs."""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(__file__))
    visual_output_dir = os.path.join(output_dir, "visual_outputs")
    os.makedirs(visual_output_dir, exist_ok=True)

    plt.figure(figsize=(14, 8))

    # Create x positions for the bars
    x = np.arange(len(config_labels))
    width = 0.35

    # Plot bars
    plt.bar(x - width / 2, quack_vs_pytorch, width, label="Quack vs PyTorch")
    plt.bar(x + width / 2, quack_vs_torchcompile, width, label="Quack vs TorchCompile")

    # Add horizontal line at y=1 (no speedup/slowdown)
    plt.axhline(y=1.0, color="r", linestyle="-", alpha=0.3)

    # Add labels and title
    plt.xlabel("Configuration (Batch Size, Sequence Length)")
    plt.ylabel("Speedup Factor (higher is better)")
    plt.title("RMSNorm Implementation Speedup Comparison")
    plt.xticks(x, config_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()

    # Save the figure
    output_path = os.path.join(visual_output_dir, "rmsnorm_speedup_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSpeedup graph saved to: {output_path}")

    # Create separate graphs for each comparison
    plt.figure(figsize=(14, 6))
    plt.bar(x, quack_vs_pytorch, color="blue", alpha=0.7)
    plt.axhline(y=1.0, color="r", linestyle="-", alpha=0.3)
    plt.xlabel("Configuration (Batch Size, Sequence Length)")
    plt.ylabel("Speedup Factor (higher is better)")
    plt.title("Quack RMSNorm vs PyTorch RMSNorm Speedup")
    plt.xticks(x, config_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    output_path = os.path.join(visual_output_dir, "quack_vs_pytorch_speedup.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Quack vs PyTorch graph saved to: {output_path}")

    plt.figure(figsize=(14, 6))
    plt.bar(x, quack_vs_torchcompile, color="green", alpha=0.7)
    plt.axhline(y=1.0, color="r", linestyle="-", alpha=0.3)
    plt.xlabel("Configuration (Batch Size, Sequence Length)")
    plt.ylabel("Speedup Factor (higher is better)")
    plt.title("Quack RMSNorm vs TorchCompile RMSNorm Speedup")
    plt.xticks(x, config_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    output_path = os.path.join(visual_output_dir, "quack_vs_torchcompile_speedup.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Quack vs TorchCompile graph saved to: {output_path}")


if __name__ == "__main__":
    # Define batch sizes and sequence lengths to benchmark
    batch_sizes = [1, 4, 8, 16, 32]
    sequence_lengths = [8192, 16384, 32768, 65536, 65536 * 2]
    hidden_features = 4096  # Fixed hidden dimension
    dtype = torch.bfloat16

    num_benchmark_iterations = 50
    num_warmup_iterations = 20

    print("Running RMSNorm benchmarks across different sequence lengths...")
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
                    print(f"Skipping BS={batch_size}, SeqLen={sequence_length} (too large)")
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
                    all_results[(batch_size, sequence_length, hidden_features)] = results
                except Exception as e:
                    print(f"Error benchmarking BS={batch_size}, SeqLen={sequence_length}: {e}")

        # Display results in a table
        print("\n=== RMSNorm Benchmark Results ===")
        display_results_table(all_results)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted. Displaying partial results...")
        if all_results:
            display_results_table(all_results)
