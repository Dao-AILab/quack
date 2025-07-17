import os

# Disable donated buffer optimization BEFORE importing PyTorch
# This needs to be set before any PyTorch modules are imported
os.environ["TORCH_COMPILE_DONATED_BUFFER"] = "0"

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import triton.testing

from quack.rmsnorm import QuackRMSNorm
from tabulate import tabulate


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):  # Changed to 1e-6 to match QuackRMSNorm
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * torch.rsqrt(torch.tensor(x.shape[-1], dtype=x.dtype))
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed


def benchmark_backward_by_subtraction(
    implementation_name,
    model,
    input_data,
    num_iterations=100,
    warmup_iterations=10,
):
    """Benchmark the backward pass by subtracting forward time from total time.

    This approach is used for all implementations to ensure fair comparison:
    1. Measure forward pass time using do_bench
    2. Measure forward + backward pass time using do_bench
    3. Calculate backward pass time by subtraction
    """
    # Create gradient tensor of the same shape as the output
    grad_output = torch.randn_like(input_data)

    # Clear cache before starting
    torch.cuda.empty_cache()

    # Convert iterations to time in ms
    warmup_ms = 25  # Default warmup time in ms
    rep_ms = 100  # Default repetition time in ms

    # Create clones outside the benchmark functions to avoid including clone time
    input_no_grad = input_data.clone()

    # Create input with gradients outside the benchmark function
    input_with_grad = input_data.clone().requires_grad_(True)

    # Define the forward-only function
    def forward_fn():
        output = model(input_no_grad)
        return output

    # Define the forward+backward function
    def forward_backward_fn():
        # Reset gradients before backward pass
        if input_with_grad.grad is not None:
            input_with_grad.grad = None

        output = model(input_with_grad)
        output.backward(grad_output)
        return input_with_grad.grad

    # Measure forward time
    forward_time_ms = triton.testing.do_bench(
        forward_fn, warmup=warmup_ms, rep=rep_ms, return_mode="mean"
    )

    # Measure total time (forward + backward)
    # Use grad_to_none to set grad to None instead of zeroing it out
    total_time_ms = triton.testing.do_bench(
        forward_backward_fn,
        warmup=warmup_ms,
        rep=rep_ms,
        return_mode="mean",
        grad_to_none=[input_with_grad],  # Set grad to None instead of zeroing
    )

    # Calculate backward time by subtraction
    backward_time_ms = total_time_ms - forward_time_ms

    # Print the timing results
    print(f"{implementation_name} forward time: {forward_time_ms:.4f} ms")
    print(f"{implementation_name} total time: {total_time_ms:.4f} ms")
    print(f"{implementation_name} backward time (subtraction): {backward_time_ms:.4f} ms")

    # Now measure memory usage for backward pass only
    torch.cuda.empty_cache()

    # Create a fresh input tensor with gradients enabled
    fresh_input = input_data.clone().requires_grad_(True)

    # First, run forward pass with gradient tracking
    output = model(fresh_input)

    # Synchronize to ensure forward pass is complete
    torch.cuda.synchronize()

    # Measure memory after forward pass
    torch.cuda.reset_peak_memory_stats()
    forward_mem = torch.cuda.memory_allocated()

    # Then run backward pass
    output.backward(grad_output)

    # Synchronize to ensure backward pass is complete
    torch.cuda.synchronize()

    # Record peak memory usage during backward pass only
    peak_mem = torch.cuda.max_memory_allocated()
    peak_mem_mb = (peak_mem - forward_mem) / (1024 * 1024)  # Convert to MB

    print(f"{implementation_name} backward-only memory: {peak_mem_mb:.2f} MB")

    # Calculate total benchmark time
    total_benchmark_time_ms = backward_time_ms * num_iterations

    return {
        "implementation": implementation_name,
        "avg_time_ms": backward_time_ms,  # Use the backward time
        "total_time_ms": total_benchmark_time_ms,
        "peak_mem_mb": peak_mem_mb,
    }


def benchmark_backward_implementation(
    implementation_name,
    model,
    input_data,
    num_iterations=100,
    warmup_iterations=10,
):
    """Benchmark function that uses the same approach for all implementations."""
    # Use the subtraction-based benchmark for all implementations
    return benchmark_backward_by_subtraction(
        implementation_name,
        model,
        input_data,
        num_iterations,
        warmup_iterations,
    )


def benchmark_rmsnorm_backward_cuda(
    input_shape,
    normalized_dim,
    num_iterations=100,
    warmup_iterations=10,
    dtype=torch.bfloat16,
):
    """Run backward pass benchmarks for different RMSNorm implementations and return results."""
    input_data = torch.randn(input_shape, device="cuda", dtype=dtype)
    results = []

    # Use the same input data for all implementations for fair comparison
    # This ensures all implementations are tested on exactly the same data
    input_data_shared = torch.randn(input_shape, device="cuda", dtype=dtype)

    # Make copies to avoid any potential memory sharing issues
    input_data_pytorch = input_data_shared.clone()
    input_data_torchcompile = input_data_shared.clone()
    input_data_quack = input_data_shared.clone()

    # Ensure all operations are completed before benchmarking
    torch.cuda.synchronize()

    # Benchmark PyTorch RMSNorm
    print("Benchmarking PyTorch RMSNorm...")
    rms_norm_layer = torch.nn.RMSNorm(normalized_dim, device="cuda", dtype=dtype)
    result = benchmark_backward_implementation(
        "PyTorch RMSNorm",
        rms_norm_layer,
        input_data_pytorch,
        num_iterations,
        warmup_iterations,
    )
    results.append(result)

    # Benchmark TorchCompile RMSNorm
    print("Benchmarking TorchCompile RMSNorm...")
    # Create and compile the model
    compiled_rms_norm = torch.compile(RMSNorm(dim=normalized_dim)).cuda().to(dtype)

    # Use the same benchmark function for all implementations
    result = benchmark_backward_implementation(
        "TorchCompile RMSNorm",
        compiled_rms_norm,
        input_data_torchcompile,
        num_iterations,
        warmup_iterations,
    )
    results.append(result)

    # Benchmark QuackRMSNorm
    print("Benchmarking Quack RMSNorm...")
    quack_rms_norm = QuackRMSNorm(dim=normalized_dim).cuda().to(dtype)
    result = benchmark_backward_implementation(
        "Quack RMSNorm",
        quack_rms_norm,
        input_data_quack,
        num_iterations,
        warmup_iterations,
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
    print("RMSNorm Backward Pass Benchmark Results")
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
    plt.title("RMSNorm Backward Pass Implementation Speedup Comparison")
    plt.xticks(x, config_labels, rotation=45, ha="right")

    # Determine appropriate y-axis tick intervals based on max value
    max_speedup = max(max(quack_vs_pytorch), max(quack_vs_torchcompile))
    y_max = max(2.0, np.ceil(max_speedup * 1.1))  # At least 2.0 or 10% above max

    # Choose appropriate tick interval based on the maximum value
    if y_max <= 5:
        tick_interval = 0.5
    elif y_max <= 10:
        tick_interval = 1.0
    elif y_max <= 20:
        tick_interval = 2.0
    else:
        tick_interval = 5.0

    plt.yticks(np.arange(0, y_max + tick_interval, tick_interval))
    plt.grid(axis="y", linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.legend()

    # Save the figure
    output_path = os.path.join(visual_output_dir, "rmsnorm_backward_speedup_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSpeedup graph saved to: {output_path}")

    # Quack vs PyTorch comparison
    plt.figure(figsize=(14, 6))
    plt.bar(x, quack_vs_pytorch, color="blue", alpha=0.7)
    plt.axhline(y=1.0, color="r", linestyle="-", alpha=0.3)
    plt.xlabel("Configuration (Batch Size, Sequence Length)")
    plt.ylabel("Speedup Factor (higher is better)")
    plt.title("Quack RMSNorm vs PyTorch RMSNorm Backward Pass Speedup")
    plt.xticks(x, config_labels, rotation=45, ha="right")

    # Determine appropriate y-axis tick intervals based on max value
    max_speedup = max(quack_vs_pytorch)
    y_max = max(2.0, np.ceil(max_speedup * 1.1))  # At least 2.0 or 10% above max

    # Choose appropriate tick interval based on the maximum value
    if y_max <= 5:
        tick_interval = 0.5
    elif y_max <= 10:
        tick_interval = 1.0
    elif y_max <= 20:
        tick_interval = 2.0
    else:
        tick_interval = 5.0

    plt.yticks(np.arange(0, y_max + tick_interval, tick_interval))
    plt.grid(axis="y", linestyle="-", alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(visual_output_dir, "quack_vs_pytorch_backward_speedup.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Quack vs PyTorch graph saved to: {output_path}")

    # Quack vs TorchCompile comparison
    plt.figure(figsize=(14, 6))
    plt.bar(x, quack_vs_torchcompile, color="green", alpha=0.7)
    plt.axhline(y=1.0, color="r", linestyle="-", alpha=0.3)
    plt.xlabel("Configuration (Batch Size, Sequence Length)")
    plt.ylabel("Speedup Factor (higher is better)")
    plt.title("Quack RMSNorm vs TorchCompile RMSNorm Backward Pass Speedup")
    plt.xticks(x, config_labels, rotation=45, ha="right")

    # Determine appropriate y-axis tick intervals based on max value
    max_speedup = max(quack_vs_torchcompile)
    y_max = max(2.0, np.ceil(max_speedup * 1.1))  # At least 2.0 or 10% above max

    # Choose appropriate tick interval based on the maximum value
    if y_max <= 5:
        tick_interval = 0.5
    elif y_max <= 10:
        tick_interval = 1.0
    elif y_max <= 20:
        tick_interval = 2.0
    else:
        tick_interval = 5.0

    plt.yticks(np.arange(0, y_max + tick_interval, tick_interval))
    plt.grid(axis="y", linestyle="-", alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(visual_output_dir, "quack_vs_torchcompile_backward_speedup.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Quack vs TorchCompile graph saved to: {output_path}")


if __name__ == "__main__":
    # Define batch sizes and sequence lengths to benchmark
    # TODO - I used llama3-8B which has 4096 hidden dim as fixed hidden dim....let's look at MoE models next
    batch_sizes = [1, 4, 8, 16, 32]
    sequence_lengths = [4096, 8192, 16384, 32768, 65536]
    hidden_features = 4096  # Fixed hidden dimension (based on llama3-8B)
    dtype = torch.bfloat16

    num_benchmark_iterations = 50
    num_warmup_iterations = 20

    print("Running RMSNorm backward pass benchmarks across different sequence lengths...")
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
                # TODO - we should figure out a better way to determine this, ala check GPU memory before running...
                if batch_size * sequence_length * hidden_features > 2**31:
                    print(f"Skipping BS={batch_size}, SeqLen={sequence_length} (too large)")
                    continue

                print(f"\nBenchmarking: BS={batch_size}, SeqLen={sequence_length}...")

                shape = (batch_size, sequence_length, hidden_features)
                norm_dim = hidden_features

                try:
                    results = benchmark_rmsnorm_backward_cuda(
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
        print("\n=== RMSNorm Backward Pass Benchmark Results ===")
        display_results_table(all_results)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted. Displaying partial results...")
        if all_results:
            display_results_table(all_results)
