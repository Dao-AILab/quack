# 🦆 QuACK: A Quirky Assortment of CuTe Kernels 🦆

Kernels are written in the [CuTe-DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

## Installation

``` bash
# For CUDA 12.9:
pip install quack-kernels

# For CUDA 13.1:
pip install 'quack-kernels[cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Or using uv (faster):
uv pip install 'quack-kernels[cu13]'
```

## Requirements

- H100 or B200/B300 GPU
- CUDA toolkit 12.9+
- Python 3.12

## Kernels 🐥

- 🦆 RMSNorm forward + backward
- 🦆 Softmax forward + backward
- 🦆 Cross entropy forward + backward
- 🦆 Layernorm forward
- 🦆 Hopper gemm + epilogue
- 🦆 Blackwell gemm + epilogue

## Usage

```
from quack import rmsnorm, softmax, cross_entropy
```

## Documentations

[2025-07-10] We have a comprehensive
[blogpost](media/2025-07-10-membound-sol.md) on how to get memory-bound kernels
to speed-of-light, right in the comfort of Python thanks to the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

## Performance

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

See our [blogpost](media/2025-07-10-membound-sol.md) for the details.

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install

# For CUDA 13.1:
pip install 'quack-kernels[dev,cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Or using uv:
uv pip install 'quack-kernels[dev,cu13]'
```
