# JAX Interface

QuACK ships optional JAX bindings for some of its kernels, layered on top of
[`jax-tvm-ffi`](https://github.com/apache/tvm-ffi). The bindings are opt-in so
users that only need the PyTorch interface do not pay the extra dependency.

## Installation

Install QuACK with the `jax` extra to pull in `jax` and `jax-tvm-ffi`:

```bash
pip install 'quack-kernels[jax]'
```

Or install the dependencies manually:

```bash
pip install jax jax-tvm-ffi
```

## Usage

```python
import jax.numpy as jnp
from quack.softmax_jax import softmax

x = jnp.ones((128, 1024), dtype=jnp.bfloat16)
y = softmax(x)
```

The wrapped function registers a `jax.custom_vjp`, so it composes naturally
with `jax.grad`, `jax.jit`, and friends.

## Adding a new JAX binding

See [`quack/softmax_jax.py`](../quack/softmax_jax.py) for a minimal end-to-end
example: it compiles the forward and backward kernels through
`quack.jax_utils.TvmFfiKernel`, calls them via the JAX FFI, and exposes a
single `softmax` function with a custom VJP. Shared helpers (dtype mapping,
shape checks, lazy target registration) live in
[`quack/jax_utils.py`](../quack/jax_utils.py).
