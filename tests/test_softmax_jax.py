import pytest


@pytest.fixture(scope="module")
def jax_env():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    softmax_mod = pytest.importorskip("quack.softmax_jax")
    return jax, jnp, softmax_mod


def _gpu_device(jax):
    try:
        return jax.devices("gpu")[0]
    except Exception:
        pytest.skip("JAX GPU backend is not available")


TOLERANCES = {
    "bfloat16": (1e-2, 1e-2),
    "float16": (1e-3, 1e-3),
    "float32": (1e-4, 1e-4),
}


def _make_inputs(jax, jnp, dtype_name):
    device = _gpu_device(jax)
    pytest.importorskip("jax_tvm_ffi")
    dtype = getattr(jnp, dtype_name)
    M, N = 7, 256
    key_x, key_dy = jax.random.split(jax.random.PRNGKey(0))
    x = 0.1 * jax.random.normal(key_x, (M, N), dtype=jnp.float32)
    dy = jax.random.normal(key_dy, (M, N), dtype=jnp.float32)
    x = jax.device_put(x.astype(dtype), device)
    dy = jax.device_put(dy.astype(dtype), device)
    return x, dy


def _check_forward(jax, jnp, x, out, dtype_name):
    ref = jax.nn.softmax(x.astype(jnp.float32), axis=-1).astype(x.dtype)
    atol, rtol = TOLERANCES[dtype_name]
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert jnp.allclose(out, ref, atol=atol, rtol=rtol)


def _check_backward(jax, jnp, x, dy, dx, dtype_name):
    ref_loss = lambda z: jnp.sum(
        jax.nn.softmax(z.astype(jnp.float32), axis=-1) * dy.astype(jnp.float32)
    )
    dx_ref = jax.grad(ref_loss)(x)
    atol, rtol = TOLERANCES[dtype_name]
    assert dx.shape == x.shape
    assert dx.dtype == x.dtype
    assert jnp.allclose(dx, dx_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize("use_jit", [False, True])
def test_jax_softmax(jax_env, dtype_name, use_jit):
    jax, jnp, softmax_mod = jax_env
    x, dy = _make_inputs(jax, jnp, dtype_name)

    fn = softmax_mod.softmax
    fn = jax.jit(fn) if use_jit else fn
    out = fn(x)
    _check_forward(jax, jnp, x, out, dtype_name)

    def loss(z):
        return jnp.sum(fn(z).astype(jnp.float32) * dy.astype(jnp.float32))

    dx = jax.grad(loss)(x)
    _check_backward(jax, jnp, x, dy, dx, dtype_name)
