import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from jax_transpose.transpose import transpose_matrix


@pytest.mark.parametrize("m,n", [(1, 1), (1, 5), (5, 1), (2, 3), (3, 2), (4, 4), (0, 3), (3, 0)])
def test_shape_and_values_vs_numpy(m, n):
    if m * n == 0:
        X = jnp.empty((m, n), dtype=jnp.float32)
    else:
        X = jnp.arange(m * n, dtype=jnp.float32).reshape(m, n)
    Y = transpose_matrix(X)
    npY = np.transpose(np.array(X))
    assert Y.shape == (n, m)
    assert jnp.allclose(Y, npY)


def test_involution_property():
    key = random.PRNGKey(0)
    X = random.normal(key, (3, 5))
    Y = transpose_matrix(X)
    Z = transpose_matrix(Y)
    assert Z.shape == X.shape
    assert jnp.allclose(Z, X, atol=0, rtol=0)


def test_dtype_preserved():
    Xf = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    Xi = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    assert transpose_matrix(Xf).dtype == Xf.dtype
    assert transpose_matrix(Xi).dtype == Xi.dtype


def test_jit_works_and_matches_eager():
    X = jnp.arange(12).reshape(3, 4)
    f_jit = jax.jit(transpose_matrix)
    Y_eager = transpose_matrix(X)
    Y_jit = f_jit(X)
    assert jnp.allclose(Y_eager, Y_jit)


def test_vmap_over_batch_dim():
    X = jnp.arange(2 * 3 * 4).reshape(2, 3, 4)
    f = jax.vmap(transpose_matrix, in_axes=0, out_axes=0)
    Y = f(X)
    assert Y.shape == (2, 4, 3)
    for b in range(2):
        assert jnp.allclose(Y[b], np.transpose(np.array(X[b])))


def test_vjp_property_of_transpose():
    key = random.PRNGKey(42)
    X = random.normal(key, (2, 3))
    Y, vjp = jax.vjp(transpose_matrix, X)
    G = jnp.arange(Y.size, dtype=Y.dtype).reshape(Y.shape)
    (dX,) = vjp(G)
    assert dX.shape == X.shape
    assert jnp.allclose(dX, jnp.transpose(G))


@pytest.mark.xfail(strict=False, reason="Optional: enforce rectangular 2D input")
def test_reject_ragged_lists_or_wrong_rank():
    ragged = [[1, 2, 3], [4, 5]]
    with pytest.raises(Exception):
        _ = transpose_matrix(ragged)
    with pytest.raises(AssertionError):
        _ = transpose_matrix(jnp.arange(6))
