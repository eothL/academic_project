"""Matrix transpose problem definition."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from .base import Problem

PROMPT = """
Implement `transpose_matrix(X)` so that it returns the transpose of a matrix X
using JAX primitives (no `np.transpose`). Your function should support JIT,
VMAP over the batch dimension, and reverse-mode autodiff (VJP).
"""

STARTER_CODE = """
import jax.numpy as jnp


def transpose_matrix(X: jnp.ndarray) -> jnp.ndarray:
    \"\"\"Return the transpose of X using JAX ops only.\"\"\"
    raise NotImplementedError("Implement transpose_matrix")
"""

EXAMPLE_SOLUTION = """
import jax.numpy as jnp


def transpose_matrix(X: jnp.ndarray) -> jnp.ndarray:
    if X.ndim != 2:
        raise AssertionError("Expected rank-2 input")
    return jnp.swapaxes(X, -2, -1)
"""


def _run_tests(module) -> None:
    if not hasattr(module, "transpose_matrix"):
        raise AssertionError("Expected transpose_matrix function")
    transpose_matrix = module.transpose_matrix
    if not callable(transpose_matrix):
        raise AssertionError("transpose_matrix must be callable")

    shapes = [(1, 1), (1, 5), (5, 1), (2, 3), (3, 2), (4, 4), (0, 3), (3, 0)]
    for m, n in shapes:
        if m * n == 0:
            X = jnp.empty((m, n), dtype=jnp.float32)
        else:
            X = jnp.arange(m * n, dtype=jnp.float32).reshape(m, n)
        Y = transpose_matrix(X)
        assert Y.shape == (n, m), f"Shape mismatch for {m}x{n}"
        npY = np.transpose(np.array(X))
        assert jnp.allclose(Y, npY)

    key = random.PRNGKey(0)
    X = random.normal(key, (3, 5))
    Y = transpose_matrix(X)
    Z = transpose_matrix(Y)
    assert jnp.allclose(Z, X)

    Xf = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    Xi = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    assert transpose_matrix(Xf).dtype == Xf.dtype
    assert transpose_matrix(Xi).dtype == Xi.dtype

    X = jnp.arange(12).reshape(3, 4)
    f_jit = jax.jit(transpose_matrix)
    assert jnp.allclose(f_jit(X), transpose_matrix(X))

    batched = jnp.arange(2 * 3 * 4).reshape(2, 3, 4)
    f = jax.vmap(transpose_matrix, in_axes=0, out_axes=0)
    Y = f(batched)
    assert Y.shape == (2, 4, 3)
    for b in range(2):
        assert jnp.allclose(Y[b], np.transpose(np.array(batched[b])))

    X = random.normal(key, (2, 3))
    Y, vjp = jax.vjp(transpose_matrix, X)
    G = jnp.arange(Y.size, dtype=Y.dtype).reshape(Y.shape)
    (dX,) = vjp(G)
    assert dX.shape == X.shape
    assert jnp.allclose(dX, jnp.transpose(G))


problem = Problem(
    slug="transpose",
    title="Implement Matrix Transpose in JAX",
    prompt=PROMPT,
    starter_code=STARTER_CODE,
    example_solution=EXAMPLE_SOLUTION,
    test_runner=_run_tests,
)
