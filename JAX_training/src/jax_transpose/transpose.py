"""Simple transpose helper implemented with JAX primitives."""

from __future__ import annotations

import jax.numpy as jnp


def transpose_matrix(X: jnp.ndarray) -> jnp.ndarray:
    """Return the transpose of a rank-2 array using JAX ops only."""
    if X.ndim != 2:
        raise AssertionError("transpose_matrix expects a rank-2 array")
    return jnp.swapaxes(X, -2, -1)
