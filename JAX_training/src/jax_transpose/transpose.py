from typing import Any
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array

def transpose_matrix(x: ArrayLike) -> Array:
    """Return the transpose of a 2D JAX array.

    Implement this yourself using JAX building blocks (e.g., jnp.swapaxes,
    jnp.transpose, lax.transpose, or a composition with stack/concatenate).
    Keep it functional: no in-place mutation.

    Rules/assumptions:
    - Input should be rank-2 after conversion with jnp.asarray.
    - Prefer staying on device; avoid converting to Python lists.
    - Preserve dtype.
    - Rectangular matrices only (no ragged lists).

    Replace the placeholder below.
    """
    X = jnp.asarray(x)
    # TODO: implement your transpose here (do not return X unchanged)
    X = jnp.transpose(X, (1,0))
    return X  # placeholder

