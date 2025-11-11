"""JAX practice playground package."""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - best effort during editable installs
    __version__ = version("jax-transpose")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
