"""
CUDA Core Library (CCCL) Python Package
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("cuda-cccl")
except Exception:
    __version__ = "0.0.0"

from .headers.include_paths import get_include_paths

# cuda.bindings is required, but instead of being listed as a required dependency,
# it is installed via an extra (e.g., [cu12] or [cu13]).
#
# One of the first things we should do is check that it is available, and raise
# a helpful error message if it is not.
try:
    import cuda.bindings as _cuda_bindings  # type: ignore
except ImportError:
    raise ImportError(
        "cuda.bindings is not installed. Please install the appropriate extra cuda-cccl[cu12] or cuda-cccl[cu13]."
    ) from None
del _cuda_bindings

__all__ = ["get_include_paths", "__version__"]
