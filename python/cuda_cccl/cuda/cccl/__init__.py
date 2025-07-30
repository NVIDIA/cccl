"""
CUDA Core Library (CCCL) Python Package
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("cuda-cccl")
except Exception:
    __version__ = "0.0.0"

from .headers.include_paths import get_include_paths

__all__ = ["get_include_paths", "__version__"]
