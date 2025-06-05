"""
CUDA Core Library (CCCL) Python Package
"""

import importlib.metadata

__version__ = importlib.metadata.version("cuda-cccl")

from .headers.include_paths import get_include_paths

__all__ = ["get_include_paths", "__version__"]
