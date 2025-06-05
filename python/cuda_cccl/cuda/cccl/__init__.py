"""
CUDA Core Library (CCCL) Python Package
"""

from ._version import __version__
from .headers.include_paths import get_include_paths

__all__ = ["get_include_paths", "__version__"]
