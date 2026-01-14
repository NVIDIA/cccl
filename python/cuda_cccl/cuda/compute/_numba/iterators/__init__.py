# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba-based iterator implementations for cuda.compute.

This package contains all Numba-dependent iterator code, organized as:
- base.py: IteratorBase and IteratorKind base classes
- simple.py: Simple iterators (RawPointer, Constant, Counting, Discard, Reverse)
- transform.py: TransformIterator
- permutation.py: PermutationIterator
- zip.py: ZipIterator
"""

from .base import IteratorBase, IteratorKind, cached_compile
from .permutation import make_permutation_iterator
from .simple import (
    CacheModifiedPointer,
    ConstantIterator,
    CountingIterator,
    DiscardIterator,
    RawPointer,
    make_reverse_iterator,
    pointer,
)
from .transform import make_transform_iterator
from .zip import make_zip_iterator

__all__ = [
    # Base classes
    "IteratorBase",
    "IteratorKind",
    "cached_compile",
    # Simple iterators
    "RawPointer",
    "pointer",
    "CacheModifiedPointer",
    "ConstantIterator",
    "CountingIterator",
    "DiscardIterator",
    # Iterator factories
    "make_reverse_iterator",
    "make_transform_iterator",
    "make_permutation_iterator",
    "make_zip_iterator",
]
