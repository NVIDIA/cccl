# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba-based iterator implementations for cuda.compute.

This module re-exports iterator classes from _numba/iterators/ for backwards
compatibility. The actual implementations are in the _numba package.

Note: These iterators require Numba to be installed.
"""

# Re-export all iterator classes from _numba for backwards compatibility
from .._numba.iterators.base import (
    CacheModifiedPointer,
    ConstantIterator,
    CountingIterator,
    DiscardIterator,
    IteratorBase,
    IteratorKind,
    RawPointer,
    cached_compile,
    make_reverse_iterator,
    make_transform_iterator,
    pointer,
)

__all__ = [
    # Base classes
    "IteratorBase",
    "IteratorKind",
    "cached_compile",
    # Iterator implementations
    "RawPointer",
    "pointer",
    "CacheModifiedPointer",
    "ConstantIterator",
    "CountingIterator",
    "DiscardIterator",
    # Iterator factories
    "make_reverse_iterator",
    "make_transform_iterator",
]
