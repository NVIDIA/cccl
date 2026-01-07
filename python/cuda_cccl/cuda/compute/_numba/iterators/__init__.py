# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba-based iterator implementations for cuda.compute.

This package contains all Numba-dependent iterator code.
"""

from .base import (
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
from .permutation import make_permutation_iterator
from .zip import make_zip_iterator

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
    "make_permutation_iterator",
    "make_zip_iterator",
]
