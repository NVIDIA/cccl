# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
PermutationIterator shim - re-exports from _numba/iterators/permutation.py.

Note: This module requires Numba to be installed.
"""

from .._numba.iterators.permutation import make_permutation_iterator

__all__ = ["make_permutation_iterator"]
