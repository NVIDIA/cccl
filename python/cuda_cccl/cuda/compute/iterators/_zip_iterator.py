# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
ZipIterator shim - re-exports from _numba/iterators/zip.py.

Note: This module requires Numba to be installed.
"""

from .._numba.iterators.zip import make_zip_iterator

__all__ = ["make_zip_iterator"]
