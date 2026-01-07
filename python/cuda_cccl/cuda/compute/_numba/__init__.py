# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba-specific functionality for cuda.compute.

This package contains all Numba-dependent code, allowing the rest of
cuda.compute to be imported without Numba when only using pre-compiled
operations (CompiledOp, CompiledIterator).
"""

# Lazy imports - only import when accessed
__all__ = [
    "interop",
    "odr_helpers",
    "numba_utils",
    "struct",
]
