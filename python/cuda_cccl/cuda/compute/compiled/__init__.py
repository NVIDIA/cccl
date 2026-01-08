# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Pre-compiled (BYOC) components for cuda.compute.

This package contains implementations for users who bring their own compiler
(BYOC) by providing pre-compiled LTOIR rather than using Numba JIT compilation.
"""

from .iterator import CompiledIterator
from .op import CompiledOp

__all__ = ["CompiledIterator", "CompiledOp"]
