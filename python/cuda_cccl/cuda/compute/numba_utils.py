# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba utility functions shim - re-exports from _numba/numba_utils.py.

Note: This module requires Numba to be installed.
"""

from ._numba.numba_utils import (
    get_inferred_return_type,
    signature_from_annotations,
    to_numba_type,
)

__all__ = [
    "get_inferred_return_type",
    "signature_from_annotations",
    "to_numba_type",
]
