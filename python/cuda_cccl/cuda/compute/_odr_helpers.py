# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
ODR helpers shim - re-exports from _numba/odr_helpers.py.

Note: This module requires Numba to be installed.
"""

from ._numba.odr_helpers import (
    create_advance_void_ptr_wrapper,
    create_input_dereference_void_ptr_wrapper,
    create_op_void_ptr_wrapper,
    create_output_dereference_void_ptr_wrapper,
)

__all__ = [
    "create_op_void_ptr_wrapper",
    "create_advance_void_ptr_wrapper",
    "create_input_dereference_void_ptr_wrapper",
    "create_output_dereference_void_ptr_wrapper",
]
