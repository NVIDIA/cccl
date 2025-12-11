# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for automatic device array capture and Op abstraction.

This tests the core functionality of device array detection and the Op class hierarchy,
not algorithm-specific behavior (those tests belong in their respective test modules).
"""

import inspect

import cupy as cp
import numpy as np
from numba import cuda

from cuda.compute._stateful import maybe_transform_to_stateful


def test_detect_globals_with_device_array():
    """Test detection of device arrays in function globals."""
    global_array = cp.zeros(1, np.int32)

    def func_with_global(x):
        cuda.atomic.add(global_array, 0, 1)
        return x * 2

    transformed_func, state_arrays = maybe_transform_to_stateful(func_with_global)
    assert len(state_arrays) == 1
    assert state_arrays[0] is global_array
    # check number of arguments of transformed function
    assert len(inspect.signature(transformed_func).parameters) == 2


def test_detect_closures_with_device_array():
    """Test detection of device arrays in function closures."""
    closure_array = cp.zeros(1, np.int32)

    def make_func():
        def func_with_closure(x):
            cuda.atomic.add(closure_array, 0, 1)
            return x * 2

        return func_with_closure

    func = make_func()
    transformed_func, state_arrays = maybe_transform_to_stateful(func)
    assert len(state_arrays) == 1
    assert state_arrays[0] is closure_array
    # check number of arguments of transformed function
    assert len(inspect.signature(transformed_func).parameters) == 2


def test_detect_no_device_arrays():
    """Test that functions without device array references return empty."""

    def func(x):
        return x * 2

    transformed_func, state_arrays = maybe_transform_to_stateful(func)
    assert len(state_arrays) == 0
    assert transformed_func is func
