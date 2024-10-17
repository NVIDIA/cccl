# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
import cupy as cp
import numpy as np
import cuda.parallel.experimental as cudax
# example-end imports

import pytest


def test_device_reduce():
    # example-begin reduce-min
    def min_op(a, b):
        return a if a < b else b

    dtype = np.int32
    h_init = np.array([42], dtype=dtype)
    d_input = cp.array([8, 6, 7, 5, 3, 0, 9], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Instantiate reduction for the given operator and initial value
    reduce_into = cudax.reduce_into(d_output, d_output, min_op, h_init)

    # Determine temporary device storage requirements
    temp_storage_size = reduce_into(None, d_input, d_output, h_init)

    # Allocate temporary storage
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    reduce_into(d_temp_storage, d_input, d_output, h_init)

    # Check the result is correct
    expected_output = 0
    assert (d_output == expected_output).all()
    # example-end reduce-min
