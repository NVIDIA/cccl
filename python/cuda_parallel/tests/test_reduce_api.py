# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy
import pytest
from numba import cuda

# example-begin imports
import cuda.parallel.experimental as cudax
# example-end imports


def test_device_reduce():
    # example-begin reduce-min
    def min_op(a, b):
        return a if a < b else b

    dtype = numpy.int32
    h_init = numpy.array([42], dtype)
    h_input = numpy.array([8, 6, 7, 5, 3, 0, 9], dtype)
    d_output = cuda.device_array(1, dtype)
    d_input = cuda.to_device(h_input)

    # Instantiate reduction for the given operator and initial value
    reduce_into = cudax.reduce_into(d_output, d_output, min_op, h_init)

    # Determine temporary device storage requirements
    temp_storage_size = reduce_into(None, d_input, d_output, h_init)

    # Allocate temporary storage
    d_temp_storage = cuda.device_array(temp_storage_size, dtype=numpy.uint8)

    # Run reduction
    reduce_into(d_temp_storage, d_input, d_output, h_init)

    expected_output = 0
    # example-end reduce-min
    h_output = d_output.copy_to_host()
    assert h_output[0] == expected_output
