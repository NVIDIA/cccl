# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy
import pytest
from numba import cuda

# example-begin imports
import cuda.parallel.experimental as cudax
# example-end imports


# example-begin reduce-min
def min_op(a, b):
    return a if a < b else b
# example-end reduce-min


def test_device_reduce_success():
    dtype = numpy.int32
    h_init = numpy.array([42], dtype)
    h_input = numpy.array([8, 6, 7, 5, 3, 0, 9], dtype)
    d_output = cuda.device_array(1, dtype)
    d_input = cuda.to_device(h_input)

    # Instantiate reduction for the given operator and initial value
    reduce_into = cudax.reduce_into(d_output, d_output, min_op, h_init)

    # Deterrmine temporary device storage requirements
    temp_storage_size = reduce_into(None, d_input, d_output, h_init)

    # Allocate temporary storage
    d_temp_storage = cuda.device_array(temp_storage_size, dtype=numpy.uint8)

    # Run reduction
    reduce_into(d_temp_storage, d_input, d_output, h_init)

    expected_output = 0
    # example-end reduce-min
    h_output = d_output.copy_to_host()
    assert h_output[0] == expected_output


def test_device_reduce_dtype_mismatch():
    dtypes = [numpy.int32, numpy.int64]
    h_inits = [numpy.array([], dt) for dt in dtypes]
    h_inputs = [numpy.array([], dt) for dt in dtypes]
    d_outputs = [cuda.device_array(1, dt) for dt in dtypes]
    d_inputs = [cuda.to_device(h_inp) for h_inp in h_inputs]

    reduce_into = cudax.reduce_into(d_inputs[0], d_outputs[0], min_op, h_inits[0])

    for ix in range(3):
        with pytest.raises(TypeError, match=r"dtype mismatch: __init__=int32, __call__=int64"):
          reduce_into(None, d_inputs[int(ix == 0)], d_outputs[int(ix == 1)], h_inits[int(ix == 2)])
