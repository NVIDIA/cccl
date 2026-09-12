import functools

import cupy as cp
import numpy as np

import cuda.compute.algorithms as algorithms

from .strided_iterator import make_ndarray_iterator


def test_strided_iterator():
    # example-begin strided-array

    def add_op(a, b):
        return a + b

    num_items = 253

    # reduce over strided array
    d_arr = cp.arange(3 * num_items + 1, dtype=np.int32)[1::3]
    d_input = make_ndarray_iterator(d_arr, (0,))

    h_init = np.array([0], dtype=np.int32)  # Initial value for the reduction
    d_output = cp.empty(1, dtype=np.int32)  # Storage for output

    # Instantiate reduction, determine storage requirements, and allocate storage
    reduce_into = algorithms.reduce_into(d_input, d_output, add_op, h_init)
    temp_storage_size = reduce_into(None, d_input, d_output, num_items, h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    reduce_into(d_temp_storage, d_input, d_output, num_items, h_init)

    expected_output = functools.reduce(
        lambda a, b: a + b, list(range(1, 3 * num_items + 1, 3))
    )
    assert cp.all(d_output == expected_output)
    # example-end strided-array
