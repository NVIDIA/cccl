import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    ConstantIterator,
    TransformOutputIterator,
    ZipIterator,
)

# example-begin
"""
Inclusive scan using zip iterator and output transform iterator to compute running average.

This example shows how to use numpy structured dtypes with type annotations.
"""

# Define struct type using numpy structured dtype
sum_and_count_dtype = np.dtype([("sum", np.float32), ("count", np.int32)], align=True)


# binary operation for the scan computes the running sum and running count
# Type annotations use the numpy dtype; return tuple is implicitly converted
def add_op(x1: sum_and_count_dtype, x2: sum_and_count_dtype) -> sum_and_count_dtype:
    return (x1.sum + x2.sum, x1.count + x2.count)


# output transform operation divides the sum by the count to get the running average
def write_op(x: sum_and_count_dtype) -> np.float32:
    return x.sum / x.count


# construct a zip iterator to pair the input with the sequence [1, 1, ..., 1]
d_input = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
it_input = ZipIterator(d_input, ConstantIterator(np.int32(1)))

# output transform iterator divides the sum by the count to get the running average
d_output = cp.empty_like(d_input)
it_output = TransformOutputIterator(d_output, write_op)

# Create initial value using np.void
h_init = np.void((0.0, 0), dtype=sum_and_count_dtype)

cuda.compute.inclusive_scan(it_input, it_output, add_op, h_init, len(d_input))

expected = np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
np.testing.assert_allclose(d_output.get(), expected)
# example-end
