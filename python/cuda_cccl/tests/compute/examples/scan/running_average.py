import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    ConstantIterator,
    TransformOutputIterator,
    ZipIterator,
    gpu_struct,
)

# example-begin
"""
Inclusive scan using zip iterator and output transform iterator to compute running average.
"""


@gpu_struct
class SumAndCount:  # data type to store the running sum and the count
    sum: np.float32
    count: np.int32


# binary operation for the scan computes the running sum and running count
def add_op(x1: SumAndCount, x2: SumAndCount) -> SumAndCount:
    return SumAndCount(x1.sum + x2.sum, x1.count + x2.count)


# output transform operation divides the sum by the count to get the running average
def write_op(x: SumAndCount) -> np.float32:
    return x.sum / x.count


# construct a zip iterator to pair the input with the sequence [1, 1, ..., 1]
d_input = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
it_input = ZipIterator(d_input, ConstantIterator(np.int32(1)))

# output transform iterator divides the sum by the count to get the running average
d_output = cp.empty_like(d_input)
it_output = TransformOutputIterator(d_output, write_op)

h_init = SumAndCount(0.0, 0)

cuda.compute.inclusive_scan(it_input, it_output, add_op, h_init, len(d_input))

expected = np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
np.testing.assert_allclose(d_output.get(), expected)
# example-end
