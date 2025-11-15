# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing tuple syntax for constructing nested gpu_struct types.

When working with nested structs in device functions, you can use tuple syntax
as a convenient shorthand for constructing the nested struct values. This can
make code more concise while maintaining the same functionality.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import gpu_struct


# Define nested structs
@gpu_struct
class Stats:
    count: np.int32
    sum: np.float32


@gpu_struct
class DataPoint:
    value: np.int64
    stats: Stats


def sum_with_tuples(d1, d2):
    """
    Reduction operation using tuple syntax for nested struct construction.

    Instead of writing: Stats(d1.stats.count + d2.stats.count, ...)
    We can use tuple syntax: (d1.stats.count + d2.stats.count, ...)
    """
    return DataPoint(
        d1.value + d2.value,
        # Tuple syntax for constructing the nested Stats struct
        (d1.stats.count + d2.stats.count, d1.stats.sum + d2.stats.sum),
    )


# Prepare the input data
num_items = 10
h_data = np.zeros(num_items, dtype=DataPoint.dtype)
for i in range(num_items):
    h_data[i]["value"] = i * 10
    h_data[i]["stats"]["count"] = 1
    h_data[i]["stats"]["sum"] = float(i)

# Copy to device
d_input = cp.empty(num_items, dtype=DataPoint.dtype)
d_input.set(h_data)

# Prepare output and initial value
d_output = cp.empty(1, dtype=DataPoint.dtype)
h_init = DataPoint(0, Stats(0, 0.0))

# Perform the reduction
cuda.compute.reduce_into(d_input, d_output, sum_with_tuples, num_items, h_init)

# Verify the result
result = d_output.get()[0]
expected_value = sum(i * 10 for i in range(num_items))
expected_count = num_items
expected_sum = sum(float(i) for i in range(num_items))

assert result["value"] == expected_value
assert result["stats"]["count"] == expected_count
assert np.isclose(result["stats"]["sum"], expected_sum)

print("Nested struct with tuple construction result:")
print(f"  value: {result['value']} (expected: {expected_value})")
print(f"  stats.count: {result['stats']['count']} (expected: {expected_count})")
print(f"  stats.sum: {result['stats']['sum']:.2f} (expected: {expected_sum:.2f})")
