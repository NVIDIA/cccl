# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing tuple syntax for constructing nested struct types.

When working with nested structs in device functions, you use tuple syntax
to construct return values. The tuples are implicitly converted to the
expected struct type. This is the standard approach when using numpy
structured dtypes.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Define nested structs using numpy structured dtypes
stats_dtype = np.dtype([("count", np.int32), ("sum", np.float32)], align=True)
datapoint_dtype = np.dtype([("value", np.int64), ("stats", stats_dtype)], align=True)


def sum_with_tuples(d1, d2):
    """
    Reduction operation using tuple syntax for struct construction.

    Return tuples which are implicitly converted to struct types.
    For nested structs, use nested tuples.
    """
    return (
        d1.value + d2.value,
        # Nested tuple for the stats field
        (d1.stats.count + d2.stats.count, d1.stats.sum + d2.stats.sum),
    )


# Prepare the input data
num_items = 10
h_data = np.zeros(num_items, dtype=datapoint_dtype)
for i in range(num_items):
    h_data[i]["value"] = i * 10
    h_data[i]["stats"]["count"] = 1
    h_data[i]["stats"]["sum"] = float(i)

# Copy to device
d_input = cp.empty(num_items, dtype=datapoint_dtype)
d_input.set(h_data)

# Prepare output and initial value using np.void with nested tuples
d_output = cp.empty(1, dtype=datapoint_dtype)
h_init = np.void((0, (0, 0.0)), dtype=datapoint_dtype)

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
