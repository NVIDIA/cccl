# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example demonstrating reductions with nested struct types.

This example shows how to define nested structs using numpy structured dtypes
and use them in reduction operations. The reduction combines values from both
the outer and inner struct fields.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Define nested struct types using numpy structured dtypes
point_dtype = np.dtype([("x", np.int32), ("y", np.int32)], align=True)
particle_dtype = np.dtype([("id", np.int64), ("position", point_dtype)], align=True)


def sum_particles(p1, p2):
    """Reduction operation that sums all fields of two particles.

    Returns a tuple which is implicitly converted to the struct type.
    For nested structs, use nested tuples.
    """
    return (
        p1.id + p2.id,
        (p1.position.x + p2.position.x, p1.position.y + p2.position.y),
    )


# Prepare the input data
num_items = 10
h_data = np.zeros(num_items, dtype=particle_dtype)
for i in range(num_items):
    h_data[i]["id"] = i * 10
    h_data[i]["position"]["x"] = i
    h_data[i]["position"]["y"] = i * 2

# Copy to device
d_input = cp.empty(num_items, dtype=particle_dtype)
d_input.set(h_data)

# Prepare output and initial value using np.void with nested tuples
d_output = cp.empty(1, dtype=particle_dtype)
h_init = np.void((0, (0, 0)), dtype=particle_dtype)

# Perform the reduction
cuda.compute.reduce_into(d_input, d_output, sum_particles, num_items, h_init)

# Verify the result
result = d_output.get()[0]
expected_id = sum(i * 10 for i in range(num_items))
expected_x = sum(range(num_items))
expected_y = sum(i * 2 for i in range(num_items))

assert result["id"] == expected_id
assert result["position"]["x"] == expected_x
assert result["position"]["y"] == expected_y

print("Nested struct reduction result:")
print(f"  id: {result['id']} (expected: {expected_id})")
print(f"  position.x: {result['position']['x']} (expected: {expected_x})")
print(f"  position.y: {result['position']['y']} (expected: {expected_y})")
