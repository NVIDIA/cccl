# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example demonstrating reductions with nested gpu_struct types.

This example shows how to define nested structs and use them in reduction
operations. The reduction combines values from both the outer and inner
struct fields.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import gpu_struct


# Define an inner struct to hold coordinate data
@gpu_struct
class Point:
    x: np.int32
    y: np.int32


# Define an outer struct that contains the inner struct
@gpu_struct
class Particle:
    id: np.int64
    position: Point


def sum_particles(p1, p2):
    """Reduction operation that sums all fields of two particles."""
    return Particle(
        p1.id + p2.id,
        Point(p1.position.x + p2.position.x, p1.position.y + p2.position.y),
    )


# Prepare the input data
num_items = 10
h_data = np.zeros(num_items, dtype=Particle.dtype)
for i in range(num_items):
    h_data[i]["id"] = i * 10
    h_data[i]["position"]["x"] = i
    h_data[i]["position"]["y"] = i * 2

# Copy to device
d_input = cp.empty(num_items, dtype=Particle.dtype)
d_input.set(h_data)

# Prepare output and initial value
d_output = cp.empty(1, dtype=Particle.dtype)
h_init = Particle(0, Point(0, 0))

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
