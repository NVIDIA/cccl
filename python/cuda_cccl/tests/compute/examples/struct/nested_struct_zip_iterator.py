# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing ZipIterator with nested gpu_struct types.

This example demonstrates combining separate arrays of nested structs using
ZipIterator, then performing a reduction that operates on the combined data.
This is useful when you have related data stored in separate arrays that need
to be processed together.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import ZipIterator, gpu_struct


# Define nested structs for geometric and color data
@gpu_struct
class Point:
    x: np.int32
    y: np.int32


@gpu_struct
class Color:
    r: np.uint8
    g: np.uint8
    b: np.uint8


@gpu_struct
class Pixel:
    position: Point
    color: Color


def sum_pixels(p1, p2):
    """Reduction operation that sums all fields of two pixels."""
    return Pixel(
        Point(p1.position.x + p2.position.x, p1.position.y + p2.position.y),
        Color(
            p1.color.r + p2.color.r, p1.color.g + p2.color.g, p1.color.b + p2.color.b
        ),
    )


# Prepare separate arrays for points and colors
num_items = 100

h_points = np.array([(i, i * 2) for i in range(num_items)], dtype=Point.dtype)
h_colors = np.array(
    [(i % 256, (i * 2) % 256, (i * 3) % 256) for i in range(num_items)],
    dtype=Color.dtype,
)

d_points = cp.empty(num_items, dtype=Point.dtype)
d_points.set(h_points)

d_colors = cp.empty(num_items, dtype=Color.dtype)
d_colors.set(h_colors)

# Create a zip iterator to combine the points and colors
zip_it = ZipIterator(d_points, d_colors)

# Prepare output and initial value
d_output = cp.empty(1, dtype=Pixel.dtype)
h_init = Pixel(Point(0, 0), Color(0, 0, 0))

# Perform the reduction on the zipped data
cuda.compute.reduce_into(zip_it, d_output, sum_pixels, num_items, h_init)

# Verify the result
result = d_output.get()[0]
expected_x = sum(range(num_items))
expected_y = sum(i * 2 for i in range(num_items))
expected_r = sum(i % 256 for i in range(num_items)) % 256
expected_g = sum((i * 2) % 256 for i in range(num_items)) % 256
expected_b = sum((i * 3) % 256 for i in range(num_items)) % 256

assert result["position"]["x"] == expected_x
assert result["position"]["y"] == expected_y
assert result["color"]["r"] == expected_r
assert result["color"]["g"] == expected_g
assert result["color"]["b"] == expected_b

print("Nested struct with ZipIterator result:")
print(f"  position.x: {result['position']['x']} (expected: {expected_x})")
print(f"  position.y: {result['position']['y']} (expected: {expected_y})")
print(f"  color.r: {result['color']['r']} (expected: {expected_r})")
print(f"  color.g: {result['color']['g']} (expected: {expected_g})")
print(f"  color.b: {result['color']['b']} (expected: {expected_b})")
