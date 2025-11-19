# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example demonstrating binary_transform with custom struct types.

When working with struct inputs in transform operations, you need to provide
type annotations to help Numba infer the correct types. Unlike reduce_into
which can infer types from h_init, transform operations require explicit
annotations when using struct inputs.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import gpu_struct


@gpu_struct
class Point2D:
    x: np.float32
    y: np.float32


def add_points(p1: Point2D, p2: Point2D) -> Point2D:
    return Point2D(p1.x + p2.x, p1.y + p2.y)


num_items = 1000

h_in1 = np.empty(num_items, dtype=Point2D.dtype)
h_in1["x"] = np.random.rand(num_items).astype(np.float32)
h_in1["y"] = np.random.rand(num_items).astype(np.float32)

h_in2 = np.empty(num_items, dtype=Point2D.dtype)
h_in2["x"] = np.random.rand(num_items).astype(np.float32)
h_in2["y"] = np.random.rand(num_items).astype(np.float32)

d_in1 = cp.empty_like(h_in1)
d_in1.set(h_in1)

d_in2 = cp.empty_like(h_in2)
d_in2.set(h_in2)

d_out = cp.empty_like(d_in1)

cuda.compute.binary_transform(d_in1, d_in2, d_out, add_points, num_items)

result = d_out.get()

np.testing.assert_allclose(result["x"], h_in1["x"] + h_in2["x"], rtol=1e-5)
np.testing.assert_allclose(result["y"], h_in1["y"] + h_in2["y"], rtol=1e-5)

print("Binary transform with structs completed successfully")
print(f"First result point: x={result[0]['x']:.4f}, y={result[0]['y']:.4f}")
