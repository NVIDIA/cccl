# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Finding the maximum green value in a sequence of pixels using `reduce_into`
with a custom data type defined as a numpy structured dtype.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Define a custom data type to store the pixel values using numpy structured dtype.
pixel_dtype = np.dtype([("r", np.int32), ("g", np.int32), ("b", np.int32)], align=True)


# Define a reduction operation that returns the pixel with the maximum green value.
# Return type is inferred from the output array dtype.
def max_g_value(x, y):
    return x if x.g > y.g else y


# Prepare the input and output arrays.
d_rgb = cp.random.randint(0, 256, (10, 3), dtype=np.int32).view(pixel_dtype)
d_out = cp.empty(1, pixel_dtype)

# Prepare the initial value for the reduction using np.void.
h_init = np.void((0, 0, 0), dtype=pixel_dtype)

# Perform the reduction.
cuda.compute.reduce_into(d_rgb, d_out, max_g_value, d_rgb.size, h_init)

# Calculate the expected result.
h_rgb = d_rgb.get()
expected = h_rgb[h_rgb.view("int32")[:, 1].argmax()]

# Verify the result.
assert expected["g"] == d_out.get()["g"]
result = d_out.get()
print(f"Pixel reduction result: {result}")
