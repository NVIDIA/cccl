# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Custom types examples demonstrating reduction with custom data types.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def pixel_reduction_example():
    """Demonstrate reduction with custom Pixel struct to find maximum green value."""

    @parallel.gpu_struct
    class Pixel:
        r: np.int32
        g: np.int32
        b: np.int32

    def max_g_value(x, y):
        return x if x.g > y.g else y

    # Create random RGB data
    d_rgb = cp.random.randint(0, 256, (10, 3), dtype=np.int32).view(Pixel.dtype)
    d_out = cp.empty(1, Pixel.dtype)

    h_init = Pixel(0, 0, 0)

    # Run reduction
    parallel.reduce_into(d_rgb, d_out, max_g_value, d_rgb.size, h_init)

    # Verify result
    h_rgb = d_rgb.get()
    expected = h_rgb[h_rgb.view("int32")[:, 1].argmax()]

    assert expected["g"] == d_out.get()["g"]
    print(f"Maximum green value: {d_out.get()['g']}")
    return d_out.get()


def minmax_reduction_example():
    """Demonstrate reduction with MinMax struct to find min and max simultaneously."""

    @parallel.gpu_struct
    class MinMax:
        min_val: np.float64
        max_val: np.float64

    def minmax_op(v1: MinMax, v2: MinMax):
        c_min = min(v1.min_val, v2.min_val)
        c_max = max(v1.max_val, v2.max_val)
        return MinMax(c_min, c_max)

    def transform_op(v):
        av = abs(v)
        return MinMax(av, av)

    nelems = 4096

    d_in = cp.random.randn(nelems)
    # input values must be transformed to MinMax structures
    # in-place to map computation to data-parallel reduction
    # algorithm that requires commutative binary operation
    # with both operands having the same type.
    tr_it = parallel.TransformIterator(d_in, transform_op)

    d_out = cp.empty(tuple(), dtype=MinMax.dtype)

    # initial value set with identity elements of
    # minimum and maximum operators
    h_init = MinMax(np.inf, -np.inf)

    # run the reduction algorithm
    parallel.reduce_into(tr_it, d_out, minmax_op, nelems, h_init)

    # display values computed on the device
    actual = d_out.get()

    h = np.abs(d_in.get())
    expected = np.asarray([(h.min(), h.max())], dtype=MinMax.dtype)

    assert actual == expected
    print(f"Min/Max result: min={actual['min_val']}, max={actual['max_val']}")
    return actual


if __name__ == "__main__":
    print("Running custom types examples...")
    pixel_reduction_example()
    minmax_reduction_example()
    print("All custom types examples completed successfully!")
