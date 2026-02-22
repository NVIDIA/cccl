# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for transform grayscale using cuda.compute.

C++ equivalent: cub/benchmarks/bench/transform/grayscale.cu

Notes:
- Input is an RGB struct with three channels
- Output is grayscale value of the same type
- Benchmark name is "grayscale" to match C++
- Migration: Python builds RGB data on host and copies to device.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import gpu_struct, make_unary_transform

# Type mapping: C++ uses float and double
TYPE_MAP = {
    "F32": np.float32,
    "F64": np.float64,
}


def bench_transform_grayscale(state: bench.State):
    # Axes
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements"))

    # Define RGB struct for this dtype
    RGB = gpu_struct({"r": dtype, "g": dtype, "b": dtype}, name=f"RGB_{type_str}")

    # Grayscale weights
    w_r = dtype(0.2989)
    w_g = dtype(0.587)
    w_b = dtype(0.114)

    def to_grayscale(pixel):
        return w_r * pixel.r + w_g * pixel.g + w_b * pixel.b

    # Generate input data on host
    h_in = np.empty(num_elements, dtype=RGB.dtype)
    h_in["r"] = np.random.random(num_elements).astype(dtype)
    h_in["g"] = np.random.random(num_elements).astype(dtype)
    h_in["b"] = np.random.random(num_elements).astype(dtype)

    # Allocate device arrays and copy
    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        d_in = cp.empty_like(h_in)
        d_out = cp.empty(num_elements, dtype=dtype)

        cp.cuda.runtime.memcpy(
            d_in.data.ptr,
            h_in.__array_interface__["data"][0],
            h_in.nbytes,
            cp.cuda.runtime.memcpyHostToDevice,
        )

    transformer = make_unary_transform(d_in=d_in, d_out=d_out, op=to_grayscale)

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        transformer(
            d_in=d_in,
            d_out=d_out,
            num_items=num_elements,
            op=to_grayscale,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_transform_grayscale)
    b.set_name("grayscale")  # Match C++ benchmark name
    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 33, 4))
    bench.run_all_benchmarks(sys.argv)
