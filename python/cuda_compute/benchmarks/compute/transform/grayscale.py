# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for transform grayscale using cuda.compute.

C++ equivalent: cub/benchmarks/bench/transform/grayscale.cu

Notes:
- Input is an RGB struct with three channels
- Output is grayscale value of the same type
- Benchmark name is "grayscale" to match C++
- Migration: Python uses AoS (`gpu_struct`) to mirror C++ `rgb_t<T>`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
from utils import FLOAT_TYPES as TYPE_MAP
from utils import as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import gpu_struct, make_unary_transform


def bench_transform_grayscale(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements{io}"))

    # Grayscale weights
    w_r = dtype(0.2989)
    w_g = dtype(0.587)
    w_b = dtype(0.114)

    RGB = gpu_struct({"r": dtype, "g": dtype, "b": dtype})

    def to_grayscale(pixel: RGB):
        return w_r * pixel.r + w_g * pixel.g + w_b * pixel.b

    alloc_stream = as_cupy_stream(state.get_stream())
    try:
        with alloc_stream:
            r_data = generate_data_with_entropy(
                num_elements, dtype, "1.000", alloc_stream
            )
            g_data = generate_data_with_entropy(
                num_elements, dtype, "1.000", alloc_stream
            )
            b_data = generate_data_with_entropy(
                num_elements, dtype, "1.000", alloc_stream
            )
            d_pixels = cp.empty(num_elements, dtype=RGB.dtype)
            d_pixels["r"] = r_data
            d_pixels["g"] = g_data
            d_pixels["b"] = b_data
            d_out = cp.empty(num_elements, dtype=dtype)
    except (MemoryError, cp.cuda.memory.OutOfMemoryError):
        state.skip("Skipping: out of memory.")
        return

    transformer = make_unary_transform(d_in=d_pixels, d_out=d_out, op=to_grayscale)

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_pixels.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        transformer(
            d_in=d_pixels,
            d_out=d_out,
            op=to_grayscale,
            num_items=num_elements,
            stream=launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_transform_grayscale)
    b.set_name("grayscale")
    b.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 33, 4))
    bench.run_all_benchmarks(sys.argv)
