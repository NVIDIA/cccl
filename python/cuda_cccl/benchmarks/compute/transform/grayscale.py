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
from utils import FLOAT_TYPES as TYPE_MAP
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import ZipIterator, make_unary_transform


def bench_transform_grayscale(state: bench.State):
    # Axes
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements"))

    # Grayscale weights
    w_r = dtype(0.2989)
    w_g = dtype(0.587)
    w_b = dtype(0.114)

    def to_grayscale(rgb):
        return w_r * rgb[0] + w_g * rgb[1] + w_b * rgb[2]

    try:
        alloc_stream = as_cupy_stream(state.get_stream())
        with alloc_stream:
            d_r = cp.random.random(num_elements).astype(dtype)
            d_g = cp.random.random(num_elements).astype(dtype)
            d_b = cp.random.random(num_elements).astype(dtype)
            d_out = cp.empty(num_elements, dtype=dtype)

        zip_in = ZipIterator(d_r, d_g, d_b)
        transformer = make_unary_transform(d_in=zip_in, d_out=d_out, op=to_grayscale)

        state.add_element_count(num_elements)
        state.add_global_memory_reads(3 * num_elements * d_r.dtype.itemsize)
        state.add_global_memory_writes(num_elements * d_out.dtype.itemsize)

        def launcher(launch: bench.Launch):
            zip_in_iter = ZipIterator(d_r, d_g, d_b)
            transformer(
                d_in=zip_in_iter,
                d_out=d_out,
                num_items=num_elements,
                op=to_grayscale,
                stream=launch.get_stream(),
            )

        state.exec(launcher)
    except (MemoryError, cp.cuda.memory.OutOfMemoryError):
        state.skip("Skipping: out of memory.")
        return


if __name__ == "__main__":
    b = bench.register(bench_transform_grayscale)
    b.set_name("grayscale")  # Match C++ benchmark name
    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 33, 4))
    bench.run_all_benchmarks(sys.argv)
