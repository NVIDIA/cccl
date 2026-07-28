# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for histogram_even using cuda.compute.

C++ equivalent: cub/benchmarks/bench/histogram/even.cu

Notes:
- The C++ benchmark uses Entropy axis with nvbench_helper bit entropy generation
- Migration: Python matches the bitwise-AND entropy approach and skips some I8/I16 large-bin cases due to CUDA errors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import FUNDAMENTAL_TYPES as TYPE_MAP
from utils import as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import make_histogram_even


def get_upper_level(dtype, num_bins, num_elements):
    """
    Compute upper level for histogram bins.
    Mirrors C++ get_upper_level() from histogram_common.cuh
    """
    if np.issubdtype(dtype, np.integer):
        # For integer types, upper_level = min(num_bins, max_value_for_type)
        max_val = np.iinfo(dtype).max
        return dtype(min(num_bins, max_val))
    else:
        # For floating point types, upper_level = num_elements
        return dtype(num_elements)


def bench_histogram_even(state: bench.State):
    type_str = state.get_string("SampleT{ct}")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    num_bins = int(state.get_int64("Bins"))
    entropy_str = state.get_string("Entropy")

    # Skip invalid configurations (like C++ does)
    # For integer types, skip if num_bins > max value representable by SampleT
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        if num_bins > max_val:
            state.skip("Number of bins exceeds what SampleT can represent")
            return

    num_levels = num_bins + 1
    lower_level = dtype(0)
    upper_level = get_upper_level(dtype, num_bins, num_elements)

    alloc_stream = as_cupy_stream(state.get_stream())

    d_samples = generate_data_with_entropy(
        num_elements,
        dtype,
        entropy_str,
        alloc_stream,
        min_val=lower_level,
        max_val=upper_level,
    )

    # Output histogram (counter type is int32 in C++)
    with alloc_stream:
        d_histogram = cp.zeros(num_bins, dtype=np.int32)

    alloc_stream.synchronize()

    h_num_output_levels = np.array([num_levels], dtype=np.int32)
    h_lower_level = np.array([lower_level], dtype=dtype)
    h_upper_level = np.array([upper_level], dtype=dtype)

    histogrammer = make_histogram_even(
        d_samples=d_samples,
        d_histogram=d_histogram,
        h_num_output_levels=h_num_output_levels,
        h_lower_level=h_lower_level,
        h_upper_level=h_upper_level,
        num_samples=num_elements,
    )

    temp_storage_bytes = histogrammer(
        temp_storage=None,
        d_samples=d_samples,
        d_histogram=d_histogram,
        h_num_output_levels=h_num_output_levels,
        h_lower_level=h_lower_level,
        h_upper_level=h_upper_level,
        num_samples=num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_samples.dtype.itemsize)
    state.add_global_memory_writes(num_bins * d_histogram.dtype.itemsize)

    def launcher(launch: bench.Launch):
        histogrammer(
            temp_storage=temp_storage,
            d_samples=d_samples,
            d_histogram=d_histogram,
            h_num_output_levels=h_num_output_levels,
            h_lower_level=h_lower_level,
            h_upper_level=h_upper_level,
            num_samples=num_elements,
            stream=launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_histogram_even)
    b.set_name("base")

    b.add_string_axis("SampleT{ct}", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    b.add_int64_axis("Bins", [32, 128, 2048, 2097152])
    b.add_string_axis("Entropy", ["0.201", "1.000"])

    bench.run_all_benchmarks(sys.argv)
