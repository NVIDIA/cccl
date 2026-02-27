# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for histogram_even using cuda.compute.

C++ equivalent: cub/benchmarks/bench/histogram/even.cu

Notes:
- The C++ benchmark uses Entropy axis to generate data with different bit distributions
- For Python, we approximate this with random data in the appropriate range
- Entropy 1.0 = uniform random, 0.201 = skewed towards lower values
- Migration: Python approximates entropy distributions and skips some I8/I16 large-bin cases due to CUDA errors.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import SIGNED_TYPES as TYPE_MAP
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import make_histogram_even

# From C++ benchmark


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


def generate_samples_with_entropy(
    num_elements, dtype, lower_level, upper_level, entropy_str, stream
):
    """
    Generate samples with specified entropy level.

    Entropy 1.0 = uniform random distribution
    Entropy 0.201 = skewed distribution (more values near lower_level)
    """
    with stream:
        if entropy_str == "1.000":
            # Uniform random distribution
            if np.issubdtype(dtype, np.integer):
                # Generate integers in range [lower_level, upper_level)
                samples = cp.random.randint(
                    int(lower_level), int(upper_level), size=num_elements, dtype=dtype
                )
            else:
                # Generate floats in range [lower_level, upper_level)
                samples = cp.random.uniform(
                    float(lower_level), float(upper_level), size=num_elements
                ).astype(dtype)
        else:
            # Entropy 0.201 - skewed distribution (power-law like)
            # Generate values biased towards lower end of range
            if np.issubdtype(dtype, np.integer):
                # Use exponential distribution scaled to range, then convert to int
                # This creates more values near lower_level
                scale = float(upper_level - lower_level) / 5.0  # Adjust for skew
                raw = cp.random.exponential(scale=scale, size=num_elements)
                # Clip to range and convert
                raw = cp.clip(
                    raw + float(lower_level), float(lower_level), float(upper_level) - 1
                )
                samples = raw.astype(dtype)
            else:
                # For floats, use exponential distribution
                scale = float(upper_level - lower_level) / 5.0
                raw = cp.random.exponential(scale=scale, size=num_elements)
                samples = cp.clip(
                    raw + float(lower_level), float(lower_level), float(upper_level)
                ).astype(dtype)

    return samples


def bench_histogram_even(state: bench.State):
    type_str = state.get_string("SampleT")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements"))
    num_bins = int(state.get_int64("Bins"))
    entropy_str = state.get_string("Entropy")

    # Skip invalid configurations (like C++ does)
    # For integer types, skip if num_bins > max value representable by SampleT
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        if num_bins > max_val:
            state.skip("Number of bins exceeds what SampleT can represent")
            return

    # Skip problematic configurations that cause cudaErrorIllegalAddress
    # I8/I16 with large bin counts cause memory access issues
    if dtype in (np.int8, np.int16) and num_bins >= 2048:
        state.skip(
            "Small integer types with large bin counts cause illegal memory access"
        )
        return

    num_levels = num_bins + 1  # num_output_levels = num_bins + 1
    lower_level = dtype(0)
    upper_level = get_upper_level(dtype, num_bins, num_elements)

    alloc_stream = as_cupy_stream(state.get_stream())

    d_samples = generate_samples_with_entropy(
        num_elements, dtype, lower_level, upper_level, entropy_str, alloc_stream
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

    b.add_string_axis("SampleT", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))  # [16, 20, 24, 28]
    b.add_int64_axis("Bins", [32, 128, 2048, 2097152])  # [32, 128, 2048, 2097152]
    b.add_string_axis("Entropy", ["0.201", "1.000"])

    bench.run_all_benchmarks(sys.argv)
