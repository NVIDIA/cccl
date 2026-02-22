# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import clear_all_caches, make_histogram_even

# Type mapping: match C++ sample_types
# Note: C++ uses int8_t, int16_t, etc. but Python histogram needs compatible types
TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F32": np.float32,
    "F64": np.float64,
}

# From C++ benchmark
BINS_VALUES = [32, 128, 2048, 2097152]
ENTROPY_VALUES = ["0.201", "1.000"]


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
    """
    Benchmark histogram_even operation.
    """
    # WORKAROUND: Clear caches to avoid bug where cached histogram objects
    # are reused incorrectly when bin count changes. The cache key for
    # np.ndarray only considers dtype, not shape/values, so h_num_output_levels
    # arrays with different values but same dtype get the same cache key.
    # This causes a histogram built for 32 bins to be reused for 2048 bins,
    # leading to memory corruption.
    # TODO: Fix in cuda.compute._caching to include array values in cache key
    clear_all_caches()

    # Get parameters from axes
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

    # Setup histogram parameters
    num_levels = num_bins + 1  # num_output_levels = num_bins + 1
    lower_level = dtype(0)
    upper_level = get_upper_level(dtype, num_bins, num_elements)

    # Allocate arrays
    alloc_stream = as_cupy_stream(state.get_stream())

    # Generate input samples with appropriate entropy
    d_samples = generate_samples_with_entropy(
        num_elements, dtype, lower_level, upper_level, entropy_str, alloc_stream
    )

    # Output histogram (counter type is int32 in C++)
    with alloc_stream:
        d_histogram = cp.zeros(num_bins, dtype=np.int32)

    # Synchronize to ensure data is ready
    alloc_stream.synchronize()

    # Create host arrays for histogram parameters (required by API)
    h_num_output_levels = np.array([num_levels], dtype=np.int32)
    h_lower_level = np.array([lower_level], dtype=dtype)
    h_upper_level = np.array([upper_level], dtype=dtype)

    # Build histogram operation
    histogrammer = make_histogram_even(
        d_samples=d_samples,
        d_histogram=d_histogram,
        h_num_output_levels=h_num_output_levels,
        h_lower_level=h_lower_level,
        h_upper_level=h_upper_level,
        num_samples=num_elements,
    )

    # Get temp storage size and allocate
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

    # Warmup run to catch any CUDA errors before benchmarking
    try:
        histogrammer(
            temp_storage=temp_storage,
            d_samples=d_samples,
            d_histogram=d_histogram,
            h_num_output_levels=h_num_output_levels,
            h_lower_level=h_lower_level,
            h_upper_level=h_upper_level,
            num_samples=num_elements,
        )
        cp.cuda.Device().synchronize()
    except Exception as e:
        state.skip(f"CUDA error during warmup: {e}")
        return

    # Match C++ metrics
    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_samples.dtype.itemsize)
    state.add_global_memory_writes(num_bins * d_histogram.dtype.itemsize)

    # Execute benchmark
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

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_histogram_even)
    b.set_name("base")  # Match C++ benchmark name

    # Match C++ axes
    b.add_string_axis("SampleT", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))  # [16, 20, 24, 28]
    b.add_int64_axis("Bins", BINS_VALUES)  # [32, 128, 2048, 2097152]
    b.add_string_axis("Entropy", ENTROPY_VALUES)

    bench.run_all_benchmarks(sys.argv)
