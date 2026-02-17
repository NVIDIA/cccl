# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for radix_sort keys using cuda.compute.

C++ equivalent: cub/benchmarks/bench/radix_sort/keys.cu

Notes:
- The C++ benchmark uses Entropy axis to control data distribution
- Sort order is always ascending (C++ benchmark hardcodes this)
- Keys only (no values) - see radix_sort/pairs.cu for key-value sorting
- begin_bit=0, end_bit=sizeof(T)*8 (full key comparison)
- Migration: Python fixes offsets, excludes int128, and approximates entropy generation.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import SortOrder, clear_all_caches, make_radix_sort

# Type mapping: match C++ fundamental_types (excluding int128)
TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F32": np.float32,
    "F64": np.float64,
}

# Entropy values from C++ benchmark
ENTROPY_VALUES = ["1.000", "0.544", "0.201"]

# Entropy to probability mapping (from nvbench_helper.cuh)
ENTROPY_TO_PROB = {
    "1.000": 1.0,
    "0.811": 0.811,
    "0.544": 0.544,
    "0.337": 0.337,
    "0.201": 0.201,
    "0.000": 0.0,
}


def generate_data_with_entropy(num_elements, dtype, entropy_str, stream):
    """
    Generate data with specified entropy level.

    Entropy controls the bit-level randomness of the data:
    - 1.000: Full random (all bits random)
    - 0.544: Medium entropy
    - 0.201: Low entropy (more structure/patterns)
    """
    probability = ENTROPY_TO_PROB[entropy_str]

    with stream:
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            if probability == 1.0:
                # Full random across entire range
                if dtype == np.int64:
                    data = cp.random.randint(
                        int(info.min),
                        int(info.max),
                        size=num_elements,
                        dtype=np.int64,
                    )
                else:
                    data = cp.random.randint(
                        int(info.min),
                        int(info.max) + 1,
                        size=num_elements,
                        dtype=np.int64,
                    ).astype(dtype)
            else:
                # Reduced entropy: limit the range of values
                # Scale the range based on probability
                range_size = int((int(info.max) - int(info.min)) * probability)
                if range_size < 1:
                    range_size = 1
                if dtype == np.int64:
                    max_high = int(info.max)
                    if range_size > max_high:
                        range_size = max_high
                data = cp.random.randint(
                    0, range_size, size=num_elements, dtype=np.int64
                ).astype(dtype)
        else:
            # Floating point
            info = np.finfo(dtype)
            if probability == 1.0:
                # Full random in [-1, 1] scaled to reasonable range
                data = cp.random.uniform(-1, 1, size=num_elements).astype(dtype)
                # Scale to larger range but avoid inf
                data = data * info.max * 0.5
            else:
                # Reduced entropy: smaller range
                scale = probability * info.max * 0.5
                data = cp.random.uniform(-scale, scale, size=num_elements).astype(dtype)

    return data


def bench_radix_sort_keys(state: bench.State):
    """
    Benchmark radix_sort keys operation.
    """
    # WORKAROUND: Clear caches to avoid caching bug
    # See BUG_REPORT_CACHING.md for details
    clear_all_caches()

    # Get parameters from axes
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements"))
    entropy_str = state.get_string("Entropy")

    # Allocate arrays
    alloc_stream = as_cupy_stream(state.get_stream())

    # Generate input data with specified entropy
    d_in_keys = generate_data_with_entropy(
        num_elements, dtype, entropy_str, alloc_stream
    )

    # Output array for sorted keys
    with alloc_stream:
        d_out_keys = cp.empty(num_elements, dtype=dtype)

    # Synchronize to ensure data is ready
    alloc_stream.synchronize()

    # Build radix sort operation (keys only, ascending order)
    sorter = make_radix_sort(
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=None,
        d_out_values=None,
        order=SortOrder.ASCENDING,
    )

    # Get temp storage size and allocate
    temp_storage_bytes = sorter(
        temp_storage=None,
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=None,
        d_out_values=None,
        num_items=num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    # Warmup run to catch any CUDA errors before benchmarking
    try:
        sorter(
            temp_storage=temp_storage,
            d_in_keys=d_in_keys,
            d_out_keys=d_out_keys,
            d_in_values=None,
            d_out_values=None,
            num_items=num_elements,
        )
        cp.cuda.Device().synchronize()
    except Exception as e:
        state.skip(f"CUDA error during warmup: {e}")
        return

    # Match C++ metrics
    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in_keys.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out_keys.dtype.itemsize)

    # Execute benchmark
    def launcher(launch: bench.Launch):
        sorter(
            temp_storage=temp_storage,
            d_in_keys=d_in_keys,
            d_out_keys=d_out_keys,
            d_in_values=None,
            d_out_values=None,
            num_items=num_elements,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_radix_sort_keys)
    b.set_name("base")  # Match C++ benchmark name

    # Match C++ axes
    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))  # [16, 20, 24, 28]
    b.add_string_axis("Entropy", ENTROPY_VALUES)
    # Note: OffsetT axis from C++ is not exposed in Python API

    bench.run_all_benchmarks(sys.argv)
