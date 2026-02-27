# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for radix_sort pairs (key-value) using cuda.compute.

C++ equivalent: cub/benchmarks/bench/radix_sort/pairs.cu

Notes:
- The C++ benchmark uses Entropy axis to control key data distribution
- Sort order is always ascending (C++ benchmark hardcodes this)
- Keys and values are sorted together (values rearranged by key order)
- begin_bit=0, end_bit=sizeof(KeyT)*8 (full key comparison)
- C++ uses integral_types for keys and int8/16/32/64 for values
- Migration: Python omits int128 values and OffsetT axis.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import INTEGER_TYPES, as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import SortOrder, clear_all_caches, make_radix_sort

# Key and value types: match C++ integral_types / value_types
KEY_TYPE_MAP = INTEGER_TYPES
VALUE_TYPE_MAP = INTEGER_TYPES

# Entropy values from C++ benchmark (pairs uses fewer values than keys)
ENTROPY_VALUES = ["1.000", "0.201"]


def generate_values(num_elements, dtype, stream):
    """
    Generate random values (no entropy control, just random data).
    """
    with stream:
        info = np.iinfo(dtype)
        data = cp.random.randint(
            int(info.min), int(info.max) + 1, size=num_elements, dtype=np.int64
        ).astype(dtype)

    return data


def bench_radix_sort_pairs(state: bench.State):
    # WORKAROUND: Clear caches to avoid caching bug
    # See BUG_REPORT_CACHING.md for details
    clear_all_caches()

    # Get parameters from axes
    key_type_str = state.get_string("KeyT")
    value_type_str = state.get_string("ValueT")
    key_dtype = KEY_TYPE_MAP[key_type_str]
    value_dtype = VALUE_TYPE_MAP[value_type_str]
    num_elements = int(state.get_int64("Elements"))
    entropy_str = state.get_string("Entropy")

    # Allocate arrays
    alloc_stream = as_cupy_stream(state.get_stream())

    # Generate input keys with specified entropy
    d_in_keys = generate_data_with_entropy(
        num_elements, key_dtype, entropy_str, alloc_stream
    )

    # Generate input values (random, no entropy control)
    d_in_values = generate_values(num_elements, value_dtype, alloc_stream)

    # Output arrays
    with alloc_stream:
        d_out_keys = cp.empty(num_elements, dtype=key_dtype)
        d_out_values = cp.empty(num_elements, dtype=value_dtype)

    # Synchronize to ensure data is ready
    alloc_stream.synchronize()

    # Build radix sort operation (keys + values, ascending order)
    sorter = make_radix_sort(
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
        order=SortOrder.ASCENDING,
    )

    # Get temp storage size and allocate
    temp_storage_bytes = sorter(
        temp_storage=None,
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
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
            d_in_values=d_in_values,
            d_out_values=d_out_values,
            num_items=num_elements,
        )
        cp.cuda.Device().synchronize()
    except Exception as e:
        state.skip(f"CUDA error during warmup: {e}")
        return

    # Match C++ metrics: reads and writes for both keys and values
    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in_keys.dtype.itemsize)
    state.add_global_memory_reads(num_elements * d_in_values.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out_keys.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out_values.dtype.itemsize)

    # Execute benchmark
    def launcher(launch: bench.Launch):
        sorter(
            temp_storage=temp_storage,
            d_in_keys=d_in_keys,
            d_out_keys=d_out_keys,
            d_in_values=d_in_values,
            d_out_values=d_out_values,
            num_items=num_elements,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_radix_sort_pairs)
    b.set_name("base")  # Match C++ benchmark name

    # Match C++ axes
    b.add_string_axis("KeyT", list(KEY_TYPE_MAP.keys()))
    b.add_string_axis("ValueT", list(VALUE_TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))  # [16, 20, 24, 28]
    b.add_string_axis("Entropy", ENTROPY_VALUES)
    # Note: OffsetT axis from C++ is not exposed in Python API

    bench.run_all_benchmarks(sys.argv)
