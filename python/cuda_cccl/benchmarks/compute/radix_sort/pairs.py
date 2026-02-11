# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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

# Key type mapping: match C++ integral_types
KEY_TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
}

# Value type mapping: match C++ value_types (int8, int16, int32, int64)
VALUE_TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
}

# Entropy values from C++ benchmark (pairs uses fewer values than keys)
ENTROPY_VALUES = ["1.000", "0.201"]

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
    - 0.201: Low entropy (more structure/patterns)
    """
    probability = ENTROPY_TO_PROB[entropy_str]

    with stream:
        info = np.iinfo(dtype)
        if probability == 1.0:
            # Full random across entire range
            data = cp.random.randint(
                int(info.min), int(info.max) + 1, size=num_elements, dtype=np.int64
            ).astype(dtype)
        else:
            # Reduced entropy: limit the range of values
            range_size = int((int(info.max) - int(info.min)) * probability)
            if range_size < 1:
                range_size = 1
            data = cp.random.randint(
                0, range_size, size=num_elements, dtype=np.int64
            ).astype(dtype)

    return data


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
    """
    Benchmark radix_sort pairs (key-value) operation.
    """
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
