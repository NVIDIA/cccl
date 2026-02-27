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
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import SIGNED_TYPES as TYPE_MAP
from utils import as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import SortOrder, clear_all_caches, make_radix_sort

# Entropy values from C++ benchmark
ENTROPY_VALUES = ["1.000", "0.544", "0.201"]


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
