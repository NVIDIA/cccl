# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for merge_sort keys using cuda.compute.

C++ equivalent: cub/benchmarks/bench/merge_sort/keys.cu

Notes:
- The C++ benchmark uses Entropy axis to control data distribution
- Uses less_t comparison operator (ascending sort)
- Keys only (no values) - see pairs.cu for key-value sorting
- Migration: Python fixes offsets and approximates entropy generation.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import TYPE_MAP, as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import OpKind, clear_all_caches, make_merge_sort

# Entropy values from C++ benchmark (keys.cu line 125)
ENTROPY_VALUES = ["1.000", "0.201"]


def bench_merge_sort_keys(state: bench.State):
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

    # Output array for sorted keys (merge_sort requires separate output)
    with alloc_stream:
        d_out_keys = cp.empty(num_elements, dtype=dtype)

    # Synchronize to ensure data is ready
    alloc_stream.synchronize()

    # Build merge sort operation (keys only, ascending order via OpKind.LESS)
    sorter = make_merge_sort(
        d_in_keys=d_in_keys,
        d_in_items=None,
        d_out_keys=d_out_keys,
        d_out_items=None,
        op=OpKind.LESS,
    )

    # Get temp storage size and allocate
    temp_storage_bytes = sorter(
        temp_storage=None,
        d_in_keys=d_in_keys,
        d_in_items=None,
        d_out_keys=d_out_keys,
        d_out_items=None,
        op=OpKind.LESS,
        num_items=num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    # Warmup run to catch any CUDA errors before benchmarking
    try:
        sorter(
            temp_storage=temp_storage,
            d_in_keys=d_in_keys,
            d_in_items=None,
            d_out_keys=d_out_keys,
            d_out_items=None,
            op=OpKind.LESS,
            num_items=num_elements,
        )
        cp.cuda.Device().synchronize()
    except Exception as e:
        state.skip(f"CUDA error during warmup: {e}")
        return

    # Match C++ metrics (lines 87-89 of keys.cu)
    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in_keys.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out_keys.dtype.itemsize)

    # Execute benchmark
    def launcher(launch: bench.Launch):
        sorter(
            temp_storage=temp_storage,
            d_in_keys=d_in_keys,
            d_in_items=None,
            d_out_keys=d_out_keys,
            d_out_items=None,
            op=OpKind.LESS,
            num_items=num_elements,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_merge_sort_keys)
    b.set_name("base")  # Match C++ benchmark name

    # Match C++ axes (keys.cu lines 121-125)
    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))  # [16, 20, 24, 28]
    b.add_string_axis("Entropy", ENTROPY_VALUES)
    # Note: OffsetT axis from C++ is not exposed in Python API

    bench.run_all_benchmarks(sys.argv)
