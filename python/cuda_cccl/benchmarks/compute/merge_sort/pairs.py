# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for merge_sort pairs using cuda.compute.

C++ equivalent: cub/benchmarks/bench/merge_sort/pairs.cu

Notes:
- Uses Entropy axis to control key distribution
- Keys and values are sorted together
- Migration: Python omits int128 values and OffsetT axis.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import (
    INTEGER_TYPES,
    SIGNED_TYPES,
    as_cupy_stream,
    generate_data_with_entropy,
)

import cuda.bench as bench
from cuda.compute import OpKind, clear_all_caches, make_merge_sort

# Key types: match C++ all_types (excluding int128 and complex)
KEY_TYPE_MAP = SIGNED_TYPES
# Value types: match C++ value_types (int8, int16, int32, int64)
VALUE_TYPE_MAP = INTEGER_TYPES

ENTROPY_VALUES = ["1.000", "0.201"]


def bench_merge_sort_pairs(state: bench.State):
    # WORKAROUND: Clear caches to avoid caching bug
    # See BUG_REPORT_CACHING.md for details
    clear_all_caches()

    key_type_str = state.get_string("KeyT")
    value_type_str = state.get_string("ValueT")
    key_dtype = KEY_TYPE_MAP[key_type_str]
    value_dtype = VALUE_TYPE_MAP[value_type_str]
    num_elements = int(state.get_int64("Elements"))
    entropy_str = state.get_string("Entropy")

    alloc_stream = as_cupy_stream(state.get_stream())

    d_in_keys = generate_data_with_entropy(
        num_elements, key_dtype, entropy_str, alloc_stream
    )

    with alloc_stream:
        if np.issubdtype(value_dtype, np.integer):
            info = np.iinfo(value_dtype)
            d_in_values = cp.random.randint(
                int(info.min), int(info.max) + 1, size=num_elements, dtype=np.int64
            ).astype(value_dtype)
        else:
            d_in_values = cp.random.uniform(-1, 1, size=num_elements).astype(
                value_dtype
            )

        d_out_keys = cp.empty(num_elements, dtype=key_dtype)
        d_out_values = cp.empty(num_elements, dtype=value_dtype)

    alloc_stream.synchronize()

    sorter = make_merge_sort(
        d_in_keys=d_in_keys,
        d_in_items=d_in_values,
        d_out_keys=d_out_keys,
        d_out_items=d_out_values,
        op=OpKind.LESS,
    )

    temp_storage_bytes = sorter(
        temp_storage=None,
        d_in_keys=d_in_keys,
        d_in_items=d_in_values,
        d_out_keys=d_out_keys,
        d_out_items=d_out_values,
        op=OpKind.LESS,
        num_items=num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    try:
        sorter(
            temp_storage=temp_storage,
            d_in_keys=d_in_keys,
            d_in_items=d_in_values,
            d_out_keys=d_out_keys,
            d_out_items=d_out_values,
            op=OpKind.LESS,
            num_items=num_elements,
        )
        cp.cuda.Device().synchronize()
    except Exception as e:
        state.skip(f"CUDA error during warmup: {e}")
        return

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in_keys.dtype.itemsize)
    state.add_global_memory_reads(num_elements * d_in_values.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out_keys.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out_values.dtype.itemsize)

    def launcher(launch: bench.Launch):
        sorter(
            temp_storage=temp_storage,
            d_in_keys=d_in_keys,
            d_in_items=d_in_values,
            d_out_keys=d_out_keys,
            d_out_items=d_out_values,
            op=OpKind.LESS,
            num_items=num_elements,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_merge_sort_pairs)
    b.set_name("base")  # Match C++ benchmark name
    b.add_string_axis("KeyT", list(KEY_TYPE_MAP.keys()))
    b.add_string_axis("ValueT", list(VALUE_TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))
    b.add_string_axis("Entropy", ENTROPY_VALUES)
    bench.run_all_benchmarks(sys.argv)
