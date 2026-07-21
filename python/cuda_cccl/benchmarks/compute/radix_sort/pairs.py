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
- C++ uses integral_types for keys and int8/16/32/64(+int128) for values
- Migration: Python matches C++ integral_types for both keys and values; omits int128 and OffsetT axis.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import INTEGRAL_TYPES, as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import SortOrder, make_radix_sort

KEY_TYPE_MAP = INTEGRAL_TYPES
VALUE_TYPE_MAP = INTEGRAL_TYPES


def bench_radix_sort_pairs(state: bench.State):
    key_type_str = state.get_string("KeyT{ct}")
    value_type_str = state.get_string("ValueT{ct}")
    key_dtype = KEY_TYPE_MAP[key_type_str]
    value_dtype = VALUE_TYPE_MAP[value_type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    entropy_str = state.get_string("Entropy")

    alloc_stream = as_cupy_stream(state.get_stream())

    d_in_keys = generate_data_with_entropy(
        num_elements, key_dtype, entropy_str, alloc_stream
    )

    d_in_values = generate_data_with_entropy(
        num_elements, value_dtype, "1.000", alloc_stream
    )

    with alloc_stream:
        d_out_keys = cp.empty(num_elements, dtype=key_dtype)
        d_out_values = cp.empty(num_elements, dtype=value_dtype)

    alloc_stream.synchronize()

    sorter = make_radix_sort(
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
        order=SortOrder.ASCENDING,
    )

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

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in_keys.dtype.itemsize)
    state.add_global_memory_reads(num_elements * d_in_values.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out_keys.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out_values.dtype.itemsize)

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

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_radix_sort_pairs)
    b.set_name("base")

    b.add_string_axis("KeyT{ct}", list(KEY_TYPE_MAP.keys()))
    b.add_string_axis("ValueT{ct}", list(VALUE_TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    b.add_string_axis("Entropy", ["1.000", "0.201"])
    # Note: OffsetT axis from C++ is not exposed in Python API

    bench.run_all_benchmarks(sys.argv)
