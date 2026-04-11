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

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import (
    INTEGRAL_TYPES,
    SIGNED_TYPES,
    as_cupy_stream,
    generate_data_with_entropy,
)

import cuda.bench as bench
from cuda.compute import OpKind, make_merge_sort

KEY_TYPE_MAP = SIGNED_TYPES
VALUE_TYPE_MAP = INTEGRAL_TYPES


def bench_merge_sort_pairs(state: bench.State):
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

    with alloc_stream:
        d_in_values = generate_data_with_entropy(
            num_elements, value_dtype, "1.000", alloc_stream
        )

        d_out_keys = cp.empty(num_elements, dtype=key_dtype)
        d_out_values = cp.empty(num_elements, dtype=value_dtype)

    alloc_stream.synchronize()

    sorter = make_merge_sort(
        d_in_keys, d_in_values, d_out_keys, d_out_values, OpKind.LESS
    )

    temp_storage_bytes = sorter(
        None,
        d_in_keys,
        d_in_values,
        d_out_keys,
        d_out_values,
        OpKind.LESS,
        num_elements,
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
            temp_storage,
            d_in_keys,
            d_in_values,
            d_out_keys,
            d_out_values,
            OpKind.LESS,
            num_elements,
            launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_merge_sort_pairs)
    b.set_name("base")
    b.add_string_axis("KeyT{ct}", list(KEY_TYPE_MAP.keys()))
    b.add_string_axis("ValueT{ct}", list(VALUE_TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    b.add_string_axis("Entropy", ["1.000", "0.201"])
    bench.run_all_benchmarks(sys.argv)
