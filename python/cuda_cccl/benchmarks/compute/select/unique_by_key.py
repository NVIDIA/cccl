# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for unique_by_key using cuda.compute.

C++ equivalent: cub/benchmarks/bench/select/unique_by_key.cu

Notes:
- The C++ benchmark uses MaxSegSize axis to control segment sizes
- Uses equal_to comparison operator for key equality
- Generates key segments with sizes between 1 and MaxSegSize
- Both keys and values are processed
- Migration: Python fixes offsets and generates key segments on GPU to mirror C++.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import INTEGRAL_TYPES, SIGNED_TYPES, as_cupy_stream, generate_key_segments

import cuda.bench as bench
from cuda.compute import OpKind, make_unique_by_key

KEY_TYPE_MAP = INTEGRAL_TYPES
VALUE_TYPE_MAP = {**SIGNED_TYPES, "C32": np.complex64}


def bench_unique_by_key(state: bench.State):
    key_type_str = state.get_string("KeyT{ct}")
    value_type_str = state.get_string("ValueT{ct}")
    key_dtype = KEY_TYPE_MAP[key_type_str]
    value_dtype = VALUE_TYPE_MAP[value_type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    max_seg_size = int(state.get_int64("MaxSegSize"))

    if num_elements > np.iinfo(np.int32).max:
        state.skip("Skipping: num_elements exceeds int32 limits")
        return

    alloc_stream = as_cupy_stream(state.get_stream())

    d_in_keys = generate_key_segments(
        num_elements,
        key_dtype,
        min_segment_size=1,
        max_segment_size=max_seg_size,
        stream=alloc_stream,
    )

    with alloc_stream:
        d_in_values = cp.zeros(num_elements, dtype=value_dtype)

        d_out_keys = cp.empty(num_elements, dtype=key_dtype)
        d_out_values = cp.empty(num_elements, dtype=value_dtype)
        d_num_selected = cp.empty(1, dtype=np.int32)

    alloc_stream.synchronize()

    uniquer = make_unique_by_key(
        d_in_keys,
        d_in_values,
        d_out_keys,
        d_out_values,
        d_num_selected,
        OpKind.EQUAL_TO,
    )

    temp_storage_bytes = uniquer(
        None,
        d_in_keys,
        d_in_values,
        d_out_keys,
        d_out_values,
        d_num_selected,
        OpKind.EQUAL_TO,
        num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    # Run once before timing to materialize the number of selected runs,
    # matching the C++ metric accounting flow.
    uniquer(
        temp_storage,
        d_in_keys,
        d_in_values,
        d_out_keys,
        d_out_values,
        d_num_selected,
        OpKind.EQUAL_TO,
        num_elements,
        alloc_stream,
    )
    alloc_stream.synchronize()
    num_runs = int(d_num_selected.get()[0])

    state.add_element_count(num_elements)
    state.add_global_memory_reads(int(num_elements * d_in_keys.dtype.itemsize))
    state.add_global_memory_reads(int(num_elements * d_in_values.dtype.itemsize))
    state.add_global_memory_writes(int(num_runs * d_out_keys.dtype.itemsize))
    state.add_global_memory_writes(int(num_runs * d_out_values.dtype.itemsize))
    state.add_global_memory_writes(int(d_num_selected.dtype.itemsize))

    def launcher(launch: bench.Launch):
        uniquer(
            temp_storage,
            d_in_keys,
            d_in_values,
            d_out_keys,
            d_out_values,
            d_num_selected,
            OpKind.EQUAL_TO,
            num_elements,
            launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_unique_by_key)
    b.set_name("base")

    b.add_string_axis("KeyT{ct}", list(KEY_TYPE_MAP.keys()))
    b.add_string_axis("ValueT{ct}", list(VALUE_TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    b.add_int64_power_of_two_axis("MaxSegSize", [1, 4, 8])
    # Note: OffsetT axis from C++ is not exposed in Python API

    bench.run_all_benchmarks(sys.argv)
