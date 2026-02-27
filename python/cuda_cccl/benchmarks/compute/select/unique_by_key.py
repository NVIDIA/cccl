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

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import INTEGER_TYPES, TYPE_MAP, as_cupy_stream, generate_key_segments

import cuda.bench as bench
from cuda.compute import OpKind, make_unique_by_key

# Key types: match C++ key_types (int8, int16, int32, int64)
KEY_TYPE_MAP = INTEGER_TYPES
# Value types: match C++ all_types (fundamental types)
VALUE_TYPE_MAP = TYPE_MAP

# MaxSegSize values from C++ benchmark (unique_by_key.cu line 176)
# These are powers of 2: 2^1=2, 2^4=16, 2^8=256
MAX_SEG_SIZE_VALUES = [2, 16, 256]


def bench_unique_by_key(state: bench.State):
    key_type_str = state.get_string("KeyT")
    value_type_str = state.get_string("ValueT")
    key_dtype = KEY_TYPE_MAP[key_type_str]
    value_dtype = VALUE_TYPE_MAP[value_type_str]
    num_elements = int(state.get_int64("Elements"))
    max_seg_size = int(state.get_int64("MaxSegSize"))

    alloc_stream = as_cupy_stream(state.get_stream())

    d_in_keys = generate_key_segments(
        num_elements,
        key_dtype,
        min_segment_size=1,
        max_segment_size=max_seg_size,
        stream=alloc_stream,
    )

    # Generate random input values
    with alloc_stream:
        if np.issubdtype(value_dtype, np.integer):
            info = np.iinfo(value_dtype)
            if value_dtype == np.int64:
                d_in_values = cp.random.randint(
                    int(info.min),
                    int(info.max),
                    size=num_elements,
                    dtype=np.int64,
                )
            else:
                d_in_values = cp.random.randint(
                    int(info.min),
                    int(info.max) + 1,
                    size=num_elements,
                    dtype=np.int64,
                ).astype(value_dtype)
        else:
            d_in_values = cp.random.uniform(-1, 1, size=num_elements).astype(
                value_dtype
            )

        d_out_keys = cp.empty(num_elements, dtype=key_dtype)
        d_out_values = cp.empty(num_elements, dtype=value_dtype)
        d_num_selected = cp.empty(1, dtype=np.int32)

    alloc_stream.synchronize()

    uniquer = make_unique_by_key(
        d_in_keys=d_in_keys,
        d_in_items=d_in_values,
        d_out_keys=d_out_keys,
        d_out_items=d_out_values,
        d_out_num_selected=d_num_selected,
        op=OpKind.EQUAL_TO,
    )

    temp_storage_bytes = uniquer(
        temp_storage=None,
        d_in_keys=d_in_keys,
        d_in_items=d_in_values,
        d_out_keys=d_out_keys,
        d_out_items=d_out_values,
        d_out_num_selected=d_num_selected,
        op=OpKind.EQUAL_TO,
        num_items=num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    # Get actual number of unique keys for accurate memory write count
    num_runs = int(d_num_selected.get()[0])

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in_keys.dtype.itemsize)  # Keys read
    state.add_global_memory_reads(
        num_elements * d_in_values.dtype.itemsize
    )  # Values read
    state.add_global_memory_writes(num_runs * d_out_keys.dtype.itemsize)  # Keys written
    state.add_global_memory_writes(
        num_runs * d_out_values.dtype.itemsize
    )  # Values written
    state.add_global_memory_writes(
        d_num_selected.dtype.itemsize
    )  # num_selected written

    def launcher(launch: bench.Launch):
        uniquer(
            temp_storage=temp_storage,
            d_in_keys=d_in_keys,
            d_in_items=d_in_values,
            d_out_keys=d_out_keys,
            d_out_items=d_out_values,
            d_out_num_selected=d_num_selected,
            op=OpKind.EQUAL_TO,
            num_items=num_elements,
            stream=launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_unique_by_key)
    b.set_name("base")

    b.add_string_axis("KeyT", list(KEY_TYPE_MAP.keys()))
    b.add_string_axis("ValueT", list(VALUE_TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))  # [16, 20, 24, 28]
    b.add_int64_power_of_two_axis("MaxSegSize", [1, 4, 8])  # 2^1=2, 2^4=16, 2^8=256
    # Note: OffsetT axis from C++ is not exposed in Python API

    bench.run_all_benchmarks(sys.argv)
