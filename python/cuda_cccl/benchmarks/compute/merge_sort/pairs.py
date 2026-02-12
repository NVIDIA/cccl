# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for merge_sort pairs using cuda.compute.

C++ equivalent: cub/benchmarks/bench/merge_sort/pairs.cu

Notes:
- Uses Entropy axis to control key distribution
- Keys and values are sorted together
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import OpKind, clear_all_caches, make_merge_sort

# Key types: match C++ all_types (excluding int128 and complex)
KEY_TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F32": np.float32,
    "F64": np.float64,
}

# Value types: match C++ value_types (int8, int16, int32, int64)
VALUE_TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
}

ENTROPY_VALUES = ["1.000", "0.201"]

ENTROPY_TO_PROB = {
    "1.000": 1.0,
    "0.811": 0.811,
    "0.544": 0.544,
    "0.337": 0.337,
    "0.201": 0.201,
    "0.000": 0.0,
}


def generate_data_with_entropy(num_elements, dtype, entropy_str, stream):
    probability = ENTROPY_TO_PROB[entropy_str]

    with stream:
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            if probability == 1.0:
                data = cp.random.randint(
                    int(info.min), int(info.max) + 1, size=num_elements, dtype=np.int64
                ).astype(dtype)
            else:
                range_size = int((int(info.max) - int(info.min)) * probability)
                if range_size < 1:
                    range_size = 1
                data = cp.random.randint(
                    0, range_size, size=num_elements, dtype=np.int64
                ).astype(dtype)
        else:
            info = np.finfo(dtype)
            if probability == 1.0:
                data = cp.random.uniform(-1, 1, size=num_elements).astype(dtype)
                data = data * info.max * 0.5
            else:
                scale = probability * info.max * 0.5
                data = cp.random.uniform(-scale, scale, size=num_elements).astype(dtype)

    return data


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
