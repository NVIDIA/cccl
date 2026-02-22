# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
- Migration: Python fixes offsets and generates key segments on CPU to mirror C++.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import OpKind, clear_all_caches, make_unique_by_key

# Type mapping for keys: match C++ key_types (int8, int16, int32, int64)
KEY_TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
}

# Type mapping for values: match C++ all_types (fundamental types)
VALUE_TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "F32": np.float32,
    "F64": np.float64,
}

# MaxSegSize values from C++ benchmark (unique_by_key.cu line 176)
# These are powers of 2: 2^1=2, 2^4=16, 2^8=256
MAX_SEG_SIZE_VALUES = [2, 16, 256]


def generate_key_segments(
    num_elements, key_dtype, min_segment_size, max_segment_size, stream
):
    """
    Generate keys with uniform random segment sizes.

    Each segment consists of consecutive identical keys.
    Segment sizes are uniformly distributed between min_segment_size and max_segment_size.
    This mimics the C++ generate.uniform.key_segments() function.
    """
    with stream:
        # Generate segment sizes
        keys = cp.empty(num_elements, dtype=key_dtype)

        # CPU generation for segment structure, then copy to GPU
        # This is simpler and matches the C++ logic
        h_keys = np.empty(num_elements, dtype=key_dtype)

        current_pos = 0
        current_key = 0

        while current_pos < num_elements:
            # Random segment size between min and max
            seg_size = np.random.randint(min_segment_size, max_segment_size + 1)
            seg_size = min(seg_size, num_elements - current_pos)

            # Fill segment with current key
            h_keys[current_pos : current_pos + seg_size] = current_key

            current_pos += seg_size
            current_key += 1

            # Wrap key value to avoid overflow
            if np.issubdtype(key_dtype, np.integer):
                info = np.iinfo(key_dtype)
                if current_key > info.max:
                    current_key = info.min

        # Copy to GPU
        keys = cp.asarray(h_keys)

    return keys


def bench_unique_by_key(state: bench.State):
    """
    Benchmark unique_by_key operation.
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
    max_seg_size = int(state.get_int64("MaxSegSize"))

    # Allocate arrays
    alloc_stream = as_cupy_stream(state.get_stream())

    # Generate input keys with segments (min_segment_size=1 from C++ code)
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

        # Output arrays
        d_out_keys = cp.empty(num_elements, dtype=key_dtype)
        d_out_values = cp.empty(num_elements, dtype=value_dtype)
        d_num_selected = cp.empty(1, dtype=np.int32)

    # Synchronize to ensure data is ready
    alloc_stream.synchronize()

    # Build unique_by_key operation with EQUAL_TO operator
    uniquer = make_unique_by_key(
        d_in_keys=d_in_keys,
        d_in_items=d_in_values,
        d_out_keys=d_out_keys,
        d_out_items=d_out_values,
        d_out_num_selected=d_num_selected,
        op=OpKind.EQUAL_TO,
    )

    # Get temp storage size and allocate
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

    # Warmup run to catch any CUDA errors before benchmarking
    try:
        uniquer(
            temp_storage=temp_storage,
            d_in_keys=d_in_keys,
            d_in_items=d_in_values,
            d_out_keys=d_out_keys,
            d_out_items=d_out_values,
            d_out_num_selected=d_num_selected,
            op=OpKind.EQUAL_TO,
            num_items=num_elements,
        )
        cp.cuda.Device().synchronize()
    except Exception as e:
        state.skip(f"CUDA error during warmup: {e}")
        return

    # Get actual number of unique keys for accurate memory write count
    num_runs = int(d_num_selected.get()[0])

    # Match C++ metrics (lines 127-132 of unique_by_key.cu)
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

    # Execute benchmark
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

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_unique_by_key)
    b.set_name("base")  # Match C++ benchmark name

    # Match C++ axes (unique_by_key.cu lines 172-176)
    b.add_string_axis("KeyT", list(KEY_TYPE_MAP.keys()))
    b.add_string_axis("ValueT", list(VALUE_TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))  # [16, 20, 24, 28]
    b.add_int64_power_of_two_axis("MaxSegSize", [1, 4, 8])  # 2^1=2, 2^4=16, 2^8=256
    # Note: OffsetT axis from C++ is not exposed in Python API

    bench.run_all_benchmarks(sys.argv)
