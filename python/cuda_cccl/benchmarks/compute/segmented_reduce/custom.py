# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for segmented_reduce custom operation using cuda.compute.

C++ equivalent: cub/benchmarks/bench/segmented_reduce/custom.cu

Notes:
- Uses a custom max operator
- Segment size ranges match segmented_reduce/sum
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import clear_all_caches, make_segmented_reduce

# Type mapping: match C++ all_types (fundamental types)
TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F32": np.float32,
    "F64": np.float64,
}

SEGMENT_SIZES_SMALL = [2**i for i in range(0, 5)]
SEGMENT_SIZES_MEDIUM = [2**i for i in range(5, 9)]
SEGMENT_SIZES_LARGE = [2**i for i in range(9, 17)]


def max_op(a, b):
    return a if a > b else b


def generate_segment_offsets(num_elements, segment_size, stream):
    num_segments = max(1, num_elements // segment_size)
    actual_elements = num_segments * segment_size

    with stream:
        start_offsets = cp.arange(0, actual_elements, segment_size, dtype=np.int64)
        end_offsets = cp.arange(
            segment_size, actual_elements + 1, segment_size, dtype=np.int64
        )

    return start_offsets, end_offsets, num_segments, actual_elements


def bench_segmented_reduce_custom(state: bench.State):
    # WORKAROUND: Clear caches to avoid caching bug
    # See BUG_REPORT_CACHING.md for details
    clear_all_caches()

    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements"))
    segment_size = int(state.get_int64("SegmentSize"))

    alloc_stream = as_cupy_stream(state.get_stream())

    start_offsets, end_offsets, num_segments, actual_elements = (
        generate_segment_offsets(num_elements, segment_size, alloc_stream)
    )

    with alloc_stream:
        if np.issubdtype(dtype, np.integer):
            max_val = min(100, np.iinfo(dtype).max)
            min_val = max(-100, np.iinfo(dtype).min)
            d_in = cp.random.randint(
                min_val, max_val + 1, size=actual_elements, dtype=dtype
            )
        else:
            d_in = cp.random.uniform(-100, 100, size=actual_elements).astype(dtype)

        d_out = cp.empty(num_segments, dtype=dtype)

    h_init = np.zeros(1, dtype=dtype)

    reducer = make_segmented_reduce(
        d_in=d_in,
        d_out=d_out,
        start_offsets=start_offsets,
        end_offsets=end_offsets,
        op=max_op,
        h_init=h_init,
    )

    temp_storage_bytes = reducer(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        start_offsets=start_offsets,
        end_offsets=end_offsets,
        op=max_op,
        num_segments=num_segments,
        h_init=h_init,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    try:
        reducer(
            temp_storage=temp_storage,
            d_in=d_in,
            d_out=d_out,
            start_offsets=start_offsets,
            end_offsets=end_offsets,
            op=max_op,
            num_segments=num_segments,
            h_init=h_init,
        )
        cp.cuda.Device().synchronize()
    except Exception as e:
        state.skip(f"CUDA error during warmup: {e}")
        return

    state.add_element_count(actual_elements)
    state.add_global_memory_reads(actual_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_segments * d_out.dtype.itemsize)
    state.add_global_memory_reads((num_segments + 1) * start_offsets.dtype.itemsize)

    def launcher(launch: bench.Launch):
        reducer(
            temp_storage=temp_storage,
            d_in=d_in,
            d_out=d_out,
            start_offsets=start_offsets,
            end_offsets=end_offsets,
            op=max_op,
            num_segments=num_segments,
            h_init=h_init,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b_small = bench.register(bench_segmented_reduce_custom)
    b_small.set_name("small")
    b_small.add_string_axis("T", list(TYPE_MAP.keys()))
    b_small.add_int64_power_of_two_axis("Elements", range(16, 29, 4))
    b_small.add_int64_axis("SegmentSize", SEGMENT_SIZES_SMALL)

    b_medium = bench.register(bench_segmented_reduce_custom)
    b_medium.set_name("medium")
    b_medium.add_string_axis("T", list(TYPE_MAP.keys()))
    b_medium.add_int64_power_of_two_axis("Elements", range(16, 29, 4))
    b_medium.add_int64_axis("SegmentSize", SEGMENT_SIZES_MEDIUM)

    b_large = bench.register(bench_segmented_reduce_custom)
    b_large.set_name("large")
    b_large.add_string_axis("T", list(TYPE_MAP.keys()))
    b_large.add_int64_power_of_two_axis("Elements", range(16, 29, 4))
    b_large.add_int64_axis("SegmentSize", SEGMENT_SIZES_LARGE)

    bench.run_all_benchmarks(sys.argv)
