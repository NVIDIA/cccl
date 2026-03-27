# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for segmented_sort keys using cuda.compute.

C++ equivalent: cub/benchmarks/bench/segmented_sort/keys.cu

Notes:
- Implements three sub-benchmarks: power, small, large
- Power uses power-law segment sizes with Entropy axis
- Small/large use uniform segment sizes with MaxSegmentSize axis
- Migration: uniform offsets use min_segment_size ~ max/2.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import (
    FUNDAMENTAL_TYPES as TYPE_MAP,
)
from utils import (
    as_cupy_stream,
    generate_data_with_entropy,
    generate_power_law_offsets,
    generate_uniform_segment_offsets,
)

import cuda.bench as bench
from cuda.compute import SortOrder, make_segmented_sort


def run_segmented_sort(
    state: bench.State,
    d_in_keys,
    d_out_keys,
    start_offsets,
    end_offsets,
    num_items,
    num_segments,
):
    sorter = make_segmented_sort(
        d_in_keys,
        d_out_keys,
        None,
        None,
        start_offsets,
        end_offsets,
        SortOrder.ASCENDING,
    )

    temp_storage_bytes = sorter(
        None,
        d_in_keys,
        d_out_keys,
        None,
        None,
        num_items,
        num_segments,
        start_offsets,
        end_offsets,
    )
    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    def launcher(launch: bench.Launch):
        sorter(
            temp_storage,
            d_in_keys,
            d_out_keys,
            None,
            None,
            num_items,
            num_segments,
            start_offsets,
            end_offsets,
            launch.get_stream(),
        )

    state.exec(launcher, batched=False, sync=True)


def bench_segmented_sort(state: bench.State, use_power_law: bool):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    alloc_stream = as_cupy_stream(state.get_stream())

    if use_power_law:
        num_segments = int(state.get_int64("Segments{io}"))
        entropy_str = state.get_string("Entropy")
    else:
        max_segment_size = int(state.get_int64("MaxSegmentSize"))
        min_segment_size = max(1, max_segment_size // 2)
        entropy_str = "1.000"

    if use_power_law:
        offsets = generate_power_law_offsets(num_elements, num_segments)
    else:
        offsets = generate_uniform_segment_offsets(
            num_elements, min_segment_size, max_segment_size
        )

    d_in_keys = generate_data_with_entropy(
        num_elements, dtype, entropy_str, alloc_stream
    )
    with alloc_stream:
        d_out_keys = cp.empty(num_elements, dtype=dtype)

        start_offsets = cp.asarray(offsets[:-1], dtype=np.int64)
        end_offsets = cp.asarray(offsets[1:], dtype=np.int64)

    alloc_stream.synchronize()
    num_segments = int(start_offsets.size)

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in_keys.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out_keys.dtype.itemsize)
    state.add_global_memory_reads((num_segments + 1) * start_offsets.dtype.itemsize)

    run_segmented_sort(
        state,
        d_in_keys,
        d_out_keys,
        start_offsets,
        end_offsets,
        num_elements,
        num_segments,
    )


def bench_segmented_sort_power(state: bench.State):
    bench_segmented_sort(state, use_power_law=True)


def bench_segmented_sort_uniform(state: bench.State):
    bench_segmented_sort(state, use_power_law=False)


if __name__ == "__main__":
    b_power = bench.register(bench_segmented_sort_power)
    b_power.set_name("power")
    b_power.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_power.add_int64_power_of_two_axis("Elements{io}", range(22, 31, 4))
    b_power.add_int64_power_of_two_axis("Segments{io}", range(12, 21, 4))
    b_power.add_string_axis("Entropy", ["1.000", "0.201"])

    b_small = bench.register(bench_segmented_sort_uniform)
    b_small.set_name("small")
    b_small.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_small.add_int64_power_of_two_axis("Elements{io}", range(22, 31, 4))
    b_small.add_int64_power_of_two_axis("MaxSegmentSize", range(1, 9, 1))

    b_large = bench.register(bench_segmented_sort_uniform)
    b_large.set_name("large")
    b_large.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_large.add_int64_power_of_two_axis("Elements{io}", range(22, 31, 4))
    b_large.add_int64_power_of_two_axis("MaxSegmentSize", range(10, 19, 2))

    bench.run_all_benchmarks(sys.argv)
