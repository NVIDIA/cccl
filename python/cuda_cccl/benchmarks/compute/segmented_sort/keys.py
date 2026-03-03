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
- Migration: Power-law offsets use a Zipf approximation; uniform offsets use min_segment_size ~ max/2.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import (
    SIGNED_TYPES as TYPE_MAP,
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
    d_in_values,
    d_out_values,
    start_offsets,
    end_offsets,
    num_items,
    num_segments,
):
    sorter = make_segmented_sort(
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
        order=SortOrder.ASCENDING,
    )

    temp_storage_bytes = sorter(
        temp_storage=None,
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
        num_items=num_items,
        num_segments=num_segments,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
    )
    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    def launcher(launch: bench.Launch):
        sorter(
            temp_storage=temp_storage,
            d_in_keys=d_in_keys,
            d_out_keys=d_out_keys,
            d_in_values=d_in_values,
            d_out_values=d_out_values,
            num_items=num_items,
            num_segments=num_segments,
            start_offsets_in=start_offsets,
            end_offsets_in=end_offsets,
            stream=launch.get_stream(),
        )

    exec_tag = getattr(bench, "exec_tag", None)
    if exec_tag is not None:
        try:
            state.exec(launcher, exec_tag=exec_tag.sync, batched=False)
            return
        except TypeError:
            try:
                state.exec(exec_tag.sync, launcher, batched=False)
                return
            except TypeError:
                pass

    try:
        state.exec(launcher, batched=False, sync=True)
    except TypeError:
        state.exec(launcher, batched=False)


def bench_segmented_sort_power(state: bench.State):
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    num_segments = int(state.get_int64("Segments{io}"))
    entropy_str = state.get_string("Entropy")

    try:
        alloc_stream = as_cupy_stream(state.get_stream())
        d_in_keys = generate_data_with_entropy(
            num_elements, dtype, entropy_str, alloc_stream
        )
        with alloc_stream:
            d_out_keys = cp.empty(num_elements, dtype=dtype)
            d_in_values = cp.arange(num_elements, dtype=dtype)
            d_out_values = cp.empty(num_elements, dtype=dtype)

            offsets = generate_power_law_offsets(num_elements, num_segments)
            start_offsets = cp.asarray(offsets[:-1], dtype=np.int64)
            end_offsets = cp.asarray(offsets[1:], dtype=np.int64)

        alloc_stream.synchronize()

        state.add_element_count(num_elements)
        state.add_global_memory_reads(num_elements * d_in_keys.dtype.itemsize)
        state.add_global_memory_reads(num_elements * d_in_values.dtype.itemsize)
        state.add_global_memory_writes(num_elements * d_out_keys.dtype.itemsize)
        state.add_global_memory_writes(num_elements * d_out_values.dtype.itemsize)
        state.add_global_memory_reads((num_segments + 1) * start_offsets.dtype.itemsize)

        run_segmented_sort(
            state,
            d_in_keys,
            d_out_keys,
            d_in_values,
            d_out_values,
            start_offsets,
            end_offsets,
            num_elements,
            num_segments,
        )
    except (MemoryError, cp.cuda.memory.OutOfMemoryError):
        state.skip("Skipping: out of memory.")
        return


def bench_segmented_sort_uniform(state: bench.State):
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    max_segment_size = int(state.get_int64("MaxSegmentSize"))

    try:
        alloc_stream = as_cupy_stream(state.get_stream())
        d_in_keys = generate_data_with_entropy(
            num_elements, dtype, "1.000", alloc_stream
        )
        with alloc_stream:
            d_out_keys = cp.empty(num_elements, dtype=dtype)
            d_in_values = cp.arange(num_elements, dtype=dtype)
            d_out_values = cp.empty(num_elements, dtype=dtype)

            min_segment_size = max(1, max_segment_size // 2)
            offsets = generate_uniform_segment_offsets(
                num_elements, min_segment_size, max_segment_size
            )
            start_offsets = cp.asarray(offsets[:-1], dtype=np.int64)
            end_offsets = cp.asarray(offsets[1:], dtype=np.int64)

        alloc_stream.synchronize()

        num_segments = int(start_offsets.size)

        state.add_element_count(num_elements)
        state.add_global_memory_reads(num_elements * d_in_keys.dtype.itemsize)
        state.add_global_memory_reads(num_elements * d_in_values.dtype.itemsize)
        state.add_global_memory_writes(num_elements * d_out_keys.dtype.itemsize)
        state.add_global_memory_writes(num_elements * d_out_values.dtype.itemsize)
        state.add_global_memory_reads((num_segments + 1) * start_offsets.dtype.itemsize)

        run_segmented_sort(
            state,
            d_in_keys,
            d_out_keys,
            d_in_values,
            d_out_values,
            start_offsets,
            end_offsets,
            num_elements,
            num_segments,
        )
    except (MemoryError, cp.cuda.memory.OutOfMemoryError):
        state.skip("Skipping: out of memory.")
        return


if __name__ == "__main__":
    b_power = bench.register(bench_segmented_sort_power)
    b_power.set_name("power")
    b_power.add_string_axis("T", list(TYPE_MAP.keys()))
    b_power.add_int64_power_of_two_axis("Elements{io}", range(22, 31, 4))
    b_power.add_int64_power_of_two_axis("Segments{io}", range(12, 21, 4))
    b_power.add_string_axis("Entropy", ["1.000", "0.201"])

    b_small = bench.register(bench_segmented_sort_uniform)
    b_small.set_name("small")
    b_small.add_string_axis("T", list(TYPE_MAP.keys()))
    b_small.add_int64_power_of_two_axis("Elements{io}", range(22, 31, 4))
    b_small.add_int64_power_of_two_axis("MaxSegmentSize", range(1, 9, 1))

    b_large = bench.register(bench_segmented_sort_uniform)
    b_large.set_name("large")
    b_large.add_string_axis("T", list(TYPE_MAP.keys()))
    b_large.add_int64_power_of_two_axis("Elements{io}", range(22, 31, 4))
    b_large.add_int64_power_of_two_axis("MaxSegmentSize", range(10, 19, 2))

    bench.run_all_benchmarks(sys.argv)
