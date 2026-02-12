# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for segmented_sort keys using cuda.compute.

C++ equivalent: cub/benchmarks/bench/segmented_sort/keys.cu

Notes:
- Implements three sub-benchmarks: power, small, large
- Power uses power-law segment sizes with Entropy axis
- Small/large use uniform segment sizes with MaxSegmentSize axis
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import SortOrder, clear_all_caches, make_segmented_sort

# Type mapping: match C++ fundamental_types (excluding int128 and complex)
TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F32": np.float32,
    "F64": np.float64,
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


def generate_power_law_offsets(num_elements, num_segments):
    sizes = np.random.zipf(1.5, size=num_segments).astype(np.int64)
    sizes = np.maximum(sizes, 1)

    if num_segments <= 0:
        return np.array([0, num_elements], dtype=np.int64)

    min_total = num_segments
    remaining = num_elements - min_total
    if remaining < 0:
        remaining = 0

    scaled = sizes / sizes.sum() * remaining
    sizes = np.floor(scaled).astype(np.int64)
    remainder = int(remaining - sizes.sum())
    if remainder > 0:
        sizes[:remainder] += 1

    sizes += 1

    offsets = np.empty(num_segments + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(sizes, dtype=np.int64)
    offsets[-1] = num_elements

    return offsets


def generate_uniform_offsets(num_elements, max_segment_size):
    min_segment_size = max(1, max_segment_size // 2)

    sizes = []
    remaining = num_elements
    while remaining > 0:
        seg_size = np.random.randint(min_segment_size, max_segment_size + 1)
        seg_size = min(seg_size, remaining)
        sizes.append(seg_size)
        remaining -= seg_size

    offsets = np.empty(len(sizes) + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(sizes, dtype=np.int64)
    offsets[-1] = num_elements

    return offsets


def run_segmented_sort(
    state: bench.State,
    d_in_keys,
    d_out_keys,
    start_offsets,
    end_offsets,
    num_items,
    num_segments,
):
    state.set_disable_blocking_kernel(True)
    state.set_run_once(True)
    sorter = make_segmented_sort(
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=None,
        d_out_values=None,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
        order=SortOrder.ASCENDING,
    )

    temp_storage_bytes = sorter(
        temp_storage=None,
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=None,
        d_out_values=None,
        num_items=num_items,
        num_segments=num_segments,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
    )
    temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    try:
        sorter(
            temp_storage=temp_storage,
            d_in_keys=d_in_keys,
            d_out_keys=d_out_keys,
            d_in_values=None,
            d_out_values=None,
            num_items=num_items,
            num_segments=num_segments,
            start_offsets_in=start_offsets,
            end_offsets_in=end_offsets,
        )
        cp.cuda.Device().synchronize()
    except Exception as e:
        state.skip(f"CUDA error during warmup: {e}")
        return

    clear_all_caches()

    def launcher(launch: bench.Launch):
        sorter(
            temp_storage=temp_storage,
            d_in_keys=d_in_keys,
            d_out_keys=d_out_keys,
            d_in_values=None,
            d_out_values=None,
            num_items=num_items,
            num_segments=num_segments,
            start_offsets_in=start_offsets,
            end_offsets_in=end_offsets,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


def bench_segmented_sort_power(state: bench.State):
    clear_all_caches()

    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements"))
    num_segments = int(state.get_int64("Segments"))
    entropy_str = state.get_string("Entropy")

    alloc_stream = as_cupy_stream(state.get_stream())
    d_in_keys = generate_data_with_entropy(
        num_elements, dtype, entropy_str, alloc_stream
    )
    with alloc_stream:
        d_out_keys = cp.empty(num_elements, dtype=dtype)

        offsets = generate_power_law_offsets(num_elements, num_segments)
        start_offsets = cp.asarray(offsets[:-1], dtype=np.int64)
        end_offsets = cp.asarray(offsets[1:], dtype=np.int64)

    alloc_stream.synchronize()

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


def bench_segmented_sort_uniform(state: bench.State):
    clear_all_caches()

    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements"))
    max_segment_size = int(state.get_int64("MaxSegmentSize"))

    alloc_stream = as_cupy_stream(state.get_stream())
    d_in_keys = generate_data_with_entropy(num_elements, dtype, "1.000", alloc_stream)
    with alloc_stream:
        d_out_keys = cp.empty(num_elements, dtype=dtype)

        offsets = generate_uniform_offsets(num_elements, max_segment_size)
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


if __name__ == "__main__":
    b_power = bench.register(bench_segmented_sort_power)
    b_power.set_name("power")
    b_power.add_string_axis("T", list(TYPE_MAP.keys()))
    b_power.add_int64_power_of_two_axis("Elements", range(22, 31, 4))
    b_power.add_int64_power_of_two_axis("Segments", range(12, 21, 4))
    b_power.add_string_axis("Entropy", ENTROPY_VALUES)

    b_small = bench.register(bench_segmented_sort_uniform)
    b_small.set_name("small")
    b_small.add_string_axis("T", list(TYPE_MAP.keys()))
    b_small.add_int64_power_of_two_axis("Elements", range(22, 31, 4))
    b_small.add_int64_power_of_two_axis("MaxSegmentSize", range(1, 9, 1))

    b_large = bench.register(bench_segmented_sort_uniform)
    b_large.set_name("large")
    b_large.add_string_axis("T", list(TYPE_MAP.keys()))
    b_large.add_int64_power_of_two_axis("Elements", range(22, 31, 4))
    b_large.add_int64_power_of_two_axis("MaxSegmentSize", range(10, 19, 2))

    bench.run_all_benchmarks(sys.argv)
