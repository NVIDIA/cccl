# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for three_way_partition using cuda.compute.

C++ equivalent: cub/benchmarks/bench/partition/three_way.cu

Notes:
- The C++ benchmark uses Entropy axis to control data distribution
- Uses less_then_t<T> predicate operators to divide data into three partitions:
  - First partition: items < left_border (max/3)
  - Second partition: items < right_border (max*2/3)
  - Third partition (unselected): items >= right_border
- T axis covers fundamental types (C++ fundamental_types minus int128)
- Migration: Python uses FUNDAMENTAL_TYPES; omits OffsetT axis.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import FUNDAMENTAL_TYPES, as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import make_three_way_partition


def bench_three_way_partition(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = FUNDAMENTAL_TYPES[type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    entropy_str = state.get_string("Entropy")

    alloc_stream = as_cupy_stream(state.get_stream())

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        min_val = 0
        max_val = info.max
    else:
        info = np.finfo(dtype)
        min_val = 0.0
        max_val = info.max

    left_border = max_val // 3 if np.issubdtype(dtype, np.integer) else max_val / 3
    right_border = left_border * 2

    d_in = generate_data_with_entropy(
        num_elements,
        dtype,
        entropy_str,
        alloc_stream,
        min_val=min_val,
        max_val=max_val,
    )

    with alloc_stream:
        d_first_part_out = cp.empty(num_elements, dtype=dtype)
        d_second_part_out = cp.empty(num_elements, dtype=dtype)
        d_unselected_out = cp.empty(num_elements, dtype=dtype)
        # d_num_selected_out stores [num_first_part, num_second_part]
        d_num_selected_out = cp.empty(2, dtype=np.int32)

    alloc_stream.synchronize()

    # Convert borders to the correct type for closure capture
    left_thresh = dtype(left_border)
    right_thresh = dtype(right_border)

    def select_first_part(x):
        return x < left_thresh

    def select_second_part(x):
        return x < right_thresh

    partitioner = make_three_way_partition(
        d_in,
        d_first_part_out,
        d_second_part_out,
        d_unselected_out,
        d_num_selected_out,
        select_first_part,
        select_second_part,
    )

    temp_storage_bytes = partitioner(
        None,
        d_in,
        d_first_part_out,
        d_second_part_out,
        d_unselected_out,
        d_num_selected_out,
        select_first_part,
        select_second_part,
        num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_in.dtype.itemsize)
    # C++ reports add_global_memory_writes<offset_t>(1) — 1 element of offset type.
    state.add_global_memory_writes(d_num_selected_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        partitioner(
            temp_storage,
            d_in,
            d_first_part_out,
            d_second_part_out,
            d_unselected_out,
            d_num_selected_out,
            select_first_part,
            select_second_part,
            num_elements,
            launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_three_way_partition)
    b.set_name("base")

    b.add_string_axis("T{ct}", list(FUNDAMENTAL_TYPES.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    b.add_string_axis("Entropy", ["1.000", "0.544", "0.000"])
    # Note: OffsetT axis from C++ is not exposed in Python API

    bench.run_all_benchmarks(sys.argv)
