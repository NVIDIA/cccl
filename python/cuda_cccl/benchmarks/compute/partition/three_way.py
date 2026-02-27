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
- T axis covers fundamental types
- Migration: Python mirrors min/max borders but omits OffsetT axis.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import ENTROPY_TO_PROB, TYPE_MAP, as_cupy_stream

import cuda.bench as bench
from cuda.compute import clear_all_caches, make_three_way_partition

# Entropy values from C++ benchmark (three_way.cu line 131)
ENTROPY_VALUES = ["1.000", "0.544", "0.000"]


def generate_data_with_entropy(
    num_elements, dtype, entropy_str, min_val, max_val, stream
):
    """
    Generate data with specified entropy level within a given range.

    Entropy controls the bit-level randomness of the data:
    - 1.000: Full random (all bits random)
    - 0.544: Medium entropy
    - 0.000: Constant value (no entropy)
    """
    probability = ENTROPY_TO_PROB[entropy_str]

    with stream:
        if probability == 0.0:
            # Zero entropy - constant value
            data = cp.full(num_elements, min_val, dtype=dtype)
        elif np.issubdtype(dtype, np.integer):
            if probability == 1.0:
                # Full random across the specified range
                data = cp.random.randint(
                    int(min_val), int(max_val) + 1, size=num_elements, dtype=np.int64
                ).astype(dtype)
            else:
                # Reduced entropy: limit the range of values
                range_size = int((int(max_val) - int(min_val)) * probability)
                if range_size < 1:
                    range_size = 1
                data = cp.random.randint(
                    int(min_val),
                    int(min_val) + range_size,
                    size=num_elements,
                    dtype=np.int64,
                ).astype(dtype)
        else:
            # Floating point
            if probability == 1.0:
                # Full random in [min_val, max_val]
                data = cp.random.uniform(
                    float(min_val), float(max_val), size=num_elements
                ).astype(dtype)
            else:
                # Reduced entropy: smaller range
                range_val = float(max_val - min_val) * probability
                data = cp.random.uniform(
                    float(min_val), float(min_val) + range_val, size=num_elements
                ).astype(dtype)

    return data


def bench_three_way_partition(state: bench.State):
    # WORKAROUND: Clear caches to avoid caching bug
    # See BUG_REPORT_CACHING.md for details
    clear_all_caches()

    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements"))
    entropy_str = state.get_string("Entropy")

    alloc_stream = as_cupy_stream(state.get_stream())

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        min_val = (
            0  # C++ uses min_val{} which is 0 for unsigned, could be min for signed
        )
        max_val = info.max
    else:
        info = np.finfo(dtype)
        min_val = 0.0
        max_val = info.max

    left_border = max_val // 3 if np.issubdtype(dtype, np.integer) else max_val / 3
    right_border = left_border * 2

    d_in = generate_data_with_entropy(
        num_elements, dtype, entropy_str, min_val, max_val, alloc_stream
    )

    with alloc_stream:
        d_first_part_out = cp.empty(num_elements, dtype=dtype)
        d_second_part_out = cp.empty(num_elements, dtype=dtype)
        d_unselected_out = cp.empty(num_elements, dtype=dtype)
        # d_num_selected_out stores [num_first_part, num_second_part]
        d_num_selected_out = cp.empty(2, dtype=np.int32)

    alloc_stream.synchronize()

    # Items where select_first_op(x) is true go to first partition
    # Items where select_second_op(x) is true (and first is false) go to second partition
    # Items where both are false go to unselected
    #
    # In C++ the predicates are:
    #   select_op_1: x < left_border (selects first partition)
    #   select_op_2: x < right_border (combined with !select_op_1, selects second partition)

    # Convert borders to the correct type for closure capture
    left_thresh = dtype(left_border)
    right_thresh = dtype(right_border)

    # Use regular Python functions - cuda.compute JIT-compiles them internally
    def select_first_part(x):
        return x < left_thresh

    def select_second_part(x):
        return x < right_thresh

    partitioner = make_three_way_partition(
        d_in=d_in,
        d_first_part_out=d_first_part_out,
        d_second_part_out=d_second_part_out,
        d_unselected_out=d_unselected_out,
        d_num_selected_out=d_num_selected_out,
        select_first_part_op=select_first_part,
        select_second_part_op=select_second_part,
    )

    temp_storage_bytes = partitioner(
        temp_storage=None,
        d_in=d_in,
        d_first_part_out=d_first_part_out,
        d_second_part_out=d_second_part_out,
        d_unselected_out=d_unselected_out,
        d_num_selected_out=d_num_selected_out,
        select_first_part_op=select_first_part,
        select_second_part_op=select_second_part,
        num_items=num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    try:
        partitioner(
            temp_storage=temp_storage,
            d_in=d_in,
            d_first_part_out=d_first_part_out,
            d_second_part_out=d_second_part_out,
            d_unselected_out=d_unselected_out,
            d_num_selected_out=d_num_selected_out,
            select_first_part_op=select_first_part,
            select_second_part_op=select_second_part,
            num_items=num_elements,
        )
        cp.cuda.Device().synchronize()
    except Exception as e:
        state.skip(f"CUDA error during warmup: {e}")
        return

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(
        d_num_selected_out.dtype.itemsize
    )  # num_selected written

    def launcher(launch: bench.Launch):
        partitioner(
            temp_storage=temp_storage,
            d_in=d_in,
            d_first_part_out=d_first_part_out,
            d_second_part_out=d_second_part_out,
            d_unselected_out=d_unselected_out,
            d_num_selected_out=d_num_selected_out,
            select_first_part_op=select_first_part,
            select_second_part_op=select_second_part,
            num_items=num_elements,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_three_way_partition)
    b.set_name("base")

    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))  # [16, 20, 24, 28]
    b.add_string_axis("Entropy", ENTROPY_VALUES)
    # Note: OffsetT axis from C++ is not exposed in Python API

    bench.run_all_benchmarks(sys.argv)
