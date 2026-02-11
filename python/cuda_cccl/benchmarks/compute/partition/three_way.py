# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numba.cuda as cuda
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import clear_all_caches, make_three_way_partition

# Type mapping: match C++ fundamental_types (excluding int128 and complex)
TYPE_MAP = {
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

# Entropy values from C++ benchmark (three_way.cu line 131)
ENTROPY_VALUES = ["1.000", "0.544", "0.000"]

# Entropy to probability mapping (from nvbench_helper.cuh)
ENTROPY_TO_PROB = {
    "1.000": 1.0,
    "0.811": 0.811,
    "0.544": 0.544,
    "0.337": 0.337,
    "0.201": 0.201,
    "0.000": 0.0,
}


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


def make_less_than_predicate(threshold):
    """
    Create a less-than predicate function for three-way partition.

    This returns a numba device function that checks if x < threshold.
    """

    @cuda.jit(device=True)
    def less_than(x):
        return x < threshold

    return less_than


def bench_three_way_partition(state: bench.State):
    """
    Benchmark three_way_partition operation.
    """
    # WORKAROUND: Clear caches to avoid caching bug
    # See BUG_REPORT_CACHING.md for details
    clear_all_caches()

    # Get parameters from axes
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements"))
    entropy_str = state.get_string("Entropy")

    # Allocate arrays
    alloc_stream = as_cupy_stream(state.get_stream())

    # Compute borders like C++ code (lines 78-85 of three_way.cu)
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

    # Compute borders: left = max/3, right = max*2/3
    left_border = max_val // 3 if np.issubdtype(dtype, np.integer) else max_val / 3
    right_border = left_border * 2

    # Generate input data with specified entropy
    d_in = generate_data_with_entropy(
        num_elements, dtype, entropy_str, min_val, max_val, alloc_stream
    )

    # Output arrays
    with alloc_stream:
        d_first_part_out = cp.empty(num_elements, dtype=dtype)
        d_second_part_out = cp.empty(num_elements, dtype=dtype)
        d_unselected_out = cp.empty(num_elements, dtype=dtype)
        # d_num_selected_out stores [num_first_part, num_second_part]
        d_num_selected_out = cp.empty(2, dtype=np.int32)

    # Synchronize to ensure data is ready
    alloc_stream.synchronize()

    # Create predicate functions for three-way partition
    # Items where select_first_op(x) is true go to first partition
    # Items where select_second_op(x) is true (and first is false) go to second partition
    # Items where both are false go to unselected
    #
    # In C++ the predicates are:
    #   select_op_1: x < left_border (selects first partition)
    #   select_op_2: x < right_border (combined with !select_op_1, selects second partition)

    # Convert borders to the correct type for numba
    left_border_typed = dtype(left_border)
    right_border_typed = dtype(right_border)

    @cuda.jit(device=True)
    def select_first_part(x):
        return x < left_border_typed

    @cuda.jit(device=True)
    def select_second_part(x):
        return x < right_border_typed

    # Build three_way_partition operation
    partitioner = make_three_way_partition(
        d_in=d_in,
        d_first_part_out=d_first_part_out,
        d_second_part_out=d_second_part_out,
        d_unselected_out=d_unselected_out,
        d_num_selected_out=d_num_selected_out,
        select_first_part_op=select_first_part,
        select_second_part_op=select_second_part,
    )

    # Get temp storage size and allocate
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

    # Warmup run to catch any CUDA errors before benchmarking
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

    # Match C++ metrics (lines 99-102 of three_way.cu)
    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(
        d_num_selected_out.dtype.itemsize
    )  # num_selected written

    # Execute benchmark
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
    b.set_name("base")  # Match C++ benchmark name

    # Match C++ axes (three_way.cu lines 127-131)
    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))  # [16, 20, 24, 28]
    b.add_string_axis("Entropy", ENTROPY_VALUES)
    # Note: OffsetT axis from C++ is not exposed in Python API

    bench.run_all_benchmarks(sys.argv)
