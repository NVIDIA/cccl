# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for radix_sort keys using cuda.compute.

C++ equivalent: cub/benchmarks/bench/radix_sort/keys.cu

Notes:
- The C++ benchmark uses Entropy axis to control data distribution
- Sort order is always ascending (C++ benchmark hardcodes this)
- Keys only (no values) - see radix_sort/pairs.cu for key-value sorting
- begin_bit=0, end_bit=sizeof(T)*8 (full key comparison)
- Migration: Python fixes offsets, excludes int128, and approximates entropy generation.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import FUNDAMENTAL_TYPES as TYPE_MAP
from utils import as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import SortOrder, make_radix_sort


def bench_radix_sort_keys(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    entropy_str = state.get_string("Entropy")

    alloc_stream = as_cupy_stream(state.get_stream())

    d_in_keys = generate_data_with_entropy(
        num_elements, dtype, entropy_str, alloc_stream
    )

    # Output array for sorted keys
    with alloc_stream:
        d_out_keys = cp.empty(num_elements, dtype=dtype)

    alloc_stream.synchronize()

    sorter = make_radix_sort(d_in_keys, d_out_keys, None, None, SortOrder.ASCENDING)

    temp_storage_bytes = sorter(
        None,
        d_in_keys,
        d_out_keys,
        None,
        None,
        num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in_keys.dtype.itemsize, "Size")
    state.add_global_memory_writes(num_elements * d_out_keys.dtype.itemsize)

    def launcher(launch: bench.Launch):
        sorter(
            temp_storage,
            d_in_keys,
            d_out_keys,
            None,
            None,
            num_elements,
            launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_radix_sort_keys)
    b.set_name("base")

    b.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    b.add_string_axis("Entropy", ["1.000", "0.544", "0.201"])
    # Note: OffsetT axis from C++ is not exposed in Python API

    bench.run_all_benchmarks(sys.argv)
