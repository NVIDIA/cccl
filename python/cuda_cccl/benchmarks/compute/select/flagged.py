# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for select flagged using cuda.compute.select.

C++ equivalent: cub/benchmarks/bench/select/flagged.cu

Notes:
- Uses a boolean flag array to select elements
- In C++, data and flags are generated from the same RNG state (correlated).
  Python mirrors this by generating data with entropy, then deriving flags as
  bool(data[i]) != 0, which matches the C++ `flags = generator` semantics.
- The Python select API only supports predicate-based selection, so we use
  ZipIterator(data, flags) with a predicate that checks the flag component.
  This means Python also writes flags to the output, unlike C++ which only
  writes selected T values. Metrics are adjusted to match C++ accounting.
- Migration: Python omits InPlace and OffsetT axes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import FUNDAMENTAL_TYPES as TYPE_MAP
from utils import as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import ZipIterator, make_select


def bench_select_flagged(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    entropy_str = state.get_string("Entropy")

    alloc_stream = as_cupy_stream(state.get_stream())

    # C++ generates both data and flags from the same generator:
    #   auto generator = generate(elements, entropy);
    #   in = generator; flags = generator;
    # So flags[i] = bool(data[i]) -- flags are derived from the data values.
    d_in = generate_data_with_entropy(num_elements, dtype, entropy_str, alloc_stream)

    with alloc_stream:
        # Derive flags from data, matching C++ correlated generation.
        # In C++, the same random values are cast to bool, so flag = (value != 0).
        flags = (d_in != 0).astype(np.uint8)

        selected_elements = int(cp.count_nonzero(flags).get())
        d_out = cp.empty(selected_elements, dtype=dtype)
        d_out_flags = cp.empty(selected_elements, dtype=np.uint8)
        d_num_selected = cp.empty(1, dtype=np.int64)

    zip_it = ZipIterator(d_in, flags)

    def flag_predicate(pair):
        return np.uint8(pair[1] != 0)

    d_out_it = ZipIterator(d_out, d_out_flags)

    selector = make_select(
        d_in=zip_it,
        d_out=d_out_it,
        d_num_selected_out=d_num_selected,
        cond=flag_predicate,
    )

    temp_storage_bytes = selector(
        temp_storage=None,
        d_in=zip_it,
        d_out=d_out_it,
        d_num_selected_out=d_num_selected,
        cond=flag_predicate,
        num_items=num_elements,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    # Metrics match C++ accounting:
    #   reads: T * elements + bool * elements
    #   writes: T * selected_elements + OffsetT * 1
    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_reads(num_elements * np.dtype(np.bool_).itemsize)
    state.add_global_memory_writes(selected_elements * d_out.dtype.itemsize)
    state.add_global_memory_writes(d_num_selected.dtype.itemsize)

    def launcher(launch: bench.Launch):
        selector(
            temp_storage=temp_storage,
            d_in=zip_it,
            d_out=d_out_it,
            d_num_selected_out=d_num_selected,
            cond=flag_predicate,
            num_items=num_elements,
            stream=launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_select_flagged)
    b.set_name("base")
    b.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    b.add_string_axis("Entropy", ["1.000", "0.544", "0.000"])
    bench.run_all_benchmarks(sys.argv)
