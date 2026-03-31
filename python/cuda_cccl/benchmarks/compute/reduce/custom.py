# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for reduce custom operation using cuda.compute.reduce_into.

C++ equivalent: cub/benchmarks/bench/reduce/custom.cu

Notes:
- Uses a custom max operator (not OpKind) to exercise generic path
- int128 and complex32 are not supported by cupy
- Migration: Python limits to basic numeric types; C++ includes int128/complex.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import SIGNED_TYPES as TYPE_MAP
from utils import as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import make_reduce_into


def max_op(a, b):
    return a if a > b else b


def bench_reduce_custom(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements{io}"))

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        d_in = generate_data_with_entropy(num_items, dtype, "1.000", alloc_stream)
        d_out = cp.empty(1, dtype=dtype)

    h_init = np.zeros(1, dtype=dtype)

    reducer = make_reduce_into(d_in, d_out, max_op, h_init)

    temp_storage_bytes = reducer(None, d_in, d_out, max_op, num_items, h_init)
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    state.add_element_count(num_items)
    state.add_global_memory_reads(num_items * d_in.dtype.itemsize, "Size")
    state.add_global_memory_writes(d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        reducer(
            temp_storage, d_in, d_out, max_op, num_items, h_init, launch.get_stream()
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_reduce_custom)
    b.set_name("base")
    b.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    bench.run_all_benchmarks(sys.argv)
