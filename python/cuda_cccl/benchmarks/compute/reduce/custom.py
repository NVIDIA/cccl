# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for reduce custom operation using cuda.compute.reduce_into.

C++ equivalent: cub/benchmarks/bench/reduce/custom.cu

Notes:
- Uses a custom max operator (not OpKind) to exercise generic path
- int128 and complex32 are not supported by cupy
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import make_reduce_into

# Type mapping: match C++ all_types (excluding int128 and complex)
TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F32": np.float32,
    "F64": np.float64,
}


def max_op(a, b):
    return a if a > b else b


def bench_reduce_custom(state: bench.State):
    # Axes
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements"))

    # Setup data - use random values like C++ generate()
    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        if np.issubdtype(dtype, np.integer):
            d_in = cp.random.randint(0, 100, size=num_items, dtype=dtype)
        else:
            d_in = cp.random.random(num_items, dtype=dtype)

        d_out = cp.empty(1, dtype=dtype)

    # Initial value for reduction (C++ uses T{})
    h_init = np.zeros(1, dtype=dtype)

    reducer = make_reduce_into(d_in=d_in, d_out=d_out, op=max_op, h_init=h_init)

    temp_storage_bytes = reducer(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        op=max_op,
        num_items=num_items,
        h_init=h_init,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    state.add_element_count(num_items)
    state.add_global_memory_reads(num_items * d_in.dtype.itemsize)
    state.add_global_memory_writes(1 * d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        reducer(
            temp_storage=temp_storage,
            d_in=d_in,
            d_out=d_out,
            op=max_op,
            num_items=num_items,
            h_init=h_init,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_reduce_custom)
    b.set_name("base")  # Match C++ benchmark name
    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))
    bench.run_all_benchmarks(sys.argv)
