# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for exclusive scan custom operation using cuda.compute.exclusive_scan.

C++ equivalent: cub/benchmarks/bench/scan/exclusive/custom.cu

Notes:
- Uses a custom max operator (not OpKind)
- int128 and complex32 are not supported by cupy
- Migration: Python fixes offsets; C++ exposes an OffsetT axis.
"""

import sys
from pathlib import Path

# Add parent directory (2 levels up) to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import make_exclusive_scan

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


def bench_scan_exclusive_custom(state: bench.State):
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements"))

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        if np.issubdtype(dtype, np.integer):
            d_in = cp.random.randint(0, 100, size=num_items, dtype=dtype)
        else:
            d_in = cp.random.random(num_items, dtype=dtype)
        d_out = cp.empty(num_items, dtype=dtype)

    h_init = np.zeros(1, dtype=dtype)

    scanner = make_exclusive_scan(d_in=d_in, d_out=d_out, op=max_op, init_value=h_init)

    temp_storage_bytes = scanner(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        op=max_op,
        num_items=num_items,
        init_value=h_init,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    state.add_element_count(num_items)
    state.add_global_memory_reads(num_items * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_items * d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        scanner(
            temp_storage=temp_storage,
            d_in=d_in,
            d_out=d_out,
            op=max_op,
            num_items=num_items,
            init_value=h_init,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_scan_exclusive_custom)
    b.set_name("base")  # Match C++ benchmark name
    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 33, 4))
    bench.run_all_benchmarks(sys.argv)
