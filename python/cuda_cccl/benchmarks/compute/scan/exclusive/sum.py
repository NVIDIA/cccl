# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for exclusive scan sum operation using cuda.compute.exclusive_scan.

C++ equivalent: cub/benchmarks/bench/scan/exclusive/sum.cu

Notes:
- int128 and complex32 are not supported by cupy
- Migration: Python fixes offsets; C++ exposes an OffsetT axis.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

# Add parent directory (2 levels up) to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import OpKind, make_exclusive_scan

# Type mapping: match C++ all_types
TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F32": np.float32,
    "F64": np.float64,
}


def bench_scan_exclusive_sum(state: bench.State):
    """
    Benchmark exclusive scan sum operation using OpKind.PLUS.
    """

    # Get parameters from axes
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

        # Output is same size as input for scan
        d_out = cp.empty(num_items, dtype=dtype)

    # Initial value for scan (identity for addition)
    h_init = np.zeros(1, dtype=dtype)

    # Build scan operation using OpKind.PLUS
    scanner = make_exclusive_scan(
        d_in=d_in, d_out=d_out, op=OpKind.PLUS, init_value=h_init
    )

    # Get temp storage size and allocate: Benchmark only execution
    temp_storage_bytes = scanner(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        op=OpKind.PLUS,
        num_items=num_items,
        init_value=h_init,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    # Match C++ metrics:
    # state.add_element_count(elements);
    # state.add_global_memory_reads<T>(elements, "Size");
    # state.add_global_memory_writes<T>(elements);
    state.add_element_count(num_items)
    state.add_global_memory_reads(num_items * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_items * d_out.dtype.itemsize)

    # Execute benchmark
    def launcher(launch: bench.Launch):
        scanner(
            temp_storage=temp_storage,
            d_in=d_in,
            d_out=d_out,
            op=OpKind.PLUS,
            num_items=num_items,
            init_value=h_init,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_scan_exclusive_sum)
    b.set_name("base")  # Match C++ benchmark name

    # Match C++ axes
    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 33, 4))  # [16, 20, 24, 28, 32]

    bench.run_all_benchmarks(sys.argv)
