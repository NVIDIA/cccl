# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for fill operation using cuda.compute.ConstantIterator.

C++ equivalent: cub/benchmarks/bench/transform/fill.cu

Notes:
- Migration: Python matches signed integer coverage (I8-I64); no tune parameters.
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
import cuda.compute
from cuda.compute import ConstantIterator, OpKind

# Type mapping: C++ types to NumPy dtypes
TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
}


def bench_fill(state: bench.State):
    """
    Benchmark transform fill operation using ConstantIterator.
    """

    # Get parameters from axes
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements"))

    # Setup data
    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        d_out = cp.empty(num_items, dtype=dtype)

    # Python equivalent of C++ return_constant<T>{42}
    constant_it = ConstantIterator(dtype(42))

    # Build transform operation: ConstantIterator -> output
    transform = cuda.compute.make_unary_transform(
        d_in=constant_it, d_out=d_out, op=OpKind.IDENTITY
    )

    # Match C++ metrics
    state.add_element_count(num_items)
    state.add_global_memory_reads(0)
    state.add_global_memory_writes(num_items * d_out.dtype.itemsize)

    # Execute benchmark
    def launcher(launch: bench.Launch):
        transform(
            d_in=constant_it,
            d_out=d_out,
            op=OpKind.IDENTITY,
            num_items=num_items,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_fill)
    b.set_name("fill")

    # Match C++ axis
    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 33, 4))

    bench.run_all_benchmarks(sys.argv)
