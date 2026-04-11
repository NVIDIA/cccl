# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for fill operation using cuda.compute.ConstantIterator.

C++ equivalent: cub/benchmarks/bench/transform/fill.cu

Notes:
- Migration: Python matches C++ integral_types (I8-I64); no tune parameters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
from utils import INTEGRAL_TYPES as TYPE_MAP
from utils import as_cupy_stream

import cuda.bench as bench
import cuda.compute
from cuda.compute import ConstantIterator, OpKind


def bench_fill(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements{io}"))

    # Setup data
    alloc_stream = as_cupy_stream(state.get_stream())
    try:
        with alloc_stream:
            d_out = cp.empty(num_items, dtype=dtype)
    except (MemoryError, cp.cuda.memory.OutOfMemoryError):
        state.skip("Skipping: out of memory.")
        return

    # Python equivalent of C++ return_constant<T>{42}
    constant_it = ConstantIterator(dtype(42))

    transform = cuda.compute.make_unary_transform(constant_it, d_out, OpKind.IDENTITY)

    state.add_element_count(num_items)
    state.add_global_memory_reads(0)
    state.add_global_memory_writes(num_items * d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        transform(constant_it, d_out, OpKind.IDENTITY, num_items, launch.get_stream())

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_fill)
    b.set_name("fill")

    b.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 33, 4))

    bench.run_all_benchmarks(sys.argv)
