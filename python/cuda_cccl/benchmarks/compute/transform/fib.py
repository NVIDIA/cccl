# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for transform fibonacci using cuda.compute.

C++ equivalent: cub/benchmarks/bench/transform/fib.cu

Notes:
- Input values are int64 in [0, 42]
- Output values are uint32
- Benchmark name is "fibonacci" to match C++
- Migration: Python fixes offsets to int64.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import make_unary_transform


def fib_op(n):
    t1 = 0
    t2 = 1

    if n < 1:
        return t1
    if n == 1:
        return t1
    if n == 2:
        return t2

    i = 3
    while i <= n:
        next_val = t1 + t2
        t1 = t2
        t2 = next_val
        i += 1

    return t2


def bench_transform_fib(state: bench.State):
    # Axes
    num_elements = int(state.get_int64("Elements{io}"))

    alloc_stream = as_cupy_stream(state.get_stream())
    try:
        with alloc_stream:
            d_in = generate_data_with_entropy(
                num_elements,
                np.int64,
                "1.000",
                alloc_stream,
                min_val=np.int64(0),
                max_val=np.int64(42),
            )
            d_out = cp.empty(num_elements, dtype=np.uint32)
    except (MemoryError, cp.cuda.memory.OutOfMemoryError):
        state.skip("Skipping: out of memory.")
        return

    transformer = make_unary_transform(d_in, d_out, fib_op)

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        transformer(d_in, d_out, fib_op, num_elements, launch.get_stream())

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_transform_fib)
    b.set_name("fibonacci")
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 33, 4))
    bench.run_all_benchmarks(sys.argv)
