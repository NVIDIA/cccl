# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for complex comparison using cuda.compute.

C++ equivalent: cub/benchmarks/bench/transform/complex_cmp.cu

Notes:
- Uses two overlapping input ranges (in[0:n-1], in[1:n])
- Output is boolean array of size n-1
- Benchmark name is "compare_complex" to match C++
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import make_binary_transform


def less_complex(a, b):
    # Lexicographic compare: real then imag
    if a.real < b.real:
        return True
    if a.real > b.real:
        return False
    return a.imag < b.imag


def bench_compare_complex(state: bench.State):
    num_elements = int(state.get_int64("Elements"))

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        real = cp.random.random(num_elements, dtype=np.float32)
        imag = cp.random.random(num_elements, dtype=np.float32)
        d_in = (real + 1j * imag).astype(np.complex64)
        d_out = cp.empty(num_elements - 1, dtype=np.bool_)

    num_items = num_elements - 1
    transformer = make_binary_transform(
        d_in1=d_in[:-1], d_in2=d_in[1:], d_out=d_out, op=less_complex
    )

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        transformer(
            d_in1=d_in[:-1],
            d_in2=d_in[1:],
            d_out=d_out,
            num_items=num_items,
            op=less_complex,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_compare_complex)
    b.set_name("compare_complex")  # Match C++ benchmark name
    b.add_int64_power_of_two_axis("Elements", range(16, 33, 4))
    bench.run_all_benchmarks(sys.argv)
