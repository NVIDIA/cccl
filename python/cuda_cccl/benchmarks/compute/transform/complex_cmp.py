# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for complex comparison using cuda.compute.

C++ equivalent: cub/benchmarks/bench/transform/complex_cmp.cu

Notes:
- Uses two overlapping input ranges (in[0:n-1], in[1:n])
- Output is boolean array of size n-1
- Benchmark name is "compare_complex" to match C++
- Migration: Python uses explicit lexicographic compare for complex64.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import cupy as cp
import numpy as np
from utils import as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import make_binary_transform

_COMPLEX_EPS = np.finfo(np.float32).eps
_COMPLEX_THRESHOLD = _COMPLEX_EPS * 2.0


def less_complex(a, b):
    mag0 = math.sqrt(a.real * a.real + a.imag * a.imag)
    mag1 = math.sqrt(b.real * b.real + b.imag * b.imag)

    if math.isnan(mag0) or math.isnan(mag1):
        return False

    if math.isinf(mag0) or math.isinf(mag1):
        scaler = 0.5
        mag0 = math.sqrt(
            (a.real * scaler) * (a.real * scaler)
            + (a.imag * scaler) * (a.imag * scaler)
        )
        mag1 = math.sqrt(
            (b.real * scaler) * (b.real * scaler)
            + (b.imag * scaler) * (b.imag * scaler)
        )

    if abs(mag0 - mag1) < _COMPLEX_THRESHOLD:
        phase0 = math.atan2(a.imag, a.real)
        phase1 = math.atan2(b.imag, b.real)
        return phase0 < phase1

    return mag0 < mag1


def bench_compare_complex(state: bench.State):
    num_elements = int(state.get_int64("Elements{io}"))

    alloc_stream = as_cupy_stream(state.get_stream())
    try:
        with alloc_stream:
            real = generate_data_with_entropy(
                num_elements, np.float32, "1.000", alloc_stream
            )
            imag = generate_data_with_entropy(
                num_elements, np.float32, "1.000", alloc_stream
            )
            d_in = (real + 1j * imag).astype(np.complex64)
            d_out = cp.empty(num_elements - 1, dtype=np.bool_)
    except (MemoryError, cp.cuda.memory.OutOfMemoryError):
        state.skip("Skipping: out of memory.")
        return

    num_items = num_elements - 1
    transformer = make_binary_transform(d_in[:-1], d_in[1:], d_out, less_complex)

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_elements * d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        transformer(
            d_in[:-1],
            d_in[1:],
            d_out,
            less_complex,
            num_items,
            launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_compare_complex)
    b.set_name("compare_complex")
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 33, 4))
    bench.run_all_benchmarks(sys.argv)
