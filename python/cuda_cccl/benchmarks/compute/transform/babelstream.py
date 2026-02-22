# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for BabelStream operations using cuda.compute transforms.

C++ equivalent: cub/benchmarks/bench/transform/babelstream.cu

Notes:
- Migration: Python omits OffsetT axis and int128 types.
- OffsetT axis is omitted because the Python API does not expose offset type.
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
from cuda.compute import ZipIterator

# Type mapping: C++ types to NumPy dtypes
TYPE_MAP = {
    "I8": np.int8,
    "I16": np.int16,
    "F32": np.float32,
    "F64": np.float64,
}

# BabelStream constants
START_A = 11
START_B = 2
START_C = 1
START_SCALAR = -2

# Verify invariant: nstream maintains consistent workload
assert START_A == START_A + START_B + START_SCALAR * START_C


def _reset_pools():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def bench_mul(state: bench.State):
    """
    Benchmark: b[i] = c[i] * scalar
    Unary transform with scalar multiplication.
    """
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements"))

    _reset_pools()

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        c = cp.full(num_items, START_C, dtype=dtype)
        b = cp.full(num_items, START_B, dtype=dtype)

    scalar = dtype(START_SCALAR)

    def mul_op(ci):
        return ci * scalar

    transform = cuda.compute.make_unary_transform(d_in=c, d_out=b, op=mul_op)

    state.add_element_count(num_items)
    state.add_global_memory_reads(num_items * c.dtype.itemsize)
    state.add_global_memory_writes(num_items * b.dtype.itemsize)

    def launcher(launch: bench.Launch):
        transform(
            d_in=c, d_out=b, op=mul_op, num_items=num_items, stream=launch.get_stream()
        )

    state.exec(launcher)


def bench_add(state: bench.State):
    """
    Benchmark: c[i] = a[i] + b[i]
    Binary transform with addition.
    """
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements"))

    _reset_pools()

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        a = cp.full(num_items, START_A, dtype=dtype)
        b = cp.full(num_items, START_B, dtype=dtype)
        c = cp.full(num_items, START_C, dtype=dtype)

    def add_op(ai, bi):
        return ai + bi

    transform = cuda.compute.make_binary_transform(d_in1=a, d_in2=b, d_out=c, op=add_op)

    state.add_element_count(num_items)
    state.add_global_memory_reads(2 * num_items * a.dtype.itemsize)
    state.add_global_memory_writes(num_items * c.dtype.itemsize)

    def launcher(launch: bench.Launch):
        transform(
            d_in1=a,
            d_in2=b,
            d_out=c,
            op=add_op,
            num_items=num_items,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


def bench_triad(state: bench.State):
    """
    Benchmark: a[i] = b[i] + scalar * c[i]
    Binary transform with fused multiply-add.
    """
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements"))

    _reset_pools()

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        a = cp.full(num_items, START_A, dtype=dtype)
        b = cp.full(num_items, START_B, dtype=dtype)
        c = cp.full(num_items, START_C, dtype=dtype)

    scalar = dtype(START_SCALAR)

    def triad_op(bi, ci):
        return bi + scalar * ci

    transform = cuda.compute.make_binary_transform(
        d_in1=b, d_in2=c, d_out=a, op=triad_op
    )

    state.add_element_count(num_items)
    state.add_global_memory_reads(2 * num_items * a.dtype.itemsize)
    state.add_global_memory_writes(num_items * a.dtype.itemsize)

    def launcher(launch: bench.Launch):
        transform(
            d_in1=b,
            d_in2=c,
            d_out=a,
            op=triad_op,
            num_items=num_items,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


def bench_nstream(state: bench.State):
    """
    Benchmark: a[i] = a[i] + b[i] + scalar * c[i]
    Ternary transform using ZipIterator to combine (a, b, c) as input.
    """
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements"))

    _reset_pools()

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        a = cp.full(num_items, START_A, dtype=dtype)
        b = cp.full(num_items, START_B, dtype=dtype)
        c = cp.full(num_items, START_C, dtype=dtype)

    scalar = dtype(START_SCALAR)

    # Use ZipIterator to combine 3 inputs into one for unary transform
    zip_in = ZipIterator(a, b, c)

    def nstream_op(abc):
        return abc[0] + abc[1] + scalar * abc[2]

    transform = cuda.compute.make_unary_transform(d_in=zip_in, d_out=a, op=nstream_op)

    state.add_element_count(num_items)
    state.add_global_memory_reads(3 * num_items * a.dtype.itemsize)
    state.add_global_memory_writes(num_items * a.dtype.itemsize)

    def launcher(launch: bench.Launch):
        # Update ZipIterator state for each iteration
        zip_in_iter = ZipIterator(a, b, c)
        transform(
            d_in=zip_in_iter,
            d_out=a,
            op=nstream_op,
            num_items=num_items,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


# Registry of all BabelStream benchmarks
BENCHMARKS = {
    "mul": bench_mul,
    "add": bench_add,
    "triad": bench_triad,
    "nstream": bench_nstream,
}


if __name__ == "__main__":
    for name, bench_fn in BENCHMARKS.items():
        b = bench.register(bench_fn)
        b.set_name(name)
        b.add_string_axis("T", list(TYPE_MAP.keys()))
        b.add_int64_power_of_two_axis("Elements", range(16, 33, 4))

    bench.run_all_benchmarks(sys.argv)
