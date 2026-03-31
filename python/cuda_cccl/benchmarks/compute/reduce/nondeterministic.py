# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for nondeterministic reduce sum using cuda.compute.reduce_into.

C++ equivalent: cub/benchmarks/bench/reduce/nondeterministic.cu

Notes:
- Uses Determinism.NOT_GUARANTEED
- C++ tests int32, int64, float, double
- Migration: Python fixes offsets; C++ exposes an OffsetT axis.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import ALL_TYPES as _ALL_TYPES
from utils import as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import Determinism, OpKind, make_reduce_into

TYPE_MAP = {k: _ALL_TYPES[k] for k in ("I32", "I64", "F32", "F64")}


def bench_reduce_nondeterministic(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements{io}"))

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        d_in = generate_data_with_entropy(num_items, dtype, "1.000", alloc_stream)
        d_out = cp.empty(1, dtype=dtype)

    h_init = np.zeros(1, dtype=dtype)

    reducer = make_reduce_into(
        d_in,
        d_out,
        OpKind.PLUS,
        h_init,
        determinism=Determinism.NOT_GUARANTEED,
    )

    temp_storage_bytes = reducer(None, d_in, d_out, OpKind.PLUS, num_items, h_init)
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    state.add_element_count(num_items)
    state.add_global_memory_reads(num_items * d_in.dtype.itemsize, "Size")
    state.add_global_memory_writes(1 * d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        reducer(
            temp_storage,
            d_in,
            d_out,
            OpKind.PLUS,
            num_items,
            h_init,
            launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_reduce_nondeterministic)
    b.set_name("base")
    b.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    bench.run_all_benchmarks(sys.argv)
