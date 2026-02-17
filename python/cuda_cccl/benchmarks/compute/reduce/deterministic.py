# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for deterministic reduce sum using cuda.compute.reduce_into.

C++ equivalent: cub/benchmarks/bench/reduce/deterministic.cu

Notes:
- Uses Determinism.RUN_TO_RUN
- C++ only tests float and double
- Migration: Python uses int64 offsets; C++ uses int offsets.
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
from cuda.compute import Determinism, OpKind, make_reduce_into

TYPE_MAP = {
    "F32": np.float32,
    "F64": np.float64,
}


def bench_reduce_deterministic(state: bench.State):
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements"))

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        d_in = cp.random.random(num_items, dtype=dtype)
        d_out = cp.empty(1, dtype=dtype)

    h_init = np.zeros(1, dtype=dtype)

    reducer = make_reduce_into(
        d_in=d_in,
        d_out=d_out,
        op=OpKind.PLUS,
        h_init=h_init,
        determinism=Determinism.RUN_TO_RUN,
    )

    temp_storage_bytes = reducer(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        op=OpKind.PLUS,
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
            op=OpKind.PLUS,
            num_items=num_items,
            h_init=h_init,
            stream=launch.get_stream(),
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_reduce_deterministic)
    b.set_name("base")  # Match C++ benchmark name
    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))
    bench.run_all_benchmarks(sys.argv)
