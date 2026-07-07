# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for reduce min operation using cuda.compute.reduce_into.

C++ equivalent: cub/benchmarks/bench/reduce/min.cu

Notes:
- Uses OpKind.MINIMUM for minimum reduction
- C++ uses cuda::minimum<> which CUB recognizes for optimized code paths (DPX on Hopper+)
- int128 and complex32 are not supported by cupy
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import FUNDAMENTAL_TYPES as TYPE_MAP
from utils import as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import OpKind, make_reduce_into


def bench_reduce_min(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements{io}"))

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        d_in = generate_data_with_entropy(num_items, dtype, "1.000", alloc_stream)
        d_out = cp.empty(1, dtype=dtype)

    # Initial value for min reduction (max value of type)
    if np.issubdtype(dtype, np.integer):
        init_val = np.iinfo(dtype).max
    else:
        init_val = np.finfo(dtype).max
    h_init = np.array([init_val], dtype=dtype)

    reducer = make_reduce_into(d_in=d_in, d_out=d_out, op=OpKind.MINIMUM, h_init=h_init)

    temp_storage_bytes = reducer(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        num_items=num_items,
        op=OpKind.MINIMUM,
        h_init=h_init,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    state.add_element_count(num_items)
    state.add_global_memory_reads(num_items * d_in.dtype.itemsize, "Size")
    state.add_global_memory_writes(d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        reducer(
            temp_storage=temp_storage,
            d_in=d_in,
            d_out=d_out,
            num_items=num_items,
            op=OpKind.MINIMUM,
            h_init=h_init,
            stream=launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_reduce_min)
    b.set_name("base")

    b.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))

    bench.run_all_benchmarks(sys.argv)
