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

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import SIGNED_TYPES as TYPE_MAP
from utils import as_cupy_stream

import cuda.bench as bench
from cuda.compute import OpKind, make_reduce_into


def bench_reduce_min(state: bench.State):
    type_str = state.get_string("T")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements"))

    # Setup data - use random values like C++ generate()
    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            d_in = cp.random.randint(info.min, info.max, size=num_items, dtype=dtype)
        else:
            # For floats, use full range
            d_in = cp.random.uniform(-1e6, 1e6, size=num_items).astype(dtype)

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
        op=OpKind.MINIMUM,
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
            op=OpKind.MINIMUM,
            num_items=num_items,
            h_init=h_init,
            stream=launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_reduce_min)
    b.set_name("base")

    b.add_string_axis("T", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements", range(16, 29, 4))  # [16, 20, 24, 28]

    bench.run_all_benchmarks(sys.argv)
