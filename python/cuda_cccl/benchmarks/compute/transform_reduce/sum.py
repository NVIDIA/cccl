#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for transform reduce sum using cuda.compute.

C++ equivalent: cub/benchmarks/bench/transform_reduce/sum.cu

Notes:
- Uses TransformIterator with a square operation
- OffsetT axis from C++ is fixed to Python default (int64)
- Migration: Python fixes offsets and omits int128/complex types.
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import SIGNED_TYPES as TYPE_MAP
from utils import as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import OpKind, TransformIterator, make_reduce_into


def square_op(x):
    return x * x


def bench_transform_reduce_sum(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_items = int(state.get_int64("Elements{io}"))

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        d_in = generate_data_with_entropy(num_items, dtype, "1.000", alloc_stream)
        d_out = cp.empty(1, dtype=dtype)

    transform_it = TransformIterator(d_in, square_op)
    h_init = np.zeros(1, dtype=dtype)

    reducer = make_reduce_into(
        d_in=transform_it, d_out=d_out, op=OpKind.PLUS, h_init=h_init
    )

    temp_storage_bytes = reducer(
        temp_storage=None,
        d_in=transform_it,
        d_out=d_out,
        num_items=num_items,
        op=OpKind.PLUS,
        h_init=h_init,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    state.add_element_count(num_items)
    state.add_global_memory_reads(num_items * d_in.dtype.itemsize, "Size")
    state.add_global_memory_writes(1 * d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        reducer(
            temp_storage=temp_storage,
            d_in=transform_it,
            d_out=d_out,
            num_items=num_items,
            op=OpKind.PLUS,
            h_init=h_init,
            stream=launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_transform_reduce_sum)
    b.set_name("base")
    b.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))

    bench.run_all_benchmarks(sys.argv)
