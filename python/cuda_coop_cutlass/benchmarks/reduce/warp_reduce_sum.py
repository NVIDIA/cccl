# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for warp reduce sum using cuda.coop.cutlass.sum.

C++ equivalent: cub/benchmarks/bench/reduce/warp_reduce_sum.cu
(with warp_reduce_base.cuh)

Notes:
- cuda.coop.cutlass has no int128, half, bfloat16 or complex value types, so
  I128, F16, BF16, C32 and C64 are absent. Restrict the C++ run to the same
  T{ct} values when comparing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cuda.bench as bench
import cutlass
from _device_side import run_benchmark, run_device_side_benchmark

import cuda.coop.cutlass as coop

TYPE_MAP = {
    "I8": cutlass.Int8,
    "I16": cutlass.Int16,
    "I32": cutlass.Int32,
    "I64": cutlass.Int64,
    "F32": cutlass.Float32,
    "F64": cutlass.Float64,
}


def warp_reduce(thread_data):
    return coop.sum(coop.this_warp(), thread_data)


def bench_warp_reduce_sum(state: bench.State):
    value_type = TYPE_MAP[state.get_string("T{ct}")]
    run_device_side_benchmark(
        state, warp_reduce, value_type, block_size=256, unroll_factor=128
    )


def configure(benchmark):
    benchmark.set_name("base")
    benchmark.add_string_axis("T{ct}", list(TYPE_MAP))


if __name__ == "__main__":
    run_benchmark(bench_warp_reduce_sum, configure, sys.argv)
