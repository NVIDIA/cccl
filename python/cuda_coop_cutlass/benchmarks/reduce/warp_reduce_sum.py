# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for warp reduce sum using cuda.coop.cutlass.sum.

C++ equivalent: cub/benchmarks/bench/reduce/warp_reduce_sum.cu
(with warp_reduce_base.cuh)

Compare the six T{ct} values shared by both providers. The native benchmark
also registers complex types and, when available, I128, F16, BF16 and
__float128. The native type list omits the unsigned integers supported by
cuda.coop.cutlass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cuda.bench as bench
import cutlass
from _device_side import run_benchmark, run_device_side_benchmark
from cutlass import cute

import cuda.coop.cutlass as coop

TYPE_MAP = {
    "I8": cutlass.Int8,
    "I16": cutlass.Int16,
    "I32": cutlass.Int32,
    "I64": cutlass.Int64,
    "F32": cutlass.Float32,
    "F64": cutlass.Float64,
}


@cute.jit
def warp_reduce_sum_rounds(seed, rounds: cutlass.Constexpr):
    lane = cute.arch.lane_idx()
    data = seed
    contribution = seed
    for _ in cutlass.range(rounds):
        total = coop.sum(coop.this_warp(), data)
        # Only lane zero owns the sum. Other lanes reuse their seed for round
        # two, then alternate its sign.
        if lane == 0:
            data = total
        else:
            data = contribution
        contribution = type(seed)(-contribution)
    return data


def warp_reduce(value_type, rounds):
    bit = (
        cutlass.Uint32(cute.arch.clock())
        >> cutlass.Uint32(cute.arch.lane_idx())
    ) & 1
    seed = value_type(2 * cutlass.Int32(bit) - 1)
    return warp_reduce_sum_rounds(seed, rounds)


def bench_warp_reduce_sum(state: bench.State):
    value_type = TYPE_MAP[state.get_string("T{ct}")]
    run_device_side_benchmark(
        state, warp_reduce, value_type, block_size=256, rounds=128
    )


def configure(benchmark):
    benchmark.set_name("base")
    benchmark.add_string_axis("T{ct}", list(TYPE_MAP))


if __name__ == "__main__":
    run_benchmark(bench_warp_reduce_sum, configure, sys.argv)
