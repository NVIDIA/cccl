# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

import numba
import numpy as np
from numba import cuda

import cuda.bench as bench

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from device_side_benchmark import (  # isort: skip # type: ignore[import-not-found] # noqa: E402
    make_unrolled_kernel,
    get_grid_size,
)

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def bench_warp_reduce(state: bench.State):
    dtype_str = state.get_string("T{ct}")
    algorithm_str = state.get_string("Algorithm{ct}")

    if algorithm_str == "warp_min" and dtype_str == "F16":
        state.skip("custom ops not supported for F16, which is needed for warp_min")

    types_map = {
        "I8": np.int8,
        "I16": np.int16,
        "I32": np.int32,
        "I64": np.int64,
        "F16": np.float16,
        "F32": np.float32,
        "F64": np.float64,
    }

    dtype = types_map[dtype_str]

    numba_dtype = numba.from_dtype(dtype)
    block_size = 256
    unroll_factor = 128

    benchmark_kernel = make_unrolled_kernel(
        block_size, algorithm_str, unroll_factor, numba_dtype
    )

    sink_buffer = cuda.device_array(16, dtype=np.int32)

    # This calls the kernel (and then immediately synchronizes the device) to
    # force compilation so we can extract occupancy info.
    grid_size = get_grid_size(
        state.get_device(), block_size, benchmark_kernel, sink_buffer
    )

    def launcher(_: bench.Launch):
        benchmark_kernel[grid_size, block_size](sink_buffer)

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_warp_reduce)
    b.add_string_axis("T{ct}", ["I8", "I16", "I32", "I64", "F16", "F32", "F64"])
    b.add_string_axis("Algorithm{ct}", ["warp_sum", "warp_min"])
    bench.run_all_benchmarks(sys.argv)
