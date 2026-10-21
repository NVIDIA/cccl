# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Device-side warp reduction benchmark, the counterpart of
``cub/benchmarks/bench/reduce/warp_reduce_sum.cu``.

Run ``check_warp_reduce.py`` first; this script only measures.
"""

import os
import sys

from numba_cuda_mlir import cuda

from cuda import bench

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from device_side_benchmark import (  # isort: skip
    NUMPY_TYPES,
    make_benchmark_kernel,
    occupancy_grid_size,
)

BLOCK_SIZE = 256
TYPES = ["I8", "I16", "I32", "I64", "F16", "F32", "F64"]
UNSUPPORTED_TYPES = {
    "F16": "cuda.coop warp sum and reduce reject float16 in both the common "
    "and the numba_mlir APIs",
}


def bench_warp_reduce(state: bench.State):
    type_name = state.get_string("T{ct}")
    algorithm = state.get_string("Algorithm{ct}")
    unroll_factor = state.get_int64("UnrollFactor")
    unroll = state.get_string("Unroll") == "full"
    if type_name in UNSUPPORTED_TYPES:
        state.skip(UNSUPPORTED_TYPES[type_name])
        return

    kernel = make_benchmark_kernel(
        algorithm,
        type_name,
        block_size=BLOCK_SIZE,
        unroll_factor=unroll_factor,
        unroll=unroll,
    )
    sink_buffer = cuda.device_array(16, dtype=NUMPY_TYPES[type_name])
    grid_size = occupancy_grid_size(kernel, BLOCK_SIZE, sink_buffer)

    _, handle = state.get_stream().__cuda_stream__()
    stream = cuda.external_stream(handle)
    configured = kernel[grid_size, BLOCK_SIZE, stream]
    # The launch shape is part of the compile key; compile it before timing.
    configured(sink_buffer)
    stream.synchronize()

    def launcher(_: bench.Launch):
        configured(sink_buffer)

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_warp_reduce)
    b.set_name("base")
    b.add_string_axis("T{ct}", TYPES)
    b.add_string_axis("Algorithm{ct}", ["sum", "reduce_add"])
    b.add_int64_axis("UnrollFactor", [128])
    b.add_string_axis("Unroll", ["loop"])
    bench.run_all_benchmarks(sys.argv)
