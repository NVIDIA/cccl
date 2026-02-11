# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python benchmark for heavy transform using cuda.compute.

C++ equivalent: cub/benchmarks/bench/transform/heavy.cu
"""

import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numba
import numpy as np
from numba import cuda as lang
from utils import as_cupy_stream

import cuda.bench as bench
import cuda.compute

HEAVINESS_VALUES = (32, 64, 128, 256)


def _heavy_op_32(data):
    reg = lang.local.array(shape=32, dtype=numba.uint32)
    reg[0] = data
    for i in range(1, 32):
        x = reg[i - 1]
        reg[i] = x * x + 1
    for i in range(32):
        x = reg[i]
        reg[i] = (x * x) % 19
    for i in range(32):
        reg[i] = reg[32 - i - 1] * reg[i]
    out = data - data  # uint32(0)
    for i in range(32):
        out += reg[i]
    return out


def _heavy_op_64(data):
    reg = lang.local.array(shape=64, dtype=numba.uint32)
    reg[0] = data
    for i in range(1, 64):
        x = reg[i - 1]
        reg[i] = x * x + 1
    for i in range(64):
        x = reg[i]
        reg[i] = (x * x) % 19
    for i in range(64):
        reg[i] = reg[64 - i - 1] * reg[i]
    out = data - data
    for i in range(64):
        out += reg[i]
    return out


def _heavy_op_128(data):
    reg = lang.local.array(shape=128, dtype=numba.uint32)
    reg[0] = data
    for i in range(1, 128):
        x = reg[i - 1]
        reg[i] = x * x + 1
    for i in range(128):
        x = reg[i]
        reg[i] = (x * x) % 19
    for i in range(128):
        reg[i] = reg[128 - i - 1] * reg[i]
    out = data - data
    for i in range(128):
        out += reg[i]
    return out


def _heavy_op_256(data):
    reg = lang.local.array(shape=256, dtype=numba.uint32)
    reg[0] = data
    for i in range(1, 256):
        x = reg[i - 1]
        reg[i] = x * x + 1
    for i in range(256):
        x = reg[i]
        reg[i] = (x * x) % 19
    for i in range(256):
        reg[i] = reg[256 - i - 1] * reg[i]
    out = data - data
    for i in range(256):
        out += reg[i]
    return out


_HEAVY_OPS = {
    32: _heavy_op_32,
    64: _heavy_op_64,
    128: _heavy_op_128,
    256: _heavy_op_256,
}


def bench_heavy(state: bench.State):
    # Axes
    n_regs = int(state.get_string("Heaviness"))
    size = state.get_int64("Elements")

    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        d_in = cp.arange(size, dtype=np.uint32)
        d_out = cp.empty(size, dtype=np.uint32)

    op = _HEAVY_OPS[n_regs]
    transform = cuda.compute.make_unary_transform(d_in=d_in, d_out=d_out, op=op)

    state.add_element_count(size)
    state.add_global_memory_reads(size * d_in.dtype.itemsize)
    state.add_global_memory_writes(size * d_out.dtype.itemsize)

    def launcher(launch: bench.Launch):
        transform(
            d_in=d_in, d_out=d_out, num_items=size, op=op, stream=launch.get_stream()
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(bench_heavy)
    b.set_name("heavy")
    b.add_string_axis("Heaviness", [str(v) for v in HEAVINESS_VALUES])
    b.add_int64_power_of_two_axis("Elements", range(16, 33, 4))
    bench.run_all_benchmarks(sys.argv)
