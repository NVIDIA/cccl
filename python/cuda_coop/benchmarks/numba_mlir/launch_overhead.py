# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Host-side cost of enqueuing the warp sum benchmark kernel from Python.

Counterpart of ``cpp/launch_overhead.cu``; prints one JSON object per launch
style. Launches go out in batches below the launch queue depth with an untimed
synchronize between batches, so this is host enqueue time, not GPU time.

    configured: ``k = kernel[grid, block, stream]`` once, then ``k(args)``
    indexed:    ``kernel[grid, block, stream](args)`` on every launch
"""

import json
import os
import statistics
import sys
import time

import numpy as np
from numba_cuda_mlir import cuda

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from device_side_benchmark import make_benchmark_kernel

BLOCK_SIZE = 256
BATCH = 256
BATCHES = 200


def measure(launch, stream):
    launch()
    stream.synchronize()
    per_launch_us = []
    for _ in range(BATCHES):
        start = time.perf_counter()
        for _ in range(BATCH):
            launch()
        stop = time.perf_counter()
        stream.synchronize()
        per_launch_us.append((stop - start) * 1e6 / BATCH)
    return per_launch_us


def main():
    kernel = make_benchmark_kernel("sum", "I32", unroll_factor=1, unroll=True)
    sink_buffer = cuda.device_array(16, dtype=np.int32)
    stream = cuda.stream()
    configured = kernel[1, BLOCK_SIZE, stream]
    styles = {
        "configured": lambda: configured(sink_buffer),
        "indexed": lambda: kernel[1, BLOCK_SIZE, stream](sink_buffer),
    }
    for style, launch in styles.items():
        samples = measure(launch, stream)
        print(
            json.dumps(
                {
                    "side": "python",
                    "launch": style,
                    "batch": BATCH,
                    "batches": BATCHES,
                    "median_us": round(statistics.median(samples), 4),
                    "min_us": round(min(samples), 4),
                    "max_us": round(max(samples), 4),
                }
            )
        )


if __name__ == "__main__":
    main()
