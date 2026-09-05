# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compose block load, scan, and store in one Numba-CUDA-MLIR kernel."""

import numpy as np
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as coop

THREADS = 32
ITEMS_PER_THREAD = 2
TILE_ITEMS = THREADS * ITEMS_PER_THREAD


# docs: start numba-mlir-load-scan-store
@cuda.jit
def load_scan_store_kernel(values, prefixes):
    block = coop.this_block()
    items = coop.ThreadData(items_per_thread=ITEMS_PER_THREAD)
    loaded = coop.load(block, values, items)
    scanned = coop.exclusive_sum(block, loaded)
    coop.store(block, prefixes, scanned)


# docs: end numba-mlir-load-scan-store


def run_example() -> np.ndarray:
    """Run one tile and return its exclusive prefix sum."""

    values = np.arange(1, TILE_ITEMS + 1, dtype=np.int32)
    prefixes = np.zeros_like(values)
    load_scan_store_kernel[1, THREADS](values, prefixes)
    expected = np.concatenate(
        (np.zeros(1, dtype=np.int32), np.cumsum(values[:-1], dtype=np.int32))
    )
    np.testing.assert_array_equal(prefixes, expected)
    return prefixes


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
