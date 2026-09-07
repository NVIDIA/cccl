# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run a portable block load, exclusive scan, and store with Numba."""

from __future__ import annotations

import numpy as np
from numba_cuda_mlir import cuda

from cuda import coop

THREADS = 32
ITEMS_PER_THREAD = 2
TILE_ITEMS = THREADS * ITEMS_PER_THREAD


# docs: start numba-common-block-scan
@cuda.jit
def block_scan_kernel(values, prefixes):
    block = coop.this_block()
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=np.int32)
    loaded = coop.load(block, values, items)
    scanned = coop.exclusive_sum(block, loaded)
    coop.store(block, prefixes, scanned)


# docs: end numba-common-block-scan


def run_example() -> np.ndarray:
    """Run one tile and return its exclusive prefix sum."""

    values = np.arange(1, TILE_ITEMS + 1, dtype=np.int32)
    prefixes = np.zeros_like(values)
    block_scan_kernel[1, THREADS](values, prefixes)
    cuda.synchronize()

    expected = np.zeros_like(values)
    expected[1:] = np.cumsum(values[:-1], dtype=np.int32)
    np.testing.assert_array_equal(prefixes, expected)
    return prefixes


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
