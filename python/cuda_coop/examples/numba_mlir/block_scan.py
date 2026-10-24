# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load one full block tile and store its exclusive prefix sum.

Each thread contributes consecutive items in blocked order. The scan returns
a separate payload; the loaded values remain available to the kernel.
"""

from __future__ import annotations

import numpy as np
from numba_cuda_mlir import cuda

from cuda import coop

THREADS = 32


# docs: start numba-block-scan
@cuda.jit
def block_scan_kernel(values, prefixes, items_per_thread):
    block = coop.this_block()
    items = coop.ThreadData(items_per_thread)
    coop.load(block, values, items)
    scanned = coop.exclusive_sum(block, items)
    coop.store(block, prefixes, scanned)


# docs: end numba-block-scan


def run_example(items_per_thread: int = 4) -> np.ndarray:
    """Run one tile and return its exclusive prefix sum."""

    tile_items = THREADS * items_per_thread
    values = np.arange(1, tile_items + 1, dtype=np.int32)
    prefixes = np.zeros_like(values)
    block_scan_kernel[1, THREADS](values, prefixes, items_per_thread)
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
