# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable root Load, Reduce, and Store with Numba-CUDA-MLIR."""

# docs: start portable-root-sum
import numpy as np
from numba_cuda_mlir import cuda

from cuda import coop

THREADS = 32
ITEMS_PER_THREAD = 2
TILE_ITEMS = THREADS * ITEMS_PER_THREAD


@cuda.jit
def portable_root_sum_kernel(values, copied, totals):
    block = coop.this_block()
    items = coop.ThreadData(2, dtype=np.int32)
    loaded = coop.load(block, values, items)
    coop.store(block, copied, loaded)
    total = coop.reduce(block, loaded)
    if block.rank() == 0:
        totals[0] = total


# docs: end portable-root-sum


def run_example() -> int:
    """Run one block, validate its round trip and sum, and return the sum."""

    values = np.arange(1, TILE_ITEMS + 1, dtype=np.int32)
    copied = np.zeros_like(values)
    totals = np.zeros(1, dtype=np.int32)

    portable_root_sum_kernel[1, THREADS](values, copied, totals)
    cuda.synchronize()

    expected_total = int(np.sum(values, dtype=np.int64))
    np.testing.assert_array_equal(copied, values)
    np.testing.assert_array_equal(
        totals,
        np.asarray([expected_total], dtype=np.int32),
    )
    return expected_total


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
