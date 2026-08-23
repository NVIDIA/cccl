# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reuse rewrite-managed temporary storage across block primitives."""

import numpy as np
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as coop

THREADS = 32
ITEMS_PER_THREAD = 2
TILE_ITEMS = THREADS * ITEMS_PER_THREAD


# docs: start numba-mlir-shared-temp-storage
@cuda.jit
def shared_temp_storage_kernel(values, result):
    block = coop.this_block()
    scratch = coop.TempStorage()
    items = coop.ThreadData(items_per_thread=ITEMS_PER_THREAD, dtype=values.dtype)
    loaded = coop.load(block, values, items, temp_storage=scratch)
    scanned = coop.exclusive_sum(block, loaded, temp_storage=scratch)
    sorted_values = coop.radix_sort_keys(
        block,
        scanned,
        begin_bit=0,
        end_bit=16,
        temp_storage=scratch,
    )
    coop.store(block, result, sorted_values, temp_storage=scratch)


# docs: end numba-mlir-shared-temp-storage


def run_example() -> np.ndarray:
    """Run the chain and return its sorted exclusive prefixes."""

    values = ((np.arange(TILE_ITEMS, dtype=np.uint32) * 7) % 17).astype(np.uint32)
    result = np.zeros_like(values)
    shared_temp_storage_kernel[1, THREADS](values, result)
    prefixes = np.concatenate(
        (np.zeros(1, dtype=np.uint32), np.cumsum(values[:-1], dtype=np.uint32))
    )
    expected = np.sort(prefixes)
    np.testing.assert_array_equal(result, expected)
    return result


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
