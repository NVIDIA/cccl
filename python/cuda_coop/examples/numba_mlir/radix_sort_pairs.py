# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block radix sort of key-value pairs with Numba-CUDA-MLIR."""

import numpy as np
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as coop

THREADS = 32
ITEMS_PER_THREAD = 2
TILE_ITEMS = THREADS * ITEMS_PER_THREAD


# docs: start numba-mlir-radix-sort-pairs
@cuda.jit
def radix_sort_pairs_kernel(keys_in, values_in, keys_out, values_out):
    tid = cuda.threadIdx.x
    block = coop.this_block()
    keys = coop.ThreadData(items_per_thread=ITEMS_PER_THREAD, dtype=keys_in.dtype)
    values = coop.ThreadData(items_per_thread=ITEMS_PER_THREAD, dtype=values_in.dtype)

    for item in range(ITEMS_PER_THREAD):
        offset = tid * ITEMS_PER_THREAD + item
        keys[item] = keys_in[offset]
        values[item] = values_in[offset]

    sorted_keys, sorted_values = coop.radix_sort_pairs(block, keys, values)

    for item in range(ITEMS_PER_THREAD):
        offset = tid * ITEMS_PER_THREAD + item
        keys_out[offset] = sorted_keys[item]
        values_out[offset] = sorted_values[item]


# docs: end numba-mlir-radix-sort-pairs


def run_example() -> tuple[np.ndarray, np.ndarray]:
    """Sort one tile and return the ordered pairs."""

    keys = np.arange(TILE_ITEMS, 0, -1, dtype=np.int32)
    values = keys * np.int32(10) + np.int32(3)
    keys_out = np.zeros_like(keys)
    values_out = np.zeros_like(values)
    radix_sort_pairs_kernel[1, THREADS](keys, values, keys_out, values_out)
    expected_keys = np.sort(keys)
    np.testing.assert_array_equal(keys_out, expected_keys)
    np.testing.assert_array_equal(values_out, expected_keys * 10 + 3)
    return keys_out, values_out


def main() -> int:
    keys, values = run_example()
    print({"keys": keys, "values": values})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
