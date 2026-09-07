# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Partial-tile TopK selection with Numba-CUDA-MLIR."""

import numpy as np
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as coop

THREADS = 32
ITEMS_PER_THREAD = 2
TILE_ITEMS = THREADS * ITEMS_PER_THREAD


# docs: start numba-mlir-topk-partial
@cuda.jit
def topk_partial_kernel(keys_in, values_in, keys_out, values_out, k, num_valid):
    tid = cuda.threadIdx.x
    block = coop.this_block()
    keys = coop.ThreadData(items_per_thread=ITEMS_PER_THREAD, dtype=keys_in.dtype)
    values = coop.ThreadData(items_per_thread=ITEMS_PER_THREAD, dtype=values_in.dtype)

    for item in range(ITEMS_PER_THREAD):
        offset = tid * ITEMS_PER_THREAD + item
        keys[item] = keys_in[offset]
        values[item] = values_in[offset]

    selected_keys, selected_values = coop.topk_min_pairs(
        block,
        keys,
        values,
        k,
        valid_items=num_valid,
    )

    for item in range(ITEMS_PER_THREAD):
        offset = tid * ITEMS_PER_THREAD + item
        keys_out[offset] = selected_keys[item]
        values_out[offset] = selected_values[item]


# docs: end numba-mlir-topk-partial


def run_example() -> list[tuple[int, int]]:
    """Select the smallest keys from the valid part of one tile."""

    k = 7
    num_valid = TILE_ITEMS - 9
    keys = ((np.arange(TILE_ITEMS, dtype=np.int32) * 29) % TILE_ITEMS).astype(np.int32)
    values = keys * np.int32(11) + np.int32(3)
    keys[num_valid:] = np.int32(-1000)
    values[num_valid:] = np.int32(-1)
    keys_out = np.zeros_like(keys)
    values_out = np.zeros_like(values)

    topk_partial_kernel[1, THREADS](
        keys,
        values,
        keys_out,
        values_out,
        np.int32(k),
        np.int32(num_valid),
    )

    actual = sorted(
        (int(key), int(value))
        for key, value in zip(keys_out[:k], values_out[:k], strict=True)
    )
    expected_keys = np.sort(keys[:num_valid])[:k]
    expected = sorted((int(key), int(key * 11 + 3)) for key in expected_keys)
    assert actual == expected
    return actual


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
