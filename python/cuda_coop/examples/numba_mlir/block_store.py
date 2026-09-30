# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Store a partial tile at an offset, preserving the rest of the destination."""

import numpy as np
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as numba_coop

_THREADS = 32
_DESTINATION_OFFSET = 5


@cuda.jit
def block_store(source, destination, valid_items, items_per_thread):
    """Store the valid tile prefix while retaining the destination suffix."""

    thread = cuda.threadIdx.x
    payload = numba_coop.ThreadData(items_per_thread)
    for item in range(items_per_thread):
        payload[item] = source[thread * items_per_thread + item]
    numba_coop.store(
        numba_coop.this_block(),
        destination,
        payload,
        algorithm="direct",
        valid_items=valid_items,
        offset=_DESTINATION_OFFSET,
    )


def main(items_per_thread: int = 4) -> None:
    tile_items = _THREADS * items_per_thread
    valid_items = tile_items - 9
    source = np.arange(tile_items, dtype=np.int32) + 100
    destination = np.full(
        _DESTINATION_OFFSET + tile_items,
        -1,
        dtype=np.int32,
    )

    block_store[1, _THREADS](
        source, destination, np.int32(valid_items), items_per_thread
    )

    expected = np.full_like(destination, -1)
    expected[_DESTINATION_OFFSET : _DESTINATION_OFFSET + valid_items] = source[
        :valid_items
    ]
    np.testing.assert_array_equal(destination, expected)


if __name__ == "__main__":
    main()
