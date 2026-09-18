# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Store a partial block tile with the qualified Numba-CUDA-MLIR API."""

import numpy as np
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as coop

_THREADS = 32
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD
_DESTINATION_OFFSET = 5


@cuda.jit
def block_store(source, destination, valid_items):
    """Store the valid tile prefix while retaining the destination suffix."""

    thread = cuda.threadIdx.x
    payload = coop.ThreadData(_ITEMS_PER_THREAD)
    for item in range(_ITEMS_PER_THREAD):
        payload[item] = source[thread * _ITEMS_PER_THREAD + item]
    coop.store(
        coop.this_block(),
        destination,
        payload,
        algorithm="direct",
        valid_items=valid_items,
        offset=_DESTINATION_OFFSET,
    )


def main() -> None:
    valid_items = _TILE_ITEMS - 9
    source = np.arange(_TILE_ITEMS, dtype=np.int32) + 100
    destination = np.full(
        _DESTINATION_OFFSET + _TILE_ITEMS,
        -1,
        dtype=np.int32,
    )

    block_store[1, _THREADS](source, destination, np.int32(valid_items))

    expected = np.full_like(destination, -1)
    expected[_DESTINATION_OFFSET : _DESTINATION_OFFSET + valid_items] = source[
        :valid_items
    ]
    np.testing.assert_array_equal(destination, expected)


if __name__ == "__main__":
    main()
