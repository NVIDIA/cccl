# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load a partial tile with an offset and a default for out-of-bounds items."""

import numpy as np
from numba_cuda_mlir import cuda

from cuda import coop

_THREADS = 32
_SOURCE_OFFSET = 3


@cuda.jit
def block_load(source, observed, valid_items, items_per_thread):
    """Load a tile; invalid payload slots receive the default value -1."""

    thread = cuda.threadIdx.x
    payload = coop.ThreadData(items_per_thread)
    coop.load(
        coop.this_block(),
        source,
        payload,
        algorithm="direct",
        valid_items=valid_items,
        oob_default=-1,
        offset=_SOURCE_OFFSET,
    )
    for item in range(items_per_thread):
        observed[thread * items_per_thread + item] = payload[item]


def main(items_per_thread: int = 4) -> None:
    tile_items = _THREADS * items_per_thread
    valid_items = tile_items - 7
    source = np.arange(_SOURCE_OFFSET + tile_items, dtype=np.int32)
    observed = np.zeros(tile_items, dtype=np.int32)

    block_load[1, _THREADS](
        source, observed, np.int32(valid_items), items_per_thread
    )

    expected = np.full(tile_items, -1, dtype=np.int32)
    expected[:valid_items] = source[
        _SOURCE_OFFSET : _SOURCE_OFFSET + valid_items
    ]
    np.testing.assert_array_equal(observed, expected)


if __name__ == "__main__":
    main()
