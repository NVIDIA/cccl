# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Observe a partial block tile loaded with the portable ``cuda.coop`` API."""

import numpy as np
from numba_cuda_mlir import cuda

from cuda import coop

_THREADS = 32
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD
_SOURCE_OFFSET = 3


@cuda.jit
def block_load(source, observed, valid_items):
    """Load a tile; invalid payload slots receive a caller-selected default."""

    thread = cuda.threadIdx.x
    payload = coop.ThreadData(_ITEMS_PER_THREAD)
    loaded = coop.load(
        coop.this_block(),
        source,
        payload,
        algorithm="direct",
        valid_items=valid_items,
        oob_default=-1,
        offset=_SOURCE_OFFSET,
    )
    for item in range(_ITEMS_PER_THREAD):
        observed[thread * _ITEMS_PER_THREAD + item] = loaded[item]


def main() -> None:
    valid_items = _TILE_ITEMS - 7
    source = np.arange(_SOURCE_OFFSET + _TILE_ITEMS, dtype=np.int32)
    observed = np.zeros(_TILE_ITEMS, dtype=np.int32)

    block_load[1, _THREADS](source, observed, np.int32(valid_items))

    expected = np.full(_TILE_ITEMS, -1, dtype=np.int32)
    expected[:valid_items] = source[_SOURCE_OFFSET : _SOURCE_OFFSET + valid_items]
    np.testing.assert_array_equal(observed, expected)


if __name__ == "__main__":
    main()
