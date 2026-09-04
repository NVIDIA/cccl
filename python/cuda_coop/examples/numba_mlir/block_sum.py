# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sum a per-thread payload with the portable ``cuda.coop`` API."""

import numpy as np
from numba_cuda_mlir import cuda, types

from cuda import coop

_THREADS = 64
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD


@cuda.jit
def block_sum(source, output):
    """Reduce a full block tile and let only the block root store the result."""

    thread = cuda.threadIdx.x
    values = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        values[item] = source[thread * _ITEMS_PER_THREAD + item]
    total = coop.sum(coop.this_block(), values, broadcast=False)
    if thread == 0:
        output[0] = total


def main() -> None:
    source = np.arange(_TILE_ITEMS, dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)

    block_sum[1, _THREADS](source, output)

    expected = np.asarray([source.sum()], dtype=np.int32)
    np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    main()
