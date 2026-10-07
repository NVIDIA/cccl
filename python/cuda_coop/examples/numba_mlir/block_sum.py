# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sum one full input tile and write the result from thread zero.

Each thread contributes several consecutive values through ``ThreadData``.
All threads reduce their values; only thread zero uses the returned sum.
"""

import numpy as np
from numba_cuda_mlir import cuda

from cuda import coop

_THREADS = 64


@cuda.jit
def block_sum(source, output, items_per_thread):
    """Reduce a full block tile and let only the block root store the result."""

    thread = cuda.threadIdx.x
    values = coop.ThreadData(items_per_thread)
    for item in range(items_per_thread):
        values[item] = source[thread * items_per_thread + item]
    total = coop.sum(coop.this_block(), values)
    if thread == 0:
        output[0] = total


def main(items_per_thread: int = 4) -> None:
    """Launch one block with a full tile and check its sum against NumPy."""

    tile_items = _THREADS * items_per_thread
    source = np.arange(tile_items, dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)

    block_sum[1, _THREADS](source, output, items_per_thread)

    expected = np.asarray([source.sum()], dtype=np.int32)
    np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    main()
