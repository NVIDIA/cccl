# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Import the kernel DSL before cuda.coop so automatic registration sees it.
# isort: off
import numpy as np
from numba_cuda_mlir import cuda

from cuda import coop  # First breakpoint: observe backend registration.
# isort: on


@cuda.jit
def copy_tile(source, destination, items_per_thread):
    block = coop.this_block()
    items = coop.ThreadData(items_per_thread)
    coop.load(block, source, items, algorithm="direct")
    coop.store(block, destination, items, algorithm="direct")


def main(items_per_thread=2):
    source = np.arange(128 * items_per_thread, dtype=np.int32)
    destination = np.zeros_like(source)

    copy_tile[1, 128](
        source, destination, items_per_thread
    )  # Compile, link, and launch.
    cuda.synchronize()
    np.testing.assert_array_equal(destination, source)
    print("First launch: copy verified")

    destination.fill(-1)
    copy_tile[1, 128](
        source, destination, items_per_thread
    )  # Reuse the compiled kernel.
    cuda.synchronize()
    np.testing.assert_array_equal(destination, source)
    print("Second launch: copy verified")


if __name__ == "__main__":
    main()
