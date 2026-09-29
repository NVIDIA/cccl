# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the Developer Overview's Debugger Walkthrough as an active Python file."""

# Import the kernel DSL before cuda.coop so automatic registration sees it.
# isort: off
import numpy as np
from numba_cuda_mlir import cuda

from cuda import coop  # First breakpoint: observe backend registration.
# isort: on


@cuda.jit
def copy_tile(source, destination):
    block = coop.this_block()
    items = coop.ThreadData(2, dtype=np.int32)
    coop.load(block, source, items, algorithm="direct")
    coop.store(block, destination, items, algorithm="direct")


def main():
    source = np.arange(256, dtype=np.int32)
    destination = np.zeros_like(source)

    copy_tile[1, 128](source, destination)  # Compile, link, and launch.
    cuda.synchronize()
    np.testing.assert_array_equal(destination, source)
    print("First launch: copy verified")

    destination.fill(-1)
    copy_tile[1, 128](source, destination)  # Reuse the compiled kernel.
    cuda.synchronize()
    np.testing.assert_array_equal(destination, source)
    print("Second launch: copy verified")


if __name__ == "__main__":
    main()
