# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reduce one scalar per block thread with Numba-CUDA-MLIR."""

import numpy as np
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as coop


@cuda.jit
def block_sum_kernel(source, output):
    """Write one block sum; the collective result is root-only."""

    thread = cuda.threadIdx.x
    total = coop.sum(coop.this_block(), source[thread])
    if thread == 0:
        output[0] = total


def main() -> None:
    source = np.arange(128, dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)
    block_sum_kernel[1, 128](source, output)
    np.testing.assert_array_equal(output, [source.sum(dtype=np.int32)])
    print(output[0])


if __name__ == "__main__":
    main()
