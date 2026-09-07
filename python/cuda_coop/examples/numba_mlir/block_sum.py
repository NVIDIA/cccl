# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block reduction with the Numba-CUDA-MLIR cuda.coop backend."""

import numpy as np
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as coop

THREADS = 32


# docs: start numba-mlir-block-sum
@cuda.jit
def block_sum_kernel(values, result):
    tid = cuda.threadIdx.x
    block = coop.this_block()
    total = coop.sum(block, values[tid])
    if block.rank() == 0:
        result[0] = total


# docs: end numba-mlir-block-sum


def run_example() -> int:
    """Run the example, check the result, and return the block sum."""

    values = np.arange(1, THREADS + 1, dtype=np.int32)
    result = np.zeros(1, dtype=np.int32)
    block_sum_kernel[1, THREADS](values, result)
    expected = int(np.sum(values, dtype=np.int64))
    assert int(result[0]) == expected
    return expected


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
