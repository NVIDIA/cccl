# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as coop

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)

THREADS = 32


def _binary_op(lhs, rhs):
    return lhs + rhs


def test_make_block_reduce_num_valid_runs_in_kernel():
    block_sum_num_valid = coop._block.make_sum(
        types.int32, threads_per_block=THREADS, num_valid=np.int32(17)
    )
    block_reduce_num_valid = coop._block.make_reduce(
        types.int32,
        threads_per_block=THREADS,
        binary_op=_binary_op,
        num_valid=np.int32(17),
    )

    @cuda.jit
    def kernel(d_in, d_out, num_valid):
        tid = cuda.threadIdx.x
        summed = block_sum_num_valid(d_in[tid], num_valid)
        reduced = block_reduce_num_valid(d_in[tid], num_valid)

        if tid == 0:
            d_out[0] = summed
            d_out[1] = reduced

    num_valid = np.int32(17)
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_output = np.zeros(2, dtype=np.int32)

    kernel[1, THREADS](h_input, h_output, num_valid)

    expected = np.sum(h_input[:num_valid]).astype(np.int32)
    np.testing.assert_array_equal(h_output, np.asarray([expected, expected]))
