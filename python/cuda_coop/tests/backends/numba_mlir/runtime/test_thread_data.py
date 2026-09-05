# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
types = pytest.importorskip("numba_cuda_mlir.types")

import cuda.coop.numba_mlir as coop

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _thread_data_extent_keywords_kernel(output):
    canonical = coop.ThreadData(items_per_thread=2, dtype=types.int32)
    aligned = coop.ThreadData(2, types.int32, alignas=16)
    canonical[0] = 3
    canonical[1] = 5
    aligned[0] = 7
    aligned[1] = 11
    output[0] = canonical[0] + canonical[1] + aligned[0] + aligned[1]


def test_thread_data_extent_keywords_compile_in_single_phase_kernel():
    output = np.zeros(1, dtype=np.int32)
    _thread_data_extent_keywords_kernel[1, 1](output)
    assert output.tolist() == [26]
