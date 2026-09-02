# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from cuda import coop as root_coop


@cuda.jit(device=True)
def _inlined_root_sum(value):
    return root_coop.sum(root_coop.this_block(), value)


@cuda.jit
def _root_sum_through_device_function(source, output):
    thread = cuda.threadIdx.x
    result = _inlined_root_sum(source[thread])
    if thread == 0:
        output[0] = result


def test_root_calls_compile_after_device_function_inlining():
    source = np.arange(64, dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)

    _root_sum_through_device_function[1, 64](source, output)

    assert output[0] == source.sum(dtype=np.int32)
