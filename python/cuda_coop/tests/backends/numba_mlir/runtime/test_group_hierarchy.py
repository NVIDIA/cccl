# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

import cuda.coop.numba_mlir as coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_THREADS = 128
_QUERY_FIELDS = 15


@cuda.jit
def _group_query_kernel(output):
    tid = cuda.threadIdx.x
    thread = coop.this_thread()
    warp = coop.this_warp()
    block = coop.this_block()
    lanes = warp.group_by(8)
    warps = block.group_by(2)

    thread.sync()
    warp.sync_aligned()
    lanes.sync()
    warps.sync_aligned()
    block.sync()

    output[0 * _THREADS + tid] = thread.rank("block")
    output[1 * _THREADS + tid] = thread.count("thread")
    output[2 * _THREADS + tid] = warp.rank("thread")
    output[3 * _THREADS + tid] = warp.count("block")
    output[4 * _THREADS + tid] = block.rank("thread")
    output[5 * _THREADS + tid] = block.count("grid")
    output[6 * _THREADS + tid] = lanes.rank("thread")
    output[7 * _THREADS + tid] = lanes.count("warp")
    output[8 * _THREADS + tid] = warps.rank("warp")
    output[9 * _THREADS + tid] = warps.count("thread")
    output[10 * _THREADS + tid] = thread.is_member()
    output[11 * _THREADS + tid] = warp.is_member()
    output[12 * _THREADS + tid] = block.is_member()
    output[13 * _THREADS + tid] = lanes.is_member()
    output[14 * _THREADS + tid] = warps.is_member()


def test_physical_and_mapped_group_runtime():
    output = np.zeros(_QUERY_FIELDS * _THREADS, dtype=np.int32)

    _group_query_kernel[1, _THREADS](output)

    tid = np.arange(_THREADS, dtype=np.int32)
    expected = np.stack(
        (
            tid,
            np.ones_like(tid),
            tid % 32,
            np.full_like(tid, 4),
            tid,
            np.ones_like(tid),
            (tid % 32) % 8,
            np.full_like(tid, 4),
            (tid % 64) // 32,
            np.full_like(tid, 64),
            *([np.ones_like(tid)] * 5),
        )
    )
    np.testing.assert_array_equal(
        output.reshape(_QUERY_FIELDS, _THREADS),
        expected,
    )
