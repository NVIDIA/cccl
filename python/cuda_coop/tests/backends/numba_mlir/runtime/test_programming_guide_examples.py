# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
from cuda import coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


def test_warp_copy():
    # coop-pg-warp-copy-begin
    @cuda.jit
    def copy_warp_tiles(source, destination, count):
        group = coop.this_warp().group_by(8)
        items = coop.ThreadData(items_per_thread=2)
        block_origin = cuda.blockIdx.x * cuda.blockDim.x * 2
        group_origin = (cuda.threadIdx.x // 8) * 16
        valid = min(max(count - block_origin - group_origin, 0), 16)

        coop.load(
            group,
            source,
            items,
            offset=block_origin,
            valid_items=valid,
            oob_default=0,
        )
        coop.store(
            group, destination, items, offset=block_origin, valid_items=valid
        )

    source = np.arange(531, dtype=np.int32)
    destination = np.full_like(source, -1)
    copy_warp_tiles[3, 128](source, destination, source.size)
    cuda.synchronize()
    np.testing.assert_array_equal(destination, source)
    # coop-pg-warp-copy-end
