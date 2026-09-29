# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executable example included in the ``cuda.coop.load`` docstring."""

import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


def test_load_example():
    # example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def copy_tiles(source, destination):
        block = coop.this_block()
        items = coop.ThreadData(2, dtype=np.int32)
        tile_size = cuda.blockDim.x * 2
        offset = cuda.blockIdx.x * tile_size
        valid = min(max(source.size - offset, 0), tile_size)
        coop.load(
            block,
            source,
            items,
            valid_items=valid,
            oob_default=0,
            offset=offset,
        )
        coop.store(
            block,
            destination,
            items,
            valid_items=valid,
            offset=offset,
        )

    expected = np.arange(1000, dtype=np.int32)
    source = cuda.to_device(expected)
    destination = cuda.device_array_like(source)
    blocks = (expected.size + 255) // 256
    copy_tiles[blocks, 128](source, destination)
    np.testing.assert_array_equal(destination.copy_to_host(), expected)
    # example-end
