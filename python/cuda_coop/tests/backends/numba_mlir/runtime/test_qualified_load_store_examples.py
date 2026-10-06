# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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


def test_qualified_load_store_example():
    # qualified-load-store-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as coop

    @cuda.jit
    def copy_tiles(source, destination, items_per_thread):
        block = coop.this_block()
        # Qualified calls also accept a fixed-size Numba local array.
        items = cuda.local.array(items_per_thread, dtype=types.int32)
        tile_size = cuda.blockDim.x * items_per_thread
        offset = cuda.blockIdx.x * tile_size
        valid = min(source.size - offset, tile_size)
        coop.load(
            block,
            source,
            items,
            offset=offset,
            valid_items=valid,
            oob_default=0,
        )
        coop.store(block, destination, items, offset=offset, valid_items=valid)

    for items_per_thread in (1, 4):
        values = np.arange(1000, dtype=np.int32)
        source = cuda.to_device(values)
        destination = cuda.device_array_like(source)
        tile_size = 64 * items_per_thread
        blocks = (values.size + tile_size - 1) // tile_size
        copy_tiles[blocks, 64](source, destination, items_per_thread)
        np.testing.assert_array_equal(destination.copy_to_host(), values)
    # qualified-load-store-example-end
