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


def test_thread_data_example():
    # thread-data-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def square_indices(destination, items_per_thread):
        block = coop.this_block()
        items = coop.ThreadData(items_per_thread, alignment=16)
        offset = cuda.blockIdx.x * cuda.blockDim.x * items_per_thread
        for i in range(items.items_per_thread):
            index = offset + cuda.threadIdx.x * items_per_thread + i
            items[i] = types.int32(index * index)
        coop.store(block, destination, items, offset=offset)

    for items_per_thread in (1, 4):
        destination = cuda.device_array(
            2 * 64 * items_per_thread, dtype=np.int32
        )
        square_indices[2, 64](destination, items_per_thread)
        expected = np.arange(2 * 64 * items_per_thread, dtype=np.int32) ** 2
        np.testing.assert_array_equal(destination.copy_to_host(), expected)
    # thread-data-example-end


def test_temp_storage_example():
    # temp-storage-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def scan_tiles(source, destination, items_per_thread):
        block = coop.this_block()
        scratch = coop.TempStorage(alignment=16, auto_sync=True)
        items = coop.ThreadData(items_per_thread)
        for tile in range(2):
            offset = tile * cuda.blockDim.x * items_per_thread
            coop.load(
                block,
                source,
                items,
                offset=offset,
                algorithm="transpose",
                temp_storage=scratch,
            )
            prefixes = coop.exclusive_sum(block, items, temp_storage=scratch)
            coop.store(
                block,
                destination,
                prefixes,
                offset=offset,
                algorithm="transpose",
                temp_storage=scratch,
            )

    for items_per_thread in (1, 4):
        values = (np.arange(2 * 128 * items_per_thread) % 7).astype(np.int32)
        source = cuda.to_device(values)
        destination = cuda.device_array_like(source)
        scan_tiles[1, 128](source, destination, items_per_thread)
        tiles = values.reshape(2, 128 * items_per_thread)
        expected = (np.cumsum(tiles, axis=1) - tiles).ravel()
        np.testing.assert_array_equal(destination.copy_to_host(), expected)
    # temp-storage-example-end
