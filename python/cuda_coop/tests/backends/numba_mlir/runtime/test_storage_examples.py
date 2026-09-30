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


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("sharing", ["shared", "exclusive"])
def test_release_inferred_region_after_cooperative_phase(
    items_per_thread, sharing
):
    # temp-storage-release-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def copy_then_handoff(source, destination, result, items_per_thread):
        block = coop.this_block()
        storage = coop.TempStorage(sharing=sharing, auto_sync=True)
        items = coop.ThreadData(items_per_thread)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(
            block,
            destination,
            items,
            algorithm="transpose",
            temp_storage=storage,
        )
        tile = storage.release(types.int32)
        rank = cuda.threadIdx.x
        tile[rank] = rank + 3
        cuda.syncthreads()
        result[rank] = tile[cuda.blockDim.x - 1 - rank]
        if rank == 0:
            result[cuda.blockDim.x] = len(tile)

    count = 128 * items_per_thread
    values = np.arange(count, dtype=np.int32)
    source = cuda.to_device(values)
    destination = cuda.device_array_like(source)
    result = cuda.device_array(129, dtype=np.int32)
    copy_then_handoff[1, 128](source, destination, result, items_per_thread)
    np.testing.assert_array_equal(destination.copy_to_host(), values)
    actual = result.copy_to_host()
    np.testing.assert_array_equal(actual[:128], np.arange(127, -1, -1) + 3)
    assert actual[128] >= count * (2 if sharing == "exclusive" else 1)
    # temp-storage-release-example-end


def test_release_preserves_independent_descriptor_regions():
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as coop

    @cuda.jit
    def kernel(source, result):
        first = coop.TempStorage(128, auto_sync=True)
        second = coop.TempStorage(256, auto_sync=True)
        items = coop.ThreadData(items_per_thread=1)
        coop.load(
            coop.this_block(),
            source,
            items,
            algorithm="transpose",
            temp_storage=first,
        )
        coop.load(
            coop.this_block(),
            source,
            items,
            algorithm="transpose",
            temp_storage=second,
        )
        left = first.release(types.int32)
        right = second.release(types.int32)
        rank = cuda.threadIdx.x
        left[rank] = rank
        right[rank] = 100 + rank
        right[32 + rank] = 200 + rank
        cuda.syncthreads()
        result[rank] = left[31 - rank] + right[rank] + right[32 + rank]
        if rank == 0:
            result[32] = len(left)
            result[33] = len(right)

    result = cuda.device_array(34, dtype=np.int32)
    source = cuda.to_device(np.arange(32, dtype=np.int32))
    kernel[1, 32](source, result)
    expected = np.concatenate((331 + np.arange(32), [32, 64])).astype(np.int32)
    np.testing.assert_array_equal(result.copy_to_host(), expected)


def test_bound_release_survives_rebind():
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as coop

    @cuda.jit
    def kernel(result):
        storage = coop.TempStorage(128)
        saved = storage.release
        storage = coop.TempStorage(256)
        left = saved()
        right = storage.release()
        rank = cuda.threadIdx.x
        left[rank] = rank
        right[rank] = 100 + rank
        cuda.syncthreads()
        result[rank] = left[31 - rank]
        result[32 + rank] = right[rank]
        if rank == 0:
            result[64] = len(left)
            result[65] = len(right)

    result = cuda.device_array(66, dtype=np.int32)
    kernel[1, 32](result)
    expected = np.concatenate(
        (np.arange(32)[::-1], 100 + np.arange(32), [128, 256])
    )
    np.testing.assert_array_equal(result.copy_to_host(), expected)
