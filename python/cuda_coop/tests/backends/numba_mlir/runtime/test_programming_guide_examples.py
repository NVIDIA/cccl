# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Executable examples included in the programming guide."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


def test_first_kernel():
    # coop-pg-first-kernel-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    from cuda import coop

    @cuda.jit
    def scan_tiles(source, destination, count):
        block = coop.this_block()
        items = coop.ThreadData(2, dtype=np.int32)
        tile_size = cuda.blockDim.x * 2
        offset = cuda.blockIdx.x * tile_size
        valid = min(max(count - offset, 0), tile_size)

        coop.load(
            block,
            source,
            items,
            offset=offset,
            valid_items=valid,
            oob_default=0,
        )
        prefixes = coop.exclusive_sum(block, items)
        coop.store(block, destination, prefixes, offset=offset, valid_items=valid)

    source = (np.arange(785) % 7).astype(np.int32)
    d_source = cuda.to_device(source)
    d_destination = cuda.device_array_like(d_source)
    blocks = (source.size + 255) // 256
    scan_tiles[blocks, 128](d_source, d_destination, source.size)
    result = d_destination.copy_to_host()

    expected = np.empty_like(source)
    for start in range(0, source.size, 256):
        tile = source[start : start + 256]
        expected[start : start + tile.size] = np.cumsum(tile) - tile
    np.testing.assert_array_equal(result, expected)
    # coop-pg-first-kernel-end


def test_qualified_import():
    # coop-pg-qualified-import-begin
    import cuda.coop.numba_mlir as numba_coop

    # coop-pg-qualified-import-end
    assert callable(numba_coop.exclusive_sum)


def test_qualified_scan():
    # coop-pg-qualified-scan-begin
    from numba_cuda_mlir import types

    @cuda.jit
    def scan_tiles_with_totals(source, destination, totals):
        block = numba_coop.this_block()
        items = cuda.local.array(2, dtype=types.int32)
        aggregate = numba_coop.ThreadData(1, dtype=np.int32)
        offset = cuda.blockIdx.x * cuda.blockDim.x * 2

        numba_coop.load(block, source, items, offset=offset)
        prefixes = numba_coop.exclusive_sum(block, items, aggregate_output=aggregate)
        numba_coop.store(block, destination, prefixes, offset=offset)
        if block.rank() == 0:
            totals[cuda.blockIdx.x] = aggregate[0]

    source = (np.arange(512) % 11).astype(np.int32)
    destination = np.empty_like(source)
    totals = np.empty(2, dtype=np.int32)
    scan_tiles_with_totals[2, 128](source, destination, totals)
    cuda.synchronize()

    tiles = source.reshape(2, 256)
    expected = np.cumsum(tiles, axis=1) - tiles
    np.testing.assert_array_equal(destination.reshape(2, 256), expected)
    np.testing.assert_array_equal(totals, tiles.sum(axis=1))
    # coop-pg-qualified-scan-end


def test_row_scan():
    # coop-pg-row-scan-begin
    @cuda.jit
    def scan_rows(source, destination):
        row_group = coop.this_warp().group_by(8)
        index = cuda.grid(1)
        destination[index] = coop.inclusive_sum(row_group, source[index])

    source = (np.arange(256) % 5).astype(np.int32)
    destination = np.empty_like(source)
    scan_rows[2, 128](source, destination)
    cuda.synchronize()
    np.testing.assert_array_equal(
        destination.reshape(32, 8), np.cumsum(source.reshape(32, 8), axis=1)
    )
    # coop-pg-row-scan-end


def test_exchange():
    # coop-pg-exchange-begin
    @cuda.jit
    def scan_striped_input(source, destination):
        block = coop.this_block()
        items = coop.ThreadData(2, dtype=np.int32)
        coop.load(block, source, items, algorithm="striped")
        blocked = coop.exchange(block, items, mode="striped_to_blocked")
        prefixes = coop.inclusive_sum(block, blocked)
        coop.store(block, destination, prefixes, algorithm="direct")

    source = (np.arange(256) % 13).astype(np.int32)
    destination = np.empty_like(source)
    scan_striped_input[1, 128](source, destination)
    cuda.synchronize()
    np.testing.assert_array_equal(destination, np.cumsum(source))
    # coop-pg-exchange-end


def test_warp_copy():
    # coop-pg-warp-copy-begin
    @cuda.jit
    def copy_warp_tiles(source, destination, count):
        group = coop.this_warp().group_by(8)
        items = coop.ThreadData(2, dtype=np.int32)
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
        coop.store(group, destination, items, offset=block_origin, valid_items=valid)

    source = np.arange(531, dtype=np.int32)
    destination = np.full_like(source, -1)
    copy_warp_tiles[3, 128](source, destination, source.size)
    cuda.synchronize()
    np.testing.assert_array_equal(destination, source)
    # coop-pg-warp-copy-end


def test_shared_scratch():
    # coop-pg-shared-scratch-begin
    @cuda.jit
    def scan_with_shared_scratch(source, destination):
        block = coop.this_block()
        scratch = coop.TempStorage()
        items = coop.ThreadData(2, dtype=np.int32)

        coop.load(block, source, items, algorithm="transpose", temp_storage=scratch)
        prefixes = coop.exclusive_sum(block, items, temp_storage=scratch)
        coop.store(
            block,
            destination,
            prefixes,
            algorithm="transpose",
            temp_storage=scratch,
        )

    source = (np.arange(256) % 7).astype(np.int32)
    destination = np.empty_like(source)
    scan_with_shared_scratch[1, 128](source, destination)
    cuda.synchronize()
    np.testing.assert_array_equal(destination, np.cumsum(source) - source)
    # coop-pg-shared-scratch-end


def test_manual_scratch():
    # coop-pg-manual-scratch-begin
    @cuda.jit
    def copy_tiles_with_manual_sync(source, destination):
        block = coop.this_block()
        scratch = coop.TempStorage(auto_sync=False)
        items = coop.ThreadData(2, dtype=np.int32)
        for tile in range(2):
            offset = tile * cuda.blockDim.x * 2
            coop.load(
                block,
                source,
                items,
                offset=offset,
                algorithm="transpose",
                temp_storage=scratch,
            )
            block.sync()
            coop.store(
                block,
                destination,
                items,
                offset=offset,
                algorithm="transpose",
                temp_storage=scratch,
            )
            block.sync()

    source = np.arange(512, dtype=np.int32)
    destination = np.empty_like(source)
    copy_tiles_with_manual_sync[1, 128](source, destination)
    cuda.synchronize()
    np.testing.assert_array_equal(destination, source)
    # coop-pg-manual-scratch-end


def test_reduce():
    # coop-pg-reduce-begin
    @cuda.jit
    def tile_sums(source, totals, count):
        block = coop.this_block()
        items = coop.ThreadData(2, dtype=np.int32)
        tile_size = cuda.blockDim.x * 2
        offset = cuda.blockIdx.x * tile_size
        valid = min(max(count - offset, 0), tile_size)
        coop.load(
            block,
            source,
            items,
            offset=offset,
            valid_items=valid,
            oob_default=0,
        )
        total = coop.sum(block, items, broadcast=False)
        if block.rank() == 0:
            totals[cuda.blockIdx.x] = total

    source = (np.arange(785) % 17).astype(np.int32)
    totals = np.empty(4, dtype=np.int32)
    tile_sums[4, 128](source, totals, source.size)
    cuda.synchronize()
    expected = [
        source[start : start + 256].sum() for start in range(0, source.size, 256)
    ]
    np.testing.assert_array_equal(totals, expected)
    # coop-pg-reduce-end


def test_custom_scan():
    # coop-pg-custom-scan-begin
    @cuda.jit(device=True)
    def maximum(left, right):
        if left > right:
            return left
        return right

    @cuda.jit
    def running_maximum(source, destination):
        block = numba_coop.this_block()
        items = numba_coop.ThreadData(2, dtype=np.int32)
        numba_coop.load(block, source, items)
        result = numba_coop.inclusive_scan(block, items, scan_op=maximum)
        numba_coop.store(block, destination, result)

    source = ((np.arange(256) * 17) % 113 - 51).astype(np.int32)
    destination = np.empty_like(source)
    running_maximum[1, 128](source, destination)
    cuda.synchronize()
    np.testing.assert_array_equal(destination, np.maximum.accumulate(source))
    # coop-pg-custom-scan-end


def test_prefix_callback():
    # coop-pg-prefix-callback-begin
    @cuda.jit(device=True)
    def carry_total(state, tile_total):
        previous = state[0]
        state[0] = previous + tile_total
        return previous

    running_prefix = numba_coop.StatefulFunction(carry_total, types.int64)

    @cuda.jit
    def scan_successive_tiles(source, destination, final_total):
        block = numba_coop.this_block()
        state = numba_coop.ThreadData(1, dtype=types.int64)
        state[0] = types.int64(0)
        scratch = numba_coop.TempStorage()
        for tile in range(3):
            index = tile * cuda.blockDim.x + cuda.threadIdx.x
            destination[index] = numba_coop.exclusive_sum(
                block,
                source[index],
                state,
                prefix_op=running_prefix,
                temp_storage=scratch,
            )
        if block.rank() == 0:
            final_total[0] = state[0]

    source = (np.arange(384) % 9).astype(np.int32)
    destination = np.empty_like(source)
    final_total = np.empty(1, dtype=np.int64)
    scan_successive_tiles[1, 128](source, destination, final_total)
    cuda.synchronize()
    np.testing.assert_array_equal(destination, np.cumsum(source) - source)
    np.testing.assert_array_equal(final_total, [source.sum()])
    # coop-pg-prefix-callback-end
