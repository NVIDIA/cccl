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
    def copy_warp_tiles(source, destination, count, items_per_thread):
        group = coop.this_warp().group_by(8)
        items = coop.ThreadData(items_per_thread)
        block_origin = cuda.blockIdx.x * cuda.blockDim.x * items_per_thread
        group_origin = (cuda.threadIdx.x // 8) * 8 * items_per_thread
        valid = min(
            max(count - block_origin - group_origin, 0), 8 * items_per_thread
        )

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

    for items_per_thread in (1, 4):
        source = np.arange(531, dtype=np.int32)
        destination = np.full_like(source, -1)
        blocks = (source.size + 128 * items_per_thread - 1) // (
            128 * items_per_thread
        )
        copy_warp_tiles[blocks, 128](
            source, destination, source.size, items_per_thread
        )
        cuda.synchronize()
        np.testing.assert_array_equal(destination, source)
    # coop-pg-warp-copy-end


def test_manual_scratch():
    # coop-pg-manual-scratch-begin
    @cuda.jit
    def copy_tiles_with_manual_sync(source, destination):
        block = coop.this_block()
        scratch = coop.TempStorage()
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

    compiled = next(
        iter(copy_tiles_with_manual_sync._launch_config_overloads.values())
    )
    # The descriptor adds no barriers to the explicit block.sync() calls.
    assert compiled.metadata["mlir_module_str"].count("gpu.barrier") == 0


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
        source[start : start + 256].sum()
        for start in range(0, source.size, 256)
    ]
    np.testing.assert_array_equal(totals, expected)
    # coop-pg-reduce-end
