# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Keep foreign shared allocations live across three scratch lanes."""

import cutlass
import numpy as np
import pytest
from cutlass import cute
from cutlass.memory import SmemAllocator

import cuda.coop.cutlass as coop
from tests.runtime_support import device_array, values_for

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_THREADS = 64
_ITEMS = 2
_TILE = _THREADS * _ITEMS
_ITERATIONS = 8
_BLOCKS = 3
_COUNTS = (67, 131, 257)
_CANARIES = sum(_COUNTS)
_ROW = 3 * _TILE + _CANARIES
_METADATA = 13 + _ITERATIONS


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (False, True))
def test_foreign_allocators_and_interleaved_scratch_lanes(sharing, auto_sync):
    """Stress independent helpers, alignment padding, and runtime reuse.

    Three foreign arrays stay live while load/scan/store, load/difference/store,
    and load/reduce/store reuse distinct descriptors. Read every foreign
    element through another thread after every iteration, then verify all six
    regions are disjoint and included in the launch's shared-memory extent.
    """

    @cute.jit
    def foreign_array(count: cutlass.Constexpr, alignment: cutlass.Constexpr):
        allocator = SmemAllocator()
        pointer = allocator.allocate_array(
            cutlass.Int32, count, byte_alignment=alignment
        )
        return cute.make_tensor(pointer, cute.make_layout(count))

    @cute.jit
    def foreign_tensor():
        return SmemAllocator().allocate_tensor(
            cutlass.Int32, cute.make_layout(_COUNTS[2]), byte_alignment=256
        )

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        metadata: cute.Pointer,
        iterations: cutlass.Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block_index, _, _ = cute.arch.block_idx()
        block = coop.this_block()
        first = foreign_array(_COUNTS[0], 16)
        lane_a = coop.TempStorage(
            sharing=sharing, auto_sync=auto_sync, alignment=64
        )
        second = foreign_array(_COUNTS[1], 128)
        lane_b = coop.TempStorage(
            sharing=sharing, auto_sync=auto_sync, alignment=128
        )
        third = foreign_tensor()
        lane_c = coop.TempStorage(
            sharing=sharing, auto_sync=auto_sync, alignment=256
        )
        foreign = (first, second, third)
        output = cute.make_tensor(
            destination, cute.make_layout(_BLOCKS * _ITERATIONS * _ROW)
        )
        details = cute.make_tensor(
            metadata, cute.make_layout(_BLOCKS * _METADATA)
        )
        for lane in cutlass.range_constexpr(3):
            for item in range(thread, _COUNTS[lane], _THREADS):
                foreign[lane][item] = (
                    10000 * block_index + 1000 * lane + 7 * item + 3
                )
        cute.arch.sync_threads()

        for iteration in range(iterations):
            tile_index = block_index * _ITERATIONS + iteration
            offset = tile_index * 3 * _TILE
            a = coop.load(
                block,
                source,
                items_per_thread=_ITEMS,
                offset=offset,
                algorithm="transpose",
                temp_storage=lane_a,
            )
            b = coop.load(
                block,
                source,
                items_per_thread=_ITEMS,
                offset=offset + _TILE,
                algorithm="transpose",
                temp_storage=lane_b,
            )
            c = coop.load(
                block,
                source,
                items_per_thread=_ITEMS,
                offset=offset + 2 * _TILE,
                algorithm="transpose",
                temp_storage=lane_c,
            )
            if cutlass.const_expr(not auto_sync):
                cute.arch.sync_threads()
            scanned = coop.inclusive_sum(block, a, temp_storage=lane_a)
            differences = coop.adjacent_difference(
                block, b, tile_predecessor_item=0, temp_storage=lane_b
            )
            total = coop.sum(block, c, temp_storage=lane_c)
            if cutlass.const_expr(not auto_sync):
                cute.arch.sync_threads()
            output_offset = tile_index * _ROW
            coop.store(
                block,
                destination,
                scanned,
                offset=output_offset,
                algorithm="transpose",
                temp_storage=lane_a,
            )
            coop.store(
                block,
                destination,
                differences,
                offset=output_offset + _TILE,
                algorithm="transpose",
                temp_storage=lane_b,
            )
            coop.store(
                block,
                destination,
                c,
                offset=output_offset + 2 * _TILE,
                algorithm="transpose",
                temp_storage=lane_c,
            )
            if cutlass.const_expr(not auto_sync):
                cute.arch.sync_threads()
            canary_offset = output_offset + 3 * _TILE
            for lane in cutlass.range_constexpr(3):
                for item in range(thread, _COUNTS[lane], _THREADS):
                    output[canary_offset + item] = foreign[lane][
                        (item + 1) % _COUNTS[lane]
                    ]
                canary_offset += _COUNTS[lane]
            if thread == 0:
                details[block_index * _METADATA + 13 + iteration] = total

        regions = (lane_a.release(), lane_b.release(), lane_c.release())
        if thread == 0:
            for lane in cutlass.range_constexpr(3):
                index = block_index * _METADATA + 2 * lane
                details[index] = foreign[lane].iterator.toint()
                details[index + 1] = _COUNTS[lane] * 4
                details[index + 6] = regions[lane].iterator.toint()
                details[index + 7] = cute.size(regions[lane])
            details[block_index * _METADATA + 12] = (
                cute.arch.dynamic_smem_size()
            )

    @cute.jit
    def launch(source, destination, metadata, iterations: cutlass.Int32):
        kernel(source, destination, metadata, iterations).launch(
            grid=_BLOCKS, block=_THREADS
        )

    source = values_for(np.int32, _BLOCKS * _ITERATIONS * 3 * _TILE, shift=19)
    output = np.full(_BLOCKS * _ITERATIONS * _ROW, -101, dtype=np.int32)
    metadata = np.zeros(_BLOCKS * _METADATA, dtype=np.int64)
    with (
        device_array(source) as src,
        device_array(output) as dst,
        device_array(metadata) as info,
    ):
        launch(src, dst, info, cutlass.Int32(_ITERATIONS))

    source = source.reshape(_BLOCKS, _ITERATIONS, 3, _TILE)
    output = output.reshape(_BLOCKS, _ITERATIONS, _ROW)
    metadata = metadata.reshape(_BLOCKS, _METADATA)
    for block in range(_BLOCKS):
        for iteration in range(_ITERATIONS):
            a, b, c = source[block, iteration]
            expected = np.concatenate((a.cumsum(), np.diff(b, prepend=0), c))
            np.testing.assert_array_equal(
                output[block, iteration, : 3 * _TILE], expected
            )
            canaries = np.concatenate(
                [
                    10000 * block
                    + 1000 * lane
                    + 7 * ((np.arange(count) + 1) % count)
                    + 3
                    for lane, count in enumerate(_COUNTS)
                ]
            )
            np.testing.assert_array_equal(
                output[block, iteration, 3 * _TILE :], canaries
            )
            assert metadata[block, 13 + iteration] == c.sum()
        regions = metadata[block, :12].reshape(6, 2)
        for (address, size), alignment in zip(
            regions, (16, 128, 256, 64, 128, 256)
        ):
            assert address % alignment == 0
            assert size > 0
        ordered = sorted(regions, key=lambda region: region[0])
        # Dynamic shared memory need not start at shared address zero.
        extent = ordered[-1][0] + ordered[-1][1] - ordered[0][0]
        assert extent <= metadata[block, 12]
        assert all(
            first[0] + first[1] <= second[0]
            for first, second in zip(ordered, ordered[1:])
        )
