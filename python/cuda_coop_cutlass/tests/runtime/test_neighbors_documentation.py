# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check neighbors examples against host results."""

import numpy as np
import pytest

from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_neighbors_example(items_per_thread):
    # cutlass-neighbors-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def compare_neighbors(
        source: cute.Pointer,
        differences_output: cute.Pointer,
        heads_output: cute.Pointer,
        tails_output: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        values = coop.load(block, source, items_per_thread=items_per_thread)
        scratch = coop.TempStorage(auto_sync=True)
        differences = coop.adjacent_difference(
            block, values, tile_predecessor_item=0, temp_storage=scratch
        )
        heads, tails = coop.discontinuity(
            block, values, mode="heads_and_tails", temp_storage=scratch
        )
        coop.store(block, differences_output, differences)
        coop.store(block, heads_output, heads)
        coop.store(block, tails_output, tails)

    @cute.jit
    def launch(
        source: cute.Pointer,
        differences_output: cute.Pointer,
        heads_output: cute.Pointer,
        tails_output: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        compare_neighbors(
            source,
            differences_output,
            heads_output,
            tails_output,
            items_per_thread,
        ).launch(grid=1, block=64)

    # cutlass-neighbors-example-end
    values = np.arange(64 * items_per_thread, dtype=np.int32) // 3 + 10
    differences, heads, tails = (np.zeros_like(values) for _ in range(3))
    with (
        device_array(values) as src,
        device_array(differences) as delta,
        device_array(heads) as head,
        device_array(tails) as tail,
    ):
        launch(src, delta, head, tail, items_per_thread)
    np.testing.assert_array_equal(differences, np.diff(values, prepend=0))
    np.testing.assert_array_equal(heads, np.r_[1, values[1:] != values[:-1]])
    np.testing.assert_array_equal(tails, np.r_[values[:-1] != values[1:], 1])


def test_neighbors_across_tiles_example():
    """Check left differences that continue across four tiles in a loop.

    Each trip takes one tile of 256 values. The first trip passes zero as the
    predecessor, and each later trip passes the last value of the previous
    tile. The host compares the output with one difference over all 1,024
    values, so a tile that ignored the tile before it would fail at its
    first item.
    """

    # cutlass-neighbors-tiles-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def difference_tiles(
        source: cute.Pointer, destination: cute.Pointer, tiles: cutlass.Int32
    ):
        block = coop.this_block()
        inputs = cute.make_tensor(source, cute.make_layout(1024))
        previous = cutlass.Int32(0)
        for tile in cutlass.range(tiles):
            offset = tile * 256
            values = coop.load(block, source, items_per_thread=4, offset=offset)
            differences = coop.adjacent_difference(
                block, values, tile_predecessor_item=previous
            )
            coop.store(block, destination, differences, offset=offset)
            # The next tile's first item follows this tile's last item.
            previous = inputs[offset + 255]

    @cute.jit
    def launch(
        source: cute.Pointer, destination: cute.Pointer, tiles: cutlass.Int32
    ):
        difference_tiles(source, destination, tiles).launch(grid=1, block=64)

    # cutlass-neighbors-tiles-example-end
    values = (np.arange(4 * 256, dtype=np.int32) * 13) % 31 - 15
    observed = np.full_like(values, -101)
    with device_array(values) as src, device_array(observed) as dst:
        launch(src, dst, cutlass.Int32(4))
    np.testing.assert_array_equal(observed, np.diff(values, prepend=0))
