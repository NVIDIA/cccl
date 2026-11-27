# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check histogram examples against host results."""

import numpy as np
import pytest

from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_histogram_example(items_per_thread):
    # cutlass-histogram-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def count_samples(
        source: cute.Pointer,
        atomic_output: cute.Pointer,
        sort_output: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        samples = coop.load(block, source, items_per_thread=items_per_thread)
        atomic_counts = coop.histogram(
            block, samples, bins=70, bins_per_thread=2
        )
        sort_counts = coop.histogram(
            block, samples, bins=70, bins_per_thread=2, algorithm="sort"
        )
        coop.store(block, atomic_output, atomic_counts, algorithm="striped")
        coop.store(block, sort_output, sort_counts, algorithm="striped")

    @cute.jit
    def launch(
        source: cute.Pointer,
        atomic_output: cute.Pointer,
        sort_output: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        count_samples(
            source, atomic_output, sort_output, items_per_thread
        ).launch(grid=1, block=64)

    # cutlass-histogram-example-end
    values = (np.arange(64 * items_per_thread, dtype=np.int32) * 13) % 70
    atomic, sorted_counts = (np.full(128, -1, dtype=np.int32) for _ in range(2))
    with (
        device_array(values) as src,
        device_array(atomic) as first,
        device_array(sorted_counts) as second,
    ):
        launch(src, first, second, items_per_thread)
    expected = np.bincount(values, minlength=128)
    np.testing.assert_array_equal(atomic, expected)
    np.testing.assert_array_equal(sorted_counts, expected)


def test_histogram_tiles_example():
    """Accumulate four tiles into 70 shared counters on 64 threads."""

    # cutlass-histogram-tiles-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def count_tiles(
        source: cute.Pointer, destination: cute.Pointer, tiles: cutlass.Int32
    ):
        block = coop.this_block()
        counters = coop.histogram_init(block, bins=70)
        for tile in cutlass.range(tiles):
            samples = coop.load(
                block, source, items_per_thread=4, offset=tile * 256
            )
            coop.histogram_composite(block, samples, counters)

        output = cute.make_tensor(destination, cute.make_layout(70))
        thread = cute.arch.thread_idx()[0]
        for bin_index in cutlass.range(thread, 70, 64):
            output[bin_index] = counters[bin_index]

    @cute.jit
    def launch(
        source: cute.Pointer, destination: cute.Pointer, tiles: cutlass.Int32
    ):
        count_tiles(source, destination, tiles).launch(grid=1, block=64)

    # cutlass-histogram-tiles-example-end
    values = (np.arange(4 * 256, dtype=np.int32) * 13) % 70
    observed = np.full(70, -1, dtype=np.int32)
    with device_array(values) as src, device_array(observed) as dst:
        launch(src, dst, cutlass.Int32(4))
    np.testing.assert_array_equal(observed, np.bincount(values, minlength=70))


def test_histogram_partial_tile_example():
    """Count the first 200 samples of a tile through an extra bin.

    The source holds a whole tile of valid bin indices, so the host bincount
    fails if the kernel counts an item at or past the count.
    """

    # cutlass-histogram-partial-tile-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def count_partial_tile(
        source: cute.Pointer, destination: cute.Pointer, count: cutlass.Int32
    ):
        block = coop.this_block()
        # Items at or past the count load as 70, the index of the extra bin.
        samples = coop.load(
            block, source, items_per_thread=4, valid_items=count, oob_default=70
        )
        counts = coop.histogram(block, samples, bins=71, bins_per_thread=2)
        # Write the counts of bins 0 to 69 and leave the extra bin out.
        coop.store(
            block, destination, counts, algorithm="striped", valid_items=70
        )

    @cute.jit
    def launch(
        source: cute.Pointer, destination: cute.Pointer, count: cutlass.Int32
    ):
        count_partial_tile(source, destination, count).launch(grid=1, block=64)

    # cutlass-histogram-partial-tile-example-end
    values = (np.arange(256, dtype=np.int32) * 13) % 70
    observed = np.full(70, -1, dtype=np.int32)
    with device_array(values) as src, device_array(observed) as dst:
        launch(src, dst, cutlass.Int32(200))
    np.testing.assert_array_equal(
        observed, np.bincount(values[:200], minlength=70)
    )
