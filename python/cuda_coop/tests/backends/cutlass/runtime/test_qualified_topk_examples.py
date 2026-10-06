# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check qualified topk examples against host results."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_topk_keys_example(items_per_thread):
    # qualified-topk-keys-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def select_extremes(
        source: cute.Pointer,
        smallest: cute.Pointer,
        largest: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys)
        low = coop.topk_min_keys(block, keys, k=7)
        high = coop.topk_max_keys(block, keys, k=7)
        # Selection is unsorted; only the first seven positions are defined.
        coop.store(block, smallest, low, valid_items=7)
        coop.store(block, largest, high, valid_items=7)

    @cute.jit
    def launch(
        source: cute.Pointer,
        smallest: cute.Pointer,
        largest: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        select_extremes(source, smallest, largest, items_per_thread).launch(
            grid=1, block=64
        )

    # qualified-topk-keys-example-end
    values = (np.arange(64 * items_per_thread, dtype=np.int32) * 13) % 127 - 63
    smallest, largest = (np.full_like(values, -101) for _ in range(2))
    with (
        device_array(values) as src,
        device_array(smallest) as low,
        device_array(largest) as high,
    ):
        launch(src, low, high, items_per_thread)
    ordered = np.sort(values)
    np.testing.assert_array_equal(np.sort(smallest[:7]), ordered[:7])
    np.testing.assert_array_equal(np.sort(largest[:7]), ordered[-7:])
    np.testing.assert_array_equal(smallest[7:], -101)
    np.testing.assert_array_equal(largest[7:], -101)


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_topk_pairs_example(items_per_thread):
    # qualified-topk-pairs-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def select_pairs(
        source: cute.Pointer,
        smallest: cute.Pointer,
        largest: cute.Pointer,
        smallest_indices: cute.Pointer,
        largest_indices: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        valid_items = 64 * items_per_thread - 3
        keys = coop.ThreadData(items_per_thread)
        indices = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys)
        for item in cutlass.range_constexpr(items_per_thread):
            indices[item] = cutlass.Int32(
                block.rank() * items_per_thread + item
            )
        low, low_indices = coop.topk_min_pairs(
            block, keys, indices, k=7, valid_items=valid_items
        )
        high, high_indices = coop.topk_max_pairs(
            block, keys, indices, k=7, valid_items=valid_items
        )
        coop.store(block, smallest, low, valid_items=7)
        coop.store(block, largest, high, valid_items=7)
        coop.store(block, smallest_indices, low_indices, valid_items=7)
        coop.store(block, largest_indices, high_indices, valid_items=7)

    @cute.jit
    def launch(
        source: cute.Pointer,
        smallest: cute.Pointer,
        largest: cute.Pointer,
        smallest_indices: cute.Pointer,
        largest_indices: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        select_pairs(
            source,
            smallest,
            largest,
            smallest_indices,
            largest_indices,
            items_per_thread,
        ).launch(grid=1, block=64)

    # qualified-topk-pairs-example-end
    values = (np.arange(64 * items_per_thread, dtype=np.int32) * 13) % 127 - 63
    smallest, largest, small_indices, large_indices = (
        np.full_like(values, -101) for _ in range(4)
    )
    with (
        device_array(values) as src,
        device_array(smallest) as low,
        device_array(largest) as high,
        device_array(small_indices) as low_index,
        device_array(large_indices) as high_index,
    ):
        launch(src, low, high, low_index, high_index, items_per_thread)
    ordered = np.sort(values[:-3])
    np.testing.assert_array_equal(np.sort(smallest[:7]), ordered[:7])
    np.testing.assert_array_equal(np.sort(largest[:7]), ordered[-7:])
    for selected, indices in (
        (smallest, small_indices),
        (largest, large_indices),
    ):
        assert np.all((indices[:7] >= 0) & (indices[:7] < len(values) - 3))
        assert len(np.unique(indices[:7])) == 7
        np.testing.assert_array_equal(values[indices[:7]], selected[:7])
        np.testing.assert_array_equal(selected[7:], -101)
        np.testing.assert_array_equal(indices[7:], -101)
