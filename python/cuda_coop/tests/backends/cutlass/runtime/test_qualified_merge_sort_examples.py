# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check qualified merge sort examples against host results."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_merge_sort_example(items_per_thread):
    # qualified-merge-sort-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def sort_prefix(
        source: cute.Pointer,
        ascending: cute.Pointer,
        descending: cute.Pointer,
        positions: cute.Pointer,
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
        scratch = coop.TempStorage(auto_sync=True)
        ordered = coop.merge_sort_keys(
            block,
            keys,
            valid_items=valid_items,
            oob_default=1000,
            temp_storage=scratch,
        )
        pair_keys, pair_indices = coop.merge_sort_pairs(
            block,
            keys,
            indices,
            descending=True,
            valid_items=valid_items,
            oob_default=-1000,
            temp_storage=scratch,
        )
        coop.store(block, ascending, ordered, valid_items=valid_items)
        coop.store(block, descending, pair_keys, valid_items=valid_items)
        coop.store(block, positions, pair_indices, valid_items=valid_items)

    @cute.jit
    def launch(
        source: cute.Pointer,
        ascending: cute.Pointer,
        descending: cute.Pointer,
        positions: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        sort_prefix(
            source, ascending, descending, positions, items_per_thread
        ).launch(grid=1, block=64)

    # qualified-merge-sort-example-end
    values = (np.arange(64 * items_per_thread, dtype=np.int32) * 13) % 31 - 15
    ascending, descending, positions = (
        np.full_like(values, -101) for _ in range(3)
    )
    with (
        device_array(values) as src,
        device_array(ascending) as low,
        device_array(descending) as high,
        device_array(positions) as index,
    ):
        launch(src, low, high, index, items_per_thread)
    valid = len(values) - 3
    np.testing.assert_array_equal(ascending[:valid], np.sort(values[:valid]))
    np.testing.assert_array_equal(
        descending[:valid], np.sort(values[:valid])[::-1]
    )
    np.testing.assert_array_equal(values[positions[:valid]], descending[:valid])
    np.testing.assert_array_equal(np.sort(positions[:valid]), np.arange(valid))
    for output in (ascending, descending, positions):
        np.testing.assert_array_equal(output[valid:], -101)
