# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check qualified radix sort examples against host results."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_radix_sort_example(items_per_thread):
    # qualified-radix-sort-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def sort_digits(
        source: cute.Pointer,
        ascending: cute.Pointer,
        descending: cute.Pointer,
        positions: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        indices = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys)
        for item in cutlass.range_constexpr(items_per_thread):
            indices[item] = cutlass.Int32(
                block.rank() * items_per_thread + item
            )
        ordered = coop.radix_sort_keys(block, keys)
        pair_keys, pair_indices = coop.radix_sort_pairs(
            block,
            keys,
            indices,
            begin_bit=0,
            end_bit=4,
            descending=True,
            blocked_to_striped=True,
        )
        coop.store(block, ascending, ordered)
        coop.store(block, descending, pair_keys, algorithm="striped")
        coop.store(block, positions, pair_indices, algorithm="striped")

    @cute.jit
    def launch(
        source: cute.Pointer,
        ascending: cute.Pointer,
        descending: cute.Pointer,
        positions: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        sort_digits(
            source, ascending, descending, positions, items_per_thread
        ).launch(grid=1, block=64)

    # qualified-radix-sort-example-end
    values = (np.arange(64 * items_per_thread, dtype=np.int32) * 13) % 127
    ascending, descending, positions = (np.zeros_like(values) for _ in range(3))
    with (
        device_array(values) as src,
        device_array(ascending) as low,
        device_array(descending) as high,
        device_array(positions) as index,
    ):
        launch(src, low, high, index, items_per_thread)
    order = np.argsort(-(values & 15), kind="stable")
    np.testing.assert_array_equal(ascending, np.sort(values))
    np.testing.assert_array_equal(descending, values[order])
    np.testing.assert_array_equal(positions, order)
