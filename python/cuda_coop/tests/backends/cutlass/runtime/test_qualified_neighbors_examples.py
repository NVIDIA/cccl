# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check qualified neighbors examples against host results."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_neighbors_example(items_per_thread):
    # qualified-neighbors-example-begin
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
        values = coop.ThreadData(items_per_thread)
        coop.load(block, source, values)
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

    # qualified-neighbors-example-end
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
