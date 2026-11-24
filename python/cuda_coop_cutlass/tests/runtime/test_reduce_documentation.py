# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check reduce examples against host results."""

import numpy as np
import pytest

from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_reduce_example(items_per_thread):
    # cutlass-reduce-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def summarize(
        source: cute.Pointer,
        totals: cute.Pointer,
        maxima: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        values = coop.load(block, source, items_per_thread=items_per_thread)
        total = coop.sum(block, values)
        maximum = coop.reduce(block, values, binary_op="max")
        total_output = cute.make_tensor(totals, cute.make_layout(1))
        maximum_output = cute.make_tensor(maxima, cute.make_layout(1))
        if block.rank() == 0:
            total_output[0] = total
            maximum_output[0] = maximum

    @cute.jit
    def launch(
        source: cute.Pointer,
        totals: cute.Pointer,
        maxima: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        summarize(source, totals, maxima, items_per_thread).launch(
            grid=1, block=64
        )

    # cutlass-reduce-example-end
    values = np.arange(64 * items_per_thread, dtype=np.int32) - 50
    totals = np.zeros(1, dtype=np.int32)
    maxima = np.zeros(1, dtype=np.int32)
    with (
        device_array(values) as src,
        device_array(totals) as total,
        device_array(maxima) as maximum,
    ):
        launch(src, total, maximum, items_per_thread)
    np.testing.assert_array_equal(totals, values.sum())
    np.testing.assert_array_equal(maxima, values.max())


def test_logical_warp_sums_across_blocks():
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def group_sums(source: cute.Pointer, destination: cute.Pointer):
        block_index = cute.arch.block_idx()[0]
        thread_rank = cute.arch.thread_idx()[0]
        # cutlass-logical-warp-reduce-example-begin
        index = block_index * 128 + thread_rank
        group = coop.this_warp().group_by(8)
        total = coop.sum(group, source[index])
        if group.rank() == 0:
            destination[index // 8] = total
        # cutlass-logical-warp-reduce-example-end

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        group_sums(source, destination).launch(grid=2, block=128)

    values = np.arange(256, dtype=np.int32)
    observed = np.full(32, -1, dtype=np.int32)
    with device_array(values) as source, device_array(observed) as destination:
        launch(source, destination)
    np.testing.assert_array_equal(observed, values.reshape(-1, 8).sum(axis=1))
