# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check qualified reduce batched examples against host results."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_reduce_batched_example(items_per_thread):
    # qualified-reduce-batched-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def sum_columns(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        warp = coop.this_warp()
        values = coop.ThreadData(items_per_thread)
        coop.load(block, source, values)
        totals = coop.reduce_batched(warp, values)
        output = cute.make_tensor(
            destination, cute.make_layout(2 * items_per_thread)
        )
        for item in cutlass.range_constexpr((items_per_thread + 31) // 32):
            batch = warp.rank() + item * 32
            if batch < items_per_thread:
                index = block.rank() // 32 * items_per_thread + batch
                output[index] = totals[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        sum_columns(source, destination, items_per_thread).launch(
            grid=1, block=64
        )

    # qualified-reduce-batched-example-end
    values = np.arange(64 * items_per_thread, dtype=np.int32) - 50
    observed = np.zeros(2 * items_per_thread, dtype=np.int32)
    with device_array(values) as src, device_array(observed) as out:
        launch(src, out, items_per_thread)
    expected = values.reshape(2, 32, items_per_thread).sum(axis=1)
    np.testing.assert_array_equal(observed, expected.ravel())
