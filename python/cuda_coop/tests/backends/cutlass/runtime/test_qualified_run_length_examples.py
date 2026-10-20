# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check qualified run length examples against host results."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_run_length_example(items_per_thread):
    # qualified-run-length-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def decode_runs(
        window_pointer: cute.Pointer,
        stream_pointer: cute.Pointer,
        totals_pointer: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        values = coop.ThreadData(items_per_thread)
        lengths = coop.ThreadData(items_per_thread)
        for item in cutlass.range_constexpr(items_per_thread):
            values[item] = cutlass.Int32(
                block.rank() * items_per_thread + item + 10
            )
            lengths[item] = cutlass.Uint32(2)
        scratch = coop.TempStorage(auto_sync=True)
        window = coop.run_length_decode(
            block,
            values,
            lengths,
            decoded_items_per_thread=2,
            decoded_window_offset=3,
            temp_storage=scratch,
        )
        coop.store(block, window_pointer, window)
        destination = cute.make_tensor(
            stream_pointer, cute.make_layout(64 * items_per_thread + 5)
        )
        total = coop.run_length_decode_into(
            block,
            values,
            lengths,
            destination,
            decoded_items_per_thread=1,
            destination_offset=5,
            temp_storage=scratch,
        )
        totals = cute.make_tensor(totals_pointer, cute.make_layout(32))
        totals[block.rank()] = total

    @cute.jit
    def launch(
        window_pointer: cute.Pointer,
        stream_pointer: cute.Pointer,
        totals_pointer: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        decode_runs(
            window_pointer, stream_pointer, totals_pointer, items_per_thread
        ).launch(grid=1, block=32)

    # qualified-run-length-example-end
    expected = np.repeat(
        np.arange(10, 10 + 32 * items_per_thread, dtype=np.int32), 2
    )
    window = np.full(64, -1, dtype=np.int32)
    stream = np.full(len(expected) + 5, -1, dtype=np.int32)
    totals = np.zeros(32, dtype=np.uint32)
    with (
        device_array(window) as win,
        device_array(stream) as dest,
        device_array(totals) as total,
    ):
        launch(win, dest, total, items_per_thread)
    np.testing.assert_array_equal(
        window, np.r_[expected[3:], np.zeros(3, dtype=np.int32)][:64]
    )
    np.testing.assert_array_equal(stream[5:], expected)
    np.testing.assert_array_equal(stream[:5], -1)
    np.testing.assert_array_equal(totals, len(expected))
