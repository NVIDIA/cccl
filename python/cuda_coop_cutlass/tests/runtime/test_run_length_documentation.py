# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare caller-controlled decode windows with whole-stream decoding."""

import numpy as np
import pytest

from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_run_length_example(items_per_thread):
    # cutlass-run-length-example-begin
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
        total_runs = 32 * items_per_thread
        total_decoded = 3 * total_runs - 2
        values = coop.ThreadData(items_per_thread)
        lengths = coop.ThreadData(items_per_thread)
        for item in cutlass.range_constexpr(items_per_thread):
            run = cutlass.Int32(block.rank()) * items_per_thread + item
            values[item] = run + 10
            length = cutlass.Uint32(3)
            if run == total_runs - 1:
                length = cutlass.Uint32(1)
            lengths[item] = length

        scratch = coop.TempStorage(auto_sync=True)

        # Decode one window of 32 items on each trip. Every call checks the
        # lengths again. valid_items keeps the zeros past the end of the
        # stream out of the store.
        for decoded_offset in cutlass.range(0, total_decoded, 32):
            window = coop.run_length_decode(
                block,
                values,
                lengths,
                decoded_items_per_thread=1,
                decoded_window_offset=decoded_offset,
                temp_storage=scratch,
            )
            valid_items = total_decoded - decoded_offset
            if valid_items > 32:
                valid_items = cutlass.Int32(32)
            coop.store(
                block,
                window_pointer,
                window,
                offset=5 + decoded_offset,
                valid_items=valid_items,
            )

        # Decode the same stream in one call. It checks the lengths once and
        # writes one window of 32 items after another.
        destination = cute.make_tensor(
            stream_pointer, cute.make_layout(total_decoded + 16)
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

    # cutlass-run-length-example-end
    run_lengths = np.full(32 * items_per_thread, 3, dtype=np.uint32)
    run_lengths[-1] = 1
    expected = np.repeat(
        np.arange(10, 10 + 32 * items_per_thread, dtype=np.int32), run_lengths
    )
    window = np.full(len(expected) + 16, -1, dtype=np.int32)
    stream = np.full(len(expected) + 16, -1, dtype=np.int32)
    totals = np.zeros(32, dtype=np.uint32)
    with (
        device_array(window) as win,
        device_array(stream) as dest,
        device_array(totals) as total,
    ):
        launch(win, dest, total, items_per_thread)
    for output in (window, stream):
        np.testing.assert_array_equal(output[5 : 5 + len(expected)], expected)
        np.testing.assert_array_equal(output[:5], -1)
        np.testing.assert_array_equal(output[5 + len(expected) :], -1)
    np.testing.assert_array_equal(totals, len(expected))
