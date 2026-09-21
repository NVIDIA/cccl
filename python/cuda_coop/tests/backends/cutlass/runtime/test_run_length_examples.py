# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Executable Run Length Decode example for the programming guide."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


def test_decode_example():
    # example-begin
    import cutlass
    from cutlass import cute

    from cuda import coop

    @cute.kernel
    def decode_runs(
        window_pointer: cute.Pointer,
        stream_pointer: cute.Pointer,
        totals_pointer: cute.Pointer,
    ):
        block = coop.this_block()
        rank = cutlass.Int32(block.rank())
        values = coop.ThreadData(1, dtype=cutlass.Int32)
        values[0] = rank + 10
        lengths = coop.ThreadData(1, dtype=cutlass.Uint32)
        lengths[0] = cutlass.Uint32(2)
        scratch = coop.TempStorage()
        window = coop.run_length_decode(
            block,
            values,
            lengths,
            decoded_items_per_thread=2,
            decoded_window_offset=3,
            temp_storage=scratch,
        )
        coop.store(block, window_pointer, window)
        destination = cute.make_tensor(stream_pointer, cute.make_layout(80))
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
        totals[rank] = total

    @cute.jit
    def launch(
        window_pointer: cute.Pointer,
        stream_pointer: cute.Pointer,
        totals_pointer: cute.Pointer,
    ):
        decode_runs(window_pointer, stream_pointer, totals_pointer).launch(
            grid=1,
            block=32,
        )

    # example-end
    expected = np.repeat(np.arange(10, 42, dtype=np.int32), 2)
    window = np.full(64, -1, dtype=np.int32)
    stream = np.full(80, -1, dtype=np.int32)
    totals = np.zeros(32, dtype=np.uint32)
    with (
        device_array(window) as win,
        device_array(stream) as dest,
        device_array(totals) as total,
    ):
        launch(win, dest, total)
    np.testing.assert_array_equal(
        window, np.concatenate((expected[3:], np.zeros(3, dtype=np.int32)))
    )
    np.testing.assert_array_equal(stream[5:69], expected)
    np.testing.assert_array_equal(stream[:5], -1)
    np.testing.assert_array_equal(stream[69:], -1)
    np.testing.assert_array_equal(totals, len(expected))
