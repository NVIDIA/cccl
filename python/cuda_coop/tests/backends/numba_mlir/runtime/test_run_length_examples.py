# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execute the documented window and bulk run-length decode examples.

Keep the marked regions self-contained for Sphinx. The window examples
check padded tails and qualified auxiliary buffers. The bulk example crosses
three internal windows. It checks that the destination offset and final-window
masking leave elements outside the decoded interval unchanged, for both
input run extents.
"""

import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


def test_common_run_length_window_example():
    """Check the common window result and its zero-padded tail."""

    # common-run-length-window-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    from cuda import coop

    @cuda.jit
    def decode_window(values, lengths, output, items_per_thread):
        block = coop.this_block()
        runs = coop.ThreadData(items_per_thread)
        sizes = coop.ThreadData(items_per_thread)
        coop.load(block, values, runs)
        coop.load(block, lengths, sizes)
        decoded = coop.run_length_decode(
            block,
            runs,
            sizes,
            decoded_items_per_thread=4,
            decoded_window_offset=2,
        )
        coop.store(block, output, decoded)

    for items_per_thread in (1, 4):
        # Zero-length runs pad the block's input tile.
        values = np.zeros(128 * items_per_thread, dtype=np.int32)
        lengths = np.zeros(128 * items_per_thread, dtype=np.uint32)
        values[:2] = [7, 9]
        lengths[:2] = [3, 2]
        output = np.empty(512, dtype=np.int32)
        decode_window[1, 128](values, lengths, output, items_per_thread)
        cuda.synchronize()
        np.testing.assert_array_equal(output[:3], [7, 9, 9])
        np.testing.assert_array_equal(output[3:], 0)
    # common-run-length-window-example-end


def test_run_length_window_example():
    """Check a shifted window, full total, and relative run positions."""

    # run-length-window-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as coop

    @cuda.jit
    def decode_window(
        values, lengths, output, offsets, totals, items_per_thread
    ):
        block = coop.this_block()
        runs = coop.ThreadData(items_per_thread)
        sizes = coop.ThreadData(items_per_thread)
        coop.load(block, values, runs)
        coop.load(block, lengths, sizes)
        relative = coop.ThreadData(items_per_thread=4)
        total = coop.ThreadData(items_per_thread=1)
        decoded = coop.run_length_decode(
            block,
            runs,
            sizes,
            decoded_items_per_thread=4,
            decoded_window_offset=2,
            relative_offsets=relative,
            total_decoded_size=total,
        )
        coop.store(block, output, decoded)
        coop.store(block, offsets, relative)
        totals[cuda.threadIdx.x] = total[0]

    for items_per_thread in (1, 4):
        # Two real runs, followed by zero-length padding for the block tile.
        values = np.zeros(128 * items_per_thread, dtype=np.int32)
        lengths = np.zeros(128 * items_per_thread, dtype=np.uint32)
        values[:2] = [7, 9]
        lengths[:2] = [3, 2]
        output = np.empty(512, dtype=np.int32)
        offsets = np.empty(512, dtype=np.uint32)
        totals = np.empty(128, dtype=np.uint32)
        decode_window[1, 128](
            values, lengths, output, offsets, totals, items_per_thread
        )
        cuda.synchronize()
        np.testing.assert_array_equal(output[:3], [7, 9, 9])
        np.testing.assert_array_equal(offsets[:3], [2, 0, 1])
        assert np.all(totals == 5)
        assert np.all(output[3:] == 0)
        assert np.all(offsets[3:] == np.iinfo(np.uint32).max)
    # run-length-window-example-end


def test_run_length_bulk_example():
    """Check three internal windows and the untouched destination margins."""

    # run-length-bulk-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    from cuda import coop

    @cuda.jit
    def decode_all(values, lengths, output, totals, items_per_thread):
        block = coop.this_block()
        runs = coop.ThreadData(items_per_thread)
        sizes = coop.ThreadData(items_per_thread)
        coop.load(block, values, runs)
        coop.load(block, lengths, sizes)
        # CUB prepares the run table once for all internal windows.
        total = coop.run_length_decode_into(
            block,
            runs,
            sizes,
            output,
            decoded_items_per_thread=4,
            destination_offset=3,
        )
        totals[cuda.threadIdx.x] = total

    for items_per_thread in (1, 4):
        values = np.zeros(128 * items_per_thread, dtype=np.int32)
        lengths = np.zeros(128 * items_per_thread, dtype=np.uint32)
        values[:3] = [7, 9, 2]
        lengths[:3] = [
            3,
            1100,
            2,
        ]  # Three windows; the final window is partial.
        output = np.full(1110, -1, dtype=np.int32)
        totals = np.empty(128, dtype=np.uint32)
        decode_all[1, 128](values, lengths, output, totals, items_per_thread)
        cuda.synchronize()
        np.testing.assert_array_equal(
            output[3:1108], np.repeat(values[:3], lengths[:3])
        )
        assert np.all(totals == 1105)
        assert np.all(output[:3] == -1) and np.all(output[1108:] == -1)
    # run-length-bulk-example-end


def test_qualified_run_length_bulk_example():
    # qualified-run-length-bulk-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as coop

    @cuda.jit
    def decode_all(values, lengths, output, relative, totals, items_per_thread):
        block = coop.this_block()
        runs = coop.ThreadData(items_per_thread)
        sizes = coop.ThreadData(items_per_thread)
        coop.load(block, values, runs)
        coop.load(block, lengths, sizes)
        # CUB prepares the run table once for all internal windows.
        total = coop.run_length_decode_into(
            block,
            runs,
            sizes,
            output,
            decoded_items_per_thread=4,
            destination_offset=3,
            relative_offsets=relative,
            decoded_offset_dtype=np.uint64,
        )
        totals[cuda.threadIdx.x] = total

    for items_per_thread in (1, 4):
        values = np.zeros(128 * items_per_thread, dtype=np.int32)
        lengths = np.zeros(128 * items_per_thread, dtype=np.uint32)
        values[:3] = [7, 9, 2]
        # Three windows; the final window is partial.
        lengths[:3] = [3, 1100, 2]
        output = np.full(1110, -1, dtype=np.int32)
        sentinel = np.iinfo(np.uint64).max
        relative = np.full(1110, sentinel, dtype=np.uint64)
        totals = np.empty(128, dtype=np.uint64)
        decode_all[1, 128](
            values, lengths, output, relative, totals, items_per_thread
        )
        cuda.synchronize()
        np.testing.assert_array_equal(
            output[3:1108], np.repeat(values[:3], lengths[:3])
        )
        expected_offsets = np.concatenate(
            [np.arange(length, dtype=np.uint64) for length in lengths[:3]]
        )
        np.testing.assert_array_equal(relative[3:1108], expected_offsets)
        assert np.all(relative[:3] == sentinel)
        assert np.all(relative[1108:] == sentinel)
        assert np.all(totals == 1105)
        assert np.all(output[:3] == -1) and np.all(output[1108:] == -1)
    # qualified-run-length-bulk-example-end
