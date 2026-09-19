# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executable examples for windowed and bulk Run Length Decode."""

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


def test_run_length_window_example():
    # run-length-window-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop

    @cuda.jit
    def decode_window(values, lengths, output, offsets, totals):
        block = numba_coop.this_block()
        runs = numba_coop.ThreadData(2)
        sizes = numba_coop.ThreadData(2)
        numba_coop.load(block, values, runs)
        numba_coop.load(block, lengths, sizes)
        relative = numba_coop.ThreadData(4, dtype=np.uint32)
        total = numba_coop.ThreadData(1, dtype=np.uint32)
        decoded = numba_coop.run_length_decode(
            block,
            runs,
            sizes,
            decoded_items_per_thread=4,
            decoded_window_offset=2,
            relative_offsets=relative,
            total_decoded_size=total,
        )
        numba_coop.store(block, output, decoded)
        numba_coop.store(block, offsets, relative)
        totals[cuda.threadIdx.x] = total[0]

    # Two real runs, followed by zero-length padding for the block tile.
    values = np.zeros(256, dtype=np.int32)
    lengths = np.zeros(256, dtype=np.uint32)
    values[:2] = [7, 9]
    lengths[:2] = [3, 2]
    output = np.empty(512, dtype=np.int32)
    offsets = np.empty(512, dtype=np.uint32)
    totals = np.empty(128, dtype=np.uint32)
    decode_window[1, 128](values, lengths, output, offsets, totals)
    cuda.synchronize()
    np.testing.assert_array_equal(output[:3], [7, 9, 9])
    np.testing.assert_array_equal(offsets[:3], [2, 0, 1])
    assert np.all(totals == 5)
    assert np.all(output[3:] == 0)
    assert np.all(offsets[3:] == np.iinfo(np.uint32).max)
    # run-length-window-example-end


def test_run_length_bulk_example():
    # run-length-bulk-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    from cuda import coop

    coop.register("numba-cuda-mlir")

    @cuda.jit
    def decode_all(values, lengths, output, totals):
        block = coop.this_block()
        runs = coop.ThreadData(2)
        sizes = coop.ThreadData(2)
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

    values = np.zeros(256, dtype=np.int32)
    lengths = np.zeros(256, dtype=np.uint32)
    values[:3] = [7, 9, 2]
    lengths[:3] = [3, 1100, 2]  # Three windows; the final window is partial.
    output = np.full(1110, -1, dtype=np.int32)
    totals = np.empty(128, dtype=np.uint32)
    decode_all[1, 128](values, lengths, output, totals)
    cuda.synchronize()
    np.testing.assert_array_equal(output[3:1108], np.repeat(values[:3], lengths[:3]))
    assert np.all(totals == 1105)
    assert np.all(output[:3] == -1) and np.all(output[1108:] == -1)
    # run-length-bulk-example-end
