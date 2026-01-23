# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
import numba
import numpy as np
from numba import cuda

from cuda import coop
from cuda.coop import BlockLoadAlgorithm

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-end imports


def _expected_decode(run_values, run_lengths):
    decoded_items = []
    for value, length in zip(run_values, run_lengths):
        for _ in range(length):
            decoded_items.append(value)
    return np.asarray(decoded_items, dtype=run_values.dtype)


def test_block_run_length_decode():
    # example-begin run-length
    threads_per_block = 32
    runs_per_thread = 2
    decoded_items_per_thread = 4

    total_runs = threads_per_block * runs_per_thread
    window_size = threads_per_block * decoded_items_per_thread

    @cuda.jit
    def kernel(run_values, run_lengths, decoded_items_out, total_decoded_size_out):
        runs_per_thread = 2
        decoded_items_per_thread = 4
        run_values_local = coop.local.array(runs_per_thread, dtype=run_values.dtype)
        run_lengths_local = coop.local.array(runs_per_thread, dtype=run_lengths.dtype)

        block_offset = cuda.blockIdx.x * runs_per_thread * cuda.blockDim.x

        coop.block.load(
            run_values[block_offset:],
            run_values_local,
            items_per_thread=runs_per_thread,
            algorithm=BlockLoadAlgorithm.DIRECT,
        )
        coop.block.load(
            run_lengths[block_offset:],
            run_lengths_local,
            items_per_thread=runs_per_thread,
            algorithm=BlockLoadAlgorithm.DIRECT,
        )

        decoded_offset_dtype = total_decoded_size_out.dtype
        total_decoded_size = coop.local.array(1, dtype=decoded_offset_dtype)
        total_decoded_size[0] = 0

        run_length = coop.block.run_length(
            run_values_local,
            run_lengths_local,
            runs_per_thread,
            decoded_items_per_thread,
            total_decoded_size,
            decoded_offset_dtype=decoded_offset_dtype,
        )

        decoded_items = coop.local.array(
            decoded_items_per_thread, dtype=run_values.dtype
        )
        decoded_window_offset = 0
        run_length.decode(decoded_items, decoded_window_offset)

        base = cuda.threadIdx.x * decoded_items_per_thread
        for i in range(decoded_items_per_thread):
            decoded_items_out[base + i] = decoded_items[i]

        if cuda.threadIdx.x == 0:
            total_decoded_size_out[cuda.blockIdx.x] = total_decoded_size[0]

    # example-end run-length

    h_run_values = np.arange(total_runs, dtype=np.uint32)
    h_run_lengths = (np.arange(total_runs, dtype=np.uint32) % 3) + 1
    h_run_lengths[-1] += window_size - int(h_run_lengths.sum())

    expected_items = _expected_decode(h_run_values, h_run_lengths)[:window_size]

    d_run_values = cuda.to_device(h_run_values)
    d_run_lengths = cuda.to_device(h_run_lengths)
    d_decoded_items = cuda.device_array(window_size, dtype=h_run_values.dtype)
    d_total_decoded_size = cuda.device_array(1, dtype=h_run_lengths.dtype)

    kernel[1, threads_per_block](
        d_run_values,
        d_run_lengths,
        d_decoded_items,
        d_total_decoded_size,
    )
    cuda.synchronize()

    h_decoded_items = d_decoded_items.copy_to_host()
    np.testing.assert_array_equal(h_decoded_items, expected_items)
