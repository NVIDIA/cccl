# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
import pytest
from numba import cuda

from cuda import coop
from cuda.coop.warp import WarpExchangeType

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_block_scan_items_per_thread_mismatch_raises():
    @cuda.jit
    def kernel(d_in, d_out):
        thread_data = coop.local.array(2, dtype=d_in.dtype)
        coop.block.load(d_in, thread_data, items_per_thread=2)
        coop.block.scan(thread_data, thread_data, items_per_thread=4)
        coop.block.store(d_out, thread_data, items_per_thread=2)

    h_input = np.arange(64, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    with pytest.raises(
        Exception,
        match="coop.block.scan items_per_thread must match the array shape",
    ):
        kernel[1, 32](d_input, d_output)


def test_block_run_length_missing_total_decoded_size_raises():
    runs_per_thread = 2
    decoded_items_per_thread = 4

    @cuda.jit
    def kernel(run_values, run_lengths):
        run_values_local = coop.local.array(runs_per_thread, dtype=run_values.dtype)
        run_lengths_local = coop.local.array(runs_per_thread, dtype=run_lengths.dtype)

        coop.block.load(
            run_values,
            run_values_local,
            items_per_thread=runs_per_thread,
        )
        coop.block.load(
            run_lengths,
            run_lengths_local,
            items_per_thread=runs_per_thread,
        )

        coop.block.run_length(
            run_values_local,
            run_lengths_local,
            runs_per_thread,
            decoded_items_per_thread,
        )

    h_run_values = np.arange(64, dtype=np.uint32)
    h_run_lengths = np.full(64, 1, dtype=np.uint32)
    d_run_values = cuda.to_device(h_run_values)
    d_run_lengths = cuda.to_device(h_run_lengths)

    with pytest.raises(
        Exception,
        match="coop.block.run_length requires at least 5 positional arguments",
    ):
        kernel[1, 32](d_run_values, d_run_lengths)


def test_block_run_length_total_decoded_size_none_raises():
    runs_per_thread = 2
    decoded_items_per_thread = 4

    @cuda.jit
    def kernel(run_values, run_lengths):
        run_values_local = coop.local.array(runs_per_thread, dtype=run_values.dtype)
        run_lengths_local = coop.local.array(runs_per_thread, dtype=run_lengths.dtype)

        coop.block.load(
            run_values,
            run_values_local,
            items_per_thread=runs_per_thread,
        )
        coop.block.load(
            run_lengths,
            run_lengths_local,
            items_per_thread=runs_per_thread,
        )

        coop.block.run_length(
            run_values_local,
            run_lengths_local,
            runs_per_thread,
            decoded_items_per_thread,
            total_decoded_size=None,
        )

    h_run_values = np.arange(64, dtype=np.uint32)
    h_run_lengths = np.full(64, 1, dtype=np.uint32)
    d_run_values = cuda.to_device(h_run_values)
    d_run_lengths = cuda.to_device(h_run_lengths)

    with pytest.raises(
        Exception,
        match="total_decoded_size must be a device array",
    ):
        kernel[1, 32](d_run_values, d_run_lengths)


def test_warp_exchange_thread_data_items_per_thread_mismatch_raises():
    items_per_thread = 4

    @cuda.jit
    def kernel(d_out):
        thread_data = coop.ThreadData(items_per_thread, dtype=numba.int32)
        output_items = coop.ThreadData(items_per_thread, dtype=numba.int32)
        coop.warp.exchange(
            thread_data,
            output_items,
            items_per_thread=5,
            warp_exchange_type=WarpExchangeType.StripedToBlocked,
            threads_in_warp=32,
        )
        for i in range(items_per_thread):
            d_out[cuda.threadIdx.x * items_per_thread + i] = output_items[i]

    d_output = cuda.device_array(32 * items_per_thread, dtype=np.int32)
    with pytest.raises(
        Exception,
        match="coop.warp.exchange items_per_thread must match array shape",
    ):
        kernel[1, 32](d_output)
