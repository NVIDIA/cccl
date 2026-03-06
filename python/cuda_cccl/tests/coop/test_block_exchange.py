# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
test_block_exchange.py

This file contains unit tests for cuda.coop.block_exchange.
"""

from functools import reduce
from operator import mul

import numba
import numpy as np
import pytest
from helpers import (
    NUMBA_TYPES_TO_NP,
    Complex,
    complex_type,
    random_int,
    row_major_tid,
)
from numba import cuda, types

from cuda import coop
from cuda.coop.block import BlockExchangeType

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

striped_to_blocked = BlockExchangeType.StripedToBlocked
blocked_to_striped = BlockExchangeType.BlockedToStriped
warp_striped_to_blocked = BlockExchangeType.WarpStripedToBlocked
blocked_to_warp_striped = BlockExchangeType.BlockedToWarpStriped
scatter_to_blocked = BlockExchangeType.ScatterToBlocked
scatter_to_striped = BlockExchangeType.ScatterToStriped
scatter_to_striped_guarded = BlockExchangeType.ScatterToStripedGuarded
scatter_to_striped_flagged = BlockExchangeType.ScatterToStripedFlagged

WARP_SIZE = 32


@pytest.mark.parametrize("T", [types.int32, types.float64])
@pytest.mark.parametrize("threads_per_block", [32, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("warp_time_slicing", [False, True])
@pytest.mark.parametrize("separate_input_output_arrays", [False, True])
def test_striped_to_blocked(
    T,
    threads_per_block,
    items_per_thread,
    warp_time_slicing,
    separate_input_output_arrays,
):
    """
    Tests the striped_to_blocked block-wide data exchange.
    """
    T_np = NUMBA_TYPES_TO_NP[T]
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if separate_input_output_arrays:

        @cuda.jit
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            thread_data_in = cuda.local.array(items_per_thread, dtype=T)
            thread_data_out = cuda.local.array(items_per_thread, dtype=T)

            # Load input data (striped)
            for i in range(items_per_thread):
                idx = tid + i * num_threads
                if idx < input_arr.shape[0]:
                    thread_data_in[i] = input_arr[idx]

            coop.block.exchange(
                thread_data_in,
                thread_data_out,
                items_per_thread=items_per_thread,
                block_exchange_type=striped_to_blocked,
                warp_time_slicing=warp_time_slicing,
            )

            # Store output data (blocked)
            for i in range(items_per_thread):
                idx = tid * items_per_thread + i
                if idx < output_arr.shape[0]:
                    output_arr[idx] = thread_data_out[i]

    else:

        @cuda.jit
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            items = cuda.local.array(items_per_thread, dtype=T)

            # Load input data (striped)
            for i in range(items_per_thread):
                idx = tid + i * num_threads
                if idx < input_arr.shape[0]:
                    items[i] = input_arr[idx]

            coop.block.exchange(
                items,
                items_per_thread=items_per_thread,
                block_exchange_type=striped_to_blocked,
                warp_time_slicing=warp_time_slicing,
            )

            # Store output data (blocked)
            for i in range(items_per_thread):
                idx = tid * items_per_thread + i
                if idx < output_arr.shape[0]:
                    output_arr[idx] = items[i]

    total_items = num_threads * items_per_thread
    h_input = random_int(total_items, T_np)

    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=T_np)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()

    # Calculate reference result
    h_ref = np.zeros(total_items, dtype=T_np)
    # Iterate over each position in the destination (blocked) array
    for blocked_idx in range(total_items):
        # Determine which source thread (s_tid) and which item within
        # that thread's stripe (s_item_idx_in_stripe) contributed to
        # h_ref[blocked_idx].
        #
        # This is based on the CUB definition for StripedToBlocked:
        #   Output element at global_blocked_idx comes from an input
        #   element defined by:
        #       source_thread_idx = global_blocked_idx % num_threads
        #       source_item_in_stripe_idx = global_blocked_idx // num_threads

        s_tid = blocked_idx % num_threads
        s_item_idx_in_stripe = blocked_idx // num_threads

        # Convert these (s_tid, s_item_idx_in_stripe) 2D-like indices
        # to a 1D index into the h_input array (which is striped).
        # In a striped layout, element (thread_k, item_j) is at
        # k + j * num_threads.
        src_striped_1d_idx = s_tid + s_item_idx_in_stripe * num_threads

        h_ref[blocked_idx] = h_input[src_striped_1d_idx]

    # We just transformed data; we didn't do any math that may have introduced
    # floating point errors, so we can use np.testing.assert_array_equal for
    # both float and int types.
    np.testing.assert_array_equal(output, h_ref)

    sig = (T[::1], T[::1])  # Signature for 1D arrays of type T
    sass = kernel.inspect_sass(sig)
    assert "LDL" not in sass  # Check for global loads
    assert "STL" not in sass  # Check for global stores


def test_block_exchange_two_phase_striped_to_blocked():
    threads_per_block = 64
    items_per_thread = 2
    dtype = np.int32

    block_exchange = coop.block.exchange(
        striped_to_blocked,
        numba.int32,
        threads_per_block,
        items_per_thread,
    )

    @cuda.jit
    def kernel(input_arr, output_arr):
        tid = row_major_tid()
        items = cuda.local.array(items_per_thread, numba.int32)
        for i in range(items_per_thread):
            idx = tid + i * threads_per_block
            items[i] = input_arr[idx]

        block_exchange(items)

        for i in range(items_per_thread):
            output_arr[tid * items_per_thread + i] = items[i]

    total_items = threads_per_block * items_per_thread
    h_input = np.arange(total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()

    expected = np.zeros_like(h_output)
    for blocked_idx in range(total_items):
        s_tid = blocked_idx % threads_per_block
        s_item_idx_in_stripe = blocked_idx // threads_per_block
        src_striped_idx = s_tid + s_item_idx_in_stripe * threads_per_block
        expected[blocked_idx] = h_input[src_striped_idx]

    np.testing.assert_array_equal(h_output, expected)


def test_block_exchange_scatter_to_blocked_thread_data_ranks_infers_items_per_thread():
    threads_per_block = 64
    items_per_thread = 4
    total_items = threads_per_block * items_per_thread

    @cuda.jit
    def kernel(d_in, d_ranks, d_out):
        input_items = coop.ThreadData(items_per_thread, dtype=d_in.dtype)
        output_items = coop.ThreadData(items_per_thread, dtype=d_in.dtype)
        ranks = coop.ThreadData(items_per_thread, dtype=d_ranks.dtype)

        coop.block.load(d_in, input_items)
        coop.block.load(d_ranks, ranks)

        coop.block.exchange(
            input_items,
            output_items,
            ranks=ranks,
            block_exchange_type=scatter_to_blocked,
        )

        coop.block.store(d_out, output_items)

    h_input = np.arange(total_items, dtype=np.int32)
    h_ranks = np.arange(total_items, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_ranks = cuda.to_device(h_ranks)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_ranks, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_input)


def test_block_exchange_scatter_to_striped_flagged_thread_data_infers_items_per_thread():
    threads_per_block = 64
    items_per_thread = 4
    total_items = threads_per_block * items_per_thread

    @cuda.jit
    def kernel(d_in, d_ranks, d_valid_flags, d_out):
        tid = row_major_tid()
        input_items = coop.ThreadData(items_per_thread, dtype=d_in.dtype)
        output_items = coop.ThreadData(items_per_thread, dtype=d_in.dtype)
        ranks = coop.ThreadData(items_per_thread, dtype=d_ranks.dtype)
        valid_flags = coop.ThreadData(items_per_thread, dtype=d_valid_flags.dtype)

        coop.block.load(d_in, input_items)
        coop.block.load(d_ranks, ranks)
        coop.block.load(d_valid_flags, valid_flags)

        coop.block.exchange(
            input_items,
            output_items,
            ranks=ranks,
            valid_flags=valid_flags,
            block_exchange_type=scatter_to_striped_flagged,
        )

        for i in range(items_per_thread):
            d_out[tid + i * threads_per_block] = output_items[i]

    h_input = random_int(total_items, np.int32)
    h_ranks = np.arange(total_items, dtype=np.int32)
    h_valid_flags = np.ones(total_items, dtype=np.uint8)
    d_input = cuda.to_device(h_input)
    d_ranks = cuda.to_device(h_ranks)
    d_valid_flags = cuda.to_device(h_valid_flags)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_ranks, d_valid_flags, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_input)


def test_block_exchange_temp_storage():
    threads_per_block = 64
    items_per_thread = 2
    dtype = np.int32

    block_exchange = coop.block.exchange(
        striped_to_blocked,
        numba.int32,
        threads_per_block,
        items_per_thread,
    )
    temp_storage_bytes = block_exchange.temp_storage_bytes
    temp_storage_alignment = block_exchange.temp_storage_alignment

    @cuda.jit
    def kernel(input_arr, output_arr):
        tid = row_major_tid()
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        items = cuda.local.array(items_per_thread, numba.int32)
        for i in range(items_per_thread):
            idx = tid + i * threads_per_block
            items[i] = input_arr[idx]

        block_exchange(items, temp_storage=temp_storage)

        for i in range(items_per_thread):
            output_arr[tid * items_per_thread + i] = items[i]

    total_items = threads_per_block * items_per_thread
    h_input = np.arange(total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()

    expected = np.zeros_like(h_output)
    for blocked_idx in range(total_items):
        s_tid = blocked_idx % threads_per_block
        s_item_idx_in_stripe = blocked_idx // threads_per_block
        src_striped_idx = s_tid + s_item_idx_in_stripe * threads_per_block
        expected[blocked_idx] = h_input[src_striped_idx]

    np.testing.assert_array_equal(h_output, expected)


def test_block_exchange_temp_storage_single_phase():
    threads_per_block = 64
    items_per_thread = 2
    dtype = np.int32

    block_exchange = coop.block.exchange(
        striped_to_blocked,
        numba.int32,
        threads_per_block,
        items_per_thread,
    )
    temp_storage_bytes = block_exchange.temp_storage_bytes
    temp_storage_alignment = block_exchange.temp_storage_alignment

    @cuda.jit
    def kernel(input_arr, output_arr):
        tid = row_major_tid()
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        items = cuda.local.array(items_per_thread, numba.int32)
        for i in range(items_per_thread):
            idx = tid + i * threads_per_block
            items[i] = input_arr[idx]

        coop.block.exchange(
            items,
            items_per_thread=items_per_thread,
            block_exchange_type=striped_to_blocked,
            temp_storage=temp_storage,
        )

        for i in range(items_per_thread):
            output_arr[tid * items_per_thread + i] = items[i]

    total_items = threads_per_block * items_per_thread
    h_input = np.arange(total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()

    expected = np.zeros_like(h_output)
    for blocked_idx in range(total_items):
        s_tid = blocked_idx % threads_per_block
        s_item_idx_in_stripe = blocked_idx // threads_per_block
        src_striped_idx = s_tid + s_item_idx_in_stripe * threads_per_block
        expected[blocked_idx] = h_input[src_striped_idx]

    np.testing.assert_array_equal(h_output, expected)


@pytest.mark.parametrize("threads_per_block", [32, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [1])
@pytest.mark.parametrize("warp_time_slicing", [False, True])
def test_striped_to_blocked_user_defined_type(
    threads_per_block, items_per_thread, warp_time_slicing
):
    """
    Tests the striped_to_blocked block-wide data exchange for Complex
    user-defined type.

    N.B. User-defined types are currently restricted to items_per_thread == 1.
         However, the test has been written to handle multiple items per thread
         for if and when this support is added.
    """
    T_complex = complex_type
    T_complex_np_component = np.int32

    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    @cuda.jit
    def kernel(input_arr, output_arr):
        tid = row_major_tid()
        thread_data_in = cuda.local.array(items_per_thread, dtype=T_complex)
        thread_data_out = cuda.local.array(items_per_thread, dtype=T_complex)

        total_complex_items_in_block = num_threads * items_per_thread

        for i in range(items_per_thread):
            striped_idx_of_complex_item = tid + i * num_threads
            if striped_idx_of_complex_item < total_complex_items_in_block:
                real_val = input_arr[striped_idx_of_complex_item]
                imag_val = input_arr[
                    striped_idx_of_complex_item + total_complex_items_in_block
                ]
                thread_data_in[i] = Complex(real_val, imag_val)

        coop.block.exchange(
            thread_data_in,
            thread_data_out,
            items_per_thread=items_per_thread,
            block_exchange_type=striped_to_blocked,
            warp_time_slicing=warp_time_slicing,
        )

        for i in range(items_per_thread):
            blocked_idx_of_complex_item = tid * items_per_thread + i
            if blocked_idx_of_complex_item < total_complex_items_in_block:
                output_arr[blocked_idx_of_complex_item] = thread_data_out[i].real
                output_arr[
                    blocked_idx_of_complex_item + total_complex_items_in_block
                ] = thread_data_out[i].imag

    total_complex_items = num_threads * items_per_thread
    h_input_real = random_int(total_complex_items, T_complex_np_component)
    h_input_imag = random_int(total_complex_items, T_complex_np_component)
    h_input_combined = np.concatenate((h_input_real, h_input_imag))

    d_input = cuda.to_device(h_input_combined)
    d_output = cuda.device_array(2 * total_complex_items, dtype=T_complex_np_component)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output_combined = d_output.copy_to_host()
    output_real = output_combined[:total_complex_items]
    output_imag = output_combined[total_complex_items:]

    h_input_complex_objects = np.array(
        [Complex(r, i) for r, i in zip(h_input_real, h_input_imag)], dtype=object
    )
    h_ref_complex_objects = np.empty(total_complex_items, dtype=object)

    # The striped_to_blocked transformation:
    for blocked_complex_idx in range(total_complex_items):
        s_tid = blocked_complex_idx % num_threads
        s_item_idx_in_stripe = blocked_complex_idx // num_threads

        # Convert to 1D index in the striped h_input_complex_objects array
        src_striped_1d_complex_idx = s_tid + s_item_idx_in_stripe * num_threads
        h_ref_complex_objects[blocked_complex_idx] = h_input_complex_objects[
            src_striped_1d_complex_idx
        ]

    # Split the reference Complex objects back into real and imaginary parts
    h_ref_real = np.array(
        [c.real for c in h_ref_complex_objects], dtype=T_complex_np_component
    )
    h_ref_imag = np.array(
        [c.imag for c in h_ref_complex_objects], dtype=T_complex_np_component
    )

    np.testing.assert_array_equal(output_real, h_ref_real)
    np.testing.assert_array_equal(output_imag, h_ref_imag)

    # Signature for SASS inspection (input and output arrays are of int32)
    sig = (
        numba.types.int32[::1],
        numba.types.int32[::1],
    )
    sass = kernel.inspect_sass(sig)
    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.int32])
@pytest.mark.parametrize("threads_per_block", [32, (4, 16)])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("warp_time_slicing", [False, True])
@pytest.mark.parametrize("separate_input_output_arrays", [False, True])
def test_blocked_to_striped(
    T,
    threads_per_block,
    items_per_thread,
    warp_time_slicing,
    separate_input_output_arrays,
):
    """
    Tests the blocked_to_striped block-wide data exchange.
    """
    T_np = NUMBA_TYPES_TO_NP[T]
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if separate_input_output_arrays:

        @cuda.jit
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            thread_data_in = cuda.local.array(items_per_thread, dtype=T)
            thread_data_out = cuda.local.array(items_per_thread, dtype=T)

            # Load input data (blocked)
            for i in range(items_per_thread):
                idx = tid * items_per_thread + i
                thread_data_in[i] = input_arr[idx]

            coop.block.exchange(
                thread_data_in,
                thread_data_out,
                items_per_thread=items_per_thread,
                block_exchange_type=blocked_to_striped,
                warp_time_slicing=warp_time_slicing,
            )

            # Store output data (striped)
            for i in range(items_per_thread):
                idx = tid + i * num_threads
                output_arr[idx] = thread_data_out[i]

    else:

        @cuda.jit
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            items = cuda.local.array(items_per_thread, dtype=T)

            # Load input data (blocked)
            for i in range(items_per_thread):
                idx = tid * items_per_thread + i
                items[i] = input_arr[idx]

            coop.block.exchange(
                items,
                items_per_thread=items_per_thread,
                block_exchange_type=blocked_to_striped,
                warp_time_slicing=warp_time_slicing,
            )

            # Store output data (striped)
            for i in range(items_per_thread):
                idx = tid + i * num_threads
                output_arr[idx] = items[i]

    total_items = num_threads * items_per_thread
    h_input = random_int(total_items, T_np)

    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=T_np)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    np.testing.assert_array_equal(output, h_input)


@pytest.mark.parametrize("T", [types.int32])
@pytest.mark.parametrize("threads_per_block", [64, 128])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("warp_time_slicing", [False, True])
@pytest.mark.parametrize("separate_input_output_arrays", [False, True])
def test_blocked_to_warp_striped(
    T,
    threads_per_block,
    items_per_thread,
    warp_time_slicing,
    separate_input_output_arrays,
):
    """
    Tests the blocked_to_warp_striped block-wide data exchange.
    """
    T_np = NUMBA_TYPES_TO_NP[T]
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if separate_input_output_arrays:

        @cuda.jit
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            thread_data_in = cuda.local.array(items_per_thread, dtype=T)
            thread_data_out = cuda.local.array(items_per_thread, dtype=T)

            # Load input data (blocked)
            for i in range(items_per_thread):
                idx = tid * items_per_thread + i
                thread_data_in[i] = input_arr[idx]

            coop.block.exchange(
                thread_data_in,
                thread_data_out,
                items_per_thread=items_per_thread,
                block_exchange_type=blocked_to_warp_striped,
                warp_time_slicing=warp_time_slicing,
            )

            # Store output data (warp-striped)
            warp_id = tid // WARP_SIZE
            lane_id = tid % WARP_SIZE
            for i in range(items_per_thread):
                idx = warp_id * WARP_SIZE * items_per_thread + lane_id + i * WARP_SIZE
                output_arr[idx] = thread_data_out[i]

    else:

        @cuda.jit
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            items = cuda.local.array(items_per_thread, dtype=T)

            # Load input data (blocked)
            for i in range(items_per_thread):
                idx = tid * items_per_thread + i
                items[i] = input_arr[idx]

            coop.block.exchange(
                items,
                items_per_thread=items_per_thread,
                block_exchange_type=blocked_to_warp_striped,
                warp_time_slicing=warp_time_slicing,
            )

            # Store output data (warp-striped)
            warp_id = tid // WARP_SIZE
            lane_id = tid % WARP_SIZE
            for i in range(items_per_thread):
                idx = warp_id * WARP_SIZE * items_per_thread + lane_id + i * WARP_SIZE
                output_arr[idx] = items[i]

    total_items = num_threads * items_per_thread
    h_input = random_int(total_items, T_np)

    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=T_np)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    np.testing.assert_array_equal(output, h_input)


@pytest.mark.parametrize("T", [types.int32])
@pytest.mark.parametrize("threads_per_block", [64, 128])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("warp_time_slicing", [False, True])
@pytest.mark.parametrize("separate_input_output_arrays", [False, True])
def test_warp_striped_to_blocked(
    T,
    threads_per_block,
    items_per_thread,
    warp_time_slicing,
    separate_input_output_arrays,
):
    """
    Tests the warp_striped_to_blocked block-wide data exchange.
    """
    T_np = NUMBA_TYPES_TO_NP[T]
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if separate_input_output_arrays:

        @cuda.jit
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            thread_data_in = cuda.local.array(items_per_thread, dtype=T)
            thread_data_out = cuda.local.array(items_per_thread, dtype=T)

            # Load input data (warp-striped)
            warp_id = tid // WARP_SIZE
            lane_id = tid % WARP_SIZE
            for i in range(items_per_thread):
                idx = warp_id * WARP_SIZE * items_per_thread + lane_id + i * WARP_SIZE
                thread_data_in[i] = input_arr[idx]

            coop.block.exchange(
                thread_data_in,
                thread_data_out,
                items_per_thread=items_per_thread,
                block_exchange_type=warp_striped_to_blocked,
                warp_time_slicing=warp_time_slicing,
            )

            # Store output data (blocked)
            for i in range(items_per_thread):
                idx = tid * items_per_thread + i
                output_arr[idx] = thread_data_out[i]

    else:

        @cuda.jit
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            items = cuda.local.array(items_per_thread, dtype=T)

            # Load input data (warp-striped)
            warp_id = tid // WARP_SIZE
            lane_id = tid % WARP_SIZE
            for i in range(items_per_thread):
                idx = warp_id * WARP_SIZE * items_per_thread + lane_id + i * WARP_SIZE
                items[i] = input_arr[idx]

            coop.block.exchange(
                items,
                items_per_thread=items_per_thread,
                block_exchange_type=warp_striped_to_blocked,
                warp_time_slicing=warp_time_slicing,
            )

            # Store output data (blocked)
            for i in range(items_per_thread):
                idx = tid * items_per_thread + i
                output_arr[idx] = items[i]

    total_items = num_threads * items_per_thread
    h_input = random_int(total_items, T_np)

    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=T_np)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    np.testing.assert_array_equal(output, h_input)


@pytest.mark.parametrize("T", [types.int32])
@pytest.mark.parametrize("threads_per_block", [64])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("warp_time_slicing", [False, True])
@pytest.mark.parametrize("separate_input_output_arrays", [False, True])
def test_scatter_to_blocked(
    T,
    threads_per_block,
    items_per_thread,
    warp_time_slicing,
    separate_input_output_arrays,
):
    """
    Tests the scatter_to_blocked block-wide data exchange.
    """
    T_np = NUMBA_TYPES_TO_NP[T]
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if separate_input_output_arrays:

        @cuda.jit
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            thread_data_in = cuda.local.array(items_per_thread, dtype=T)
            thread_data_out = cuda.local.array(items_per_thread, dtype=T)
            ranks = cuda.local.array(items_per_thread, dtype=types.int32)

            # Load input data (striped)
            for i in range(items_per_thread):
                idx = tid + i * num_threads
                thread_data_in[i] = input_arr[idx]
                ranks[i] = tid + i * num_threads

            coop.block.exchange(
                thread_data_in,
                thread_data_out,
                items_per_thread=items_per_thread,
                ranks=ranks,
                block_exchange_type=scatter_to_blocked,
                warp_time_slicing=warp_time_slicing,
            )

            # Store output data (blocked)
            for i in range(items_per_thread):
                idx = tid * items_per_thread + i
                output_arr[idx] = thread_data_out[i]

    else:

        @cuda.jit
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            items = cuda.local.array(items_per_thread, dtype=T)
            ranks = cuda.local.array(items_per_thread, dtype=types.int32)

            # Load input data (striped)
            for i in range(items_per_thread):
                idx = tid + i * num_threads
                items[i] = input_arr[idx]
                ranks[i] = tid + i * num_threads

            coop.block.exchange(
                items,
                items_per_thread=items_per_thread,
                ranks=ranks,
                block_exchange_type=scatter_to_blocked,
                warp_time_slicing=warp_time_slicing,
            )

            # Store output data (blocked)
            for i in range(items_per_thread):
                idx = tid * items_per_thread + i
                output_arr[idx] = items[i]

    total_items = num_threads * items_per_thread
    h_input = random_int(total_items, T_np)

    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=T_np)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    np.testing.assert_array_equal(output, h_input)


@pytest.mark.parametrize("T", [types.int32])
@pytest.mark.parametrize("threads_per_block", [64])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("warp_time_slicing", [False, True])
@pytest.mark.parametrize("separate_input_output_arrays", [False, True])
def test_scatter_to_striped(
    T,
    threads_per_block,
    items_per_thread,
    warp_time_slicing,
    separate_input_output_arrays,
):
    """
    Tests the scatter_to_striped block-wide data exchange.
    """
    T_np = NUMBA_TYPES_TO_NP[T]
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if separate_input_output_arrays:

        @cuda.jit
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            thread_data_in = cuda.local.array(items_per_thread, dtype=T)
            thread_data_out = cuda.local.array(items_per_thread, dtype=T)
            ranks = cuda.local.array(items_per_thread, dtype=types.int32)

            # Load input data (blocked)
            for i in range(items_per_thread):
                idx = tid * items_per_thread + i
                thread_data_in[i] = input_arr[idx]
                ranks[i] = tid * items_per_thread + i

            coop.block.exchange(
                thread_data_in,
                thread_data_out,
                items_per_thread=items_per_thread,
                ranks=ranks,
                block_exchange_type=scatter_to_striped,
                warp_time_slicing=warp_time_slicing,
            )

            # Store output data (striped)
            for i in range(items_per_thread):
                idx = tid + i * num_threads
                output_arr[idx] = thread_data_out[i]

    else:

        @cuda.jit
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            items = cuda.local.array(items_per_thread, dtype=T)
            ranks = cuda.local.array(items_per_thread, dtype=types.int32)

            # Load input data (blocked)
            for i in range(items_per_thread):
                idx = tid * items_per_thread + i
                items[i] = input_arr[idx]
                ranks[i] = tid * items_per_thread + i

            coop.block.exchange(
                items,
                items_per_thread=items_per_thread,
                ranks=ranks,
                block_exchange_type=scatter_to_striped,
                warp_time_slicing=warp_time_slicing,
            )

            # Store output data (striped)
            for i in range(items_per_thread):
                idx = tid + i * num_threads
                output_arr[idx] = items[i]

    total_items = num_threads * items_per_thread
    h_input = random_int(total_items, T_np)

    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=T_np)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    np.testing.assert_array_equal(output, h_input)


@pytest.mark.parametrize(
    "block_exchange_type, use_valid_flags, valid_flags_dtype",
    [
        (scatter_to_striped_guarded, False, types.uint8),
        (scatter_to_striped_flagged, True, types.boolean),
        (scatter_to_striped_flagged, True, types.uint8),
    ],
)
def test_scatter_to_striped_guarded_and_flagged(
    block_exchange_type, use_valid_flags, valid_flags_dtype
):
    """
    Tests the scatter_to_striped_guarded and scatter_to_striped_flagged exchanges.
    """
    T = types.int32
    T_np = NUMBA_TYPES_TO_NP[T]
    threads_per_block = 64
    items_per_thread = 4
    num_threads = threads_per_block

    @cuda.jit
    def kernel(input_arr, output_arr):
        tid = row_major_tid()
        items = cuda.local.array(items_per_thread, dtype=T)
        ranks = cuda.local.array(items_per_thread, dtype=types.int32)
        valid_flags = cuda.local.array(items_per_thread, dtype=valid_flags_dtype)

        # Load input data (blocked)
        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            items[i] = input_arr[idx]
            ranks[i] = tid * items_per_thread + i
            if use_valid_flags:
                valid_flags[i] = 1

        if use_valid_flags:
            coop.block.exchange(
                items,
                items_per_thread=items_per_thread,
                ranks=ranks,
                valid_flags=valid_flags,
                block_exchange_type=block_exchange_type,
            )
        else:
            coop.block.exchange(
                items,
                items_per_thread=items_per_thread,
                ranks=ranks,
                block_exchange_type=block_exchange_type,
            )

        # Store output data (striped)
        for i in range(items_per_thread):
            idx = tid + i * num_threads
            output_arr[idx] = items[i]

    total_items = num_threads * items_per_thread
    h_input = random_int(total_items, T_np)

    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=T_np)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    np.testing.assert_array_equal(output, h_input)
