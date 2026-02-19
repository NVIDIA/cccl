# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import operator

import numba
import numpy as np
from numba import cuda

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def _exclusive_scan_host(values, op, identity):
    out = np.empty_like(values)
    acc = identity
    for idx, val in enumerate(values):
        out[idx] = acc
        acc = op(acc, val)
    return out


def _inclusive_scan_host(values, op, identity):
    out = np.empty_like(values)
    acc = identity
    for idx, val in enumerate(values):
        acc = op(acc, val)
        out[idx] = acc
    return out


def test_block_exclusive_sum_single_phase():
    threads_per_block = 128
    items_per_thread = 4
    total_items = threads_per_block * items_per_thread

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(d_in, thread_data, items_per_thread=items_per_thread)
        coop.block.exclusive_sum(
            thread_data,
            thread_data,
            items_per_thread=items_per_thread,
        )
        coop.block.store(d_out, thread_data, items_per_thread=items_per_thread)

    h_input = np.ones(total_items, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output, items_per_thread)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_reference = _exclusive_scan_host(h_input, operator.add, np.int32(0))
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_inclusive_sum_single_phase():
    threads_per_block = 128
    items_per_thread = 4
    total_items = threads_per_block * items_per_thread

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(d_in, thread_data, items_per_thread=items_per_thread)
        coop.block.inclusive_sum(
            thread_data,
            thread_data,
            items_per_thread=items_per_thread,
        )
        coop.block.store(d_out, thread_data, items_per_thread=items_per_thread)

    h_input = np.ones(total_items, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output, items_per_thread)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_reference = _inclusive_scan_host(h_input, operator.add, np.int32(0))
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_exclusive_scan_single_phase_bit_xor():
    threads_per_block = 64
    items_per_thread = 4
    total_items = threads_per_block * items_per_thread

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(d_in, thread_data, items_per_thread=items_per_thread)
        coop.block.exclusive_scan(
            thread_data,
            thread_data,
            items_per_thread=items_per_thread,
            scan_op="bit_xor",
        )
        coop.block.store(d_out, thread_data, items_per_thread=items_per_thread)

    h_input = np.random.randint(0, 256, total_items, dtype=np.uint32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output, items_per_thread)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_reference = _exclusive_scan_host(h_input, operator.xor, np.uint32(0))
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_inclusive_scan_single_phase_bit_xor():
    threads_per_block = 64
    items_per_thread = 4
    total_items = threads_per_block * items_per_thread

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(d_in, thread_data, items_per_thread=items_per_thread)
        coop.block.inclusive_scan(
            thread_data,
            thread_data,
            items_per_thread=items_per_thread,
            scan_op="bit_xor",
        )
        coop.block.store(d_out, thread_data, items_per_thread=items_per_thread)

    h_input = np.random.randint(0, 256, total_items, dtype=np.uint32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output, items_per_thread)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_reference = _inclusive_scan_host(h_input, operator.xor, np.uint32(0))
    np.testing.assert_array_equal(h_output, h_reference)
