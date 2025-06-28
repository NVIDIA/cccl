# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda

import cuda.cccl.cooperative.experimental as coop

# coop._init_extension()

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_block_load_store_single_phase1():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(d_in, thread_data, items_per_thread)
        coop.block.store(d_out, thread_data, items_per_thread)

    threads_per_block = 128
    items_per_thread = 4
    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    k = kernel[1, threads_per_block]
    k(d_input, d_output, items_per_thread)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_block_load_store_two_phase1():
    dtype = np.int32
    dim = 128
    items_per_thread = 16

    block_load = coop.block.load(dtype, dim, items_per_thread)

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        block_load(d_in, thread_data)
        coop.block.store(d_out, thread_data, items_per_thread)

    h_input = np.random.randint(0, 42, dim * items_per_thread, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    k = kernel[1, dim]
    k(d_input, d_output, items_per_thread)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_block_load_store_two_phase2():
    dtype = np.int32
    dim = 128
    items_per_thread = 16

    # block_load = coop.block.load(dtype, dim, items_per_thread)
    block_load = coop.block.load.create(dtype, dim, items_per_thread)

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        block_load(d_in, thread_data)
        coop.block.store(d_out, thread_data, items_per_thread)

    h_input = np.random.randint(0, 42, dim * items_per_thread, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    k = kernel[1, dim]
    k(d_input, d_output, items_per_thread)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_block_load_store_single_phase2():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        temp_storage = coop.TempStorage()
        block_load = coop.BlockLoad(temp_storage)
        block_load(d_in, thread_data, items_per_thread)
        block_store = coop.BlockStore(temp_storage)
        block_store(d_out, thread_data, items_per_thread)

    threads_per_block = 128
    items_per_thread = 4
    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    k = kernel[1, threads_per_block]
    k(d_input, d_output, items_per_thread)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_block_load_store_single_phase_temp_storage():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.ThreadData(items_per_thread)
        temp_storage = coop.TempStorage()
        coop.block.load[temp_storage](d_in, thread_data)
        coop.block.store[temp_storage](d_out, thread_data)

    threads_per_block = 128
    items_per_thread = 4
    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    k = kernel[1, threads_per_block]
    k(d_input, d_output, items_per_thread)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)
