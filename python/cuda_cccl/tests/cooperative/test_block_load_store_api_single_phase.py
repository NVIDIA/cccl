# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass

import numba
import numpy as np
from numba import cuda

import cuda.cccl.cooperative.experimental as coop

# coop._init_extension()

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_block_load_store_single_phase():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(d_in, thread_data, items_per_thread)
        coop.block.store(d_out, thread_data, items_per_thread)

    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4
    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    k = kernel[1, threads_per_block]
    k(d_input, d_output, items_per_thread)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_block_load_store_single_phase_num_valid_items():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        # thread_offset = cuda.threadIdx.x * items_per_thread

        # Allocate local memory per thread
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)

        # This loop allows handling arrays larger than the grid size
        while block_offset < num_total_items:
            # Calculate num_valid_items for the current block load/store
            if block_offset + items_per_block <= num_total_items:
                num_valid_items = items_per_block
            else:
                num_valid_items = num_total_items - block_offset

            if num_valid_items == items_per_block:
                coop.block.load(
                    d_in[block_offset:],
                    thread_data,
                    items_per_thread=items_per_thread,
                    algorithm=coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
                )

                coop.block.store(
                    d_out[block_offset:],
                    thread_data,
                    items_per_thread=items_per_thread,
                    algorithm=coop.BlockStoreAlgorithm.DIRECT,
                )

            else:
                coop.block.load(
                    d_in[block_offset:],
                    thread_data,
                    items_per_thread=items_per_thread,
                    algorithm=coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
                    num_valid_items=num_valid_items,
                )

                coop.block.store(
                    d_out[block_offset:],
                    thread_data,
                    items_per_thread=items_per_thread,
                    algorithm=coop.BlockStoreAlgorithm.DIRECT,
                    num_valid_items=num_valid_items,
                )

            # Move to next data block
            block_offset += items_per_block * cuda.gridDim.x

    dtype = np.int32
    threads_per_block = 128
    num_total_items = 1000
    items_per_thread = 4

    h_input = np.random.randint(0, 42, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    # Calculate number of blocks required
    threads_per_block = 128
    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    # Launch kernel
    kernel[blocks_per_grid, threads_per_block](
        d_input,
        d_output,
        items_per_thread,
        num_total_items,
    )

    h_output = d_output.copy_to_host()

    np.testing.assert_array_equal(h_output, h_input)


def test_block_load_store_two_phase():
    dtype = np.int32
    dim = 128
    items_per_thread = 16

    block_load = coop.block.load(dtype, dim, items_per_thread)

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        block_load(d_in, thread_data)
        coop.block.store(d_out, thread_data, items_per_thread)

    h_input = np.random.randint(0, 42, dim * items_per_thread, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    k = kernel[1, dim]
    k(d_input, d_output, items_per_thread)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


_global_test_dtype = np.int32
_global_test_dim = 128
_global_test_items_per_thread = 16
_global_test_block_load = coop.block.load(
    _global_test_dtype,
    _global_test_dim,
    _global_test_items_per_thread,
)


def test_block_load_store_two_phase_global():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        _global_test_block_load(d_in, thread_data)
        coop.block.store(d_out, thread_data, items_per_thread)

    dtype = _global_test_dtype
    dim = _global_test_dim
    items_per_thread = _global_test_items_per_thread

    h_input = np.random.randint(0, 42, dim * items_per_thread, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    k = kernel[1, dim]
    k(d_input, d_output, items_per_thread)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_block_load_store_two_phase_camel_constructor():
    dtype = np.int32
    dim = 128
    items_per_thread = 16

    block_load = coop.BlockLoad(dtype, dim, items_per_thread)

    @cuda.jit(link=block_load.files)
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


def test_block_load_store_two_phase_kernel_param():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, block_load):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        block_load(d_in, thread_data)
        coop.block.store(d_out, thread_data, items_per_thread)

    dtype = np.int32
    dim = 128
    items_per_thread = 16

    block_load = coop.block.load(dtype, dim, items_per_thread)

    h_input = np.random.randint(0, 42, dim * items_per_thread, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    k = kernel[1, dim]
    k(d_input, d_output, items_per_thread, block_load)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


# Crashes with:
#  ... Call to cuLaunchKernel results in CUDA_ERROR_INVALID_VALUE
def disabled_test_block_load_store_two_phase_gpu_dataclass():
    # XXX: this only seems to pass when *debugged* in VS Code/Cursor.  Running
    # it any other way results in a seg fault.
    dtype = np.int32
    dim = 128
    items_per_thread = 16

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, kp):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        kp.block_load(d_in, thread_data)
        coop.block.store(d_out, thread_data, items_per_thread)

    def make_kernel_params(dtype, dim, items_per_thread):
        block_load = coop.block.load(dtype, dim, items_per_thread)
        block_store = coop.block.store(dtype, dim, items_per_thread)

        @dataclass
        class KernelParams:
            items_per_thread: int
            block_load: coop.block.load
            block_store: coop.block.store

        kp = KernelParams(
            items_per_thread=items_per_thread,
            block_load=block_load,
            block_store=block_store,
        )

        kp = coop.gpu_dataclass(kp)
        return kp

    h_input = np.random.randint(0, 42, dim * items_per_thread, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    k = kernel[1, dim]
    kp = make_kernel_params(dtype, dim, items_per_thread)

    assert kp.temp_storage_bytes_max == 1
    assert kp.temp_storage_bytes_sum == 2
    assert kp.temp_storage_alignment == 1
    assert kp.items_per_thread == items_per_thread

    k(d_input, d_output, items_per_thread, kp)
    h_output = d_output.copy_to_host()

    np.testing.assert_array_equal(h_output, h_input)
