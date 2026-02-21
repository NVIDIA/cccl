# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
from dataclasses import dataclass

import numba
import numpy as np
import pytest
from helpers import random_int
from numba import cuda

from cuda import coop
from cuda.coop import BlockLoadAlgorithm, BlockStoreAlgorithm

# example-end imports

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_block_load_store():
    # example-begin load_store
    threads_per_block = 32
    items_per_thread = 4
    block_load = coop.block.make_load(
        numba.int32, threads_per_block, items_per_thread, "striped"
    )
    block_store = coop.block.make_store(
        numba.int32, threads_per_block, items_per_thread, "striped"
    )

    @cuda.jit
    def kernel(input, output):
        tmp = cuda.local.array(items_per_thread, numba.int32)
        block_load(input, tmp)
        block_store(output, tmp)

    # example-end load_store

    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_block_load_store_single_phase_implicit_temp_storage():
    # example-begin load_store_single_phase_implicit_temp_storage_kernel
    import cuda.coop as coop

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(d_in, thread_data, items_per_thread)
        coop.block.store(d_out, thread_data, items_per_thread)

    # example-end load_store_single_phase_implicit_temp_storage_kernel

    # example-begin load_store_single_phase_implicit_temp_storage_usage
    threads_per_block = 128
    items_per_thread = 4
    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    kernel[1, threads_per_block](d_input, d_output, items_per_thread)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)
    # example-end load_store_single_phase_implicit_temp_storage_usage


def test_block_load_store_single_phase():
    # example-begin load-store-single-phase
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(d_in, thread_data, items_per_thread)
        coop.block.store(d_out, thread_data, items_per_thread)

    # example-end load-store-single-phase

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


def test_block_load_store_single_phase_thread_data():
    # example-begin load-store-thread-data
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(d_in, thread_data)
        coop.block.store(d_out, thread_data)

    # example-end load-store-thread-data

    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4
    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    kernel[1, threads_per_block](d_input, d_output, items_per_thread)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_thread_data_dtype_mismatch_raises():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(d_in, thread_data)
        coop.block.store(d_out, thread_data)

    threads_per_block = 128
    items_per_thread = 4
    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(h_input.size, dtype=np.float32)

    with pytest.raises(Exception, match="consistent dtype for ThreadData"):
        kernel[1, threads_per_block](d_input, d_output, items_per_thread)


def test_block_load_store_single_phase_thread_data_temp_storage():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    block_load = coop.block.load(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=BlockLoadAlgorithm.TRANSPOSE,
    )
    block_store = coop.block.store(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=BlockStoreAlgorithm.TRANSPOSE,
    )
    temp_storage_bytes = max(
        block_load.temp_storage_bytes,
        block_store.temp_storage_bytes,
    )
    temp_storage_alignment = max(
        block_load.temp_storage_alignment,
        block_store.temp_storage_alignment,
    )

    # example-begin load-store-thread-data-temp-storage
    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(
            d_in,
            thread_data,
            algorithm=BlockLoadAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )
        coop.block.store(
            d_out,
            thread_data,
            algorithm=BlockStoreAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )

    # example-end load-store-thread-data-temp-storage

    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_block_load_store_single_phase_thread_data_temp_storage_infer_from_omitted_size_alignment():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage()
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(
            d_in,
            thread_data,
            algorithm=BlockLoadAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )
        coop.block.store(
            d_out,
            thread_data,
            algorithm=BlockStoreAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )

    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_temp_storage_shared_partial_inference_size_only():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    block_load = coop.block.load(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=BlockLoadAlgorithm.TRANSPOSE,
    )
    block_store = coop.block.store(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=BlockStoreAlgorithm.TRANSPOSE,
    )
    temp_storage_bytes = max(
        block_load.temp_storage_bytes,
        block_store.temp_storage_bytes,
    )

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
        )
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(
            d_in,
            thread_data,
            algorithm=BlockLoadAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )
        coop.block.store(
            d_out,
            thread_data,
            algorithm=BlockStoreAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )

    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()
    np.testing.assert_allclose(h_output, h_input)


def test_temp_storage_shared_partial_inference_alignment_only():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    block_load = coop.block.load(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=BlockLoadAlgorithm.TRANSPOSE,
    )
    block_store = coop.block.store(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=BlockStoreAlgorithm.TRANSPOSE,
    )
    temp_storage_alignment = max(
        block_load.temp_storage_alignment,
        block_store.temp_storage_alignment,
    )

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            alignment=temp_storage_alignment,
        )
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(
            d_in,
            thread_data,
            algorithm=BlockLoadAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )
        coop.block.store(
            d_out,
            thread_data,
            algorithm=BlockStoreAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )

    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()
    np.testing.assert_allclose(h_output, h_input)


def test_temp_storage_shared_rejects_explicit_size_too_small():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(1)
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(
            d_in,
            thread_data,
            algorithm=BlockLoadAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )
        coop.block.store(
            d_out,
            thread_data,
            algorithm=BlockStoreAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )

    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    with pytest.raises(Exception, match="size_in_bytes is smaller than required"):
        kernel[1, threads_per_block](d_input, d_output)


def test_temp_storage_shared_rejects_explicit_alignment_too_small():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(8192, alignment=1)
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(
            d_in,
            thread_data,
            algorithm=BlockLoadAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )
        coop.block.store(
            d_out,
            thread_data,
            algorithm=BlockStoreAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )

    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    with pytest.raises(Exception, match="alignment is smaller than required"):
        kernel[1, threads_per_block](d_input, d_output)


def test_temp_storage_shared_manual_sync():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(auto_sync=False)
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(
            d_in,
            thread_data,
            algorithm=BlockLoadAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )
        cuda.syncthreads()
        coop.block.store(
            d_out,
            thread_data,
            algorithm=BlockStoreAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )

    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()
    np.testing.assert_allclose(h_output, h_input)


def test_temp_storage_exclusive_infer():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(sharing="exclusive")
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(
            d_in,
            thread_data,
            algorithm=BlockLoadAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )
        coop.block.store(
            d_out,
            thread_data,
            algorithm=BlockStoreAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )

    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()
    np.testing.assert_allclose(h_output, h_input)


def test_temp_storage_exclusive_rejects_auto_sync_true():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(sharing="exclusive", auto_sync=True)
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(
            d_in,
            thread_data,
            algorithm=BlockLoadAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )
        coop.block.store(
            d_out,
            thread_data,
            algorithm=BlockStoreAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )

    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    with pytest.raises(Exception, match="sharing='exclusive'.*auto_sync=True"):
        kernel[1, threads_per_block](d_input, d_output)


def test_temp_storage_rejects_invalid_sharing():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(sharing="bogus")
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(
            d_in,
            thread_data,
            algorithm=BlockLoadAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )
        coop.block.store(
            d_out,
            thread_data,
            algorithm=BlockStoreAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )

    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    with pytest.raises(Exception, match="sharing must be 'shared' or 'exclusive'"):
        kernel[1, threads_per_block](d_input, d_output)


def test_block_load_store_single_phase_num_valid_items():
    # example-begin load-store-num-valid-items
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
                    algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
                )

                coop.block.store(
                    d_out[block_offset:],
                    thread_data,
                    items_per_thread=items_per_thread,
                    algorithm=BlockStoreAlgorithm.DIRECT,
                )

            else:
                coop.block.load(
                    d_in[block_offset:],
                    thread_data,
                    items_per_thread=items_per_thread,
                    algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
                    num_valid_items=num_valid_items,
                )

                coop.block.store(
                    d_out[block_offset:],
                    thread_data,
                    items_per_thread=items_per_thread,
                    algorithm=BlockStoreAlgorithm.DIRECT,
                    num_valid_items=num_valid_items,
                )

            # Move to next data block
            block_offset += items_per_block * cuda.gridDim.x

    # example-end load-store-num-valid-items

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

    # example-begin load-store-two-phase
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        block_load(d_in, thread_data)
        coop.block.store(d_out, thread_data, items_per_thread)

    # example-end load-store-two-phase

    h_input = np.random.randint(0, 42, dim * items_per_thread, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    k = kernel[1, dim]
    k(d_input, d_output, items_per_thread)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)


def test_block_load_single_phase_oob_default():
    dtype = np.int32
    threads_per_block = 64
    items_per_thread = 4
    total_items = threads_per_block * items_per_thread
    num_valid = total_items - 3
    oob_default = np.int32(-123)

    # example-begin load-single-phase-oob-default
    @cuda.jit
    def kernel(d_in, d_out, num_valid_items):
        thread_data = cuda.local.array(items_per_thread, dtype=numba.int32)
        coop.block.load(
            d_in,
            thread_data,
            items_per_thread=items_per_thread,
            num_valid_items=num_valid_items,
            oob_default=oob_default,
        )
        coop.block.store(d_out, thread_data, items_per_thread=items_per_thread)

    # example-end load-single-phase-oob-default

    h_input = np.random.randint(0, 42, total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=dtype)

    kernel[1, threads_per_block](d_input, d_output, num_valid)
    h_output = d_output.copy_to_host()

    expected = np.full(total_items, oob_default, dtype=dtype)
    expected[:num_valid] = h_input[:num_valid]
    np.testing.assert_array_equal(h_output, expected)


def test_block_load_two_phase_oob_default():
    dtype = np.int32
    threads_per_block = 64
    items_per_thread = 4
    total_items = threads_per_block * items_per_thread
    num_valid = total_items - 5
    oob_default = np.int32(-7)

    block_load = coop.block.load(dtype, threads_per_block, items_per_thread)

    # example-begin load-two-phase-oob-default
    @cuda.jit
    def kernel(d_in, d_out, num_valid_items):
        thread_data = cuda.local.array(items_per_thread, dtype=numba.int32)
        block_load(
            d_in,
            thread_data,
            num_valid_items=num_valid_items,
            oob_default=oob_default,
        )
        coop.block.store(d_out, thread_data, items_per_thread=items_per_thread)

    # example-end load-two-phase-oob-default

    h_input = np.random.randint(0, 42, total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=dtype)

    kernel[1, threads_per_block](d_input, d_output, num_valid)
    h_output = d_output.copy_to_host()

    expected = np.full(total_items, oob_default, dtype=dtype)
    expected[:num_valid] = h_input[:num_valid]
    np.testing.assert_array_equal(h_output, expected)


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


def test_block_load_store_two_phase_constructor():
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


def test_block_load_store_two_phase_gpu_dataclass():
    dtype = np.int32
    dim = 128
    items_per_thread = 16

    @cuda.jit
    def kernel(d_in, d_out, kp):
        items_per_thread = kp.items_per_thread
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        kp.block_load(d_in, thread_data)
        kp.block_store(d_out, thread_data)

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

    k(d_input, d_output, kp)
    h_output = d_output.copy_to_host()

    np.testing.assert_array_equal(h_output, h_input)


def _run_load_store_kernel(load_algo, store_algo, items_per_thread, offset=0):
    threads_per_block = 128
    total_items = threads_per_block * items_per_thread

    h_input = random_int(total_items + offset + 1, np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    @cuda.jit
    def kernel(d_in, d_out):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(
            d_in,
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=load_algo,
        )
        coop.block.store(
            d_out,
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=store_algo,
        )

    kernel[1, threads_per_block](d_input[offset:], d_output[offset:])
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(
        h_output[offset : offset + total_items],
        h_input[offset : offset + total_items],
    )


@pytest.mark.parametrize(
    "load_algo",
    [
        BlockLoadAlgorithm.TRANSPOSE,
        BlockLoadAlgorithm.WARP_TRANSPOSE,
        BlockLoadAlgorithm.WARP_TRANSPOSE_TIMESLICED,
    ],
)
def test_block_load_shared_memory_algorithms(load_algo):
    _run_load_store_kernel(load_algo, BlockStoreAlgorithm.DIRECT, 4)


@pytest.mark.parametrize(
    "store_algo",
    [
        BlockStoreAlgorithm.TRANSPOSE,
        BlockStoreAlgorithm.WARP_TRANSPOSE,
        BlockStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
    ],
)
def test_block_store_shared_memory_algorithms(store_algo):
    _run_load_store_kernel(BlockLoadAlgorithm.DIRECT, store_algo, 4)


@pytest.mark.parametrize("offset", [0, 1])
def test_block_load_vectorize_alignment(offset):
    _run_load_store_kernel(
        BlockLoadAlgorithm.VECTORIZE,
        BlockStoreAlgorithm.DIRECT,
        4,
        offset=offset,
    )


@pytest.mark.parametrize("offset", [0, 1])
def test_block_store_vectorize_alignment(offset):
    _run_load_store_kernel(
        BlockLoadAlgorithm.DIRECT,
        BlockStoreAlgorithm.VECTORIZE,
        4,
        offset=offset,
    )


def test_block_load_vectorize_odd_items_per_thread():
    _run_load_store_kernel(
        BlockLoadAlgorithm.VECTORIZE,
        BlockStoreAlgorithm.DIRECT,
        3,
    )


def test_block_store_vectorize_odd_items_per_thread():
    _run_load_store_kernel(
        BlockLoadAlgorithm.DIRECT,
        BlockStoreAlgorithm.VECTORIZE,
        3,
    )
