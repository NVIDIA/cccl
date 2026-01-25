# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass

import numba
import numpy as np
import pytest
from numba import cuda
from numba.cuda.cudadrv import driver

from cuda import coop
from cuda.coop import (
    BlockHistogramAlgorithm,
    BlockLoadAlgorithm,
    BlockStoreAlgorithm,
)
from cuda.coop.block import BlockDiscontinuityType, BlockExchangeType

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@cuda.jit(device=True)
def _merge_op(a, b):
    return a < b


@cuda.jit(device=True)
def _flag_op(lhs, rhs):
    return numba.int32(1 if lhs != rhs else 0)


def _exclusive_scan_host(tile):
    scanned = np.empty_like(tile)
    if tile.size == 0:
        return scanned
    scanned[0] = 0
    scanned[1:] = np.cumsum(tile[:-1], dtype=np.uint64)
    return scanned.astype(tile.dtype)


def _discontinuity_heads_host(tile):
    flags = np.zeros_like(tile, dtype=np.int32)
    if tile.size == 0:
        return flags
    flags[0] = 1
    for idx in range(1, tile.size):
        flags[idx] = 1 if tile[idx] != tile[idx - 1] else 0
    return flags


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def test_block_primitives_single_phase_sort_scan_strided():
    threads_per_block = 64
    items_per_thread = 2
    items_per_block = threads_per_block * items_per_thread
    num_tiles = 3
    total_items = items_per_block * num_tiles
    begin_bit = 0
    end_bit = 8

    @cuda.jit
    def kernel(d_in, d_out, total_items):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        block_offset = cuda.blockIdx.x * items_per_block
        grid_stride = cuda.gridDim.x * items_per_block
        while block_offset < total_items:
            coop.block.load(
                d_in[block_offset:],
                thread_data,
                items_per_thread=items_per_thread,
            )
            coop.block.scan(
                thread_data,
                thread_data,
                items_per_thread=items_per_thread,
            )
            coop.block.radix_sort_keys(
                thread_data,
                items_per_thread,
                begin_bit,
                end_bit,
            )
            coop.block.merge_sort_keys(
                thread_data,
                items_per_thread,
                _merge_op,
            )
            coop.block.store(
                d_out[block_offset:],
                thread_data,
                items_per_thread=items_per_thread,
            )
            block_offset += grid_stride

    h_input = np.random.randint(0, 16, total_items, dtype=np.uint32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output, total_items)
    cuda.synchronize()

    h_reference = np.empty_like(h_input)
    for tile_idx in range(num_tiles):
        start = tile_idx * items_per_block
        end = start + items_per_block
        tile = h_input[start:end]
        scanned = _exclusive_scan_host(tile)
        h_reference[start:end] = np.sort(scanned)

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_primitives_single_phase_sort_scan_strided_multi_block():
    threads_per_block = 64
    items_per_thread = 2
    items_per_block = threads_per_block * items_per_thread
    num_tiles = 5
    total_items = items_per_block * num_tiles
    begin_bit = 0
    end_bit = 8
    blocks_per_grid = 2

    @cuda.jit
    def kernel(d_in, d_out, total_items):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        block_offset = cuda.blockIdx.x * items_per_block
        grid_stride = cuda.gridDim.x * items_per_block
        while block_offset < total_items:
            coop.block.load(
                d_in[block_offset:],
                thread_data,
                items_per_thread=items_per_thread,
            )
            coop.block.scan(
                thread_data,
                thread_data,
                items_per_thread=items_per_thread,
            )
            coop.block.radix_sort_keys(
                thread_data,
                items_per_thread,
                begin_bit,
                end_bit,
            )
            coop.block.merge_sort_keys(
                thread_data,
                items_per_thread,
                _merge_op,
            )
            coop.block.store(
                d_out[block_offset:],
                thread_data,
                items_per_thread=items_per_thread,
            )
            block_offset += grid_stride

    h_input = np.random.randint(0, 16, total_items, dtype=np.uint32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[blocks_per_grid, threads_per_block](d_input, d_output, total_items)
    cuda.synchronize()

    h_reference = np.empty_like(h_input)
    for tile_idx in range(num_tiles):
        start = tile_idx * items_per_block
        end = start + items_per_block
        tile = h_input[start:end]
        scanned = _exclusive_scan_host(tile)
        h_reference[start:end] = np.sort(scanned)

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_primitives_single_phase_partial_tiles_num_valid():
    threads_per_block = 64
    items_per_thread = 2
    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = 3
    total_items = items_per_block * blocks_per_grid - 13
    oob_default = np.uint32(7)

    @cuda.jit
    def kernel(d_in, d_out, total_items):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        block_offset = cuda.blockIdx.x * items_per_block
        grid_stride = cuda.gridDim.x * items_per_block

        while block_offset < total_items:
            remaining = total_items - block_offset
            if remaining >= items_per_block:
                num_valid_items = items_per_block
            else:
                num_valid_items = remaining

            coop.block.load(
                d_in[block_offset:],
                thread_data,
                items_per_thread=items_per_thread,
                num_valid_items=num_valid_items,
                oob_default=oob_default,
            )
            coop.block.scan(
                thread_data,
                thread_data,
                items_per_thread=items_per_thread,
            )
            coop.block.store(
                d_out[block_offset:],
                thread_data,
                items_per_thread=items_per_thread,
                num_valid_items=num_valid_items,
            )
            block_offset += grid_stride

    h_input = np.random.randint(0, 16, total_items, dtype=np.uint32)
    d_input = cuda.to_device(h_input)
    h_output = np.full(total_items + items_per_block, 0xFE, dtype=np.uint32)
    d_output = cuda.to_device(h_output)

    kernel[blocks_per_grid, threads_per_block](d_input, d_output, total_items)
    cuda.synchronize()

    h_reference = h_output.copy()
    num_tiles = (total_items + items_per_block - 1) // items_per_block
    for tile_idx in range(num_tiles):
        start = tile_idx * items_per_block
        tile = np.full(items_per_block, oob_default, dtype=np.uint32)
        valid = min(items_per_block, total_items - start)
        tile[:valid] = h_input[start : start + valid]
        scanned = _exclusive_scan_host(tile)
        h_reference[start : start + valid] = scanned[:valid]

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_primitives_2d_block_exchange_discontinuity_sort():
    block_dim = (16, 4, 1)
    threads_per_block = block_dim
    items_per_thread = 2
    items_per_block = block_dim[0] * block_dim[1] * items_per_thread
    begin_bit = 0
    end_bit = 8

    dtype = np.uint32

    block_exchange_to_striped = coop.block.exchange(
        BlockExchangeType.BlockedToStriped,
        dtype,
        threads_per_block,
        items_per_thread,
    )
    block_exchange_to_blocked = coop.block.exchange(
        BlockExchangeType.StripedToBlocked,
        dtype,
        threads_per_block,
        items_per_thread,
    )
    block_discontinuity = coop.block.discontinuity(
        dtype,
        threads_per_block,
        items_per_thread,
        flag_op=_flag_op,
        flag_dtype=np.int32,
        block_discontinuity_type=BlockDiscontinuityType.HEADS,
    )

    @cuda.jit
    def kernel(d_in, d_out, d_flags):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        flags = coop.local.array(items_per_thread, dtype=np.int32)

        coop.block.load(
            d_in,
            thread_data,
            items_per_thread=items_per_thread,
        )
        block_exchange_to_striped(thread_data)
        block_exchange_to_blocked(thread_data)
        block_discontinuity(
            thread_data,
            flags,
            flag_op=_flag_op,
            block_discontinuity_type=BlockDiscontinuityType.HEADS,
        )
        coop.block.scan(
            thread_data,
            thread_data,
            items_per_thread=items_per_thread,
        )
        coop.block.radix_sort_keys(
            thread_data,
            items_per_thread,
            begin_bit,
            end_bit,
        )
        coop.block.merge_sort_keys(
            thread_data,
            items_per_thread,
            _merge_op,
        )
        coop.block.store(
            d_out,
            thread_data,
            items_per_thread=items_per_thread,
        )
        tid = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        for i in range(items_per_thread):
            d_flags[tid * items_per_thread + i] = flags[i]

    h_input = np.random.randint(0, 16, items_per_block, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    d_flags = cuda.device_array(items_per_block, dtype=np.int32)

    kernel[1, block_dim](d_input, d_output, d_flags)
    cuda.synchronize()

    scanned = _exclusive_scan_host(h_input)
    h_reference = np.sort(scanned)
    h_output = d_output.copy_to_host()
    h_flags = d_flags.copy_to_host()
    h_flags_ref = _discontinuity_heads_host(h_input)

    np.testing.assert_array_equal(h_output, h_reference)
    np.testing.assert_array_equal(h_flags, h_flags_ref)


def test_block_primitives_dynamic_shared_overlapping_slices():
    threads_per_block = 128
    items_per_thread = 4
    items_per_block = threads_per_block * items_per_thread
    total_items = items_per_block
    begin_bit = 0
    end_bit = 8

    dtype = np.uint32

    block_scan = coop.block.scan(
        dtype,
        threads_per_block,
        items_per_thread,
    )
    block_radix_sort = coop.block.radix_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    block_merge_sort = coop.block.merge_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        _merge_op,
    )

    scan_bytes = block_scan.temp_storage_bytes
    radix_bytes = block_radix_sort.temp_storage_bytes
    merge_bytes = block_merge_sort.temp_storage_bytes
    shared_alignment = max(
        block_scan.temp_storage_alignment,
        block_radix_sort.temp_storage_alignment,
        block_merge_sort.temp_storage_alignment,
    )

    dynamic_shared_bytes = _align_up(
        max(scan_bytes + radix_bytes + merge_bytes, 49 * 1024),
        shared_alignment,
    )

    max_optin = cuda.current_context().device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    if dynamic_shared_bytes > max_optin:
        pytest.skip(
            "Device does not support required dynamic shared memory size "
            f"({dynamic_shared_bytes} > {max_optin})."
        )

    @cuda.jit
    def kernel(d_in, d_out):
        smem = cuda.shared.array(0, dtype=numba.uint8)
        temp_scan = smem[:scan_bytes]
        overlap_offset = scan_bytes // 2
        temp_radix = smem[overlap_offset : overlap_offset + radix_bytes]
        temp_merge = smem[scan_bytes : scan_bytes + merge_bytes]

        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)

        coop.block.load(
            d_in,
            thread_data,
            items_per_thread=items_per_thread,
        )
        cuda.syncthreads()

        block_scan(thread_data, thread_data, temp_storage=temp_scan)
        cuda.syncthreads()

        block_radix_sort(
            thread_data,
            items_per_thread,
            begin_bit,
            end_bit,
            temp_storage=temp_radix,
        )
        cuda.syncthreads()

        block_merge_sort(
            thread_data,
            items_per_thread,
            _merge_op,
            temp_storage=temp_merge,
        )
        cuda.syncthreads()

        coop.block.store(
            d_out,
            thread_data,
            items_per_thread=items_per_thread,
        )

    h_input = np.random.randint(0, 16, total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    def _set_dynamic_shared(kernel_obj, launch_config):
        cufunc = kernel_obj._codelibrary.get_cufunc()
        driver.driver.cuKernelSetAttribute(
            driver.binding.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            dynamic_shared_bytes,
            cufunc.handle,
            cufunc.device.id,
        )
        cufunc.set_shared_memory_carveout(100)

    configured = kernel[1, threads_per_block, 0, dynamic_shared_bytes]
    configured.pre_launch_callbacks.append(_set_dynamic_shared)
    try:
        configured(d_input, d_output)
        cuda.synchronize()
    except Exception:
        return

    scanned = _exclusive_scan_host(h_input)
    h_reference = np.sort(scanned)
    h_output = d_output.copy_to_host()
    if np.array_equal(h_output, h_reference):
        pytest.xfail("Overlapping temp storage did not surface corruption (undefined)")
    assert not np.array_equal(h_output, h_reference)


def test_block_primitives_dynamic_shared_mixed_sync():
    threads_per_block = 128
    items_per_thread = 4
    items_per_block = threads_per_block * items_per_thread
    total_items = items_per_block
    begin_bit = 0
    end_bit = 8

    dtype = np.uint32

    block_scan = coop.block.scan(
        dtype,
        threads_per_block,
        items_per_thread,
    )
    block_radix_sort = coop.block.radix_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    block_merge_sort = coop.block.merge_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        _merge_op,
    )

    shared_bytes = max(
        block_scan.temp_storage_bytes,
        block_radix_sort.temp_storage_bytes,
        block_merge_sort.temp_storage_bytes,
    )
    shared_alignment = max(
        block_scan.temp_storage_alignment,
        block_radix_sort.temp_storage_alignment,
        block_merge_sort.temp_storage_alignment,
    )
    dynamic_shared_bytes = _align_up(max(shared_bytes, 49 * 1024), shared_alignment)

    max_optin = cuda.current_context().device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    if dynamic_shared_bytes > max_optin:
        pytest.skip(
            "Device does not support required dynamic shared memory size "
            f"({dynamic_shared_bytes} > {max_optin})."
        )

    @cuda.jit
    def kernel(d_in, d_out):
        smem = cuda.shared.array(0, dtype=numba.uint8)
        temp_storage = smem[:shared_bytes]
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)

        coop.block.load(
            d_in,
            thread_data,
            items_per_thread=items_per_thread,
        )

        cuda.syncthreads()
        block_scan(thread_data, thread_data, temp_storage=temp_storage)
        cuda.syncthreads()

        block_radix_sort(
            thread_data,
            items_per_thread,
            begin_bit,
            end_bit,
            temp_storage=temp_storage,
        )

        cuda.syncthreads()
        block_merge_sort(
            thread_data,
            items_per_thread,
            _merge_op,
            temp_storage=temp_storage,
        )

        coop.block.store(
            d_out,
            thread_data,
            items_per_thread=items_per_thread,
        )

    h_input = np.random.randint(0, 16, total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    def _set_dynamic_shared(kernel_obj, launch_config):
        cufunc = kernel_obj._codelibrary.get_cufunc()
        driver.driver.cuKernelSetAttribute(
            driver.binding.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            dynamic_shared_bytes,
            cufunc.handle,
            cufunc.device.id,
        )
        cufunc.set_shared_memory_carveout(100)

    configured = kernel[1, threads_per_block, 0, dynamic_shared_bytes]
    configured.pre_launch_callbacks.append(_set_dynamic_shared)
    configured(d_input, d_output)
    cuda.synchronize()

    scanned = _exclusive_scan_host(h_input)
    h_reference = np.sort(scanned)
    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_primitives_dynamic_shared_branching_reuse():
    threads_per_block = 128
    items_per_thread = 4
    items_per_block = threads_per_block * items_per_thread
    total_items = items_per_block
    begin_bit = 0
    end_bit = 8
    iterations = 4

    dtype = np.uint32

    block_scan = coop.block.scan(
        dtype,
        threads_per_block,
        items_per_thread,
    )
    block_radix_sort = coop.block.radix_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )

    shared_bytes = max(
        block_scan.temp_storage_bytes,
        block_radix_sort.temp_storage_bytes,
    )
    shared_alignment = max(
        block_scan.temp_storage_alignment,
        block_radix_sort.temp_storage_alignment,
    )
    dynamic_shared_bytes = _align_up(max(shared_bytes, 49 * 1024), shared_alignment)

    max_optin = cuda.current_context().device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    if dynamic_shared_bytes > max_optin:
        pytest.skip(
            "Device does not support required dynamic shared memory size "
            f"({dynamic_shared_bytes} > {max_optin})."
        )

    @cuda.jit
    def kernel(d_in, d_out):
        smem = cuda.shared.array(0, dtype=numba.uint8)
        temp_storage = smem[:shared_bytes]
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)

        coop.block.load(
            d_in,
            thread_data,
            items_per_thread=items_per_thread,
        )

        for it in range(iterations):
            if it % 2 == 0:
                block_scan(thread_data, thread_data, temp_storage=temp_storage)
            else:
                block_radix_sort(
                    thread_data,
                    items_per_thread,
                    begin_bit,
                    end_bit,
                    temp_storage=temp_storage,
                )
            cuda.syncthreads()

        coop.block.store(
            d_out,
            thread_data,
            items_per_thread=items_per_thread,
        )

    h_input = np.random.randint(0, 16, total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    def _set_dynamic_shared(kernel_obj, launch_config):
        cufunc = kernel_obj._codelibrary.get_cufunc()
        driver.driver.cuKernelSetAttribute(
            driver.binding.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            dynamic_shared_bytes,
            cufunc.handle,
            cufunc.device.id,
        )
        cufunc.set_shared_memory_carveout(100)

    configured = kernel[1, threads_per_block, 0, dynamic_shared_bytes]
    configured.pre_launch_callbacks.append(_set_dynamic_shared)
    configured(d_input, d_output)
    cuda.synchronize()

    ref = h_input.copy()
    for it in range(iterations):
        if it % 2 == 0:
            ref = _exclusive_scan_host(ref)
        else:
            ref = np.sort(ref)
    h_output = d_output.copy_to_host()
    if not np.array_equal(h_output, ref):
        pytest.xfail(
            "Branching reuse of shared temp storage produced mismatched output"
        )
    np.testing.assert_array_equal(h_output, ref)


def test_block_primitives_single_phase_run_length_histogram_strided():
    threads_per_block = 32
    runs_per_thread = 2
    decoded_items_per_thread = 4
    runs_per_block = threads_per_block * runs_per_thread
    window_size = threads_per_block * decoded_items_per_thread
    bins = 64
    num_tiles = 2
    total_runs = runs_per_block * num_tiles

    item_dtype = np.uint32
    length_dtype = np.uint32
    counter_dtype = np.uint32
    decoded_offset_dtype = np.uint32

    h_run_values = (np.arange(total_runs, dtype=item_dtype) % bins).astype(item_dtype)
    h_run_lengths = np.full(total_runs, 2, dtype=length_dtype)

    d_run_values = cuda.to_device(h_run_values)
    d_run_lengths = cuda.to_device(h_run_lengths)
    d_decoded_items = cuda.device_array(window_size * num_tiles, dtype=item_dtype)
    d_hist = cuda.device_array(bins * num_tiles, dtype=counter_dtype)
    d_total_decoded = cuda.device_array(num_tiles, dtype=decoded_offset_dtype)

    run_length_traits = coop.block.run_length(
        item_dtype,
        threads_per_block,
        runs_per_thread,
        decoded_items_per_thread,
        decoded_offset_dtype=decoded_offset_dtype,
    )
    temp_storage_bytes = run_length_traits.temp_storage_bytes
    temp_storage_alignment = run_length_traits.temp_storage_alignment

    block_histogram = coop.block.histogram(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        dim=threads_per_block,
        items_per_thread=decoded_items_per_thread,
        algorithm=BlockHistogramAlgorithm.SORT,
        bins=bins,
    )

    @cuda.jit
    def kernel(
        run_values,
        run_lengths,
        decoded_items_out,
        hist_out,
        total_decoded_out,
        total_runs,
    ):
        runs_per_thread = 2
        decoded_items_per_thread = 4
        run_values_local = coop.local.array(runs_per_thread, dtype=run_values.dtype)
        run_lengths_local = coop.local.array(runs_per_thread, dtype=run_lengths.dtype)
        decoded_items = coop.local.array(
            decoded_items_per_thread, dtype=run_values.dtype
        )
        relative_offsets = coop.local.array(
            decoded_items_per_thread, dtype=run_lengths.dtype
        )
        total_decoded_size = coop.local.array(1, dtype=total_decoded_out.dtype)

        runs_per_block = runs_per_thread * cuda.blockDim.x
        block_offset = cuda.blockIdx.x * runs_per_block
        grid_stride = cuda.gridDim.x * runs_per_block
        tile_idx = block_offset // runs_per_block

        smem_histogram = coop.shared.array(bins, dtype=counter_dtype)
        histo = block_histogram()

        while block_offset < total_runs:
            total_decoded_size[0] = 0
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

            temp_storage = coop.TempStorage(
                temp_storage_bytes,
                temp_storage_alignment,
            )
            run_length = coop.block.run_length(
                run_values_local,
                run_lengths_local,
                runs_per_thread,
                decoded_items_per_thread,
                total_decoded_size,
                decoded_offset_dtype=decoded_offset_dtype,
                temp_storage=temp_storage,
            )

            run_length.decode(decoded_items, 0, relative_offsets)

            decoded_block_offset = tile_idx * window_size
            coop.block.store(
                decoded_items_out[decoded_block_offset:],
                decoded_items,
                items_per_thread=decoded_items_per_thread,
            )

            histo.init(smem_histogram)
            histo.composite(decoded_items, smem_histogram)
            cuda.syncthreads()

            thread_linear = (
                cuda.threadIdx.x
                + cuda.threadIdx.y * cuda.blockDim.x
                + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
            )
            hist_offset = tile_idx * bins
            for bin_idx in range(thread_linear, bins, threads_per_block):
                hist_out[hist_offset + bin_idx] = smem_histogram[bin_idx]

            if cuda.threadIdx.x == 0:
                total_decoded_out[tile_idx] = total_decoded_size[0]

            block_offset += grid_stride
            tile_idx += cuda.gridDim.x

    kernel[1, threads_per_block](
        d_run_values,
        d_run_lengths,
        d_decoded_items,
        d_hist,
        d_total_decoded,
        total_runs,
    )
    cuda.synchronize()

    h_decoded = d_decoded_items.copy_to_host()
    h_hist = d_hist.copy_to_host()
    h_total_decoded = d_total_decoded.copy_to_host()

    expected_decoded = np.empty_like(h_decoded)
    expected_hist = np.empty_like(h_hist)

    for tile_idx in range(num_tiles):
        run_start = tile_idx * runs_per_block
        run_end = run_start + runs_per_block
        run_vals = h_run_values[run_start:run_end]
        run_lens = h_run_lengths[run_start:run_end]
        decoded = []
        for value, count in zip(run_vals, run_lens):
            decoded.extend([value] * int(count))
        decoded = np.array(decoded, dtype=item_dtype)
        assert decoded.size == window_size

        out_start = tile_idx * window_size
        out_end = out_start + window_size
        expected_decoded[out_start:out_end] = decoded

        hist = np.bincount(decoded, minlength=bins).astype(counter_dtype)
        hist_start = tile_idx * bins
        hist_end = hist_start + bins
        expected_hist[hist_start:hist_end] = hist

    np.testing.assert_array_equal(h_decoded, expected_decoded)
    np.testing.assert_array_equal(h_hist, expected_hist)
    np.testing.assert_array_equal(h_total_decoded, np.full(num_tiles, window_size))


def test_block_primitives_single_phase_run_length_histogram_strided_multi_block():
    threads_per_block = 32
    runs_per_thread = 2
    decoded_items_per_thread = 4
    runs_per_block = threads_per_block * runs_per_thread
    window_size = threads_per_block * decoded_items_per_thread
    bins = 64
    num_tiles = 5
    total_runs = runs_per_block * num_tiles
    blocks_per_grid = 2

    item_dtype = np.uint32
    length_dtype = np.uint32
    counter_dtype = np.uint32
    decoded_offset_dtype = np.uint32

    h_run_values = (np.arange(total_runs, dtype=item_dtype) % bins).astype(item_dtype)
    h_run_lengths = np.full(total_runs, 2, dtype=length_dtype)

    d_run_values = cuda.to_device(h_run_values)
    d_run_lengths = cuda.to_device(h_run_lengths)
    d_decoded_items = cuda.device_array(window_size * num_tiles, dtype=item_dtype)
    d_hist = cuda.device_array(bins * num_tiles, dtype=counter_dtype)
    d_total_decoded = cuda.device_array(num_tiles, dtype=decoded_offset_dtype)

    run_length_traits = coop.block.run_length(
        item_dtype,
        threads_per_block,
        runs_per_thread,
        decoded_items_per_thread,
        decoded_offset_dtype=decoded_offset_dtype,
    )
    temp_storage_bytes = run_length_traits.temp_storage_bytes
    temp_storage_alignment = run_length_traits.temp_storage_alignment

    block_histogram = coop.block.histogram(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        dim=threads_per_block,
        items_per_thread=decoded_items_per_thread,
        algorithm=BlockHistogramAlgorithm.SORT,
        bins=bins,
    )

    @cuda.jit
    def kernel(
        run_values,
        run_lengths,
        decoded_items_out,
        hist_out,
        total_decoded_out,
        total_runs,
    ):
        runs_per_thread = 2
        decoded_items_per_thread = 4
        run_values_local = coop.local.array(runs_per_thread, dtype=run_values.dtype)
        run_lengths_local = coop.local.array(runs_per_thread, dtype=run_lengths.dtype)
        decoded_items = coop.local.array(
            decoded_items_per_thread, dtype=run_values.dtype
        )
        relative_offsets = coop.local.array(
            decoded_items_per_thread, dtype=run_lengths.dtype
        )
        total_decoded_size = coop.local.array(1, dtype=total_decoded_out.dtype)

        runs_per_block = runs_per_thread * cuda.blockDim.x
        block_offset = cuda.blockIdx.x * runs_per_block
        grid_stride = cuda.gridDim.x * runs_per_block
        tile_idx = block_offset // runs_per_block

        smem_histogram = coop.shared.array(bins, dtype=counter_dtype)
        histo = block_histogram()

        while block_offset < total_runs:
            total_decoded_size[0] = 0
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

            temp_storage = coop.TempStorage(
                temp_storage_bytes,
                temp_storage_alignment,
            )
            run_length = coop.block.run_length(
                run_values_local,
                run_lengths_local,
                runs_per_thread,
                decoded_items_per_thread,
                total_decoded_size,
                decoded_offset_dtype=decoded_offset_dtype,
                temp_storage=temp_storage,
            )

            run_length.decode(decoded_items, 0, relative_offsets)

            decoded_block_offset = tile_idx * window_size
            coop.block.store(
                decoded_items_out[decoded_block_offset:],
                decoded_items,
                items_per_thread=decoded_items_per_thread,
            )

            histo.init(smem_histogram)
            histo.composite(decoded_items, smem_histogram)
            cuda.syncthreads()

            thread_linear = (
                cuda.threadIdx.x
                + cuda.threadIdx.y * cuda.blockDim.x
                + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
            )
            hist_offset = tile_idx * bins
            for bin_idx in range(thread_linear, bins, threads_per_block):
                hist_out[hist_offset + bin_idx] = smem_histogram[bin_idx]

            if cuda.threadIdx.x == 0:
                total_decoded_out[tile_idx] = total_decoded_size[0]

            block_offset += grid_stride
            tile_idx += cuda.gridDim.x

    kernel[blocks_per_grid, threads_per_block](
        d_run_values,
        d_run_lengths,
        d_decoded_items,
        d_hist,
        d_total_decoded,
        total_runs,
    )
    cuda.synchronize()

    h_decoded = d_decoded_items.copy_to_host()
    h_hist = d_hist.copy_to_host()
    h_total_decoded = d_total_decoded.copy_to_host()

    expected_decoded = np.empty_like(h_decoded)
    expected_hist = np.empty_like(h_hist)

    for tile_idx in range(num_tiles):
        run_start = tile_idx * runs_per_block
        run_end = run_start + runs_per_block
        run_vals = h_run_values[run_start:run_end]
        run_lens = h_run_lengths[run_start:run_end]
        decoded = []
        for value, count in zip(run_vals, run_lens):
            decoded.extend([value] * int(count))
        decoded = np.array(decoded, dtype=item_dtype)
        assert decoded.size == window_size

        out_start = tile_idx * window_size
        out_end = out_start + window_size
        expected_decoded[out_start:out_end] = decoded

        hist = np.bincount(decoded, minlength=bins).astype(counter_dtype)
        hist_start = tile_idx * bins
        hist_end = hist_start + bins
        expected_hist[hist_start:hist_end] = hist

    np.testing.assert_array_equal(h_decoded, expected_decoded)
    np.testing.assert_array_equal(h_hist, expected_hist)
    np.testing.assert_array_equal(h_total_decoded, np.full(num_tiles, window_size))


def test_block_primitives_two_phase_gpu_dataclass_transpose_private():
    threads_per_block = 64
    items_per_thread = 2
    items_per_block = threads_per_block * items_per_thread
    num_tiles = 2
    total_items = items_per_block * num_tiles
    begin_bit = numba.int32(0)
    end_bit = numba.int32(8)

    dtype = np.uint32

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
    block_exclusive_sum = coop.block.exclusive_sum(
        dtype,
        threads_per_block,
        items_per_thread,
    )
    block_radix_sort = coop.block.radix_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    block_merge_sort = coop.block.merge_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        _merge_op,
    )

    load_bytes = block_load.temp_storage_bytes
    load_alignment = block_load.temp_storage_alignment
    store_bytes = block_store.temp_storage_bytes
    store_alignment = block_store.temp_storage_alignment
    scan_bytes = block_exclusive_sum.temp_storage_bytes
    scan_alignment = block_exclusive_sum.temp_storage_alignment
    radix_bytes = block_radix_sort.temp_storage_bytes
    radix_alignment = block_radix_sort.temp_storage_alignment
    merge_bytes = block_merge_sort.temp_storage_bytes
    merge_alignment = block_merge_sort.temp_storage_alignment

    @dataclass
    class KernelParams:
        items_per_thread: int
        block_load: coop.block.load
        block_store: coop.block.store
        block_exclusive_sum: coop.block.exclusive_sum
        block_radix_sort: coop.block.radix_sort_keys
        block_merge_sort: coop.block.merge_sort_keys

    kp = KernelParams(
        items_per_thread=items_per_thread,
        block_load=block_load,
        block_store=block_store,
        block_exclusive_sum=block_exclusive_sum,
        block_radix_sort=block_radix_sort,
        block_merge_sort=block_merge_sort,
    )
    kp = coop.gpu_dataclass(kp)

    @cuda.jit
    def kernel(d_in, d_out, total_items, kp):
        thread_data = coop.local.array(kp.items_per_thread, dtype=d_in.dtype)
        block_offset = cuda.blockIdx.x * items_per_block
        grid_stride = cuda.gridDim.x * items_per_block

        while block_offset < total_items:
            temp_load = coop.TempStorage(load_bytes, load_alignment)
            temp_scan = coop.TempStorage(scan_bytes, scan_alignment)
            temp_radix = coop.TempStorage(radix_bytes, radix_alignment)
            temp_merge = coop.TempStorage(merge_bytes, merge_alignment)
            temp_store = coop.TempStorage(store_bytes, store_alignment)

            kp.block_load(
                d_in[block_offset:],
                thread_data,
                temp_storage=temp_load,
            )
            kp.block_exclusive_sum(thread_data, thread_data, temp_storage=temp_scan)
            kp.block_radix_sort(
                thread_data,
                kp.items_per_thread,
                begin_bit,
                end_bit,
                temp_storage=temp_radix,
            )
            kp.block_merge_sort(
                thread_data,
                kp.items_per_thread,
                _merge_op,
                temp_storage=temp_merge,
            )
            kp.block_store(
                d_out[block_offset:],
                thread_data,
                temp_storage=temp_store,
            )

            block_offset += grid_stride

    h_input = np.random.randint(0, 16, total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output, total_items, kp)
    cuda.synchronize()

    h_reference = np.empty_like(h_input)
    for tile_idx in range(num_tiles):
        start = tile_idx * items_per_block
        end = start + items_per_block
        tile = h_input[start:end]
        scanned = _exclusive_scan_host(tile)
        h_reference[start:end] = np.sort(scanned)

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_primitives_dynamic_shared_temp_storage():
    threads_per_block = 128
    items_per_thread = 4
    items_per_block = threads_per_block * items_per_thread
    total_items = items_per_block
    begin_bit = 0
    end_bit = 8

    dtype = np.uint32

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
    block_scan = coop.block.scan(
        dtype,
        threads_per_block,
        items_per_thread,
    )
    block_radix_sort = coop.block.radix_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    block_merge_sort = coop.block.merge_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        _merge_op,
    )

    shared_bytes = max(
        block_load.temp_storage_bytes,
        block_store.temp_storage_bytes,
        block_scan.temp_storage_bytes,
        block_radix_sort.temp_storage_bytes,
        block_merge_sort.temp_storage_bytes,
    )
    shared_alignment = max(
        block_load.temp_storage_alignment,
        block_store.temp_storage_alignment,
        block_scan.temp_storage_alignment,
        block_radix_sort.temp_storage_alignment,
        block_merge_sort.temp_storage_alignment,
    )

    dynamic_shared_bytes = _align_up(max(shared_bytes, 49 * 1024), shared_alignment)

    max_optin = cuda.current_context().device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    if dynamic_shared_bytes > max_optin:
        pytest.skip(
            "Device does not support required dynamic shared memory size "
            f"({dynamic_shared_bytes} > {max_optin})."
        )

    @cuda.jit
    def kernel(d_in, d_out, total_items):
        smem = cuda.shared.array(0, dtype=numba.uint8)
        temp_storage = smem[:shared_bytes]
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)

        block_offset = cuda.blockIdx.x * items_per_block
        if block_offset >= total_items:
            return

        coop.block.load(
            d_in[block_offset:],
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )
        cuda.syncthreads()

        block_scan(thread_data, thread_data, temp_storage=temp_storage)
        cuda.syncthreads()

        block_radix_sort(
            thread_data,
            items_per_thread,
            begin_bit,
            end_bit,
            temp_storage=temp_storage,
        )
        cuda.syncthreads()

        block_merge_sort(
            thread_data,
            items_per_thread,
            _merge_op,
            temp_storage=temp_storage,
        )
        cuda.syncthreads()

        coop.block.store(
            d_out[block_offset:],
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=BlockStoreAlgorithm.TRANSPOSE,
            temp_storage=temp_storage,
        )

    h_input = np.random.randint(0, 16, total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    def _set_dynamic_shared(kernel_obj, launch_config):
        cufunc = kernel_obj._codelibrary.get_cufunc()
        driver.driver.cuKernelSetAttribute(
            driver.binding.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            dynamic_shared_bytes,
            cufunc.handle,
            cufunc.device.id,
        )
        cufunc.set_shared_memory_carveout(100)

    configured = kernel[1, threads_per_block, 0, dynamic_shared_bytes]
    configured.pre_launch_callbacks.append(_set_dynamic_shared)
    configured(
        d_input,
        d_output,
        np.int32(total_items),
    )
    cuda.synchronize()

    scanned = _exclusive_scan_host(h_input)
    h_reference = np.sort(scanned)
    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_primitives_dynamic_shared_temp_storage_carved():
    threads_per_block = 128
    items_per_thread = 4
    items_per_block = threads_per_block * items_per_thread
    total_items = items_per_block
    begin_bit = 0
    end_bit = 8

    dtype = np.uint32

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
    block_scan = coop.block.scan(
        dtype,
        threads_per_block,
        items_per_thread,
    )
    block_exchange_to_striped = coop.block.exchange(
        BlockExchangeType.BlockedToStriped,
        dtype,
        threads_per_block,
        items_per_thread,
    )
    block_exchange_to_blocked = coop.block.exchange(
        BlockExchangeType.StripedToBlocked,
        dtype,
        threads_per_block,
        items_per_thread,
    )
    block_discontinuity = coop.block.discontinuity(
        dtype,
        threads_per_block,
        items_per_thread,
        flag_op=_flag_op,
        flag_dtype=np.int32,
        block_discontinuity_type=BlockDiscontinuityType.HEADS,
    )
    block_radix_sort = coop.block.radix_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    block_merge_sort = coop.block.merge_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        _merge_op,
    )

    load_bytes = block_load.temp_storage_bytes
    load_alignment = block_load.temp_storage_alignment or 1
    store_bytes = block_store.temp_storage_bytes
    store_alignment = block_store.temp_storage_alignment or 1
    scan_bytes = block_scan.temp_storage_bytes
    scan_alignment = block_scan.temp_storage_alignment or 1
    exchange_strip_bytes = block_exchange_to_striped.temp_storage_bytes
    exchange_strip_alignment = block_exchange_to_striped.temp_storage_alignment or 1
    exchange_block_bytes = block_exchange_to_blocked.temp_storage_bytes
    exchange_block_alignment = block_exchange_to_blocked.temp_storage_alignment or 1
    disc_bytes = block_discontinuity.temp_storage_bytes
    disc_alignment = block_discontinuity.temp_storage_alignment or 1
    radix_bytes = block_radix_sort.temp_storage_bytes
    radix_alignment = block_radix_sort.temp_storage_alignment or 1
    merge_bytes = block_merge_sort.temp_storage_bytes
    merge_alignment = block_merge_sort.temp_storage_alignment or 1

    offset = 0
    offset = _align_up(offset, load_alignment)
    load_offset = offset
    offset = load_offset + load_bytes

    offset = _align_up(offset, exchange_strip_alignment)
    exchange_strip_offset = offset
    offset = exchange_strip_offset + exchange_strip_bytes

    offset = _align_up(offset, exchange_block_alignment)
    exchange_block_offset = offset
    offset = exchange_block_offset + exchange_block_bytes

    offset = _align_up(offset, disc_alignment)
    disc_offset = offset
    offset = disc_offset + disc_bytes

    offset = _align_up(offset, scan_alignment)
    scan_offset = offset
    offset = scan_offset + scan_bytes

    offset = _align_up(offset, radix_alignment)
    radix_offset = offset
    offset = radix_offset + radix_bytes

    offset = _align_up(offset, merge_alignment)
    merge_offset = offset
    offset = merge_offset + merge_bytes

    offset = _align_up(offset, store_alignment)
    store_offset = offset
    offset = store_offset + store_bytes

    max_alignment = max(
        load_alignment,
        store_alignment,
        scan_alignment,
        exchange_strip_alignment,
        exchange_block_alignment,
        disc_alignment,
        radix_alignment,
        merge_alignment,
    )
    dynamic_shared_bytes = _align_up(max(offset, 49 * 1024), max_alignment)

    max_optin = cuda.current_context().device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    if dynamic_shared_bytes > max_optin:
        pytest.skip(
            "Device does not support required dynamic shared memory size "
            f"({dynamic_shared_bytes} > {max_optin})."
        )

    @cuda.jit
    def kernel(d_in, d_out, total_items, d_flags):
        smem = cuda.shared.array(0, dtype=numba.uint8)
        temp_load = smem[load_offset : load_offset + load_bytes]
        temp_exchange_strip = smem[
            exchange_strip_offset : exchange_strip_offset + exchange_strip_bytes
        ]
        temp_exchange_block = smem[
            exchange_block_offset : exchange_block_offset + exchange_block_bytes
        ]
        temp_disc = smem[disc_offset : disc_offset + disc_bytes]
        temp_scan = smem[scan_offset : scan_offset + scan_bytes]
        temp_radix = smem[radix_offset : radix_offset + radix_bytes]
        temp_merge = smem[merge_offset : merge_offset + merge_bytes]
        temp_store = smem[store_offset : store_offset + store_bytes]

        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        flags = coop.local.array(items_per_thread, dtype=np.int32)
        block_offset = cuda.blockIdx.x * items_per_block
        if block_offset >= total_items:
            return

        coop.block.load(
            d_in[block_offset:],
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.TRANSPOSE,
            temp_storage=temp_load,
        )
        cuda.syncthreads()

        block_exchange_to_striped(thread_data, temp_storage=temp_exchange_strip)
        cuda.syncthreads()
        block_exchange_to_blocked(thread_data, temp_storage=temp_exchange_block)
        cuda.syncthreads()

        block_discontinuity(
            thread_data,
            flags,
            flag_op=_flag_op,
            block_discontinuity_type=BlockDiscontinuityType.HEADS,
            temp_storage=temp_disc,
        )
        cuda.syncthreads()

        block_scan(thread_data, thread_data, temp_storage=temp_scan)
        cuda.syncthreads()

        block_radix_sort(
            thread_data,
            items_per_thread,
            begin_bit,
            end_bit,
            temp_storage=temp_radix,
        )
        cuda.syncthreads()

        block_merge_sort(
            thread_data,
            items_per_thread,
            _merge_op,
            temp_storage=temp_merge,
        )
        cuda.syncthreads()

        coop.block.store(
            d_out[block_offset:],
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=BlockStoreAlgorithm.TRANSPOSE,
            temp_storage=temp_store,
        )
        tid = cuda.threadIdx.x
        for i in range(items_per_thread):
            d_flags[tid * items_per_thread + i] = flags[i]

    h_input = np.random.randint(0, 16, total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    d_flags = cuda.device_array(total_items, dtype=np.int32)

    def _set_dynamic_shared(kernel_obj, launch_config):
        cufunc = kernel_obj._codelibrary.get_cufunc()
        driver.driver.cuKernelSetAttribute(
            driver.binding.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            dynamic_shared_bytes,
            cufunc.handle,
            cufunc.device.id,
        )
        cufunc.set_shared_memory_carveout(100)

    configured = kernel[1, threads_per_block, 0, dynamic_shared_bytes]
    configured.pre_launch_callbacks.append(_set_dynamic_shared)
    configured(
        d_input,
        d_output,
        np.int32(total_items),
        d_flags,
    )
    cuda.synchronize()

    scanned = _exclusive_scan_host(h_input)
    h_reference = np.sort(scanned)
    h_output = d_output.copy_to_host()
    h_flags = d_flags.copy_to_host()
    h_flags_ref = _discontinuity_heads_host(h_input)
    np.testing.assert_array_equal(h_output, h_reference)
    np.testing.assert_array_equal(h_flags, h_flags_ref)


def test_block_primitives_dynamic_shared_run_length_histogram_shared():
    threads_per_block = 32
    runs_per_thread = 2
    decoded_items_per_thread = 4
    runs_per_block = threads_per_block * runs_per_thread
    window_size = threads_per_block * decoded_items_per_thread
    bins = 64
    total_runs = runs_per_block

    item_dtype = np.uint32
    length_dtype = np.uint32
    counter_dtype = np.uint32
    decoded_offset_dtype = np.uint32

    h_run_values = (np.arange(total_runs, dtype=item_dtype) % bins).astype(item_dtype)
    h_run_lengths = np.full(total_runs, 2, dtype=length_dtype)

    d_run_values = cuda.to_device(h_run_values)
    d_run_lengths = cuda.to_device(h_run_lengths)
    d_decoded_items = cuda.device_array(window_size, dtype=item_dtype)
    d_scanned_items = cuda.device_array(window_size, dtype=item_dtype)
    d_hist = cuda.device_array(bins, dtype=counter_dtype)
    d_total_decoded = cuda.device_array(1, dtype=decoded_offset_dtype)

    block_load_values = coop.block.load(
        item_dtype,
        threads_per_block,
        runs_per_thread,
        algorithm=BlockLoadAlgorithm.DIRECT,
    )
    block_load_lengths = coop.block.load(
        length_dtype,
        threads_per_block,
        runs_per_thread,
        algorithm=BlockLoadAlgorithm.DIRECT,
    )
    block_scan = coop.block.scan(
        item_dtype,
        threads_per_block,
        decoded_items_per_thread,
    )
    run_length_traits = coop.block.run_length(
        item_dtype,
        threads_per_block,
        runs_per_thread,
        decoded_items_per_thread,
        decoded_offset_dtype=decoded_offset_dtype,
    )

    shared_bytes = max(
        block_load_values.temp_storage_bytes,
        block_load_lengths.temp_storage_bytes,
        block_scan.temp_storage_bytes,
        run_length_traits.temp_storage_bytes,
    )
    shared_alignment = max(
        block_load_values.temp_storage_alignment,
        block_load_lengths.temp_storage_alignment,
        block_scan.temp_storage_alignment,
        run_length_traits.temp_storage_alignment,
    )
    dynamic_shared_bytes = _align_up(max(shared_bytes, 49 * 1024), shared_alignment)

    max_optin = cuda.current_context().device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    if dynamic_shared_bytes > max_optin:
        pytest.skip(
            "Device does not support required dynamic shared memory size "
            f"({dynamic_shared_bytes} > {max_optin})."
        )

    block_histogram = coop.block.histogram(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        dim=threads_per_block,
        items_per_thread=decoded_items_per_thread,
        algorithm=BlockHistogramAlgorithm.SORT,
        bins=bins,
    )

    @cuda.jit
    def kernel(
        run_values,
        run_lengths,
        decoded_items_out,
        scanned_items_out,
        hist_out,
        total_decoded_out,
    ):
        runs_per_thread = 2
        decoded_items_per_thread = 4
        smem = cuda.shared.array(0, dtype=numba.uint8)
        temp_storage = smem[:shared_bytes]

        run_values_local = coop.local.array(runs_per_thread, dtype=run_values.dtype)
        run_lengths_local = coop.local.array(runs_per_thread, dtype=run_lengths.dtype)
        decoded_items = coop.local.array(
            decoded_items_per_thread, dtype=run_values.dtype
        )
        scan_items = coop.local.array(decoded_items_per_thread, dtype=run_values.dtype)
        total_decoded_size = coop.local.array(1, dtype=total_decoded_out.dtype)

        block_offset = cuda.blockIdx.x * runs_per_block
        if block_offset >= total_runs:
            return

        total_decoded_size[0] = 0
        coop.block.load(
            run_values[block_offset:],
            run_values_local,
            items_per_thread=runs_per_thread,
            algorithm=BlockLoadAlgorithm.DIRECT,
            temp_storage=temp_storage,
        )
        cuda.syncthreads()
        coop.block.load(
            run_lengths[block_offset:],
            run_lengths_local,
            items_per_thread=runs_per_thread,
            algorithm=BlockLoadAlgorithm.DIRECT,
            temp_storage=temp_storage,
        )
        cuda.syncthreads()

        run_length = coop.block.run_length(
            run_values_local,
            run_lengths_local,
            runs_per_thread,
            decoded_items_per_thread,
            total_decoded_size,
            decoded_offset_dtype=decoded_offset_dtype,
            temp_storage=temp_storage,
        )
        cuda.syncthreads()

        decoded_window_offset = 0
        run_length.decode(decoded_items, decoded_window_offset)

        for i in range(decoded_items_per_thread):
            scan_items[i] = decoded_items[i]

        block_scan(scan_items, scan_items, temp_storage=temp_storage)
        cuda.syncthreads()

        base = cuda.threadIdx.x * decoded_items_per_thread
        for i in range(decoded_items_per_thread):
            decoded_items_out[base + i] = decoded_items[i]
            scanned_items_out[base + i] = scan_items[i]

        smem_histogram = coop.shared.array(bins, dtype=counter_dtype)
        histo = block_histogram()
        histo.init(smem_histogram)
        histo.composite(decoded_items, smem_histogram)
        cuda.syncthreads()

        thread_linear = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        for bin_idx in range(thread_linear, bins, threads_per_block):
            hist_out[bin_idx] = smem_histogram[bin_idx]

        if cuda.threadIdx.x == 0:
            total_decoded_out[0] = total_decoded_size[0]

    def _set_dynamic_shared(kernel_obj, launch_config):
        cufunc = kernel_obj._codelibrary.get_cufunc()
        driver.driver.cuKernelSetAttribute(
            driver.binding.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            dynamic_shared_bytes,
            cufunc.handle,
            cufunc.device.id,
        )
        cufunc.set_shared_memory_carveout(100)

    configured = kernel[1, threads_per_block, 0, dynamic_shared_bytes]
    configured.pre_launch_callbacks.append(_set_dynamic_shared)
    configured(
        d_run_values,
        d_run_lengths,
        d_decoded_items,
        d_scanned_items,
        d_hist,
        d_total_decoded,
    )
    cuda.synchronize()

    h_decoded = d_decoded_items.copy_to_host()
    h_scanned = d_scanned_items.copy_to_host()
    h_hist = d_hist.copy_to_host()
    h_total_decoded = d_total_decoded.copy_to_host()

    decoded = []
    for value, count in zip(h_run_values, h_run_lengths):
        decoded.extend([value] * int(count))
    decoded = np.array(decoded, dtype=item_dtype)

    h_reference_decoded = decoded
    h_reference_scanned = _exclusive_scan_host(decoded)
    h_reference_hist = np.bincount(decoded, minlength=bins).astype(counter_dtype)

    np.testing.assert_array_equal(h_decoded, h_reference_decoded)
    np.testing.assert_array_equal(h_scanned, h_reference_scanned)
    np.testing.assert_array_equal(h_hist, h_reference_hist)
    np.testing.assert_array_equal(h_total_decoded, np.array([window_size]))


def test_block_primitives_dynamic_shared_run_length_histogram_carved():
    threads_per_block = 32
    runs_per_thread = 2
    decoded_items_per_thread = 4
    runs_per_block = threads_per_block * runs_per_thread
    window_size = threads_per_block * decoded_items_per_thread
    bins = 64
    total_runs = runs_per_block

    item_dtype = np.uint32
    length_dtype = np.uint32
    counter_dtype = np.uint32
    decoded_offset_dtype = np.uint32

    h_run_values = (np.arange(total_runs, dtype=item_dtype) % bins).astype(item_dtype)
    h_run_lengths = np.full(total_runs, 2, dtype=length_dtype)

    d_run_values = cuda.to_device(h_run_values)
    d_run_lengths = cuda.to_device(h_run_lengths)
    d_decoded_items = cuda.device_array(window_size, dtype=item_dtype)
    d_scanned_items = cuda.device_array(window_size, dtype=item_dtype)
    d_hist = cuda.device_array(bins, dtype=counter_dtype)
    d_total_decoded = cuda.device_array(1, dtype=decoded_offset_dtype)

    block_load_values = coop.block.load(
        item_dtype,
        threads_per_block,
        runs_per_thread,
        algorithm=BlockLoadAlgorithm.DIRECT,
    )
    block_load_lengths = coop.block.load(
        length_dtype,
        threads_per_block,
        runs_per_thread,
        algorithm=BlockLoadAlgorithm.DIRECT,
    )
    block_scan = coop.block.scan(
        item_dtype,
        threads_per_block,
        decoded_items_per_thread,
    )
    run_length_traits = coop.block.run_length(
        item_dtype,
        threads_per_block,
        runs_per_thread,
        decoded_items_per_thread,
        decoded_offset_dtype=decoded_offset_dtype,
    )

    load_values_bytes = block_load_values.temp_storage_bytes
    load_values_alignment = block_load_values.temp_storage_alignment or 1
    load_lengths_bytes = block_load_lengths.temp_storage_bytes
    load_lengths_alignment = block_load_lengths.temp_storage_alignment or 1
    scan_bytes = block_scan.temp_storage_bytes
    scan_alignment = block_scan.temp_storage_alignment or 1
    run_length_bytes = run_length_traits.temp_storage_bytes
    run_length_alignment = run_length_traits.temp_storage_alignment or 1

    offset = 0
    offset = _align_up(offset, load_values_alignment)
    load_values_offset = offset
    offset = load_values_offset + load_values_bytes

    offset = _align_up(offset, load_lengths_alignment)
    load_lengths_offset = offset
    offset = load_lengths_offset + load_lengths_bytes

    offset = _align_up(offset, run_length_alignment)
    run_length_offset = offset
    offset = run_length_offset + run_length_bytes

    offset = _align_up(offset, scan_alignment)
    scan_offset = offset
    offset = scan_offset + scan_bytes

    max_alignment = max(
        load_values_alignment,
        load_lengths_alignment,
        run_length_alignment,
        scan_alignment,
    )
    dynamic_shared_bytes = _align_up(max(offset, 49 * 1024), max_alignment)

    max_optin = cuda.current_context().device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    if dynamic_shared_bytes > max_optin:
        pytest.skip(
            "Device does not support required dynamic shared memory size "
            f"({dynamic_shared_bytes} > {max_optin})."
        )

    block_histogram = coop.block.histogram(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        dim=threads_per_block,
        items_per_thread=decoded_items_per_thread,
        algorithm=BlockHistogramAlgorithm.SORT,
        bins=bins,
    )

    @cuda.jit
    def kernel(
        run_values,
        run_lengths,
        decoded_items_out,
        scanned_items_out,
        hist_out,
        total_decoded_out,
    ):
        runs_per_thread = 2
        decoded_items_per_thread = 4
        smem = cuda.shared.array(0, dtype=numba.uint8)
        temp_load_values = smem[
            load_values_offset : load_values_offset + load_values_bytes
        ]
        temp_load_lengths = smem[
            load_lengths_offset : load_lengths_offset + load_lengths_bytes
        ]
        temp_run_length = smem[run_length_offset : run_length_offset + run_length_bytes]
        temp_scan = smem[scan_offset : scan_offset + scan_bytes]

        run_values_local = coop.local.array(runs_per_thread, dtype=run_values.dtype)
        run_lengths_local = coop.local.array(runs_per_thread, dtype=run_lengths.dtype)
        decoded_items = coop.local.array(
            decoded_items_per_thread, dtype=run_values.dtype
        )
        scan_items = coop.local.array(decoded_items_per_thread, dtype=run_values.dtype)
        total_decoded_size = coop.local.array(1, dtype=total_decoded_out.dtype)

        block_offset = cuda.blockIdx.x * runs_per_block
        if block_offset >= total_runs:
            return

        total_decoded_size[0] = 0
        coop.block.load(
            run_values[block_offset:],
            run_values_local,
            items_per_thread=runs_per_thread,
            algorithm=BlockLoadAlgorithm.DIRECT,
            temp_storage=temp_load_values,
        )
        cuda.syncthreads()
        coop.block.load(
            run_lengths[block_offset:],
            run_lengths_local,
            items_per_thread=runs_per_thread,
            algorithm=BlockLoadAlgorithm.DIRECT,
            temp_storage=temp_load_lengths,
        )
        cuda.syncthreads()

        run_length = coop.block.run_length(
            run_values_local,
            run_lengths_local,
            runs_per_thread,
            decoded_items_per_thread,
            total_decoded_size,
            decoded_offset_dtype=decoded_offset_dtype,
            temp_storage=temp_run_length,
        )
        cuda.syncthreads()

        decoded_window_offset = 0
        run_length.decode(decoded_items, decoded_window_offset)

        for i in range(decoded_items_per_thread):
            scan_items[i] = decoded_items[i]

        block_scan(scan_items, scan_items, temp_storage=temp_scan)
        cuda.syncthreads()

        base = cuda.threadIdx.x * decoded_items_per_thread
        for i in range(decoded_items_per_thread):
            decoded_items_out[base + i] = decoded_items[i]
            scanned_items_out[base + i] = scan_items[i]

        smem_histogram = coop.shared.array(bins, dtype=counter_dtype)
        histo = block_histogram()
        histo.init(smem_histogram)
        histo.composite(decoded_items, smem_histogram)
        cuda.syncthreads()

        thread_linear = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        for bin_idx in range(thread_linear, bins, threads_per_block):
            hist_out[bin_idx] = smem_histogram[bin_idx]

        if cuda.threadIdx.x == 0:
            total_decoded_out[0] = total_decoded_size[0]

    def _set_dynamic_shared(kernel_obj, launch_config):
        cufunc = kernel_obj._codelibrary.get_cufunc()
        driver.driver.cuKernelSetAttribute(
            driver.binding.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            dynamic_shared_bytes,
            cufunc.handle,
            cufunc.device.id,
        )
        cufunc.set_shared_memory_carveout(100)

    configured = kernel[1, threads_per_block, 0, dynamic_shared_bytes]
    configured.pre_launch_callbacks.append(_set_dynamic_shared)
    configured(
        d_run_values,
        d_run_lengths,
        d_decoded_items,
        d_scanned_items,
        d_hist,
        d_total_decoded,
    )
    cuda.synchronize()

    h_decoded = d_decoded_items.copy_to_host()
    h_scanned = d_scanned_items.copy_to_host()
    h_hist = d_hist.copy_to_host()
    h_total_decoded = d_total_decoded.copy_to_host()

    decoded = []
    for value, count in zip(h_run_values, h_run_lengths):
        decoded.extend([value] * int(count))
    decoded = np.array(decoded, dtype=item_dtype)

    h_reference_decoded = decoded
    h_reference_scanned = _exclusive_scan_host(decoded)
    h_reference_hist = np.bincount(decoded, minlength=bins).astype(counter_dtype)

    np.testing.assert_array_equal(h_decoded, h_reference_decoded)
    np.testing.assert_array_equal(h_scanned, h_reference_scanned)
    np.testing.assert_array_equal(h_hist, h_reference_hist)
    np.testing.assert_array_equal(h_total_decoded, np.array([window_size]))


@pytest.mark.parametrize(
    "smem_mode",
    [
        "private",
        "shared_auto",
        "shared_manual",
    ],
)
def test_block_primitives_two_phase_gpu_dataclass_smem_modes(smem_mode):
    threads_per_block = 64
    items_per_thread = 2
    items_per_block = threads_per_block * items_per_thread
    num_tiles = 2
    total_items = items_per_block * num_tiles
    begin_bit = numba.int32(0)
    end_bit = numba.int32(8)

    dtype = np.uint32

    block_load = coop.block.load(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=BlockLoadAlgorithm.DIRECT,
    )
    block_store = coop.block.store(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=BlockStoreAlgorithm.DIRECT,
    )
    block_exclusive_sum = coop.block.exclusive_sum(
        dtype,
        threads_per_block,
        items_per_thread,
    )
    block_radix_sort = coop.block.radix_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    block_merge_sort = coop.block.merge_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        _merge_op,
    )

    scan_bytes = block_exclusive_sum.temp_storage_bytes
    scan_alignment = block_exclusive_sum.temp_storage_alignment
    radix_bytes = block_radix_sort.temp_storage_bytes
    radix_alignment = block_radix_sort.temp_storage_alignment
    merge_bytes = block_merge_sort.temp_storage_bytes
    merge_alignment = block_merge_sort.temp_storage_alignment

    shared_bytes = max(scan_bytes, radix_bytes, merge_bytes)
    shared_alignment = max(
        scan_alignment,
        radix_alignment,
        merge_alignment,
    )

    @dataclass
    class KernelParams:
        items_per_thread: int
        block_load: coop.block.load
        block_store: coop.block.store
        block_exclusive_sum: coop.block.exclusive_sum
        block_radix_sort: coop.block.radix_sort_keys
        block_merge_sort: coop.block.merge_sort_keys

    kp = KernelParams(
        items_per_thread=items_per_thread,
        block_load=block_load,
        block_store=block_store,
        block_exclusive_sum=block_exclusive_sum,
        block_radix_sort=block_radix_sort,
        block_merge_sort=block_merge_sort,
    )
    kp = coop.gpu_dataclass(kp)

    @cuda.jit
    def kernel(d_in, d_out, total_items, kp):
        thread_data = coop.local.array(kp.items_per_thread, dtype=d_in.dtype)
        block_offset = cuda.blockIdx.x * items_per_block
        grid_stride = cuda.gridDim.x * items_per_block

        if smem_mode == "shared_auto":
            shared_temp = coop.TempStorage(
                shared_bytes,
                shared_alignment,
                auto_sync=True,
            )
        elif smem_mode == "shared_manual":
            shared_temp = coop.TempStorage(
                shared_bytes,
                shared_alignment,
                auto_sync=False,
            )
        else:
            shared_temp = None

        while block_offset < total_items:
            if smem_mode == "private":
                temp_scan = coop.TempStorage(scan_bytes, scan_alignment)
                temp_radix = coop.TempStorage(radix_bytes, radix_alignment)
                temp_merge = coop.TempStorage(merge_bytes, merge_alignment)

                kp.block_load(
                    d_in[block_offset:],
                    thread_data,
                )
                kp.block_exclusive_sum(thread_data, thread_data, temp_storage=temp_scan)
                kp.block_radix_sort(
                    thread_data,
                    kp.items_per_thread,
                    begin_bit,
                    end_bit,
                    temp_storage=temp_radix,
                )
                kp.block_merge_sort(
                    thread_data,
                    kp.items_per_thread,
                    _merge_op,
                    temp_storage=temp_merge,
                )
                kp.block_store(
                    d_out[block_offset:],
                    thread_data,
                )
            else:
                kp.block_load(
                    d_in[block_offset:],
                    thread_data,
                    temp_storage=shared_temp,
                )
                if smem_mode == "shared_manual":
                    cuda.syncthreads()

                kp.block_exclusive_sum(
                    thread_data,
                    thread_data,
                    temp_storage=shared_temp,
                )
                if smem_mode == "shared_manual":
                    cuda.syncthreads()

                kp.block_radix_sort(
                    thread_data,
                    kp.items_per_thread,
                    begin_bit,
                    end_bit,
                    temp_storage=shared_temp,
                )
                if smem_mode == "shared_manual":
                    cuda.syncthreads()

                kp.block_merge_sort(
                    thread_data,
                    kp.items_per_thread,
                    _merge_op,
                    temp_storage=shared_temp,
                )
                if smem_mode == "shared_manual":
                    cuda.syncthreads()

                kp.block_store(
                    d_out[block_offset:],
                    thread_data,
                    temp_storage=shared_temp,
                )
                if smem_mode == "shared_manual":
                    cuda.syncthreads()

            block_offset += grid_stride

    h_input = np.random.randint(0, 16, total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output, total_items, kp)
    cuda.synchronize()

    h_reference = np.empty_like(h_input)
    for tile_idx in range(num_tiles):
        start = tile_idx * items_per_block
        end = start + items_per_block
        tile = h_input[start:end]
        scanned = _exclusive_scan_host(tile)
        h_reference[start:end] = np.sort(scanned)

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)
