# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as coop

from ..support.runtime import (
    ITEMS_PER_THREAD,
    THREADS,
    _less,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


def test_device_function_single_phase_block_and_warp_primitives():
    threads_per_block = 64
    warps_per_block = threads_per_block // 32

    @cuda.jit(device=True)
    def device_block_sum(val):
        return coop._block.sum(val, items_per_thread=1)

    @cuda.jit(device=True)
    def device_warp_sum(val):
        return coop._warp.sum(val)

    @cuda.jit
    def kernel(d_in, d_out_block, d_out_warp):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        warp_sum = device_warp_sum(val)
        block_sum = device_block_sum(val)
        if tid == 0:
            d_out_block[0] = block_sum
        if tid % 32 == 0:
            d_out_warp[tid // 32] = warp_sum

    h_input = np.random.randint(0, 64, threads_per_block, dtype=np.int32)
    h_out_block = np.zeros(1, dtype=np.int32)
    h_out_warp = np.zeros(warps_per_block, dtype=np.int32)

    kernel[1, threads_per_block](h_input, h_out_block, h_out_warp)

    expected_warp = np.asarray(
        [np.sum(h_input[i * 32 : (i + 1) * 32]) for i in range(warps_per_block)],
        dtype=np.int32,
    )

    assert h_out_block[0] == np.sum(h_input)
    np.testing.assert_array_equal(h_out_warp, expected_warp)


@cuda.jit
def _block_single_phase_dim_alias_kernel(d_in, d_roundtrip, d_scan, d_sum):
    tid = cuda.threadIdx.x
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    scanned = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    coop._block.load(
        d_in,
        items,
        dtype="int32",
        dim=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    coop._block.store(
        d_roundtrip,
        items,
        dtype="int32",
        dim=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    coop._block.inclusive_sum(
        items,
        scanned,
        dtype="int32",
        dim=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    for item in range(ITEMS_PER_THREAD):
        d_scan[tid * ITEMS_PER_THREAD + item] = scanned[item]

    total = coop._block.sum(d_in[tid], dtype="int32", dim=THREADS)
    if tid == 0:
        d_sum[0] = total


def test_block_single_phase_accepts_dim_alias():
    h_input = np.arange(1, THREADS * ITEMS_PER_THREAD + 1, dtype=np.int32)
    h_roundtrip = np.zeros_like(h_input)
    h_scan = np.zeros_like(h_input)
    h_sum = np.zeros(1, dtype=np.int32)

    _block_single_phase_dim_alias_kernel[1, THREADS](
        h_input, h_roundtrip, h_scan, h_sum
    )

    np.testing.assert_array_equal(h_roundtrip, h_input)
    np.testing.assert_array_equal(h_scan, np.cumsum(h_input).astype(np.int32))
    np.testing.assert_array_equal(
        h_sum, np.asarray([np.sum(h_input[:THREADS])], dtype=np.int32)
    )


def test_two_phase_invocables_accept_temp_storage_keyword():
    block_sum = coop._block.make_sum(types.int32, threads_per_block=THREADS)
    block_scan = coop._block.make_inclusive_sum(
        types.int32,
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    warp_sum = coop._warp.make_sum(types.int32, threads_in_warp=THREADS)
    warp_scan = coop._warp.make_inclusive_scan(
        types.int32,
        "+",
        threads_in_warp=THREADS,
    )

    @cuda.jit
    def kernel(d_in, d_out, d_scan):
        tid = cuda.threadIdx.x
        temp_storage = coop.TempStorage()

        block_total = block_sum(d_in[tid], temp_storage=temp_storage)
        warp_total = warp_sum(d_in[tid], temp_storage=temp_storage)
        if tid == 0:
            d_out[0] = block_total
            d_out[1] = warp_total

        d_out[tid + 2] = warp_scan(d_in[tid], temp_storage=temp_storage)

        items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
        scanned = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
        for item in range(ITEMS_PER_THREAD):
            items[item] = d_in[tid * ITEMS_PER_THREAD + item]
        block_scan(items, scanned, temp_storage=temp_storage)
        for item in range(ITEMS_PER_THREAD):
            d_scan[tid * ITEMS_PER_THREAD + item] = scanned[item]

    h_input = np.arange(1, THREADS * ITEMS_PER_THREAD + 1, dtype=np.int32)
    h_out = np.zeros(THREADS + 2, dtype=np.int32)
    h_scan = np.zeros_like(h_input)

    kernel[1, THREADS](h_input, h_out, h_scan)

    np.testing.assert_array_equal(
        h_out[:2],
        np.asarray(
            [np.sum(h_input[:THREADS]), np.sum(h_input[:THREADS])],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        h_out[2:], np.cumsum(h_input[:THREADS]).astype(np.int32)
    )
    np.testing.assert_array_equal(h_scan, np.cumsum(h_input).astype(np.int32))


@cuda.jit
def _warp_sum_temp_storage_getitem_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    temp_storage = coop.TempStorage()
    total = coop._warp.sum[temp_storage](
        d_in[tid],
        dtype="int32",
        threads_in_warp=THREADS,
    )
    if tid == 0:
        d_out[0] = total


def test_single_phase_factory_temp_storage_getitem_sugar():
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_output = np.zeros(1, dtype=np.int32)

    _warp_sum_temp_storage_getitem_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(
        h_output, np.asarray([np.sum(h_input)], dtype=np.int32)
    )


def test_block_sort_invocables_accept_temp_storage_keyword():
    merge_sort = coop._block.make_merge_sort_keys(
        types.int32,
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        compare_op=_less,
    )
    radix_sort = coop._block.make_radix_sort_keys(
        types.int32,
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    merge_sort_bytes = merge_sort.temp_storage_bytes
    merge_sort_alignment = merge_sort.temp_storage_alignment
    radix_sort_bytes = radix_sort.temp_storage_bytes
    radix_sort_alignment = radix_sort.temp_storage_alignment

    @cuda.jit
    def merge_kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        temp_storage = coop.TempStorage(
            merge_sort_bytes,
            merge_sort_alignment,
        )
        keys = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
        for item in range(ITEMS_PER_THREAD):
            keys[item] = d_in[tid * ITEMS_PER_THREAD + item]
        merge_sort(keys, temp_storage=temp_storage)
        for item in range(ITEMS_PER_THREAD):
            d_out[tid * ITEMS_PER_THREAD + item] = keys[item]

    @cuda.jit
    def radix_kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        temp_storage = coop.TempStorage(
            radix_sort_bytes,
            radix_sort_alignment,
        )
        keys = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
        for item in range(ITEMS_PER_THREAD):
            keys[item] = d_in[tid * ITEMS_PER_THREAD + item]
        radix_sort(keys, temp_storage=temp_storage)
        for item in range(ITEMS_PER_THREAD):
            d_out[tid * ITEMS_PER_THREAD + item] = keys[item]

    h_keys = np.arange(THREADS * ITEMS_PER_THREAD, 0, -1, dtype=np.int32)
    h_merge = np.zeros_like(h_keys)
    h_radix = np.zeros_like(h_keys)

    merge_kernel[1, THREADS](h_keys, h_merge)
    radix_kernel[1, THREADS](h_keys, h_radix)

    expected = np.sort(h_keys)
    np.testing.assert_array_equal(h_merge, expected)
    np.testing.assert_array_equal(h_radix, expected)


@cuda.jit
def _block_sort_scan_grid_stride_thread_data_kernel(d_in, d_out, total_items):
    items_per_block = THREADS * ITEMS_PER_THREAD
    block_offset = cuda.blockIdx.x * items_per_block
    grid_stride = cuda.gridDim.x * items_per_block

    while block_offset < total_items:
        thread_data = coop.ThreadData(ITEMS_PER_THREAD)
        coop._block.load(d_in, thread_data, offset=block_offset)
        coop._block.scan(thread_data, thread_data)
        coop._block.radix_sort_keys(thread_data, begin_bit=0, end_bit=8)
        coop._block.merge_sort_keys(thread_data, compare_op=_less)
        coop._block.store(d_out, thread_data, offset=block_offset)
        block_offset += grid_stride


def test_block_sort_scan_grid_stride_thread_data():
    items_per_block = THREADS * ITEMS_PER_THREAD
    total_items = items_per_block * 3
    h_input = ((np.arange(total_items, dtype=np.uint32) * 17) % 16).astype(np.uint32)
    h_output = np.zeros_like(h_input)

    _block_sort_scan_grid_stride_thread_data_kernel[1, THREADS](
        h_input, h_output, total_items
    )

    expected = np.empty_like(h_input)
    for start in range(0, total_items, items_per_block):
        end = start + items_per_block
        tile = h_input[start:end]
        scanned = np.concatenate(
            [np.asarray([0], dtype=np.uint32), np.cumsum(tile[:-1], dtype=np.uint32)]
        ).astype(np.uint32)
        expected[start:end] = np.sort(scanned)

    np.testing.assert_array_equal(h_output, expected)


@cuda.jit
def _block_temp_storage_chain_shared_auto_kernel(d_in, d_out):
    temp_storage = coop.TempStorage()
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
    scanned = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)

    coop._block.load(d_in, items, temp_storage=temp_storage)
    coop._block.exclusive_sum(items, scanned, temp_storage=temp_storage)
    coop._block.radix_sort_keys(
        scanned,
        begin_bit=0,
        end_bit=8,
        temp_storage=temp_storage,
    )
    coop._block.merge_sort_keys(
        scanned,
        compare_op=_less,
        temp_storage=temp_storage,
    )
    coop._block.store(d_out, scanned, temp_storage=temp_storage)


@cuda.jit
def _block_temp_storage_chain_shared_manual_kernel(d_in, d_out):
    temp_storage = coop.TempStorage(auto_sync=False)
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
    scanned = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)

    coop._block.load(d_in, items, temp_storage=temp_storage)
    cuda.syncthreads()
    coop._block.exclusive_sum(items, scanned, temp_storage=temp_storage)
    cuda.syncthreads()
    coop._block.radix_sort_keys(
        scanned,
        begin_bit=0,
        end_bit=8,
        temp_storage=temp_storage,
    )
    cuda.syncthreads()
    coop._block.merge_sort_keys(
        scanned,
        compare_op=_less,
        temp_storage=temp_storage,
    )
    cuda.syncthreads()
    coop._block.store(d_out, scanned, temp_storage=temp_storage)
    cuda.syncthreads()


@cuda.jit
def _block_temp_storage_chain_exclusive_kernel(d_in, d_out):
    temp_storage = coop.TempStorage(sharing="exclusive")
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
    scanned = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)

    coop._block.load(d_in, items, temp_storage=temp_storage)
    coop._block.exclusive_sum(items, scanned, temp_storage=temp_storage)
    coop._block.radix_sort_keys(
        scanned,
        begin_bit=0,
        end_bit=8,
        temp_storage=temp_storage,
    )
    coop._block.merge_sort_keys(
        scanned,
        compare_op=_less,
        temp_storage=temp_storage,
    )
    coop._block.store(d_out, scanned, temp_storage=temp_storage)


@pytest.mark.parametrize(
    "kernel",
    [
        _block_temp_storage_chain_shared_auto_kernel,
        _block_temp_storage_chain_shared_manual_kernel,
        _block_temp_storage_chain_exclusive_kernel,
    ],
)
def test_block_temp_storage_chained_primitives(kernel):
    h_input = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.uint32) * 7) % 17
    h_output = np.zeros_like(h_input)

    kernel[1, THREADS](h_input, h_output)

    expected = np.concatenate(
        [np.asarray([0], dtype=np.uint32), np.cumsum(h_input[:-1], dtype=np.uint64)]
    ).astype(np.uint32)
    expected = np.sort(expected)
    np.testing.assert_array_equal(h_output, expected)


def test_temp_storage_dynamic_shared_launch_minimum():
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        _query_device_shared_memory_limits,
    )

    limits = _query_device_shared_memory_limits()
    max_default = int(limits["max_default_shared_memory_per_block"])
    max_optin = int(limits["max_optin_shared_memory_per_block"])
    if max_optin <= max_default:
        pytest.skip("Device does not support shared memory opt-in.")

    block_scan = coop._block.make_exclusive_sum(
        types.uint32,
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    temp_storage_alignment = max(1, int(block_scan.temp_storage_alignment or 1))
    min_dynamic = max(
        max_default + temp_storage_alignment,
        int(block_scan.temp_storage_bytes),
    )
    temp_storage_bytes = (
        (min_dynamic + temp_storage_alignment - 1) // temp_storage_alignment
    ) * temp_storage_alignment
    if temp_storage_bytes > max_optin:
        pytest.skip(
            "Device does not support required dynamic shared memory size "
            f"({temp_storage_bytes} > {max_optin})."
        )

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
        scanned = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)

        coop._block.load(d_in, items)
        coop._block.exclusive_sum(items, scanned, temp_storage=temp_storage)
        coop._block.store(d_out, scanned)

    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.uint32)
    h_output = np.zeros_like(h_input)

    kernel[1, THREADS](h_input, h_output)

    compile_results = [
        *kernel.overloads.values(),
        *getattr(kernel, "_launch_config_overloads", {}).values(),
    ]
    min_sharedmem = max(
        int(result.metadata.get("required_dynamic_shared_memory", 0) or 0)
        for result in compile_results
    )
    assert min_sharedmem >= temp_storage_bytes
    expected = np.concatenate(
        [np.asarray([0], dtype=np.uint32), np.cumsum(h_input[:-1], dtype=np.uint64)]
    ).astype(np.uint32)
    np.testing.assert_array_equal(h_output, expected)
