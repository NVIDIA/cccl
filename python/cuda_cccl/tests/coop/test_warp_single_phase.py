# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
import pytest
from helpers import NUMBA_TYPES_TO_NP, random_int
from numba import cuda, types

from cuda import coop
from cuda.coop import WarpLoadAlgorithm, WarpStoreAlgorithm

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_warp_reduce_sum_single_phase():
    @cuda.jit(device=True)
    def max_op(a, b):
        return a if a > b else b

    threads_in_warp = 32

    @cuda.jit
    def kernel(d_in, d_out_max, d_out_sum):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        warp_max = coop.warp.reduce(val, max_op)
        warp_sum = coop.warp.sum(val)
        if tid == 0:
            d_out_max[0] = warp_max
            d_out_sum[0] = warp_sum

    h_input = np.random.randint(0, 100, threads_in_warp, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_max = cuda.device_array(1, dtype=np.int32)
    d_out_sum = cuda.device_array(1, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_out_max, d_out_sum)
    cuda.synchronize()

    h_out_max = d_out_max.copy_to_host()
    h_out_sum = d_out_sum.copy_to_host()

    assert h_out_max[0] == np.max(h_input)
    assert h_out_sum[0] == np.sum(h_input)


def test_warp_sum_single_phase_valid_items():
    threads_in_warp = 32
    valid_items = 10

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        warp_sum = coop.warp.sum(val, valid_items=valid_items)
        if tid == 0:
            d_out[0] = warp_sum

    h_input = np.arange(threads_in_warp, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    assert h_output[0] == np.sum(h_input[:valid_items])


def test_warp_reduce_single_phase_valid_items():
    @cuda.jit(device=True)
    def max_op(a, b):
        return a if a > b else b

    threads_in_warp = 32
    valid_items = 7

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        warp_max = coop.warp.reduce(val, max_op, valid_items=valid_items)
        if tid == 0:
            d_out[0] = warp_max

    h_input = np.random.randint(0, 100, threads_in_warp, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    assert h_output[0] == np.max(h_input[:valid_items])


@pytest.mark.parametrize("T", [types.int32])
def test_warp_exclusive_sum_single_phase(T):
    threads_in_warp = 32

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        d_out[tid] = coop.warp.exclusive_sum(val)

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = np.ones(threads_in_warp, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(threads_in_warp, dtype=dtype)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.arange(threads_in_warp, dtype=dtype)
    np.testing.assert_array_equal(h_output, expected)


@pytest.mark.parametrize("T", [types.int32])
def test_warp_inclusive_sum_single_phase(T):
    threads_in_warp = 32

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        d_out[tid] = coop.warp.inclusive_sum(val)

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = np.ones(threads_in_warp, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(threads_in_warp, dtype=dtype)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.arange(1, threads_in_warp + 1, dtype=dtype)
    np.testing.assert_array_equal(h_output, expected)


@pytest.mark.parametrize("T", [types.int32])
def test_warp_inclusive_sum_single_phase_warp_aggregate(T):
    threads_in_warp = 32

    @cuda.jit
    def kernel(d_in, d_out, d_aggregate):
        tid = cuda.threadIdx.x
        warp_aggregate = cuda.local.array(1, dtype=dtype)
        d_out[tid] = coop.warp.inclusive_sum(d_in[tid], warp_aggregate=warp_aggregate)
        d_aggregate[tid] = warp_aggregate[0]

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(threads_in_warp, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(threads_in_warp, dtype=dtype)
    d_aggregate = cuda.device_array(threads_in_warp, dtype=dtype)

    kernel[1, threads_in_warp](d_input, d_output, d_aggregate)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_aggregate = d_aggregate.copy_to_host()
    np.testing.assert_array_equal(h_output, np.cumsum(h_input))
    expected_aggregate = np.sum(h_input)
    np.testing.assert_array_equal(
        h_aggregate, np.full_like(h_aggregate, expected_aggregate)
    )


@pytest.mark.parametrize("T", [types.int32])
def test_warp_exclusive_scan_single_phase(T):
    threads_in_warp = 32

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        d_out[tid] = coop.warp.exclusive_scan(val, "max", 0)

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(threads_in_warp, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(threads_in_warp, dtype=dtype)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    reference = np.empty_like(h_input)
    running = 0
    for i in range(threads_in_warp):
        reference[i] = running
        running = max(running, int(h_input[i]))
    np.testing.assert_array_equal(h_output, reference)


@pytest.mark.parametrize("T", [types.int32])
def test_warp_inclusive_scan_single_phase(T):
    threads_in_warp = 32

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        d_out[tid] = coop.warp.inclusive_scan(val, "max")

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(threads_in_warp, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(threads_in_warp, dtype=dtype)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    reference = np.maximum.accumulate(h_input)
    np.testing.assert_array_equal(h_output, reference)


def test_warp_inclusive_scan_valid_items_single_phase():
    threads_in_warp = 32
    valid_items = 11

    @cuda.jit
    def kernel(d_in, d_out, d_aggregate):
        tid = cuda.threadIdx.x
        warp_aggregate = cuda.local.array(1, dtype=np.int32)
        d_out[tid] = coop.warp.inclusive_scan(
            d_in[tid],
            "max",
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
        )
        d_aggregate[tid] = warp_aggregate[0]

    h_input = random_int(threads_in_warp, np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(threads_in_warp, dtype=np.int32)
    d_aggregate = cuda.device_array(threads_in_warp, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_output, d_aggregate)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_aggregate = d_aggregate.copy_to_host()
    reference = np.maximum.accumulate(h_input[:valid_items])
    expected_aggregate = np.max(h_input[:valid_items])
    np.testing.assert_array_equal(h_output[:valid_items], reference)
    np.testing.assert_array_equal(
        h_aggregate[:valid_items],
        np.full(valid_items, expected_aggregate, dtype=np.int32),
    )


def test_warp_inclusive_scan_sum_valid_items_single_phase():
    threads_in_warp = 32
    valid_items = 9

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        d_out[tid] = coop.warp.inclusive_scan(
            d_in[tid],
            "+",
            valid_items=valid_items,
        )

    h_input = random_int(threads_in_warp, np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(threads_in_warp, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    reference = np.cumsum(h_input[:valid_items])
    np.testing.assert_array_equal(h_output[:valid_items], reference)


def test_warp_exclusive_scan_sum_valid_items_single_phase():
    threads_in_warp = 32
    valid_items = 12

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        d_out[tid] = coop.warp.exclusive_scan(
            d_in[tid],
            "+",
            0,
            valid_items=valid_items,
        )

    h_input = random_int(threads_in_warp, np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(threads_in_warp, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    reference = np.empty(valid_items, dtype=np.int32)
    running = 0
    for i in range(valid_items):
        reference[i] = running
        running += h_input[i]
    np.testing.assert_array_equal(h_output[:valid_items], reference)


def test_warp_inclusive_scan_temp_storage_single_phase():
    threads_in_warp = 32
    warp_inclusive_scan = coop.warp.inclusive_scan(dtype=types.int32, scan_op="max")
    temp_storage_bytes = warp_inclusive_scan.temp_storage_bytes
    temp_storage_alignment = warp_inclusive_scan.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out_single, d_out_two_phase):
        tid = cuda.threadIdx.x
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        d_out_single[tid] = coop.warp.inclusive_scan(
            d_in[tid],
            "max",
            temp_storage=temp_storage,
        )
        d_out_two_phase[tid] = warp_inclusive_scan(d_in[tid])

    h_input = random_int(threads_in_warp, np.int32)
    d_input = cuda.to_device(h_input)
    d_out_single = cuda.device_array(threads_in_warp, dtype=np.int32)
    d_out_two_phase = cuda.device_array(threads_in_warp, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_out_single, d_out_two_phase)
    cuda.synchronize()

    h_out_single = d_out_single.copy_to_host()
    h_out_two_phase = d_out_two_phase.copy_to_host()
    reference = np.maximum.accumulate(h_input)
    np.testing.assert_array_equal(h_out_single, reference)
    np.testing.assert_array_equal(h_out_two_phase, reference)


def test_warp_reduce_temp_storage_single_phase():
    @cuda.jit(device=True)
    def max_op_single(a, b):
        return a if a > b else b

    @cuda.jit(device=True)
    def max_op_two_phase(a, b):
        return a if a > b else b

    threads_in_warp = 32
    warp_reduce = coop.warp.reduce(types.int32, max_op_two_phase)
    temp_storage_bytes = warp_reduce.temp_storage_bytes
    temp_storage_alignment = warp_reduce.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out_single, d_out_two_phase):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        warp_out_single = coop.warp.reduce(
            val, max_op_single, temp_storage=temp_storage
        )
        warp_out_two_phase = warp_reduce(val, temp_storage=temp_storage)
        if tid == 0:
            d_out_single[0] = warp_out_single
            d_out_two_phase[0] = warp_out_two_phase

    h_input = np.random.randint(0, 100, threads_in_warp, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_single = cuda.device_array(1, dtype=np.int32)
    d_out_two_phase = cuda.device_array(1, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_out_single, d_out_two_phase)
    cuda.synchronize()

    h_out_single = d_out_single.copy_to_host()
    h_out_two_phase = d_out_two_phase.copy_to_host()
    expected = np.max(h_input)
    assert h_out_single[0] == expected
    assert h_out_two_phase[0] == expected


def test_warp_sum_temp_storage_single_phase():
    threads_in_warp = 32
    warp_sum = coop.warp.sum(types.int32)
    temp_storage_bytes = warp_sum.temp_storage_bytes
    temp_storage_alignment = warp_sum.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out_single, d_out_two_phase):
        tid = cuda.threadIdx.x
        val = d_in[tid]
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        warp_out_single = coop.warp.sum(val, temp_storage=temp_storage)
        warp_out_two_phase = warp_sum(val, temp_storage=temp_storage)
        if tid == 0:
            d_out_single[0] = warp_out_single
            d_out_two_phase[0] = warp_out_two_phase

    h_input = np.random.randint(0, 100, threads_in_warp, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_out_single = cuda.device_array(1, dtype=np.int32)
    d_out_two_phase = cuda.device_array(1, dtype=np.int32)

    kernel[1, threads_in_warp](d_input, d_out_single, d_out_two_phase)
    cuda.synchronize()

    h_out_single = d_out_single.copy_to_host()
    h_out_two_phase = d_out_two_phase.copy_to_host()
    expected = np.sum(h_input)
    assert h_out_single[0] == expected
    assert h_out_two_phase[0] == expected


def test_warp_load_store_single_phase():
    threads_in_warp = 32
    items_per_thread = 4

    @cuda.jit
    def kernel(d_in, d_out):
        items = cuda.local.array(items_per_thread, dtype=numba.int32)
        coop.warp.load(
            d_in,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpLoadAlgorithm.STRIPED,
        )
        coop.warp.store(
            d_out,
            items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=WarpStoreAlgorithm.STRIPED,
        )

    h_input = np.random.randint(
        0, 42, threads_in_warp * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    np.testing.assert_allclose(h_output, h_input)


def test_warp_merge_sort_single_phase():
    T = types.int32
    items_per_thread = 2
    threads_in_warp = 32

    @cuda.jit(device=True)
    def op(a, b):
        return a < b

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = d_in[tid * items_per_thread + i]
        coop.warp.merge_sort_keys(
            thread_data,
            items_per_thread=items_per_thread,
            compare_op=op,
            threads_in_warp=threads_in_warp,
        )
        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = thread_data[i]

    dtype = NUMBA_TYPES_TO_NP[T]
    items_per_tile = threads_in_warp * items_per_thread
    h_input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)

    kernel[1, threads_in_warp](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    reference = sorted(h_input)
    for i in range(items_per_tile):
        assert h_output[i] == reference[i]


def test_warp_merge_sort_key_value_single_phase():
    T = types.int32
    items_per_thread = 2
    threads_in_warp = 32

    @cuda.jit(device=True)
    def op(a, b):
        return a < b

    @cuda.jit
    def kernel(keys_in, values_in, keys_out, values_out):
        tid = cuda.threadIdx.x
        thread_keys = cuda.local.array(shape=items_per_thread, dtype=dtype)
        thread_values = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            thread_keys[i] = keys_in[idx]
            thread_values[i] = values_in[idx]
        coop.warp.merge_sort_keys(
            thread_keys,
            items_per_thread=items_per_thread,
            compare_op=op,
            threads_in_warp=threads_in_warp,
            values=thread_values,
        )
        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            keys_out[idx] = thread_keys[i]
            values_out[idx] = thread_values[i]

    dtype = NUMBA_TYPES_TO_NP[T]
    items_per_tile = threads_in_warp * items_per_thread
    h_keys = random_int(items_per_tile, dtype)
    h_values = (h_keys * 4 + 5).astype(dtype)
    d_keys = cuda.to_device(h_keys)
    d_values = cuda.to_device(h_values)
    d_keys_out = cuda.device_array(items_per_tile, dtype=dtype)
    d_values_out = cuda.device_array(items_per_tile, dtype=dtype)

    kernel[1, threads_in_warp](d_keys, d_values, d_keys_out, d_values_out)
    cuda.synchronize()

    keys_out = d_keys_out.copy_to_host()
    values_out = d_values_out.copy_to_host()

    expected = sorted(zip(h_keys, h_values), key=lambda kv: kv[0])
    exp_keys = [kv[0] for kv in expected]
    exp_values = [kv[1] for kv in expected]

    assert list(keys_out) == exp_keys
    assert list(values_out) == exp_values
