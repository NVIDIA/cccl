# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
import pytest
from helpers import NUMBA_TYPES_TO_NP, random_int
from numba import cuda, types

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
def test_warp_exclusive_sum(T):
    warp_exclusive_sum = coop.warp.exclusive_sum(dtype=T)

    @cuda.jit
    def kernel(input, output):
        tid = cuda.threadIdx.x
        output[tid] = warp_exclusive_sum(input[tid])

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(32, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = np.cumsum(h_input) - h_input
    for i in range(32):
        assert output[i] == reference[i]

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
def test_warp_inclusive_sum(T):
    warp_inclusive_sum = coop.warp.inclusive_sum(dtype=T)

    @cuda.jit
    def kernel(input, output):
        tid = cuda.threadIdx.x
        output[tid] = warp_inclusive_sum(input[tid])

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(32, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = np.cumsum(h_input)
    for i in range(32):
        assert output[i] == reference[i]


@pytest.mark.parametrize("T", [types.uint32])
def test_warp_inclusive_sum_warp_aggregate(T):
    warp_inclusive_sum = coop.warp.inclusive_sum(dtype=T)
    dtype = NUMBA_TYPES_TO_NP[T]

    @cuda.jit
    def kernel(input, output, aggregate):
        tid = cuda.threadIdx.x
        warp_aggregate = cuda.local.array(1, dtype=dtype)
        output[tid] = warp_inclusive_sum(input[tid], warp_aggregate=warp_aggregate)
        aggregate[tid] = warp_aggregate[0]

    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(32, dtype=dtype)
    d_aggregate = cuda.device_array(32, dtype=dtype)
    kernel[1, 32](d_input, d_output, d_aggregate)
    cuda.synchronize()

    output = d_output.copy_to_host()
    aggregate = d_aggregate.copy_to_host()
    reference = np.cumsum(h_input)
    np.testing.assert_array_equal(output, reference)
    expected_aggregate = np.sum(h_input)
    np.testing.assert_array_equal(
        aggregate, np.full_like(aggregate, expected_aggregate)
    )


@pytest.mark.parametrize("T", [types.int32])
def test_warp_exclusive_scan_max(T):
    warp_exclusive_scan = coop.warp.exclusive_scan(
        dtype=T, scan_op="max", initial_value=0
    )

    @cuda.jit
    def kernel(input, output):
        tid = cuda.threadIdx.x
        output[tid] = warp_exclusive_scan(input[tid], 0)

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(32, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = np.empty_like(h_input)
    running = 0
    for i in range(32):
        reference[i] = running
        running = max(running, int(h_input[i]))
    np.testing.assert_array_equal(output, reference)


@pytest.mark.parametrize("T", [types.int32])
def test_warp_inclusive_scan_max(T):
    warp_inclusive_scan = coop.warp.inclusive_scan(dtype=T, scan_op="max")

    @cuda.jit
    def kernel(input, output):
        tid = cuda.threadIdx.x
        output[tid] = warp_inclusive_scan(input[tid])

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(32, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = np.maximum.accumulate(h_input)
    np.testing.assert_array_equal(output, reference)


@pytest.mark.parametrize("T", [types.int32])
def test_warp_inclusive_scan_valid_items_warp_aggregate(T):
    warp_inclusive_scan = coop.warp.inclusive_scan(dtype=T, scan_op="max")
    valid_items = 13
    dtype = NUMBA_TYPES_TO_NP[T]

    @cuda.jit
    def kernel(input, output, aggregate):
        tid = cuda.threadIdx.x
        warp_aggregate = cuda.local.array(1, dtype=dtype)
        output[tid] = warp_inclusive_scan(
            input[tid],
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
        )
        aggregate[tid] = warp_aggregate[0]

    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(32, dtype=dtype)
    d_aggregate = cuda.device_array(32, dtype=dtype)
    kernel[1, 32](d_input, d_output, d_aggregate)
    cuda.synchronize()

    output = d_output.copy_to_host()
    aggregate = d_aggregate.copy_to_host()
    reference = np.maximum.accumulate(h_input[:valid_items])
    expected_aggregate = np.max(h_input[:valid_items])
    np.testing.assert_array_equal(output[:valid_items], reference)
    np.testing.assert_array_equal(
        aggregate[:valid_items],
        np.full(valid_items, expected_aggregate, dtype=dtype),
    )


@pytest.mark.parametrize("T", [types.int32])
def test_warp_inclusive_scan_sum_valid_items(T):
    warp_inclusive_scan = coop.warp.inclusive_scan(dtype=T, scan_op="+")
    valid_items = 17

    @cuda.jit
    def kernel(input, output):
        tid = cuda.threadIdx.x
        output[tid] = warp_inclusive_scan(input[tid], valid_items=valid_items)

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(32, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = np.cumsum(h_input[:valid_items])
    np.testing.assert_array_equal(output[:valid_items], reference)


@pytest.mark.parametrize("T", [types.int32])
def test_warp_exclusive_scan_sum_valid_items(T):
    warp_exclusive_scan = coop.warp.exclusive_scan(
        dtype=T, scan_op="+", initial_value=0
    )
    valid_items = 19

    @cuda.jit
    def kernel(input, output):
        tid = cuda.threadIdx.x
        output[tid] = warp_exclusive_scan(input[tid], 0, valid_items=valid_items)

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(32, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = np.empty(valid_items, dtype=dtype)
    running = 0
    for i in range(valid_items):
        reference[i] = running
        running += h_input[i]
    np.testing.assert_array_equal(output[:valid_items], reference)


@pytest.mark.parametrize("T", [types.int32])
def test_warp_inclusive_scan_temp_storage_two_phase(T):
    warp_inclusive_scan = coop.warp.inclusive_scan(dtype=T, scan_op="max")
    temp_storage_bytes = warp_inclusive_scan.temp_storage_bytes
    temp_storage_alignment = warp_inclusive_scan.temp_storage_alignment

    @cuda.jit
    def kernel(input, output):
        tid = cuda.threadIdx.x
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        output[tid] = warp_inclusive_scan(input[tid], temp_storage=temp_storage)

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(32, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = np.maximum.accumulate(h_input)
    np.testing.assert_array_equal(output, reference)
