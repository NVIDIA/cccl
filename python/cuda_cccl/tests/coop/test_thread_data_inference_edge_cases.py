# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
import pytest
from numba import cuda

from cuda import coop
from cuda.coop.block import BlockExchangeType

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

striped_to_blocked = BlockExchangeType.StripedToBlocked


def _exclusive_scan_host(values):
    if values.size == 0:
        return np.empty_like(values)
    output = np.empty_like(values)
    output[0] = values.dtype.type(0)
    output[1:] = np.cumsum(values[:-1], dtype=values.dtype)
    return output


def test_thread_data_multiple_instances_control_flow():
    threads_per_block = 128
    items_per_thread = 4
    total_items = threads_per_block * items_per_thread

    @cuda.jit
    def kernel(d_in_i32, d_in_f32, d_out_i32, d_out_f32, use_i32):
        td_i32 = coop.ThreadData(items_per_thread)
        td_f32 = coop.ThreadData(items_per_thread)

        coop.block.load(d_in_i32, td_i32, items_per_thread=items_per_thread)
        coop.block.load(d_in_f32, td_f32, items_per_thread=items_per_thread)

        if use_i32:
            coop.block.scan(
                td_i32,
                td_i32,
                items_per_thread=items_per_thread,
            )
        else:
            coop.block.scan(
                td_f32,
                td_f32,
                items_per_thread=items_per_thread,
            )

        coop.block.store(d_out_i32, td_i32, items_per_thread=items_per_thread)
        coop.block.store(d_out_f32, td_f32, items_per_thread=items_per_thread)

    h_i32 = np.random.randint(0, 16, total_items, dtype=np.int32)
    h_f32 = np.random.random(total_items).astype(np.float32)

    d_i32 = cuda.to_device(h_i32)
    d_f32 = cuda.to_device(h_f32)
    d_out_i32 = cuda.device_array_like(d_i32)
    d_out_f32 = cuda.device_array_like(d_f32)

    kernel[1, threads_per_block](d_i32, d_f32, d_out_i32, d_out_f32, True)
    cuda.synchronize()

    out_i32 = d_out_i32.copy_to_host()
    out_f32 = d_out_f32.copy_to_host()

    np.testing.assert_array_equal(out_i32, _exclusive_scan_host(h_i32))
    np.testing.assert_allclose(out_f32, h_f32)

    kernel[1, threads_per_block](d_i32, d_f32, d_out_i32, d_out_f32, False)
    cuda.synchronize()

    out_i32 = d_out_i32.copy_to_host()
    out_f32 = d_out_f32.copy_to_host()

    np.testing.assert_array_equal(out_i32, h_i32)
    np.testing.assert_allclose(
        out_f32,
        _exclusive_scan_host(h_f32),
        rtol=1e-5,
        atol=1e-6,
    )


def test_thread_data_dtype_mismatch_across_primitives_raises():
    threads_per_block = 128
    items_per_thread = 4

    @cuda.jit
    def kernel(d_in):
        td = coop.ThreadData(items_per_thread)
        tmp = cuda.local.array(items_per_thread, dtype=numba.float32)
        coop.block.load(d_in, td, items_per_thread=items_per_thread)
        coop.block.exchange(
            td,
            tmp,
            items_per_thread=items_per_thread,
            block_exchange_type=striped_to_blocked,
        )

    h_input = np.random.randint(
        0, 16, threads_per_block * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)

    with pytest.raises(Exception, match="consistent dtype for ThreadData"):
        kernel[1, threads_per_block](d_input)


def test_thread_data_mixed_items_per_thread():
    threads_per_block = 128
    scalar_items_per_thread = 1
    vector_items_per_thread = 4

    total_scalar = threads_per_block * scalar_items_per_thread
    total_vector = threads_per_block * vector_items_per_thread

    @cuda.jit
    def kernel(d_in_scalar, d_in_vec, d_out_scalar, d_out_vec):
        td_scalar = coop.ThreadData(scalar_items_per_thread)
        td_vec = coop.ThreadData(vector_items_per_thread)

        coop.block.load(
            d_in_scalar,
            td_scalar,
            items_per_thread=scalar_items_per_thread,
        )
        coop.block.load(
            d_in_vec,
            td_vec,
            items_per_thread=vector_items_per_thread,
        )

        coop.block.exchange(
            td_scalar,
            items_per_thread=scalar_items_per_thread,
            block_exchange_type=striped_to_blocked,
        )
        coop.block.scan(
            td_vec,
            td_vec,
            items_per_thread=vector_items_per_thread,
        )

        coop.block.store(
            d_out_scalar,
            td_scalar,
            items_per_thread=scalar_items_per_thread,
        )
        coop.block.store(
            d_out_vec,
            td_vec,
            items_per_thread=vector_items_per_thread,
        )

    h_scalar = np.random.randint(0, 16, total_scalar, dtype=np.int32)
    h_vec = np.random.randint(0, 16, total_vector, dtype=np.int32)

    d_scalar = cuda.to_device(h_scalar)
    d_vec = cuda.to_device(h_vec)
    d_out_scalar = cuda.device_array_like(d_scalar)
    d_out_vec = cuda.device_array_like(d_vec)

    kernel[1, threads_per_block](d_scalar, d_vec, d_out_scalar, d_out_vec)
    cuda.synchronize()

    out_scalar = d_out_scalar.copy_to_host()
    out_vec = d_out_vec.copy_to_host()

    np.testing.assert_array_equal(out_scalar, h_scalar)
    np.testing.assert_array_equal(out_vec, _exclusive_scan_host(h_vec))
