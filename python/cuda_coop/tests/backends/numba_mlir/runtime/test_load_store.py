# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")


import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir._block import (
    BlockExchangeType,
)

from ..support.runtime import (
    ITEMS_PER_THREAD,
    THREADS,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _warp_load_store_kernel(d_in, d_out):
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    coop._warp.load(
        d_in,
        items,
        dtype="int32",
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        algorithm="direct",
    )
    coop._warp.store(
        d_out,
        items,
        dtype="int32",
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        algorithm="direct",
    )


def test_warp_load_store_round_trip():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _warp_load_store_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


@cuda.jit
def _warp_load_store_thread_data_kernel(d_in, d_out):
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)

    coop._warp.load(
        d_in,
        items,
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        algorithm="direct",
    )
    coop._warp.store(
        d_out,
        items,
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        algorithm="direct",
    )


@cuda.jit
def _warp_load_store_thread_data_temp_storage_kernel(d_in, d_out):
    temp_storage = coop.TempStorage()
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)

    coop._warp.load(
        d_in,
        items,
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        algorithm="transpose",
        temp_storage=temp_storage,
    )
    coop._warp.store(
        d_out,
        items,
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        algorithm="transpose",
        temp_storage=temp_storage,
    )


@cuda.jit
def _warp_load_store_num_valid_all_kernel(d_in, d_out_all, valid_items, oob):
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    coop._warp.load(
        d_in,
        items,
        valid_items,
        oob,
        dtype="int32",
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        algorithm="striped",
    )
    coop._warp.store(
        d_out_all,
        items,
        dtype="int32",
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        algorithm="striped",
    )


@cuda.jit
def _warp_load_store_num_valid_valid_kernel(d_in, d_out_valid, valid_items, oob):
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    coop._warp.load(
        d_in,
        items,
        valid_items,
        oob,
        dtype="int32",
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        algorithm="striped",
    )
    coop._warp.store(
        d_out_valid,
        items,
        valid_items,
        dtype="int32",
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        algorithm="striped",
    )


def test_warp_load_store_thread_data_round_trip():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _warp_load_store_thread_data_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


def test_warp_load_store_thread_data_temp_storage_round_trip():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _warp_load_store_thread_data_temp_storage_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


def test_warp_load_store_num_valid_oob_default():
    total_items = THREADS * ITEMS_PER_THREAD
    valid_items = np.int32(total_items - 7)
    oob_default = np.int32(-123)
    sentinel = np.int32(-999)
    h_input = np.arange(total_items, dtype=np.int32)
    h_out_all = np.zeros_like(h_input)
    h_out_valid = np.full(total_items, sentinel, dtype=np.int32)

    _warp_load_store_num_valid_all_kernel[1, THREADS](
        h_input, h_out_all, valid_items, oob_default
    )
    _warp_load_store_num_valid_valid_kernel[1, THREADS](
        h_input, h_out_valid, valid_items, oob_default
    )

    expected_all = np.full(total_items, oob_default, dtype=np.int32)
    expected_all[:valid_items] = h_input[:valid_items]
    expected_valid = np.full(total_items, sentinel, dtype=np.int32)
    expected_valid[:valid_items] = h_input[:valid_items]
    np.testing.assert_array_equal(h_out_all, expected_all)
    np.testing.assert_array_equal(h_out_valid, expected_valid)


@cuda.jit
def _block_load_store_kernel(d_in, d_out):
    items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    coop._block.load(
        d_in,
        items,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        algorithm="direct",
    )
    coop._block.store(
        d_out,
        items,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        algorithm="direct",
    )


def test_block_load_store_round_trip():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _block_load_store_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


@cuda.jit
def _thread_data_explicit_dtype_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    thread_data = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)

    for item in range(ITEMS_PER_THREAD):
        thread_data[item] = d_in[tid * ITEMS_PER_THREAD + item]

    for item in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + item] = thread_data[item]


def test_thread_data_explicit_dtype_direct_round_trip():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _thread_data_explicit_dtype_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


@cuda.jit
def _block_load_store_thread_data_kernel(d_in, d_out):
    thread_data = coop.ThreadData(ITEMS_PER_THREAD)
    coop._block.load(d_in, thread_data)
    coop._block.store(d_out, thread_data)


def test_block_load_store_thread_data_round_trip():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _block_load_store_thread_data_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


@cuda.jit
def _thread_data_multiple_instances_control_flow_kernel(
    d_in_i32, d_in_f32, d_out_i32, d_out_f32, use_i32
):
    td_i32 = coop.ThreadData(ITEMS_PER_THREAD)
    td_f32 = coop.ThreadData(ITEMS_PER_THREAD)

    coop._block.load(d_in_i32, td_i32)
    coop._block.load(d_in_f32, td_f32)

    if use_i32:
        coop._block.scan(td_i32, td_i32)
    else:
        coop._block.scan(td_f32, td_f32)

    coop._block.store(d_out_i32, td_i32)
    coop._block.store(d_out_f32, td_f32)


def test_thread_data_multiple_instances_control_flow():
    h_i32 = np.arange(1, THREADS * ITEMS_PER_THREAD + 1, dtype=np.int32)
    h_f32 = np.linspace(1.0, 2.0, THREADS * ITEMS_PER_THREAD, dtype=np.float32)
    h_out_i32 = np.zeros_like(h_i32)
    h_out_f32 = np.zeros_like(h_f32)

    _thread_data_multiple_instances_control_flow_kernel[1, THREADS](
        h_i32, h_f32, h_out_i32, h_out_f32, True
    )

    np.testing.assert_array_equal(
        h_out_i32,
        np.concatenate([np.asarray([0], dtype=np.int32), np.cumsum(h_i32[:-1])]),
    )
    np.testing.assert_allclose(h_out_f32, h_f32)

    _thread_data_multiple_instances_control_flow_kernel[1, THREADS](
        h_i32, h_f32, h_out_i32, h_out_f32, False
    )

    np.testing.assert_array_equal(h_out_i32, h_i32)
    np.testing.assert_allclose(
        h_out_f32,
        np.concatenate([np.asarray([0], dtype=np.float32), np.cumsum(h_f32[:-1])]),
        rtol=1e-5,
        atol=1e-6,
    )


@cuda.jit
def _thread_data_mixed_items_per_thread_kernel(
    d_in_scalar, d_in_vec, d_out_scalar, d_out_vec
):
    td_scalar = coop.ThreadData(1)
    td_vec = coop.ThreadData(ITEMS_PER_THREAD)

    coop._block.load(d_in_scalar, td_scalar)
    coop._block.load(d_in_vec, td_vec)

    coop._block.exchange(
        td_scalar,
        block_exchange_type=BlockExchangeType.StripedToBlocked,
    )
    coop._block.scan(td_vec, td_vec)

    coop._block.store(d_out_scalar, td_scalar)
    coop._block.store(d_out_vec, td_vec)


def test_thread_data_mixed_items_per_thread():
    h_scalar = np.arange(1, THREADS + 1, dtype=np.int32)
    h_vec = np.arange(1, THREADS * ITEMS_PER_THREAD + 1, dtype=np.int32)
    h_out_scalar = np.zeros_like(h_scalar)
    h_out_vec = np.zeros_like(h_vec)

    _thread_data_mixed_items_per_thread_kernel[1, THREADS](
        h_scalar, h_vec, h_out_scalar, h_out_vec
    )

    np.testing.assert_array_equal(h_out_scalar, h_scalar)
    np.testing.assert_array_equal(
        h_out_vec,
        np.concatenate([np.asarray([0], dtype=np.int32), np.cumsum(h_vec[:-1])]),
    )


@cuda.jit
def _block_load_store_temp_storage_kernel(d_in, d_out):
    temp_storage = coop.TempStorage()
    thread_data = coop.ThreadData(ITEMS_PER_THREAD)
    coop._block.load(
        d_in,
        thread_data,
        algorithm="transpose",
        temp_storage=temp_storage,
    )
    coop._block.store(
        d_out,
        thread_data,
        algorithm="transpose",
        temp_storage=temp_storage,
    )


@cuda.jit
def _block_load_store_exclusive_temp_storage_kernel(d_in, d_out):
    temp_storage = coop.TempStorage(sharing="exclusive")
    thread_data = coop.ThreadData(ITEMS_PER_THREAD)
    coop._block.load(
        d_in,
        thread_data,
        algorithm="transpose",
        temp_storage=temp_storage,
    )
    coop._block.store(
        d_out,
        thread_data,
        algorithm="transpose",
        temp_storage=temp_storage,
    )


@cuda.jit
def _block_load_num_valid_oob_kernel(d_in, d_out, valid_items, oob):
    thread_data = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    coop._block.load(
        d_in,
        thread_data,
        valid_items,
        oob,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        algorithm="direct",
    )
    coop._block.store(
        d_out,
        thread_data,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        algorithm="direct",
    )


@cuda.jit
def _block_store_num_valid_kernel(d_in, d_out, valid_items):
    thread_data = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    coop._block.load(
        d_in,
        thread_data,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        algorithm="direct",
    )
    coop._block.store(
        d_out,
        thread_data,
        valid_items,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        algorithm="direct",
    )


def test_block_load_store_temp_storage_round_trip():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _block_load_store_temp_storage_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


def test_block_load_store_exclusive_temp_storage_round_trip():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _block_load_store_exclusive_temp_storage_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


def test_block_load_store_num_valid_oob_default():
    total_items = THREADS * ITEMS_PER_THREAD
    valid_items = np.int32(total_items - 5)
    oob_default = np.int32(-321)
    sentinel = np.int32(-999)
    h_input = np.arange(total_items, dtype=np.int32)
    h_out_all = np.zeros_like(h_input)
    h_out_valid = np.full(total_items, sentinel, dtype=np.int32)

    _block_load_num_valid_oob_kernel[1, THREADS](
        h_input, h_out_all, valid_items, oob_default
    )
    _block_store_num_valid_kernel[1, THREADS](h_input, h_out_valid, valid_items)

    expected_all = np.full(total_items, oob_default, dtype=np.int32)
    expected_all[:valid_items] = h_input[:valid_items]
    expected_valid = np.full(total_items, sentinel, dtype=np.int32)
    expected_valid[:valid_items] = h_input[:valid_items]
    np.testing.assert_array_equal(h_out_all, expected_all)
    np.testing.assert_array_equal(h_out_valid, expected_valid)
