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
from cuda.coop.numba_mlir._block import (
    BlockDiscontinuityType,
    BlockExchangeType,
)
from cuda.coop.numba_mlir._warp import WarpExchangeType

from ..support.runtime import (
    ITEMS_PER_THREAD,
    THREADS,
    _different,
    _striped_to_blocked_reference,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _warp_exchange_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    input_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        input_items[i] = d_in[tid + i * THREADS]

    coop._warp.exchange(
        input_items,
        output_items,
        dtype="int32",
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        warp_exchange_type=WarpExchangeType.StripedToBlocked,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = output_items[i]


@cuda.jit
def _warp_exchange_blocked_to_striped_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    input_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        input_items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._warp.exchange(
        input_items,
        output_items,
        dtype="int32",
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        warp_exchange_type=WarpExchangeType.BlockedToStriped,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid + i * THREADS] = output_items[i]


@cuda.jit
def _warp_exchange_round_trip_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    striped_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    blocked_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    round_trip_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        striped_items[i] = d_in[tid + i * THREADS]

    coop._warp.exchange(
        striped_items,
        blocked_items,
        dtype="int32",
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        warp_exchange_type=WarpExchangeType.StripedToBlocked,
    )
    coop._warp.exchange(
        blocked_items,
        round_trip_items,
        dtype="int32",
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        warp_exchange_type=WarpExchangeType.BlockedToStriped,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid + i * THREADS] = round_trip_items[i]


@cuda.jit
def _warp_exchange_thread_data_temp_storage_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    temp_storage = coop.TempStorage()
    input_items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
    output_items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_out.dtype)

    for i in range(ITEMS_PER_THREAD):
        input_items[i] = d_in[tid + i * THREADS]

    coop._warp.exchange(
        input_items,
        output_items,
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        warp_exchange_type=WarpExchangeType.StripedToBlocked,
        temp_storage=temp_storage,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = output_items[i]


def test_warp_exchange_striped_to_blocked():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _warp_exchange_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, _striped_to_blocked_reference(h_input))


def test_warp_exchange_blocked_to_striped():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _warp_exchange_blocked_to_striped_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


def test_warp_exchange_round_trip_links_both_modes():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _warp_exchange_round_trip_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


def test_warp_exchange_thread_data_temp_storage_striped_to_blocked():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _warp_exchange_thread_data_temp_storage_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, _striped_to_blocked_reference(h_input))


@cuda.jit
def _warp_exchange_scatter_to_striped_kernel(d_in, d_ranks, d_out):
    tid = cuda.threadIdx.x
    input_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    ranks = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        input_items[i] = d_in[idx]
        ranks[i] = d_ranks[idx]

    coop._warp.exchange(
        input_items,
        output_items,
        ranks,
        dtype="int32",
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=THREADS,
        warp_exchange_type=WarpExchangeType.ScatterToStriped,
        offset_dtype="int32",
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid + i * THREADS] = output_items[i]


def test_warp_exchange_scatter_to_striped():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_ranks = np.arange(h_input.size - 1, -1, -1, dtype=np.int32)
    h_output = np.zeros_like(h_input)
    h_expected = np.empty_like(h_input)
    for idx, rank in enumerate(h_ranks):
        h_expected[rank] = h_input[idx]

    _warp_exchange_scatter_to_striped_kernel[1, THREADS](h_input, h_ranks, h_output)

    np.testing.assert_array_equal(h_output, h_expected)


@cuda.jit
def _block_exchange_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    input_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        input_items[i] = d_in[tid + i * THREADS]

    coop._block.exchange(
        input_items,
        output_items,
        block_exchange_type=BlockExchangeType.StripedToBlocked,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = output_items[i]


def test_block_exchange_striped_to_blocked():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _block_exchange_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, _striped_to_blocked_reference(h_input))


@cuda.jit
def _block_exchange_thread_data_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    input_items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
    output_items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_out.dtype)

    for i in range(ITEMS_PER_THREAD):
        input_items[i] = d_in[tid + i * THREADS]

    coop._block.exchange(
        input_items,
        output_items,
        block_exchange_type=BlockExchangeType.StripedToBlocked,
        items_per_thread=ITEMS_PER_THREAD,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = output_items[i]


@cuda.jit
def _block_exchange_blocked_to_striped_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    input_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        input_items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.exchange(
        input_items,
        output_items,
        block_exchange_type=BlockExchangeType.BlockedToStriped,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid + i * THREADS] = output_items[i]


@cuda.jit
def _block_exchange_thread_data_temp_storage_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    temp_storage = coop.TempStorage()
    input_items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
    output_items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_out.dtype)

    for i in range(ITEMS_PER_THREAD):
        input_items[i] = d_in[tid + i * THREADS]

    coop._block.exchange(
        input_items,
        output_items,
        block_exchange_type=BlockExchangeType.StripedToBlocked,
        items_per_thread=ITEMS_PER_THREAD,
        temp_storage=temp_storage,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = output_items[i]


def test_block_exchange_thread_data_striped_to_blocked():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _block_exchange_thread_data_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, _striped_to_blocked_reference(h_input))


def test_block_exchange_thread_data_temp_storage_striped_to_blocked():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _block_exchange_thread_data_temp_storage_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, _striped_to_blocked_reference(h_input))


def test_block_exchange_blocked_to_striped():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _block_exchange_blocked_to_striped_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


@cuda.jit
def _block_exchange_blocked_to_warp_striped_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    input_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        input_items[i] = d_in[tid * ITEMS_PER_THREAD + i]

    coop._block.exchange(
        input_items,
        output_items,
        block_exchange_type=BlockExchangeType.BlockedToWarpStriped,
        dtype="int32",
        threads_per_block=64,
        items_per_thread=ITEMS_PER_THREAD,
    )

    warp_id = tid // 32
    lane_id = tid % 32
    for i in range(ITEMS_PER_THREAD):
        d_out[warp_id * 32 * ITEMS_PER_THREAD + lane_id + i * 32] = output_items[i]


def test_block_exchange_blocked_to_warp_striped():
    h_input = np.arange(64 * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _block_exchange_blocked_to_warp_striped_kernel[1, 64](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


@cuda.jit
def _block_exchange_warp_striped_to_blocked_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    input_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    warp_id = tid // 32
    lane_id = tid % 32
    for i in range(ITEMS_PER_THREAD):
        input_items[i] = d_in[warp_id * 32 * ITEMS_PER_THREAD + lane_id + i * 32]

    coop._block.exchange(
        input_items,
        output_items,
        block_exchange_type=BlockExchangeType.WarpStripedToBlocked,
        dtype="int32",
        threads_per_block=64,
        items_per_thread=ITEMS_PER_THREAD,
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = output_items[i]


def test_block_exchange_warp_striped_to_blocked():
    h_input = np.arange(64 * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _block_exchange_warp_striped_to_blocked_kernel[1, 64](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


@cuda.jit
def _block_exchange_scatter_to_blocked_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    input_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    ranks = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        idx = tid + i * THREADS
        input_items[i] = d_in[idx]
        ranks[i] = idx

    coop._block.exchange(
        input_items,
        output_items,
        ranks,
        block_exchange_type=BlockExchangeType.ScatterToBlocked,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        offset_dtype="int32",
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid * ITEMS_PER_THREAD + i] = output_items[i]


def test_block_exchange_scatter_to_blocked():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _block_exchange_scatter_to_blocked_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)


@cuda.jit
def _block_exchange_scatter_to_striped_flagged_kernel(d_in, d_ranks, d_out):
    tid = cuda.threadIdx.x
    input_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    ranks = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    valid_flags = cuda.local.array(ITEMS_PER_THREAD, cuda.uint8)

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        input_items[i] = d_in[idx]
        ranks[i] = d_ranks[idx]
        valid_flags[i] = 1

    coop._block.exchange(
        input_items,
        output_items,
        ranks,
        valid_flags,
        block_exchange_type=BlockExchangeType.ScatterToStripedFlagged,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        offset_dtype="int32",
        valid_flag_dtype="uint8",
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid + i * THREADS] = output_items[i]


def test_block_exchange_scatter_to_striped_flagged():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_ranks = np.arange(h_input.size - 1, -1, -1, dtype=np.int32)
    h_output = np.zeros_like(h_input)
    h_expected = np.empty_like(h_input)
    for idx, rank in enumerate(h_ranks):
        h_expected[rank] = h_input[idx]

    _block_exchange_scatter_to_striped_flagged_kernel[1, THREADS](
        h_input, h_ranks, h_output
    )

    np.testing.assert_array_equal(h_output, h_expected)


@cuda.jit
def _block_exchange_scatter_to_striped_guarded_kernel(d_in, d_ranks, d_out):
    tid = cuda.threadIdx.x
    input_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    output_items = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)
    ranks = cuda.local.array(ITEMS_PER_THREAD, cuda.int32)

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        input_items[i] = d_in[idx]
        ranks[i] = d_ranks[idx]

    coop._block.exchange(
        input_items,
        output_items,
        ranks,
        block_exchange_type=BlockExchangeType.ScatterToStripedGuarded,
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        offset_dtype="int32",
    )

    for i in range(ITEMS_PER_THREAD):
        d_out[tid + i * THREADS] = output_items[i]


def test_block_exchange_scatter_to_striped_guarded():
    h_input = np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32)
    h_ranks = np.arange(h_input.size - 1, -1, -1, dtype=np.int32)
    h_output = np.zeros_like(h_input)
    h_expected = np.empty_like(h_input)
    for idx, rank in enumerate(h_ranks):
        h_expected[rank] = h_input[idx]

    _block_exchange_scatter_to_striped_guarded_kernel[1, THREADS](
        h_input, h_ranks, h_output
    )

    np.testing.assert_array_equal(h_output, h_expected)


@cuda.jit
def _block_exchange_discontinuity_3d_dim_kernel(d_in, d_out, d_flags):
    tid = (
        cuda.threadIdx.x
        + cuda.threadIdx.y * cuda.blockDim.x
        + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
    )
    threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_in.dtype)
    flags = coop.ThreadData(ITEMS_PER_THREAD, dtype=types.boolean)

    for i in range(ITEMS_PER_THREAD):
        items[i] = d_in[tid + i * threads_per_block]

    coop._block.exchange(
        items,
        block_exchange_type=BlockExchangeType.StripedToBlocked,
    )
    coop._block.discontinuity(
        items,
        flags,
        flag_op=_different,
        block_discontinuity_type=BlockDiscontinuityType.HEADS,
    )

    for i in range(ITEMS_PER_THREAD):
        idx = tid * ITEMS_PER_THREAD + i
        d_out[idx] = items[i]
        d_flags[idx] = flags[i]


def test_block_exchange_discontinuity_3d_dim_inference():
    threads_per_block = (8, 2, 2)
    total_threads = int(np.prod(np.asarray(threads_per_block, dtype=np.int32)))
    total_items = total_threads * ITEMS_PER_THREAD
    h_input = (np.arange(total_items, dtype=np.int32) // 3).astype(np.int32)
    h_output = np.zeros_like(h_input)
    h_flags = np.zeros(total_items, dtype=np.bool_)

    _block_exchange_discontinuity_3d_dim_kernel[1, threads_per_block](
        h_input, h_output, h_flags
    )

    expected_flags = np.zeros_like(h_flags)
    expected_flags[0] = True
    expected_flags[1:] = h_input[1:] != h_input[:-1]
    np.testing.assert_array_equal(h_output, h_input)
    np.testing.assert_array_equal(h_flags, expected_flags)
