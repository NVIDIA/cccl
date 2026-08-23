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
from cuda.coop.numba_mlir._single_phase_rewrites import (
    CoopSinglePhaseRewriteError,
)

from ..support.runtime import (
    THREADS,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _block_run_length_fractional_count_kernel(run_values, run_lengths):
    run_values_local = coop.local.array(1, dtype=cuda.uint32)
    run_lengths_local = coop.local.array(1, dtype=cuda.uint32)
    total_decoded_size = coop.local.array(1, dtype=cuda.uint32)
    decoded_items = coop.local.array(1, dtype=cuda.uint32)
    run_values_local[0] = run_values[cuda.threadIdx.x]
    run_lengths_local[0] = run_lengths[cuda.threadIdx.x]
    total_decoded_size[0] = 0
    run_length = coop._block.run_length(
        run_values_local,
        run_lengths_local,
        1.5,
        1,
        total_decoded_size,
        decoded_offset_dtype="uint32",
    )
    run_length.decode(decoded_items, run_lengths_local[0] - run_lengths_local[0])


def test_block_run_length_decode_rejects_fractional_single_phase_count():
    h_run_values = np.arange(THREADS, dtype=np.uint32)
    h_run_lengths = np.ones(THREADS, dtype=np.uint32)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="runs_per_thread must be a compile-time positive integer",
    ):
        _block_run_length_fractional_count_kernel[1, THREADS](
            h_run_values,
            h_run_lengths,
        )


@cuda.jit
def _block_run_length_kernel(run_values, run_lengths, decoded_items_out, size_out):
    tid = cuda.threadIdx.x
    runs_per_thread = 1
    decoded_items_per_thread = 2

    run_values_local = coop.local.array(runs_per_thread, dtype=cuda.uint32)
    run_lengths_local = coop.local.array(runs_per_thread, dtype=cuda.uint32)
    total_decoded_size = coop.local.array(1, dtype=cuda.uint32)
    decoded_items = coop.local.array(decoded_items_per_thread, dtype=cuda.uint32)

    run_values_local[0] = run_values[tid]
    run_lengths_local[0] = run_lengths[tid]
    total_decoded_size[0] = 0

    run_length = coop._block.run_length(
        run_values_local,
        run_lengths_local,
        runs_per_thread,
        decoded_items_per_thread,
        total_decoded_size,
        decoded_offset_dtype="uint32",
    )
    decoded_window_offset = run_lengths_local[0] - run_lengths_local[0]
    run_length.decode(decoded_items, decoded_window_offset)

    base = tid * decoded_items_per_thread
    for i in range(decoded_items_per_thread):
        decoded_items_out[base + i] = decoded_items[i]

    if tid == 0:
        size_out[0] = total_decoded_size[0]


@cuda.jit
def _block_run_length_offsets_kernel(
    run_values, run_lengths, decoded_items_out, relative_offsets_out, size_out
):
    tid = cuda.threadIdx.x
    runs_per_thread = 1
    decoded_items_per_thread = 2

    run_values_local = coop.local.array(runs_per_thread, dtype=cuda.uint32)
    run_lengths_local = coop.local.array(runs_per_thread, dtype=cuda.uint32)
    total_decoded_size = coop.local.array(1, dtype=cuda.uint32)
    decoded_items = coop.local.array(decoded_items_per_thread, dtype=cuda.uint32)
    relative_offsets = coop.local.array(decoded_items_per_thread, dtype=cuda.uint32)

    run_values_local[0] = run_values[tid]
    run_lengths_local[0] = run_lengths[tid]
    total_decoded_size[0] = 0

    run_length = coop._block.run_length(
        run_values_local,
        run_lengths_local,
        runs_per_thread,
        decoded_items_per_thread,
        total_decoded_size,
        decoded_offset_dtype="uint32",
    )
    decoded_window_offset = run_lengths_local[0] - run_lengths_local[0]
    run_length.decode(decoded_items, decoded_window_offset, relative_offsets)

    base = tid * decoded_items_per_thread
    for i in range(decoded_items_per_thread):
        decoded_items_out[base + i] = decoded_items[i]
        relative_offsets_out[base + i] = relative_offsets[i]

    if tid == 0:
        size_out[0] = total_decoded_size[0]


@cuda.jit
def _block_run_length_temp_storage_kernel(
    run_values, run_lengths, decoded_items_out, relative_offsets_out, size_out
):
    tid = cuda.threadIdx.x
    runs_per_thread = 2
    decoded_items_per_thread = 4

    run_values_local = coop.local.array(runs_per_thread, dtype=cuda.uint32)
    run_lengths_local = coop.local.array(runs_per_thread, dtype=cuda.uint32)
    total_decoded_size = coop.local.array(1, dtype=cuda.uint32)
    decoded_items = coop.local.array(decoded_items_per_thread, dtype=cuda.uint32)
    relative_offsets = coop.local.array(decoded_items_per_thread, dtype=cuda.uint32)
    temp_storage = coop.TempStorage()

    coop._block.load(
        run_values,
        run_values_local,
        items_per_thread=runs_per_thread,
        algorithm="direct",
    )
    coop._block.load(
        run_lengths,
        run_lengths_local,
        items_per_thread=runs_per_thread,
        algorithm="direct",
    )
    total_decoded_size[0] = 0

    run_length = coop._block.run_length(
        run_values_local,
        run_lengths_local,
        runs_per_thread,
        decoded_items_per_thread,
        total_decoded_size,
        decoded_offset_dtype="uint32",
        temp_storage=temp_storage,
    )
    decoded_window_offset = run_lengths_local[0] - run_lengths_local[0]
    run_length.decode(decoded_items, decoded_window_offset, relative_offsets)

    base = tid * decoded_items_per_thread
    for i in range(decoded_items_per_thread):
        decoded_items_out[base + i] = decoded_items[i]
        relative_offsets_out[base + i] = relative_offsets[i]

    if tid == 0:
        size_out[0] = total_decoded_size[0]


def _expected_run_length_decode(run_values, run_lengths):
    decoded_items = np.repeat(run_values, run_lengths.astype(np.int64))
    relative_offsets = np.concatenate(
        [np.arange(int(length), dtype=run_lengths.dtype) for length in run_lengths]
    )
    return decoded_items, relative_offsets


def test_block_run_length_decode():
    h_run_values = np.arange(THREADS, dtype=np.uint32)
    h_run_lengths = np.full(THREADS, 2, dtype=np.uint32)
    h_decoded_items = np.zeros(THREADS * 2, dtype=np.uint32)
    h_size = np.zeros(1, dtype=np.uint32)

    _block_run_length_kernel[1, THREADS](
        h_run_values, h_run_lengths, h_decoded_items, h_size
    )

    expected_items = np.repeat(h_run_values, 2)
    np.testing.assert_array_equal(h_decoded_items, expected_items)
    np.testing.assert_array_equal(h_size, np.asarray([THREADS * 2], dtype=np.uint32))


def test_block_run_length_decode_relative_offsets():
    h_run_values = np.arange(THREADS, dtype=np.uint32)
    h_run_lengths = np.full(THREADS, 2, dtype=np.uint32)
    h_decoded_items = np.zeros(THREADS * 2, dtype=np.uint32)
    h_relative_offsets = np.zeros(THREADS * 2, dtype=np.uint32)
    h_size = np.zeros(1, dtype=np.uint32)

    _block_run_length_offsets_kernel[1, THREADS](
        h_run_values,
        h_run_lengths,
        h_decoded_items,
        h_relative_offsets,
        h_size,
    )

    expected_items = np.repeat(h_run_values, 2)
    expected_offsets = np.tile(np.arange(2, dtype=np.uint32), THREADS)
    np.testing.assert_array_equal(h_decoded_items, expected_items)
    np.testing.assert_array_equal(h_relative_offsets, expected_offsets)
    np.testing.assert_array_equal(h_size, np.asarray([THREADS * 2], dtype=np.uint32))


def test_block_run_length_decode_two_phase_parent():
    runs_per_thread = 2
    decoded_items_per_thread = 4
    total_runs = THREADS * runs_per_thread
    window_size = THREADS * decoded_items_per_thread
    run_length_instance = coop._block.run_length(
        types.uint32,
        THREADS,
        runs_per_thread,
        decoded_items_per_thread,
        decoded_offset_dtype=types.uint32,
    )

    @cuda.jit
    def kernel(
        run_values,
        run_lengths,
        decoded_items_out,
        relative_offsets_out,
        size_out,
    ):
        tid = cuda.threadIdx.x
        run_values_local = coop.local.array(runs_per_thread, dtype=run_values.dtype)
        run_lengths_local = coop.local.array(runs_per_thread, dtype=run_lengths.dtype)
        total_decoded_size = coop.local.array(1, dtype=types.uint32)
        decoded_items = coop.local.array(
            decoded_items_per_thread, dtype=run_values.dtype
        )
        relative_offsets = coop.local.array(
            decoded_items_per_thread, dtype=run_lengths.dtype
        )

        for item in range(runs_per_thread):
            idx = tid * runs_per_thread + item
            run_values_local[item] = run_values[idx]
            run_lengths_local[item] = run_lengths[idx]
        total_decoded_size[0] = 0

        run_length = run_length_instance(
            run_values_local,
            run_lengths_local,
            runs_per_thread,
            decoded_items_per_thread,
            total_decoded_size,
            decoded_offset_dtype=types.uint32,
        )
        run_length.decode(decoded_items, 0, relative_offsets)

        base = tid * decoded_items_per_thread
        for item in range(decoded_items_per_thread):
            decoded_items_out[base + item] = decoded_items[item]
            relative_offsets_out[base + item] = relative_offsets[item]
        if tid == 0:
            size_out[0] = total_decoded_size[0]

    h_run_values = np.arange(total_runs, dtype=np.uint32)
    h_run_lengths = (np.arange(total_runs, dtype=np.uint32) % 3) + np.uint32(1)
    h_run_lengths[-1] += np.uint32(window_size - int(h_run_lengths.sum()))
    h_decoded_items = np.zeros(window_size, dtype=np.uint32)
    h_relative_offsets = np.zeros(window_size, dtype=np.uint32)
    h_size = np.zeros(1, dtype=np.uint32)

    kernel[1, THREADS](
        h_run_values,
        h_run_lengths,
        h_decoded_items,
        h_relative_offsets,
        h_size,
    )

    expected_items, expected_offsets = _expected_run_length_decode(
        h_run_values, h_run_lengths
    )
    np.testing.assert_array_equal(h_decoded_items, expected_items)
    np.testing.assert_array_equal(h_relative_offsets, expected_offsets)
    np.testing.assert_array_equal(h_size, np.asarray([window_size], dtype=np.uint32))


def test_block_run_length_decode_two_phase_temp_storage():
    runs_per_thread = 2
    decoded_items_per_thread = 4
    total_runs = THREADS * runs_per_thread
    window_size = THREADS * decoded_items_per_thread
    run_length_instance = coop._block.run_length(
        types.uint32,
        THREADS,
        runs_per_thread,
        decoded_items_per_thread,
        decoded_offset_dtype=types.uint32,
    )
    decode_invocable = run_length_instance.decode(
        types.uint32,
        types.uint32,
        with_relative_offsets=True,
        with_decoded_window_offset=True,
        relative_offset_dtype=types.uint32,
    )
    temp_storage_bytes = int(run_length_instance.temp_storage_bytes)
    temp_storage_alignment = int(run_length_instance.temp_storage_alignment)
    assert temp_storage_bytes == int(decode_invocable.temp_storage_bytes)
    assert temp_storage_alignment == int(decode_invocable.temp_storage_alignment)

    @cuda.jit
    def kernel(
        run_values,
        run_lengths,
        decoded_items_out,
        relative_offsets_out,
        size_out,
    ):
        tid = cuda.threadIdx.x
        run_values_local = coop.local.array(runs_per_thread, dtype=run_values.dtype)
        run_lengths_local = coop.local.array(runs_per_thread, dtype=run_lengths.dtype)
        total_decoded_size = coop.local.array(1, dtype=types.uint32)
        decoded_items = coop.local.array(
            decoded_items_per_thread, dtype=run_values.dtype
        )
        relative_offsets = coop.local.array(
            decoded_items_per_thread, dtype=run_lengths.dtype
        )
        temp_storage = coop.TempStorage(temp_storage_bytes, temp_storage_alignment)

        for item in range(runs_per_thread):
            idx = tid * runs_per_thread + item
            run_values_local[item] = run_values[idx]
            run_lengths_local[item] = run_lengths[idx]
        total_decoded_size[0] = 0

        run_length = run_length_instance(
            run_values_local,
            run_lengths_local,
            runs_per_thread,
            decoded_items_per_thread,
            total_decoded_size,
            decoded_offset_dtype=types.uint32,
            temp_storage=temp_storage,
        )
        run_length.decode(decoded_items, 0, relative_offsets)

        base = tid * decoded_items_per_thread
        for item in range(decoded_items_per_thread):
            decoded_items_out[base + item] = decoded_items[item]
            relative_offsets_out[base + item] = relative_offsets[item]
        if tid == 0:
            size_out[0] = total_decoded_size[0]

    h_run_values = np.arange(total_runs, dtype=np.uint32)
    h_run_lengths = (np.arange(total_runs, dtype=np.uint32) % 3) + np.uint32(1)
    h_run_lengths[-1] += np.uint32(window_size - int(h_run_lengths.sum()))
    h_decoded_items = np.zeros(window_size, dtype=np.uint32)
    h_relative_offsets = np.zeros(window_size, dtype=np.uint32)
    h_size = np.zeros(1, dtype=np.uint32)

    kernel[1, THREADS](
        h_run_values,
        h_run_lengths,
        h_decoded_items,
        h_relative_offsets,
        h_size,
    )

    expected_items, expected_offsets = _expected_run_length_decode(
        h_run_values, h_run_lengths
    )
    np.testing.assert_array_equal(h_decoded_items, expected_items)
    np.testing.assert_array_equal(h_relative_offsets, expected_offsets)
    np.testing.assert_array_equal(h_size, np.asarray([window_size], dtype=np.uint32))


def test_block_run_length_decode_temp_storage():
    runs_per_thread = 2
    decoded_items_per_thread = 4
    total_runs = THREADS * runs_per_thread
    window_size = THREADS * decoded_items_per_thread

    h_run_values = np.arange(total_runs, dtype=np.uint32)
    h_run_lengths = (np.arange(total_runs, dtype=np.uint32) % 3) + np.uint32(1)
    h_run_lengths[-1] += np.uint32(window_size - int(h_run_lengths.sum()))
    h_decoded_items = np.zeros(window_size, dtype=np.uint32)
    h_relative_offsets = np.zeros(window_size, dtype=np.uint32)
    h_size = np.zeros(1, dtype=np.uint32)

    _block_run_length_temp_storage_kernel[1, THREADS](
        h_run_values,
        h_run_lengths,
        h_decoded_items,
        h_relative_offsets,
        h_size,
    )

    expected_items, expected_offsets = _expected_run_length_decode(
        h_run_values, h_run_lengths
    )
    np.testing.assert_array_equal(h_decoded_items, expected_items)
    np.testing.assert_array_equal(h_relative_offsets, expected_offsets)
    np.testing.assert_array_equal(h_size, np.asarray([window_size], dtype=np.uint32))


def test_block_run_length_decode_direct_invocable():
    runs_per_thread = 2
    decoded_items_per_thread = 4
    total_runs = THREADS * runs_per_thread
    window_size = THREADS * decoded_items_per_thread
    run_length_factory = coop._block.run_length(
        types.uint32,
        THREADS,
        runs_per_thread,
        decoded_items_per_thread,
        decoded_offset_dtype=types.uint32,
    )
    decode_invocable = run_length_factory.decode(
        types.uint32,
        types.uint32,
        with_relative_offsets=True,
        with_decoded_window_offset=True,
        relative_offset_dtype=types.uint32,
    )
    temp_storage_bytes = int(decode_invocable.temp_storage_bytes)
    temp_storage_alignment = int(decode_invocable.temp_storage_alignment)

    @cuda.jit
    def kernel(
        run_values,
        run_lengths,
        decoded_items_out,
        relative_offsets_out,
        size_out,
    ):
        tid = cuda.threadIdx.x
        run_values_local = coop.local.array(runs_per_thread, dtype=types.uint32)
        run_lengths_local = coop.local.array(runs_per_thread, dtype=types.uint32)
        total_decoded_size = coop.local.array(1, dtype=types.uint32)
        decoded_items = coop.local.array(decoded_items_per_thread, dtype=types.uint32)
        relative_offsets = coop.local.array(
            decoded_items_per_thread, dtype=types.uint32
        )
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )

        for item in range(runs_per_thread):
            idx = tid * runs_per_thread + item
            run_values_local[item] = run_values[idx]
            run_lengths_local[item] = run_lengths[idx]
        total_decoded_size[0] = 0

        decode_invocable(
            run_values_local,
            run_lengths_local,
            total_decoded_size,
            decoded_items,
            relative_offsets,
            types.uint32(0),
            temp_storage=temp_storage,
        )

        base = tid * decoded_items_per_thread
        for item in range(decoded_items_per_thread):
            decoded_items_out[base + item] = decoded_items[item]
            relative_offsets_out[base + item] = relative_offsets[item]
        if tid == 0:
            size_out[0] = total_decoded_size[0]

    h_run_values = np.arange(total_runs, dtype=np.uint32)
    h_run_lengths = (np.arange(total_runs, dtype=np.uint32) % 3) + np.uint32(1)
    h_run_lengths[-1] += np.uint32(window_size - int(h_run_lengths.sum()))
    h_decoded_items = np.zeros(window_size, dtype=np.uint32)
    h_relative_offsets = np.zeros(window_size, dtype=np.uint32)
    h_size = np.zeros(1, dtype=np.uint32)

    kernel[1, THREADS](
        h_run_values,
        h_run_lengths,
        h_decoded_items,
        h_relative_offsets,
        h_size,
    )

    expected_items, expected_offsets = _expected_run_length_decode(
        h_run_values, h_run_lengths
    )
    np.testing.assert_array_equal(h_decoded_items, expected_items[:window_size])
    np.testing.assert_array_equal(h_relative_offsets, expected_offsets[:window_size])
    np.testing.assert_array_equal(
        h_size,
        np.asarray([window_size], dtype=np.uint32),
    )
