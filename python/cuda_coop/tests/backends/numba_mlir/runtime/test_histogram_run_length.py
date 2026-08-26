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

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_THREADS = 32
_HISTOGRAM_ITEMS_PER_THREAD = 2
_BINS = 17
_DECODED_ITEMS_PER_THREAD = 2
_DECODED_CAPACITY = _THREADS * _DECODED_ITEMS_PER_THREAD


@cuda.jit
def _histogram_kernel(samples, common_output, qualified_output, preserved):
    rank = cuda.threadIdx.x
    common_samples = coop.ThreadData(
        _HISTOGRAM_ITEMS_PER_THREAD,
        dtype=types.int32,
    )
    qualified_samples = numba_coop.ThreadData(
        _HISTOGRAM_ITEMS_PER_THREAD,
        dtype=types.int32,
    )
    for item in range(_HISTOGRAM_ITEMS_PER_THREAD):
        index = rank * _HISTOGRAM_ITEMS_PER_THREAD + item
        value = samples[index]
        common_samples[item] = value
        qualified_samples[item] = value

    common_counts = coop.histogram(
        coop.this_block(),
        common_samples,
        bins=_BINS,
    )
    qualified_counts = numba_coop.histogram(
        numba_coop.this_block(),
        qualified_samples,
        bins=_BINS,
        algorithm="sort",
    )
    common_output[rank] = common_counts[0]
    qualified_output[rank] = qualified_counts[0]
    for item in range(_HISTOGRAM_ITEMS_PER_THREAD):
        index = rank * _HISTOGRAM_ITEMS_PER_THREAD + item
        preserved[index] = qualified_samples[item]


@cuda.jit
def _run_length_decode_kernel(
    values,
    lengths,
    window_offset,
    common_output,
    qualified_output,
    relative_output,
    total_output,
    preserved_values,
    preserved_lengths,
):
    rank = cuda.threadIdx.x
    common_values = coop.ThreadData(1, dtype=types.int32)
    common_lengths = coop.ThreadData(1, dtype=types.uint32)
    qualified_values = numba_coop.ThreadData(1, dtype=types.int32)
    qualified_lengths = numba_coop.ThreadData(1, dtype=types.uint32)
    common_values[0] = values[rank]
    common_lengths[0] = lengths[rank]
    qualified_values[0] = values[rank]
    qualified_lengths[0] = lengths[rank]

    common_decoded = coop.run_length_decode(
        coop.this_block(),
        common_values,
        common_lengths,
        decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
        decoded_window_offset=window_offset,
    )
    relative = numba_coop.ThreadData(
        _DECODED_ITEMS_PER_THREAD,
        dtype=types.uint32,
    )
    total = numba_coop.ThreadData(1, dtype=types.uint32)
    qualified_decoded = numba_coop.run_length_decode(
        numba_coop.this_block(),
        qualified_values,
        qualified_lengths,
        decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
        decoded_window_offset=window_offset,
        relative_offsets=relative,
        total_decoded_size=total,
    )

    base = rank * _DECODED_ITEMS_PER_THREAD
    for item in range(_DECODED_ITEMS_PER_THREAD):
        common_output[base + item] = common_decoded[item]
        qualified_output[base + item] = qualified_decoded[item]
        relative_output[base + item] = relative[item]
    total_output[rank] = total[0]
    preserved_values[rank] = qualified_values[0]
    preserved_lengths[rank] = qualified_lengths[0]


@cuda.jit
def _static_run_length_decode_kernel(values, lengths, output):
    rank = cuda.threadIdx.x
    run_values = numba_coop.ThreadData(1, dtype=types.int32)
    run_lengths = numba_coop.ThreadData(1, dtype=types.uint32)
    run_values[0] = values[rank]
    run_lengths[0] = lengths[rank]
    decoded = numba_coop.run_length_decode(
        numba_coop.this_block(),
        run_values,
        run_lengths,
        decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
        decoded_window_offset=3,
    )
    base = rank * _DECODED_ITEMS_PER_THREAD
    for item in range(_DECODED_ITEMS_PER_THREAD):
        output[base + item] = decoded[item]


def test_common_and_qualified_histograms_match_the_same_oracle() -> None:
    samples = (
        np.arange(_THREADS * _HISTOGRAM_ITEMS_PER_THREAD, dtype=np.int32) * 7 + 3
    ) % _BINS
    common_output = np.full(_THREADS, -1, dtype=np.int32)
    qualified_output = np.full(_THREADS, -1, dtype=np.int32)
    preserved = np.full_like(samples, -1)

    _histogram_kernel[1, _THREADS](
        samples,
        common_output,
        qualified_output,
        preserved,
    )
    cuda.synchronize()

    expected = np.zeros(_THREADS, dtype=np.int32)
    expected[:_BINS] = np.bincount(samples, minlength=_BINS)
    np.testing.assert_array_equal(common_output, expected)
    np.testing.assert_array_equal(qualified_output, expected)
    np.testing.assert_array_equal(preserved, samples)


@pytest.mark.parametrize("offset", [3, 100])
def test_decode_matches_window_oracle_and_masks_beyond_total(offset) -> None:
    values = np.arange(_THREADS, dtype=np.int32) + 20
    lengths = np.zeros(_THREADS, dtype=np.uint32)
    values[:4] = np.asarray([7, 8, 9, 10], dtype=np.int32)
    lengths[:4] = np.asarray([2, 1, 3, 4], dtype=np.uint32)
    common_output = np.full(_DECODED_CAPACITY, -777, dtype=np.int32)
    qualified_output = np.full_like(common_output, -777)
    relative_output = np.zeros(_DECODED_CAPACITY, dtype=np.uint32)
    total_output = np.zeros(_THREADS, dtype=np.uint32)
    preserved_values = np.zeros_like(values)
    preserved_lengths = np.zeros_like(lengths)

    _run_length_decode_kernel[1, _THREADS](
        values,
        lengths,
        np.uint32(offset),
        common_output,
        qualified_output,
        relative_output,
        total_output,
        preserved_values,
        preserved_lengths,
    )
    cuda.synchronize()

    stream = np.asarray([7, 7, 8, 9, 9, 9, 10, 10, 10, 10], dtype=np.int32)
    relative_stream = np.asarray([0, 1, 0, 0, 1, 2, 0, 1, 2, 3], dtype=np.uint32)
    expected = np.zeros(_DECODED_CAPACITY, dtype=np.int32)
    expected_relative = np.full(
        _DECODED_CAPACITY,
        np.iinfo(np.uint32).max,
        dtype=np.uint32,
    )
    valid = max(0, stream.size - offset)
    if valid:
        expected[:valid] = stream[offset:]
        expected_relative[:valid] = relative_stream[offset:]

    np.testing.assert_array_equal(common_output, expected)
    np.testing.assert_array_equal(qualified_output, expected)
    np.testing.assert_array_equal(relative_output, expected_relative)
    np.testing.assert_array_equal(
        total_output,
        np.full(_THREADS, stream.size, dtype=np.uint32),
    )
    np.testing.assert_array_equal(preserved_values, values)
    np.testing.assert_array_equal(preserved_lengths, lengths)


def test_static_decode_offset_matches_window_oracle() -> None:
    values = np.arange(_THREADS, dtype=np.int32) + 20
    lengths = np.zeros(_THREADS, dtype=np.uint32)
    values[:4] = np.asarray([7, 8, 9, 10], dtype=np.int32)
    lengths[:4] = np.asarray([2, 1, 3, 4], dtype=np.uint32)
    output = np.full(_DECODED_CAPACITY, -777, dtype=np.int32)

    _static_run_length_decode_kernel[1, _THREADS](values, lengths, output)
    cuda.synchronize()

    stream = np.asarray([7, 7, 8, 9, 9, 9, 10, 10, 10, 10], dtype=np.int32)
    expected = np.zeros(_DECODED_CAPACITY, dtype=np.int32)
    expected[: stream.size - 3] = stream[3:]
    np.testing.assert_array_equal(output, expected)
