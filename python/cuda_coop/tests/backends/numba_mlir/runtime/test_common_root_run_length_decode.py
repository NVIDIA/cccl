# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Differential GPU evidence for portable block Run Length Decode."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

_BLOCK = (8, 4, 2)
_THREADS = 64
_RUNS_PER_THREAD = 2
_DECODED_ITEMS_PER_THREAD = 3
_RUN_COUNT = _THREADS * _RUNS_PER_THREAD
_DECODED_CAPACITY = _THREADS * _DECODED_ITEMS_PER_THREAD

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _run_length_decode_kernel(
    values,
    lengths,
    window_offset,
    common_output,
    qualified_output,
    relative_output,
    total_output,
    common_values_after,
    common_lengths_after,
    qualified_values_after,
    qualified_lengths_after,
):
    tid = (
        cuda.threadIdx.x
        + cuda.threadIdx.y * cuda.blockDim.x
        + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
    )
    common_values = coop.ThreadData(
        _RUNS_PER_THREAD,
        dtype=values.dtype,
    )
    common_lengths = coop.ThreadData(
        _RUNS_PER_THREAD,
        dtype=lengths.dtype,
    )
    qualified_values = numba_coop.ThreadData(
        _RUNS_PER_THREAD,
        dtype=values.dtype,
    )
    qualified_lengths = numba_coop.ThreadData(
        _RUNS_PER_THREAD,
        dtype=lengths.dtype,
    )
    for item in range(_RUNS_PER_THREAD):
        index = tid * _RUNS_PER_THREAD + item
        value = values[index]
        length = lengths[index]
        common_values[item] = value
        common_lengths[item] = length
        qualified_values[item] = value
        qualified_lengths[item] = length

    common_decoded = coop.run_length_decode(
        coop.this_block(),
        common_values,
        common_lengths,
        decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
        decoded_window_offset=window_offset,
    )
    qualified_relative = numba_coop.ThreadData(
        _DECODED_ITEMS_PER_THREAD,
        dtype=lengths.dtype,
    )
    qualified_total = numba_coop.ThreadData(1, dtype=lengths.dtype)
    qualified_decoded = numba_coop.run_length_decode(
        numba_coop.this_block(),
        qualified_values,
        qualified_lengths,
        decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
        decoded_window_offset=window_offset,
        relative_offsets=qualified_relative,
        total_decoded_size=qualified_total,
    )

    for item in range(_RUNS_PER_THREAD):
        index = tid * _RUNS_PER_THREAD + item
        common_values_after[index] = common_values[item]
        common_lengths_after[index] = common_lengths[item]
        qualified_values_after[index] = qualified_values[item]
        qualified_lengths_after[index] = qualified_lengths[item]
    for item in range(_DECODED_ITEMS_PER_THREAD):
        index = tid * _DECODED_ITEMS_PER_THREAD + item
        common_output[index] = common_decoded[item]
        qualified_output[index] = qualified_decoded[item]
        relative_output[index] = qualified_relative[item]
    total_output[tid] = qualified_total[0]


@cuda.jit
def _qualified_float_tail_kernel(values, lengths, window_offset, output):
    tid = cuda.threadIdx.x
    run_values = numba_coop.ThreadData(1, dtype=values.dtype)
    run_lengths = numba_coop.ThreadData(1, dtype=lengths.dtype)
    run_values[0] = values[tid]
    run_lengths[0] = lengths[tid]
    decoded = numba_coop.run_length_decode(
        numba_coop.this_block(),
        run_values,
        run_lengths,
        decoded_items_per_thread=2,
        decoded_window_offset=window_offset,
    )
    output[tid * 2] = decoded[0]
    output[tid * 2 + 1] = decoded[1]


def _inputs(value_dtype, length_dtype):
    values = (np.arange(_RUN_COUNT, dtype=np.uint64) + 11).astype(value_dtype)
    lengths = np.zeros(_RUN_COUNT, dtype=length_dtype)
    # CUB permits out-of-bounds run slots to be trailing zero-length padding.
    # Zero-length entries interspersed among actual runs are not portable.
    lengths[:7] = np.asarray(
        [2, 1, 3, 4, 1, 2, 3],
        dtype=length_dtype,
    )
    return values, lengths


def _oracle(values, lengths, window_offset):
    decoded = []
    relative = []
    for value, length in zip(values, lengths, strict=True):
        count = int(length)
        decoded.extend([value] * count)
        relative.extend(range(count))

    output = np.zeros(_DECODED_CAPACITY, dtype=values.dtype)
    if np.issubdtype(lengths.dtype, np.unsignedinteger):
        sentinel = np.iinfo(lengths.dtype).max
    else:
        sentinel = -1
    relative_output = np.full(
        _DECODED_CAPACITY,
        sentinel,
        dtype=lengths.dtype,
    )
    available = max(0, len(decoded) - int(window_offset))
    valid = min(_DECODED_CAPACITY, available)
    if valid:
        start = int(window_offset)
        output[:valid] = np.asarray(decoded[start : start + valid], dtype=values.dtype)
        relative_output[:valid] = np.asarray(
            relative[start : start + valid],
            dtype=lengths.dtype,
        )
    return output, relative_output, len(decoded)


@pytest.mark.parametrize(
    ("value_dtype", "length_dtype"),
    [
        (np.uint8, np.uint64),
        (np.int32, np.int32),
        (np.uint32, np.uint32),
        (np.int64, np.int64),
        (np.uint64, np.uint32),
    ],
)
@pytest.mark.evidence_for(
    "group.run_length_decode",
    backend="numba_mlir",
    evidence="runtime",
)
def test_common_and_qualified_run_length_decode_match_window_oracle_and_preserve_inputs(
    value_dtype,
    length_dtype,
):
    values, lengths = _inputs(value_dtype, length_dtype)
    total = int(lengths.astype(np.uint64).sum())

    for offset in (4, total + 3):
        common_output = np.full(_DECODED_CAPACITY, 77, dtype=value_dtype)
        qualified_output = np.full(_DECODED_CAPACITY, 77, dtype=value_dtype)
        relative_output = np.zeros(_DECODED_CAPACITY, dtype=length_dtype)
        total_output = np.zeros(_THREADS, dtype=length_dtype)
        common_values_after = np.zeros_like(values)
        common_lengths_after = np.zeros_like(lengths)
        qualified_values_after = np.zeros_like(values)
        qualified_lengths_after = np.zeros_like(lengths)
        typed_offset = np.asarray(offset, dtype=length_dtype)[()]

        _run_length_decode_kernel[1, _BLOCK](
            values,
            lengths,
            typed_offset,
            common_output,
            qualified_output,
            relative_output,
            total_output,
            common_values_after,
            common_lengths_after,
            qualified_values_after,
            qualified_lengths_after,
        )

        expected, expected_relative, expected_total = _oracle(
            values,
            lengths,
            offset,
        )
        np.testing.assert_array_equal(common_output, qualified_output)
        np.testing.assert_array_equal(common_output, expected)
        np.testing.assert_array_equal(relative_output, expected_relative)
        np.testing.assert_array_equal(
            total_output,
            np.full(_THREADS, expected_total, dtype=length_dtype),
        )
        np.testing.assert_array_equal(common_values_after, values)
        np.testing.assert_array_equal(common_lengths_after, lengths)
        np.testing.assert_array_equal(qualified_values_after, values)
        np.testing.assert_array_equal(qualified_lengths_after, lengths)


def test_qualified_float_run_length_decode_zero_fills_nan_tail():
    threads = 32
    values = np.arange(threads, dtype=np.float32)
    values[0] = np.nan
    lengths = np.zeros(threads, dtype=np.uint32)
    lengths[0] = 1
    output = np.full(threads * 2, np.nan, dtype=np.float32)

    _qualified_float_tail_kernel[1, threads](
        values,
        lengths,
        np.uint32(1),
        output,
    )

    np.testing.assert_array_equal(output, np.zeros_like(output))


@pytest.mark.evidence_for(
    "group.run_length_decode",
    backend="numba_mlir",
    evidence="runtime",
)
def test_uint64_window_near_max_cannot_wrap_into_valid_output():
    values, lengths = _inputs(np.uint8, np.uint64)
    common_output = np.full(_DECODED_CAPACITY, 77, dtype=np.uint8)
    qualified_output = np.full(_DECODED_CAPACITY, 77, dtype=np.uint8)
    relative_output = np.zeros(_DECODED_CAPACITY, dtype=np.uint64)
    total_output = np.zeros(_THREADS, dtype=np.uint64)
    common_values_after = np.zeros_like(values)
    common_lengths_after = np.zeros_like(lengths)
    qualified_values_after = np.zeros_like(values)
    qualified_lengths_after = np.zeros_like(lengths)
    window_offset = np.uint64(np.iinfo(np.uint64).max - 4)

    _run_length_decode_kernel[1, _BLOCK](
        values,
        lengths,
        window_offset,
        common_output,
        qualified_output,
        relative_output,
        total_output,
        common_values_after,
        common_lengths_after,
        qualified_values_after,
        qualified_lengths_after,
    )

    np.testing.assert_array_equal(
        common_output,
        np.zeros(_DECODED_CAPACITY, dtype=np.uint8),
    )
    np.testing.assert_array_equal(common_output, qualified_output)
    np.testing.assert_array_equal(
        relative_output,
        np.full(_DECODED_CAPACITY, np.iinfo(np.uint64).max, dtype=np.uint64),
    )


@pytest.mark.evidence_for(
    "group.run_length_decode",
    backend="numba_mlir",
    evidence="runtime",
)
@pytest.mark.parametrize("length_dtype", [np.int64, np.uint64], ids=["int64", "uint64"])
def test_max_total_preserves_five_item_boundary_window(length_dtype):
    values = np.arange(_RUN_COUNT, dtype=np.uint8)
    values[0] = 42
    lengths = np.zeros(_RUN_COUNT, dtype=length_dtype)
    length_max = np.iinfo(length_dtype).max
    lengths[0] = length_max
    common_output = np.full(_DECODED_CAPACITY, 77, dtype=np.uint8)
    qualified_output = np.full(_DECODED_CAPACITY, 77, dtype=np.uint8)
    relative_output = np.zeros(_DECODED_CAPACITY, dtype=length_dtype)
    total_output = np.zeros(_THREADS, dtype=length_dtype)
    common_values_after = np.zeros_like(values)
    common_lengths_after = np.zeros_like(lengths)
    qualified_values_after = np.zeros_like(values)
    qualified_lengths_after = np.zeros_like(lengths)
    window_offset = length_dtype(length_max - 5)

    _run_length_decode_kernel[1, _BLOCK](
        values,
        lengths,
        window_offset,
        common_output,
        qualified_output,
        relative_output,
        total_output,
        common_values_after,
        common_lengths_after,
        qualified_values_after,
        qualified_lengths_after,
    )

    expected = np.zeros(_DECODED_CAPACITY, dtype=np.uint8)
    expected[:5] = 42
    relative_sentinel = (
        length_max if np.issubdtype(length_dtype, np.unsignedinteger) else -1
    )
    expected_relative = np.full(
        _DECODED_CAPACITY,
        relative_sentinel,
        dtype=length_dtype,
    )
    expected_relative[:5] = np.arange(
        length_max - 5,
        length_max,
        dtype=length_dtype,
    )
    np.testing.assert_array_equal(common_output, expected)
    np.testing.assert_array_equal(common_output, qualified_output)
    np.testing.assert_array_equal(relative_output, expected_relative)
    np.testing.assert_array_equal(
        total_output,
        np.full(_THREADS, length_max, dtype=length_dtype),
    )
