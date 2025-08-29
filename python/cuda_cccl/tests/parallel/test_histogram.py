# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import pytest

import cuda.cccl.parallel.experimental as parallel

DTYPE_LIST = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
]


def get_mark(dt, log_size):
    if log_size + np.log2(np.dtype(dt).itemsize) < 21:
        return tuple()
    return pytest.mark.large


def type_to_problem_sizes(dtype):
    if dtype in [np.uint8, np.int8]:
        return [8, 10, 12, 14]
    elif dtype in [np.float16, np.uint16, np.int16]:
        return [10, 12, 14, 16]
    elif dtype in [np.uint32, np.int32, np.float32]:
        return [12, 14, 16, 18]
    elif dtype in [np.uint64, np.int64, np.float64]:
        return [12, 14, 16, 18]
    else:
        raise ValueError("Unsupported dtype")


dtype_size_pairs = [
    pytest.param(dt, 2**log_size, marks=get_mark(dt, log_size))
    for dt in DTYPE_LIST
    for log_size in type_to_problem_sizes(dt)
]


def random_int_array(size, dtype):
    if np.issubdtype(dtype, np.integer):
        if dtype in [np.uint8, np.int8]:
            max_val = 126
        else:
            max_val = 1024
        return np.random.randint(0, max_val, size=size).astype(dtype)
    else:
        # For floating point, generate values in similar range
        return (np.random.random(size) * 1024).astype(dtype)


def compute_reference_histogram(h_samples, num_levels, lower_level, upper_level):
    # Filter samples within range [lower_level, upper_level)
    valid_mask = (h_samples >= lower_level) & (h_samples < upper_level)
    valid_samples = h_samples[valid_mask]

    if len(valid_samples) == 0:
        return np.zeros(num_levels - 1, dtype=np.int32)

    # Compute bin indices for valid samples
    bin_indices = (
        (valid_samples - lower_level) * (num_levels - 1) / (upper_level - lower_level)
    ).astype(int)

    # Ensure indices are within valid range [0, num_levels-2]
    bin_indices = np.clip(bin_indices, 0, num_levels - 2)

    # Use bincount to get histogram
    histogram = np.bincount(bin_indices, minlength=num_levels - 1)

    return histogram.astype(np.int32)


@pytest.mark.parametrize("dtype,num_samples", dtype_size_pairs)
def test_device_histogram_basic_use(dtype, num_samples):
    if dtype in [np.uint8, np.int8]:
        max_level = 126.0
        max_level_count = 127
    else:
        max_level = 1024.0
        max_level_count = 1025

    num_levels = max_level_count
    lower_level = np.float64(0.0)
    upper_level = np.float64(max_level)

    h_samples = random_int_array(num_samples, dtype)
    d_samples = cp.asarray(h_samples)

    d_histogram = cp.zeros(num_levels - 1, dtype=np.int32)

    parallel.histogram_even(
        d_samples,
        d_histogram,
        num_levels,
        lower_level,
        upper_level,
        num_samples,
    )

    h_expected = compute_reference_histogram(
        h_samples, num_levels, lower_level, upper_level
    )
    h_result = cp.asnumpy(d_histogram)

    np.testing.assert_array_equal(h_result, h_expected)


@pytest.mark.no_verify_sass(reason="LDL/STL instructions emitted for this test.")
def test_device_histogram_sample_iterator():
    max_level_count = 1025
    num_levels = max_level_count
    num_bins = num_levels - 1

    samples_per_bin = 10
    adjusted_total_samples = num_bins * samples_per_bin

    counting_it = parallel.CountingIterator(np.int32(0))

    d_histogram = cp.zeros(num_levels - 1, dtype=np.int32)

    # Set up levels so that values 0 to adjusted_total_samples-1 are evenly distributed
    lower_level = np.float64(0.0)
    upper_level = np.float64(adjusted_total_samples)

    parallel.histogram_even(
        counting_it,
        d_histogram,
        num_levels,
        lower_level,
        upper_level,
        adjusted_total_samples,
    )

    # Each bin should have exactly samples_per_bin elements
    h_expected = np.full(num_bins, samples_per_bin, dtype=np.int32)
    h_result = cp.asnumpy(d_histogram)

    np.testing.assert_array_equal(h_result, h_expected)


def test_device_histogram_single_sample():
    h_samples = np.array([5.0], dtype=np.float32)
    d_samples = cp.asarray(h_samples)

    num_levels = 5
    lower_level = np.float64(0.0)
    upper_level = np.float64(10.0)

    d_histogram = cp.zeros(num_levels - 1, dtype=np.int32)

    parallel.histogram_even(
        d_samples, d_histogram, num_levels, lower_level, upper_level, 1
    )

    # Sample 5.0 should go into bin 2 (bins: [0,2.5), [2.5,5), [5,7.5), [7.5,10))
    h_expected = np.array([0, 0, 1, 0], dtype=np.int32)
    h_result = cp.asnumpy(d_histogram)

    np.testing.assert_array_equal(h_result, h_expected)


def test_device_histogram_out_of_range():
    h_samples = np.array([-1.0, 0.5, 5.5, 10.5, 15.0], dtype=np.float32)
    d_samples = cp.asarray(h_samples)

    num_levels = 3  # 2 bins: [0,5), [5,10)
    lower_level = np.float64(0.0)
    upper_level = np.float64(10.0)

    d_histogram = cp.zeros(num_levels - 1, dtype=np.int32)

    parallel.histogram_even(
        d_samples,
        d_histogram,
        num_levels,
        lower_level,
        upper_level,
        len(h_samples),
    )

    # Only 0.5 (bin 0) and 5.5 (bin 1) should be counted
    # -1.0, 10.5, and 15.0 are out of range
    h_expected = np.array([1, 1], dtype=np.int32)
    h_result = cp.asnumpy(d_histogram)

    np.testing.assert_array_equal(h_result, h_expected)


def test_device_histogram_with_stream(cuda_stream):
    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)

    h_samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    d_samples = cp.asarray(h_samples)

    num_levels = 5  # 4 bins: [0,2), [2,4), [4,6), [6,8)
    lower_level = np.float64(0.0)
    upper_level = np.float64(8.0)

    d_histogram = cp.zeros(num_levels - 1, dtype=np.int32)

    with cp_stream:
        d_samples = cp.asarray(h_samples)
        d_histogram = cp.zeros(num_levels - 1, dtype=np.int32)

    parallel.histogram_even(
        d_samples,
        d_histogram,
        num_levels,
        lower_level,
        upper_level,
        len(h_samples),
        stream=cuda_stream,
    )

    with cp_stream:
        h_result = cp.asnumpy(d_histogram)

    # Expected: bin 0: [1.0, 2.0), bin 1: [2.0, 4.0), bin 2: [4.0, 6.0), bin 3: [6.0, 8.0)
    # Values: 1.0->bin0, 2.0->bin1, 3.0->bin1, 4.0->bin2, 5.0->bin2, 6.0->bin3, 7.0->bin3, 8.0->out_of_range
    h_expected = np.array([1, 2, 2, 2], dtype=np.int32)

    np.testing.assert_array_equal(h_result, h_expected)


@pytest.mark.no_verify_sass(reason="LDL/STL instructions emitted for this test.")
def test_device_histogram_with_constant_iterator():
    constant_it = parallel.ConstantIterator(np.float32(3.0))

    num_samples = 10
    num_levels = 5  # 4 bins: [0,2), [2,4), [4,6), [6,8)
    lower_level = np.float64(0.0)
    upper_level = np.float64(8.0)

    d_histogram = cp.zeros(num_levels - 1, dtype=np.int32)

    parallel.histogram_even(
        constant_it,
        d_histogram,
        num_levels,
        lower_level,
        upper_level,
        num_samples,
    )

    h_result = cp.asnumpy(d_histogram)

    # Expected: All 10 samples have value 3.0, which falls in bin 1 [2,4)
    h_expected = np.array([0, 10, 0, 0], dtype=np.int32)

    np.testing.assert_array_equal(h_result, h_expected)
