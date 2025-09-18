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


def random_array(size, dtype, max_value=None) -> np.typing.NDArray:
    rng = np.random.default_rng()
    if np.isdtype(dtype, "integral"):
        if max_value is None:
            max_value = np.iinfo(dtype).max
        return rng.integers(max_value, size=size, dtype=dtype)
    elif np.isdtype(dtype, "real floating"):
        if dtype == np.float16:  # Cannot generate float16 directly
            return rng.random(size=size, dtype=np.float32).astype(dtype)
        return rng.random(size=size, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


three_way_partition_params = [
    (dt, 2**log_size) for dt in DTYPE_LIST for log_size in [2, 4, 6, 8, 10, 16, 20]
]


def _host_three_way_partition(h_in: np.ndarray, less_than_op, greater_equal_op):
    # Vectorize ops to produce boolean masks
    first_mask = np.vectorize(less_than_op, otypes=[np.uint8])(h_in).astype(bool)
    remaining = h_in[~first_mask]
    second_mask = np.vectorize(greater_equal_op, otypes=[np.uint8])(remaining).astype(
        bool
    )

    first_part = h_in[first_mask]
    second_part = remaining[second_mask]
    unselected = remaining[~second_mask]

    return (
        first_part,
        second_part,
        unselected,
        np.int64(first_part.size),
        np.int64(second_part.size),
        np.int64(unselected.size),
    )


@pytest.mark.parametrize("dtype,num_items", three_way_partition_params)
def test_three_way_partition_basic(dtype, num_items):
    h_in = random_array(num_items, dtype, max_value=100)

    rng = np.random.default_rng()
    if np.issubdtype(dtype, np.floating):
        thr = dtype(rng.random())
    else:
        thr = dtype(rng.integers(0, 100))

    def less_than_op(x, t=thr):
        return np.uint8(x < t)

    def greater_equal_op(x, t=thr):
        return np.uint8(x >= t)

    d_in = cp.asarray(h_in)
    d_first = cp.empty_like(d_in)
    d_second = cp.empty_like(d_in)
    d_unselected = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.int64)

    parallel.three_way_partition(
        d_in,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
    )

    num_selected = d_num_selected.get()
    got_first = d_first.get()[: int(num_selected[0])]
    got_second = d_second.get()[: int(num_selected[1])]
    got_unselected = d_unselected.get()[
        : int(num_items) - int(num_selected[0]) - int(num_selected[1])
    ]

    expected_first, expected_second, expected_unselected, n1, n2, n3 = (
        _host_three_way_partition(h_in, less_than_op, greater_equal_op)
    )

    np.testing.assert_array_equal(got_first, expected_first)
    np.testing.assert_array_equal(got_second, expected_second)
    np.testing.assert_array_equal(got_unselected, expected_unselected)
    assert num_selected[0] == n1 and num_selected[1] == n2


def test_three_way_partition_empty():
    dtype = np.int32
    d_in = cp.empty(0, dtype=dtype)
    d_first = cp.empty(0, dtype=dtype)
    d_second = cp.empty(0, dtype=dtype)
    d_unselected = cp.empty(0, dtype=dtype)
    d_num_selected = cp.zeros(2, dtype=np.int64)

    # Random thresholds, though they don't affect empty input
    rng = np.random.default_rng()
    thr0 = dtype(rng.integers(0, 100))
    thr1 = dtype(rng.integers(0, 100))

    def less_than_op(x, t=thr0):
        return np.uint8(x < t)

    def greater_equal_op(x, t=thr1):
        return np.uint8(x >= t)

    parallel.three_way_partition(
        d_in,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
    )

    np.testing.assert_array_equal(d_num_selected.get(), np.array([0, 0]))


def test_three_way_partition_with_iterators():
    dtype = np.int32
    num_items = 10_000
    rng = np.random.default_rng()
    h_in = random_array(num_items, dtype, max_value=100)
    thr = dtype(rng.integers(0, 100))

    def less_than_op(x, t=thr):
        return np.uint8(x < t)

    def greater_equal_op(x, t=thr):
        return np.uint8(x >= t)

    expected_first, expected_second, expected_unselected, _, _, _ = (
        _host_three_way_partition(h_in, less_than_op, greater_equal_op)
    )

    d_in = cp.asarray(h_in)
    in_it = parallel.CacheModifiedInputIterator(d_in, modifier="stream")

    d_first = cp.empty_like(d_in)
    d_second = cp.empty_like(d_in)
    d_unselected = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.int64)

    parallel.three_way_partition(
        in_it,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
    )

    num_selected = d_num_selected.get()
    got_first = d_first.get()[: int(num_selected[0])]
    got_second = d_second.get()[: int(num_selected[1])]
    got_unselected = d_unselected.get()[
        : int(num_items) - int(num_selected[0]) - int(num_selected[1])
    ]

    np.testing.assert_array_equal(got_first, expected_first)
    np.testing.assert_array_equal(got_second, expected_second)
    np.testing.assert_array_equal(got_unselected, expected_unselected)


def test_three_way_partition_struct_type():
    @parallel.gpu_struct
    class pair_type:
        a: np.int32
        b: np.uint64

    rng = np.random.default_rng()
    comparison_value = np.int32(rng.integers(0, 100))

    def less_than_op(x: pair_type, t=comparison_value):
        return np.uint8(x.a < t)

    def greater_equal_op(x: pair_type, t=comparison_value):
        return np.uint8(x.a >= t)

    num_items = 20_000
    a_vals = np.random.default_rng(0).integers(0, 100, size=num_items, dtype=np.int32)
    b_vals = a_vals.astype(np.uint64)

    h_in = np.empty(num_items, dtype=pair_type.dtype)
    h_in["a"] = a_vals
    h_in["b"] = b_vals

    expected_first_mask = a_vals < comparison_value
    remaining_mask = ~expected_first_mask
    expected_second_mask = a_vals[remaining_mask] >= comparison_value

    expected_first = h_in[expected_first_mask]
    expected_second = h_in[remaining_mask][expected_second_mask]
    expected_unselected = h_in[remaining_mask][~expected_second_mask]

    d_in = cp.asarray(h_in).view(pair_type.dtype)
    d_first = cp.empty_like(d_in)
    d_second = cp.empty_like(d_in)
    d_unselected = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.int64)

    parallel.three_way_partition(
        d_in,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
    )

    num_selected = d_num_selected.get()
    got_first = d_first.get()[: int(num_selected[0])]
    got_second = d_second.get()[: int(num_selected[1])]
    got_unselected = d_unselected.get()[
        : int(num_items) - int(num_selected[0]) - int(num_selected[1])
    ]

    np.testing.assert_array_equal(got_first, expected_first)
    np.testing.assert_array_equal(got_second, expected_second)
    np.testing.assert_array_equal(got_unselected, expected_unselected)


def test_three_way_partition_with_stream(cuda_stream):
    dtype = np.int32
    num_items = 50_000
    rng = np.random.default_rng()
    h_in = random_array(num_items, dtype, max_value=100)
    thr = dtype(rng.integers(0, 100))

    def less_than_op(x, t=thr):
        return np.uint8(x < t)

    def greater_equal_op(x, t=thr):
        return np.uint8(x >= t)

    expected_first, expected_second, expected_unselected, _, _, _ = (
        _host_three_way_partition(h_in, less_than_op, greater_equal_op)
    )

    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)
    with cp_stream:
        d_in = cp.asarray(h_in)
        d_first = cp.empty_like(d_in)
        d_second = cp.empty_like(d_in)
        d_unselected = cp.empty_like(d_in)
        d_num_selected = cp.empty(2, dtype=np.int64)

    parallel.three_way_partition(
        d_in,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
        stream=cuda_stream,
    )

    num_selected = d_num_selected.get()
    got_first = d_first.get()[: int(num_selected[0])]
    got_second = d_second.get()[: int(num_selected[1])]
    got_unselected = d_unselected.get()[
        : int(num_items) - int(num_selected[0]) - int(num_selected[1])
    ]

    np.testing.assert_array_equal(got_first, expected_first)
    np.testing.assert_array_equal(got_second, expected_second)
    np.testing.assert_array_equal(got_unselected, expected_unselected)
