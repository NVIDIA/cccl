# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import pytest

import cuda.compute
from cuda.compute import CacheModifiedInputIterator, gpu_struct

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
def test_three_way_partition_basic(dtype, num_items, monkeypatch):
    # NOTE: the SASS check failure is seen only with NVRTC 13.1:
    if np.isdtype(dtype, np.float16):
        import cuda.compute._cccl_interop as cccl_interop

        monkeypatch.setattr(
            cccl_interop,
            "_check_sass",
            False,
        )

    h_in = random_array(num_items, dtype, max_value=100)

    def less_than_op(x):
        return x < 42

    def greater_equal_op(x):
        return x >= 42

    d_in = cp.asarray(h_in)
    d_first = cp.empty_like(d_in)
    d_second = cp.empty_like(d_in)
    d_unselected = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.int32)
    cuda.compute.three_way_partition(
        d_in,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
        num_items,
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

    def less_than_op(x):
        return x < 42

    def greater_equal_op(x):
        return x >= 42

    cuda.compute.three_way_partition(
        d_in,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
        0,
    )

    np.testing.assert_array_equal(d_num_selected.get(), np.array([0, 0]))


def test_three_way_partition_with_iterators():
    dtype = np.int32
    num_items = 10_000
    h_in = random_array(num_items, dtype, max_value=100)

    def less_than_op(x):
        return np.uint8(x < 42)

    def greater_equal_op(x):
        return np.uint8(x >= 42)

    expected_first, expected_second, expected_unselected, _, _, _ = (
        _host_three_way_partition(h_in, less_than_op, greater_equal_op)
    )

    d_in = cp.asarray(h_in)
    in_it = CacheModifiedInputIterator(d_in, modifier="stream")

    d_first = cp.empty_like(d_in)
    d_second = cp.empty_like(d_in)
    d_unselected = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint32)

    cuda.compute.three_way_partition(
        in_it,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
        num_items,
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
    @gpu_struct
    class pair_type:
        a: np.int32
        b: np.uint64

    comparison_value = np.int32(42)

    def less_than_op(x: pair_type):
        return (x.a < 42) & (x.b < 42)

    def greater_equal_op(x: pair_type):
        return (x.a >= 42) & (x.b >= 42)

    num_items = 20_000
    a_vals = random_array(num_items, np.int32, max_value=100)
    b_vals = a_vals.astype(np.uint64)

    h_in = np.empty(num_items, dtype=pair_type.dtype)
    h_in["a"] = a_vals
    h_in["b"] = b_vals

    expected_first_mask = (a_vals < comparison_value) & (b_vals < comparison_value)
    remaining_mask = ~expected_first_mask
    expected_second_mask = (a_vals[remaining_mask] >= comparison_value) & (
        b_vals[remaining_mask] >= comparison_value
    )

    expected_first = h_in[expected_first_mask]
    expected_second = h_in[remaining_mask][expected_second_mask]
    expected_unselected = h_in[remaining_mask][~expected_second_mask]

    h_in_i32 = h_in.view(np.int32).reshape(num_items, 4)
    d_in = cp.asarray(h_in_i32).view(pair_type.dtype).reshape(num_items)
    d_first = cp.empty_like(d_in)
    d_second = cp.empty_like(d_in)
    d_unselected = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    cuda.compute.three_way_partition(
        d_in,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
        num_items,
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
    h_in = random_array(num_items, dtype, max_value=100)

    def less_than_op(x):
        return x < 42

    def greater_equal_op(x):
        return x >= 42

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

    cuda.compute.three_way_partition(
        d_in,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
        num_items,
        stream=cuda_stream,
    )

    with cp_stream:
        num_selected = d_num_selected.get()
        got_first = d_first.get()[: int(num_selected[0])]
        got_second = d_second.get()[: int(num_selected[1])]
        got_unselected = d_unselected.get()[
            : int(num_items) - int(num_selected[0]) - int(num_selected[1])
        ]

    np.testing.assert_array_equal(got_first, expected_first)
    np.testing.assert_array_equal(got_second, expected_second)
    np.testing.assert_array_equal(got_unselected, expected_unselected)


def test_three_way_partition_no_selection():
    dtype = np.int32
    num_items = 10_000
    h_in = random_array(num_items, dtype, max_value=100)

    def less_than_op(x):
        return x == 101

    def greater_equal_op(x):
        return x == 102

    d_in = cp.asarray(h_in)
    d_first = cp.empty_like(d_in)
    d_second = cp.empty_like(d_in)
    d_unselected = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.int64)

    cuda.compute.three_way_partition(
        d_in,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
        num_items,
    )

    num_selected = d_num_selected.get()
    assert int(num_selected[0]) == 0 and int(num_selected[1]) == 0

    got_first = d_first.get()[: int(num_selected[0])]
    got_second = d_second.get()[: int(num_selected[1])]
    got_unselected = d_unselected.get()[:num_items]

    np.testing.assert_array_equal(got_first, np.empty(0, dtype=dtype))
    np.testing.assert_array_equal(got_second, np.empty(0, dtype=dtype))
    np.testing.assert_array_equal(got_unselected, h_in)


def test_three_way_partition_same_predicate():
    dtype = np.int32
    num_items = 100
    h_in = random_array(num_items, dtype, max_value=100)

    def always_true(x):
        return True

    d_in = cp.asarray(h_in)
    d_first = cp.empty_like(d_in)
    d_second = cp.empty_like(d_in)
    d_unselected = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.int64)

    cuda.compute.three_way_partition(
        d_in,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        always_true,
        always_true,
        num_items,
    )

    num_selected = d_num_selected.get()
    assert int(num_selected[0]) == num_items
    assert int(num_selected[1]) == 0


def test_three_way_partition_all_selected_first():
    dtype = np.int32
    num_items = 20_000
    h_in = np.full(num_items, 37, dtype=dtype)

    def less_than_op(x):
        return x == 37

    def greater_equal_op(x):
        return x == 42

    d_in = cp.asarray(h_in)
    d_first = cp.empty_like(d_in)
    d_second = cp.empty_like(d_in)
    d_unselected = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.int64)

    cuda.compute.three_way_partition(
        d_in,
        d_first,
        d_second,
        d_unselected,
        d_num_selected,
        less_than_op,
        greater_equal_op,
        num_items,
    )

    num_selected = d_num_selected.get()
    assert int(num_selected[0]) == num_items and int(num_selected[1]) == 0

    got_first = d_first.get()[: int(num_selected[0])]
    got_second = d_second.get()[: int(num_selected[1])]
    got_unselected = d_unselected.get()[
        : int(num_items) - int(num_selected[0]) - int(num_selected[1])
    ]

    np.testing.assert_array_equal(got_first, h_in)
    np.testing.assert_array_equal(got_second, np.empty(0, dtype=dtype))
    np.testing.assert_array_equal(got_unselected, np.empty(0, dtype=dtype))
