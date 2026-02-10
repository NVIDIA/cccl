# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import pytest
from numba.np.numpy_support import carray  # noqa: F401

import cuda.compute
from cuda.compute import CacheModifiedInputIterator, ZipIterator, gpu_struct

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


select_params = [
    (dt, 2**log_size) for dt in DTYPE_LIST for log_size in [2, 4, 6, 8, 10, 16, 20]
]


@pytest.fixture(scope="function", autouse=True)
def disable_sass_check(monkeypatch):
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )


def _host_select(h_in: np.ndarray, cond):
    # Vectorize condition to produce boolean mask
    mask = np.vectorize(cond, otypes=[np.uint8])(h_in).astype(bool)
    selected = h_in[mask]
    return selected, np.int64(selected.size)


@pytest.mark.parametrize("dtype,num_items", select_params)
def test_select_basic(dtype, num_items):
    h_in = random_array(num_items, dtype, max_value=100)

    def even_op(x):
        return x % 2 == 0

    d_in = cp.empty(num_items, dtype=dtype)
    d_in.set(h_in)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    cuda.compute.select(
        d_in,
        d_out,
        d_num_selected,
        even_op,
        num_items,
    )

    num_selected = int(d_num_selected[0].get())
    got = d_out.get()[:num_selected]

    expected, expected_count = _host_select(h_in, even_op)

    assert num_selected == expected_count
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("dtype,num_items", select_params)
def test_select_greater_than(dtype, num_items):
    h_in = random_array(num_items, dtype, max_value=100)

    def greater_than_42(x):
        return x > 42

    d_in = cp.asarray(h_in)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    cuda.compute.select(
        d_in,
        d_out,
        d_num_selected,
        greater_than_42,
        num_items,
    )

    num_selected = int(d_num_selected[0].get())
    got = d_out.get()[:num_selected]

    expected, expected_count = _host_select(h_in, greater_than_42)

    assert num_selected == expected_count
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_select_all_pass(dtype):
    num_items = 1000
    h_in = random_array(num_items, dtype, max_value=100)

    def always_true(x):
        return True

    d_in = cp.asarray(h_in)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    cuda.compute.select(
        d_in,
        d_out,
        d_num_selected,
        always_true,
        num_items,
    )

    num_selected = int(d_num_selected[0].get())
    got = d_out.get()[:num_selected]

    assert num_selected == num_items
    assert np.array_equal(got, h_in)


@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_select_none_pass(monkeypatch, dtype):
    num_items = 1000
    h_in = random_array(num_items, dtype, max_value=100)

    def always_false(x):
        return False

    d_in = cp.asarray(h_in)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.int32)

    cuda.compute.select(
        d_in,
        d_out,
        d_num_selected,
        always_false,
        num_items,
    )

    num_selected = int(d_num_selected[0].get())

    assert num_selected == 0


def test_select_empty():
    dtype = np.int32
    num_items = 0
    h_in = np.array([], dtype=dtype)

    def even_op(x):
        return x % 2 == 0

    d_in = cp.asarray(h_in)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    cuda.compute.select(
        d_in,
        d_out,
        d_num_selected,
        even_op,
        num_items,
    )

    num_selected = int(d_num_selected[0].get())

    assert num_selected == 0


@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_select_with_iterator(dtype):
    num_items = 10000
    h_in = random_array(num_items, dtype, max_value=100)

    def less_than_50(x):
        return x < 50

    d_in = cp.asarray(h_in)
    d_in_iter = CacheModifiedInputIterator(d_in, modifier="stream")
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    cuda.compute.select(
        d_in_iter,
        d_out,
        d_num_selected,
        less_than_50,
        num_items,
    )

    num_selected = int(d_num_selected[0].get())
    got = d_out.get()[:num_selected]

    expected, expected_count = _host_select(h_in, less_than_50)

    assert num_selected == expected_count
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_select_object_api(dtype):
    num_items = 10000
    h_in = random_array(num_items, dtype, max_value=100)

    def divisible_by_3(x):
        return x % 3 == 0

    d_in = cp.asarray(h_in)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    # Create select object
    selector = cuda.compute.make_select(
        d_in,
        d_out,
        d_num_selected,
        divisible_by_3,
    )

    # Get temp storage size
    temp_storage_bytes = selector(
        None,
        d_in,
        d_out,
        d_num_selected,
        divisible_by_3,
        num_items,
    )

    # Allocate temp storage
    d_temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    # Execute select
    selector(
        d_temp_storage,
        d_in,
        d_out,
        d_num_selected,
        divisible_by_3,
        num_items,
    )

    num_selected = int(d_num_selected[0].get())
    got = d_out.get()[:num_selected]

    expected, expected_count = _host_select(h_in, divisible_by_3)

    assert num_selected == expected_count
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_select_reuse_object(dtype):
    """Test that the select object can be reused multiple times with different inputs"""
    num_items = 1000

    def positive_op(x):
        return x > 0

    d_out = cp.empty(num_items, dtype=dtype)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    # Create select object with initial input
    h_in1 = random_array(num_items, dtype, max_value=100) - 50
    d_in1 = cp.asarray(h_in1)
    selector = cuda.compute.make_select(
        d_in1,
        d_out,
        d_num_selected,
        positive_op,
    )

    # First execution
    temp_storage_bytes = selector(
        None, d_in1, d_out, d_num_selected, positive_op, num_items
    )
    d_temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
    selector(d_temp_storage, d_in1, d_out, d_num_selected, positive_op, num_items)

    num_selected1 = int(d_num_selected[0].get())
    got1 = d_out.get()[:num_selected1]
    expected1, expected_count1 = _host_select(h_in1, positive_op)

    assert num_selected1 == expected_count1
    assert np.array_equal(got1, expected1)

    # Reuse with different input
    h_in2 = random_array(num_items, dtype, max_value=100) - 50
    d_in2 = cp.asarray(h_in2)

    selector(d_temp_storage, d_in2, d_out, d_num_selected, positive_op, num_items)

    num_selected2 = int(d_num_selected[0].get())
    got2 = d_out.get()[:num_selected2]
    expected2, expected_count2 = _host_select(h_in2, positive_op)

    assert num_selected2 == expected_count2
    assert np.array_equal(got2, expected2)


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_select_with_struct(dtype):
    """Test selection with custom struct types"""

    @gpu_struct
    class Point:
        x: dtype
        y: dtype

    num_items = 1000
    h_x = random_array(num_items, dtype, max_value=100)
    h_y = random_array(num_items, dtype, max_value=100)

    h_in = np.empty(num_items, dtype=Point.dtype)
    h_in["x"] = h_x
    h_in["y"] = h_y

    def in_first_quadrant(p: Point) -> np.uint8:
        return (p.x > 50) and (p.y > 50)

    d_in = cp.empty(num_items, dtype=Point.dtype)
    d_in.set(h_in)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    cuda.compute.select(
        d_in,
        d_out,
        d_num_selected,
        in_first_quadrant,
        num_items,
    )

    num_selected = int(d_num_selected[0].get())
    got = d_out.get()[:num_selected]

    # Host reference
    def host_in_first_quadrant(p):
        return (p[0] > 50) and (p[1] > 50)

    expected, expected_count = _host_select(h_in, host_in_first_quadrant)

    assert num_selected == expected_count
    assert np.array_equal(got["x"], expected["x"])
    assert np.array_equal(got["y"], expected["y"])


def test_select_with_zip_iterator(monkeypatch):
    """Test select with ZipIterator input and output"""

    dtype = np.int32
    num_items = 10_000

    # Create two arrays
    h_in1 = random_array(num_items, dtype, max_value=100)
    h_in2 = random_array(num_items, dtype, max_value=100)

    # Select condition: sum of elements < 70
    def condition(pair):
        return (pair[0] + pair[1]) < 70

    # Device arrays
    d_in1 = cp.asarray(h_in1)
    d_in2 = cp.asarray(h_in2)

    # Create zip iterator for input
    zip_in = ZipIterator(d_in1, d_in2)

    # Allocate output arrays
    d_out1 = cp.empty_like(d_in1)
    d_out2 = cp.empty_like(d_in2)

    # Create zip iterator for output
    zip_out = ZipIterator(d_out1, d_out2)
    d_num_selected = cp.empty(1, dtype=np.int32)

    cuda.compute.select(
        zip_in,
        zip_out,
        d_num_selected,
        condition,
        num_items,
    )

    num_selected = int(d_num_selected[0].get())

    # Get results
    got1 = d_out1.get()[:num_selected]
    got2 = d_out2.get()[:num_selected]

    # Verify results: all elements should satisfy the condition
    for i in range(num_selected):
        assert (got1[i] + got2[i]) < 70

    # Verify count
    h_sums = h_in1 + h_in2
    expected_count = np.sum(h_sums < 70)

    assert num_selected == expected_count


def test_select_stateful_threshold():
    """Test stateful select that uses state for threshold"""
    num_items = 1000
    h_in = random_array(num_items, np.int32, max_value=100)

    # Create device state containing threshold value
    threshold_value = 50
    threshold_state = cp.array([threshold_value], dtype=np.int32)

    # Define condition that references state as closure
    def threshold_select(x):
        return x > threshold_state[0]

    d_in = cp.asarray(h_in)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    cuda.compute.select(
        d_in,
        d_out,
        d_num_selected,
        threshold_select,
        num_items,
    )

    # Check selected output
    num_selected = int(d_num_selected[0].get())
    got = d_out.get()[:num_selected]

    # Verify all output values are > threshold
    assert np.all(got > threshold_value)

    # Verify we got the expected number of items
    expected_selected = h_in[h_in > threshold_value]
    expected_count = len(expected_selected)

    assert num_selected == expected_count

    # Verify exact results
    assert np.array_equal(got, expected_selected)


def test_select_stateful_atomic():
    """Test stateful select with atomic operations to count rejected items"""
    from numba import cuda as numba_cuda

    num_items = 1000
    h_in = random_array(num_items, np.int32, max_value=100)

    # Create device state for counting rejected items
    reject_counter = cp.zeros(1, dtype=np.int32)

    # Define condition that references state as closure
    def count_rejects(x):
        if x > 50:
            return True
        else:
            numba_cuda.atomic.add(reject_counter, 0, 1)
            return False

    d_in = cp.asarray(h_in)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    cuda.compute.select(
        d_in,
        d_out,
        d_num_selected,
        count_rejects,
        num_items,
    )

    # Check selected output
    num_selected = int(d_num_selected[0].get())
    got = d_out.get()[:num_selected]

    # Verify all output values are > 50
    assert np.all(got > 50)

    # Verify we got the expected number of items
    expected_selected = h_in[h_in > 50]
    expected_count = len(expected_selected)

    assert num_selected == expected_count

    # Verify exact results
    assert np.array_equal(got, expected_selected)

    # Verify state contains count of rejected items
    rejected_count = int(reject_counter[0].get())
    expected_rejected = len(h_in[h_in <= 50])
    assert rejected_count == expected_rejected, (
        f"Expected {expected_rejected} rejections, got {rejected_count}"
    )


def test_select_with_side_effect_counting_rejects():
    """Select with side effect that counts rejected items"""
    from numba import cuda as numba_cuda

    d_in = cp.arange(100, dtype=np.int32)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(1, dtype=np.uint64)

    reject_count = cp.zeros(1, dtype=np.int32)

    # Define condition that references state as closure
    def count_rejects(x):
        if x >= 50:
            return True
        else:
            numba_cuda.atomic.add(reject_count, 0, 1)
            return False

    cuda.compute.select(d_in, d_out, d_num_selected, count_rejects, len(d_in))

    num_selected = int(d_num_selected.get()[0])
    num_rejected = int(reject_count.get()[0])

    assert num_selected == 50  # Values 50-99
    assert num_rejected == 50  # Values 0-49


def test_select_with_lambda():
    """Test select with a lambda function as predicate."""
    num_items = 100
    h_in = np.arange(num_items, dtype=np.int32)

    d_in = cp.asarray(h_in)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    # Use a lambda function directly as the predicate
    cuda.compute.select(d_in, d_out, d_num_selected, lambda x: x % 2 == 0, num_items)

    num_selected = int(d_num_selected.get()[0])
    expected_selected = [x for x in h_in if x % 2 == 0]

    assert num_selected == len(expected_selected)
    np.testing.assert_array_equal(d_out.get()[:num_selected], expected_selected)


def test_select_stateful_state_updates():
    """Test that select correctly updates state between calls with different thresholds."""
    num_items = 20
    d_in = cp.arange(num_items, dtype=np.int32)
    d_out = cp.empty_like(d_in)
    d_count = cp.zeros(2, dtype=np.uint64)

    # Create two different thresholds
    threshold_5 = cp.array([5], dtype=np.int32)
    threshold_15 = cp.array([15], dtype=np.int32)

    # Call 1: Select items > 5 (should get 14 items: 6-19)
    def select_gt_5(x):
        return x > threshold_5[0]

    cuda.compute.select(d_in, d_out, d_count, select_gt_5, num_items)
    count1 = int(d_count[0].get())
    assert count1 == 14
    expected_1 = list(range(6, 20))
    np.testing.assert_array_equal(d_out.get()[:count1], expected_1)

    # Call 2: Select items > 15 (should get 4 items: 16-19)
    def select_gt_15(x):
        return x > threshold_15[0]

    d_count.fill(0)
    cuda.compute.select(d_in, d_out, d_count, select_gt_15, num_items)
    count2 = int(d_count[0].get())
    assert count2 == 4
    expected_2 = list(range(16, 20))
    np.testing.assert_array_equal(d_out.get()[:count2], expected_2)

    # Call 3: Back to first threshold (test cache reuse with updated state)
    d_count.fill(0)
    cuda.compute.select(d_in, d_out, d_count, select_gt_5, num_items)
    count3 = int(d_count[0].get())
    assert count3 == 14
    np.testing.assert_array_equal(d_out.get()[:count3], expected_1)


def test_select_stateful_same_bytecode_different_state():
    """
    Test that select works correctly when using factory functions that produce
    identical bytecode but capture different state arrays.

    This is a regression test for the cache collision bug where functions with
    the same bytecode but different captured arrays would reuse stale state.
    """
    num_items = 20
    d_in = cp.arange(num_items, dtype=np.int32)
    d_out = cp.empty_like(d_in)
    d_count = cp.zeros(2, dtype=np.uint64)

    # Factory that creates functions with identical bytecode
    def make_selector(threshold_array):
        def selector(x):
            return x > threshold_array[0]

        return selector

    threshold_5 = cp.array([5], dtype=np.int32)
    threshold_15 = cp.array([15], dtype=np.int32)

    select_5 = make_selector(threshold_5)
    select_15 = make_selector(threshold_15)

    # Call 1: threshold > 5
    cuda.compute.select(d_in, d_out, d_count, select_5, num_items)
    count1 = int(d_count[0].get())
    assert count1 == 14

    # Call 2: threshold > 15 (different state, same bytecode)
    d_count.fill(0)
    cuda.compute.select(d_in, d_out, d_count, select_15, num_items)
    count2 = int(d_count[0].get())
    assert count2 == 4  # If this fails, cache collision bug is present


def test_stateful_caching_same_dtype_different_values():
    """
    Test that stateful ops with same dtype but different values work correctly.
    After transformation, values are runtime parameters, so they should use the
    same compiled code.
    """
    import cupy as cp
    import numpy as np

    import cuda.compute

    num_items = 100
    d_in = cp.arange(num_items, dtype=np.int32)
    d_out = cp.empty_like(d_in)
    d_count = cp.zeros(2, dtype=np.uint64)

    # Two thresholds with SAME dtype, SAME size, DIFFERENT values
    threshold_30 = cp.array([30], dtype=np.int32)
    threshold_70 = cp.array([70], dtype=np.int32)

    # Test with threshold_30
    def select_gt_30(x):
        return x > threshold_30[0]

    cuda.compute.select(d_in, d_out, d_count, select_gt_30, num_items)
    count_30 = int(d_count[0].get())

    # Test with threshold_70
    def select_gt_70(x):
        return x > threshold_70[0]

    d_out.fill(0)
    d_count.fill(0)
    cuda.compute.select(d_in, d_out, d_count, select_gt_70, num_items)
    count_70 = int(d_count[0].get())

    # Verify correct results (not cache collision)
    assert count_30 == 69  # Values 31-99
    assert count_70 == 29  # Values 71-99
