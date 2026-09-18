# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute


def cpu_inclusive(keys, values, op):
    """Inclusive scan that restarts at every boundary between unequal adjacent keys."""
    out = []
    run = None
    for i, (k, v) in enumerate(zip(keys, values)):
        run = v if i == 0 or k != keys[i - 1] else op(run, v)
        out.append(run)
    return out


def cpu_exclusive(keys, values, op, init):
    """Exclusive scan; init is applied at the head of every run, not just the first."""
    out = []
    run = init
    for i, (k, v) in enumerate(zip(keys, values)):
        if i > 0 and k != keys[i - 1]:
            run = init
        out.append(run)
        run = op(run, v)
    return out


def add(a, b):
    return a + b


def eq(a, b):
    return a == b


def run_inclusive(keys, values):
    d_keys = DeviceArray.from_numpy(keys)
    d_values = DeviceArray.from_numpy(values)
    d_out = DeviceArray.empty(values.shape, values.dtype)
    cuda.compute.inclusive_scan_by_key(
        d_keys_in=d_keys,
        d_values_in=d_values,
        d_values_out=d_out,
        op=add,
        equality_op=eq,
        num_items=values.size,
    )
    return d_out.copy_to_host()


def run_exclusive(keys, values, init):
    d_keys = DeviceArray.from_numpy(keys)
    d_values = DeviceArray.from_numpy(values)
    d_out = DeviceArray.empty(values.shape, values.dtype)
    cuda.compute.exclusive_scan_by_key(
        d_keys_in=d_keys,
        d_values_in=d_values,
        d_values_out=d_out,
        op=add,
        equality_op=eq,
        init_value=np.array([init], dtype=values.dtype),
        num_items=values.size,
    )
    return d_out.copy_to_host()


def test_worked_example_from_the_issue():
    """The hand-computed example that defines the semantics: the final 0 starts a new run."""
    keys = np.array([0, 0, 3, 3, 3, 0], dtype=np.int32)
    values = np.array([2, 5, 1, 4, 6, 7], dtype=np.int32)

    assert list(run_inclusive(keys, values)) == [2, 7, 1, 5, 11, 7]
    assert list(run_exclusive(keys, values, 10)) == [10, 12, 10, 11, 15, 10]


@pytest.mark.parametrize("n", [0, 1, 2, 31, 32, 33, 127, 128, 129, 1024, 4097])
@pytest.mark.parametrize(
    "key_pattern",
    ["all_same", "all_distinct", "alternating", "non_adjacent_reuse", "runs_of_three"],
)
def test_inclusive_matches_cpu(n, key_pattern):
    rng = np.random.default_rng(1234 + n)
    values = rng.integers(0, 100, size=n, dtype=np.int32)

    if key_pattern == "all_same":
        keys = np.zeros(n, dtype=np.int32)
    elif key_pattern == "all_distinct":
        keys = np.arange(n, dtype=np.int32)
    elif key_pattern == "alternating":
        keys = (np.arange(n) % 2).astype(np.int32)
    elif key_pattern == "non_adjacent_reuse":
        keys = (np.arange(n) % 3).astype(np.int32)
    else:
        keys = (np.arange(n) // 3).astype(np.int32)

    expected = np.array(cpu_inclusive(keys, values, add), dtype=np.int32)
    assert np.array_equal(run_inclusive(keys, values), expected)


@pytest.mark.parametrize("n", [0, 1, 33, 128, 4097])
def test_exclusive_applies_init_per_run(n):
    rng = np.random.default_rng(99 + n)
    values = rng.integers(0, 50, size=n, dtype=np.int32)
    keys = (np.arange(n) // 5).astype(np.int32)

    expected = np.array(cpu_exclusive(keys, values, add, 7), dtype=np.int32)
    assert np.array_equal(run_exclusive(keys, values, 7), expected)


def test_run_spanning_many_tiles():
    """A single run far longer than one tile, followed by a one-element trailing run."""
    n = 100_003
    keys = np.zeros(n, dtype=np.int32)
    keys[n - 1] = 9
    values = np.ones(n, dtype=np.int32)

    got = run_inclusive(keys, values)
    assert got[n - 2] == n - 1
    assert got[n - 1] == 1  # the trailing run restarts


def test_reused_object_across_different_inputs():
    from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer

    keys_a = np.array([1, 1, 2, 2], dtype=np.int32)
    vals_a = np.array([1, 2, 3, 4], dtype=np.int32)
    keys_b = np.array([5, 5, 5, 5], dtype=np.int32)
    vals_b = np.array([10, 20, 30, 40], dtype=np.int32)

    d_keys = DeviceArray.empty(keys_a.shape, keys_a.dtype)
    d_values = DeviceArray.empty(vals_a.shape, vals_a.dtype)
    d_out = DeviceArray.empty(vals_a.shape, vals_a.dtype)

    scan = cuda.compute.make_inclusive_scan_by_key(
        d_keys_in=d_keys, d_values_in=d_values, d_values_out=d_out, op=add, equality_op=eq
    )

    for keys, values in ((keys_a, vals_a), (keys_b, vals_b), (keys_a, vals_a)):
        d_keys.copy_from_host(keys)
        d_values.copy_from_host(values)
        # Two-phase: query the workspace size, allocate, then execute through the same object.
        nbytes = scan(
            temp_storage=None,
            d_keys_in=d_keys,
            d_values_in=d_values,
            d_values_out=d_out,
            op=add,
            equality_op=eq,
            init_value=None,
            num_items=values.size,
        )
        tmp = TempStorageBuffer(nbytes)
        scan(
            temp_storage=tmp,
            d_keys_in=d_keys,
            d_values_in=d_values,
            d_values_out=d_out,
            op=add,
            equality_op=eq,
            init_value=None,
            num_items=values.size,
        )
        expected = np.array(cpu_inclusive(keys, values, add), dtype=np.int32)
        assert np.array_equal(d_out.copy_to_host(), expected)


def test_key_and_value_dtypes():
    keys = np.array([0, 0, 1, 1], dtype=np.int64)
    values = np.array([1.5, 2.5, 3.0, 4.0], dtype=np.float32)
    d_keys = DeviceArray.from_numpy(keys)
    d_values = DeviceArray.from_numpy(values)
    d_out = DeviceArray.empty(values.shape, values.dtype)
    cuda.compute.inclusive_scan_by_key(
        d_keys_in=d_keys,
        d_values_in=d_values,
        d_values_out=d_out,
        op=add,
        equality_op=eq,
        num_items=4,
    )
    assert np.allclose(d_out.copy_to_host(), [1.5, 4.0, 3.0, 7.0])


def test_future_value_init_is_rejected():
    """A device-resident init would need a different CUB overload; refusing beats guessing."""
    keys = np.array([0, 0], dtype=np.int32)
    values = np.array([1, 2], dtype=np.int32)
    device_init = DeviceArray.empty((1,), np.int32)
    with pytest.raises(NotImplementedError, match="future value"):
        cuda.compute.make_exclusive_scan_by_key(
            d_keys_in=DeviceArray.from_numpy(keys),
            d_values_in=DeviceArray.from_numpy(values),
            d_values_out=DeviceArray.empty(values.shape, values.dtype),
            op=add,
            equality_op=eq,
            init_value=device_init,
        )


