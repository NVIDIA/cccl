# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer


def add(a, b):
    return a + b


def maximum(a, b):
    return a if a > b else b


def cpu_reduce_by_key(keys, values, op):
    """Reference: reduce each run of equal adjacent keys.

    The representative key is the run's *last* key, which is the CUB contract -- not the first key the
    way unique_by_key behaves.
    """
    out_keys, out_aggs = [], []
    i = 0
    n = len(keys)
    while i < n:
        j = i
        acc = values[i]
        while j + 1 < n and keys[j + 1] == keys[i]:
            j += 1
            acc = op(acc, values[j])
        out_keys.append(keys[j])
        out_aggs.append(acc)
        i = j + 1
    return out_keys, out_aggs


def run(keys, values, op=add, kdtype=None, vdtype=None):
    kdtype = kdtype or keys.dtype
    vdtype = vdtype or values.dtype
    keys = keys.astype(kdtype)
    values = values.astype(vdtype)
    n = keys.size
    dk = DeviceArray.from_numpy(keys)
    dv = DeviceArray.from_numpy(values)
    du = DeviceArray.empty((n,), kdtype)
    da = DeviceArray.empty((n,), vdtype)
    dn = DeviceArray.empty((1,), np.int32)
    cuda.compute.reduce_by_key(
        d_keys_in=dk,
        d_values_in=dv,
        d_unique_out=du,
        d_aggregates_out=da,
        d_num_runs_out=dn,
        op=op,
        num_items=n,
    )
    num_runs = int(dn.copy_to_host()[0])
    return (
        du.copy_to_host()[:num_runs],
        da.copy_to_host()[:num_runs],
        num_runs,
    )


def test_worked_example_from_the_issue():
    """The hand-computed example, including the representative-key rule."""
    keys = np.array([0, 0, 3, 3, 3, 0], dtype=np.int32)
    values = np.array([2, 5, 1, 4, 6, 7], dtype=np.int32)

    uk, aggs, runs = run(keys, values)
    assert runs == 3
    assert list(uk) == [0, 3, 0]  # last key of each run
    assert list(aggs) == [7, 11, 7]


@pytest.mark.parametrize("n", [0, 1, 2, 33, 128, 129, 4097])
@pytest.mark.parametrize(
    "key_pattern",
    ["all_same", "all_distinct", "alternating", "non_adjacent_reuse", "runs_of_three"],
)
def test_matches_cpu(n, key_pattern):
    rng = np.random.default_rng(4321 + n)
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

    want_k, want_a = cpu_reduce_by_key(keys, values, add)
    got_k, got_a, runs = run(keys, values)
    assert runs == len(want_k)
    assert list(got_k) == want_k
    assert list(got_a) == want_a


def test_run_spanning_many_tiles_and_trailing_single():
    n = 100_003
    keys = np.zeros(n, dtype=np.int32)
    keys[n - 1] = 9
    values = np.ones(n, dtype=np.int32)

    uk, aggs, runs = run(keys, values)
    assert runs == 2
    assert list(uk) == [0, 9]
    assert list(aggs) == [n - 1, 1]


def test_min_max_operator():
    keys = np.array([1, 1, 2, 2, 2], dtype=np.int32)
    values = np.array([5, 3, 9, 7, 8], dtype=np.int32)
    _, aggs, runs = run(keys, values, op=maximum)
    assert runs == 2
    assert list(aggs) == [5, 9]


def test_distinct_key_and_value_dtypes():
    """Keys and aggregates may differ; the aggregate must not be truncated to the key type."""
    keys = np.array([0, 0, 1, 1], dtype=np.int64)
    values = np.array([1.5, 2.5, 3.0, 4.0], dtype=np.float32)
    uk, aggs, runs = run(keys, values)
    assert runs == 2
    assert list(uk) == [0, 1]
    assert np.allclose(aggs, [4.0, 7.0])


def test_reused_object_across_different_inputs():
    a_keys = np.array([1, 1, 2], dtype=np.int32)
    a_vals = np.array([1, 2, 3], dtype=np.int32)
    b_keys = np.array([7, 7, 7], dtype=np.int32)
    b_vals = np.array([10, 20, 30], dtype=np.int32)

    dk = DeviceArray.empty((3,), np.int32)
    dv = DeviceArray.empty((3,), np.int32)
    du = DeviceArray.empty((3,), np.int32)
    da = DeviceArray.empty((3,), np.int32)
    dn = DeviceArray.empty((1,), np.int32)

    reducer = cuda.compute.make_reduce_by_key(
        d_keys_in=dk,
        d_values_in=dv,
        d_unique_out=du,
        d_aggregates_out=da,
        d_num_runs_out=dn,
        op=add,
    )

    for keys, values, want_k, want_a in (
        (a_keys, a_vals, [1, 2], [3, 3]),
        (b_keys, b_vals, [7], [60]),
        (a_keys, a_vals, [1, 2], [3, 3]),
    ):
        dk.copy_from_host(keys)
        dv.copy_from_host(values)
        nbytes = reducer(
            temp_storage=None,
            d_keys_in=dk,
            d_values_in=dv,
            d_unique_out=du,
            d_aggregates_out=da,
            d_num_runs_out=dn,
            op=add,
            num_items=3,
        )
        tmp = TempStorageBuffer(nbytes)
        reducer(
            temp_storage=tmp,
            d_keys_in=dk,
            d_values_in=dv,
            d_unique_out=du,
            d_aggregates_out=da,
            d_num_runs_out=dn,
            op=add,
            num_items=3,
        )
        runs = int(dn.copy_to_host()[0])
        assert runs == len(want_k)
        assert list(du.copy_to_host()[:runs]) == want_k
        assert list(da.copy_to_host()[:runs]) == want_a

