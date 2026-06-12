# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import cupy as cp
import numpy as np
import pytest

import cuda.compute
from cuda.compute import OpKind

DTYPE_LIST = [
    np.int32,
    np.int64,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
]


def random_sorted_array(size, dtype, max_value=1000):
    rng = np.random.default_rng()
    if np.isdtype(dtype, "integral"):
        data = rng.integers(max_value, size=size, dtype=dtype)
    else:
        if dtype == np.float16:  # pragma: no cover - float16 not used here
            data = rng.random(size=size, dtype=np.float32).astype(dtype)
        else:
            data = rng.random(size=size, dtype=dtype)
    data.sort()
    return data


@pytest.fixture(scope="function", autouse=True)
def disable_sass_check(monkeypatch):
    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        False,
    )


@pytest.mark.parametrize(
    "search, side",
    [
        (cuda.compute.lower_bound, "left"),
        (cuda.compute.upper_bound, "right"),
    ],
)
<<<<<<< HEAD
@pytest.mark.no_numba
=======
>>>>>>> main
def test_binary_search_explicit_opkind_less(search, side):
    h_data = np.array([1, 3, 3, 7, 9], dtype=np.int32)
    h_values = np.array([0, 3, 4, 10], dtype=np.int32)

    d_data = cp.asarray(h_data)
    d_values = cp.asarray(h_values)
    d_out = cp.empty(len(h_values), dtype=np.uintp)

    search(
        d_data=d_data,
        num_items=len(d_data),
        d_values=d_values,
        num_values=len(d_values),
        d_out=d_out,
        comp=OpKind.LESS,
    )

    expected = np.searchsorted(h_data, h_values, side=side).astype(np.uintp)
    np.testing.assert_array_equal(d_out.get(), expected)


@pytest.mark.parametrize(
    "search, side",
    [
        (cuda.compute.lower_bound, "left"),
        (cuda.compute.upper_bound, "right"),
    ],
)
def test_binary_search_custom_comparator(search, side):
    h_data = np.array([9, 7, 3, 3, 1], dtype=np.int32)
    h_values = np.array([10, 4, 3, 0], dtype=np.int32)

    def greater(lhs, rhs):
        return lhs > rhs

    d_data = cp.asarray(h_data)
    d_values = cp.asarray(h_values)
    d_out = cp.empty(len(h_values), dtype=np.uintp)

    search(
        d_data=d_data,
        num_items=len(d_data),
        d_values=d_values,
        num_values=len(d_values),
        d_out=d_out,
        comp=greater,
    )

    expected = np.searchsorted(-h_data, -h_values, side=side).astype(np.uintp)
    np.testing.assert_array_equal(d_out.get(), expected)


@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize(
    "num_items,num_values", [(0, 0), (0, 128), (128, 0), (512, 128)]
)
def test_lower_bound_basic(dtype, num_items, num_values):
    h_data = random_sorted_array(num_items, dtype)
    h_values = random_sorted_array(num_values, dtype)

    d_data = cp.asarray(h_data)
    d_values = cp.asarray(h_values)
    d_out = cp.empty(num_values, dtype=np.uintp)

    cuda.compute.lower_bound(
        d_data=d_data,
        num_items=num_items,
        d_values=d_values,
        num_values=num_values,
        d_out=d_out,
    )

    expected = np.searchsorted(h_data, h_values, side="left").astype(np.uintp)
    got = cp.asnumpy(d_out)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize(
    "num_items,num_values", [(0, 0), (0, 128), (128, 0), (512, 128)]
)
def test_upper_bound_basic(dtype, num_items, num_values):
    h_data = random_sorted_array(num_items, dtype)
    h_values = random_sorted_array(num_values, dtype)

    d_data = cp.asarray(h_data)
    d_values = cp.asarray(h_values)
    d_out = cp.empty(num_values, dtype=np.uintp)

    cuda.compute.upper_bound(
        d_data=d_data,
        num_items=num_items,
        d_values=d_values,
        num_values=num_values,
        d_out=d_out,
    )

    expected = np.searchsorted(h_data, h_values, side="right").astype(np.uintp)
    got = cp.asnumpy(d_out)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_binary_search_with_duplicates(dtype):
    rng = np.random.default_rng()
    h_data = (
        rng.integers(10, size=1024, dtype=dtype)
        if np.isdtype(dtype, "integral")
        else rng.random(1024, dtype=dtype)
    )
    h_data.sort()
    h_values = (
        rng.integers(10, size=128, dtype=dtype)
        if np.isdtype(dtype, "integral")
        else rng.random(128, dtype=dtype)
    )

    d_data = cp.asarray(h_data)
    d_values = cp.asarray(h_values)
    d_out = cp.empty(len(h_values), dtype=np.uintp)

    cuda.compute.lower_bound(
        d_data=d_data,
        num_items=len(h_data),
        d_values=d_values,
        num_values=len(h_values),
        d_out=d_out,
    )
    expected = np.searchsorted(h_data, h_values, side="left").astype(np.uintp)
    got = cp.asnumpy(d_out)
    assert np.array_equal(got, expected)

    cuda.compute.upper_bound(
        d_data=d_data,
        num_items=len(h_data),
        d_values=d_values,
        num_values=len(h_values),
        d_out=d_out,
    )
    expected = np.searchsorted(h_data, h_values, side="right").astype(np.uintp)
    got = cp.asnumpy(d_out)
    assert np.array_equal(got, expected)


def test_binary_search_requires_unsigned_output():
    """Output must be unsigned integer dtype for indices."""
    d_data = cp.asarray(np.array([1, 2, 3, 4], dtype=np.int32))
    d_values = cp.asarray(np.array([2, 3], dtype=np.int32))
    d_out = cp.empty(len(d_values), dtype=np.int32)  # signed, should fail

    with pytest.raises(TypeError, match="unsigned integer"):
        cuda.compute.lower_bound(
            d_data=d_data,
            num_items=len(d_data),
            d_values=d_values,
            num_values=len(d_values),
            d_out=d_out,
        )


def test_binary_search_requires_pointer_sized_output():
    """Output must be pointer-sized (np.uintp) to hold any valid index."""
    d_data = cp.asarray(np.array([1, 2, 3, 4], dtype=np.int32))
    d_values = cp.asarray(np.array([2, 3], dtype=np.int32))
    d_out = cp.empty(
        len(d_values), dtype=np.uint32
    )  # unsigned but not pointer-sized (on 64-bit)

    with pytest.raises(ValueError, match="pointer-sized"):
        cuda.compute.lower_bound(
            d_data=d_data,
            num_items=len(d_data),
            d_values=d_values,
            num_values=len(d_values),
            d_out=d_out,
        )
