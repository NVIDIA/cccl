# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute.iterators import (
    PermutationIterator,
    ShuffleIterator,
)


def test_shuffle_iterator_bijectivity():
    num_items = 100
    seed = 42

    shuffle_it = ShuffleIterator(num_items, seed)

    d_output = DeviceArray.empty(num_items, np.int64)
    cuda.compute.unary_transform(
        d_in=shuffle_it, d_out=d_output, op=lambda x: x, num_items=num_items
    )

    result = d_output.copy_to_host()

    assert len(set(result)) == num_items
    assert set(result) == set(range(num_items))


def test_shuffle_iterator_determinism():
    num_items = 50
    seed = 12345

    shuffle_it1 = ShuffleIterator(num_items, seed)
    shuffle_it2 = ShuffleIterator(num_items, seed)

    d_output1 = DeviceArray.empty(num_items, np.int64)
    d_output2 = DeviceArray.empty(num_items, np.int64)

    cuda.compute.unary_transform(
        d_in=shuffle_it1, d_out=d_output1, op=lambda x: x, num_items=num_items
    )
    cuda.compute.unary_transform(
        d_in=shuffle_it2, d_out=d_output2, op=lambda x: x, num_items=num_items
    )

    np.testing.assert_array_equal(d_output1.copy_to_host(), d_output2.copy_to_host())


@pytest.mark.parametrize("num_items", [1, 2, 7, 16, 17, 100, 1000, 1023, 1024, 1025])
def test_shuffle_iterator_various_sizes(num_items):
    seed = 42

    shuffle_it = ShuffleIterator(num_items, seed)

    d_output = DeviceArray.empty(num_items, np.int64)
    cuda.compute.unary_transform(
        d_in=shuffle_it, d_out=d_output, op=lambda x: x, num_items=num_items
    )

    result = d_output.copy_to_host()

    assert len(set(result)) == num_items
    assert set(result) == set(range(num_items))


def test_shuffle_iterator_with_permutation_iterator():
    num_items = 10
    seed = 42

    h_values = np.asarray([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int32)
    d_values = DeviceArray.from_numpy(h_values)

    shuffle_it = ShuffleIterator(num_items, seed)
    perm_it = PermutationIterator(d_values, shuffle_it)

    d_output = DeviceArray.empty(num_items, np.int32)
    cuda.compute.unary_transform(
        d_in=perm_it, d_out=d_output, op=lambda x: x, num_items=num_items
    )

    result = d_output.copy_to_host()

    assert result.sum() == h_values.sum()
    assert sorted(result) == sorted(h_values)


def test_shuffle_iterator_invalid_num_items():
    with pytest.raises(ValueError, match="num_items must be > 0"):
        ShuffleIterator(0, seed=42)

    with pytest.raises(ValueError, match="num_items must be > 0"):
        ShuffleIterator(-1, seed=42)
