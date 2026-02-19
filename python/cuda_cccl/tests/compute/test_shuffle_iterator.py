# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import pytest

import cuda.compute
from cuda.compute.iterators import (
    PermutationIterator,
    ShuffleIterator,
)


def test_shuffle_iterator_bijectivity():
    num_items = 100
    seed = 42

    shuffle_it = ShuffleIterator(num_items, seed)

    d_output = cp.empty(num_items, dtype=np.int64)
    cuda.compute.unary_transform(shuffle_it, d_output, lambda x: x, num_items)

    result = d_output.get()

    assert len(set(result)) == num_items
    assert set(result) == set(range(num_items))


def test_shuffle_iterator_determinism():
    num_items = 50
    seed = 12345

    shuffle_it1 = ShuffleIterator(num_items, seed)
    shuffle_it2 = ShuffleIterator(num_items, seed)

    d_output1 = cp.empty(num_items, dtype=np.int64)
    d_output2 = cp.empty(num_items, dtype=np.int64)

    cuda.compute.unary_transform(shuffle_it1, d_output1, lambda x: x, num_items)
    cuda.compute.unary_transform(shuffle_it2, d_output2, lambda x: x, num_items)

    cp.testing.assert_array_equal(d_output1, d_output2)


@pytest.mark.parametrize("num_items", [1, 2, 7, 16, 17, 100, 1000, 1023, 1024, 1025])
def test_shuffle_iterator_various_sizes(num_items):
    seed = 42

    shuffle_it = ShuffleIterator(num_items, seed)

    d_output = cp.empty(num_items, dtype=np.int64)
    cuda.compute.unary_transform(shuffle_it, d_output, lambda x: x, num_items)

    result = d_output.get()

    assert len(set(result)) == num_items
    assert set(result) == set(range(num_items))


def test_shuffle_iterator_with_permutation_iterator():
    num_items = 10
    seed = 42

    d_values = cp.asarray([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int32)

    shuffle_it = ShuffleIterator(num_items, seed)
    perm_it = PermutationIterator(d_values, shuffle_it)

    d_output = cp.empty(num_items, dtype=np.int32)
    cuda.compute.unary_transform(perm_it, d_output, lambda x: x, num_items)

    result = d_output.get()

    assert result.sum() == d_values.sum()
    assert sorted(result) == sorted(d_values.get())


def test_shuffle_iterator_invalid_num_items():
    with pytest.raises(ValueError, match="num_items must be > 0"):
        ShuffleIterator(0, seed=42)

    with pytest.raises(ValueError, match="num_items must be > 0"):
        ShuffleIterator(-1, seed=42)
