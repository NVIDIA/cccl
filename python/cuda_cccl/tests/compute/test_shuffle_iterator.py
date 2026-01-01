# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import pytest

import cuda.compute
from cuda.compute import OpKind
from cuda.compute.iterators import (
    PermutationIterator,
    ShuffleIterator,
)


def test_shuffle_iterator_bijectivity():
    """Test that ShuffleIterator produces a valid permutation (bijective)."""
    num_items = 100
    seed = 42

    shuffle_it = ShuffleIterator(num_items, seed)

    # Use unary_transform to collect all shuffled indices
    d_output = cp.empty(num_items, dtype=np.int64)
    cuda.compute.unary_transform(shuffle_it, d_output, lambda x: x, num_items)

    result = d_output.get()

    # Every index from 0 to num_items-1 should appear exactly once
    assert len(set(result)) == num_items
    assert set(result) == set(range(num_items))


def test_shuffle_iterator_determinism():
    """Test that same seed produces same permutation."""
    num_items = 50
    seed = 12345

    shuffle_it1 = ShuffleIterator(num_items, seed)
    shuffle_it2 = ShuffleIterator(num_items, seed)

    d_output1 = cp.empty(num_items, dtype=np.int64)
    d_output2 = cp.empty(num_items, dtype=np.int64)

    cuda.compute.unary_transform(shuffle_it1, d_output1, lambda x: x, num_items)
    cuda.compute.unary_transform(shuffle_it2, d_output2, lambda x: x, num_items)

    cp.testing.assert_array_equal(d_output1, d_output2)


def test_shuffle_iterator_different_seeds():
    """Test that different seeds produce different permutations."""
    num_items = 50

    shuffle_it1 = ShuffleIterator(num_items, seed=1)
    shuffle_it2 = ShuffleIterator(num_items, seed=2)

    d_output1 = cp.empty(num_items, dtype=np.int64)
    d_output2 = cp.empty(num_items, dtype=np.int64)

    cuda.compute.unary_transform(shuffle_it1, d_output1, lambda x: x, num_items)
    cuda.compute.unary_transform(shuffle_it2, d_output2, lambda x: x, num_items)

    # Very unlikely that two different seeds produce the same permutation
    assert not np.array_equal(d_output1.get(), d_output2.get())


@pytest.mark.parametrize("num_items", [1, 2, 7, 16, 17, 100, 1000, 1023, 1024, 1025])
def test_shuffle_iterator_various_sizes(num_items):
    """Test ShuffleIterator works correctly for various sizes."""
    seed = 42

    shuffle_it = ShuffleIterator(num_items, seed)

    d_output = cp.empty(num_items, dtype=np.int64)
    cuda.compute.unary_transform(shuffle_it, d_output, lambda x: x, num_items)

    result = d_output.get()

    # Should be a valid permutation
    assert len(set(result)) == num_items
    assert set(result) == set(range(num_items))


def test_shuffle_iterator_with_reduction():
    """Test ShuffleIterator with a reduction operation."""
    num_items = 100
    seed = 42

    shuffle_it = ShuffleIterator(num_items, seed)

    h_init = np.array([0], dtype=np.int64)
    d_output = cp.empty(1, dtype=np.int64)

    cuda.compute.reduce_into(shuffle_it, d_output, OpKind.PLUS, num_items, h_init)

    # Sum of a permutation of [0, num_items) should equal sum(0..num_items-1)
    expected = sum(range(num_items))
    assert d_output.get()[0] == expected


def test_shuffle_iterator_with_permutation_iterator():
    """Test ShuffleIterator composed with PermutationIterator for shuffled data access."""
    num_items = 10
    seed = 42

    # Create data array
    d_values = cp.asarray([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int32)

    # Create a shuffle iterator to generate shuffled indices
    shuffle_it = ShuffleIterator(num_items, seed)

    # Get the shuffled indices to verify correctness
    d_indices = cp.empty(num_items, dtype=np.int64)
    cuda.compute.unary_transform(shuffle_it, d_indices, lambda x: x, num_items)

    # Create permutation iterator using shuffle iterator as indices
    shuffle_it2 = ShuffleIterator(num_items, seed)  # Fresh iterator
    perm_it = PermutationIterator(d_values, shuffle_it2)

    # Reduce the permuted values
    h_init = np.array([0], dtype=np.int32)
    d_output = cp.empty(1, dtype=np.int32)

    cuda.compute.reduce_into(perm_it, d_output, OpKind.PLUS, num_items, h_init)

    # Sum should equal sum of all values (since it's a permutation)
    expected = d_values.sum()
    assert d_output.get()[0] == expected


def test_shuffle_iterator_invalid_num_items():
    """Test that ShuffleIterator raises error for invalid num_items."""
    with pytest.raises(ValueError, match="num_items must be > 0"):
        ShuffleIterator(0, seed=42)

    with pytest.raises(ValueError, match="num_items must be > 0"):
        ShuffleIterator(-1, seed=42)


def test_shuffle_iterator_rounds():
    """Test ShuffleIterator with different round counts."""
    num_items = 50
    seed = 42

    # Test with minimum rounds (6)
    shuffle_it1 = ShuffleIterator(num_items, seed, rounds=3)  # Will be clamped to 6

    d_output1 = cp.empty(num_items, dtype=np.int64)
    cuda.compute.unary_transform(shuffle_it1, d_output1, lambda x: x, num_items)

    result1 = set(d_output1.get())
    assert result1 == set(range(num_items))

    # Test with more rounds
    shuffle_it2 = ShuffleIterator(num_items, seed, rounds=12)

    d_output2 = cp.empty(num_items, dtype=np.int64)
    cuda.compute.unary_transform(shuffle_it2, d_output2, lambda x: x, num_items)

    result2 = set(d_output2.get())
    assert result2 == set(range(num_items))


def test_shuffle_iterator_large():
    """Test ShuffleIterator with a larger dataset."""
    num_items = 10000
    seed = 12345

    shuffle_it = ShuffleIterator(num_items, seed)

    # Just check sum to verify it's a valid permutation
    h_init = np.array([0], dtype=np.int64)
    d_output = cp.empty(1, dtype=np.int64)

    cuda.compute.reduce_into(shuffle_it, d_output, OpKind.PLUS, num_items, h_init)

    expected = sum(range(num_items))
    assert d_output.get()[0] == expected
