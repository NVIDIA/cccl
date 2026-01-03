# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Demonstrate using ShuffleIterator for deterministic random permutation of indices in ``[0, num_items)``.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import (
    OpKind,
    PermutationIterator,
    ShuffleIterator,
)

# Create a shuffle iterator for 10 elements with a fixed seed
num_items = 10
seed = 42
shuffle_it = ShuffleIterator(num_items, seed)

# Collect the shuffled indices using unary_transform
d_indices = cp.empty(num_items, dtype=np.int64)
cuda.compute.unary_transform(shuffle_it, d_indices, lambda x: x, num_items)

print(f"Shuffled indices: {d_indices.get()}")
# Verify it's a valid permutation (all indices 0 to num_items-1 appear exactly once)
assert set(d_indices.get()) == set(range(num_items))

# Use ShuffleIterator with PermutationIterator to access data in shuffled order
d_values = cp.asarray([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int32)

# Create a new shuffle iterator (same seed for same order)
shuffle_it2 = ShuffleIterator(num_items, seed)

# Combine with PermutationIterator to access values in shuffled order
perm_it = PermutationIterator(d_values, shuffle_it2)

# Reduce the shuffled values - sum should equal sum of all values
h_init = np.array([0], dtype=np.int32)
d_output = cp.empty(1, dtype=np.int32)

cuda.compute.reduce_into(perm_it, d_output, OpKind.PLUS, num_items, h_init)

# Since shuffle is a permutation, sum equals sum of all values
expected_sum = d_values.sum()
print(f"Sum of shuffled values: {d_output[0]} (expected: {expected_sum})")
assert d_output[0] == expected_sum

# Different seeds produce different permutations
shuffle_it_a = ShuffleIterator(num_items, seed=1)
shuffle_it_b = ShuffleIterator(num_items, seed=2)

d_perm_a = cp.empty(num_items, dtype=np.int64)
d_perm_b = cp.empty(num_items, dtype=np.int64)

cuda.compute.unary_transform(shuffle_it_a, d_perm_a, lambda x: x, num_items)
cuda.compute.unary_transform(shuffle_it_b, d_perm_b, lambda x: x, num_items)

print(f"Permutation with seed=1: {d_perm_a.get()}")
print(f"Permutation with seed=2: {d_perm_b.get()}")
assert not np.array_equal(d_perm_a.get(), d_perm_b.get())
