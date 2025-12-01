# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute.iterators import (
    CountingIterator,
    PermutationIterator,
    TransformIterator,
    ZipIterator,
)


def test_permutation_iterator_equality():
    values1 = cp.asarray([10, 20, 30, 40, 50], dtype="int32")
    values2 = cp.asarray([100, 200, 300], dtype="int32")
    values3 = cp.asarray([10, 20, 30, 40, 50], dtype="int64")

    indices1 = cp.asarray([0, 2, 1], dtype="int32")
    indices2 = cp.asarray([1, 0, 2], dtype="int32")
    indices3 = cp.asarray([0, 2, 1], dtype="int64")

    # Same value and index types should have same kind
    it1 = PermutationIterator(values1, indices1)
    it2 = PermutationIterator(values1, indices2)
    it3 = PermutationIterator(values2, indices1)

    assert it1.kind == it2.kind == it3.kind

    # Different value type should have different kind
    it4 = PermutationIterator(values3, indices1)
    assert it1.kind != it4.kind

    # Different index type should have different kind
    it5 = PermutationIterator(values1, indices3)
    assert it1.kind != it5.kind


def test_permutation_iterator_with_array_values():
    values = cp.asarray([10, 20, 30, 40, 50], dtype="int32")
    indices = cp.asarray([2, 0, 4, 1], dtype="int32")
    perm_it = PermutationIterator(values, indices)

    h_init = np.array([0], dtype="int32")
    d_output = cp.empty(1, dtype="int32")
    cuda.compute.reduce_into(
        perm_it, d_output, cuda.compute.OpKind.PLUS, len(indices), h_init
    )
    assert d_output[0] == values[indices].sum()


def test_permutation_iterator_with_iterator_values():
    values_it = CountingIterator(np.int32(10))
    indices = cp.asarray([2, 0, 4, 1], dtype="int32")
    perm_it = PermutationIterator(values_it, indices)

    h_init = np.array([0], dtype="int32")
    d_output = cp.empty(1, dtype="int32")

    cuda.compute.reduce_into(
        perm_it, d_output, cuda.compute.OpKind.PLUS, len(indices), h_init
    )

    expected = cp.arange(10, 20)[indices].sum()
    assert d_output[0] == expected


def test_permutation_iterator_of_zip_iterator():
    @cuda.compute.gpu_struct
    class Pair:
        value_0: np.int32
        value_1: np.int32

    d_values1 = cp.asarray([10, 20, 30, 40, 50], dtype="int32")
    d_values2 = cp.asarray([1, 2, 3, 4, 5], dtype="int32")
    zip_it = ZipIterator(d_values1, d_values2)
    indices = cp.asarray([2, 0, 4], dtype="int32")
    perm_it = PermutationIterator(zip_it, indices)

    def sum_both_fields(a, b):
        return Pair(a.value_0 + b.value_0, a.value_1 + b.value_1)

    h_init = Pair(0, 0)
    d_output = cp.empty(1, dtype=Pair.dtype)

    cuda.compute.reduce_into(perm_it, d_output, sum_both_fields, len(indices), h_init)

    result = d_output.get()[0]
    assert result["value_0"] == d_values1[indices].sum()
    assert result["value_1"] == d_values2[indices].sum()


def test_zip_iterator_of_permutation_iterators():
    @cuda.compute.gpu_struct
    class Pair:
        value_0: np.int32
        value_1: np.int32

    d_values1 = cp.asarray([10, 20, 30, 40, 50], dtype="int32")
    d_values2 = cp.asarray([100, 200, 300, 400, 500], dtype="int32")
    indices1 = cp.asarray([4, 1, 3, 0], dtype="int32")
    indices2 = cp.asarray([2, 4, 0, 1], dtype="int32")
    perm_it1 = PermutationIterator(d_values1, indices1)
    perm_it2 = PermutationIterator(d_values2, indices2)

    zip_it = ZipIterator(perm_it1, perm_it2)

    def sum_both_fields(a, b):
        return Pair(a.value_0 + b.value_0, a.value_1 + b.value_1)

    h_init = Pair(0, 0)
    d_output = cp.empty(1, dtype=Pair.dtype)

    num_items = len(indices1)
    cuda.compute.reduce_into(zip_it, d_output, sum_both_fields, num_items, h_init)

    result = d_output.get()[0]
    assert result["value_0"] == d_values1[indices1].sum()
    assert result["value_1"] == d_values2[indices2].sum()


def test_unary_transform_of_permutation_iterator():
    values = cp.asarray([10, 20, 30, 40, 50], dtype="int32")
    indices = cp.asarray([2, 0, 4, 1], dtype="int32")
    perm_it = PermutationIterator(values, indices)

    def op(a):
        return a + 1

    d_out = cp.empty_like(values, shape=(len(indices),))
    cuda.compute.unary_transform(perm_it, d_out, op, len(indices))

    expected = values[indices] + 1
    assert cp.all(d_out == expected)


def test_caching_permutation_iterator():
    # same value type, same index type:
    it1 = PermutationIterator(
        cp.arange(10, dtype=np.int32), cp.arange(10, dtype=np.int32)
    )
    it2 = PermutationIterator(
        cp.arange(10, dtype=np.int32), cp.arange(10, dtype=np.int32)
    )
    assert it1.advance is it2.advance
    assert it1.input_dereference is it2.input_dereference
    assert it1.output_dereference is it2.output_dereference

    # same value type, different index types:
    it1 = PermutationIterator(
        cp.arange(10, dtype=np.int32), cp.arange(10, dtype=np.int32)
    )
    it2 = PermutationIterator(
        cp.arange(10, dtype=np.int32), cp.arange(10, dtype=np.int64)
    )
    assert it1.advance is not it2.advance
    assert it1.input_dereference is not it2.input_dereference
    assert it1.output_dereference is not it2.output_dereference

    # different value types, same index type:
    it1 = PermutationIterator(
        cp.arange(10, dtype=np.int32), cp.arange(10, dtype=np.int32)
    )
    it2 = PermutationIterator(
        cp.arange(10, dtype=np.int64), cp.arange(10, dtype=np.int32)
    )
    assert it1.advance is not it2.advance
    assert it1.input_dereference is not it2.input_dereference
    assert it1.output_dereference is not it2.output_dereference

    # permutation iterator with transform iterator value type (same op):
    def op(x):
        return x + 1

    it1 = PermutationIterator(
        TransformIterator(cp.arange(10, dtype=np.int32), op),
        cp.arange(10, dtype=np.int32),
    )
    it2 = PermutationIterator(
        TransformIterator(cp.arange(10, dtype=np.int32), op),
        cp.arange(10, dtype=np.int32),
    )
    assert it1.advance is it2.advance
    assert it1.input_dereference is it2.input_dereference
    assert it1.output_dereference is it2.output_dereference

    # permutation iterator with transform iterator value type (different op):
    def op2(x):
        return x + 2

    it1 = PermutationIterator(
        TransformIterator(cp.arange(10, dtype=np.int32), op),
        cp.arange(10, dtype=np.int32),
    )
    it2 = PermutationIterator(
        TransformIterator(cp.arange(10, dtype=np.int32), op2),
        cp.arange(10, dtype=np.int32),
    )
    assert it1.advance is not it2.advance
    assert it1.input_dereference is not it2.input_dereference
    assert it1.output_dereference is not it2.output_dereference
