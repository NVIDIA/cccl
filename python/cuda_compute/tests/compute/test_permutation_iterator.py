# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute.iterators import (
    CountingIterator,
    PermutationIterator,
    ZipIterator,
)


def test_permutation_iterator_equality():
    d_values1 = DeviceArray.from_numpy(np.asarray([10, 20, 30, 40, 50], dtype="int32"))
    d_values2 = DeviceArray.from_numpy(np.asarray([100, 200, 300], dtype="int32"))
    d_values3 = DeviceArray.from_numpy(np.asarray([10, 20, 30, 40, 50], dtype="int64"))

    d_indices1 = DeviceArray.from_numpy(np.asarray([0, 2, 1], dtype="int32"))
    d_indices2 = DeviceArray.from_numpy(np.asarray([1, 0, 2], dtype="int32"))
    d_indices3 = DeviceArray.from_numpy(np.asarray([0, 2, 1], dtype="int64"))

    # Same value and index types should have same kind
    it1 = PermutationIterator(d_values1, d_indices1)
    it2 = PermutationIterator(d_values1, d_indices2)
    it3 = PermutationIterator(d_values2, d_indices1)

    assert it1.kind == it2.kind == it3.kind

    # Different value type should have different kind
    it4 = PermutationIterator(d_values3, d_indices1)
    assert it1.kind != it4.kind

    # Different index type should have different kind
    it5 = PermutationIterator(d_values1, d_indices3)
    assert it1.kind != it5.kind


def test_permutation_iterator_with_array_values():
    h_values = np.asarray([10, 20, 30, 40, 50], dtype="int32")
    h_indices = np.asarray([2, 0, 4, 1], dtype="int32")
    d_values = DeviceArray.from_numpy(h_values)
    d_indices = DeviceArray.from_numpy(h_indices)
    perm_it = PermutationIterator(d_values, d_indices)

    h_init = np.array([0], dtype="int32")
    d_output = DeviceArray.empty(1, np.int32)
    cuda.compute.reduce_into(
        d_in=perm_it,
        d_out=d_output,
        num_items=len(h_indices),
        op=cuda.compute.OpKind.PLUS,
        h_init=h_init,
    )
    assert d_output.copy_to_host()[0] == h_values[h_indices].sum()


def test_permutation_iterator_with_iterator_values():
    values_it = CountingIterator(np.int32(10))
    h_indices = np.asarray([2, 0, 4, 1], dtype="int32")
    d_indices = DeviceArray.from_numpy(h_indices)
    perm_it = PermutationIterator(values_it, d_indices)

    h_init = np.array([0], dtype="int32")
    d_output = DeviceArray.empty(1, np.int32)

    cuda.compute.reduce_into(
        d_in=perm_it,
        d_out=d_output,
        num_items=len(h_indices),
        op=cuda.compute.OpKind.PLUS,
        h_init=h_init,
    )

    expected = np.arange(10, 20)[h_indices].sum()
    assert d_output.copy_to_host()[0] == expected


def test_permutation_iterator_of_zip_iterator():
    @cuda.compute.gpu_struct
    class Pair:
        value_0: np.int32
        value_1: np.int32

    h_values1 = np.asarray([10, 20, 30, 40, 50], dtype="int32")
    h_values2 = np.asarray([1, 2, 3, 4, 5], dtype="int32")
    d_values1 = DeviceArray.from_numpy(h_values1)
    d_values2 = DeviceArray.from_numpy(h_values2)
    zip_it = ZipIterator(d_values1, d_values2)
    h_indices = np.asarray([2, 0, 4], dtype="int32")
    d_indices = DeviceArray.from_numpy(h_indices)
    perm_it = PermutationIterator(zip_it, d_indices)

    def sum_both_fields(a, b):
        return Pair(a.value_0 + b.value_0, a.value_1 + b.value_1)

    h_init = Pair(0, 0)
    d_output = DeviceArray.empty(1, Pair.dtype)

    cuda.compute.reduce_into(
        d_in=perm_it,
        d_out=d_output,
        num_items=len(h_indices),
        op=sum_both_fields,
        h_init=h_init,
    )

    result = d_output.copy_to_host()[0]
    assert result["value_0"] == h_values1[h_indices].sum()
    assert result["value_1"] == h_values2[h_indices].sum()


def test_zip_iterator_of_permutation_iterators():
    @cuda.compute.gpu_struct
    class Pair:
        value_0: np.int32
        value_1: np.int32

    h_values1 = np.asarray([10, 20, 30, 40, 50], dtype="int32")
    h_values2 = np.asarray([100, 200, 300, 400, 500], dtype="int32")
    h_indices1 = np.asarray([4, 1, 3, 0], dtype="int32")
    h_indices2 = np.asarray([2, 4, 0, 1], dtype="int32")
    d_values1 = DeviceArray.from_numpy(h_values1)
    d_values2 = DeviceArray.from_numpy(h_values2)
    d_indices1 = DeviceArray.from_numpy(h_indices1)
    d_indices2 = DeviceArray.from_numpy(h_indices2)
    perm_it1 = PermutationIterator(d_values1, d_indices1)
    perm_it2 = PermutationIterator(d_values2, d_indices2)

    zip_it = ZipIterator(perm_it1, perm_it2)

    def sum_both_fields(a, b):
        return Pair(a.value_0 + b.value_0, a.value_1 + b.value_1)

    h_init = Pair(0, 0)
    d_output = DeviceArray.empty(1, Pair.dtype)

    num_items = len(h_indices1)
    cuda.compute.reduce_into(
        d_in=zip_it,
        d_out=d_output,
        num_items=num_items,
        op=sum_both_fields,
        h_init=h_init,
    )

    result = d_output.copy_to_host()[0]
    assert result["value_0"] == h_values1[h_indices1].sum()
    assert result["value_1"] == h_values2[h_indices2].sum()


def test_unary_transform_of_permutation_iterator():
    h_values = np.asarray([10, 20, 30, 40, 50], dtype="int32")
    h_indices = np.asarray([2, 0, 4, 1], dtype="int32")
    d_values = DeviceArray.from_numpy(h_values)
    d_indices = DeviceArray.from_numpy(h_indices)
    perm_it = PermutationIterator(d_values, d_indices)

    def op(a):
        return a + 1

    d_out = DeviceArray.empty(len(h_indices), h_values.dtype)
    cuda.compute.unary_transform(
        d_in=perm_it, d_out=d_out, op=op, num_items=len(h_indices)
    )

    expected = h_values[h_indices] + 1
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_caching_permutation_iterator():
    """Test that iterator compilation is cached across instances with the same structure."""
    from cuda.compute._cpp_compile import compile_cpp_op_code

    # Test 1: Same structure → same kind
    it1 = PermutationIterator(
        DeviceArray.from_numpy(np.arange(10, dtype=np.int32)),
        DeviceArray.from_numpy(np.arange(10, dtype=np.int32)),
    )
    it2 = PermutationIterator(
        DeviceArray.from_numpy(np.arange(20, dtype=np.int32)),
        DeviceArray.from_numpy(np.arange(5, dtype=np.int32)),
    )
    assert it1.kind == it2.kind, "Same structure should have same kind"

    # Test 2: Different index type → different kind
    it3 = PermutationIterator(
        DeviceArray.from_numpy(np.arange(10, dtype=np.int32)),
        DeviceArray.from_numpy(np.arange(10, dtype=np.int64)),
    )
    assert it1.kind != it3.kind, "Different index type should have different kind"

    # Test 3: Different value type → different kind
    it4 = PermutationIterator(
        DeviceArray.from_numpy(np.arange(10, dtype=np.int64)),
        DeviceArray.from_numpy(np.arange(10, dtype=np.int32)),
    )
    assert it1.kind != it4.kind, "Different value type should have different kind"

    # Test 4: Verify compilation caching with cache statistics
    compile_cpp_op_code.cache_clear()

    # Create multiple instances with same structure
    iterators = []
    for i in range(3):
        it = PermutationIterator(
            DeviceArray.from_numpy(np.arange(i * 10, (i + 1) * 10, dtype=np.float32)),
            DeviceArray.from_numpy(np.arange(5, dtype=np.int32)),
        )
        # Trigger compilation by accessing Op objects
        it.get_advance_op()
        it.get_input_deref_op()
        iterators.append(it)

    cache_info = compile_cpp_op_code.cache_info()
    assert cache_info.hits >= 2, (
        f"Expected cache hits for same structure, got {cache_info.hits} hits, "
        f"{cache_info.misses} misses"
    )


def test_permutation_iterator_advance():
    """Test PermutationIterator.__add__ only advances indices, not values."""
    # Create values array [10, 20, 30, 40, 50, 60, 70]
    h_values = np.asarray([10, 20, 30, 40, 50, 60, 70], dtype="int32")
    d_values = DeviceArray.from_numpy(h_values)

    # Create indices array [2, 0, 4, 1, 3, 5]
    # indices[0] = 2 -> values[2] = 30
    # indices[1] = 0 -> values[0] = 10
    # indices[2] = 4 -> values[4] = 50
    # indices[3] = 1 -> values[1] = 20
    # indices[4] = 3 -> values[3] = 40
    # indices[5] = 5 -> values[5] = 60
    h_indices = np.asarray([2, 0, 4, 1, 3, 5], dtype="int32")
    d_indices = DeviceArray.from_numpy(h_indices)

    perm_it = PermutationIterator(d_values, d_indices)

    # Advance by 2 positions (should skip first 2 indices)
    offset = 2
    advanced_perm_it = perm_it + offset

    # Reduce from the advanced position
    # Should process indices[2:] = [4, 1, 3, 5]
    # Which accesses values[4, 1, 3, 5] = [50, 20, 40, 60]
    h_init = np.array([0], dtype="int32")
    d_output = DeviceArray.empty(1, np.int32)

    remaining_items = len(h_indices) - offset
    cuda.compute.reduce_into(
        d_in=advanced_perm_it,
        d_out=d_output,
        num_items=remaining_items,
        op=cuda.compute.OpKind.PLUS,
        h_init=h_init,
    )

    # Expected: values[indices[2:]] = values[[4, 1, 3, 5]] = [50, 20, 40, 60]
    expected = h_values[h_indices[offset:]].sum()
    result = d_output.copy_to_host()[0]
    assert result == expected, f"Expected {expected}, got {result}"
